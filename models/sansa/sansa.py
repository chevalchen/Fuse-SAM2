import os
from typing import Any, Dict, List, Optional, Tuple

import py3_wget
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
import torch.nn.functional as F

from models.sam2.modeling.sam2_utils import preprocess
from models.sam2.modeling.sam2_base import SAM2Base
from models.sansa.model_utils import BackboneOutput, DecoderOutput
from models.sansa.uncertainty import UncertaintyHead
from models.sansa.hppa import HPPA
from models.sansa.mlpa import MLPA
from util.path_utils import SAM2_PATHS_CONFIG, SAM2_WEIGHTS_URL
from util.promptable_utils import rescale_prompt


class SANSA(nn.Module):
    def __init__(
        self,
        sam: SAM2Base,
        device: torch.device,
        use_uncertainty: bool = True,
        mlpa: Optional[MLPA] = None,
        hppa: Optional[HPPA] = None,
    ):
        super().__init__()
        self.sam = sam
        self.device = device
        self.use_uncertainty = use_uncertainty
        if self.use_uncertainty:
            self.uncertainty_head = UncertaintyHead(in_channels=256)

        # New modules — both None means original SANSA behaviour is preserved
        self.mlpa = mlpa
        self.hppa = hppa

    # ------------------------------------------------------------------ #
    #  Public forward                                                       #
    # ------------------------------------------------------------------ #

    def forward(self, samples: torch.Tensor, prompt_dict: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Args:
            samples: [B, T, C, H, W]  (pseudo-video: support frames then query frames)
            prompt_dict: list of per-batch prompt dicts

        Returns:
            {"pred_masks": [B*T, H', W']}
        """
        samples, B, T, orig_size = self._preprocess_visual_features(samples, self.sam.image_size)
        backbone_output: BackboneOutput = self._forward_backbone(samples, orig_size)
        outputs = {"masks": [], "uncertainties": []}

        n_shots = prompt_dict['shots']

        for b in range(B):
            self.memory_bank = {}
            ref_info: Optional[Dict] = None  # built lazily on first query frame

            for idx in range(T):
                absolute_idx = b * T + idx

                if idx < n_shots:
                    # ── Support frame ──────────────────────────────────────
                    frame_prompt = prompt_dict[b][idx]['prompt']
                    prompt_type  = prompt_dict[b][idx]['prompt_type']
                    frame_prompt = rescale_prompt(
                        frame_prompt, prompt_type, orig_size[b], self.sam.image_size
                    )
                    if prompt_type == 'mask':
                        decoder_out: DecoderOutput = self.sam._use_mask_as_output(
                            backbone_output, frame_prompt, absolute_idx
                        )
                    else:
                        decoder_out: DecoderOutput = self._compute_decoder_out_no_mem(
                            backbone_output, absolute_idx, prompt_input=frame_prompt
                        )

                else:
                    # ── Query frame ────────────────────────────────────────
                    # Build ref_info once (after all support frames are in memory_bank)
                    if ref_info is None and (self.mlpa is not None or self.hppa is not None):
                        ref_info = self._build_ref_info(b, T, n_shots, backbone_output)

                    decoder_out: DecoderOutput = self._compute_decoder_out_w_mem(
                        backbone_output, absolute_idx, idx, self.memory_bank,
                        ref_info=ref_info,
                    )

                # Update memory bank and collect outputs
                mem_entry = self._compute_memory_bank_dict(decoder_out, backbone_output, absolute_idx)
                self.memory_bank[idx] = mem_entry
                outputs["masks"].append(decoder_out.masks[0])
                if decoder_out.uncertainty is not None:
                    outputs["uncertainties"].append(decoder_out.uncertainty)

        masks = torch.cat(outputs["masks"])
        masks = F.interpolate(masks[None], size=orig_size[0], mode='bilinear', align_corners=False)[0]
        result = {"pred_masks": masks}
        if outputs["uncertainties"]:
            uncertainty_maps = torch.cat(outputs["uncertainties"], dim=0)
            uncertainty_maps = F.interpolate(
                uncertainty_maps, size=orig_size[0], mode='bilinear', align_corners=False
            )
            result["uncertainty"] = uncertainty_maps
        return result

    # ------------------------------------------------------------------ #
    #  Preprocessing                                                        #
    # ------------------------------------------------------------------ #

    def _preprocess_visual_features(
        self, samples: torch.Tensor, image_size: int
    ) -> Tuple[torch.Tensor, int, int, List[Tuple[int, int]]]:
        B, T, C, H, W = samples.shape
        samples = samples.view(B * T, C, H, W)
        orig_size = [tuple(x.shape[-2:]) for x in samples]
        samples = torch.stack([preprocess(x, image_size) for x in samples], dim=0)
        return samples, B, T, orig_size

    # ------------------------------------------------------------------ #
    #  Backbone                                                             #
    # ------------------------------------------------------------------ #

    def _forward_backbone(
        self, samples: torch.Tensor, orig_size: List[Tuple[int, int]]
    ) -> BackboneOutput:
        """Run SAM2 image encoder; also stores raw Hiera stage features for MLPA."""
        # vis: list of 4 tensors [B*T, C_i, H_i, W_i], finest (stage0) to coarsest (stage3)
        vis = self.sam.image_encoder.trunk(samples)
        feats, pos = self.sam.image_encoder.neck(vis)

        # Discard lowest resolution (coarsest FPN level)
        feats, pos = feats[:-1], pos[:-1]

        feats[0] = self.sam.sam_mask_decoder.conv_s0(feats[0])
        feats[1] = self.sam.sam_mask_decoder.conv_s1(feats[1])

        bb = {
            "vision_features": feats[-1],
            "vision_pos_enc": pos,
            "backbone_fpn": feats,
        }
        vision_feats, vision_pos, sizes = self.sam._prepare_backbone_features(bb)

        # Store raw Hiera outputs only when MLPA is active (saves memory otherwise)
        hiera_stage_feats = vis if self.mlpa is not None else None

        return BackboneOutput(orig_size, vision_feats, vision_pos, sizes, hiera_stage_feats)

    # ------------------------------------------------------------------ #
    #  Per-frame decoding                                                   #
    # ------------------------------------------------------------------ #

    def _compute_decoder_out_no_mem(
        self,
        backbone_out: BackboneOutput,
        idx: int,
        prompt_input: Optional[Dict[str, torch.Tensor]],
    ) -> DecoderOutput:
        current_vision_feats = backbone_out.get_current_feats(idx)
        high_res_features    = backbone_out.get_high_res_features(current_vision_feats)

        pix_feat_no_mem = current_vision_feats[-1:][-1] + self.sam.no_mem_embed
        pix_feat_no_mem = pix_feat_no_mem.permute(1, 2, 0).view(1, 256, 64, 64)
        decoder_out: DecoderOutput = self.sam._forward_sam_heads(
            backbone_features=pix_feat_no_mem,
            point_inputs=prompt_input,
            high_res_features=high_res_features,
        )
        return decoder_out

    def _compute_decoder_out_w_mem(
        self,
        backbone_out: BackboneOutput,
        idx: int,
        memory_idx: int,
        memory_bank: Dict[int, Dict[str, torch.Tensor]],
        ref_info: Optional[Dict] = None,
    ) -> DecoderOutput:
        """Decode a query frame, optionally enhanced by MLPA + HPPA."""
        current_vision_feats     = backbone_out.get_current_feats(idx)
        current_vision_pos_embeds = backbone_out.get_current_pos_embeds(idx)
        high_res_features        = backbone_out.get_high_res_features(current_vision_feats)

        # ── Apply MLPA + HPPA before Memory Attention ─────────────────────
        if ref_info is not None:
            current_vision_feats = self._apply_modules(
                current_vision_feats, backbone_out, idx, ref_info, backbone_out.feat_sizes
            )

        pix_feat_with_mem = self.sam._prepare_memory_conditioned_features(
            frame_idx=memory_idx,
            current_vision_feats=current_vision_feats[-1:],
            current_vision_pos_embeds=current_vision_pos_embeds[-1:],
            feat_sizes=backbone_out.feat_sizes[-1:],
            num_frames=memory_idx + 1,
            memory_bank=memory_bank,
        )

        uncertainty = None
        if self.use_uncertainty:
            uncertainty = self.uncertainty_head(pix_feat_with_mem)

        decoder_out: DecoderOutput = self.sam._forward_sam_heads(
            backbone_features=pix_feat_with_mem,
            high_res_features=high_res_features,
            multimask_output=True if memory_idx > 0 else False,
        )
        decoder_out.uncertainty = uncertainty
        return decoder_out

    def _compute_memory_bank_dict(
        self, decoder_out: DecoderOutput, backbone_out: BackboneOutput, idx: int
    ) -> Dict[str, torch.Tensor]:
        current_vision_feats = backbone_out.get_current_feats(idx)
        feat_sizes           = backbone_out.feat_sizes

        mem_feats, mem_pos = self.sam._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=decoder_out.high_res_masks,
            is_mask_from_pts=False,
        )
        return {
            "maskmem_features": mem_feats,
            "maskmem_pos_enc":  mem_pos,
            "pred_masks":       decoder_out.low_res_masks,
            "obj_ptr":          decoder_out.obj_ptr,
        }

    # ------------------------------------------------------------------ #
    #  MLPA + HPPA integration helpers                                      #
    # ------------------------------------------------------------------ #

    def _build_ref_info(
        self,
        b: int,
        T: int,
        n_shots: int,
        backbone_out: BackboneOutput,
    ) -> Dict:
        """
        Collect reference features and masks from all support frames and average
        them.  Called once per query sequence (after all support frames are in
        memory_bank).

        Returns a dict with:
            ref_feat  [HW, 1, 256]     — averaged FPN feature over shots
            ref_mask  [1, 1, H, W]     — averaged soft mask at feat resolution
        """
        feat_H, feat_W = backbone_out.feat_sizes[-1]  # e.g. (64, 64)

        ref_feats = []
        ref_masks  = []

        for shot_idx in range(n_shots):
            abs_shot_idx = b * T + shot_idx

            # FPN feature for this reference frame: [HW, 1, 256]
            ref_feat_i = backbone_out.get_current_feats(abs_shot_idx)[-1]
            ref_feats.append(ref_feat_i)

            # Reference mask from memory bank (set during support-frame processing)
            pred_masks_i = self.memory_bank[shot_idx].get("pred_masks", None)
            if pred_masks_i is not None:
                # Convert logits → prob, resize to feature resolution
                mask_i = F.interpolate(
                    pred_masks_i.sigmoid()[:, :1],      # take first mask channel
                    size=(feat_H, feat_W),
                    mode='bilinear',
                    align_corners=False,
                )  # [1, 1, feat_H, feat_W]
            else:
                # Fallback: uniform mask (no mask signal → neutral part token)
                mask_i = torch.ones(
                    1, 1, feat_H, feat_W,
                    device=ref_feat_i.device, dtype=ref_feat_i.dtype
                )
            ref_masks.append(mask_i)

        # Average over shots: [HW, 1, 256] and [1, 1, H, W]
        ref_feat_avg = torch.stack(ref_feats, dim=0).mean(dim=0)   # [HW, 1, 256]
        ref_mask_avg = torch.stack(ref_masks, dim=0).mean(dim=0)   # [1, 1, H, W]

        return {"ref_feat": ref_feat_avg, "ref_mask": ref_mask_avg}

    def _apply_modules(
        self,
        current_vision_feats: List[torch.Tensor],
        backbone_out: BackboneOutput,
        abs_idx: int,
        ref_info: Dict,
        feat_sizes: List[Tuple[int, int]],
    ) -> List[torch.Tensor]:
        """
        Apply MLPA then HPPA to the lowest-resolution FPN feature (the one
        consumed by Memory Attention).

        Flow:
            Hiera stage feats → MLPA → upsample → add to FPN feat
            Enhanced FPN feat → HPPA (conditioned on ref) → output
        """
        target_H, target_W = feat_sizes[-1]
        enhanced = list(current_vision_feats)  # shallow copy; modifying [-1] only

        # ── MLPA: multi-scale aggregation ─────────────────────────────────
        if self.mlpa is not None and backbone_out.hiera_stage_feats is not None:
            # Use stages 1, 2, 3 (skip stage 0 — too large, mostly texture)
            hiera_feats_frame = [
                backbone_out.hiera_stage_feats[s][abs_idx:abs_idx + 1]
                for s in range(1, len(backbone_out.hiera_stage_feats))
            ]  # list of [1, C_i, H_i, W_i]

            mlpa_out = self.mlpa(hiera_feats_frame)  # [1, 256, H_stage_deep, W_stage_deep]

            # Upsample MLPA output to FPN feature resolution if needed
            if (mlpa_out.shape[-2], mlpa_out.shape[-1]) != (target_H, target_W):
                mlpa_out = F.interpolate(
                    mlpa_out, size=(target_H, target_W),
                    mode='bilinear', align_corners=False,
                )  # [1, 256, target_H, target_W]

            # Convert [1, 256, H, W] → [HW, 1, 256] (seq-first format)
            mlpa_flat = mlpa_out.flatten(2).permute(2, 0, 1)

            enhanced[-1] = current_vision_feats[-1] + mlpa_flat

        # ── HPPA: reference-conditioned adaptation ─────────────────────────
        if self.hppa is not None:
            enhanced[-1] = self.hppa(
                target_feat=enhanced[-1],           # [HW, 1, 256]
                ref_feat=ref_info["ref_feat"],      # [HW, 1, 256]
                ref_mask=ref_info["ref_mask"],      # [1, 1, feat_H, feat_W]
            )

        return enhanced


# --------------------------------------------------------------------------- #
#  Factory                                                                      #
# --------------------------------------------------------------------------- #

def build_sansa(
    sam2_version: str = 'large',
    adaptformer_stages: List[int] = [2, 3],
    channel_factor: float = 0.3,
    device: str = 'cuda',
    use_uncertainty: bool = True,
    use_mlpa: bool = True,
    use_hppa: bool = True,
    hppa_part_token_dim: int = 128,
    hppa_bottleneck_dim: int = 64,
    mlpa_num_heads: int = 8,
) -> 'SANSA':
    assert sam2_version in SAM2_PATHS_CONFIG.keys(), \
        f'Unknown sam2_version: {sam2_version}'

    sam2_weights, sam2_config = SAM2_PATHS_CONFIG[sam2_version]
    if not os.path.isfile(sam2_weights):
        print(f"Downloading SAM2-{sam2_version}")
        py3_wget.download_file(SAM2_WEIGHTS_URL[sam2_version], sam2_weights)

    with initialize(version_base=None, config_path=".", job_name="test_app"):
        cfg = compose(config_name=sam2_config, overrides=[
            f"++model.image_encoder.trunk.adaptformer_stages={adaptformer_stages}",
            f"++model.image_encoder.trunk.adapt_dim={channel_factor}",
        ])
        OmegaConf.resolve(cfg)
        cfg.model.pred_obj_scores       = False
        cfg.model.pred_obj_scores_mlp   = False
        cfg.model.fixed_no_obj_ptr      = False
        sam = instantiate(cfg.model, _recursive_=True)

    state_dict = torch.load(sam2_weights, map_location="cpu", weights_only=False)["model"]
    sam.load_state_dict(state_dict, strict=False)

    # ── Build MLPA + HPPA if requested ────────────────────────────────────
    mlpa_module: Optional[MLPA] = None
    hppa_module: Optional[HPPA] = None

    if use_mlpa or use_hppa:
        trunk = sam.image_encoder.trunk
        # channel_list = [stage3, stage2, stage1, stage0] (coarsest→finest)
        all_dims = list(reversed(trunk.channel_list))  # [stage0, stage1, stage2, stage3]
        # Stages 1, 2, 3 for MLPA (skip stage 0)
        mlpa_stage_dims = all_dims[1:]                 # e.g. [288, 576, 1152] for large

        if use_mlpa:
            mlpa_module = MLPA(
                stage_dims=mlpa_stage_dims,
                target_dim=256,
                num_heads=mlpa_num_heads,
            )
        if use_hppa:
            hppa_module = HPPA(
                dim=256,
                part_token_dim=hppa_part_token_dim,
                bottleneck_dim=hppa_bottleneck_dim,
            )

    model = SANSA(
        sam=sam,
        device=torch.device(device),
        use_uncertainty=use_uncertainty,
        mlpa=mlpa_module,
        hppa=hppa_module,
    )

    # ── Freeze: everything except adapters, uncertainty head, HPPA, MLPA ─
    for name, p in model.named_parameters():
        p.requires_grad = (
            "adapter"          in name or
            "uncertainty_head" in name or
            "hppa"             in name or
            "mlpa"             in name
        )

    return model
