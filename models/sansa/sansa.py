import os
from typing import Any, Dict, List, Tuple

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
from util.path_utils import SAM2_PATHS_CONFIG, SAM2_WEIGHTS_URL
from util.promptable_utils import rescale_prompt


class SANSA(nn.Module):
    def __init__(
        self,
        sam: SAM2Base,
        device: torch.device,
        use_uncertainty: bool = True,
        use_part_proto_ptr: bool = False,
        use_corr_dense_prompt: bool = False,
        part_proto_temperature: float = 1.0,
        corr_clamp: float = 6.0,
    ):
        super().__init__()
        self.sam = sam
        self.device = device
        self.use_uncertainty = use_uncertainty
        self.use_part_proto_ptr = use_part_proto_ptr
        self.use_corr_dense_prompt = use_corr_dense_prompt
        self.part_proto_temperature = part_proto_temperature
        self.corr_clamp = corr_clamp
        if self.use_uncertainty:
            self.uncertainty_head = UncertaintyHead(in_channels=256)
        if self.use_part_proto_ptr or self.use_corr_dense_prompt:
            self.part_proto_proj = nn.Linear(256, 256)
            nn.init.zeros_(self.part_proto_proj.weight)
            nn.init.zeros_(self.part_proto_proj.bias)

    def forward(self, samples: torch.Tensor, prompt_dict: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run SANSA.
        Args:
            samples: [B, T, C, H, W].
            targets: list (len B) of dicts with:
                - 'is_support': list[bool] of len T
                - 'masks': Tensor [T, H, W]

        Returns:
            {"pred_masks": Tensor [B*T, H', W']}
        """

        samples, B, T, orig_size = self._preprocess_visual_features(samples, self.sam.image_size)
        backbone_output: BackboneOutput = self._forward_backbone(samples, orig_size)
        outputs = {"masks": [], "uncertainties": []}

        n_shots = prompt_dict['shots']
        for b in range(B):
            self.memory_bank = {}
            for idx in range(T):
                absolute_idx = b * T + idx

                if idx < n_shots:
                    frame_prompt = prompt_dict[b][idx]['prompt']
                    prompt_type = prompt_dict[b][idx]['prompt_type']
                    frame_prompt = rescale_prompt(frame_prompt, prompt_type, orig_size[b], self.sam.image_size)
                    if prompt_type == 'mask':
                        decoder_out: DecoderOutput = self.sam._use_mask_as_output(backbone_output, frame_prompt, absolute_idx)
                    else:
                        decoder_out: DecoderOutput = self._compute_decoder_out_no_mem(backbone_output, absolute_idx, prompt_input=frame_prompt)
                        
                else:
                    decoder_out: DecoderOutput = self._compute_decoder_out_w_mem(backbone_output, absolute_idx, idx, self.memory_bank)

                # update memory bank
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

    def _preprocess_visual_features(
        self, samples: torch.Tensor, image_size: int
    ) -> Tuple[torch.Tensor, int, int, List[Tuple[int, int]]]:
        """
        Flatten [B,T,C,H,W] -> [B*T,C,H,W], store original sizes, and apply SAM2 preprocess.

        Args:
            samples:   Tensor [B, T, C, H, W].
            image_size: target side for SAM2 preprocessing.

        Returns:
            (samples_bt, B, T, orig_sizes)
        """

        B, T, C, H, W = samples.shape
        samples = samples.view(B * T, C, H, W)
        orig_size = [tuple(x.shape[-2:]) for x in samples]
        samples = torch.stack([preprocess(x, image_size) for x in samples], dim=0)
        return samples, B, T, orig_size

    def _compute_decoder_out_no_mem(
        self,
        backbone_out: BackboneOutput,
        idx: int,
        prompt_input: Dict[str, torch.Tensor] | None,
    ) -> DecoderOutput:
        """
        Decode a frame without memory: used for reference frames;

        Args:
            backbone_out: backbone features.
            idx: absolute idx.
            prompt:       "mask" | "point" | "scribble" | "box".
            prompt_input: inputs for point/scribble/box (ignored for "mask").

        Returns:
            DecoderOutput.
        """
        current_vision_feats = backbone_out.get_current_feats(idx)

        high_res_features = backbone_out.get_high_res_features(current_vision_feats)

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
    ) -> DecoderOutput:
        """
        Decode a frame with memory: used for target frames;

        Args:
            backbone_out: backbone features.
            idx: absolute idx.
            memory_idx:   temporal index t (0-based).
            memory_bank:  dict of memory entries from previous frames.

        Returns:
            DecoderOutput
        """
        current_vision_feats = backbone_out.get_current_feats(idx)
        current_vision_pos_embeds = backbone_out.get_current_pos_embeds(idx)

        # take only the highest res feature map
        high_res_features = backbone_out.get_high_res_features(current_vision_feats)
        
        pix_feat_with_mem = self.sam._prepare_memory_conditioned_features(
            frame_idx=memory_idx,
            current_vision_feats=current_vision_feats[-1:],
            current_vision_pos_embeds=current_vision_pos_embeds[-1:],
            feat_sizes=backbone_out.feat_sizes[-1:],
            num_frames=memory_idx+1,
            memory_bank=memory_bank
        )

        # 仅对经过 memory 融合的 query 帧估计不确定性
        uncertainty = None
        if self.use_uncertainty:
            uncertainty = self.uncertainty_head(pix_feat_with_mem)

        dense_prompt = None
        if self.use_corr_dense_prompt:
            ref_proto = self._get_reference_part_proto(memory_bank)
            if ref_proto is not None:
                dense_prompt = self._build_corr_dense_prompt(
                    current_vision_feats=current_vision_feats,
                    feat_sizes=backbone_out.feat_sizes,
                    part_proto=ref_proto,
                )

        decoder_out: DecoderOutput = self.sam._forward_sam_heads(
            backbone_features=pix_feat_with_mem,
            mask_inputs=dense_prompt,
            high_res_features=high_res_features,
            multimask_output=True if memory_idx > 0 else False
        )
        decoder_out.uncertainty = uncertainty
        return decoder_out

    def _compute_memory_bank_dict(
        self, decoder_out: DecoderOutput, backbone_out: BackboneOutput, idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Encode current prediction into memory for later frames.

        Args:
            decoder_out: decoder output with high_res/low_res masks.
            backbone_out:  backbone features.
            idx: absolute idx.

        Returns:
            Memory entry dict.
        """
        current_vision_feats = backbone_out.get_current_feats(idx)
        feat_sizes = backbone_out.feat_sizes

        mem_feats, mem_pos = self.sam._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=decoder_out.high_res_masks,
            is_mask_from_pts=False,
        )
        part_proto = None
        enhanced_obj_ptr = decoder_out.obj_ptr
        if self.use_part_proto_ptr or self.use_corr_dense_prompt:
            part_proto = self._extract_part_prototype(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                high_res_masks=decoder_out.high_res_masks,
            )
            if part_proto is not None:
                part_proto = self.part_proto_proj(part_proto)
                if self.use_part_proto_ptr:
                    enhanced_obj_ptr = self._enhance_obj_ptr(decoder_out.obj_ptr, part_proto)
        return {
            "maskmem_features": mem_feats,
            "maskmem_pos_enc": mem_pos,
            "pred_masks": decoder_out.low_res_masks,
            "obj_ptr": enhanced_obj_ptr,
            "part_proto": part_proto,
        }

    def _extract_part_prototype(
        self,
        current_vision_feats: List[torch.Tensor],
        feat_sizes: List[Tuple[int, int]],
        high_res_masks: torch.Tensor,
    ) -> torch.Tensor | None:
        if high_res_masks is None:
            return None
        feat = current_vision_feats[-1]
        bsz = feat.size(1)
        channels = feat.size(2)
        h, w = feat_sizes[-1]
        feat_bchw = feat.permute(1, 2, 0).contiguous().view(bsz, channels, h, w)
        mask = torch.sigmoid(high_res_masks).float()
        mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
        denom = mask.sum(dim=(2, 3)).clamp_min(1e-6)
        proto = (feat_bchw * mask).sum(dim=(2, 3)) / denom
        return proto

    def _enhance_obj_ptr(self, obj_ptr: torch.Tensor, part_proto: torch.Tensor) -> torch.Tensor:
        return obj_ptr + part_proto

    def _get_reference_part_proto(self, memory_bank: Dict[int, Dict[str, torch.Tensor]]) -> torch.Tensor | None:
        if not memory_bank:
            return None
        first = memory_bank.get(0, None)
        if first is not None and first.get("part_proto", None) is not None:
            return first["part_proto"]
        for t in sorted(memory_bank.keys(), reverse=True):
            candidate = memory_bank[t].get("part_proto", None)
            if candidate is not None:
                return candidate
        return None

    def _build_corr_dense_prompt(
        self,
        current_vision_feats: List[torch.Tensor],
        feat_sizes: List[Tuple[int, int]],
        part_proto: torch.Tensor,
    ) -> torch.Tensor:
        feat = current_vision_feats[-1]
        bsz = feat.size(1)
        channels = feat.size(2)
        h, w = feat_sizes[-1]
        query_feat = feat.permute(1, 2, 0).contiguous().view(bsz, channels, h, w)

        query_norm = F.normalize(query_feat, dim=1)
        proto_norm = F.normalize(part_proto, dim=1).unsqueeze(-1).unsqueeze(-1)
        corr_map = (query_norm * proto_norm).sum(dim=1, keepdim=True)

        temperature = max(float(self.part_proto_temperature), 1e-6)
        corr_map = corr_map / temperature
        if self.corr_clamp is not None and self.corr_clamp > 0:
            corr_map = corr_map.clamp(min=-self.corr_clamp, max=self.corr_clamp)

        dense_size = self.sam.sam_prompt_encoder.mask_input_size
        corr_prompt = F.interpolate(corr_map, size=dense_size, mode="bilinear", align_corners=False)
        return corr_prompt

    def _forward_backbone(
        self, samples: torch.Tensor, orig_size: List[Tuple[int, int]]
    ) -> BackboneOutput:
        """
        Run SAM2 image encoder and prepare backbone features for decoding.

        Args:
            samples:  Tensor [B*T, C, H, W] after preprocessing.
            orig_size:   list of original frame sizes.

        Returns:
            BackboneOutput.
        """
        vis = self.sam.image_encoder.trunk(samples)
        feats, pos = self.sam.image_encoder.neck(vis)

        # discard lowest resolution
        feats, pos = feats[:-1], pos[:-1]

        feats[0] = self.sam.sam_mask_decoder.conv_s0(feats[0])
        feats[1] = self.sam.sam_mask_decoder.conv_s1(feats[1])

        bb = {
            "vision_features": feats[-1],
            "vision_pos_enc": pos,
            "backbone_fpn": feats,
        }
        vision_feats, vision_pos, sizes = self.sam._prepare_backbone_features(bb)
        return BackboneOutput(orig_size, vision_feats, vision_pos, sizes)


def build_sansa(
    sam2_version: str = 'large',
    adaptformer_stages: List[int] = [2, 3],
    channel_factor: float = 0.3,
    device: str = 'cuda',
    use_uncertainty: bool = True,
    use_part_proto_ptr: bool = False,
    use_corr_dense_prompt: bool = False,
    part_proto_temperature: float = 1.0,
    corr_clamp: float = 6.0,
) -> SANSA:
    assert sam2_version in SAM2_PATHS_CONFIG.keys(), f'wrong argument sam2_version: {sam2_version}'
    
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
        cfg.model.pred_obj_scores = False
        cfg.model.pred_obj_scores_mlp = False
        cfg.model.fixed_no_obj_ptr = False
        sam = instantiate(cfg.model, _recursive_=True)

    state_dict = torch.load(sam2_weights, map_location="cpu", weights_only=False)["model"]
    sam.load_state_dict(state_dict, strict=False)
    model = SANSA(
        sam=sam,
        device=torch.device(device),
        use_uncertainty=use_uncertainty,
        use_part_proto_ptr=use_part_proto_ptr,
        use_corr_dense_prompt=use_corr_dense_prompt,
        part_proto_temperature=part_proto_temperature,
        corr_clamp=corr_clamp,
    )

    # freeze everything except adapters and uncertainty head
    for name, p in model.named_parameters():
        p.requires_grad = (
            "adapter" in name
            or "uncertainty_head" in name
            or "part_proto_proj" in name
        )

    return model
