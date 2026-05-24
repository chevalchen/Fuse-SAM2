"""
Hyper-Prompted Part Adapter (HPPA).

Adapts target frame features conditioned on the reference frame's part semantics.
Inspiration: HyP-Adpt in Med-SA (Wu et al., MedIA 2025), which generates adapter
weights from prompt embeddings via a hypernetwork MLP.

Key departure from Med-SA:
  - "Prompt" is not a click/box; it is the masked region of the reference image.
  - Adapter is placed before Memory Attention (not inside the mask decoder).
  - Handles k-shot by receiving pre-averaged reference features/masks.

Tensor conventions (matching SANSA's internal format):
  - Visual features: [HW, B, D]  (seq-first, as used by MemoryAttention)
  - Spatial masks:   [B, 1, H, W]  (already resized to feature resolution)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HPPA(nn.Module):
    """
    Hyper-Prompted Part Adapter.

    Args:
        dim:            Feature dimension (256 for all SAM2 variants after FPN).
        part_token_dim: Hidden dim of the part token extracted from reference.
                        Larger → more expressive but more params (try 64/128).
        bottleneck_dim: Bottleneck rank K for the dynamic weight matrices W_down, W_up.
    """

    def __init__(
        self,
        dim: int = 256,
        part_token_dim: int = 128,
        bottleneck_dim: int = 64,
    ):
        super().__init__()
        self.dim = dim
        self.bottleneck_dim = bottleneck_dim

        # Step 1 — extract a part-specific token from masked reference features
        self.part_extractor = nn.Sequential(
            nn.Linear(dim, part_token_dim),
            nn.LayerNorm(part_token_dim),
            nn.GELU(),
        )

        # Step 2 — generate dynamic adapter weights from part token
        # W_down: [D → K],  W_up: [K → D]
        self.W_down_gen = nn.Linear(part_token_dim, dim * bottleneck_dim)
        self.W_up_gen   = nn.Linear(part_token_dim, bottleneck_dim * dim)

        self.norm  = nn.LayerNorm(dim)
        # scale starts at 0 → adapter is identity at init, learned during training
        self.scale = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        # Small-variance init so the generated weights start near zero
        nn.init.normal_(self.W_down_gen.weight, std=0.01)
        nn.init.zeros_(self.W_down_gen.bias)
        nn.init.normal_(self.W_up_gen.weight, std=0.01)
        nn.init.zeros_(self.W_up_gen.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        target_feat: torch.Tensor,   # [HW, B, D]
        ref_feat:    torch.Tensor,   # [HW, B, D]  (avg over k shots)
        ref_mask:    torch.Tensor,   # [B, 1, H_feat, W_feat]  soft mask ∈ [0,1]
    ) -> torch.Tensor:
        """
        Returns:
            Tensor [HW, B, D] — part-conditioned target features.
        """
        HW, B, D = target_feat.shape
        K = self.bottleneck_dim

        # ── Step 1: masked average pooling ref_feat → part token ──────────
        # ref_mask [B,1,H,W] → flatten & transpose to [HW, B, 1]
        mask_flat = ref_mask.flatten(2).permute(2, 0, 1)      # [HW, B, 1]

        masked_sum      = (ref_feat * mask_flat).sum(dim=0)   # [B, D]
        count           = mask_flat.sum(dim=0).clamp(min=1.0) # [B, 1]
        part_token_raw  = masked_sum / count                   # [B, D]

        part_token = self.part_extractor(part_token_raw)       # [B, part_token_dim]

        # ── Step 2: generate dynamic weights ──────────────────────────────
        W_down = self.W_down_gen(part_token).view(B, D, K)    # [B, D, K]
        W_up   = self.W_up_gen(part_token).view(B, K, D)      # [B, K, D]

        # ── Step 3: dynamic bottleneck on target features ─────────────────
        tgt = self.norm(target_feat).permute(1, 0, 2)         # [B, HW, D]
        x   = torch.bmm(tgt, W_down)                          # [B, HW, K]
        x   = F.relu(x)
        x   = torch.bmm(x, W_up)                              # [B, HW, D]
        x   = x.permute(1, 0, 2)                              # [HW, B, D]

        # ── Step 4: learned residual ───────────────────────────────────────
        return target_feat + self.scale * x
