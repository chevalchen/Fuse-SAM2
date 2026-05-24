"""
Multi-Layer Part-aware Aggregation (MLPA).

Aggregates multi-scale Hiera backbone features (before the FPN neck) using
lightweight cross-attention, enriching the deep semantic feature with spatial
detail from shallower stages.

Inspiration: multi-layer aggregation decoder in 3DSAM-Adapter (Gong et al.,
MedIA 2024), which collects intermediate ViT features and concatenates them
before decoding.  Key departures:
  - 2-D spatial (not 3-D volumetric).
  - Aggregation is placed *before* Memory Attention (encoder-side), not in
    the mask decoder.
  - Uses cross-attention (deep queries shallow K/V) instead of simple concat.
  - Output is added residually to the FPN feature consumed by Memory Attention.

Stage index convention (Hiera-large, 1024×1024 input):
  stage 0 → 256×256 × 144 ch   (not used here)
  stage 1 → 128×128 × 288 ch   ← shallow spatial detail
  stage 2 →  64× 64 × 576 ch   ← mid-level features
  stage 3 →  32× 32 × 1152 ch  ← deep semantics (query anchor)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MLPA(nn.Module):
    """
    Multi-Layer Part-aware Aggregation.

    Args:
        stage_dims:  Channel dimensions of the Hiera stages to aggregate.
                     E.g. [288, 576, 1152] for SAM2-large stages 1-2-3.
                     The *last* entry is treated as the "deep" anchor.
        target_dim:  Output channel dimension (256 = SAM2 FPN hidden dim).
        num_heads:   Attention heads for cross-attention.
    """

    def __init__(
        self,
        stage_dims: List[int],
        target_dim: int = 256,
        num_heads:  int = 8,
    ):
        super().__init__()
        assert len(stage_dims) >= 2, "Need at least 2 stages (one deep, one shallow)."

        # Project each stage to target_dim with 1×1 conv + GroupNorm
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, target_dim, kernel_size=1, bias=False),
                nn.GroupNorm(1, target_dim),  # instance norm over C (= LayerNorm for CHW)
            )
            for d in stage_dims
        ])

        # Cross-attention: deep (stage_dims[-1]) queries shallow K/V
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=target_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm_q  = nn.LayerNorm(target_dim)
        self.norm_kv = nn.LayerNorm(target_dim)

        # Learned residual scale — starts at 0 (identity) and is trained
        self.scale = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    def forward(self, stage_feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            stage_feats: list of [B, C_i, H_i, W_i] tensors in ascending
                         depth order (shallowest first, deepest last).
                         Lengths must match ``len(stage_dims)``.

        Returns:
            [B, target_dim, H_deep, W_deep]  — deep feature enhanced with
            multi-scale context (same spatial resolution as the deepest input).
        """
        # ── Step 1: project every stage to target_dim ─────────────────────
        projected = [proj(f) for proj, f in zip(self.projectors, stage_feats)]
        # projected[i]: [B, target_dim, H_i, W_i]

        # ── Step 2: deep feature is the query anchor ───────────────────────
        deep = projected[-1]                 # [B, C, H_deep, W_deep]
        B, C, H, W = deep.shape

        # ── Step 3: downsample shallower stages to deep resolution ────────
        # (avg-pool → cheap, no extra params; alternatives: stride conv, bilinear)
        shallow_aligned = [
            F.adaptive_avg_pool2d(p, (H, W)) for p in projected[:-1]
        ]  # each [B, C, H, W]

        # ── Step 4: cross-attention (deep queries, shallow K/V) ───────────
        deep_seq    = deep.flatten(2).permute(0, 2, 1)        # [B, HW, C]
        shallow_kv  = torch.cat(
            [p.flatten(2).permute(0, 2, 1) for p in shallow_aligned], dim=1
        )  # [B, N_shallow_stages × HW, C]

        enhanced, _ = self.cross_attn(
            self.norm_q(deep_seq),
            self.norm_kv(shallow_kv),
            self.norm_kv(shallow_kv),
        )  # [B, HW, C]

        # ── Step 5: residual + reshape ─────────────────────────────────────
        out = deep_seq + self.scale * enhanced          # [B, HW, C]
        return out.permute(0, 2, 1).reshape(B, C, H, W)
