# CLAUDE.md - SANSA Agent Reference

## Purpose

This file is for agents, not humans.
Read this first to understand the repository layout, invariants, and safe modification points.

## Project Summary

SANSA is a few-shot semantic segmentation framework built on top of SAM2.

Core idea:
- keep all SAM2 weights frozen
- inject lightweight AdaptFormer adapters into selected Hiera stages
- reinterpret few-shot segmentation as a pseudo-video task: `supports -> query`
- optionally attach an uncertainty head on query memory-conditioned features

Current ground truth about the codebase:
- the real training/eval pipeline is **single-query per episode**
- sequence format is `[support_1, ..., support_K, query]`
- `args.J` exists, but the current code should **not** be read as true multi-query support
- uncertainty is query-only and optional

## Top-Level Map

```text
SANSA/
├── main.py                     # training entry
├── inference_fss.py            # evaluation / inference entry
├── engine.py                   # one-epoch training loop
├── opts.py                     # CLI arguments
├── datasets/                   # dataset classes and dataset factory
├── models/
│   ├── sam2/                   # Meta SAM2 code with local modifications
│   └── sansa/                  # SANSA-specific model code
├── util/                       # losses, metrics, logging, prompts, misc utils
├── pretrain/                   # adapter checkpoints
├── output/                     # experiment outputs
├── docs/                       # data preparation docs
└── sansa_demo.ipynb            # notebook demo
```

## Key Entry Points

### `main.py`
- builds SANSA via `build_sansa(...)`
- creates dataset / sampler / optimizer / LR scheduler
- runs `train_one_epoch(...)`
- runs validation through `eval_fss(...)`
- saves only the best checkpoint

### `inference_fss.py`
- builds the same model
- loads adapter-only checkpoint with `strict=False`
- evaluates mIoU on the selected dataset/fold
- supports horizontal-flip TTA via `--tta flip` (query frame only; probabilities averaged in sigmoid space)
- optionally writes visualizations

### `engine.py`
- builds the pseudo-video tensor from supports and one query
- builds support prompts via `build_prompt_dict(...)`
- computes segmentation loss
- if enabled, computes uncertainty NLL on the query frame

## Core Model Structure

### `models/sansa/sansa.py`

`SANSA` wraps a frozen `SAM2Base`.

High-level forward path:

```text
input: [B, T, C, H, W], where T = n_shots + 1
-> preprocess + flatten to [B*T, C, H, W]
-> SAM2 trunk + neck
-> iterate frames in order
   - support frames:
     - mask prompt: `sam._use_mask_as_output(...)`
     - other prompts: `_compute_decoder_out_no_mem(...)`
   - query frame:
     - `_compute_decoder_out_w_mem(...)`
     - read from memory bank
     - optional uncertainty prediction / feature recalibration
-> write each frame prediction back into memory
-> return `pred_masks`
```

Current uncertainty behavior:
- enabled by `use_uncertainty`
- only used on the query frame after memory conditioning
- predicts `log_var` from `pix_feat_with_mem`
- computes `confidence = 1 - sigmoid(log_var)`
- if `unc_recalib_active`, multiplies `pix_feat_with_mem * confidence` before decoding
- returns:
  - `pred_masks`
  - `uncertainty_feat` at feature resolution for NLL loss
  - `uncertainty` upsampled to image resolution for visualization/debugging

Important model methods:
- `_compute_decoder_out_no_mem(...)`: support decoding without memory
- `_compute_decoder_out_w_mem(...)`: query decoding with memory
- `_compute_memory_bank_dict(...)`: converts decoder outputs into SAM2 memory entries
- `_forward_backbone(...)`: SAM2 trunk + neck + feature packing
- `build_sansa(...)`: weight loading, adapter injection, parameter freezing

### `models/sansa/adapter.py`

AdaptFormer structure:

```text
LayerNorm -> down_proj -> ReLU -> up_proj -> scaled residual branch
```

Invariants:
- `up_proj` must stay zero-initialized for stable startup
- only selected Hiera stages receive adapters
- injection happens inside `MultiScaleBlock` in `models/sam2/modeling/backbones/hieradet.py`

### `models/sansa/uncertainty.py`

Current head:

```text
LayerNorm2d(C) -> Conv1x1(C -> C/4, bias=False) -> ReLU -> Conv1x1(C/4 -> 1) -> clamp(-6, 6)
```

Invariants:
- input is typically query `pix_feat_with_mem` with shape `[B, 256, 64, 64]`
- output is `log_var` with shape `[B, 1, 64, 64]`
- last conv is zero-initialized
- head is instantiated only when `use_uncertainty=True`

### `models/sansa/model_utils.py`

Contains:
- `DDPWrapper`
- `BackboneOutput`
- `DecoderOutput`

Important note:
- `DecoderOutput` includes optional `uncertainty`
- `move_to_cpu()` must also move uncertainty if present

## Dataset Contract

All datasets follow the episode pattern:
- sample one query and `K` supports
- return a batch dict containing:
  - `query_img`
  - `query_mask`
  - `support_imgs`
  - `support_masks`
  - `class_id`

Dataset factory:
- `datasets.build_dataset(name, image_set, args)`

Special cases:
- COCO/LVIS may also return base-class masks
- multi-dataset training uses `ConcatDataset` with weighted sampling

## Prompt System

Main file:
- `util/promptable_utils.py`

Prompt types:
- `mask`
- `box`
- `point`
- `scribble`
- `multi` (random training-time selection)

Core function:
- `build_prompt_dict(masks, prompt_type, n_shots, train_mode, device)`

## Losses

Main file:
- `util/losses.py`

Current losses:
- `loss_masks(...)`: segmentation loss = focal/BCE-style term + Dice
- `uncertainty_nll_loss(...)`: heteroscedastic NLL on query logits and `log_var`

Important invariants:
- segmentation supervision is query-only
- uncertainty loss uses `uncertainty_feat`, not the upsampled visualization map
- `log_var` is clamped to `[-6, 6]`

## CLI / Runtime

Main argument groups in `opts.py`:
- general: seed, device, resume
- I/O: output dir, experiment name
- data: dataset selection, multi-train config
- episode config: prompt, shots, fold, `J`
- model: SAM2 version, adapter stages, channel factor, uncertainty flags
- optimization: lr, weight decay, epochs, batch size, grad clip
- inference: threshold, visualize, tta, tta_scales, shot_permutations, postprocess_min_area, adaptive_threshold

Important runtime facts:
- `--use_uncertainty` defaults to off
- `--uncertainty_warmup_epochs` disables feature recalibration early while keeping NLL active
- `--tta` is a composable list (`nargs="+"`, choices: `none` / `flip` / `scale`); default `["none"]`
  - `flip`: horizontal-flip query frame, un-flip prediction, average probabilities
  - `scale`: center-crop zoom-in / zero-pad zoom-out on query at scales given by `--tta_scales` (default `[0.75, 1.25]`); 1.0 always included; each prediction is inverse-warped before averaging
  - Passes = |flip variants| × |scale variants|; e.g. `--tta flip scale` → 6 passes
- `--shot_permutations N`: ensemble N random support orderings (K>1 only); silently no-ops for K=1
- `--postprocess_min_area K`: drop connected components smaller than K pixels post-binarization
- `--adaptive_threshold otsu`: per-query Otsu threshold guarded by `max(prob) > 0.3` confidence floor; falls back to `--threshold` for empty-query episodes
- `--area_calibrate_threshold`: scales threshold by support-mask FG area ratio — small parts (< 3 % FG) get threshold × 0.65 (recall boost), large parts (> 25 %) get × 1.10 (precision boost); zero cost; composable with Otsu
- `--self_refine`: two-pass cascaded inference — Pass 1 result becomes a K+1-th pseudo-support with mask prompt; Pass 2 replaces Pass 1 (direct substitution); skipped when Pass-1 mask < `--self_refine_min_area` pixels (default 50) to avoid reinforcing empty predictions; ~2× cost; no model changes
- All inference flags default to their zero-cost values (no-op); the ensemble loop in `eval_fss` always runs but degenerates to a single forward pass at defaults
- DDP is supported; `--no_distributed` forces single-process mode

Inference helpers live in `util/tta_utils.py`:
- `build_tta_passes`, `_apply_scale` — Module A (Geometric TTA)
- `get_permutations`, `permute_supports` — Module B (Support permutation)
- `drop_small_components`, `binarize`, `_otsu_threshold` — Module C (Post-processing)
- `calibrate_threshold_by_support` — Module D (Support-area calibration)

## Checkpoint Rules

Main logic:
- save only the best validation checkpoint
- path: `output/<name_exp>/checkpoint_best.pth`
- saved model weights are filtered through `adapter_state_dict(...)`

Saved trainable modules:
- adapter weights
- uncertainty head weights

Checkpoint loading:
- uses `strict=False` to remain compatible with frozen SAM2 backbone weights

## Safe Modification Guide

### Add a new dataset
- create a dataset file in `datasets/`
- register it in `datasets/__init__.py`
- add it to `opts.py` dataset choices if needed

### Change adapter architecture
- edit `models/sansa/adapter.py`
- preserve stable initialization behavior
- if stage selection changes, update Hiera injection logic

### Change adapter injection point
- edit `models/sam2/modeling/backbones/hieradet.py`
- check both constructor-time adapter attachment and forward residual fusion

### Add a new prompt type
- extend `util/promptable_utils.py`
- update prompt choices in `opts.py`
- ensure `models/sansa/sansa.py` handles the new prompt path correctly

### Extend query-only auxiliary heads
- update `DecoderOutput`
- wire training loss in `engine.py`
- include trainable params in `build_sansa(...)`
- include them in checkpoint filtering in `util/commons.py`

## Do Not Assume

- do not assume true multi-query support from `J`
- do not assume SAM2 backbone weights are trainable
- do not assume every output branch should be saved in checkpoints
- do not assume visualization tensors are appropriate for training losses

## Current Architectural Invariants

- SAM2 core weights stay frozen
- only adapters and uncertainty head are trainable
- memory bank stores decoder mask outputs, not uncertainty outputs
- support frames build memory; query frame reads memory
- uncertainty is applied after memory conditioning and before mask decoding
- the production codepath is single-query
