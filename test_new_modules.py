"""
Minimal sanity-check for the HPPA + MLPA integration.

What this script does
---------------------
1. Builds SANSA with the new modules (uses dummy tensors — no real checkpoint needed
   if you pass --skip_weights, otherwise loads the generalist checkpoint).
2. Constructs a synthetic (reference, target) pair at the expected resolution.
3. Runs a full forward pass through the model.
4. Prints a parameter breakdown: frozen vs. trainable, and the count for each new module.

Usage
-----
# With a real checkpoint (set SANSA_CKPT env var or pass --resume):
python test_new_modules.py --resume pretrain/generalist_checkpoint.pth

# Without a checkpoint (random weights, just for shape/param checks):
python test_new_modules.py --skip_weights

Requirements
------------
Same conda env as SANSA training.  The script does NOT run actual SAM2 inference
(the model forward is called, but no GPU acceleration is assumed).
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F


# ── helpers ──────────────────────────────────────────────────────────────────

def count_params(module):
    total     = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def make_prompt_dict(n_shots: int, H: int, W: int, device):
    """Build a minimal prompt_dict with mask prompts for all support frames."""
    prompt_dict = {"shots": n_shots}
    # Single batch item (B=1)
    per_sample = {}
    for i in range(n_shots):
        # Random binary mask: 1 in a centred 50×50 patch
        mask = torch.zeros(1, H, W, device=device)
        mask[0, H//4:3*H//4, W//4:3*W//4] = 1.0
        per_sample[i] = {"prompt": mask, "prompt_type": "mask"}
    prompt_dict[0] = per_sample
    return prompt_dict


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",       default=None, help="Path to SANSA checkpoint")
    parser.add_argument("--skip_weights", action="store_true",
                        help="Skip checkpoint loading (random weights, shape-only test)")
    parser.add_argument("--sam2_version", default="large")
    parser.add_argument("--device",       default="cpu",
                        help="'cpu' or 'cuda' — CPU works for shape / param checks")
    parser.add_argument("--image_size",   type=int, default=1024)
    parser.add_argument("--n_shots",      type=int, default=1)
    parser.add_argument("--n_query",      type=int, default=1,
                        help="Number of query frames (J in SANSA notation)")
    # ablation flags
    parser.add_argument("--no_mlpa", action="store_true")
    parser.add_argument("--no_hppa", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── 1. Build model ────────────────────────────────────────────────────
    print("Building SANSA with MLPA + HPPA …")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from models.sansa.sansa import build_sansa
    from util.commons import resume_from_checkpoint

    model = build_sansa(
        sam2_version=args.sam2_version,
        device=str(device),
        use_mlpa=not args.no_mlpa,
        use_hppa=not args.no_hppa,
    )
    model.to(device)
    model.eval()

    if args.resume and not args.skip_weights:
        print(f"Loading checkpoint: {args.resume}")
        resume_from_checkpoint(args.resume, model, optimizer=None, lr_scheduler=None,
                               args=argparse.Namespace(start_epoch=0))

    # ── 2. Parameter summary ──────────────────────────────────────────────
    total, trainable = count_params(model)
    print(f"\n{'='*55}")
    print(f"  Total parameters    : {total:>12,}")
    print(f"  Trainable parameters: {trainable:>12,}")
    if model.mlpa is not None:
        n, _ = count_params(model.mlpa)
        print(f"  └─ MLPA             : {n:>12,}")
    if model.hppa is not None:
        n, _ = count_params(model.hppa)
        print(f"  └─ HPPA             : {n:>12,}")
    print(f"  (Target ≈ 5 M new trainable params)")
    print(f"{'='*55}\n")

    # ── 3. Build synthetic input ──────────────────────────────────────────
    T = args.n_shots + args.n_query
    H = W = args.image_size
    # [B=1, T, C=3, H, W] — random RGB images in [0, 1]
    images = torch.rand(1, T, 3, H, W, device=device)
    prompt_dict = make_prompt_dict(args.n_shots, H, W, device)

    # ── 4. Forward pass ───────────────────────────────────────────────────
    print("Running forward pass … (this may take ~30 s on CPU for SAM2-large)")
    with torch.no_grad():
        output = model(images, prompt_dict)

    pred = output["pred_masks"]
    print(f"Forward pass succeeded!")
    print(f"  Output shape: {tuple(pred.shape)}")
    print(f"  Value range : [{pred.min():.3f}, {pred.max():.3f}]")
    if "uncertainty" in output:
        print(f"  Uncertainty : {tuple(output['uncertainty'].shape)}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
