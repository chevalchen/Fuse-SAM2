# Experiment Configurations

## Pascal-Part Fine-tune

Fine-tune the generalist SANSA checkpoint on Pascal-Part, training only HPPA + MLPA
(AdaptFormers and SAM2 backbone remain frozen).

```bash
python main.py \
  --name_exp pascal_part_hppa_mlpa \
  --dataset_file pascal_part \
  --fold -1 \
  --sam2_version large \
  --adaptformer_stages 2 3 \
  --channel_factor 0.8 \
  --prompt mask \
  --shots 1 \
  --J 1 \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --clip_max_norm 1.0 \
  --resume pretrain/generalist_checkpoint.pth
```

Key hyper-parameters and rationale:

| Parameter       | Value  | Rationale |
|-----------------|--------|-----------|
| `--lr`          | 1e-4   | 10× smaller than training from scratch; modules start at identity (scale=0) |
| `--batch_size`  | 16     | Reduced vs. 32 to fit extra memory from HPPA/MLPA on 80 GB GPU |
| `--epochs`      | 30     | Short fine-tune; cosine LR with T_max = 30 × len(dataset) |
| `--weight_decay`| 1e-4   | Standard AdamW for small adapter fine-tuning |
| `--shots`       | 1      | 1-shot Pascal-Part protocol (can repeat for 5-shot) |

---

## Ablation Experiment Variants

Run each variant with the same Pascal-Part setup above, changing only the flags below.

### Variant 1 — Baseline (SANSA only, no new modules)

```bash
python main.py \
  --name_exp ablation_baseline \
  --no_mlpa --no_hppa \
  ... (same Pascal-Part flags as above)
```

*Trainable params*: AdaptFormer only (~2 M).

---

### Variant 2 — HPPA only

```bash
python main.py \
  --name_exp ablation_hppa_only \
  --no_mlpa \
  ... (same Pascal-Part flags)
```

*Trainable params*: AdaptFormer + HPPA (~6 M).

---

### Variant 3 — MLPA only

```bash
python main.py \
  --name_exp ablation_mlpa_only \
  --no_hppa \
  ... (same Pascal-Part flags)
```

*Trainable params*: AdaptFormer + MLPA (~3 M).

---

### Variant 4 — Full model (HPPA + MLPA) ← primary

```bash
python main.py \
  --name_exp ablation_hppa_mlpa \
  ... (same Pascal-Part flags, both modules active by default)
```

*Trainable params*: AdaptFormer + HPPA + MLPA (~7 M total).

---

### Variant 5 — Full model + Boundary Loss

Requires adding boundary loss support to `util/losses.py` and `engine.py`.

Boundary loss (Sobel-based):
```python
# In engine.py, after loss_masks():
if args.use_boundary_loss:
    L_boundary = boundary_loss(pred_masks, gt_masks)   # see util/losses.py
    loss = loss + args.boundary_loss_weight * L_boundary
```

```bash
python main.py \
  --name_exp ablation_hppa_mlpa_boundary \
  --use_boundary_loss \
  --boundary_loss_weight 0.5 \
  ... (same Pascal-Part flags)
```

*Expected effect*: sharper part boundaries (hand/arm, head/neck separation).

---

## Notes

- **To add `--no_mlpa` / `--no_hppa` / `--use_boundary_loss` CLI flags**, extend
  `opts.py`:
  ```python
  parser.add_argument('--no_mlpa',   action='store_true')
  parser.add_argument('--no_hppa',   action='store_true')
  parser.add_argument('--use_boundary_loss',    action='store_true')
  parser.add_argument('--boundary_loss_weight', type=float, default=0.5)
  ```
  Then pass `use_mlpa=not args.no_mlpa, use_hppa=not args.no_hppa` to `build_sansa()`.

- **Evaluation**: use `inference_fss.py --dataset_file pascal_part` with the same
  `--adaptformer_stages 2 3` and new module flags.

- **5-shot ablation**: add `--shots 5` (HPPA averages all 5 reference features/masks
  automatically via `_build_ref_info`).
