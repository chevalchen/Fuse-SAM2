# TASK.md — 在位置 B 引入不确定性模块

## 任务目标

在 `_compute_decoder_out_w_mem` 中、Memory Attention 融合之后、MaskDecoder 解码之前，
插入一个轻量**不确定性估计模块（UncertaintyHead）**，使模型能够：

1. 输出逐像素不确定性图（uncertainty map），与预测 mask 一同返回
2. 将不确定性图作为辅助损失（KL / NLL）参与训练，引导模型在 support-query 对齐置信度低的区域给出较高不确定性
3. 不影响 SAM2 任何冻结权重；新增参数随 adapter 一同可训练

---

## 涉及文件清单

| 文件 | 操作类型 |
|------|---------|
| `models/sansa/uncertainty.py` | **新建** — UncertaintyHead 模块定义 |
| `models/sansa/model_utils.py` | **修改** — DecoderOutput 增加 `uncertainty` 字段 |
| `models/sansa/sansa.py` | **修改** — SANSA 集成 UncertaintyHead；`_compute_decoder_out_w_mem` 调用它 |
| `models/sansa/__init__.py` | **修改** — 导出 `uncertainty` 模块 |
| `util/losses.py` | **修改** — 新增 `uncertainty_loss` 函数 |
| `engine.py` | **修改** — 训练循环中加入不确定性损失项 |
| `opts.py` | **修改** — 新增不确定性相关命令行参数 |

---

## 详细改动说明

---

### 1. 新建 `models/sansa/uncertainty.py`

定义 `UncertaintyHead`，输入为 `pix_feat_with_mem`（形状 `[B, C, H, W]`，C=256，H=W=64），
输出为逐像素对数方差 `log_var`（形状 `[B, 1, H, W]`）。

结构（从简到繁可选其一，默认用选项 A）：

**选项 A（推荐）：两层轻量 Conv**
```
Conv2d(C, C//4, 1) → LayerNorm2d(C//4) → ReLU → Conv2d(C//4, 1, 1)
```
- 无 Dropout，参数量约 16K（C=256 时）
- 输出不加激活，表示 log_var（可为任意实数）

**选项 B：带 Dropout 的 MC Dropout 版本**（供对比实验，暂不实现）

实现要点：
- 类名 `UncertaintyHead(nn.Module)`
- `__init__(self, in_channels: int)` — `in_channels` 默认 256
- `forward(self, x: Tensor) -> Tensor` — 返回 `log_var`，shape `[B, 1, H, W]`
- 权重初始化：最后一层 `nn.init.zeros_`，使训练初期 log_var≈0（方差≈1），保持稳定

---

### 2. 修改 `models/sansa/model_utils.py`

在 `DecoderOutput` dataclass 中新增一个可选字段：

```python
uncertainty: Optional[Tensor] = None   # [B, 1, H, W]，逐像素 log_var；仅 query 帧有值
```

放在现有字段 `pix_feat_with_mem` 之后。`__post_init__` 和 `move_to_cpu` 均需同步处理该字段：
- `move_to_cpu` 中将 `"uncertainty"` 加入遍历的字段列表

---

### 3. 修改 `models/sansa/sansa.py`

#### 3a. `SANSA.__init__` 中实例化 UncertaintyHead

```python
from models.sansa.uncertainty import UncertaintyHead

class SANSA(nn.Module):
    def __init__(self, sam: SAM2Base, device: torch.device, use_uncertainty: bool = True):
        super().__init__()
        self.sam = sam
        self.device = device
        self.use_uncertainty = use_uncertainty
        if use_uncertainty:
            self.uncertainty_head = UncertaintyHead(in_channels=256)
```

#### 3b. `_compute_decoder_out_w_mem` 中调用 UncertaintyHead

在现有代码：
```python
pix_feat_with_mem = self.sam._prepare_memory_conditioned_features(...)

decoder_out: DecoderOutput = self.sam._forward_sam_heads(
    backbone_features=pix_feat_with_mem, ...
)
return decoder_out
```

修改为（在两行之间插入）：
```python
pix_feat_with_mem = self.sam._prepare_memory_conditioned_features(...)

# ── 不确定性估计（仅 query 帧，即经过 memory 融合的帧）──────────────
uncertainty = None
if self.use_uncertainty:
    uncertainty = self.uncertainty_head(pix_feat_with_mem)  # [B, 1, H, W]
# ───────────────────────────────────────────────────────────────────

decoder_out: DecoderOutput = self.sam._forward_sam_heads(
    backbone_features=pix_feat_with_mem, ...
)
decoder_out.uncertainty = uncertainty
return decoder_out
```

#### 3c. `SANSA.forward` 中收集不确定性图

在 `outputs` dict 中新增 `"uncertainties"` 列表，与 `"masks"` 并行收集：

```python
outputs = {"masks": [], "uncertainties": []}
```

在循环内 `outputs["masks"].append(decoder_out.masks[0])` 之后追加：
```python
if decoder_out.uncertainty is not None:
    outputs["uncertainties"].append(decoder_out.uncertainty)
```

在 `forward` 返回前：
```python
result = {"pred_masks": masks}
if outputs["uncertainties"]:
    uncertainty_maps = torch.cat(outputs["uncertainties"])           # [B*T_query, 1, H, W]
    uncertainty_maps = F.interpolate(
        uncertainty_maps, size=orig_size[0], mode='bilinear', align_corners=False
    )
    result["uncertainty"] = uncertainty_maps
return result
```

注意：只有 query 帧（`idx >= n_shots`）才有 `decoder_out.uncertainty`，support 帧的 uncertainty 为 None，不收集进列表，因此 `uncertainty_maps` 的帧数为 `T - n_shots`，与 `loss_masks` 中 `num_frames` 逻辑对齐。

#### 3d. `build_sansa` 函数签名更新

```python
def build_sansa(
    sam2_version: str = 'large',
    adaptformer_stages: List[int] = [2, 3],
    channel_factor: float = 0.3,
    device: str = 'cuda',
    use_uncertainty: bool = True,       # ← 新增
) -> SANSA:
```

实例化时传入：
```python
model = SANSA(sam=sam, device=torch.device(device), use_uncertainty=use_uncertainty)
```

冻结逻辑保持不变（名称含 `"adapter"` 或 `"uncertainty_head"` 均可训练）：
```python
for name, p in model.named_parameters():
    p.requires_grad = ("adapter" in name or "uncertainty_head" in name)
```

---

### 4. 修改 `models/sansa/__init__.py`

在现有导出行之后新增：
```python
from .uncertainty import UncertaintyHead
```

并更新 `__all__`：
```python
__all__ = ['sansa', 'adapter', 'model_utils', 'uncertainty']
```

---

### 5. 修改 `util/losses.py`

新增 `uncertainty_loss` 函数，实现**基于预测方差的 NLL（负对数似然）损失**：

```
L_unc = mean( 0.5 * exp(-log_var) * (pred - gt)^2 + 0.5 * log_var )
```

函数签名：
```python
def uncertainty_loss(
    pred_masks: torch.Tensor,    # [B, T_query, H, W]，logits（未 sigmoid）
    gt_masks: torch.Tensor,      # [B, T_query, H, W]，二值 0/1
    log_var: torch.Tensor,       # [B*T_query, 1, H, W] 或 [B, T_query, H, W]
    num_frames: int,
) -> torch.Tensor:
```

实现要点：
- `pred_masks` 先过 sigmoid 得到概率
- 误差平方项 `err = (prob - gt)^2`
- 损失 = `(0.5 * torch.exp(-log_var) * err + 0.5 * log_var).mean()`
- 返回标量 tensor
- 加入 log_var 的上下界 clamp（如 `log_var.clamp(-6, 6)`）防止训练初期数值爆炸

---

### 6. 修改 `engine.py`

在 `train_one_epoch` 的损失计算段，当 `outputs` 中存在 `"uncertainty"` 键时，计算并累加不确定性损失：

```python
losses_dict = loss_masks(outputs["pred_masks"], masks, num_frames=use_frames)

# 不确定性损失（如果模型输出了 uncertainty）
if "uncertainty" in outputs and outputs["uncertainty"] is not None:
    from util.losses import uncertainty_loss
    unc_loss = uncertainty_loss(
        pred_masks=...,   # 从 outputs["pred_masks"] 取 query 帧对应部分
        gt_masks=...,     # 从 masks 取 query 帧对应部分
        log_var=outputs["uncertainty"],
        num_frames=use_frames,
    )
    losses_dict["loss_uncertainty"] = unc_loss * args.uncertainty_loss_weight

loss = sum(losses_dict.values())
```

注意：`uncertainty_loss` 的 `pred_masks` 和 `gt_masks` 需要与 `loss_masks` 保持一致的帧切片逻辑（`start = max(0, T - num_frames)`），建议在 `engine.py` 中统一做切片后再传入。

---

### 7. 修改 `opts.py`

在 `get_args_parser()` 中新增以下参数（建议放在 `# SAM2 / Backbone` 分组下）：

```python
parser.add_argument(
    "--use_uncertainty", action="store_true", default=False,
    help="Enable UncertaintyHead at position B (after Memory Attention)."
)
parser.add_argument(
    "--uncertainty_loss_weight", type=float, default=0.1,
    help="Weight for the uncertainty NLL loss term (lambda in total loss)."
)
```

同时在调用 `build_sansa` 的所有位置（`main.py` 和 `inference_fss.py`）传入 `use_uncertainty=args.use_uncertainty`。

---

## 数据流与 Tensor Shape 速查

```
pix_feat_with_mem              [B=1, C=256, H=64, W=64]
        ↓ UncertaintyHead
log_var (uncertainty)          [B=1, 1, H=64, W=64]
        ↓ _forward_sam_heads（不改变）
low_res_masks                  [B=1, 1, H=256, W=256]
high_res_masks                 [B=1, 1, H=1024, W=1024]
        ↓ SANSA.forward 插值到 orig_size
pred_masks（返回）             [B*T, H_orig, W_orig]
uncertainty（返回）            [B*T_query, 1, H_orig, W_orig]
```

---

## 训练命令示例（新增参数）

```bash
python main.py \
  --batch_size 4 \
  --name_exp pascalpart_f1_u \
  --dataset_file pascal_part \
  --fold 0 \
  --resume pretrain/adapter_sansa_universal.pth \
  --adaptformer_stages 2 3 \
  --channel_factor 0.3 \
  --prompt mask \
  --lr 1e-5 \
  --epochs 3 \
  --use_uncertainty \
  --uncertainty_loss_weight 0.1
```

---

## 验收标准

1. `--use_uncertainty` 未设置时，所有行为与原代码完全一致（`outputs` 中无 `"uncertainty"` 键，损失函数无变化）
2. `--use_uncertainty` 设置后：
   - `model.uncertainty_head` 参数数量约 16K（C=256），`requires_grad=True`
   - `outputs["uncertainty"]` shape 为 `[T_query, 1, H_orig, W_orig]`
   - `losses_dict` 中出现 `"loss_uncertainty"` 键
   - `inference_fss.py` 中 `with torch.no_grad()` 内仍能正常前向（`uncertainty` 字段存在但不参与损失）
3. 不修改 `models/sam2/` 下任何文件
4. 检查点保存（`adapter_state_dict`）需确认 `uncertainty_head` 权重被包含

---

## 检查点兼容性说明

`util/commons.py` 中的 `adapter_state_dict` 函数当前过滤逻辑为仅保留名称含 `"adapter"` 的参数。
需确认该函数是否需要扩展为同时保留 `"uncertainty_head"` 的参数；
若不修改，`uncertainty_head` 权重将不被保存，重新加载后需重训。
**建议**：将过滤条件改为 `"adapter" in name or "uncertainty_head" in name`。