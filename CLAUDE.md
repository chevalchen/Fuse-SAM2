# CLAUDE.md — SANSA 项目指南

## 项目概述

**SANSA**（Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation）是一个基于 SAM2 的小样本图像分割框架，发表于 **NeurIPS 2025 Spotlight**。

核心思想：冻结 SAM2（Segment Anything 2）的所有权重，仅在 Hiera 编码器的指定阶段插入轻量级 **AdaptFormer 适配器**，将 SAM2 的时序记忆机制重新用于 few-shot 语义分割（support → query）。

- **不微调 SAM2 权重**，只训练适配器
- 支持 point / box / scribble / mask 四种提示方式
- 同时支持物体级和零件级分割

---

## 目录结构

```
SANSA/
├── main.py                  # 训练入口
├── inference_fss.py         # 推理/评估入口
├── engine.py                # 单 epoch 训练逻辑
├── opts.py                  # 所有命令行参数定义
├── data/                    # 数据集存放目录（不含在代码库中）
├── datasets/                # 数据集 Dataset 类
│   ├── __init__.py          # build_dataset 工厂函数
│   ├── coco.py              # COCO-20i
│   ├── lvis.py              # LVIS-92i
│   ├── fss.py               # FSS-1000
│   ├── pascal_part.py       # Pascal-Part
│   ├── paco_part.py         # PACO-Part
│   ├── ade20k.py            # ADE20K
│   ├── isic.py              # ISIC 皮肤病变
│   ├── samplers.py          # 分布式采样器
│   └── transform_utils.py   # 多边形→掩码等工具
├── models/
│   ├── sam2/                # SAM2 原始代码（来自 Meta）
│   │   └── modeling/
│   │       ├── backbones/
│   │       │   ├── hieradet.py      # Hiera 编码器（含 adapter 注入逻辑）
│   │       │   └── image_encoder.py # FPN Neck
│   │       ├── sam/
│   │       │   └── transformer.py   # SAM Transformer（带 RoPE）
│   │       └── sam2_base.py         # SAM2 基类（_use_mask_as_output 等核心方法）
│   └── sansa/
│       ├── __init__.py
│       ├── sansa.py         # SANSA 主模型类 + build_sansa 工厂函数
│       ├── adapter.py       # AdaptFormer 适配器实现
│       └── model_utils.py   # DDPWrapper、BackboneOutput、DecoderOutput 数据类
├── util/
│   ├── commons.py           # 检查点加载/保存、日志等公共工具
│   ├── losses.py            # 损失函数（loss_masks）
│   ├── metrics.py           # AverageMeter、Evaluator（IoU 计算）
│   ├── misc.py              # 分布式初始化等杂项
│   ├── path_utils.py        # SAM2 权重路径配置
│   ├── promptable_utils.py  # prompt 构建（build_prompt_dict）、mask→box/point/scribble 转换
│   └── demo_sansa.py        # 交互式 demo 工具函数
├── pretrain/                # 预训练适配器权重存放目录
├── output/                  # 训练输出（检查点、日志）
├── docs/                    # 数据准备文档
├── assets/                  # README 图片/GIF
├── requirements.txt
└── sansa_demo.ipynb         # Colab 交互式演示
```

---

## 核心模块详解

### 1. SANSA 主模型（`models/sansa/sansa.py`）

`SANSA` 是一个 `nn.Module`，内部持有冻结的 `SAM2Base` 实例。

**前向流程（`forward`）**：
```
输入: [B, T, C, H, W]  (T = n_shots + J 张图像构成"伪视频")
  ↓ _preprocess_visual_features  → SAM2 归一化，展平为 [B*T, C, H, W]
  ↓ _forward_backbone             → Hiera + FPN 提取特征 → BackboneOutput
  ↓ 逐帧处理（循环 T 帧）：
      前 n_shots 帧（support）：
        mask prompt  → sam._use_mask_as_output（直接用 mask 编码进记忆库）
        其他 prompt  → _compute_decoder_out_no_mem（正常 SAM 解码）
      后 J 帧（query）：
        _compute_decoder_out_w_mem（从记忆库提取 memory，条件解码）
  ↓ 将每帧预测 mask 写入 memory_bank
输出: {"pred_masks": [B*T, H', W']}
```

**关键方法**：
- `_compute_decoder_out_w_mem`: 使用记忆库做条件解码（query 帧推理）
- `_compute_memory_bank_dict`: 将当前帧预测编码进记忆字典
- `_forward_backbone`: 调用 SAM2 编码器 + neck，返回 `BackboneOutput`

**`build_sansa(sam2_version, adaptformer_stages, channel_factor, device)`**：
- 自动下载 SAM2 权重（如不存在）
- 通过 Hydra 配置注入 adapter 参数
- **冻结所有参数，只解冻名称含 `"adapter"` 的参数**

---

### 2. AdaptFormer 适配器（`models/sansa/adapter.py`）

```
输入 x → LayerNorm → Linear↓(dim→bottleneck) → ReLU → Linear↑(bottleneck→dim) → × scale → 输出
```

- `bottleneck = int(dim * channel_factor)`（由 `--channel_factor` 控制大小）
- up_proj 权重初始化为零 → 训练开始时等效于恒等映射
- 注入位置：`MultiScaleBlock.forward` 中，在 MLP 残差之后并联加上 adapter 输出

适配器注入逻辑在 `models/sam2/modeling/backbones/hieradet.py` 的 `MultiScaleBlock` 中：
```python
if hasattr(self, "adapter"):
    x_adapt = self.adapter(x)
    x_mlp = self.drop_path(self.mlp(self.norm2(x)))
    x = x + x_mlp + x_adapt
```

---

### 3. 数据集（`datasets/`）

所有 Dataset 遵循统一 episode 采样范式：
- `__getitem__` 随机采样一个 (query, support×K) 元组
- 返回 batch dict，固定包含 `query_img`, `query_mask`, `support_imgs`, `support_masks`, `class_id`
- COCO/LVIS 额外返回 `base_masks`（用于基类 mask 监督，某些训练配置需要）

工厂函数 `datasets.build_dataset(name, image_set, args)` 统一创建数据集实例。

**多数据集训练**：使用 `torch.utils.data.ConcatDataset` + 按权重 WeightedRandomSampler（由 `--ds_weight` 控制）。

---

### 4. 命令行参数（`opts.py`）

所有实验参数集中在 `get_args_parser()` 中，分组如下：

| 参数组 | 关键参数 |
|--------|---------|
| 通用 | `--seed`, `--device`, `--resume` |
| 实验 I/O | `--output_dir`, `--name_exp` |
| 数据 | `--data_root`, `--dataset_file`, `--multi_train`, `--ds_weight` |
| 提示/镜头/折叠 | `--prompt`, `--shots`, `--J`, `--fold` |
| 模型 | `--sam2_version`, `--adaptformer_stages`, `--channel_factor` |
| 优化 | `--lr`, `--weight_decay`, `--epochs`, `--batch_size`, `--clip_max_norm` |
| 推理 | `--threshold`, `--visualize` |

---

### 5. 训练引擎（`engine.py`）

`train_one_epoch` 的核心步骤：
1. 从 batch 中取 `support_imgs/masks` + `query_img/mask`
2. 拼接成伪视频 `[B, T, C, H, W]`，T = shots + J
3. 调用 `build_prompt_dict` 将 support masks 转为指定 prompt 格式
4. 前向 `model(imgs, prompt_dict)`
5. 计算 `loss_masks`（二值交叉熵 + Dice loss）
6. 反向传播 + 梯度裁剪 + 优化器步进 + LR 调度器步进（每步更新）

---

### 6. Prompt 工具（`util/promptable_utils.py`）

`build_prompt_dict(support_masks, prompt_type, n_shots, train_mode, device)` 将 support mask 转换为 SAM2 所需格式：

- `mask`: 直接使用 mask tensor
- `box`: 从 mask 提取最大连通域的 bounding box（XYXY 格式，以 SAM2 point_coords 形式编码）
- `point`: 从 mask 随机采样前景点（训练时随机数量，测试时固定 20 个）
- `scribble`: 使用随机曲线笔画模拟人工涂抹
- `multi`: 训练时在四种方式中随机选一种

---

## 常用命令

### 环境安装
```bash
conda create --name sansa python=3.10 -y
conda activate sansa
pip install -r requirements.txt
```

### 训练（严格小样本分割）
```bash
# COCO-20i, fold 0
python main.py \
  --batch_size 32 \
  --name_exp train_coco_f0 \
  --dataset_file coco \
  --fold 0 \
  --adaptformer_stages 2 3 \
  --prompt mask
```

### 训练（多数据集泛化模型）
```bash
python main.py \
  --batch_size 32 \
  --name_exp train_generalist \
  --multi_train \
  --dataset_file lvis coco ade20k paco_part \
  --ds_weight 0.4 0.45 0.1 0.05 \
  --fold -1 \
  --adaptformer_stages 2 3 \
  --channel_factor 0.8 \
  --prompt mask
```

### 推理/评估
```bash
python inference_fss.py \
  --dataset_file coco \
  --fold 0 \
  --resume pretrain/adapter_coco_fold0.pth \
  --name_exp eval_coco_f0 \
  --shot 1 \
  --adaptformer_stages 2,3 \
  --prompt mask
```

### 可视化结果
在推理命令末尾添加 `--visualize`。

---

## 关键设计决策与注意事项

### Adapter 注入阶段（`--adaptformer_stages`）
- 默认 `[2, 3]`（Hiera 编码器的最后两个阶段）
- 通过 Hydra 配置动态注入到 `MultiScaleBlock`，无需修改模型代码

### Fold 约定
- `--fold F`：在第 F 折上评估，在其余折上训练（严格小样本协议）
- `--fold -1`：在所有折上训练（泛化模型，不做分离评估）
- FSS-1000 无折，省略 `--fold` 参数

### 检查点格式
- 仅在验证 mIoU 提升时保存最佳权重，路径：`output/<name_exp>/checkpoint_best.pth`
- 检查点仅包含适配器与不确定性头参数（通过 `adapter_state_dict` 过滤），并同时保存 `optimizer`、`lr_scheduler`、`epoch`、`best_miou` 与 `args`
- 取消每个 epoch 的常规保存（不再写入 `checkpoint.pth` 与按轮次编号的检查点）
- 加载时使用 `strict=False`，兼容 SAM2 基础权重

### 损失函数（`util/losses.py`）
- 二值交叉熵（BCE）+ Dice Loss 的加权组合
- 仅在 query 帧上计算损失（support 帧无标签监督）

### 分布式训练
- 支持 DDP（`torch.nn.parallel.DistributedDataParallel`）
- 通过 `--no_distributed` 强制单进程
- 支持 SLURM 环境（`util/misc.py` 中的 `init_distributed_mode`）

### 数据格式（伪视频）
SANSA 将小样本分割重构为"视频分割"：
```
[support_1, support_2, ..., support_K, query_1, ..., query_J]
  ↑ 提供 prompt，写入记忆库          ↑ 从记忆库读取，预测分割
```

---

## 依赖与外部代码

- **SAM2**（`models/sam2/`）：来自 Meta，已修改 `hieradet.py` 以支持 adapter 注入
- **AdaptFormer**：adapter 设计参考 [ShoufaChen/AdaptFormer](https://github.com/ShoufaChen/AdaptFormer)
- **Hydra**：用于 SAM2 模型配置管理
- **py3_wget**：自动下载 SAM2 预训练权重

---

## 修改代码时的注意事项

1. **新增数据集**：在 `datasets/` 下创建新 Dataset 类，并在 `datasets/__init__.py` 的 `build_dataset` 中注册，同时在 `opts.py` 的 `--dataset_file` choices 中添加。

2. **修改 adapter 结构**：`models/sansa/adapter.py` 中的 `Adapter` 类，注意保持 up_proj 零初始化以避免训练初期不稳定。

3. **调整 adapter 注入位置**：修改 `models/sam2/modeling/backbones/hieradet.py` 中 `Hiera.__init__` 的 `adaptformer_stages` 逻辑，以及 `MultiScaleBlock.forward` 中的残差融合方式。

4. **新增 prompt 类型**：在 `util/promptable_utils.py` 的 `build_prompt_dict` 和 `get_*_mask` 函数族中扩展，同时在 `opts.py` 的 `--prompt` choices 中添加，并在 `models/sansa/sansa.py` 的 `forward` 中处理新类型的分支逻辑。

5. **SAM2 权重路径**：在 `util/path_utils.py` 的 `SAM2_PATHS_CONFIG` 和 `SAM2_WEIGHTS_URL` 中管理。
