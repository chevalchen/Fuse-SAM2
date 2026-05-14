# Modify Log

## 背景与目标
- 基于 `修改思路.md` 落地两条增强路径：
  - 方向一：support part prototype 增强 memory `obj_ptr`。
  - 方向二：prototype-query 相关图作为 SAM dense prompt。
- 约束：默认关闭新功能，不影响现有功能路径（包括已存在的不确定性分支、可视化修复、geocrack 数据集）。

## 修改清单

### 1) 参数与开关
- 文件：`opts.py`
- 修改：
  - 新增 `--use_part_proto_ptr`（默认 `False`）
  - 新增 `--use_corr_dense_prompt`（默认 `False`）
  - 新增 `--part_proto_temperature`（默认 `1.0`）
  - 新增 `--corr_clamp`（默认 `6.0`）
- 目的：支持独立双开关和数值稳定控制，保持默认行为一致。

### 2) SANSA 主逻辑
- 文件：`models/sansa/sansa.py`
- 修改：
  - `SANSA.__init__` 新增四个配置参数并保存。
  - 按需新增 `part_proto_proj: Linear(256->256)`，并零初始化（稳定起步）。
  - 新增 `_extract_part_prototype(...)`：
    - 从顶层特征提取 `[B, C, H, W]`；
    - 对 `high_res_masks` 做 `sigmoid + resize`；
    - masked average pooling 得到 `[B, C]` prototype。
  - 新增 `_enhance_obj_ptr(...)`：残差增强 `obj_ptr + projected_proto`。
  - 新增 `_get_reference_part_proto(...)`：优先取 `memory_bank[0]`，缺失则回退最近有效 prototype。
  - 新增 `_build_corr_dense_prompt(...)`：
    - query feature 与 prototype 做通道余弦相似度；
    - 温度缩放 + clamp；
    - resize 到 `sam_prompt_encoder.mask_input_size`。
  - `_compute_memory_bank_dict(...)`：
    - 写入 `part_proto`；
    - 若启用方向一，写入增强后的 `obj_ptr`。
  - `_compute_decoder_out_w_mem(...)`：
    - 若启用方向二，构建 `dense_prompt` 并传给 `_forward_sam_heads(mask_inputs=...)`。
  - `build_sansa(...)`：
    - 透传新开关；
    - 冻结策略扩展为允许训练 `part_proto_proj`。

### 3) 训练/推理入口透传
- 文件：
  - `main.py`
  - `inference_fss.py`
- 修改：
  - 调用 `build_sansa(...)` 时透传新开关与数值参数。
- 目的：训练与评估统一配置、可直接做消融。

### 4) checkpoint 过滤同步
- 文件：`util/commons.py`
- 修改：
  - `adapter_state_dict(...)` 过滤键新增 `part_proto_proj`。
- 目的：避免新增可训练参数未保存导致恢复不一致。

## 兼容性说明
- 默认配置下（新开关均为 `False`）：
  - query 解码不注入 dense prompt；
  - memory bank 的 `obj_ptr` 不做增强；
  - 行为与原流程保持一致。
- 与现有模块关系：
  - `use_uncertainty` 分支不变，可与新开关任意组合。
  - geocrack、可视化修复路径无侵入改动。

## 最小验证建议
- 参数解析：
  - `python main.py --help` 确认四个新参数可见。
- 形状/前向检查（建议小 batch）：
  - 全关、仅方向一、仅方向二、全开 四组分别跑一次前向。
  - 观察 `pred_masks` 形状一致、无 NaN、无运行时报错。
- checkpoint 检查：
  - 保存后确认 state_dict 中包含 `part_proto_proj` 键（在相关开关启用时）。

## 风险与后续建议
- 风险：
  - 相关图分布与 SAM dense prompt 分布可能不匹配，需关注早期训练稳定性。
  - prototype 来源依赖当前 mask 质量，弱监督阶段可能引入噪声。
- 建议：
  - 做 2x2 消融：`ptr on/off` × `corr on/off`；
  - 记录每组 mIoU、收敛速度与小目标类别表现。
