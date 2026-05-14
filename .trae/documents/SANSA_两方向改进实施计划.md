# SANSA 两方向改进实施计划

## 1. Summary
- 目标：基于 `修改思路.md` 将两条改进方向落地到当前仓库实现，并保持默认行为不变（新能力默认关闭）。
- 范围：实现
  1) support 侧 `part prototype -> obj_ptr` 增强；
  2) query 侧 `prototype-query` 相关图作为 SAM dense mask prompt。
- 可控性：新增独立双开关，支持单独启用、联合启用与消融实验。
- 可审阅性：在仓库根目录新增 `Modify.md` 记录每一步修改、动机、影响与验证结果。

## 2. Current State Analysis
- 方案来源：`/data6/chensq/SANSA/修改思路.md` 已明确两条技术方向与插入点。
- 主模型入口：`/data6/chensq/SANSA/models/sansa/sansa.py`
  - 记忆写入位于 `_compute_memory_bank_dict`，当前仅写入 `maskmem_features/maskmem_pos_enc/pred_masks/obj_ptr`。
  - query 解码位于 `_compute_decoder_out_w_mem`，当前流程为 memory attention -> （可选不确定性头）-> `_forward_sam_heads`，未传 `mask_inputs`。
  - 参数冻结策略位于 `build_sansa`，当前仅训练 `adapter` 与 `uncertainty_head` 参数。
- SAM2 接口确认：`/data6/chensq/SANSA/models/sam2/modeling/sam2_base.py`
  - `_forward_sam_heads(..., mask_inputs=...)` 已原生支持 dense mask prompt，输入形状 `[B,1,H*16,W*16]`，会自动插值到 `sam_prompt_encoder.mask_input_size`。
  - `_prepare_memory_conditioned_features` 会读取 `memory_bank[t]["obj_ptr"]` 作为 encoder 端 object pointer token。
- 数据结构：`/data6/chensq/SANSA/models/sansa/model_utils.py`
  - `DecoderOutput` 已有 `pix_feat_with_mem`/`uncertainty` 可扩展字段。
- 参数与训练链路：
  - `opts.py` 已集中管理开关；
  - `main.py`、`inference_fss.py` 共用 parser 构建模型；
  - `util/commons.py::adapter_state_dict` 当前只保存 `adapter|uncertainty_head`，若新增可训练层需同步纳入。

## 3. Proposed Changes

### A. 新增配置开关（默认关闭）
- 修改文件：`/data6/chensq/SANSA/opts.py`
- 变更内容：
  - 新增 `--use_part_proto_ptr`（`store_true`，默认 `False`）：控制方向一。
  - 新增 `--use_corr_dense_prompt`（`store_true`，默认 `False`）：控制方向二。
  - 可选新增 `--part_proto_temperature`（float，默认 `1.0`）与 `--corr_clamp`（float，默认 `6.0`）用于数值稳定。
- 原因：满足“独立双开关 + 默认关闭”的需求，确保与现有行为兼容。

### B. build_sansa 与构造函数透传新开关
- 修改文件：
  - `/data6/chensq/SANSA/models/sansa/sansa.py`
  - `/data6/chensq/SANSA/main.py`
  - `/data6/chensq/SANSA/inference_fss.py`
- 变更内容：
  - `SANSA.__init__` 增加新布尔开关参数并保存。
  - `build_sansa(...)` 增加对应参数并向 `SANSA(...)` 透传。
  - `main.py` 与 `inference_fss.py` 在调用 `build_sansa` 时传入两个开关。
- 原因：训练/推理统一控制，同一 checkpoint 评估链路可复现。

### C. 方向一：Part Prototype 增强 obj_ptr
- 修改文件：`/data6/chensq/SANSA/models/sansa/sansa.py`
- 变更内容：
  - 在 `SANSA.__init__` 中按需创建 `self.part_proto_proj = nn.Linear(256, 256)`（仅当 `use_part_proto_ptr` 或 `use_corr_dense_prompt` 时需要 prototype）。
  - 新增私有工具函数（命名示例）：
    - `_extract_part_prototype(current_vision_feats, feat_sizes, high_res_mask)`：
      - 使用 top-level 特征 `(HW, B, C) -> (B, C, H, W)`；
      - 将 `decoder_out.high_res_masks` 先 `sigmoid` 后插值到特征分辨率；
      - 采用 masked average pooling 得到 `[B, C]` prototype；
      - 对小掩码面积加入 `eps` 防止除零。
    - `_enhance_obj_ptr(obj_ptr, part_proto)`：
      - 残差注入：`obj_ptr + part_proto_proj(part_proto)`。
  - 在 `_compute_memory_bank_dict` 中：
    - 计算 `part_proto`；
    - 当 `use_part_proto_ptr=True` 时写入增强后的 `obj_ptr`；
    - 将 `part_proto` 一并存入 `memory_bank` 供方向二复用。
- 原因：通过 memory encoder 已消费的 support mask 提取部件语义并注入 pointer token，增强 query cross-attention 对部件 identity 的可分性。

### D. 方向二：相关图作为 Dense Prompt
- 修改文件：`/data6/chensq/SANSA/models/sansa/sansa.py`
- 变更内容：
  - 在 `_compute_decoder_out_w_mem` 中，`pix_feat_with_mem` 生成后、`_forward_sam_heads` 前新增分支：
    - 从 `memory_bank` 读取 reference `part_proto`（优先 `memory_bank[0]`，若缺失回退最近可用条目）；
    - 取当前 query 顶层特征 `[B,C,H,W]`，与 `part_proto` 计算通道维 cosine 相似度，得到 `[B,1,H,W]` corr map；
    - 进行温度缩放/裁剪后，上采样到 SAM dense prompt size（建议直接对齐 `self.sam.sam_prompt_encoder.mask_input_size`）；
    - 以 `mask_inputs=corr_prompt` 传入 `_forward_sam_heads`。
  - 若开关关闭或 prototype 不可用，则保持原先无 `mask_inputs` 路径。
- 原因：给 SAM decoder 显式空间先验，补足隐式 memory attention 对小部件定位不足的问题。

### E. 参数冻结与 checkpoint 过滤同步
- 修改文件：
  - `/data6/chensq/SANSA/models/sansa/sansa.py`（`build_sansa` 冻结规则）
  - `/data6/chensq/SANSA/util/commons.py`（`adapter_state_dict`）
- 变更内容：
  - 冻结策略扩展为训练：`adapter | uncertainty_head | part_proto_proj`（如该模块存在）。
  - checkpoint 保存过滤同步纳入 `part_proto_proj`，避免“训练了但未保存”。
- 原因：保持训练参数集合与保存集合一致，防止恢复时性能退化。

### F. 结构化修改记录文档
- 新增文件：`/data6/chensq/SANSA/Modify.md`
- 记录模板（执行阶段填充）：
  - 变更背景与目标
  - 文件级修改清单（what/why）
  - 关键实现片段说明（prototype 提取、corr prompt 注入、兼容分支）
  - 兼容性说明（默认关闭、与 uncertainty/geocrack/visualization 的关系）
  - 验证结果（命令、关键日志、指标）
  - 风险与后续建议（消融矩阵、失败模式）

## 4. Assumptions & Decisions
- 决策已确认：
  - 两个方向都实现；
  - 新功能默认关闭；
  - 暴露独立双开关；
  - `Modify.md` 放仓库根目录。
- 假设：
  - 当前 `hidden_dim=256`（从 `sansa.py` 与 `sam2_base.py` 使用方式可见），`part_proto_proj` 采用 256->256。
  - prototype 以“当前帧预测高分辨率 mask”构建；对于 support 的 mask prompt 路径，`_use_mask_as_output` 已提供可用 `high_res_masks`。
- 兼容策略：
  - 任一新开关为 `False` 时不改变原逻辑。
  - 若 `use_corr_dense_prompt=True` 但缺少有效 prototype，安全回退原路径，不抛异常。

## 5. Verification Steps
- 静态检查：
  - 确认 `opts.py` 新参数可被训练/推理入口正确解析与透传。
  - 确认 `build_sansa` 可训练参数统计包含 `part_proto_proj`（仅开关开启时）。
  - 确认 `adapter_state_dict` 保存键包含新增模块键。
- 最小功能验证（不跑大训练）：
  - 使用一个小 batch 前向，分别验证四种配置：
    1) 全关；
    2) 仅方向一；
    3) 仅方向二；
    4) 两者全开；
  - 检查输出中 `pred_masks` 形状一致、无 NaN、推理不报错。
- 回归与兼容：
  - 默认配置（全关）下与修改前关键日志/行为一致。
  - 现有不确定性分支可正常共存（`use_uncertainty` 与新开关组合）。
- 文档完整性：
  - `Modify.md` 覆盖全部改动文件及验证结果，便于 reviewer 逐项对照。
