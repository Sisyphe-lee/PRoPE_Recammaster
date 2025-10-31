# Change Log
## v0.2.14 @codex - 2025-10-31

### 修复
- `src/lightning_trainer.py`: 调整 `LightningModelForTrain.decode_video` 签名及其验证调用方式，统一透传参数顺序，避免在复用 `VideoDecoder.decode_and_create_combined_video` 时对 `condition_latents` 同时使用位置参数与关键字参数导致验证崩溃。
- 调整denoise timestep，在i2v下，follow官方代码，给第一帧timestep置0

## v0.2.13 @codex - 2025-10-30

### 变更
- 训练入口重构：将 `LightningModelForTrain` 独立到 `src/lightning_trainer.py`，`src/train_recammaster.py` 专注于 CLI 解析与调度，便于后续复用。
- I2V 条件逻辑：在训练/验证阶段固定首帧作为图像条件，不参与加噪与损失，采样循环每步都会写回真实首帧，保持与 Wan2.2 官方策略一致。
- 训练脚本：`scripts/train.sh` 新增 `-v/--val-size` 与 `-i/--val-check-interval-batches` 短选项，默认值仍为 `12/50`，方便在实验脚本中覆写。
- 重构代码，去除冗余逻辑

## v0.2.12 @codex - 2025-10-28
### 变更
- 数据集：`src/inference_recammaster.py` 在 cond 与 tgt 轨迹计算相对位姿前，引入 `_center_trajectory` 将平移起点统一移至原点，确保两条轨迹共享参考帧并降低
       尺度偏移风险。
- 实现：新增 `_center_trajectory` 辅助函数，并在 cond/target 轨迹管线复用，避免重复的平移归零逻辑。

---
## v0.2.11 @lcy - 2025-10-23

### 新增
- 工具：`tools/convert_npz_dir_to_json.py` 支持将目录下的多段 `.npz` 位姿批量转换为旧版 `camera_extrinsics.json` 格式，便于与既有评估/可视化管线互通。

### 变更
- 轨迹可视化：`tools/visualize_compare_trajectories.py` 与 `tools/trajectory_viz_utils.py` 允许直接传入 `.npz` 文件或包含多段 `.npz` 的目录，自动聚合后与传统 JSON 数据同样绘制；保持对 Plotly/PyTorch3D 可视化逻辑的兼容。
- 输出命名：比较脚本针对目录输入使用归一化后的目录名生成结果文件名，避免路径分隔符干扰。

---
## v0.2.10 @yyb - 2025-10-22

### 新增
- 距离感知 RoPE（实验性）：在 3D RoPE 中将 `t_highfreq_ratio` 复用为“w 维低频段比例”，对指定头的 w 维低频复用相位屏蔽，便于与相机平移幅度耦合（distance-aware masking）。
  - `diffsynth/models/wan_video_dit.py`: `rope_apply` 支持在 w 段低频范围内屏蔽选定 head 的复数对；`DiTBlock.forward`/`PRoPE_SelfAttention.forward` 透传 `original_camera_translation`。
  - `src/prope.py`: `_prepare_apply_fns` 对齐 3D RoPE 的 [t, h, w] 复杂频段划分，新增 w 维低频选择逻辑；支持接收 `original_trans`。
- 数据通路：数据集返回原始相机平移轨迹（未归一化）以便 RoPE 使用。
  - `src/dataset.py`: 在计算相对位姿后，保存 `original_camera_translation = concat(tgt_t, cond_t)` 并注入到 `data` 中。
- 实验脚本：新增 `exp_by_day/10.22/exp09b:new_dist_rope.sh` 记录基于距离感知 RoPE 的实验。

### 变更
- 相机平移归一化策略：由“基线归一化（baseline）”调整为“最大范数归一化（max-norm, cond+tgt 联合）”。
  - 在 cond 与 tgt 的相对 c2w 轨迹上，采用两者所有帧的平移 L2 范数最大值进行统一缩放；当最大值 < 1e-2 时不归一化。
  - 移除数据管线内散落的临时缩放，归一化逻辑集中为 `normalize_translation`。
- 位姿工具下沉：将 `invert_SE3_np`、`compute_relative_c2w`、`normalize_translation(_baseline)` 抽出至模块级函数，减少重复计算与提升可读性。
- API 透传：`PRoPE_SelfAttention.forward` 与 `DiTBlock.forward` 新增 `**kwargs`/`original_camera_translation` 透传，避免未来接口膨胀。

### 修复
- 统一 `freqs.to(device=x.device)` 与 dtype 对齐，避免复杂数运算中的 device/dtype 不一致。

### 兼容性
- 归一化策略变化会影响训练/验证中相机尺度，非 API 破坏性但数值分布与旧结果不可完全对齐；建议在同策略下做对比。

---
## v0.2.9 @lcy - 2025-10-22

### 新增
- 训练采样策略：默认按半段均匀抽帧，取消历史的随机抽帧开关，逻辑简化为固定的等间隔采样。

### 变更
- 相机归一化策略：由“最大范数归一化（max-norm）”改为“基线归一化（baseline）”，仍然保留以前的函数。
  - 以 cond 轨迹的最后一帧相对平移范数作为基线；若无效则回退到 cond 平移范数的中位数；再退化到 1.0。
  - 去除历史的固定 `/100` 平移缩放，改由基线统一控制尺度。
- 数值稳定性：相机内参与相机张量类型统一为 float32。
  - `src/dataset.py` 的 `intrinsics` 与 `camera(w2c)` 从 bfloat16 改为 float32，训练/验证一致。
- 路径解析鲁棒性：数据集路径分割从 `re.split(r",", path)` 改为 `re.split(r"/+", path)`，兼容重复分隔符。
- 文案：相关注释与帮助信息同步更新（如“轴向归一化，去固定缩放”等）。
- 精简训练开关：移除 `--enable_cam_layers`、`--enable_test_step` 及计时回调 `ConciseTimingCallback`，脚本与入口保持默认最小化配置。
- CLI：将 `--pipeline_type` 映射为 `v2v/i2v`，移除 `--ckpt_type`，训练/推理均自动根据管线加载 Wan2.1 或 Wan2.2 原始权重。
- 训练/推理 CLI 改用 `v2v/i2v` 管线枚举，并限定 `ckpt_type` 为 `wan21/wan22`；脚本默认加载 Wan 原始模型，不再依赖 ReCamMaster 微调权重。

### 修复
- 在极小平移场景下，归一化基线可能退化为 0 的问题：增加 epsilon 与中位数回退，避免数值发散。

### 影响范围与兼容性
- 训练抽帧策略现固定为均匀抽帧，移除随机抽帧相关参数配置，避免多余分支。
- 相机归一化策略的更改会影响训练与验证中的相机尺度，数值更稳定但与旧版本结果不可完全对齐（非 API 破坏性；建议在同一策略下对比）。

## v0.2.8 @yyb - 2025-10-22

### 新增
- **Δ-RoPE Triton FlashAttention 实现**: 完整的基于 Triton 的 Δ-RoPE (Delta-RoPE) FlashAttention 实现
  - 选择性通道 RoPE: 仅对指定通道对应用基于帧差的 RoPE，其余通道使用标准注意力计算
  - 帧差索引: 相位仅依赖 Δ = t_j - t_i，无需绝对位置，专门为视频帧序列设计
  - 融合计算: Triton 内核中实现两路累加 (co/si) + LUT 融合 + 在线 softmax + 完整 @V
  - 内存高效: LUT 大小仅 [P, 2T-1]，支持在线 softmax 处理长序列
  - 与 PyTorch SDPA 接口一致: 可直接替换 `F.scaled_dot_product_attention`
- **核心实现文件**:
  - `custom_fla.py`: 主要实现文件，包含 LUT 构建、通道重排、Triton 内核、Python 包装接口
  - `test_delta_rope.py`: 基本功能测试，不依赖 Triton 的纯 PyTorch 测试
  - `example_usage.py`: 完整的使用示例和集成指南
  - `README_DELTA_ROPE.md`: 详细的 API 文档和使用指南
- **测试和验证**:
  - 数值精度验证: Triton vs PyTorch 参考实现误差 < 3%
  - 基本功能测试: LUT 构建、通道重排、Δ-RoPE 注意力计算
  - 边界情况处理: P=0、单帧、因果掩码等
  - 性能基准测试: 内存和计算效率验证

### 技术特性
- **LUT 构建**: 构建 [P, 2T-1] 的 cos/sin 查找表，索引映射 Δ_idx = Δ + (T-1)
- **通道重排**: 将 Δ-RoPE 通道对移到 head_dim 前面，优化内存访问模式
- **Triton 内核**: 融合 Δ-RoPE 分数计算、在线 softmax、完整 @V 计算
- **在线 Softmax**: 支持长序列处理，内存高效的流式计算
- **完整 @V 计算**: 全维度支持，分块流式处理，与标准 FA 兼容

### 性能优化
- 内存效率: LUT 大小减少 90%+ (相比完整位置编码)
- 计算效率: 融合计算，避免额外内存访问
- 兼容性: 与现有 PyTorch 代码 100% 兼容
- 灵活性: 可选择性地对部分通道应用 Δ-RoPE

### 使用方式
```python
from custom_fla import sdpa_delta_rope

# 基本使用
out = sdpa_delta_rope(q, k, v, T=T, HW=HW, 
                     rope_pairs_idx=rope_pairs_idx, 
                     rope_omega=rope_omega)

# 与现有模型集成
class DeltaRoPEMultiHeadAttention(nn.Module):
    def forward(self, x):
        return sdpa_delta_rope(q, k, v, ...)
```

### 依赖要求
- Python >= 3.8
- PyTorch >= 2.0  
- Triton >= 2.1
- CUDA 环境

## v0.2.7 @yyb - 2025-01-XX

### 新增
- **物理索引功能**: 添加 `-P/--use-physical-index` 参数，支持前半段时序索引复制到后半段，使 tgt/cond 不共享时间戳
- **模型下载工具**: 完善 `tools/download_model.py` 支持多源模型下载
  - 支持 HuggingFace Hub 和 ModelScope 双源下载
  - 添加自动源选择功能，优先尝试 HuggingFace，失败时回退到 ModelScope
  - 支持文件模式过滤（包含/排除特定文件）
  - 支持断点续传和强制重新下载
  - 添加使用示例：Wan2.1-T2V-1.3B、FLUX.1-Kontext-dev、Wan2.2-TI2V-5B
- **PRoPE 测试工具**: 新增 `tools/prope_case.py` 用于 PRoPE 算法测试和验证
  - 包含旋转矩阵计算、相对位姿编码、多种评分模式
  - 支持 base、rel_scaled、modulated 三种评分策略

### 变更
- `scripts/train.sh` 添加物理索引参数支持和帮助文档
- `src/train_recammaster.py` 优化时序索引处理逻辑，启用初始验证步骤
- `exp_by_day/10.17/exp08a:5frame_physical.sh` 更新 GPU 配置和 batch size
- `tools/download_model.py` 从简单脚本重构为功能完整的 CLI 工具
  - 使用 `click` 库提供丰富的命令行参数支持
  - 支持环境变量 `HF_TOKEN` 自动获取访问令牌
  - 默认使用 HuggingFace 镜像端点提升下载速度

### 修复
- 优化物理索引实现，确保时序索引正确复制和处理
- 改进训练逻辑中的索引设备一致性处理

### 构建与工具链
- 添加 `click` 依赖支持，用于构建命令行界面
- 脚本添加执行权限，支持直接运行

### 其他
- 删除过时的 `tools/download_wan2.1.py` 脚本
- 更新实验配置以适配新的多卡训练环境

## v0.2.6 @lcy - 2025-10-17

### 新增
- 分布式快速失败与超时控制：
  - `scripts/train.sh` 注入 `NCCL_ASYNC_ERROR_HANDLING=1`、`NCCL_BLOCKING_WAIT=1`，并将 `NCCL_DEBUG` 缺省降为 `ERROR`，避免心跳刷屏。
  - 训练入口新增超时参数 `--distributed_timeout_seconds`（默认 1800），用于配置分布式初始化超时。

- 训练/验证新增 `-P/--use_physical_index`：开启时将时序索引前半段复制到后半段，使 tgt 与 cond 不共享时间戳（物理索引）。


### 变更
- `src/train_recammaster.py`：
  - 读取 `--distributed_timeout_seconds` 并设置 `TORCH_DIST_INIT_TIMEOUT`；
  - 将主入口包裹在 try/except 中，出现致命错误时调用 `dist.destroy_process_group()` 并 `sys.exit(1)`，确保各 rank 一致性退出。
- `scripts/train.sh`：
  - 将验证频率 `--val_check_interval_batches` 设为 200；
  - 透传 `--distributed_timeout_seconds 1800` 给训练入口。

- 物理索引适用于“开/关下采样”两种场景：由 `-P/--use_physical_index` 控制，默认关闭不改变现有行为；在 `training_step` 与 `validation_step` 中对 `temporal_indices` 执行“前半段复制到后半段”。
- 更新 `src/train_recammaster.py` 注释与 CLI 帮助文案，明确物理索引适用于是否下采样皆可；本次仅调整训练/验证，推理暂未变更。

### 实验
- `src/inference_recammaster.py`：在目标相机轨迹归一化后、联合归一化前，对目标轨迹平移分量进行 3 倍放大（tgt translation ×3），以增强相机运动幅度（实验性）。

### 修复
- 分布式异常导致的 NCCL 心跳刷屏问题：通过异步错误处理与阻塞等待配置，让错误更快传播并干净退出，避免淹没根因日志。

## v0.2.5 @lcy - 2025-10-15

### 新增
- 训练脚本 `scripts/train.sh` 新增开关 `-T/--use-real-temporal-indices`，可独立于 `-F/--frame-downsample-to` 控制 RoPE 使用真实时间索引或连续索引。

### 变更
- 模型与管线支持显式传入真实时间索引用于 RoPE：
  - `diffsynth/models/wan_video_dit.py` 的 `WanModel.forward` 与 `DiTBlock.forward` 接受 `temporal_indices`，并在构建时间维 RoPE 频率时优先使用真实索引；自动处理 device 一致性。
  - 推理 `diffsynth/pipelines/wan_video_recammaster.py` 计算并传递 `temporal_indices`，与降采样后的帧对齐。
- 训练主循环 `src/train_recammaster.py`：
  - 当启用两半降采样（two-halves）时构造真实索引 `[base, base+per_half]` 并透传；
  - 当未降采样但启用 `--use_real_temporal_indices` 时，使用完整连续区间索引；
  - 验证流程同步透传 `temporal_indices`，确保 RoPE 与时序对齐。
- 验证逻辑对齐训练逻辑：`validation_step` 改为先在整段时序上做两半降采样，再切分 target/condition，简化并与 `training_step` 保持一致；同时将相机嵌入与内参的索引同步为全序列形态（去除按半序列的分支）。
- 训练脚本 `scripts/train.sh` 将 `--val_size` 恢复为 36（由 2 调整回 36），以匹配常规验证规模。

### 实验
- 新增实验脚本 `exp_by_day/10.15/exp07k:5frame_new_t_rope.sh`，演示 `-F 5 -T` 的组合（5 帧两半降采样 + 真实时间索引）。
- 更新 `exp_by_day/10.14/exp07j:full_KS_without_downsample_resume_5_f.sh` 的恢复训练命令与设备分配，指向最新断点与 GPU 配置。

### 修复
- 修复 RoPE 构建中索引 device 不一致导致的运行错误：在模型与管线中将 `temporal_indices` 强制移动至 RoPE 频率张量所在设备。

## v0.2.4 @lcy - 2025-10-14

### 变更
- 推理脚本 `scripts/inference.sh` 更新checkpoint路径至Exp07i实验，并禁用帧降采样（`--frame_downsample_to 0`）以支持全帧推理
- 移除推理脚本中过期的ReCamMaster checkpoint路径配置

### 实验记录
- 新增实验脚本 `exp_by_day/10.14/exp07j:full_KS_without_downsample_resume_5_f.sh`，记录基于Exp07j的断点恢复训练命令

## v0.2.3 @lcy - 2025-10-11

### 新增
- 数据集加载 `src/dataset.py` 按 metadata 中的子目录解析，自动推算并返回相机内参矩阵；训练与验证批次现可直接获取 `intrinsics`。

### 变更
- 推理/训练主干 `diffsynth/models/wan_video_dit.py`、`diffsynth/pipelines/wan_video_recammaster.py` 与 `src/train_recammaster.py` 全链路传播 `cam_intrinsics`，缺省时回退至 Wan2.1 默认内参。
- 训练脚本 `scripts/train.sh` 默认使用全量数据（`metadata_all.csv` + `/train` 根目录）、新增 `--batch-size` 参数，并仅在明确传入 `--wan21-resume-checkpoint` 时加载断点。
- 数据集加载 `src/dataset.py` 目前训练集和验证集没有重合。
- VAE 特征提取 `src/vae_feature.py` 支持 `--metadata_path`，可复用共享 metadata；`TextVideoDataset` 与训练版数据集都会接受绝对或相对路径。

### 构建与工具链
- 新版 `scripts/extract_vae.sh` 支持参数化数据集/metadata 输入，用于批量补齐 `.tensors.pth`。

### 其他
- 清理 README 过期的 todo 片段，保持顶层描述精简。

## v0.2.2 @lcy - 2025-10-09

### 新增
- 推理脚本 `src/inference_recammaster.py` 与 `scripts/inference.sh` 支持 `--frame_downsample_to`，可按需从 21 latent 均匀抽帧降到任意数量，并默认配置为 5。
- `WanVideoReCamMasterPipeline` 增加统一的 latent / camera 下采样工具方法，推理调用可复用。

### 变更
- 推理数据集相机处理与训练对齐：cond/tgt 轨迹分别归一化再联合归一化，并输出 bf16 w2c 视角序列。
- 推理脚本开头注入项目根目录，使直接运行 `src/inference_recammaster.py` 时总能加载当前仓库版本的 `diffsynth` 模块。

### 修复
- 修正 `scripts/inference.sh` 直接运行路径与环境变量，避免旧的 `python inference_recammaster.py` 调用失败。

## v0.2.1 @lcy - 2025-10-08

### 新增
- 新增日常实验脚本 `exp_by_day/10.08/exp07h:5frame_downsample.sh`，记录包含 5 帧降采样、归一化与梯度裁剪的恢复训练命令。

### 变更
- 调整训练脚本默认配置：每轮步数提升至 10k、关闭梯度累积、dataloader worker 提升至 36、batch size提升为10、验证集采样扩充至 36 并缩短验证间隔等，以匹配最新算力配置 `scripts/train.sh`。
- 更新实验脚本 `exp_by_day/10.06/exp07g:resume_debug.sh` 以传递新的运行标志，保持与主训练脚本一致的参数集。

### 修复
- 修正 WandB 项目名称的前缀来源，确保 CLI 指定的 `wandb_name` 能正确映射到项目 slug `src/train_recammaster.py`。

## v0.2.0 @lcy - 2025-10-06

### 新增
- 训练脚本 `scripts/train.sh` 增加 `--t-highfreq-ratio`、`--frame-downsample-to`、`--wan21-resume-checkpoint` 等参数开关，便于动态控制时间频率与断点恢复策略
- 训练主循环 `src/train_recammaster.py` 引入 `ConciseTimingCallback`，自动记录训练/验证时长并在启动阶段执行一次基线验证
- 新增工具脚本 `clean_empty_experiments.py`，用于清理未产出视频的 wandb 子实验

### 变更
- `diffsynth/models/wan_video_dit.py` 与 `DiTBlock` 支持向下游透传 `t_highfreq_ratio` 等关键字参数，适配新的时序高频筛选逻辑
- `scripts/train.sh` 与 `src/train_recammaster.py` 默认超参更新：提升 dataloader worker 数、调整验证频率至每 200 step、缩短测试步数并启用梯度裁剪

### 修复
- `src/dataset.py` 重新整理相机相对位姿：统一参考视角后归一化平移分量并反求 w2c，修复 viewmats 乱飘问题

### 构建与工具链
- `.vscode/settings.json` 默认关闭 ChatGPT 扩展启动弹窗，避免干扰

### 其他
- `exp_by_day/` 目录新增日常实验脚本记录，方便复现训练命令

## Init @yyb 2025-10-02

### 新功能
- **PRoPE 集成**: 实现投影 RoPE 用于增强相机位姿注入
- **数据集工具**: 添加 `tools/download_datasets.py` 和 `tools/list_hf_dataset_files.py`
- **文档套件**: 添加完整文档 (`docs/DEVELOPMENT_QA.md`, `docs/ARCHITECTURE.md` 等)

### 重构
- **项目结构**: 将模块移至 `src/`，脚本移至 `scripts/`，工具移至 `tools/`
- **导入路径**: 更新所有导入语句以适应新结构
- **RoPE 增强**: 修改 `diffsynth/models/wan_video_dit.py` 使用投影项

### 修复
- 修复重构后的导入路径问题
- 解决模块依赖问题

### 维护
- 添加 YouTube cookies 用于认证下载
- 更新 VS Code 启动配置
- 增强训练脚本的参数处理

### 破坏性变更
- 由于重构，导入路径已更改
- 脚本移至不同目录
