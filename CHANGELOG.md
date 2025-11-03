# Change Log
## v0.2.16 @codex - 2025-10-31

### 修复
- `src/inference_unified.py`: 统一 `compute_relative_c2w` 逻辑，与训练脚本保持一致，对相对 w2c 结果补充 `invert_se3`，避免推理阶段相机平移幅度异常。
- `tools/compare_target_pose.py`: 同步更新相对位姿计算，保证对标工具与推理主流程使用一致的坐标系语义。

## v0.2.15 @codex - 2025-10-31

### 新增
- `evaluation/render_pointodyssey.py`: 支持通过 `--timestamp` 指定已有时间戳目录继续推理，并在循环内跳过已生成的视频，允许中断后续跑。

### 变更
- `evaluation/run_render_pointodyssey.sh`: 精简 GPU 侦测与启动逻辑，自动解析 `CUDA_VISIBLE_DEVICES` 数量并统一调用 `torch.distributed.run`/单卡入口，便于传递新参数（含 `--timestamp`）。

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
- 路径解析鲁棒性：数据集路径分割从 `re.split(r\",\", path)` 改为 `re.split(r\"/+\", path)`，兼容重复分隔符。
- 文案：相关注释与帮助信息同步更新（如“轴向归一化，去固定缩放”等）。
- 精简训练开关：移除 `--enable
