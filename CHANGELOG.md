# Change Log

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
