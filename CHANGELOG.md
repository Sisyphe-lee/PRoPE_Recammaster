# Change Log

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