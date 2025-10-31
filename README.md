# ReCamMaster + PRoPE 简介

本仓库在 DiffSynth Wan 系列模型基础上实现了 ReCamMaster 相机控制方案，并结合 PRoPE（Projection RoPE）对时空自注意力进行改造。项目支持：

- **Wan2.1 V2V（视频条件）** 与 **Wan2.2 I2V（图像条件）** 两种训练/推理模式；
- 通过修改三维旋转位置编码，将相机位姿投影项注入自注意力，实现更稳定的镜头运动控制；
- 与官方 DiffSynth 流程保持兼容的 Lightning 训练器、数据集与验证脚本。

## 目录概览

- `src/`：核心训练脚本、Lightning 模块、数据集定义、PRoPE 实现、推理工具；
- `scripts/`：多 GPU 与可复现实验的启动脚本（如 `train.sh`、`exp_by_day/<date>/`）；
- `delta_prope_tests/`：Δ-RoPE Triton 参考实现与回归测试；
- `third_party/DiffSynth-Studio/`：上游 DiffSynth 依赖（保持原样，仅在同步时改动）；
- 其他目录（`docs/`、`metadata/`、`assets/`、`test_output/`）存放实验记录、元数据与可视化资产。

## 主要组件

- **LightningModelForTrain** (`src/lightning_trainer.py`)  
  同时适配 V2V 与 I2V 管线，集成 PRoPE、自定义相机编码与 WandB 视频日志。I2V 模式对首帧潜变量与时间步进行了官方同款处理。
- **Dataset 系列** (`src/dataset.py`)  
  `TensorDataset` 用于 wan2.1 V2V；`ImageConditionTensorDataset` 针对 wan2.2 I2V，将首帧潜变量作为条件，其余帧作为目标噪声预测。
- **PRoPE 实现** (`src/prope.py`)  
  在自注意力的旋转位置编码中注入投影项，替换原 MLP 位姿注入方案。

## 环境与依赖

1. Python ≥ 3.8，CUDA 12.x GPU。
2. 安装依赖：`pip install -r requirements.txt && pip install -e .`
3. 设置 PYTHONPATH：  
   ```bash
   export PYTHONPATH=$PWD/third_party/DiffSynth-Studio:$PYTHONPATH
   ```
4. 大模型权重请提前下载并在 `train.sh` 或 CLI flag 中指定路径。

## 训练与推理

- 训练：  
  ```bash
  python src/train_recammaster.py \
    --dataset_path <数据集根目录> \
    --output_path ./models/train \
    --pipeline_type {v2v|i2v} \
    [更多 PRoPE / WandB / 采样参数]
  ```
  推荐通过 `scripts/train.sh` 或预设脚本启动分布式训练。

- 推理：  
  ```bash
  python src/inference_recammaster.py \
    --dataset_path <数据集根目录> \
    --ckpt_path ./models/ReCamMaster/checkpoints/stepXXXX.ckpt \
    --output_dir ./test_output
  ```
  I2V 模式下输入图像潜变量会自动广播到首帧，并在调度过程中保持不变。

## 验证与测试

- 快速验证 Δ-RoPE：`python delta_prope_tests/test_delta_rope.py`
- I2V 生成质量回归：可对照 DiffSynth 官方脚本 `third_party/DiffSynth-Studio/examples/wanvideo/model_inference/Wan2.2-TI2V-5B.py`
- 建议在提交前运行必要的 GPU 端到端验证，并将关键可视化保存到 `test_output/`。

## 备注

- 请勿直接修改 `third_party/`，本地补丁需在 `docs/` 中登记。
- PR 需附带训练/验证命令、W&B 链接或离线指标，并说明所需数据/权重。
