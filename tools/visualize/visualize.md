# 可视化脚本速览

| 脚本 | 用途 | 常用参数 | 输出 |
| ---- | ---- | -------- | ---- |
| `trajectory_viz_utils.py` | 共享的解析/绘图工具函数 | `load_trajectory_data(path)`、`overlay_p3d_frustums(...)` 等 | 供其他脚本调用 |
| `visualize_single_trajectory.py` | 查看单份轨迹（JSON / NPZ / 目录） | `--input`、`--align-centroid`、`--normalize-01`、`--translation-scale`、`--cam-every` | `trajectory_vis/view_<dataset>.html` |
| `visualize_compare_trajectories.py` | 同图对比两份轨迹 | `--input`、`--input2`、`--align-centers`、`--normalize-01`、`--match`、`--translation-scale`、`--cam-every` | `trajectory_vis/compare_<A>_vs_<B>.html` |

所有脚本都会：
- 自动把每条轨迹的首帧平移到原点；
- 支持放大 translation（默认 `--translation-scale 5`）以便观察；
- 可通过 `--save-image`（依赖 `kaleido`）导出静态 PNG。
- `visualize_compare_trajectories.py` 在启用 `--match` 时，会按轨迹名逐一匹配组 B 的尺度与朝向到组 A。

### 单轨迹示例
```bash
python tools/visualize/visualize_single_trajectory.py \
    --input tools/visualize/debug2 \
    --align-centroid \
    --normalize-01 \
    --cam-every 6
```

### 轨迹对比示例
```bash
python tools/visualize/visualize_compare_trajectories.py \
    --input tools/visualize/debug \
    --input2 tools/visualize/debug2 \
    --align-centers \
    --normalize-01 \
    --match \
    --cam-every 6
```
 
> 输入支持 JSON、单个 NPZ 或包含多个 NPZ 的目录；矩阵需为 c2w（单位厘米），加载时会自动转换为 PyTorch3D 约定并换算成米。
