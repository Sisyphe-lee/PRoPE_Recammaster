# 可视化脚本速览

| 脚本 | 用途 | 常用参数 | 输出 |
| ---- | ---- | -------- | ---- |
| `trajectory_viz_utils.py` | 共享的解析/绘图工具函数 | `load_trajectory_data(path)`、`overlay_p3d_frustums(...)` 等 | 供其他脚本调用 |
| `visualize_single_trajectory.py` | 查看单份轨迹（JSON / NPZ / 目录） | `--input`、`--align-centroid`、`--normalize-01`、`--translation-scale`、`--cam-every` | `trajectory_vis/view_<dataset>.html` |
| `visualize_compare_trajectories.py` | viser 实时对比两份轨迹（同名键一键显示） | `--input`、`--input2`、`--reorder-input1`、`--reorder-input2`、`--align-centers`、`--normalize-01`、`--match`、`--translation-scale`、`--cam-every`、`--port` | 本地 viser 前端（默认 `http://127.0.0.1:8080`） |

通用行为：
- 自动把每条轨迹的首帧平移到原点；
- 支持放大 translation（默认 `--translation-scale 5`）以便观察；
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
    --cam-every 6 \
    --port 8080
```
 
> 输入支持 JSON、单个 NPZ 或包含多个 NPZ 的目录；默认按矩阵原样解析，必要时可以在对比脚本中开启 `--reorder-input1/--reorder-input2` 进行坐标轴重排。对比脚本启动后默认不渲染任何轨迹，左侧面板点击对应名称即可加载同名 A/B 轨迹。

- 对比前端中，输入1 轨迹颜色为蓝→青渐变，输入2 为橙→黄渐变，起点以绿色标记、终点以红色标记；提供两个 log10 滑条（0.01–100）可分别实时缩放当前显示的 A/B 平移量。
