import numpy as np
from pathlib import Path

npz_path = Path("evaluation/example_eval/20251103_165044/pose/1_cam05.npz")  # 修改为你的文件

data = np.load(npz_path, allow_pickle=False)
pose_key = "data" if "data" in data else list(data.keys())[0]
c2w = np.asarray(data[pose_key])
# print(c2w.shape)
# exit(0)
# c2w = c2w.transpose(0,2,1)
c2w = c2w[:, :, [1, 2, 0, 3]]
# 只读第4列平移
t = c2w[:, :3, 3].astype(np.float64)
# 重心化到原点
t -= t[0]
# 归一化到最大范数为1
norms = np.linalg.norm(t, axis=1)
scale = norms.max() if norms.size > 0 else 1.0
scale = scale if scale > 0 else 1.0
t /= scale

indices = list(range(0, len(t), 10))
if indices[-1] != len(t) - 1:
    indices.append(len(t) - 1)

print(f"max_norm_before_norm: {norms.max() if norms.size>0 else 0}")
for i in indices:
    print(f"frame {i:02d}: {t[i]}")