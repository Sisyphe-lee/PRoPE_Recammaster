import math
import argparse

import numpy as np


def rotation_matrix_from_euler(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
	"""Create rotation matrix R = Rz(yaw) * Ry(pitch) * Rx(roll).

	Angles in degrees. Right-handed, intrinsic Z-Y-X.
	"""
	y = math.radians(yaw_deg)
	p = math.radians(pitch_deg)
	r = math.radians(roll_deg)

	cz, sz = math.cos(y), math.sin(y)
	cy, sy = math.cos(p), math.sin(p)
	cx, sx = math.cos(r), math.sin(r)

	Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
	Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
	Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)

	return (Rz @ Ry) @ Rx


def rotation_angle_cos_from_R(R: np.ndarray) -> float:
	# 不再使用，保留占位以兼容旧引用
	val = (float(np.trace(R)) - 1.0) / 2.0
	return float(max(-1.0, min(1.0, val)))


# def prope_encode(R: np.ndarray, t: np.ndarray) -> np.ndarray:
# 	"""Projection positional encoding (simplified) on SE(3) element (R, t).

# 	- 仅使用 ||t|| 的函数: dist_enc = 1/(1+||t||)，保证近大远小、且对 t 的符号对称。
# 	- 旋转部分使用 cos(theta) = (tr(R) - 1)/2，范围 [-1, 1]。
# 	"""
# 	assert R.shape == (3, 3)
# 	assert t.shape == (3,)

# 	dist = float(np.linalg.norm(t, ord=2))
# 	dist_enc = 1.0 / (1.0 + dist)  # in (0, 1]
# 	rot_cos = rotation_angle_cos_from_R(R)  # in [-1, 1]

# 	# Two-dimensional encoding is sufficient for tests
# 	return np.array([dist_enc, rot_cos], dtype=np.float64)

def prope_encode(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Paper-only encoding: concat(R.flatten(9), t(3)) -> 12-D."""
    assert R.shape == (3, 3)
    assert t.shape == (3,)
    return np.concatenate([R.reshape(-1), t.astype(np.float64)], axis=0)


def sim_dot(q: np.ndarray, k: np.ndarray) -> float:
	return float(np.dot(q, k))


def sim_softmax_like(q: np.ndarray, k: np.ndarray) -> float:
	return math.exp(sim_dot(q, k))


def print_header(title: str) -> None:
	print("\n" + "=" * 8 + f" {title} " + "=" * 8)


# ===== Scoring variants =====
def rel_R_t(R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Compute relative pose in cam1 frame: T_rel = inv(T1) @ T2."""
	R_rel = R1.T @ R2
	t_rel = R1.T @ (t2 - t1)
	return R_rel, t_rel


def score_qP1invP2K_relScaled(q4: np.ndarray, k4: np.ndarray, R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray) -> float:
	"""Chain-preserving variant: replace t_rel by v_rel = s * u, where s=1/(1+||t_rel||)."""
	R_rel, t_rel = rel_R_t(R1, t1, R2, t2)
	d = float(np.linalg.norm(t_rel, ord=2))
	if d < 1e-12:
		v_rel = np.zeros(3, dtype=np.float64)
	else:
		u = (t_rel / d).astype(np.float64)
		s = 1.0 / (1.0 + d)
		v_rel = s * u
	T_rel = np.eye(4, dtype=np.float64)
	T_rel[:3, :3] = R_rel
	T_rel[:3, 3] = v_rel
	return float(q4 @ (T_rel @ k4))


def score_qP1invP2K_modulated(q4: np.ndarray, k4: np.ndarray, R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray) -> float:
	"""Multiplicative modulation: base score * s, where s=1/(1+||t_rel||)."""
	base = score_qP1invP2K(q4, k4, R1, t1, R2, t2)
	_, t_rel = rel_R_t(R1, t1, R2, t2)
	s = 1.0 / (1.0 + float(np.linalg.norm(t_rel, ord=2)))
	return base * s


def se3_to_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
	"""Build 4x4 homogeneous transform from R,t."""
	T = np.eye(4, dtype=np.float64)
	T[:3, :3] = R
	T[:3, 3] = t
	return T


def se3_inv_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
	"""Inverse of SE(3) in homogeneous form."""
	R_inv = R.T
	t_inv = -R_inv @ t
	T = np.eye(4, dtype=np.float64)
	T[:3, :3] = R_inv
	T[:3, 3] = t_inv
	return T


def score_qP1invP2K(q4: np.ndarray, k4: np.ndarray, R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray) -> float:
	"""Compute q[1x4] @ inv(T1)[4x4] @ T2[4x4] @ k[4x1]."""
	P1_inv = se3_inv_homogeneous(R1, t1)
	P2 = se3_to_homogeneous(R2, t2)
	val = q4 @ (P1_inv @ (P2 @ k4))
	return float(val)


# def relative_t(R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray) -> np.ndarray:
# 	"""Compute t_rel from inv(T1) @ T2 acting on [0,0,0,1]."""
# 	R_rel = R1.T @ R2
# 	t_rel = (R1.T @ (t2 - t1)).astype(np.float64)
# 	return t_rel


# def score_inv_t(R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray) -> float:
# 	"""1 / (1 + ||t_rel||) using t_rel between (R1,t1) and (R2,t2)."""
# 	t_rel = relative_t(R1, t1, R2, t2)
# 	d = float(np.linalg.norm(t_rel, ord=2))
# 	return 1.0 / (1.0 + d)


def encode_relative(Ri_cw: np.ndarray, ti_cw: np.ndarray, Rj_cw: np.ndarray, tj_cw: np.ndarray) -> np.ndarray:
	"""按照忽略 K 的公式编码相对位姿：P_ij = T_i^{cw} (T_j^{cw})^{-1}。

	给定相机到世界的位姿 (Ri_cw, ti_cw) 与 (Rj_cw, tj_cw)，
	R_rel = Ri_cw @ Rj_cw^T
	t_rel = ti_cw - Ri_cw @ Rj_cw^T @ tj_cw
	"""
	R_rel = Ri_cw @ Rj_cw.T
	t_rel = ti_cw - R_rel @ tj_cw
	return prope_encode(R_rel, t_rel)



def main() -> None:
	parser = argparse.ArgumentParser(description="ProPE tests")
	parser.add_argument("--score-mode", type=str, default="base", choices=["base", "rel_scaled", "modulated"], help="Scoring mode: base | rel_scaled | modulated")
	args = parser.parse_args()

	# 仅保留链式齐次乘法方案
	# 固定相机1在原点，R1为单位阵
	R1 = np.eye(3, dtype=np.float64)
	t1 = np.zeros(3, dtype=np.float64)
	# q,k 为 4 维齐次点向量（均为 1）
	q4 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
	k4 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

	print_header("计算表达式")
	print("score = q[1x4] @ P_i^{-1}[4x4] @ P_j[4x4] @ k[4x1]; 其中 K1=K2=I, P_i=Ti, P_j=Tj")

	# scorer 分发器
	def score(Rj: np.ndarray, tj: np.ndarray) -> float:
		if args.score_mode == "base":
			return score_qP1invP2K(q4, k4, R1, t1, Rj, tj)
		if args.score_mode == "rel_scaled":
			return score_qP1invP2K_relScaled(q4, k4, R1, t1, Rj, tj)
		if args.score_mode == "modulated":
			return score_qP1invP2K_modulated(q4, k4, R1, t1, Rj, tj)
		raise ValueError(f"Unknown score_mode: {args.score_mode}")

    # T 对称性：tj 在 (1,1,1) 与 (-1,-1,-1)
	print_header("T 对称性 (k 在 (1,1,1) 与 (-1,-1,-1))")
	for vec in [(1.0, 1.0, 1.0), (-1.0, -1.0, -1.0)]:
		Rj = np.eye(3, dtype=np.float64)
		tj = np.array(vec, dtype=np.float64)
		dot = score(Rj, tj)
		print(f"t2={vec}, dot={dot:.6f}, exp(dot)={float(np.exp(dot)):.6f}")

	# T 有界性：k 在 ±(1e7,1e7,1e6)
	print_header("T 有界性 (k 在 ±(1e7,1e7,1e6))")
	for vec in [(1e7, 1e7, 1e6), (-1e7, -1e7, -1e6)]:
		Rj = np.eye(3, dtype=np.float64)
		tj = np.array(vec, dtype=np.float64)
		dot = score(Rj, tj)
		print(f"t2={vec}, dot={dot:.12f}, exp(dot)={float(np.exp(dot)):.12f}")

	# 近大远小：k 在 (0.1,0.1,0.1), (1,1,1), (5,5,5), (10,10,10)
	print_header("近大远小 (按距离增大)")
	near_far_list = [
		(0.1, 0.1, 0.1),
		(1.0, 1.0, 1.0),
		(5.0, 5.0, 5.0),
		(10.0, 10.0, 10.0),
	]
	for vec in near_far_list:
		Rj = np.eye(3, dtype=np.float64)
		tj = np.array(vec, dtype=np.float64)
		d = np.linalg.norm(tj)
		dot = score(Rj, tj)
		print(f"t2={vec}, |t2|={d:.4f}, dot={dot:.6f}, exp(dot)={float(np.exp(dot)):.6f}")

	# 仅沿 x 轴正负扫描：幅值 0.1, 1, 5, 10, 20, 100
	print_header("仅沿 x 轴正负扫描：0.1, 1, 5, 10, 20, 100")
	magnitudes = [0.1, 1.0, 5.0, 10.0, 20.0, 100.0]
	for sgn in [1.0, -1.0]:
		for mag in magnitudes:
			Rj = np.eye(3, dtype=np.float64)
			tj = np.array([sgn * mag, 0.0, 0.0], dtype=np.float64)
			d = np.linalg.norm(tj)
			dot = score(Rj, tj)
			print(f"sign={int(sgn):+d}, x={sgn*mag:.4f}, |t2|={d:.4f}, dot={dot:.6f}, exp(dot)={float(np.exp(dot)):.6f}")

	# xyz 同时正负扫描：幅值 0.1, 1, 5, 10, 20, 100
	print_header("xyz 同时正负扫描：0.1, 1, 5, 10, 20, 100")
	for sgn in [1.0, -1.0]:
		for mag in magnitudes:
			Rj = np.eye(3, dtype=np.float64)
			tj = np.array([sgn * mag, sgn * mag, sgn * mag], dtype=np.float64)
			d = np.linalg.norm(tj)
			dot = score(Rj, tj)
			print(f"sign={int(sgn):+d}, xyz={sgn*mag:.4f}, |t2|={d:.4f}, dot={dot:.6f}, exp(dot)={float(np.exp(dot)):.6f}")

	# R 验证：先左右/上下各旋转 30°（用 yaw=±30, pitch=±30），再沿一个方向 10° -> 180°
	print_header("R 角度验证：t=0，仅旋转")
	base_cases = [
		{"name": "yaw +30", "yaw": 30.0, "pitch": 0.0, "roll": 0.0},
		{"name": "yaw -30", "yaw": -30.0, "pitch": 0.0, "roll": 0.0},
		{"name": "pitch +30", "yaw": 0.0, "pitch": 30.0, "roll": 0.0},
		{"name": "pitch -30", "yaw": 0.0, "pitch": -30.0, "roll": 0.0},
	]
	for case in base_cases:
		Rj = rotation_matrix_from_euler(case["yaw"], case["pitch"], case["roll"])
		tj = np.zeros(3, dtype=np.float64)
		dot = score(Rj, tj)
		print(f"{case['name']}: dot={dot:.6f}, exp(dot)={float(np.exp(dot)):.6f}")

	print("\nYaw 扫描 10° -> 180° (步长 10°)，t=0")
	for yaw in range(10, 181, 10):
		Rj = rotation_matrix_from_euler(float(yaw), 0.0, 0.0)
		tj = np.zeros(3, dtype=np.float64)
		dot = score(Rj, tj)
		print(f"yaw={yaw:3d}°, dot={dot:.6f}, exp(dot)={float(np.exp(dot)):.6f}")



if __name__ == "__main__":
	main()


