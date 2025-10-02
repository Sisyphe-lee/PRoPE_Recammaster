### Q1: SE(3) 变换 H = [R t; 0 1] 的逆平移为何是 −R^T t？

结论：H^{-1} = [R^T, −R^T t; 0, 1]。

两种等价推导：

1) 从运动方程反解
- 正向：x_c = R x_w + t
- 反向：x_w = R^T (x_c − t) = R^T x_c + (−R^T t)
- 因此逆的旋转为 R^T，逆的平移为 −R^T t。

2) 从群律（叠加）推导
- 组合律：[R1, t1] ∘ [R2, t2] = [R1 R2, R1 t2 + t1]
- 设 H^{-1} = [R', t']，要求 H^{-1} ∘ H = I：
  - 旋转：R' R = I ⇒ R' = R^T
  - 平移：R' t + t' = 0 ⇒ t' = −R' t = −R^T t

与代码对应：
- Rinv = R^T
- t' = −Rinv @ t  对应实现：`out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])`

公式形式：

\[ H = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix},\quad H^{-1} = \begin{bmatrix} R^{\top} & -R^{\top} t \\ 0 & 1 \end{bmatrix}. \]

由组合恒等式 \(H^{-1} H = I\) 推得：
\[ \begin{aligned}
H^{-1} H &= \begin{bmatrix} R' & t' \\ 0 & 1 \end{bmatrix} \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix} \\
&= \begin{bmatrix} R'R & R' t + t' \\ 0 & 1 \end{bmatrix} \\
&= I = \begin{bmatrix} I & 0 \\ 0 & 1 \end{bmatrix}.
\end{aligned} \]
因此：
\[ R'R = I \Rightarrow R' = R^{\top}, \qquad R' t + t' = 0 \Rightarrow t' = -R' t = -R^{\top} t. \]
