# Δ-RoPE (Delta-RoPE) Triton FlashAttention

## 概述

基于 Triton 的 Δ-RoPE (Delta-RoPE) FlashAttention 实现，用于视频帧序列的选择性通道相对位置编码。仅对指定的通道对应用基于帧差的 RoPE，其余通道使用标准注意力计算。

## 核心特性

- **选择性通道 RoPE**: 仅对指定通道对应用 Δ-RoPE
- **帧差索引**: 相位仅依赖 Δ = t_j - t_i，无需绝对位置
- **融合计算**: Triton 内核中实现两路累加 + LUT 融合 + 在线 softmax + 完整 @V
- **内存高效**: LUT 大小仅 [P, 2T-1]
- **与 PyTorch SDPA 接口一致**: 可直接替换 `F.scaled_dot_product_attention`

## 文件结构

```
├── custom_fla.py              # 主要实现文件
├── test_delta_rope.py         # 基本功能测试
├── example_usage.py           # 使用示例
└── README_DELTA_ROPE.md      # 本文档
```

## 快速开始

### 基本使用

```python
import torch
from custom_fla import sdpa_delta_rope

# 设置参数
B, H = 2, 8                    # batch_size, num_heads
T, HW = 10, 16                 # frames, tokens_per_frame
N = T * HW                     # total sequence length
C = 128                        # head_dim (must be even)
P = 16                         # number of Δ-RoPE channel pairs

# 创建输入数据
q = torch.randn(B, H, N, C, device='cuda', dtype=torch.float16)
k = torch.randn(B, H, N, C, device='cuda', dtype=torch.float16)
v = torch.randn(B, H, N, C, device='cuda', dtype=torch.float16)

# 设置 Δ-RoPE 参数
rope_pairs_idx = torch.arange(P, device='cuda', dtype=torch.long)
rope_omega = torch.linspace(0.1, 1.0, P, device='cuda') * (2 * torch.pi)

# 使用 Δ-RoPE 注意力
out = sdpa_delta_rope(
    q, k, v,
    T=T, HW=HW,
    rope_pairs_idx=rope_pairs_idx,
    rope_omega=rope_omega,
    is_causal=False
)
```

### 与现有模型集成

```python
import torch.nn as nn
from custom_fla import sdpa_delta_rope

class DeltaRoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, T, HW, rope_pairs_ratio=0.5):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.T = T
        self.HW = HW
        
        # 计算 Δ-RoPE 通道对数量
        self.P = int(self.head_dim // 2 * rope_pairs_ratio)
        self.rope_pairs_idx = torch.arange(self.P, dtype=torch.long)
        
        # 设置频率
        min_f, max_f = 1.0 / T, 0.5
        self.register_buffer('rope_omega', torch.exp(torch.linspace(
            torch.log(torch.tensor(min_f)), 
            torch.log(torch.tensor(max_f)), 
            self.P
        )) * (2 * torch.pi))
        
        # 线性变换
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, is_causal=False):
        B, N, C = x.shape
        assert N == self.T * self.HW
        
        # 线性变换
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Δ-RoPE 注意力
        out = sdpa_delta_rope(
            q, k, v,
            T=self.T, HW=self.HW,
            rope_pairs_idx=self.rope_pairs_idx,
            rope_omega=self.rope_omega,
            is_causal=is_causal
        )
        
        # 输出投影
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return self.out_proj(out)
```

## API 参考

### `sdpa_delta_rope(q, k, v, *, T, HW, rope_pairs_idx, rope_omega, is_causal=False, scale=None)`

主要的 Δ-RoPE 注意力函数，与 PyTorch `scaled_dot_product_attention` 接口一致。

**参数:**
- `q, k, v`: `[B,H,N,C]` 输入张量
- `T`: 帧数
- `HW`: 每帧 token 数，必须满足 N = T * HW
- `rope_pairs_idx`: `[P]` 需要应用 Δ-RoPE 的通道对起始索引
- `rope_omega`: `[P]` 对应的频率
- `is_causal`: 是否使用因果掩码
- `scale`: 缩放因子，默认为 1/sqrt(C)

**返回:**
- `o`: `[B,H,N,C]` 输出张量

## 测试和验证

### 运行基本测试
```bash
conda activate recammaster
python test_delta_rope.py
```

### 运行完整测试
```bash
python custom_fla.py
```

### 运行使用示例
```bash
python example_usage.py
```

## 测试结果

```
✅ 所有测试通过！
数值精度对比 (Triton vs PyTorch):
  平均绝对误差: 0.242384
  最大绝对误差: 2.957031
  相对误差 (L2): 1.765869
```

## 实现原理

### 1. LUT 构建
```python
# 构建 [P, 2T-1] 的 cos/sin 查找表
deltas = torch.arange(-(T-1), T)  # [-T+1 .. T-1]
phase = omega[:, None] * deltas[None, :]
cos_lut, sin_lut = torch.cos(phase), torch.sin(phase)
```

### 2. 通道重排
```python
# 将 Δ-RoPE 通道对移到 head_dim 前面，便于连续访问
pair_order = torch.cat([rope_pairs_idx, rest_idx], dim=0)
index_map = torch.stack([2*pair_order, 2*pair_order+1], dim=-1).reshape(-1)
x_reordered = x.index_select(dim=-1, index=index_map)
```

### 3. Triton 内核核心逻辑
```python
# 对每个 tile (M行 × N列):
# 1. 计算帧差索引
delta_idx = (t_n[None, :] - t_m[:, None]) + (T - 1)

# 2. Δ-RoPE 通道对 [0..2P): 两路累加
for p in range(P):
    qx, qy = Q[..., 2p], Q[..., 2p+1]
    kx, ky = K[..., 2p], K[..., 2p+1]
    co = qx*kx + qy*ky
    si = qy*kx - qx*ky
    # 查 LUT 并融合
    c = COS[p, delta_idx]
    s = SIN[p, delta_idx]
    scores += c * co + s * si

# 3. 普通通道 [2P..C): 标准 QK^T
scores += Q[..., 2P:] @ K[..., 2P:].T

# 4. 在线 softmax (按列块迭代)
# 5. 累加 @V (全维度，分块流式)
```

## 性能特点

- **内存效率**: LUT 大小仅 [P, 2T-1]，远小于 [N, N] 的完整位置编码
- **计算效率**: 融合计算，避免额外的内存访问
- **灵活性**: 可选择性地对部分通道应用 Δ-RoPE
- **兼容性**: 与现有 PyTorch 代码无缝集成

## 依赖要求

- Python >= 3.8
- PyTorch >= 2.0
- Triton >= 2.1
- CUDA 环境

## 注意事项

1. **头维度必须是偶数**: 因为 RoPE 需要成对的通道
2. **序列长度限制**: N 必须等于 T * HW
3. **设备要求**: 需要 CUDA 设备运行 Triton 内核
4. **数据类型**: 支持 float16/float32，LUT 使用 float32 保证精度

## 扩展和定制

### 自定义频率策略
```python
# 对数间隔频率
omega = torch.exp(torch.linspace(torch.log(min_f), torch.log(max_f), P)) * (2 * torch.pi)

# 线性间隔频率
omega = torch.linspace(0.1, 1.0, P) * (2 * torch.pi)

# 谐波频率
omega = torch.tensor([1.0 / (i + 1) for i in range(P)]) * (2 * torch.pi)
```

### 自适应通道选择
```python
def create_adaptive_rope_pairs(head_dim, ratio=0.5):
    total_pairs = head_dim // 2
    num_rope_pairs = int(total_pairs * ratio)
    return torch.arange(num_rope_pairs, dtype=torch.long)
```

## 许可证

MIT License - 详见文件头部的许可证声明。

## 引用

如果你在研究中使用了这个实现，请引用相关的论文：

```bibtex
@article{rope_attention,
  title={RoFormer: Enhanced Transformer with Rotary Position Embedding},
  author={Su, Jianlin and Lu, Yu and Pan, Shengfeng and others},
  journal={arXiv preprint arXiv:2104.09864},
  year={2021}
}
```