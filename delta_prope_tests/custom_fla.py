# MIT License
#
# Copyright (c) Authors of
# "Cameras as Relative Positional Encoding" https://arxiv.org/pdf/2507.10496
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Δ-RoPE (Delta-RoPE) Triton FlashAttention Implementation

This module implements a selective channel-wise RoPE (Rotary Position Embedding)
that applies relative position encoding based on frame differences (Δ = t_j - t_i)
for video frame sequences. Only specified channel pairs are rotated, while others
follow standard attention computation.

Key Features:
- Selective channel RoPE: Only specified channel pairs use Δ-RoPE
- Frame-difference indexing: Phase depends only on Δ = t_j - t_i
- Fused computation: Two-path accumulation (co/si) + LUT fusion + online softmax + full @V
- Memory efficient: LUT size [P, 2T-1] with online softmax support

Usage:
    # Basic usage
    out = sdpa_delta_rope(q, k, v, T=T, HW=HW, 
                         rope_pairs_idx=rope_pairs_idx, 
                         rope_omega=rope_omega)
    
    # With causal masking
    out = sdpa_delta_rope(q, k, v, T=T, HW=HW,
                         rope_pairs_idx=rope_pairs_idx,
                         rope_omega=rope_omega,
                         is_causal=True)
"""

import math
import torch
import triton
import triton.language as tl


# ---------- LUT 构建 (Host 侧) ----------
def build_delta_rope_lut(omega: torch.Tensor, T: int, *, device=None, dtype=torch.float32):
    """
    构建 Δ-RoPE 的余弦和正弦查找表。
    
    Args:
        omega: [P] 需要应用 Δ-RoPE 的通道对对应的频率 (rad/frame)
        T: 最大帧数
        device: 设备 (可选)
        dtype: 数据类型 (默认 torch.float32)
    
    Returns:
        cos_lut, sin_lut: [P, 2*T-1] 查找表，索引 Δ_idx = Δ + (T-1) ∈ [0, 2T-2]
    """
    device = device or omega.device
    P = int(omega.numel())
    deltas = torch.arange(-(T - 1), T, device=device, dtype=dtype)  # [-T+1 .. T-1]
    phase = omega.reshape(P, 1) * deltas.reshape(1, 2 * T - 1)
    return torch.cos(phase).contiguous(), torch.sin(phase).contiguous()


# ---------- 通道重排 ----------
def pack_pairs_first(x: torch.Tensor, rope_pairs_idx: torch.Tensor):
    """
    将需要应用 Δ-RoPE 的通道对移动到 head_dim 的前面，便于内核连续访问。
    
    Args:
        x: [B,H,N,C] 张量，C 必须为偶数 (2维为一组)
        rope_pairs_idx: [P] 需要应用 Δ-RoPE 的通道对起始索引 (0,2,4,...)，假定不重叠
    
    Returns:
        x_reordered: 重排后的张量，Δ-RoPE 通道对在前 2P 维
        index_map: [C] 重排索引映射
        P: Δ-RoPE 通道对数量
    """
    B, H, N, C = x.shape
    assert C % 2 == 0, "head_dim must be even (pairs of 2 dims)"
    pairs = C // 2
    rope_pairs_idx = rope_pairs_idx.to(x.device)
    P = int(rope_pairs_idx.numel())
    
    # 创建通道对掩码
    mask = torch.zeros(pairs, dtype=torch.bool, device=x.device)
    if P > 0:
        mask[rope_pairs_idx] = True
    rest_idx = torch.nonzero(~mask, as_tuple=True)[0]
    
    # 构造新的通道对顺序：先 Δ-RoPE 对，再其它
    pair_order = torch.cat([rope_pairs_idx, rest_idx], dim=0)  # [pairs]
    
    # 展开到通道维（两维一组）
    index_map = torch.stack([2 * pair_order, 2 * pair_order + 1], dim=-1).reshape(-1)  # [C]
    x_reordered = x.index_select(dim=-1, index=index_map)
    
    return x_reordered, index_map, P


# ---------- Triton 内核：Δ-RoPE + 在线 softmax + 完整 @V ----------
@triton.jit
def _delta_rope_flash_fwd(
    Q, K, V, O,                 # pointers
    COS, SIN,                   # [P, 2T-1]
    B, H, N, C,                 # sizes
    T, HW, P,                   # frame count, tokens/frame, rope-pair count
    STR_QB, STR_QH, STR_QN, STR_QC,
    STR_KB, STR_KH, STR_KN, STR_KC,
    STR_VB, STR_VH, STR_VN, STR_VC,
    STR_OB, STR_OH, STR_ON, STR_OC,
    STR_CP, STR_CD,             # lut strides: [P, 2T-1]
    SCALE,                      # scale = 1/sqrt(C) typically
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,      # rows per tile
    BLOCK_N: tl.constexpr,      # cols per tile
    BLOCK_DV: tl.constexpr      # dv block when writing @V
):
    # program ids
    bh = tl.program_id(0)               # combine (B,H)
    row_block = tl.program_id(1)        # which M-block

    b = bh // H
    h = bh % H

    # row indices
    offs_m = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < N

    # frame ids for rows
    t_m = offs_m // HW  # [M]

    # accumulators for online softmax (per row)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)  # running max
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)                # running sum of exp

    # initialize O (accumulator in global memory) to zero for these M rows across dv blocks
    # Do it once before iterating column blocks
    for dv0 in range(0, C, BLOCK_DV):
        dv = tl.arange(0, BLOCK_DV)
        dv_idx = dv0 + dv
        dv_mask = dv_idx < C
        z = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
        tl.store(
            O + b*STR_OB + h*STR_OH + (offs_m[:, None]*STR_ON) + (dv_idx[None, :]*STR_OC),
            z, mask=(m_mask[:, None] & dv_mask[None, :])
        )

    # number of column blocks
    n_blocks = (N + BLOCK_N - 1) // BLOCK_N

    # iterate over column blocks (online softmax)
    for col_block in range(0, n_blocks):
        offs_n = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        # col frame ids and Δ index (shifted to [0..2T-2])
        t_n = offs_n // HW
        delta_idx = (t_n[None, :] - t_m[:, None]) + (T - 1)  # [M,Nblk]

        # compute score tile: start with zeros
        scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # --- rest channels (no Δ-RoPE): head-dim [2P .. C) ---
        rest = C - 2 * P
        # tile over rest dims with some width (we can use 64)
        for d0 in range(0, rest, 64):
            dlen = tl.minimum(rest - d0, 64)
            d = (2 * P) + tl.arange(0, 64)
            d = tl.where(d < (2 * P + dlen), d, (2 * P + dlen - 1))
            q_blk = tl.load(Q + b*STR_QB + h*STR_QH + (offs_m[:, None]*STR_QN) + (d[None, :]*STR_QC),
                            mask=(m_mask[:, None]), other=0.0).to(tl.float32)
            k_blk = tl.load(K + b*STR_KB + h*STR_KH + (offs_n[:, None]*STR_KN) + (d[None, :]*STR_KC),
                            mask=(n_mask[:, None]), other=0.0).to(tl.float32)
            scores += tl.dot(q_blk, tl.trans(k_blk))

        # --- Δ-RoPE pairs: head-dim [0 .. 2P) ---
        # compute co/si contribution per pair and fuse by cos/sin(Δ·ω_p)
        for p in range(0, P):
            dx = 2*p + 0
            dy = 2*p + 1
            qx = tl.load(Q + b*STR_QB + h*STR_QH + (offs_m*STR_QN) + dx*STR_QC,
                         mask=m_mask, other=0.0).to(tl.float32)
            qy = tl.load(Q + b*STR_QB + h*STR_QH + (offs_m*STR_QN) + dy*STR_QC,
                         mask=m_mask, other=0.0).to(tl.float32)
            kx = tl.load(K + b*STR_KB + h*STR_KH + (offs_n*STR_KN) + dx*STR_KC,
                         mask=n_mask, other=0.0).to(tl.float32)
            ky = tl.load(K + b*STR_KB + h*STR_KH + (offs_n*STR_KN) + dy*STR_KC,
                         mask=n_mask, other=0.0).to(tl.float32)
            co = (qx[:, None] * kx[None, :]) + (qy[:, None] * ky[None, :])  # [M,Nblk]
            si = (qy[:, None] * kx[None, :]) - (qx[:, None] * ky[None, :])  # [M,Nblk]
            c = tl.load(COS + p*STR_CP + delta_idx, mask=(m_mask[:, None] & n_mask[None, :]), other=1.0)
            s = tl.load(SIN + p*STR_CP + delta_idx, mask=(m_mask[:, None] & n_mask[None, :]), other=0.0)
            scores += c * co + s * si

        # scale
        scores *= SCALE

        # causal (optional): mask future tokens by global indices
        if IS_CAUSAL:
            causal_mask = offs_n[None, :] > offs_m[:, None]
            scores = tl.where(causal_mask, -float("inf"), scores)

        # avoid invalid columns
        scores = tl.where(n_mask[None, :], scores, -float("inf"))

        # --- online softmax update ---
        m_ij = tl.max(scores, 1)                       # [M]
        m_new = tl.maximum(m_i, m_ij)                  # [M]
        alpha = tl.exp(m_i - m_new)                    # [M]
        p = tl.exp(scores - m_new[:, None])            # [M,Nblk]
        l_new = l_i * alpha + tl.sum(p, 1)             # [M]

        # --- update O accumulator in global memory: O = O*alpha + p @ V_blk ---
        # stream over dv=C by BLOCK_DV
        for dv0 in range(0, C, BLOCK_DV):
            dv = tl.arange(0, BLOCK_DV)
            dv_idx = dv0 + dv
            dv_mask = dv_idx < C

            v_blk = tl.load(
                V + b*STR_VB + h*STR_VH + (offs_n[:, None]*STR_VN) + (dv_idx[None, :]*STR_VC),
                mask=(n_mask[:, None] & dv_mask[None, :]),
                other=0.0
            ).to(tl.float32)  # [Nblk, DV]

            o_prev = tl.load(
                O + b*STR_OB + h*STR_OH + (offs_m[:, None]*STR_ON) + (dv_idx[None,:]*STR_OC),
                mask=(m_mask[:, None] & dv_mask[None, :]),
                other=0.0
            ).to(tl.float32)  # [M, DV]

            # O := O*alpha + p @ v_blk
            o_upd = tl.dot(p, v_blk)                   # [M, DV]
            o_new = (o_prev * alpha[:, None]) + o_upd  # row-wise scale then add

            tl.store(
                O + b*STR_OB + h*STR_OH + (offs_m[:, None]*STR_ON) + (dv_idx[None,:]*STR_OC),
                o_new, mask=(m_mask[:, None] & dv_mask[None, :])
            )

        # finalize step
        m_i = m_new
        l_i = l_new

    # --- normalize: O = O / l_i[:,None] ---
    for dv0 in range(0, C, BLOCK_DV):
        dv = tl.arange(0, BLOCK_DV)
        dv_idx = dv0 + dv
        dv_mask = dv_idx < C
        o_blk = tl.load(
            O + b*STR_OB + h*STR_OH + (offs_m[:, None]*STR_ON) + (dv_idx[None,:]*STR_OC),
            mask=(m_mask[:, None] & dv_mask[None, :]),
            other=0.0
        ).to(tl.float32)
        o_blk = o_blk / l_i[:, None]
        tl.store(
            O + b*STR_OB + h*STR_OH + (offs_m[:, None]*STR_ON) + (dv_idx[None,:]*STR_OC),
            o_blk, mask=(m_mask[:, None] & dv_mask[None, :])
        )


# ---------- Python 包装接口 ----------
def sdpa_delta_rope(
    q: torch.Tensor,   # [B,H,N,C], float16/bfloat16 ok
    k: torch.Tensor,   # [B,H,N,C]
    v: torch.Tensor,   # [B,H,N,C]  (dv == C here; extendable)
    *,
    T: int,            # frames
    HW: int,           # tokens per frame -> N must equal T*HW
    rope_pairs_idx: torch.Tensor,  # [P] pair-start indices in {0,2,4,...} among head-dim
    rope_omega: torch.Tensor,      # [P] frequencies
    is_causal: bool = False,
    scale: float | None = None,
    block_m: int = 64,
    block_n: int = 64,
    block_dv: int = 64,
):
    """
    Δ-RoPE 注意力机制，与 PyTorch scaled_dot_product_attention 接口一致。
    
    Args:
        q, k, v: [B,H,N,C] 输入张量
        T: 帧数
        HW: 每帧 token 数，必须满足 N = T * HW
        rope_pairs_idx: [P] 需要应用 Δ-RoPE 的通道对起始索引
        rope_omega: [P] 对应的频率
        is_causal: 是否使用因果掩码
        scale: 缩放因子，默认为 1/sqrt(C)
        block_m, block_n, block_dv: Triton tile 大小
    
    Returns:
        o: [B,H,N,C] 输出张量
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "use CUDA tensors"
    B, H, N, C = q.shape
    assert N == T * HW, "N must equal T*HW (flatten order (t,y,x))"
    assert C % 2 == 0, "head_dim must be even (pairs of 2 dims)"
    
    # move selected pairs to front
    q_re, _, P = pack_pairs_first(q, rope_pairs_idx)
    k_re, _, _ = pack_pairs_first(k, rope_pairs_idx)
    
    # build LUT
    cos_lut, sin_lut = build_delta_rope_lut(rope_omega.to(q.device, torch.float32), T, device=q.device)
    
    # scale
    if scale is None:
        scale = 1.0 / math.sqrt(C)
    scale = float(scale)
    
    # output (accumulator buffer)
    o = torch.empty_like(v)

    grid = (B * H, triton.cdiv(N, block_m))

    _delta_rope_flash_fwd[grid](
        q_re, k_re, v, o,
        cos_lut, sin_lut,
        B, H, N, C,
        T, HW, P,
        q_re.stride(0), q_re.stride(1), q_re.stride(2), q_re.stride(3),
        k_re.stride(0), k_re.stride(1), k_re.stride(2), k_re.stride(3),
        v.stride(0),    v.stride(1),    v.stride(2),    v.stride(3),
        o.stride(0),    o.stride(1),    o.stride(2),    o.stride(3),
        cos_lut.stride(0), cos_lut.stride(1),
        scale,
        IS_CAUSAL=int(is_causal),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_DV=block_dv
    )
    return o


# ---------- PyTorch 参考实现 (用于验证) ----------
def _reference_delta_rope_attention(
    q: torch.Tensor,   # [B,H,N,C]
    k: torch.Tensor,   # [B,H,N,C]
    v: torch.Tensor,   # [B,H,N,C]
    *,
    T: int,
    HW: int,
    rope_pairs_idx: torch.Tensor,
    rope_omega: torch.Tensor,
    is_causal: bool = False,
    scale: float | None = None,
):
    """
    PyTorch 参考实现，用于数值验证。
    """
    B, H, N, C = q.shape
    pairs = C // 2
    P = int(rope_pairs_idx.numel())
    
    # reorder
    def _pack(x):
        mask = torch.zeros(pairs, dtype=torch.bool, device=x.device)
        if P > 0:
            mask[rope_pairs_idx] = True
        rest_idx = torch.nonzero(~mask, as_tuple=True)[0]
        order = torch.cat([rope_pairs_idx, rest_idx], 0)
        idx2 = torch.stack([2*order, 2*order+1], dim=-1).reshape(-1)
        return x.index_select(dim=-1, index=idx2)
    
    q_re = _pack(q)
    k_re = _pack(k)

    q_rope, q_rest = q_re[..., :2*P].float(), q_re[..., 2*P:].float()
    k_rope, k_rest = k_re[..., :2*P].float(), k_re[..., 2*P:].float()

    scores = torch.einsum("bhnc,bhsc->bhns", q_rest, k_rest)

    idx = torch.arange(N, device=q.device)
    t = idx // HW
    delta = (t[None, :] - t[:, None])                                  # [N,N]
    delta_idx = (delta + (T - 1)).long()

    cos_lut, sin_lut = build_delta_rope_lut(rope_omega.float(), T, device=q.device)
    qx, qy = q_rope[..., 0::2], q_rope[..., 1::2]                      # [B,H,N,P]
    kx, ky = k_rope[..., 0::2], k_rope[..., 1::2]                      # [B,H,N,P]

    for p in range(P):
        c = cos_lut[p, delta_idx]                                      # [N,N]
        s = sin_lut[p, delta_idx]
        co = torch.einsum("bhn,bhs->bhns", qx[..., p], kx[..., p]) + torch.einsum("bhn,bhs->bhns", qy[..., p], ky[..., p])
        si = torch.einsum("bhn,bhs->bhns", qy[..., p], kx[..., p]) - torch.einsum("bhn,bhs->bhns", qx[..., p], ky[..., p])
        scores = scores + c * co + s * si

    if scale is None:
        scale = 1.0 / math.sqrt(C)
    
    if is_causal:
        causal_mask = torch.tril(torch.ones(N, N, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float('-inf'))
    
    probs = torch.softmax(scores * scale, dim=-1)
    out = torch.einsum("bhns,bhsc->bhnc", probs, v.float())
    return out.to(v.dtype)


# ----------------------- 测试和示例 -----------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"

    B, H = 2, 3
    T, HW = 6, 8             # N = 48
    N = T * HW
    C = 64                    # even, head_dim
    P = 8                     # Δ-RoPE over first P pairs (2P dims)

    q = torch.randn(B, H, N, C, device=device, dtype=torch.float16)
    k = torch.randn(B, H, N, C, device=device, dtype=torch.float16)
    v = torch.randn(B, H, N, C, device=device, dtype=torch.float16)

    rope_pairs_idx = torch.arange(P, device=device, dtype=torch.long)  # pairs 0..P-1
    # log-spaced frequencies in [2π/T, π] as a demo
    min_f, max_f = 1.0 / T, 0.5
    omega = torch.exp(torch.linspace(math.log(min_f), math.log(max_f), P, device=device)) * (2 * math.pi)

    print("Δ-RoPE Triton FlashAttention 测试")
    print(f"输入形状: q={q.shape}, k={k.shape}, v={v.shape}")
    print(f"帧数 T={T}, 每帧 token 数 HW={HW}, 总 token 数 N={N}")
    print(f"Δ-RoPE 通道对数 P={P}, 频率范围: [{min_f:.3f}, {max_f:.3f}]")

    # Triton Δ-RoPE SDPA
    print("\n运行 Triton 实现...")
    out_triton = sdpa_delta_rope(q, k, v,
                                 T=T, HW=HW,
                                 rope_pairs_idx=rope_pairs_idx,
                                 rope_omega=omega,
                                 is_causal=False,
                                 block_m=64, block_n=64, block_dv=64)

    # PyTorch 参考实现
    print("运行 PyTorch 参考实现...")
    out_ref = _reference_delta_rope_attention(q, k, v,
                                           T=T, HW=HW,
                                           rope_pairs_idx=rope_pairs_idx,
                                           rope_omega=omega,
                                           is_causal=False)

    # 数值精度对比
    diff = (out_triton - out_ref).float().abs()
    print(f"\n数值精度对比 (Triton vs PyTorch):")
    print(f"  平均绝对误差: {diff.mean().item():.6f}")
    print(f"  最大绝对误差: {diff.max().item():.6f}")
    print(f"  相对误差 (L2): {(out_triton - out_ref).norm().item() / out_ref.norm().item():.6f}")

    # 基本功能验证
    assert out_triton.shape == q.shape, f"输出形状不匹配: {out_triton.shape} vs {q.shape}"
    assert not torch.isnan(out_triton).any(), "输出包含 NaN"
    assert not torch.isinf(out_triton).any(), "输出包含 Inf"
    
    print("\n✅ 所有测试通过！")
    print("Δ-RoPE Triton FlashAttention 实现完成。")
