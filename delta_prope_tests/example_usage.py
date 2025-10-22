#!/usr/bin/env python3
"""
Δ-RoPE 使用示例和集成指南

本文件展示如何在你的项目中使用 Δ-RoPE Triton FlashAttention。
"""

import torch
import torch.nn.functional as F
from custom_fla import sdpa_delta_rope, build_delta_rope_lut, pack_pairs_first

def example_basic_usage():
    """基本使用示例"""
    print("🚀 Δ-RoPE 基本使用示例")
    print("=" * 50)
    
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
    rope_pairs_idx = torch.arange(P, device='cuda', dtype=torch.long)  # 前 P 个通道对
    # 对数间隔的频率，范围 [2π/T, π]
    min_f, max_f = 1.0 / T, 0.5
    rope_omega = torch.exp(torch.linspace(
        torch.log(torch.tensor(min_f)), 
        torch.log(torch.tensor(max_f)), 
        P, device='cuda'
    )) * (2 * torch.pi)
    
    print(f"📊 输入参数:")
    print(f"  批次大小: {B}, 注意力头数: {H}")
    print(f"  帧数: {T}, 每帧 token 数: {HW}, 总序列长度: {N}")
    print(f"  头维度: {C}, Δ-RoPE 通道对数: {P}")
    print(f"  频率范围: [{min_f:.3f}, {max_f:.3f}]")
    
    # 使用 Δ-RoPE 注意力
    print(f"\n⚡ 运行 Δ-RoPE 注意力...")
    out = sdpa_delta_rope(
        q, k, v,
        T=T, HW=HW,
        rope_pairs_idx=rope_pairs_idx,
        rope_omega=rope_omega,
        is_causal=False
    )
    
    print(f"✅ 输出形状: {out.shape}")
    print(f"  数据类型: {out.dtype}")
    print(f"  设备: {out.device}")
    
    return out

def example_causal_attention():
    """因果注意力示例"""
    print("\n🎭 因果注意力示例")
    print("=" * 30)
    
    B, H, T, HW, C = 1, 4, 5, 8, 64
    N = T * HW
    P = 8
    
    q = torch.randn(B, H, N, C, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, C, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, C, device='cuda', dtype=torch.float16)
    
    rope_pairs_idx = torch.arange(P, device='cuda', dtype=torch.long)
    rope_omega = torch.linspace(0.1, 1.0, P, device='cuda') * (2 * torch.pi)
    
    # 因果注意力
    out_causal = sdpa_delta_rope(
        q, k, v,
        T=T, HW=HW,
        rope_pairs_idx=rope_pairs_idx,
        rope_omega=rope_omega,
        is_causal=True
    )
    
    print(f"✅ 因果注意力输出形状: {out_causal.shape}")
    return out_causal

def example_integration_with_existing_attention():
    """与现有注意力机制集成示例"""
    print("\n🔗 与现有注意力机制集成")
    print("=" * 40)
    
    class DeltaRoPEMultiHeadAttention(torch.nn.Module):
        """集成 Δ-RoPE 的多头注意力模块"""
        
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
            self.q_proj = torch.nn.Linear(d_model, d_model, bias=False)
            self.k_proj = torch.nn.Linear(d_model, d_model, bias=False)
            self.v_proj = torch.nn.Linear(d_model, d_model, bias=False)
            self.out_proj = torch.nn.Linear(d_model, d_model, bias=False)
        
        def forward(self, x, is_causal=False):
            B, N, C = x.shape
            assert N == self.T * self.HW, f"序列长度 {N} 必须等于 T*HW = {self.T}*{self.HW}"
            
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
    
    # 测试集成模块
    d_model, num_heads = 256, 8
    T, HW = 6, 12
    N = T * HW
    
    model = DeltaRoPEMultiHeadAttention(d_model, num_heads, T, HW).cuda().half()
    x = torch.randn(2, N, d_model, device='cuda', dtype=torch.float16)
    
    # 前向传播
    out = model(x, is_causal=False)
    print(f"✅ 集成模块输出形状: {out.shape}")
    
    # 因果注意力
    out_causal = model(x, is_causal=True)
    print(f"✅ 因果集成模块输出形状: {out_causal.shape}")
    
    return model, out

def example_performance_comparison():
    """性能对比示例"""
    print("\n⚡ 性能对比示例")
    print("=" * 30)
    
    import time
    
    B, H, T, HW, C = 4, 8, 8, 16, 128
    N = T * HW
    P = 16
    
    q = torch.randn(B, H, N, C, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, C, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, C, device='cuda', dtype=torch.float16)
    
    rope_pairs_idx = torch.arange(P, device='cuda', dtype=torch.long)
    rope_omega = torch.linspace(0.1, 1.0, P, device='cuda') * (2 * torch.pi)
    
    # 预热
    for _ in range(3):
        _ = sdpa_delta_rope(q, k, v, T=T, HW=HW, rope_pairs_idx=rope_pairs_idx, rope_omega=rope_omega)
        _ = F.scaled_dot_product_attention(q, k, v)
    
    torch.cuda.synchronize()
    
    # 测试 Δ-RoPE
    start_time = time.time()
    for _ in range(10):
        out_delta_rope = sdpa_delta_rope(q, k, v, T=T, HW=HW, rope_pairs_idx=rope_pairs_idx, rope_omega=rope_omega)
    torch.cuda.synchronize()
    delta_rope_time = time.time() - start_time
    
    # 测试标准 SDPA
    start_time = time.time()
    for _ in range(10):
        out_sdpa = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    sdpa_time = time.time() - start_time
    
    print(f"📊 性能对比 (10次运行平均):")
    print(f"  Δ-RoPE 时间: {delta_rope_time/10*1000:.2f} ms")
    print(f"  标准 SDPA 时间: {sdpa_time/10*1000:.2f} ms")
    print(f"  性能开销: {delta_rope_time/sdpa_time:.2f}x")
    
    return delta_rope_time, sdpa_time

def example_advanced_usage():
    """高级使用示例"""
    print("\n🔬 高级使用示例")
    print("=" * 30)
    
    # 自定义频率设置
    def create_custom_frequencies(P, T, freq_type='log'):
        """创建自定义频率"""
        if freq_type == 'log':
            # 对数间隔
            min_f, max_f = 1.0 / T, 0.5
            return torch.exp(torch.linspace(torch.log(torch.tensor(min_f)), torch.log(torch.tensor(max_f)), P)) * (2 * torch.pi)
        elif freq_type == 'linear':
            # 线性间隔
            return torch.linspace(0.1, 1.0, P) * (2 * torch.pi)
        elif freq_type == 'harmonic':
            # 谐波频率
            return torch.tensor([1.0 / (i + 1) for i in range(P)]) * (2 * torch.pi)
        else:
            raise ValueError(f"未知频率类型: {freq_type}")
    
    # 测试不同频率类型
    P, T = 8, 6
    freq_types = ['log', 'linear', 'harmonic']
    
    for freq_type in freq_types:
        omega = create_custom_frequencies(P, T, freq_type)
        print(f"  {freq_type} 频率: {omega[:3].tolist()}... (前3个)")
    
    # 动态调整通道对
    def create_adaptive_rope_pairs(head_dim, ratio=0.5):
        """根据头维度自适应选择通道对"""
        total_pairs = head_dim // 2
        num_rope_pairs = int(total_pairs * ratio)
        return torch.arange(num_rope_pairs, dtype=torch.long)
    
    head_dim = 128
    rope_pairs_idx = create_adaptive_rope_pairs(head_dim, ratio=0.3)
    print(f"\n自适应通道对选择:")
    print(f"  总通道对数: {head_dim // 2}")
    print(f"  Δ-RoPE 通道对数: {len(rope_pairs_idx)}")
    print(f"  比例: {len(rope_pairs_idx) / (head_dim // 2):.1%}")

if __name__ == "__main__":
    print("🎯 Δ-RoPE 使用示例集合")
    print("=" * 60)
    
    try:
        # 基本使用
        example_basic_usage()
        
        # 因果注意力
        example_causal_attention()
        
        # 集成示例
        example_integration_with_existing_attention()
        
        # 性能对比
        example_performance_comparison()
        
        # 高级使用
        example_advanced_usage()
        
        print("\n" + "=" * 60)
        print("🎉 所有示例运行完成！")
        print("\n📚 使用指南:")
        print("  1. 基本使用: 直接调用 sdpa_delta_rope()")
        print("  2. 集成使用: 创建自定义注意力模块")
        print("  3. 性能优化: 根据需求调整通道对比例")
        print("  4. 高级功能: 自定义频率和通道选择策略")
        
    except Exception as e:
        print(f"\n❌ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()
