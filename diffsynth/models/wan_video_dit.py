import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from prope import _prepare_apply_fns
from typing import Tuple, Optional
from einops import rearrange
from .utils import hash_state_dict_keys
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    # Check if inputs are already in multi-head format by checking tensor dimensions
    # len(shape) == 4 means (batch, num_heads, seqlen, head_dim)
    # len(shape) == 3 means (batch, seqlen, dim)
    already_multihead = len(q.shape) == 4
    
    if already_multihead:
        # Inputs are already in (batch, num_heads, seqlen, head_dim) format
        if compatibility_mode:
            x = F.scaled_dot_product_attention(q, k, v)
        elif FLASH_ATTN_3_AVAILABLE:
            # Rearrange to (batch, seqlen, num_heads, head_dim) for flash_attn_3
            q = rearrange(q, "b n s d -> b s n d", n=num_heads)
            k = rearrange(k, "b n s d -> b s n d", n=num_heads)
            v = rearrange(v, "b n s d -> b s n d", n=num_heads)
            x = flash_attn_interface.flash_attn_func(q, k, v)
            x = rearrange(x, "b s n d -> b n s d", n=num_heads)
        elif FLASH_ATTN_2_AVAILABLE:
            # Rearrange to (batch, seqlen, num_heads, head_dim) for flash_attn_2
            q = rearrange(q, "b n s d -> b s n d", n=num_heads)
            k = rearrange(k, "b n s d -> b s n d", n=num_heads)
            v = rearrange(v, "b n s d -> b s n d", n=num_heads)
            x = flash_attn.flash_attn_func(q, k, v)
            x = rearrange(x, "b s n d -> b n s d", n=num_heads)
        elif SAGE_ATTN_AVAILABLE:
            x = sageattn(q, k, v)
        else:
            x = F.scaled_dot_product_attention(q, k, v)
    else:
        # Original behavior: inputs are in (batch, seqlen, dim) format
        if compatibility_mode:
            q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
            k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
            v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
            x = F.scaled_dot_product_attention(q, k, v)
            x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
        elif FLASH_ATTN_3_AVAILABLE:
            q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
            k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
            v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
            x = flash_attn_interface.flash_attn_func(q, k, v)
            x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
        elif FLASH_ATTN_2_AVAILABLE:
            q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
            k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
            v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
            x = flash_attn.flash_attn_func(q, k, v)
            x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
        elif SAGE_ATTN_AVAILABLE:
            q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
            k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
            v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
            x = sageattn(q, k, v)
            x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
        else:
            q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
            k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
            v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
            x = F.scaled_dot_product_attention(q, k, v)
            x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def rope_apply_(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)

def rope_apply(x, freqs, num_heads, *, mask_first_head_fraction: float = 0.0, t_highfreq_ratio: float = 0.0):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    # Ensure freqs has the same dtype as x for complex operations
    freqs = freqs.to( device=x.device)
    x_out = torch.view_as_complex(x.to(torch.float32).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))

    if mask_first_head_fraction > 0 and t_highfreq_ratio > 0:
        # Build per-head complex-frequency coefficients so we can mask specific heads
        # freqs shape: (seqlen, 1, Lc)
        seqlen, _, Lc = freqs.shape
        heads_to_mask = max(1, int(num_heads * mask_first_head_fraction))
        # Assume 3D split order [t, h, w] across complex bins
        tLc = Lc - 2 * (Lc // 3)
        t_start = 0
        t_end = t_start + tLc
        t_lo = max(1, int(tLc * t_highfreq_ratio + 1e-6))
        # Lowest frequencies are at the end of the t segment (from back to front)
        t_lo_start = max(t_start, t_end - t_lo)
        t_lo_end = t_end
        # Create head-aware freqs tensor aligned as (1, seqlen, num_heads, Lc) to avoid s x s broadcast
        freqs_heads = freqs.view(1, seqlen, 1, Lc).expand(1, seqlen, num_heads, Lc).clone()
        # Mask rotation for selected heads on t-lowfreq complex bins by setting multiplier to 1+0j
        one_c = torch.ones(1, dtype=freqs_heads.dtype, device=freqs_heads.device)
        freqs_heads[:, :, :heads_to_mask, t_lo_start:t_lo_end] = one_c
        # Broadcast to x_out shape (b, s, n, Lc)
        freqs_effective = freqs_heads
    else:
        # Broadcast original freqs across heads aligned as (1, s, n, Lc)
        seqlen, _, Lc = freqs.shape
        freqs_effective = freqs.view(1, seqlen, 1, Lc).expand(1, seqlen, num_heads, Lc)

    x_out = torch.view_as_real(x_out * freqs_effective).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class  PRoPE_SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs, viewmats, Ks=None, *, mask_first_head_fraction: float = 1, t_highfreq_ratio: float = 0.5):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        
        # Apply 3D RoPE first, but mask low-frequency t channels on the first fraction of heads
        q = rope_apply(q, freqs, self.num_heads, mask_first_head_fraction=mask_first_head_fraction, t_highfreq_ratio=t_highfreq_ratio)
        k = rope_apply(k, freqs, self.num_heads, mask_first_head_fraction=mask_first_head_fraction, t_highfreq_ratio=t_highfreq_ratio)
        
        # Rearrange to multi-head format: (batch, seqlen, dim) -> (batch, num_heads, seqlen, head_dim)
        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
        
        # Ensure data type consistency for PRoPE operations
        target_dtype = q.dtype
        target_device = q.device
        
        # Convert viewmats and Ks to match input tensor dtype and device
        viewmats = viewmats.to(dtype=target_dtype, device=target_device)
        if Ks is not None:
            Ks = Ks.to(dtype=target_dtype, device=target_device)
        
        apply_fn_q, apply_fn_kv, apply_fn_o = _prepare_apply_fns(
            head_dim=self.head_dim,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=52,
            patches_y=30,
            image_width=832,
            image_height=480,
            num_heads=self.num_heads,
            head_fraction=mask_first_head_fraction,
            t_highfreq_ratio=t_highfreq_ratio,
        )
        
        # Apply PRoPE transforms
        q = apply_fn_q(q)
        k = apply_fn_kv(k)
        # v = apply_fn_kv(v)
        
        # Apply attention (inputs are already in multi-head format)
        x = self.attn(q, k, v)
        
        # Apply output transform
        # x = apply_fn_o(x)
        
        # Rearrange back to original format: (batch, num_heads, seqlen, head_dim) -> (batch, seqlen, dim)
        x = rearrange(x, "b n s d -> b s (n d)", n=self.num_heads)
        
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6, enable_cam_layers: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.enable_cam_layers = enable_cam_layers

        self.self_attn = PRoPE_SelfAttention(dim, num_heads, eps)
        # self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        
        # Camera layers are registered externally in training script
        # This class only controls whether to use projector or not

    def forward(self, x, context, cam_emb, t_mod, freqs):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        
        
        
        ## TODO:
        # cam_emb: torch.Size([1, 21, 12])
        # x.shape   torch.Size([1, 65520, 1536]) 
        # encode camera

        B, N, _ = cam_emb.shape
        reshaped_cam_emb = cam_emb.view(B, N, 3, 4)
        bottom_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=reshaped_cam_emb.device, dtype=reshaped_cam_emb.dtype)
        bottom_row = bottom_row.unsqueeze(0).expand(N, -1, -1)
        bottom_row = bottom_row.unsqueeze(0).expand(B, -1, -1, -1)
        cam_emb = torch.cat([reshaped_cam_emb, bottom_row], dim=2)
        # cam_emb = cam_emb.unsqueeze(0)

        N = cam_emb.shape[1]
        
        # Ensure Ks has the same dtype and device as input_x
        Ks = torch.tensor([[818.18,0,540],[0,818.18,540],[0,0,1]], device=input_x.device, dtype=input_x.dtype).unsqueeze(0).repeat(N,1,1).unsqueeze(0)
        
        if self.enable_cam_layers:
            x = x + gate_msa * self.projector(self.self_attn(input_x, freqs, cam_emb, Ks))
        else:
            x = x + gate_msa * self.self_attn(input_x, freqs, cam_emb, Ks)


        # cam_emb = self.cam_encoder(cam_emb)
        # cam_emb = cam_emb.repeat(1, 2, 1)
        # cam_emb = cam_emb.unsqueeze(2).unsqueeze(3).repeat(1, 1, 30, 52, 1)
        # cam_emb = rearrange(cam_emb, 'b f h w d -> b (f h w) d')
        # input_x = input_x + cam_emb
        # x = x + gate_msa * self.projector(self.self_attn(input_x, freqs))

        
        
        
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(input_x)
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        enable_cam_layers: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)  # Always start with False, will be set dynamically
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim)  # clip_feature_dim = 1280
    

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                cam_emb: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x)
        
        # 获取 self.freqs[0][:f] 并修改倒数五个通道
        # freq_0 = self.freqs[0][:f].clone()  # 形状: [f, 22]
        # 将倒数五个通道设置为 1.0000+0.0000e+00j
        # freq_0[:, -5:] = torch.complex(torch.ones_like(freq_0[:, -5:].real), torch.zeros_like(freq_0[:, -5:].imag))
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, cam_emb, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, cam_emb, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, cam_emb, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
    
    
class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config
    
    def from_civitai(self, state_dict):
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        else:
            config = {}
        return state_dict, config
