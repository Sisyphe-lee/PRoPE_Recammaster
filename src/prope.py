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

# How to use PRoPE attention for self-attention:
# 
# 1. Easiest way (fast):
#    attn = PropeDotProductAttention(...)
#    o = attn(q, k, v, viewmats, Ks)
#
# 2. More flexible way (fast):
#    attn = PropeDotProductAttention(...)
#    attn._precompute_and_cache_apply_fns(viewmats, Ks)
#    q = attn._apply_to_q(q)
#    k = attn._apply_to_kv(k)
#    v = attn._apply_to_kv(v)
#    o = F.scaled_dot_product_attention(q, k, v, **kwargs)
#    o = attn._apply_to_o(o)
# 
# 3. The most flexible way (but slower because repeated computation of RoPE coefficients):
#    o = prope_dot_product_attention(q, k, v, ...)
# 
# How to use PRoPE attention for cross-attention:
# 
#    attn_src = PropeDotProductAttention(...)
#    attn_tgt = PropeDotProductAttention(...)
#    attn_src._precompute_and_cache_apply_fns(viewmats_src, Ks_src)
#    attn_tgt._precompute_and_cache_apply_fns(viewmats_tgt, Ks_tgt)
#    q_src = attn_src._apply_to_q(q_src)
#    k_tgt = attn_tgt._apply_to_kv(k_tgt)
#    v_tgt = attn_tgt._apply_to_kv(v_tgt)
#    o_src = F.scaled_dot_product_attention(q_src, k_tgt, v_tgt, **kwargs)
#    o_src = attn_src._apply_to_o(o_src)

from functools import partial
from typing import Callable, Optional, Tuple, List

import torch
import torch.nn.functional as F


class PropeDotProductAttention(torch.nn.Module):
    """PRoPE attention with precomputed RoPE coefficients."""

    coeffs_x_0: torch.Tensor
    coeffs_x_1: torch.Tensor
    coeffs_y_0: torch.Tensor
    coeffs_y_1: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        patches_x: int,
        patches_y: int,
        image_width: int,
        image_height: int,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.image_width = image_width
        self.image_height = image_height

        coeffs_x: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
            torch.tile(torch.arange(patches_x), (patches_y,)),
            freq_base=freq_base,
            freq_scale=freq_scale,
            feat_dim=head_dim // 4,
        )
        coeffs_y: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
            torch.repeat_interleave(torch.arange(patches_y), patches_x),
            freq_base=freq_base,
            freq_scale=freq_scale,
            feat_dim=head_dim // 4,
        )
        # Do not save coeffs to checkpoint as `cameras` might change during testing.
        self.register_buffer("coeffs_x_0", coeffs_x[0], persistent=False)
        self.register_buffer("coeffs_x_1", coeffs_x[1], persistent=False)
        self.register_buffer("coeffs_y_0", coeffs_y[0], persistent=False)
        self.register_buffer("coeffs_y_1", coeffs_y[1], persistent=False)

    # override load_state_dict to not load coeffs if they exist (for backward compatibility)
    def load_state_dict(self, state_dict, strict=True):
        # remove coeffs from state_dict
        state_dict.pop("coeffs_x_0", None)
        state_dict.pop("coeffs_x_1", None)
        state_dict.pop("coeffs_y_0", None)
        state_dict.pop("coeffs_y_1", None)
        super().load_state_dict(state_dict, strict)

    def forward(
        self,
        q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
        Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
        **kwargs,
    ) -> torch.Tensor:
        return prope_dot_product_attention(
            q,
            k,
            v,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=self.patches_x,
            patches_y=self.patches_y,
            image_width=self.image_width,
            image_height=self.image_height,
            coeffs_x=(self.coeffs_x_0, self.coeffs_x_1),
            coeffs_y=(self.coeffs_y_0, self.coeffs_y_1),
            **kwargs,
        )

    def _precompute_and_cache_apply_fns(
        self, viewmats: torch.Tensor, Ks: Optional[torch.Tensor]
    ):
        (batch, cameras, _, _) = viewmats.shape
        assert viewmats.shape == (batch, cameras, 4, 4)
        assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
        self.cameras = cameras

        self.apply_fn_q, self.apply_fn_kv, self.apply_fn_o = _prepare_apply_fns(
            head_dim=self.head_dim,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=self.patches_x,
            patches_y=self.patches_y,
            image_width=self.image_width,
            image_height=self.image_height,
            coeffs_x=(self.coeffs_x_0, self.coeffs_x_1),
            coeffs_y=(self.coeffs_y_0, self.coeffs_y_1),
        )

    def _apply_to_q(self, q: torch.Tensor) -> torch.Tensor:
        (batch, num_heads, seqlen, head_dim) = q.shape
        assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        assert q.shape == (batch, num_heads, seqlen, head_dim)
        assert self.apply_fn_q is not None
        return self.apply_fn_q(q)

    def _apply_to_kv(self, kv: torch.Tensor) -> torch.Tensor:
        (batch, num_heads, seqlen, head_dim) = kv.shape
        assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        assert kv.shape == (batch, num_heads, seqlen, head_dim)
        assert self.apply_fn_kv is not None
        return self.apply_fn_kv(kv)

    def _apply_to_o(self, o: torch.Tensor) -> torch.Tensor:
        (batch, num_heads, seqlen, head_dim) = o.shape
        assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        assert o.shape == (batch, num_heads, seqlen, head_dim)
        assert self.apply_fn_o is not None
        return self.apply_fn_o(o)


def prope_dot_product_attention(
    q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    *,
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
    patches_x: int,  # How many patches wide is each image?
    patches_y: int,  # How many patches tall is each image?
    image_width: int,  # Width of the image. Used to normalize intrinsics.
    image_height: int,  # Height of the image. Used to normalize intrinsics.
    coeffs_x: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    coeffs_y: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    """Similar to torch.nn.functional.scaled_dot_product_attention, but applies PRoPE-style
    positional encoding.

    Currently, we assume that the sequence length is equal to:

        cameras * patches_x * patches_y

    And token ordering allows the `(seqlen,)` axis to be reshaped into
    `(cameras, patches_x, patches_y)`.
    """
    # We're going to assume self-attention: all inputs are the same shape.
    (batch, num_heads, seqlen, head_dim) = q.shape
    cameras = viewmats.shape[1]
    assert q.shape == k.shape == v.shape
    assert viewmats.shape == (batch, cameras, 4, 4)
    assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
    assert seqlen == cameras * patches_x * patches_y

    apply_fn_q, apply_fn_kv, apply_fn_o = _prepare_apply_fns(
        head_dim=head_dim,
        viewmats=viewmats,
        Ks=Ks,
        patches_x=patches_x,
        patches_y=patches_y,
        image_width=image_width,
        image_height=image_height,
        coeffs_x=coeffs_x,
        coeffs_y=coeffs_y,
    )

    out = F.scaled_dot_product_attention(
        query=apply_fn_q(q),
        key=apply_fn_kv(k),
        value=apply_fn_kv(v),
        **kwargs,
    )
    out = apply_fn_o(out)
    assert out.shape == (batch, num_heads, seqlen, head_dim)
    return out


def _prepare_apply_fns(
    head_dim: int,  # Q/K/V will have this last dimension
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
    patches_x: int,  # How many patches wide is each image?
    patches_y: int,  # How many patches tall is each image?
    image_width: int,  # Width of the image. Used to normalize intrinsics.
    image_height: int,  # Height of the image. Used to normalize intrinsics.
    *,
    num_heads: Optional[int] = None,
    head_fraction: float = 0.0,  # e.g., 0.25 means first quarter heads
    t_highfreq_ratio: float = 0.0,  # reused as w-lowfreq ratio for distance RoPE
    original_trans: Optional[torch.Tensor] = None,  # (batch, cameras, 3)
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Prepare transforms for PRoPE-style positional encoding.

    Modified: remove x/y RoPE; only apply projection matrices to a subset of heads
    and only to the low-frequency tail of the t-channel block (last portion of head_dim).
    """
    device = viewmats.device
    (batch, cameras, _, _) = viewmats.shape

    # Normalize camera intrinsics.
    if Ks is not None:
        Ks_norm = torch.zeros_like(Ks)
        Ks_norm[..., 0, 0] = Ks[..., 0, 0] / image_width
        Ks_norm[..., 1, 1] = Ks[..., 1, 1] / image_height
        Ks_norm[..., 0, 2] = Ks[..., 0, 2] / image_width - 0.5
        Ks_norm[..., 1, 2] = Ks[..., 1, 2] / image_height - 0.5
        Ks_norm[..., 2, 2] = 1.0
        del Ks

        # Compute the camera projection matrices we use in PRoPE.
        # - K is an `image<-camera` transform.
        # - viewmats is a `camera<-world` transform.
        # - P = lift(K) @ viewmats is an `image<-world` transform.
        P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
        P_T = P.transpose(-1, -2)
        P_inv = torch.einsum(
            "...ij,...jk->...ik",
            _invert_SE3(viewmats),
            _lift_K(_invert_K(Ks_norm)),
        )

    else:
        # GTA formula. P is `camera<-world` transform.
        P = viewmats
        P_T = P.transpose(-1, -2)
        P_inv = _invert_SE3(viewmats)

    assert P.shape == P_inv.shape == (batch, cameras, 4, 4)

    # Configure subset selection
    assert head_dim % 4 == 0
    # Align 3D RoPE blocks: total complex bins Lc = head_dim//2; split [t,h,w] in complex bins
    Lc = head_dim // 2
    tLc = Lc - 2 * (Lc // 3)
    hLc = (Lc // 3)
    wLc = (Lc // 3)
    # Real-channel index helpers
    def complex_to_real_span(start_bin: int, end_bin: int) -> Tuple[int, int]:
        # map complex-bin range [start_bin, end_bin) to real-channel [2*start, 2*end)
        return 2 * start_bin, 2 * end_bin

    # PRoPE projection still uses t low-frequency tail selected by t_highfreq_ratio
    t_block_start = 0
    t_block_end = 2 * tLc
    if t_highfreq_ratio > 0:
        t_lo_len_bins = int((t_block_end - t_block_start) * t_highfreq_ratio) // 2  # convert to complex bins
        t_lo_len_bins = max(0, (t_lo_len_bins // 2) * 2)  # ensure multiple of 2 bins => 4 real channels
    else:
        t_lo_len_bins = 0
    t_lo_start_bin = max(t_block_start // 2, (t_block_end // 2) - t_lo_len_bins)
    t_lo_end_bin = t_block_end // 2
    t_real_start, t_real_end = complex_to_real_span(t_lo_start_bin, t_lo_end_bin)

    # Distance RoPE uses w-dim low-frequency tail with the same ratio
    w_start_bin = tLc + hLc
    w_end_bin = tLc + hLc + wLc
    if t_highfreq_ratio > 0:
        w_lo_len_bins = int(wLc * t_highfreq_ratio)
        w_lo_len_bins = max(0, (w_lo_len_bins // 2) * 2)  # even number of pairs
    else:
        w_lo_len_bins = 0
    w_lo_start_bin = max(w_start_bin, w_end_bin - w_lo_len_bins)
    w_lo_end_bin = w_end_bin
    w_real_start, w_real_end = complex_to_real_span(w_lo_start_bin, w_lo_end_bin)
    num_pairs_w = (w_real_end - w_real_start) // 2

    if num_heads is None or head_fraction <= 0:
        head_indices = None  # no-op
    else:
        head_indices = torch.arange(max(1, int(num_heads * head_fraction)), device=device)

    # Fixed 16 unit vectors covering axes, plane diagonals, and space diagonals
    U_const = torch.tensor([
        [ 1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0, -1.0,  0.0],
        [ 0.0,  0.0,  1.0],
        [ 0.0,  0.0, -1.0],
        [ 0.70710678,  0.70710678,  0.0],
        [ 0.70710678, -0.70710678,  0.0],
        [ 0.70710678,  0.0,  0.70710678],
        [ 0.70710678,  0.0, -0.70710678],
        [ 0.0,  0.70710678,  0.70710678],
        [ 0.0,  0.70710678, -0.70710678],
        [ 0.57735027,  0.57735027,  0.57735027],
        [ 0.57735027,  0.57735027, -0.57735027],
        [ 0.57735027, -0.57735027,  0.57735027],
        [-0.57735027,  0.57735027,  0.57735027],
    ], dtype=torch.float32, device=device)

    def _apply_proj_subset(feats: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """
        Apply projection matrix to a subset of heads and the low-frequency tail of the t-channel block.
        """
        if t_lo_len_bins == 0 or head_indices is None:
            return feats
        (B, H, S, D) = feats.shape
        if (t_real_end - t_real_start) == 0:
            return feats
        assert ((t_real_end - t_real_start) % 4) == 0
        feats[:, head_indices, :, t_real_start:t_real_end] = _apply_tiled_projmat(
            feats[:, head_indices, :, t_real_start:t_real_end], matrix
        )
        return feats

    def _apply_distance_rope(feats: torch.Tensor) -> torch.Tensor:
        """Apply distance-based rotary on w-dim low-frequency real channels (2-ch pairs).
        feats: (B,H,S,D)
        """
        if num_pairs_w == 0 or head_indices is None:
            return feats
        (B, H, S, D) = feats.shape
        if (w_real_end - w_real_start) == 0:
            return feats
        # Determine cameras and patches
        # Camera translations (original scale preferred), shape (B,C,3)
        if original_trans is not None:
            p = original_trans.to(device=device, dtype=feats.dtype)
        else:
            # use c2w translation from P_inv
            p = P_inv[..., :3, 3].to(dtype=feats.dtype)
        # fixed scaling by 1/100 
        p = p / 100.0
        # Align provided translations to cameras dimension from viewmats
        # Ensure p shape is (B, cameras, 3)
        if p.dim() == 2:
            p = p.unsqueeze(0)
        if p.shape[1] != cameras:
            if p.shape[1] > cameras:
                p = p[:, :cameras, :]
            else:
                raise RuntimeError(f"original_trans cams ({p.shape[1]}) != viewmats cams ({cameras})")
        C = cameras
        assert S % C == 0, f"S={S} not divisible by C={C}"
        P = S // C
        # Determine number of rotation pairs from span length to avoid mismatch
        Kpairs = max(0, (w_real_end - w_real_start) // 2)
        if Kpairs == 0:
            return feats
        # Prepare frequency magnitudes s_k with base=10000 for Kpairs
        k_idx = torch.arange(Kpairs, device=device, dtype=torch.float32)
        # s_k = 10000^(-k/Kpairs)
        s_k = torch.pow(torch.tensor(10000.0, device=device, dtype=torch.float32), -k_idx / float(Kpairs))
        s_k = s_k.to(feats.dtype)
        # omega_k = s_k * u_k
        U = U_const[:Kpairs, :].to(dtype=feats.dtype)
        omega = U * s_k.reshape(-1, 1)
        # theta[b,c,k] = dot(p[b,c], omega[k])
        theta = torch.einsum('bcj,kj->bck', p, omega)  # (B,C,Kpairs)
        cos_t = torch.cos(theta).unsqueeze(-1)  # (B,C,Kpairs,1)
        sin_t = torch.sin(theta).unsqueeze(-1)  # (B,C,Kpairs,1)
        # Reshape feats to (B,H,C,P,D)
        x = feats.reshape(B, H, C, P, D)
        sel = x[:, head_indices, :, :, w_real_start:w_real_end]
        # Pairwise rotation along last dim (size = 2*Kpairs)
        x1 = sel[..., 0::2]
        x2 = sel[..., 1::2]
        # Broadcast theta over heads and patches
        cos_b = cos_t.reshape(B, 1, C, 1, Kpairs, 1)
        sin_b = sin_t.reshape(B, 1, C, 1, Kpairs, 1)
        x1r = x1.reshape(B, -1, C, P, Kpairs, 1)
        x2r = x2.reshape(B, -1, C, P, Kpairs, 1)
        rot1 = x1r * cos_b - x2r * sin_b
        rot2 = x1r * sin_b + x2r * cos_b
        sel_rot = torch.empty_like(sel)
        sel_rot[..., 0::2] = rot1.reshape_as(x1)
        sel_rot[..., 1::2] = rot2.reshape_as(x2)
        x[:, head_indices, :, :, w_real_start:w_real_end] = sel_rot
        return x.reshape(B, H, S, D)

    def apply_fn_q(feats: torch.Tensor) -> torch.Tensor:
        feats = _apply_distance_rope(feats)
        feats = _apply_proj_subset(feats, P_T)
        return feats

    def apply_fn_kv(feats: torch.Tensor) -> torch.Tensor:
        feats = _apply_distance_rope(feats)
        feats = _apply_proj_subset(feats, P_inv)
        return feats

    def apply_fn_o(feats: torch.Tensor) -> torch.Tensor:
        feats = _apply_distance_rope(feats)
        feats = _apply_proj_subset(feats, P)
        return feats

    return apply_fn_q, apply_fn_kv, apply_fn_o
   # relative projection matrix RP = torch.einsum('amn,bnk->abmk', P_T[0], P_inv[0].transpose(2,1)).shape
   # RP[i,j] = RT 

def _apply_tiled_projmat(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    matrix: torch.Tensor,  # (batch, cameras, D, D)
) -> torch.Tensor:
    """Apply projection matrix to features."""
    # - seqlen => (cameras, patches_x * patches_y)
    # - feat_dim => (feat_dim // 4, 4)
    (batch, num_heads, seqlen, feat_dim) = feats.shape
    cameras = matrix.shape[1]
    assert seqlen > cameras and seqlen % cameras == 0
    D = matrix.shape[-1]
    assert matrix.shape == (batch, cameras, D, D)
    assert feat_dim % D == 0
    # Equivalent readable form (for reference only):
    # Shapes:
    #   matrix: (batch, cameras, D, D)
    #   feats:  (batch, num_heads, seqlen, feat_dim)
    #   where seqlen = cameras * patches, feat_dim = (feat_dim // D) * D
    # Reshape feats -> (batch, num_heads, cameras, patches, feat_dim // D, D)
    # For each (batch, cameras), apply matrix (D x D) to the last dim D of feats.
    # Pseudocode:
    #   x = feats.reshape(batch, num_heads, cameras, patches, feat_dim // D, D)
    #   # move dims to (batch, cameras, num_heads, patches, feat_dim // D, D)
    #   x = x.permute(0, 2, 1, 3, 4, 5)
    #   # matmul along last dim: (D, D) @ (D) -> (D)
    #   x = torch.matmul(matrix[:, :, None, None, None, : , :], x)
    #   # move back and reshape to original
    #   x = x.permute(0, 2, 1, 3, 4, 5).reshape_as(feats)
    # einsum below compactly does the same operation.
    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix,
        feats.reshape((batch, num_heads, cameras, -1, feat_dim // D, D)), #  feats @ matrix !!!!
    ).reshape(feats.shape)


def _rope_precompute_coeffs(
    positions: torch.Tensor,  # (seqlen,)
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE coefficients."""
    assert len(positions.shape) == 1
    assert feat_dim % 2 == 0
    num_freqs = feat_dim // 2
    freqs = freq_scale * (
        freq_base
        ** (
            -torch.arange(num_freqs, device=positions.device, dtype=positions.dtype)[None, None, None, :]
            / num_freqs
        )
    )
    angles = positions[None, None, :, None] * freqs
    # Shape should be: `(batch, num_heads, seqlen, num_freqs)`; we're
    # broadcasting across `batch` and `num_heads`.
    assert angles.shape == (1, 1, positions.shape[0], num_freqs)
    return torch.cos(angles), torch.sin(angles)


def _rope_apply_coeffs(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    coeffs: Tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    """Apply RoPE coefficients to features. We adopt a 'split' ordering
    convention. (in contrast to 'interleaved')"""
    cos, sin = coeffs
    # We allow (cos, sin) to be either with shape (1, 1, seqlen, feat_dim // 2),
    # or (1, 1, seqlen_per_image, feat_dim // 2) and we repeat it to
    # match the shape of feats.
    if cos.shape[2] != feats.shape[2]:
        n_repeats = feats.shape[2] // cos.shape[2]
        cos = cos.repeat(1, 1, n_repeats, 1)
        sin = sin.repeat(1, 1, n_repeats, 1)
    assert len(feats.shape) == len(cos.shape) == len(sin.shape) == 4
    assert cos.shape[-1] == sin.shape[-1] == feats.shape[-1] // 2
    x_in = feats[..., : feats.shape[-1] // 2]
    y_in = feats[..., feats.shape[-1] // 2 :]
    return torch.cat(
        (
            [cos * x_in + sin * y_in, -sin * x_in + cos * y_in]
            if not inverse
            else [cos * x_in - sin * y_in, sin * x_in + cos * y_in]
        ),
        dim=-1,
    )


def _apply_block_diagonal(
    feats: torch.Tensor,  # (..., dim)
    func_size_pairs: List[Tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Apply a block-diagonal function to an input array.

    Each function is specified as a tuple with form:

        ((Tensor) -> Tensor, int)

    Where the integer is the size of the input to the function.
    """
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat(
        [f(x_block) for f, x_block in zip(funcs, x_blocks)],
        dim=-1,
    )
    assert out.shape == feats.shape, "Input/output shapes should match."
    return out


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """Lift 3x3 matrices to homogeneous 4x4 matrices."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device, dtype=Ks.dtype)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert 3x3 intrinsics matrices. Assumes no skew."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out