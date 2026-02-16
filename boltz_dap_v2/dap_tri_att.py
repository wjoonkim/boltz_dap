"""
DAP-aware Triangle Attention for Boltz 2 — SDPA edition.

Uses F.scaled_dot_product_attention (FlashAttention / mem-efficient backend)
to eliminate the O(N²) attention matrix that was the #1 memory bottleneck.

Confirmed by profiling:
  - Q@K^T creates a f32 [cs, H, N, N] matrix = 4,734 MB per chunk
  - add biases creates a copy → 9,470 MB peak per chunk
  - SDPA eliminates both by computing attention in O(N) tiles

Starting node: row-scattered z works directly (all N columns available).
Ending node: needs row_to_col to get all N rows, then operates like starting.
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from boltz_distributed.comm import row_to_col, col_to_row, gather
from boltz_distributed.core import get_dap_size, get_dap_rank

from boltz.model.layers.triangular_attention.utils import permute_final_dims


def _sdpa_tri_attn(x, mha, mask_bias, full_bias, chunk_size, diag=False):
    """SDPA-based triangle attention, chunked over the row (batch) dimension.

    Memory optimization: fold mask_bias into full_bias ONCE before the chunk
    loop to avoid materializing a [cs, H, N, N] combined_bias per chunk.

    Args:
        x: [B, I, N, D] — layer-normed input
        mha: the Attention module (has linear_q/k/v/g/o, no_heads, c_hidden)
        mask_bias: [B, I, 1, 1, N] — additive mask (-inf for padding)
        full_bias: [B, 1, H, N, N] — gathered triangle bias
        chunk_size: int — how many rows to process at a time
        diag: bool — if True, print per-step memory profiling
    Returns:
        output: [B, I, N, D]
    """
    B, I, N, D = x.shape
    H = mha.no_heads
    ch = mha.c_hidden
    cs = chunk_size or 128
    scale = 1.0 / math.sqrt(ch)

    rank0 = get_dap_rank() == 0

    # ──────── Fold mask_bias into full_bias ONCE ────────
    # mask_bias: [B, I, 1, 1, N] — key-level mask (same for all query rows)
    # full_bias: [B, 1, H, N, N]
    # Take the mask from the first row (mask is uniform across rows — it only
    # marks which key positions are valid/padded, independent of query position).
    # Result: [B, 1, H, N, N] — only ~37 MB for B=1, H=4, N=1557
    mask_row0 = mask_bias[:, 0:1, :, :, :]  # [B, 1, 1, 1, N] — view, no copy
    # Add in-place to full_bias to avoid extra allocation
    # full_bias [B, 1, H, N, N] + mask_row0 [B, 1, 1, 1, N] → broadcasts on H and query-N dims
    attn_bias = full_bias + mask_row0  # [B, 1, H, N, N] — same size as full_bias (~37 MB)
    del mask_row0
    # Squeeze the I-broadcast dim: [B, 1, H, N, N] -> [B, H, N, N]
    attn_bias = attn_bias.squeeze(-4)  # [B, H, N, N]

    if diag and rank0:
        ab_mb = attn_bias.nelement() * attn_bias.element_size() // (1024 * 1024)
        print(f"          [SDPA-DIAG] attn_bias.shape={list(attn_bias.shape)}, "
              f"dtype={attn_bias.dtype}, size={ab_mb}MB (mask folded in)", flush=True)

    # Allocate output
    output = torch.empty(B, I, N, D, dtype=x.dtype, device=x.device)

    def _reset():
        if not diag or not rank0:
            return 0
        torch.cuda.synchronize()
        a = torch.cuda.memory_allocated(0) // (1024 * 1024)
        torch.cuda.reset_peak_memory_stats(0)
        return a

    def _measure(label, a0):
        if not diag or not rank0:
            return
        torch.cuda.synchronize()
        a1 = torch.cuda.memory_allocated(0) // (1024 * 1024)
        p = torch.cuda.max_memory_allocated(0) // (1024 * 1024)
        print(f"          [SDPA-STEP] {label:24s} | alloc: {a0}→{a1}MB "
              f"(Δ{a1 - a0:+d}) | peak={p}MB | TRANSIENT={p - a0}MB", flush=True)

    # Helper: run SDPA with mem_efficient backend
    def _run_sdpa(q, k, v, bias):
        # Attempt 1: PyTorch >= 2.2 API
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=bias, scale=scale,
                )
        except Exception:
            pass
        # Attempt 2: PyTorch 2.0-2.1 API
        try:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=False, enable_mem_efficient=True
            ):
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=bias, scale=scale,
                )
        except Exception:
            pass
        # Attempt 3: fallback — any backend
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, scale=scale,
        )

    for start in range(0, I, cs):
        end = min(start + cs, I)
        actual_cs = end - start
        is_first = (start == 0)

        # Slice input for this chunk
        x_c = x[:, start:end]  # [B, cs, N, D]

        if is_first and diag:
            a0 = _reset()

        # QKV projection per chunk
        q = mha.linear_q(x_c)  # [B, cs, N, H*ch]
        k = mha.linear_k(x_c)
        v = mha.linear_v(x_c)

        # Reshape: [B, cs, N, H*ch] -> [B*cs, N, H, ch] -> [B*cs, H, N, ch]
        q = q.reshape(B * actual_cs, N, H, ch).transpose(1, 2)
        k = k.reshape(B * actual_cs, N, H, ch).transpose(1, 2)
        v = v.reshape(B * actual_cs, N, H, ch).transpose(1, 2)

        if is_first and diag:
            _measure("QKV projection", a0)

        # SDPA with the pre-merged bias [B, H, N, N]
        # SDPA broadcasts this to [B*cs, H, N, N] internally without materializing!
        if is_first and diag:
            a0 = _reset()

        o = _run_sdpa(q, k, v, attn_bias)
        del q, k, v

        if is_first and diag:
            _measure("SDPA", a0)

        # Reshape back: [B*cs, H, N, ch] -> [B*cs, N, H, ch] -> [B, cs, N, H*ch]
        o = o.transpose(1, 2).reshape(B, actual_cs, N, H * ch)

        # Gating
        if mha.linear_g is not None:
            if is_first and diag:
                a0 = _reset()
            g = mha.sigmoid(mha.linear_g(x_c))
            o = o * g
            del g
            if is_first and diag:
                _measure("gating", a0)

        # Output projection
        if is_first and diag:
            a0 = _reset()

        o = mha.linear_o(o)

        if is_first and diag:
            _measure("output projection", a0)

        output[:, start:end] = o
        del o, x_c

    del attn_bias
    return output


class DAPTriAttStart(nn.Module):
    """DAP wrapper for TriangleAttentionStartingNode.

    Operates on row-scattered z [B, N/dap, N, D].
    Starting node attention: iterate over rows (N/dap, local),
    attend across columns (N, full). Only the bias is gathered.
    Now uses SDPA to avoid materializing the N×N attention matrix.
    """

    def __init__(self, original_tri_att):
        super().__init__()
        self.inner = original_tri_att
        self._diag_count = 0

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        chunk_size: Optional[int] = None,
        use_kernels: bool = False,
    ) -> Tensor:
        dap_size = get_dap_size()
        if dap_size <= 1:
            return self.inner(x, mask, chunk_size, use_kernels)

        if mask is None:
            mask = x.new_ones(x.shape[:-1])

        # Layer norm (pointwise)
        x = self.inner.layer_norm(x)

        # Mask bias: [B, N/dap, 1, 1, N] — cast to bf16 to avoid f32 from Python float
        mask_bias = (self.inner.inf * (mask[..., :, None, None, :] - 1)).to(dtype=x.dtype)

        # Triangle bias: gather only H channels (not D)
        local_bias = self.inner.linear(x)
        local_bias = permute_final_dims(local_bias, (2, 0, 1))

        # Gather dim 2 (N/dap -> N): [B, H, N, N]
        N = x.shape[2]
        full_bias = gather(local_bias.contiguous(), dim=2, original_size=N)
        del local_bias
        full_bias = full_bias.unsqueeze(1)  # [B, 1, H, N, N]

        # Diagnostic profiling for first 2 calls
        diag = self._diag_count < 2 and get_dap_rank() == 0
        if diag:
            print(f"        [TRI-ATT-START] SDPA path #{self._diag_count}, "
                  f"chunk_size={chunk_size}, x.shape={list(x.shape)}", flush=True)
            self._diag_count += 1

        x = _sdpa_tri_attn(x, self.inner.mha, mask_bias, full_bias, chunk_size, diag=diag)
        return x


class DAPTriAttEnd(nn.Module):
    """DAP wrapper for TriangleAttentionEndingNode.

    Ending node needs all N rows for keys/queries.
    Strategy: row_to_col -> transpose -> operate like starting node.
    Only the bias is gathered. Uses SDPA for memory efficiency.
    """

    def __init__(self, original_tri_att):
        super().__init__()
        self.inner = original_tri_att
        self._diag_count = 0

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        chunk_size: Optional[int] = None,
        use_kernels: bool = False,
    ) -> Tensor:
        dap_size = get_dap_size()
        if dap_size <= 1:
            return self.inner(x, mask, chunk_size, use_kernels)

        if mask is None:
            mask = x.new_ones(x.shape[:-1])

        original_N = x.shape[2]

        # 1. row_to_col: [B, N/dap, N, D] -> [B, N_pad, N/dap, D]
        x_col = row_to_col(x)
        mask_col = row_to_col(mask.unsqueeze(-1)).squeeze(-1)

        # 2. Transpose for ending node: [B, N/dap, N_pad, D]
        x_t = x_col.transpose(-2, -3)
        mask_t = mask_col.transpose(-1, -2)
        del x_col, mask_col

        N_pad = x_t.shape[2]

        # 3. Layer norm (pointwise)
        x_t = self.inner.layer_norm(x_t)

        # 4. Mask bias: [B, N/dap, 1, 1, N_pad] — cast to bf16
        mask_bias = (self.inner.inf * (mask_t[..., :, None, None, :] - 1)).to(dtype=x_t.dtype)

        # 5. Triangle bias: gather the small bias
        local_bias = self.inner.linear(x_t)
        local_bias = permute_final_dims(local_bias, (2, 0, 1))
        full_bias = gather(local_bias.contiguous(), dim=2, original_size=N_pad)
        del local_bias
        full_bias = full_bias.unsqueeze(1)  # [B, 1, H, N_pad, N_pad]

        # Diagnostic profiling
        diag = self._diag_count < 2 and get_dap_rank() == 0
        if diag:
            print(f"        [TRI-ATT-END] SDPA path #{self._diag_count}, "
                  f"chunk_size={chunk_size}, x_t.shape={list(x_t.shape)}", flush=True)
            self._diag_count += 1

        # 6. SDPA-based attention
        x_t = _sdpa_tri_attn(x_t, self.inner.mha, mask_bias, full_bias, chunk_size, diag=diag)

        # 7. Transpose back + col_to_row
        x_col_out = x_t.transpose(-2, -3)
        del x_t
        x_out = col_to_row(x_col_out)
        del x_col_out

        # 8. Trim padding
        if x_out.shape[2] > original_N:
            x_out = x_out[:, :, :original_N, :]

        return x_out
