"""Optional CUDA/Triton kernels used by GPU feature-parity inference."""

from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by deployment environments
    triton = None
    tl = None


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    half = tensor.shape[-1] // 2
    return torch.cat((-tensor[..., half:], tensor[..., :half]), dim=-1)


if triton is not None:

    @triton.jit
    def _round_bfloat16_to_float32(value):
        """Force an observable BF16 rounding point in one fused kernel."""
        return tl.inline_asm_elementwise(
            asm="""
            {
                .reg .b32 bits;
                .reg .b32 lsb;
                mov.b32 bits, $1;
                shr.u32 lsb, bits, 16;
                and.b32 lsb, lsb, 1;
                add.u32 bits, bits, 0x7fff;
                add.u32 bits, bits, lsb;
                and.b32 bits, bits, 0xffff0000;
                mov.b32 $0, bits;
            }
            """,
            constraints="=f,f",
            args=[value],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _fused_rotary_kernel(
        query,
        key,
        cos,
        sin,
        query_output,
        key_output,
        query_numel,
        key_numel,
        QUERY_HEADS: tl.constexpr,
        KEY_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        query_mask = offsets < query_numel
        query_offsets = tl.where(query_mask, offsets, 0)
        query_dim = query_offsets % HEAD_DIM
        query_row_start = query_offsets - query_dim
        query_rotary_dim = tl.where(
            query_dim < HEAD_DIM // 2,
            query_dim + HEAD_DIM // 2,
            query_dim - HEAD_DIM // 2,
        )
        query_sequence = query_offsets // (QUERY_HEADS * HEAD_DIM)
        query_value = tl.load(query + query_offsets, mask=query_mask, other=0.0)
        query_rotary = tl.load(
            query + query_row_start + query_rotary_dim,
            mask=query_mask,
            other=0.0,
        )
        query_rotary = tl.where(
            query_dim < HEAD_DIM // 2, -query_rotary, query_rotary
        )
        query_cos = tl.load(
            cos + query_sequence * HEAD_DIM + query_dim,
            mask=query_mask,
            other=0.0,
        )
        query_sin = tl.load(
            sin + query_sequence * HEAD_DIM + query_dim,
            mask=query_mask,
            other=0.0,
        )
        query_left = _round_bfloat16_to_float32(
            query_value.to(tl.float32) * query_cos.to(tl.float32)
        )
        query_right = _round_bfloat16_to_float32(
            query_rotary.to(tl.float32) * query_sin.to(tl.float32)
        )
        tl.store(
            query_output + query_offsets,
            (query_left + query_right).to(tl.bfloat16),
            mask=query_mask,
        )

        key_linear = offsets - query_numel
        key_mask = (key_linear >= 0) & (key_linear < key_numel)
        key_offsets = tl.where(key_mask, key_linear, 0)
        key_dim = key_offsets % HEAD_DIM
        key_row_start = key_offsets - key_dim
        key_rotary_dim = tl.where(
            key_dim < HEAD_DIM // 2,
            key_dim + HEAD_DIM // 2,
            key_dim - HEAD_DIM // 2,
        )
        key_sequence = key_offsets // (KEY_HEADS * HEAD_DIM)
        key_value = tl.load(key + key_offsets, mask=key_mask, other=0.0)
        key_rotary = tl.load(
            key + key_row_start + key_rotary_dim,
            mask=key_mask,
            other=0.0,
        )
        key_rotary = tl.where(
            key_dim < HEAD_DIM // 2, -key_rotary, key_rotary
        )
        key_cos = tl.load(
            cos + key_sequence * HEAD_DIM + key_dim,
            mask=key_mask,
            other=0.0,
        )
        key_sin = tl.load(
            sin + key_sequence * HEAD_DIM + key_dim,
            mask=key_mask,
            other=0.0,
        )
        key_left = _round_bfloat16_to_float32(
            key_value.to(tl.float32) * key_cos.to(tl.float32)
        )
        key_right = _round_bfloat16_to_float32(
            key_rotary.to(tl.float32) * key_sin.to(tl.float32)
        )
        tl.store(
            key_output + key_offsets,
            (key_left + key_right).to(tl.bfloat16),
            mask=key_mask,
        )

    @triton.jit
    def _fused_swiglu_kernel(
        gate_up,
        output,
        output_numel,
        INTERMEDIATE_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < output_numel
        column = offsets % INTERMEDIATE_SIZE
        row = offsets // INTERMEDIATE_SIZE
        gate_offset = row * (2 * INTERMEDIATE_SIZE) + column
        gate = tl.load(gate_up + gate_offset, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(
            gate_up + gate_offset + INTERMEDIATE_SIZE,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        silu = (gate * tl.sigmoid(gate)).to(tl.bfloat16)
        tl.store(
            output + offsets,
            (silu * up.to(tl.bfloat16)).to(tl.bfloat16),
            mask=mask,
        )


def fused_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply half-rotation RoPE with one CUDA kernel for Q and K."""
    eligible = (
        triton is not None
        and query.device.type == "cuda"
        and key.device == query.device
        and query.ndim == 3
        and key.ndim == 3
        and cos.ndim == 2
        and sin.ndim == 2
        and query.shape[0] == key.shape[0] == cos.shape[0] == sin.shape[0]
        and query.shape[-1] == key.shape[-1] == cos.shape[-1] == sin.shape[-1]
        and query.shape[-1] % 2 == 0
        and query.dtype == key.dtype == cos.dtype == sin.dtype == torch.bfloat16
        and query.is_contiguous()
        and key.is_contiguous()
        and cos.is_contiguous()
        and sin.is_contiguous()
    )
    if not eligible:
        cos_expanded = cos.unsqueeze(1)
        sin_expanded = sin.unsqueeze(1)
        return (
            query * cos_expanded + _rotate_half(query) * sin_expanded,
            key * cos_expanded + _rotate_half(key) * sin_expanded,
        )

    query_output = torch.empty_like(query)
    key_output = torch.empty_like(key)
    total = query.numel() + key.numel()
    grid = (triton.cdiv(total, 256),)
    _fused_rotary_kernel[grid](
        query,
        key,
        cos,
        sin,
        query_output,
        key_output,
        query.numel(),
        key.numel(),
        QUERY_HEADS=query.shape[1],
        KEY_HEADS=key.shape[1],
        HEAD_DIM=query.shape[-1],
        BLOCK_SIZE=256,
    )
    return query_output, key_output


def fused_swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """Apply SiLU(gate) * up with one CUDA kernel."""
    intermediate_size = gate_up.shape[-1] // 2
    eligible = (
        triton is not None
        and gate_up.device.type == "cuda"
        and gate_up.shape[-1] % 2 == 0
        and gate_up.dtype == torch.bfloat16
        and gate_up.is_contiguous()
    )
    if not eligible:
        gate, up = gate_up.split(intermediate_size, dim=-1)
        return torch.nn.functional.silu(gate) * up

    output = torch.empty(
        (*gate_up.shape[:-1], intermediate_size),
        dtype=gate_up.dtype,
        device=gate_up.device,
    )
    grid = (triton.cdiv(output.numel(), 256),)
    _fused_swiglu_kernel[grid](
        gate_up,
        output,
        output.numel(),
        INTERMEDIATE_SIZE=intermediate_size,
        BLOCK_SIZE=256,
    )
    return output
