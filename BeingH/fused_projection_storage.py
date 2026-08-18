"""Inference-only shared storage for fused projection weights.

The helper is device agnostic.  It replaces each projection parameter with a
slice of one contiguous fused allocation, so the original per-projection
allocations are no longer retained while the original state-dict keys remain
valid.  This is intentionally an inference-only transformation.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn


def combined_storage_view(tensors: Iterable[torch.Tensor]) -> Optional[torch.Tensor]:
    """Return one row-major view when tensors are adjacent slices of a storage."""
    items = tuple(tensors)
    if not items:
        return None
    first = items[0]
    if not first.is_contiguous() or first.ndim < 1:
        return None
    storage_ptr = first.untyped_storage().data_ptr()
    expected_offset = first.storage_offset()
    trailing_shape = first.shape[1:]
    for tensor in items:
        if (
            tensor.untyped_storage().data_ptr() != storage_ptr
            or tensor.storage_offset() != expected_offset
            or not tensor.is_contiguous()
            or tensor.shape[1:] != trailing_shape
            or tensor.dtype != first.dtype
            or tensor.device != first.device
        ):
            return None
        expected_offset += tensor.numel()
    rows = sum(tensor.shape[0] for tensor in items)
    return first.as_strided(
        (rows, *trailing_shape),
        first.stride(),
        first.storage_offset(),
    )


def _pack_linear_group(linears: Iterable[nn.Linear]) -> dict[str, int]:
    modules = tuple(linears)
    if len(modules) < 2:
        raise ValueError("projection storage group requires at least two linears")
    weights = tuple(module.weight for module in modules)
    if any(weight.shape[1:] != weights[0].shape[1:] for weight in weights):
        raise ValueError("projection weights must share their input shape")
    if any(weight.device != weights[0].device for weight in weights):
        raise ValueError("projection weights must share one device")
    if any(weight.dtype != weights[0].dtype for weight in weights):
        raise ValueError("projection weights must share one dtype")

    fused_weight = torch.cat([weight.detach() for weight in weights], dim=0)
    offset = 0
    for module, weight in zip(modules, weights, strict=True):
        row_count = weight.shape[0]
        module.weight = nn.Parameter(
            fused_weight.narrow(0, offset, row_count),
            requires_grad=False,
        )
        offset += row_count

    biases = tuple(module.bias for module in modules)
    if any(bias is not None for bias in biases):
        if not all(bias is not None for bias in biases):
            raise ValueError("projection biases must be either all present or all absent")
        fused_bias = torch.cat([bias.detach() for bias in biases], dim=0)  # type: ignore[union-attr]
        offset = 0
        for module, bias in zip(modules, biases, strict=True):
            assert bias is not None
            row_count = bias.shape[0]
            module.bias = nn.Parameter(
                fused_bias.narrow(0, offset, row_count),
                requires_grad=False,
            )
            offset += row_count

    return {
        "weight_bytes": fused_weight.numel() * fused_weight.element_size(),
        "bias_bytes": 0
        if not all(bias is not None for bias in biases)
        else sum(bias.numel() * bias.element_size() for bias in biases),  # type: ignore[union-attr]
    }


def prepare_fused_projection_storage(model: nn.Module) -> dict[str, int]:
    """Pack QKV and Gate/Up groups while preserving state-dict keys.

    The operation works on CUDA, NPU, or CPU tensors.  Each module parameter is
    rebound to a row view into a single fused allocation; callers that later
    request a fused projection can recover that allocation with
    :func:`combined_storage_view` without creating another ``cat`` copy.
    """
    qkv_groups = 0
    gate_up_groups = 0
    shared_bytes = 0
    for module in model.modules():
        if all(
            isinstance(getattr(module, name, None), nn.Linear)
            for name in ("q_proj_mot_gen", "k_proj_mot_gen", "v_proj_mot_gen")
        ):
            stats = _pack_linear_group(
                (
                    module.q_proj_mot_gen,
                    module.k_proj_mot_gen,
                    module.v_proj_mot_gen,
                )
            )
            module._npu_fused_qkv_mot_gen_weight = None
            module._npu_fused_qkv_mot_gen_bias = None
            qkv_groups += 1
            shared_bytes += stats["weight_bytes"] + stats["bias_bytes"]
        if all(
            isinstance(getattr(module, name, None), nn.Linear)
            for name in ("gate_proj", "up_proj")
        ):
            stats = _pack_linear_group((module.gate_proj, module.up_proj))
            module._npu_fused_gate_up_weight = None
            gate_up_groups += 1
            shared_bytes += stats["weight_bytes"] + stats["bias_bytes"]
    return {
        "qkv_groups": qkv_groups,
        "gate_up_groups": gate_up_groups,
        "shared_storage_bytes": shared_bytes,
        "avoided_fused_cache_bytes": shared_bytes,
    }
