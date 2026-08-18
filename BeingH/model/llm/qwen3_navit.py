# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import glob
import torch
import re
from functools import partial
from typing import Any, Dict, List, Optional, Tuple
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.functional import scaled_dot_product_attention
from safetensors import safe_open
try:
    import torch_npu
except ImportError:
    torch_npu = None
from BeingH.npu_single_sample_fast_path import (
    real_sample_lens as _real_sample_lens,
)
from BeingH.npu_prefix_segment_route import (
    PrefixSegment,
    PrefixSegmentRoute,
)
from BeingH.cuda_fused_ops import fused_rotary
from BeingH.fused_projection_storage import combined_storage_view
from .qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP, 
    Qwen3PreTrainedModel, 
    Qwen3RMSNorm, 
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

from .qwen3.configuration_qwen3 import Qwen3Config as _Qwen3Config
from .qwen2_navit import get_layer_mapping_strategy, interpolate_layer_params
from .qwen2_navit import NaiveCache, BaseNavitOutputWithPast, pad_sequence

torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096
flex_attention = torch.compile(flex_attention)


def _apply_rotary_pos_emb(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    enable_npu_fused_rotary: bool,
    enable_cuda_fused_rotary: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply half-rotation RoPE, optionally using the native NPU kernel.

    Being-H stores attention projections as ``[S, N, D]`` and the shared
    rotary tables as ``[S, D]``.  ``npu_rotary_mul`` consumes four-dimensional
    BSND tensors, so the added batch/head singleton dimensions are metadata
    only.  Non-NPU execution deliberately retains the checkpoint's original
    PyTorch expression.
    """
    if (
        enable_cuda_fused_rotary
        and query_states.device.type == "cuda"
    ):
        return fused_rotary(query_states, key_states, cos, sin)

    if not (
        enable_npu_fused_rotary
        and torch_npu is not None
        and query_states.device.type == "npu"
    ):
        return apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            unsqueeze_dim=1,
        )

    if query_states.ndim != 3 or key_states.ndim != 3:
        raise ValueError("NPU fused RoPE expects [S, N, D] query/key tensors")
    if cos.ndim != 2 or sin.ndim != 2:
        raise ValueError("NPU fused RoPE expects [S, D] cosine/sine tables")
    cos_bsnd = cos.unsqueeze(0).unsqueeze(2)
    sin_bsnd = sin.unsqueeze(0).unsqueeze(2)
    query_output = torch_npu.npu_rotary_mul(
        query_states.unsqueeze(0), cos_bsnd, sin_bsnd, rotary_mode="half"
    )
    key_output = torch_npu.npu_rotary_mul(
        key_states.unsqueeze(0), cos_bsnd, sin_bsnd, rotary_mode="half"
    )
    return query_output.squeeze(0), key_output.squeeze(0)


def _npu_fusion_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    head_num: int,
    scale: float,
    input_layout: str = "BNSD",
) -> torch.Tensor:
    """Run fused attention with native GQA for logical [N, S, D] tensors."""
    if torch_npu is None or query_states.device.type != "npu":
        raise RuntimeError("NPU fusion attention requires torch-npu NPU tensors")

    # The fused op uses True for masked positions. The existing dense path
    # represents the same mask as additive 0/-inf values.
    bool_mask = attention_mask.to(torch.bool)
    if input_layout == "BSND":
        # Callers hold transposed [N, S, D] views of contiguous [S, N, D]
        # projections.  Transposing the views back is metadata-only and lets
        # the fused op consume the naturally contiguous BSND representation.
        query_input = query_states.transpose(0, 1).unsqueeze(0)
        key_input = key_states.transpose(0, 1).unsqueeze(0)
        value_input = value_states.transpose(0, 1).unsqueeze(0)
    elif input_layout == "BNSD":
        query_input = query_states.unsqueeze(0)
        key_input = key_states.unsqueeze(0)
        value_input = value_states.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported NPU fusion-attention layout: {input_layout}")

    output = torch_npu.npu_fusion_attention(
        query_input,
        key_input,
        value_input,
        head_num=head_num,
        input_layout=input_layout,
        atten_mask=bool_mask,
        scale=scale,
        keep_prob=1.0,
        sparse_mode=1,
        sync=True,
    )[0]
    output = output.squeeze(0)
    return output.transpose(0, 1) if input_layout == "BSND" else output


def _cuda_grouped_query_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run memory-efficient CUDA GQA without materialized K/V repeats.

    Folding the KV-head index into the batch dimension lets SDPA see
    ``groups`` query heads and stride-zero expanded K/V heads.  It retains the
    legacy memory-efficient kernel's BF16 reduction order while removing the
    physical K/V repeat.
    """
    if query.device.type != "cuda":
        raise RuntimeError("CUDA grouped-query attention requires CUDA tensors")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("CUDA grouped-query attention expects [B, N, S, D]")
    batch_size, query_heads, query_length, head_dim = query.shape
    key_batch, key_heads, key_length, key_dim = key.shape
    if (
        key_batch != batch_size
        or value.shape != key.shape
        or key_dim != head_dim
        or query_heads % key_heads != 0
    ):
        raise ValueError("incompatible CUDA grouped-query attention shapes")
    groups = query_heads // key_heads
    grouped_query = query.reshape(
        batch_size * key_heads, groups, query_length, head_dim
    )
    grouped_key = key.reshape(
        batch_size * key_heads, 1, key_length, head_dim
    ).expand(batch_size * key_heads, groups, key_length, head_dim)
    grouped_value = value.reshape(
        batch_size * key_heads, 1, key_length, head_dim
    ).expand(batch_size * key_heads, groups, key_length, head_dim)
    if attention_mask.ndim == 3:
        attention_mask = attention_mask.unsqueeze(1)
    with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
        output = scaled_dot_product_attention(
            grouped_query,
            grouped_key,
            grouped_value,
            attention_mask,
        )
    return output.reshape(batch_size, query_heads, query_length, head_dim)


def _effective_npu_attention_layout(
    requested_layout: str,
    real_sample_count: int,
    prefer_bnsd: bool = False,
) -> str:
    """Keep packed multi-sample inference on the established BNSD path."""
    if prefer_bnsd:
        return "BNSD"
    if requested_layout == "BSND" and real_sample_count == 1:
        return "BSND"
    return "BNSD"


class Qwen3Config(_Qwen3Config):
    model_type = "qwen3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, qk_norm=True, layer_module="Qwen3DecoderLayer", **kwargs):
        super().__init__(**kwargs)
        self.qk_norm = qk_norm
        self.layer_module = layer_module


class PackedAttention(Qwen3Attention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        if self.config.qk_norm:
            self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        self.enable_npu_fusion_attention = False
        self.npu_fusion_attention_input_layout = "BNSD"
        self.enable_npu_hybrid_attention_layout = False
        self.enable_npu_qkv_fusion = False
        self.enable_npu_dtype_fast_path = False
        self.enable_npu_fused_rotary = False
        self.enable_cuda_fused_rotary = False
        self.enable_cuda_gqa_attention = False
        self.use_npu_single_sample_fast_path = False

    def forward(self, *args, **kwargs):
        return self.forward_train(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask: List[torch.Tensor],
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
    ):
        total_seq_len = packed_sequence.shape[0]
        dtype, device = packed_sequence.dtype, packed_sequence.device
        packed_query_states = torch.zeros((total_seq_len, self.num_heads * self.head_dim), dtype=dtype, device=device)
        packed_key_states = torch.zeros((total_seq_len, self.num_key_value_heads * self.head_dim), dtype=dtype, device=device)
        packed_value_states = torch.zeros((total_seq_len, self.num_key_value_heads * self.head_dim), dtype=dtype, device=device)

        packed_query_states[packed_und_token_indexes] = self.q_proj(packed_sequence).to(dtype)
        packed_key_states[packed_und_token_indexes] = self.k_proj(packed_sequence).to(dtype)
        packed_value_states[packed_und_token_indexes] = self.v_proj(packed_sequence).to(dtype)

        packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
        packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)

        packed_query_states_ = packed_query_states.new_zeros(packed_query_states.shape)
        packed_key_states_ = packed_key_states.new_zeros(packed_key_states.shape)

        packed_query_states_[packed_und_token_indexes] = self.q_norm(packed_query_states[packed_und_token_indexes])
        packed_key_states_[packed_und_token_indexes] = self.k_norm(packed_key_states[packed_und_token_indexes])
        
        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states, packed_key_states = _apply_rotary_pos_emb(
            packed_query_states_,
            packed_key_states_,
            packed_cos,
            packed_sin,
            enable_npu_fused_rotary=self.enable_npu_fused_rotary,
            enable_cuda_fused_rotary=self.enable_cuda_fused_rotary,
        )

        if isinstance(attention_mask, List):
            real_sample_lens = _real_sample_lens(
                sample_lens, packed_query_states_.shape[0]
            )
            use_cuda_gqa = (
                self.enable_cuda_gqa_attention and device.type == "cuda"
            )
            use_native_gqa = self.enable_npu_fusion_attention or use_cuda_gqa
            fusion_attention_input_layout = _effective_npu_attention_layout(
                self.npu_fusion_attention_input_layout,
                len(real_sample_lens),
                self.enable_npu_hybrid_attention_layout,
            )
            if not use_native_gqa:
                # Keep the pre-OPT-02 dense path byte-for-byte equivalent:
                # expand [S, Nkv, D] before transposing/splitting. Expanding
                # after the transpose would reshape [Nkv, S, G, D] and mix
                # sequence positions with GQA groups.
                packed_key_states_ = packed_key_states_[:, :, None, :].repeat(
                    1, 1, self.num_key_value_groups, 1
                ).reshape(-1, self.num_heads, self.head_dim)
                packed_value_states = packed_value_states[
                    :, :, None, :
                ].repeat(
                    1, 1, self.num_key_value_groups, 1
                ).reshape(-1, self.num_heads, self.head_dim)
            if self.use_npu_single_sample_fast_path:
                if len(real_sample_lens) != 1:
                    raise RuntimeError(
                        "OPT-03 fast path reached attention with "
                        f"{len(real_sample_lens)} real samples"
                    )
                query_states = packed_query_states_.transpose(0, 1)
                key_states = packed_key_states_.transpose(0, 1)
                value_states = packed_value_states.transpose(0, 1)
                attention_mask_per_sample = attention_mask[0]
                if self.enable_npu_fusion_attention:
                    packed_attn_output = _npu_fusion_attention(
                        query_states.to(torch.bfloat16),
                        key_states.to(torch.bfloat16),
                        value_states.to(torch.bfloat16),
                        attention_mask_per_sample,
                        head_num=self.num_heads,
                        scale=self.head_dim**-0.5,
                        input_layout=fusion_attention_input_layout,
                    )
                elif use_cuda_gqa:
                    packed_attn_output = _cuda_grouped_query_attention(
                        query_states.to(torch.bfloat16).unsqueeze(0),
                        key_states.to(torch.bfloat16).unsqueeze(0),
                        value_states.to(torch.bfloat16).unsqueeze(0),
                        attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                    ).squeeze(0)
                else:
                    with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                        packed_attn_output = scaled_dot_product_attention(
                            query_states.to(torch.bfloat16).unsqueeze(0),
                            key_states.to(torch.bfloat16).unsqueeze(0),
                            value_states.to(torch.bfloat16).unsqueeze(0),
                            attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                        ).squeeze(0)
            else:
                unpacked_query_states = packed_query_states_.transpose(0, 1).split(real_sample_lens, dim=1)
                unpacked_key_states = packed_key_states_.transpose(0, 1).split(real_sample_lens, dim=1)
                unpacked_value_states = packed_value_states.transpose(0, 1).split(real_sample_lens, dim=1)
                upacked_attn_output = []
                for query_states, key_states, value_states, attention_mask_per_sample in zip(
                    unpacked_query_states, unpacked_key_states, unpacked_value_states, attention_mask
                ):
                    if self.enable_npu_fusion_attention:
                        attn_output = _npu_fusion_attention(
                            query_states.to(torch.bfloat16),
                            key_states.to(torch.bfloat16),
                            value_states.to(torch.bfloat16),
                            attention_mask_per_sample,
                            head_num=self.num_heads,
                            scale=self.head_dim**-0.5,
                            input_layout=fusion_attention_input_layout,
                        )
                    elif use_cuda_gqa:
                        attn_output = _cuda_grouped_query_attention(
                            query_states.to(torch.bfloat16).unsqueeze(0),
                            key_states.to(torch.bfloat16).unsqueeze(0),
                            value_states.to(torch.bfloat16).unsqueeze(0),
                            attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                        ).squeeze(0)
                    else:
                        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                            attn_output = scaled_dot_product_attention(
                                query_states.to(torch.bfloat16).unsqueeze(0),
                                key_states.to(torch.bfloat16).unsqueeze(0),
                                value_states.to(torch.bfloat16).unsqueeze(0),
                                attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                            ).squeeze(0)
                    upacked_attn_output.append(attn_output)
                packed_attn_output = torch.cat(upacked_attn_output, dim=1)
        else:
            pad_size = sum(sample_lens) - packed_query_states.shape[0]
            packed_query_states_ = pad_sequence(packed_query_states_.permute(1, 0, 2), pad_size)
            packed_key_states_ = pad_sequence(packed_key_states_.permute(1, 0, 2), pad_size)
            packed_value_states = pad_sequence(packed_value_states.permute(1, 0, 2), pad_size)
            packed_attn_output = flex_attention(
                packed_query_states_.unsqueeze(0), # 1, num_head, L, head_dim
                packed_key_states_.unsqueeze(0), 
                packed_value_states.unsqueeze(0), 
                enable_gqa=True,
                block_mask=attention_mask,
            )
            end_index = packed_attn_output.shape[2] - pad_size
            packed_attn_output = packed_attn_output[0, :, :end_index, :]

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, self.hidden_size)
        packed_attn_output = self.o_proj(packed_attn_output[packed_und_token_indexes])

        return packed_attn_output


class PackedAttentionMoT(Qwen3Attention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
   
        if self.config.qk_norm:
            self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.q_norm_mot_gen = Qwen3RMSNorm(self.head_dim, eps=config.expert_config.rms_norm_eps)
            self.k_norm_mot_gen = Qwen3RMSNorm(self.head_dim, eps=config.expert_config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.q_norm_mot_gen = nn.Identity()
            self.k_norm_mot_gen = nn.Identity()
        self.enable_npu_fusion_attention = False
        self.npu_fusion_attention_input_layout = "BNSD"
        self.enable_npu_hybrid_attention_layout = False
        self.enable_npu_qkv_fusion = False
        self.enable_npu_dtype_fast_path = False
        self.enable_npu_fused_rotary = False
        self.enable_cuda_fused_rotary = False
        self.enable_cuda_gqa_attention = False
        self.use_npu_single_sample_fast_path = False
        
        # if llm and expert are the same, then use the pretrained weight, else reinitializaiton
        mot_gen_hidden_size = config.expert_config.hidden_size
        
        self.q_proj_mot_gen = nn.Linear(mot_gen_hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj_mot_gen = nn.Linear(mot_gen_hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj_mot_gen = nn.Linear(mot_gen_hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj_mot_gen = nn.Linear(self.num_heads * self.head_dim, mot_gen_hidden_size, bias=config.attention_bias)
        self._npu_fused_qkv_mot_gen_weight = None
        self._npu_fused_qkv_mot_gen_bias = None

    def _get_npu_fused_qkv_mot_gen(self):
        weight = self._npu_fused_qkv_mot_gen_weight
        expected_rows = (
            self.q_proj_mot_gen.weight.shape[0]
            + self.k_proj_mot_gen.weight.shape[0]
            + self.v_proj_mot_gen.weight.shape[0]
        )
        if (
            weight is None
            or weight.device != self.q_proj_mot_gen.weight.device
            or weight.dtype != self.q_proj_mot_gen.weight.dtype
            or weight.shape[0] != expected_rows
        ):
            weight = combined_storage_view(
                (
                    self.q_proj_mot_gen.weight,
                    self.k_proj_mot_gen.weight,
                    self.v_proj_mot_gen.weight,
                )
            )
            if weight is not None:
                weight = weight.detach()
            else:
                weight = torch.cat(
                    (
                        self.q_proj_mot_gen.weight,
                        self.k_proj_mot_gen.weight,
                        self.v_proj_mot_gen.weight,
                    ),
                    dim=0,
                ).detach()
            biases = (
                self.q_proj_mot_gen.bias,
                self.k_proj_mot_gen.bias,
                self.v_proj_mot_gen.bias,
            )
            if all(item is None for item in biases):
                bias = None
            else:
                bias = combined_storage_view(biases)  # type: ignore[arg-type]
                if bias is not None:
                    bias = bias.detach()
                else:
                    bias = torch.cat(biases, dim=0).detach()
            self._npu_fused_qkv_mot_gen_weight = weight
            self._npu_fused_qkv_mot_gen_bias = bias
        return weight, self._npu_fused_qkv_mot_gen_bias

    def _project_action_qkv(self, action_sequence):
        if not self.enable_npu_qkv_fusion:
            return (
                self.q_proj_mot_gen(action_sequence),
                self.k_proj_mot_gen(action_sequence),
                self.v_proj_mot_gen(action_sequence),
            )
        weight, bias = self._get_npu_fused_qkv_mot_gen()
        qkv = F.linear(action_sequence, weight, bias)
        q_rows = self.num_heads * self.head_dim
        kv_rows = self.num_key_value_heads * self.head_dim
        return qkv.split((q_rows, kv_rows, kv_rows), dim=-1)

    def _to_attention_bfloat16(self, tensor):
        if (
            self.enable_npu_dtype_fast_path
            and tensor.dtype == torch.bfloat16
        ):
            return tensor
        return tensor.to(torch.bfloat16)
        
    def forward(self, *args, **kwargs):
        return self.forward_train(*args, **kwargs)

    @staticmethod
    def _join_global_prefix_segments(
        packed_sequence_und: torch.Tensor,
        packed_sequence_gen: torch.Tensor,
        segments: Tuple[PrefixSegment, ...],
    ) -> torch.Tensor:
        parts = []
        for segment in segments:
            source = (
                packed_sequence_und
                if segment.branch == "und"
                else packed_sequence_gen
            )
            parts.append(source[segment.source_start : segment.source_end])
        return torch.cat(parts, dim=0)

    @staticmethod
    def _restore_prefix_branch_order(
        packed_sequence: torch.Tensor,
        segments: Tuple[PrefixSegment, ...],
    ) -> torch.Tensor:
        if not segments:
            return packed_sequence[:0]
        parts = [
            packed_sequence[segment.global_start : segment.global_end]
            for segment in segments
        ]
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=0)

    def _project_static_prefix_qkv(
        self,
        packed_sequence_und: torch.Tensor,
        packed_sequence_gen: torch.Tensor,
        prefix_segment_route: PrefixSegmentRoute,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project branch-major tokens, then restore exact global token order."""
        query_states_und = self.q_proj(packed_sequence_und).view(
            -1, self.num_heads, self.head_dim
        )
        query_states_gen = self.q_proj_mot_gen(packed_sequence_gen).view(
            -1, self.num_heads, self.head_dim
        )
        key_states_und = self.k_proj(packed_sequence_und).view(
            -1, self.num_key_value_heads, self.head_dim
        )
        key_states_gen = self.k_proj_mot_gen(packed_sequence_gen).view(
            -1, self.num_key_value_heads, self.head_dim
        )
        value_states_und = self.v_proj(packed_sequence_und).view(
            -1, self.num_key_value_heads, self.head_dim
        )
        value_states_gen = self.v_proj_mot_gen(packed_sequence_gen).view(
            -1, self.num_key_value_heads, self.head_dim
        )

        query_states_und = self.q_norm(query_states_und)
        query_states_gen = self.q_norm_mot_gen(query_states_gen)
        key_states_und = self.k_norm(key_states_und)
        key_states_gen = self.k_norm_mot_gen(key_states_gen)

        return (
            self._join_global_prefix_segments(
                query_states_und,
                query_states_gen,
                prefix_segment_route.global_segments,
            ),
            self._join_global_prefix_segments(
                key_states_und,
                key_states_gen,
                prefix_segment_route.global_segments,
            ),
            self._join_global_prefix_segments(
                value_states_und,
                value_states_gen,
                prefix_segment_route.global_segments,
            ),
        )

    def forward_train(
        self,
        packed_sequence_und: torch.Tensor,
        packed_sequence_gen: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
        return_kv_cache: bool = False,
        prefix_segment_route: Optional[PrefixSegmentRoute] = None,
    ):
        
        total_seq_len = packed_sequence_und.shape[0] + packed_sequence_gen.shape[0]
        dtype, device = packed_sequence_und.dtype, packed_sequence_und.device

        if prefix_segment_route is not None:
            (
                packed_query_states_,
                packed_key_states_,
                packed_value_states,
            ) = self._project_static_prefix_qkv(
                packed_sequence_und,
                packed_sequence_gen,
                prefix_segment_route,
            )
            packed_query_states = packed_query_states_
        else:
            packed_query_states = torch.zeros((total_seq_len, self.num_heads * self.head_dim), dtype=dtype, device=device)
            packed_key_states = torch.zeros((total_seq_len, self.num_key_value_heads * self.head_dim), dtype=dtype, device=device)
            packed_value_states = torch.zeros((total_seq_len, self.num_key_value_heads * self.head_dim), dtype=dtype, device=device)

            # 0123 6789 45 1011         <-> 0123 6789 45 1011
            packed_query_states[packed_und_token_indexes] = self.q_proj(packed_sequence_und)
            packed_query_states[packed_gen_token_indexes] = self.q_proj_mot_gen(packed_sequence_gen)

            packed_key_states[packed_und_token_indexes] = self.k_proj(packed_sequence_und)
            packed_key_states[packed_gen_token_indexes] = self.k_proj_mot_gen(packed_sequence_gen)

            packed_value_states[packed_und_token_indexes] = self.v_proj(packed_sequence_und)
            packed_value_states[packed_gen_token_indexes] = self.v_proj_mot_gen(packed_sequence_gen)

            packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
            packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)

            packed_query_states_ = packed_query_states.new_zeros(packed_query_states.shape)
            packed_key_states_ = packed_key_states.new_zeros(packed_key_states.shape)

            packed_query_states_[packed_und_token_indexes] = self.q_norm(packed_query_states[packed_und_token_indexes])
            packed_query_states_[packed_gen_token_indexes] = self.q_norm_mot_gen(packed_query_states[packed_gen_token_indexes])

            packed_key_states_[packed_und_token_indexes] = self.k_norm(packed_key_states[packed_und_token_indexes])
            packed_key_states_[packed_gen_token_indexes] = self.k_norm_mot_gen(packed_key_states[packed_gen_token_indexes])
        
        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states_, packed_key_states_ = _apply_rotary_pos_emb(
            packed_query_states_,
            packed_key_states_,
            packed_cos,
            packed_sin,
            enable_npu_fused_rotary=self.enable_npu_fused_rotary,
            enable_cuda_fused_rotary=self.enable_cuda_fused_rotary,
        )
        
        cached_key_states = None
        cached_value_states = None
        if isinstance(attention_mask, List):
            real_sample_lens = _real_sample_lens(
                sample_lens, packed_query_states_.shape[0]
            )
            use_cuda_gqa = (
                self.enable_cuda_gqa_attention and device.type == "cuda"
            )
            use_native_gqa = self.enable_npu_fusion_attention or use_cuda_gqa
            fusion_attention_input_layout = _effective_npu_attention_layout(
                self.npu_fusion_attention_input_layout,
                len(real_sample_lens),
                self.enable_npu_hybrid_attention_layout,
            )
            if use_native_gqa:
                cached_key_states = packed_key_states_
                cached_value_states = packed_value_states
            else:
                # Preserve the OPT-01 cache shape and dense attention ordering:
                # [S, Nkv, D] -> [S, Nq, D] before transpose/split.
                packed_key_states_ = packed_key_states_[:, :, None, :].repeat(
                    1, 1, self.num_key_value_groups, 1
                ).reshape(-1, self.num_heads, self.head_dim)
                packed_value_states = packed_value_states[
                    :, :, None, :
                ].repeat(
                    1, 1, self.num_key_value_groups, 1
                ).reshape(-1, self.num_heads, self.head_dim)
                cached_key_states = packed_key_states_
                cached_value_states = packed_value_states
            if self.use_npu_single_sample_fast_path:
                if len(real_sample_lens) != 1:
                    raise RuntimeError(
                        "OPT-03 fast path reached attention with "
                        f"{len(real_sample_lens)} real samples"
                    )
                query_states = packed_query_states_.transpose(0, 1)
                key_states = packed_key_states_.transpose(0, 1)
                value_states = packed_value_states.transpose(0, 1)
                attention_mask_per_sample = attention_mask[0]
                if self.enable_npu_fusion_attention:
                    packed_attn_output = _npu_fusion_attention(
                        query_states.to(torch.bfloat16),
                        key_states.to(torch.bfloat16),
                        value_states.to(torch.bfloat16),
                        attention_mask_per_sample,
                        head_num=self.num_heads,
                        scale=self.head_dim**-0.5,
                        input_layout=fusion_attention_input_layout,
                    )
                elif use_cuda_gqa:
                    packed_attn_output = _cuda_grouped_query_attention(
                        query_states.to(torch.bfloat16).unsqueeze(0),
                        key_states.to(torch.bfloat16).unsqueeze(0),
                        value_states.to(torch.bfloat16).unsqueeze(0),
                        attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                    ).squeeze(0)
                else:
                    with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                        packed_attn_output = scaled_dot_product_attention(
                            query_states.to(torch.bfloat16).unsqueeze(0),
                            key_states.to(torch.bfloat16).unsqueeze(0),
                            value_states.to(torch.bfloat16).unsqueeze(0),
                            attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                        ).squeeze(0)
            else:
                unpacked_query_states = packed_query_states_.transpose(0, 1).split(real_sample_lens, dim=1)
                unpacked_key_states = packed_key_states_.transpose(0, 1).split(real_sample_lens, dim=1)
                unpacked_value_states = packed_value_states.transpose(0, 1).split(real_sample_lens, dim=1)
                upacked_attn_output = []
                for query_states, key_states, value_states, attention_mask_per_sample in zip(
                    unpacked_query_states, unpacked_key_states, unpacked_value_states, attention_mask
                ):
                    if self.enable_npu_fusion_attention:
                        attn_output = _npu_fusion_attention(
                            query_states.to(torch.bfloat16),
                            key_states.to(torch.bfloat16),
                            value_states.to(torch.bfloat16),
                            attention_mask_per_sample,
                            head_num=self.num_heads,
                            scale=self.head_dim**-0.5,
                            input_layout=fusion_attention_input_layout,
                        )
                    elif use_cuda_gqa:
                        attn_output = _cuda_grouped_query_attention(
                            query_states.to(torch.bfloat16).unsqueeze(0),
                            key_states.to(torch.bfloat16).unsqueeze(0),
                            value_states.to(torch.bfloat16).unsqueeze(0),
                            attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                        ).squeeze(0)
                    else:
                        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                            attn_output = scaled_dot_product_attention(
                                query_states.to(torch.bfloat16).unsqueeze(0),
                                key_states.to(torch.bfloat16).unsqueeze(0),
                                value_states.to(torch.bfloat16).unsqueeze(0),
                                attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                            ).squeeze(0)
                    upacked_attn_output.append(attn_output)
                packed_attn_output = torch.cat(upacked_attn_output, dim=1)
        else:
            pad_size = sum(sample_lens) - packed_query_states.shape[0]
            packed_query_states_ = pad_sequence(packed_query_states_.permute(1, 0, 2), pad_size)
            packed_key_states_ = pad_sequence(packed_key_states_.permute(1, 0, 2), pad_size)
            packed_value_states = pad_sequence(packed_value_states.permute(1, 0, 2), pad_size)
            packed_attn_output = flex_attention(
                packed_query_states_.unsqueeze(0), # 1, num_head, L, head_dim
                packed_key_states_.unsqueeze(0),
                packed_value_states.unsqueeze(0), 
                enable_gqa=True,
                block_mask=attention_mask,
            )
            #breakpoint()
            end_index = packed_attn_output.shape[2] - pad_size
            packed_attn_output = packed_attn_output[0, :, :end_index, :]

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, self.num_heads * self.head_dim)
        if prefix_segment_route is not None:
            packed_attn_output_und = self.o_proj(
                self._restore_prefix_branch_order(
                    packed_attn_output,
                    prefix_segment_route.und_segments,
                )
            )
            packed_attn_output_gen = self.o_proj_mot_gen(
                self._restore_prefix_branch_order(
                    packed_attn_output,
                    prefix_segment_route.gen_segments,
                )
            )
        else:
            packed_attn_output_und = self.o_proj(packed_attn_output[packed_und_token_indexes])
            packed_attn_output_gen = self.o_proj_mot_gen(packed_attn_output[packed_gen_token_indexes])
     
        if return_kv_cache:
            if cached_key_states is None or cached_value_states is None:
                raise RuntimeError(
                    "Prefix KV caching requires the dense accelerator attention path"
                )
            return (
                packed_attn_output_und,
                packed_attn_output_gen,
                (cached_key_states, cached_value_states),
            )
        return packed_attn_output_und, packed_attn_output_gen

    def forward_action_with_prefix_cache(
        self,
        action_sequence: torch.Tensor,
        prefix_key_states: torch.Tensor,
        prefix_value_states: torch.Tensor,
        action_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        key_workspace: Optional[torch.Tensor] = None,
        value_workspace: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run causal attention only for action tokens using a static prefix KV."""
        query_states, key_states, value_states = self._project_action_qkv(
            action_sequence
        )

        query_states = query_states.view(-1, self.num_heads, self.head_dim)
        key_states = key_states.view(-1, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(
            -1, self.num_key_value_heads, self.head_dim
        )

        query_states = self.q_norm_mot_gen(query_states)
        key_states = self.k_norm_mot_gen(key_states)
        cos, sin = action_position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            enable_npu_fused_rotary=self.enable_npu_fused_rotary,
            enable_cuda_fused_rotary=self.enable_cuda_fused_rotary,
        )

        use_cuda_gqa = (
            self.enable_cuda_gqa_attention
            and action_sequence.device.type == "cuda"
        )
        if self.enable_npu_fusion_attention or use_cuda_gqa:
            if (key_workspace is None) != (value_workspace is None):
                raise ValueError("key/value workspaces must be provided together")
            if key_workspace is None:
                all_key_states = torch.cat(
                    [prefix_key_states, key_states], dim=0
                )
                all_value_states = torch.cat(
                    [prefix_value_states, value_states], dim=0
                )
            else:
                prefix_length = prefix_key_states.shape[0]
                expected_key_shape = (
                    prefix_length + key_states.shape[0],
                    *key_states.shape[1:],
                )
                expected_value_shape = (
                    prefix_value_states.shape[0] + value_states.shape[0],
                    *value_states.shape[1:],
                )
                if tuple(key_workspace.shape) != expected_key_shape:
                    raise ValueError(
                        "key workspace shape mismatch: "
                        f"expected {expected_key_shape}, got "
                        f"{tuple(key_workspace.shape)}"
                    )
                if tuple(value_workspace.shape) != expected_value_shape:
                    raise ValueError(
                        "value workspace shape mismatch: "
                        f"expected {expected_value_shape}, got "
                        f"{tuple(value_workspace.shape)}"
                    )
                key_workspace[prefix_length:].copy_(key_states)
                value_workspace[prefix_length:].copy_(value_states)
                all_key_states = key_workspace
                all_value_states = value_workspace
            if self.enable_npu_fusion_attention:
                attention_output = _npu_fusion_attention(
                    self._to_attention_bfloat16(
                        query_states.transpose(0, 1)
                    ),
                    self._to_attention_bfloat16(
                        all_key_states.transpose(0, 1)
                    ),
                    self._to_attention_bfloat16(
                        all_value_states.transpose(0, 1)
                    ),
                    attention_mask,
                    head_num=self.num_heads,
                    scale=self.head_dim**-0.5,
                    # Prefix-cache action suffix is constructed only by the
                    # validated single-sample route.  Packed multi-sample requests
                    # use forward_train above and are forced to BNSD there.
                    input_layout=self.npu_fusion_attention_input_layout,
                )
            else:
                attention_output = _cuda_grouped_query_attention(
                    query_states.transpose(0, 1)
                    .unsqueeze(0)
                    .to(torch.bfloat16),
                    all_key_states.transpose(0, 1)
                    .unsqueeze(0)
                    .to(torch.bfloat16),
                    all_value_states.transpose(0, 1)
                    .unsqueeze(0)
                    .to(torch.bfloat16),
                    attention_mask.unsqueeze(0).unsqueeze(0),
                ).squeeze(0)
        else:
            key_states = key_states[:, :, None, :].repeat(
                1, 1, self.num_key_value_groups, 1
            ).reshape(-1, self.num_heads, self.head_dim)
            value_states = value_states[:, :, None, :].repeat(
                1, 1, self.num_key_value_groups, 1
            ).reshape(-1, self.num_heads, self.head_dim)
            all_key_states = torch.cat(
                [prefix_key_states, key_states], dim=0
            )
            all_value_states = torch.cat(
                [prefix_value_states, value_states], dim=0
            )
            with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                attention_output = scaled_dot_product_attention(
                    query_states.transpose(0, 1).unsqueeze(0).to(torch.bfloat16),
                    all_key_states.transpose(0, 1).unsqueeze(0).to(torch.bfloat16),
                    all_value_states.transpose(0, 1)
                    .unsqueeze(0)
                    .to(torch.bfloat16),
                    attention_mask.unsqueeze(0).unsqueeze(0),
                ).squeeze(0)
        attention_output = attention_output.transpose(0, 1).reshape(
            -1, self.num_heads * self.head_dim
        )
        return self.o_proj_mot_gen(attention_output)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = PackedAttention(config=config, layer_idx=layer_idx)
        
        self.mlp = Qwen3MLP(config)

        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, *args, **kwargs):
        return self.forward_train(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence_und: torch.Tensor,
        packed_sequence_gen: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
    ) -> torch.Tensor:
        # [seq_len, dim]
        residual = packed_sequence_und
        packed_sequence = self.input_layernorm(packed_sequence_und)

        # Self Attention
        packed_sequence = self.self_attn(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
            packed_und_token_indexes=packed_und_token_indexes
        )

        packed_sequence = residual + packed_sequence

        # Fully Connected
        residual = packed_sequence
        packed_sequence = self.post_attention_layernorm(packed_sequence)
        packed_sequence = self.mlp(packed_sequence)
        packed_sequence = residual + packed_sequence

        return packed_sequence, packed_sequence_gen
    

class Qwen3MoTDecoderLayer(nn.Module):
    def __init__(
        self, 
        config, 
        # gen_config
        layer_idx: Optional[int] = None, 
        attn_module: Optional[Qwen3Attention] = PackedAttentionMoT,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = attn_module(config, layer_idx)

        self.mlp = Qwen3MLP(config)
        self.mlp_mot_gen = Qwen3MLP(config.expert_config)

        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_mot_gen = Qwen3RMSNorm(config.expert_config.hidden_size, eps=config.expert_config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_mot_gen = Qwen3RMSNorm(config.expert_config.hidden_size, eps=config.expert_config.rms_norm_eps)
        self.enable_npu_add_rms_norm = False

    def forward(self, *args, **kwargs):
        return self.forward_train(*args, **kwargs)
    
    def forward_train(
        self,
        packed_sequence_und: torch.Tensor,
        packed_sequence_gen: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
        return_kv_cache: bool = False,
        prefix_segment_route: Optional[PrefixSegmentRoute] = None,
    ) -> torch.Tensor:
        
        residual_und = packed_sequence_und
        residual_gen = packed_sequence_gen
    
        packed_sequence_und_ = self.input_layernorm(packed_sequence_und)
        packed_sequence_gen_ = self.input_layernorm_mot_gen(packed_sequence_gen)

        # Self Attention
        attention_outputs = self.self_attn(
            packed_sequence_und=packed_sequence_und_,
            packed_sequence_gen=packed_sequence_gen_,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
            return_kv_cache=return_kv_cache,
            prefix_segment_route=prefix_segment_route,
        )
        if return_kv_cache:
            (
                packed_sequence_und_,
                packed_sequence_gen_,
                kv_cache,
            ) = attention_outputs
        else:
            packed_sequence_und_, packed_sequence_gen_ = attention_outputs

        packed_sequence_und = residual_und + packed_sequence_und_
        packed_sequence_gen = residual_gen + packed_sequence_gen_
        
        # Fully Connected
        residual_und = packed_sequence_und
        residual_gen = packed_sequence_gen

        packed_sequence_und_ = packed_sequence_und.new_zeros(packed_sequence_und.shape)
        packed_sequence_gen_ = packed_sequence_gen.new_zeros(packed_sequence_gen.shape)

        packed_sequence_und_ = self.mlp(self.post_attention_layernorm(packed_sequence_und))
        packed_sequence_gen_ = self.mlp_mot_gen(
            self.post_attention_layernorm_mot_gen(packed_sequence_gen)
        )
     
        packed_sequence_und = residual_und + packed_sequence_und_
        packed_sequence_gen = residual_gen + packed_sequence_gen_

        if return_kv_cache:
            return packed_sequence_und, packed_sequence_gen, kv_cache
        return packed_sequence_und, packed_sequence_gen

    def forward_action_with_prefix_cache(
        self,
        action_sequence: torch.Tensor,
        prefix_key_states: torch.Tensor,
        prefix_value_states: torch.Tensor,
        action_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        key_workspace: Optional[torch.Tensor] = None,
        value_workspace: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = action_sequence
        normalized_action = self.input_layernorm_mot_gen(action_sequence)
        attention_delta = self.self_attn.forward_action_with_prefix_cache(
            action_sequence=normalized_action,
            prefix_key_states=prefix_key_states,
            prefix_value_states=prefix_value_states,
            action_position_embeddings=action_position_embeddings,
            attention_mask=attention_mask,
            key_workspace=key_workspace,
            value_workspace=value_workspace,
        )
        if (
            self.enable_npu_add_rms_norm
            and attention_delta.device.type == "npu"
        ):
            if torch_npu is None:
                raise RuntimeError("NPU add RMSNorm requires torch-npu")
            normalized_action, _, action_sequence = (
                torch_npu.npu_add_rms_norm(
                    attention_delta,
                    residual,
                    self.post_attention_layernorm_mot_gen.weight,
                    self.post_attention_layernorm_mot_gen.variance_epsilon,
                )
            )
        else:
            action_sequence = residual + attention_delta
            normalized_action = self.post_attention_layernorm_mot_gen(
                action_sequence
            )
        mlp_delta = self.mlp_mot_gen(normalized_action)
        return action_sequence + mlp_delta

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:

        residual = packed_query_sequence
        if mode == "und":
            packed_query_sequence = self.input_layernorm(packed_query_sequence)
        elif mode == "gen":
            packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
            packed_query_sequence_[packed_text_indexes] = self.input_layernorm(packed_query_sequence[packed_text_indexes])
            packed_query_sequence_[packed_vae_token_indexes] = self.input_layernorm_mot_gen(packed_query_sequence[packed_vae_token_indexes])
            packed_query_sequence = packed_query_sequence_

        # Self Attention
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )
        packed_query_sequence = residual + packed_query_sequence

        # Fully Connected
        residual = packed_query_sequence
        if mode == "und":
            packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
            packed_query_sequence = self.mlp(packed_query_sequence)
        elif mode == "gen":
            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]
            packed_text_query_sequence = self.post_attention_layernorm(packed_text_query_sequence).to(torch.bfloat16)
            packed_vae_query_sequence = self.post_attention_layernorm_mot_gen(packed_vae_query_sequence).to(torch.bfloat16)

            packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(torch.bfloat16)
            packed_query_sequence_[packed_text_indexes] = self.mlp(packed_text_query_sequence)
            packed_query_sequence_[packed_vae_token_indexes] = self.mlp_mot_gen(packed_vae_query_sequence)
            packed_query_sequence = packed_query_sequence_

        packed_query_sequence = residual + packed_query_sequence
        return packed_query_sequence, past_key_values
    

Decoder_layer_dict = {
    "Qwen3DecoderLayer": Qwen3DecoderLayer,
    "Qwen3MoTDecoderLayer": partial(Qwen3MoTDecoderLayer, attn_module=PackedAttentionMoT),
}

class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.use_mot = config.use_mot
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        layer_module = Decoder_layer_dict[config.layer_module]

        self.layers = nn.ModuleList(
            [layer_module(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
       
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.use_mot:
            self.norm_mot_gen = Qwen3RMSNorm(config.expert_config.hidden_size, eps=config.expert_config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        self.post_init()

    def set_npu_fusion_attention(
        self,
        enabled: bool,
        input_layout: str = "BNSD",
        hybrid_prefix_bnsd: bool = False,
    ) -> None:
        """Toggle OPT-02 on every packed attention layer."""
        if input_layout not in {"BNSD", "BSND"}:
            raise ValueError(
                f"Unsupported NPU fusion-attention layout: {input_layout}"
            )
        for decoder_layer in self.layers:
            decoder_layer.self_attn.enable_npu_fusion_attention = enabled
            decoder_layer.self_attn.npu_fusion_attention_input_layout = (
                input_layout
            )
            decoder_layer.self_attn.enable_npu_hybrid_attention_layout = (
                hybrid_prefix_bnsd
            )

    def set_cuda_gqa_attention(self, enabled: bool) -> None:
        """Use CUDA SDPA's native GQA path without materializing K/V heads."""
        for decoder_layer in self.layers:
            decoder_layer.self_attn.enable_cuda_gqa_attention = enabled

    def set_npu_projection_fusion(self, enabled: bool) -> None:
        """Toggle OPT-06 fused projections and native NPU RMSNorm."""
        for decoder_layer in self.layers:
            decoder_layer.self_attn.enable_npu_qkv_fusion = enabled
            decoder_layer.mlp.enable_npu_gate_up_fusion = enabled
            if hasattr(decoder_layer, "mlp_mot_gen"):
                decoder_layer.mlp_mot_gen.enable_npu_gate_up_fusion = enabled
            for name in (
                "input_layernorm",
                "post_attention_layernorm",
                "input_layernorm_mot_gen",
                "post_attention_layernorm_mot_gen",
            ):
                norm = getattr(decoder_layer, name, None)
                if norm is not None:
                    norm.enable_npu_fused_rms_norm = enabled
            for name in (
                "q_norm",
                "k_norm",
                "q_norm_mot_gen",
                "k_norm_mot_gen",
            ):
                norm = getattr(decoder_layer.self_attn, name, None)
                if isinstance(norm, Qwen3RMSNorm):
                    norm.enable_npu_fused_rms_norm = enabled
        self.norm.enable_npu_fused_rms_norm = enabled
        if self.use_mot:
            self.norm_mot_gen.enable_npu_fused_rms_norm = enabled

    def set_npu_dtype_fast_path(self, enabled: bool) -> None:
        """Toggle OPT-09 proven no-op BF16 cast elimination."""
        for decoder_layer in self.layers:
            decoder_layer.self_attn.enable_npu_dtype_fast_path = enabled

    def set_npu_fused_rotary(self, enabled: bool) -> None:
        """Use the native NPU half-rotation kernel for query/key RoPE."""
        for decoder_layer in self.layers:
            decoder_layer.self_attn.enable_npu_fused_rotary = enabled

    def set_cuda_fused_rotary(self, enabled: bool) -> None:
        """Use the CUDA/Triton fused half-rotation kernel for query/key RoPE."""
        for decoder_layer in self.layers:
            decoder_layer.self_attn.enable_cuda_fused_rotary = enabled

    def set_npu_fused_swiglu(self, enabled: bool) -> None:
        """Fuse SwiGLU activation/multiply after the packed gate/up GEMM."""
        for decoder_layer in self.layers:
            decoder_layer.mlp.enable_npu_fused_swiglu = enabled
            if hasattr(decoder_layer, "mlp_mot_gen"):
                decoder_layer.mlp_mot_gen.enable_npu_fused_swiglu = enabled

    def set_cuda_fused_swiglu(self, enabled: bool) -> None:
        """Use the CUDA/Triton fused SwiGLU kernel after gate/up projection."""
        for decoder_layer in self.layers:
            decoder_layer.mlp.enable_cuda_fused_swiglu = enabled
            if hasattr(decoder_layer, "mlp_mot_gen"):
                decoder_layer.mlp_mot_gen.enable_cuda_fused_swiglu = enabled

    def set_npu_add_rms_norm(self, enabled: bool) -> None:
        """Fuse the action-suffix attention residual add and RMSNorm."""
        for decoder_layer in self.layers:
            if isinstance(decoder_layer, Qwen3MoTDecoderLayer):
                decoder_layer.enable_npu_add_rms_norm = enabled

    def set_npu_single_sample_fast_path(self, enabled: bool) -> None:
        """Route packed attention layers through the OPT-03 fast path."""
        for decoder_layer in self.layers:
            decoder_layer.self_attn.use_npu_single_sample_fast_path = enabled

    def forward(self, *args, **kwargs):
        return self.forward_train(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence_und: torch.Tensor,
        packed_sequence_gen: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:

        # create position embeddings to be shared across the decoder layers
        cos, sin = self.rotary_emb(packed_sequence_und, packed_position_ids.unsqueeze(0))
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_position_embeddings = (cos, sin)

        extra_inputs = {}
        if self.use_mot:
            assert packed_und_token_indexes is not None
            if packed_gen_token_indexes is None:
                packed_gen_token_indexes = packed_und_token_indexes.new_ones(size=[0])
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )
        else:
            extra_inputs.update(packed_und_token_indexes=packed_und_token_indexes)

        for decoder_layer in self.layers:
            packed_sequence_und, packed_sequence_gen = decoder_layer(
                packed_sequence_und=packed_sequence_und,
                packed_sequence_gen=packed_sequence_gen,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_embeddings=packed_position_embeddings,
                **extra_inputs
            )

        if self.use_mot:
            # packed_sequence_ = torch.zeros((packed_sequence_und.shape[0]+packed_sequence_gen.shape[0], packed_sequence_und.shape[1]), dtype=dtype, device=device)
            packed_sequence_und_ = torch.zeros_like(packed_sequence_und)
            packed_sequence_gen_ = torch.zeros_like(packed_sequence_gen)
            # packed_sequence_[packed_und_token_indexes] = self.norm(packed_sequence_und)
            packed_sequence_und_ = self.norm(packed_sequence_und)
            packed_sequence_gen_ = self.norm_mot_gen(packed_sequence_gen)
            return packed_sequence_und_, packed_sequence_gen_
        else:
            assert packed_sequence_gen.shape[0]==0
            return self.norm(packed_sequence_und), packed_sequence_gen

    def build_static_prefix_cache(
        self,
        packed_sequence_und: torch.Tensor,
        packed_sequence_gen: torch.Tensor,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
        attention_mask: torch.Tensor,
        prefix_segment_route: Optional[PrefixSegmentRoute] = None,
    ) -> Dict[str, Any]:
        """Prefill the invariant causal prefix and retain per-layer expanded KV."""
        if not self.use_mot:
            raise RuntimeError("Static prefix caching requires the MoT model")

        cos, sin = self.rotary_emb(
            packed_sequence_und, packed_position_ids.unsqueeze(0)
        )
        position_embeddings = (cos.squeeze(0), sin.squeeze(0))
        layer_caches = []
        prefix_len = len(packed_position_ids)
        sample_lens = [prefix_len]
        if (
            prefix_segment_route is not None
            and not prefix_segment_route.matches(
                prefix_length=prefix_len,
                und_length=packed_sequence_und.shape[0],
                gen_length=packed_sequence_gen.shape[0],
            )
        ):
            prefix_segment_route = None

        for decoder_layer in self.layers:
            if not isinstance(decoder_layer, Qwen3MoTDecoderLayer):
                raise RuntimeError(
                    "Static prefix caching requires Qwen3MoTDecoderLayer"
                )
            (
                packed_sequence_und,
                packed_sequence_gen,
                layer_cache,
            ) = decoder_layer(
                packed_sequence_und=packed_sequence_und,
                packed_sequence_gen=packed_sequence_gen,
                sample_lens=sample_lens,
                attention_mask=[attention_mask],
                packed_position_embeddings=position_embeddings,
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
                return_kv_cache=True,
                prefix_segment_route=prefix_segment_route,
            )
            layer_caches.append(layer_cache)

        return {
            "layers": layer_caches,
            "prefix_length": prefix_len,
        }

    def forward_action_with_prefix_cache(
        self,
        action_sequence: torch.Tensor,
        packed_position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_cache: Dict[str, Any],
    ) -> torch.Tensor:
        """Run the dynamic action suffix against a previously prefetched prefix."""
        if len(prefix_cache["layers"]) != len(self.layers):
            raise RuntimeError("Prefix cache layer count does not match model")

        cos, sin = self.rotary_emb(
            action_sequence, packed_position_ids.unsqueeze(0)
        )
        action_position_embeddings = (cos.squeeze(0), sin.squeeze(0))

        full_layers = prefix_cache.get("full_layers")
        if full_layers is not None and len(full_layers) != len(self.layers):
            raise RuntimeError("Full KV workspace layer count does not match model")

        for layer_index, (
            decoder_layer,
            (prefix_key_states, prefix_value_states),
        ) in enumerate(zip(self.layers, prefix_cache["layers"])):
            key_workspace = None
            value_workspace = None
            if full_layers is not None:
                key_workspace, value_workspace = full_layers[layer_index]
            action_sequence = (
                decoder_layer.forward_action_with_prefix_cache(
                    action_sequence=action_sequence,
                    prefix_key_states=prefix_key_states,
                    prefix_value_states=prefix_value_states,
                    action_position_embeddings=action_position_embeddings,
                    attention_mask=attention_mask,
                    key_workspace=key_workspace,
                    value_workspace=value_workspace,
                )
            )
        return self.norm_mot_gen(action_sequence)


class Qwen3ForCausalLM(Qwen3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.llm_layers = config.num_hidden_layers
        self.use_mot = config.use_mot
        if self.use_mot:
            self.expert_layers = config.expert_config.num_hidden_layers
            self.llm_identical = config.hidden_size==config.expert_config.hidden_size

        # Initialize weights and apply final processing
        self.post_init()

    def custom_init_pretrained(self, mllm_state_dict, logger):
        logger.info("Initializing LLM decoder from MLLM weights") 
        for name, param in self.named_parameters():
            if "mot_gen" in name:
                continue
            
            original_name = "language_model."+name.replace("_und", "") if "norm_und" in name \
                            else "language_model."+name
            try:
                param.data.copy_(mllm_state_dict.pop(original_name).data)
            except:
                logger.info(f"'{name}' not initialized")
        return mllm_state_dict

    def init_pretrained(self, mllm_state_dict, enable=True):
        if not enable:
            print("Initializing LLM decoder randomly")
            return mllm_state_dict
        
        print("Initializing LLM decoder from MLLM weights") 
        for name, param in self.named_parameters():
            if "mot_gen" in name:
                continue
            
            original_name = "language_model."+name.replace("_und", "") if "norm_und" in name \
                            else "language_model."+name
            try:
                param.data.copy_(mllm_state_dict.pop(original_name).data)
            except:
                print(f"'{name}' not initialized")
        return mllm_state_dict
    
    def init_expert(self, expert_path, from_scratch=True):
        """initialize parameters of action expert using the pretrained llm ckpt"""

        if from_scratch:
            print("INFO: `from_scratch=True`. Expert (`_mot_gen`) parameters will be randomly initialized.")
            return
        assert 1==0
        safetensor_files = glob.glob(f"{expert_path}/*.safetensors")
        expert_state_dict = dict()
        for file_path in safetensor_files:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    expert_state_dict[key] = f.get_tensor(key)

        layer_mapping = get_layer_mapping_strategy(self.llm_layers, self.expert_layers)
        
        for name, param in self.named_parameters():
            if "mot_gen" not in name:
                continue

            original_name = name.replace("_mot_gen", "")  
            
            if any(n in name for n in ["q_norm_mot_gen", "k_norm_mot_gen"]):      
                param.data.copy_(self.state_dict()[original_name].data)    
            elif any(proj in name for proj in ["q_proj_mot_gen", "k_proj_mot_gen", "v_proj_mot_gen", "o_proj_mot_gen"]):
                if self.llm_identical and not expert_path:
                    param.data.copy_(expert_state_dict[original_name].data)
            else:
                layer_match = re.search(r'layers\.(\d+)\.', name)
                if layer_match:
                    curr_layer_idx = int(layer_match.group(1))
                    if curr_layer_idx < len(layer_mapping):
                        expert_pos = layer_mapping[curr_layer_idx]
                        original_name_template = name.replace("_mot_gen", "").replace(f"layers.{curr_layer_idx}.", "layers.{}.") # Build parameter name template
                 
                        interpolated_param = interpolate_layer_params(
                            expert_state_dict, expert_pos, original_name_template,
                            num_expert_layers=self.expert_layers
                        )

                        if interpolated_param is not None:
                            param.data.copy_(interpolated_param)
                        else:
                            fallback_idx = min(int(expert_pos), self.expert_layers-1)
                            fallback_name = original_name_template.format(fallback_idx)
                            if fallback_name in expert_state_dict:
                                param.data.copy_(expert_state_dict[fallback_name])
                                print(f"Layer {curr_layer_idx}: fallback to expert layer {fallback_idx}")
                else:
                    if original_name in expert_state_dict:
                        param.data.copy_(expert_state_dict[original_name].data)
    
    def init_mot(self):
        """initialize the action expert by direct param copy, requiring llm and expert sharing the same arch"""
        for name, param in self.named_parameters():
            if "mot_gen" in name:
                original_name = name.replace("_mot_gen", "")
                param.data.copy_(self.state_dict()[original_name].data)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(self, *args, **kwargs):
        return self.forward_train(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence_und: torch.Tensor,
        packed_sequence_gen: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:

        outputs = self.model(
            packed_sequence_und=packed_sequence_und,
            packed_sequence_gen=packed_sequence_gen,
            sample_lens=sample_lens,
            packed_position_ids=packed_position_ids,
            attention_mask=attention_mask,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        return outputs

    def build_static_prefix_cache(self, **kwargs):
        return self.model.build_static_prefix_cache(**kwargs)

    def forward_action_with_prefix_cache(self, **kwargs):
        return self.model.forward_action_with_prefix_cache(**kwargs)
