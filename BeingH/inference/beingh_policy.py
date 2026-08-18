# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import glob
import json
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
 
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoModel, AutoTokenizer

from configs.data_config import DATA_CONFIG_MAP, ModalityConfig
from BeingH.dataset.preprocess import build_vit_transform_base
from BeingH.model import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM
from BeingH.model.beingvla import BeingH, BeingHConfig
from BeingH.model.layers import InternVLConnector
from BeingH.model.llm.qwen2 import Qwen2Tokenizer
from BeingH.model.vit_model.internvit_navit import InternVisionConfig, InternVisionModel
from BeingH.npu_single_sample_fast_path import (
    VALID_NPU_SINGLE_SAMPLE_FAST_PATH_MODES,
)
from BeingH.npu_capture_replay import NPUCaptureProcessUnhealthyError
from BeingH.npu_prefix_segment_route import build_prefix_segment_route
from BeingH.fused_projection_storage import prepare_fused_projection_storage
from BeingH.utils.schema import DatasetMetadata, EmbodimentTag
 
# Register custom config
AutoConfig.register("beingh", BeingHConfig)
AutoModel.register(BeingHConfig, BeingH)


# ==============================================================================
# Constants
# ==============================================================================
 
BLOCK_SIZE = 128
 
# Model version configurations
VERSION_CONFIGS = {
    "qwen2.5": (Qwen2Config, Qwen2ForCausalLM, Qwen2Tokenizer),
    "qwen3": (Qwen3Config, Qwen3ForCausalLM, AutoTokenizer),
}


def load_safetensors(path):
    safetensor_files = glob.glob(f"{path}/*.safetensors")
    # if safetensor_files:
    print(f"Found {len(safetensor_files)} .safetensors files, loading...")
    state_dict = {}
    for file_path in safetensor_files:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    print("Weights loaded from .safetensors files.")
    return state_dict


def configure_npu_linear_weight_prelayout(enabled: bool) -> None:
    """Configure the process-global torch-npu MatMul format policy for OPT-16."""
    if not enabled:
        return
    import torch_npu

    torch.npu.config.allow_internal_format = True
    torch.npu.set_mm_bmm_format_nd(False)


def prelayout_npu_linear_weights(model: torch.nn.Module) -> int:
    """Convert unique 2-D ``nn.Linear`` weights to FRACTAL_NZ in place."""
    import torch_npu

    converted = 0
    seen_parameters: set[int] = set()
    target_format = int(torch_npu.Format.FRACTAL_NZ)
    for module in model.modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        weight = module.weight
        if id(weight) in seen_parameters:
            continue
        seen_parameters.add(id(weight))
        if weight.device.type != "npu" or weight.ndim != 2:
            continue
        if int(torch_npu.get_npu_format(weight)) == target_format:
            continue
        weight.data = torch_npu.npu_format_cast(weight.data, target_format)
        converted += 1
    return converted


# Helper functions
def unsqueeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unsqueeze the values of a dictionary.
    This converts the data to be batched of size 1.
    """
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, list):
            unsqueezed_data[k] = np.array(v)
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data


def squeeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Squeeze the values of a dictionary. This removes the batch dimension (axis=0).
    NOTE: We use axis=0 explicitly to avoid removing other size-1 dimensions
    (e.g., gripper_position with shape (1, 16, 1) should become (16, 1), not (16,))
    """
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v, axis=0)  # Only remove batch dimension
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze(0)  # Only remove batch dimension
        else:
            squeezed_data[k] = v
    return squeezed_data


# ==============================================================================
# BeingH Policy
# ==============================================================================
class BasePolicy(ABC):
    """Abstract base class for robot control policies."""

    @abstractmethod
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method to get the action for a given state.

        Args:
            observations: The observations from the environment.

        Returns:
            The action to take in the environment in dictionary format.
        """
        raise NotImplementedError

    @abstractmethod
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Return the modality config of the policy.
        """
        raise NotImplementedError
    

class BeingHPolicy(BasePolicy):
    """
    Inference wrapper for trained Being-H VLA model.
    
    Features:
    - Multi-view image processing
    - State and action normalization
    - RTC (Real-Time Chunking) support
    - MPG parameter overrides
    - Hierarchical metadata variant selection
    """
    def __init__(
        self,
        model_path: str,
        data_config_name: str,  # Data configuration name
        dataset_name: str,
        embodiment_tag: str,
        instruction_template: str,
        prop_pos: str = "front",
        max_view_num: int = -1,
        use_fixed_view: bool = False,
        action_attn_mode: str = "causal",
        device: Union[int, str] = "cuda" if torch.cuda.is_available() else "cpu",
        # MPG overrides
        use_mpg: bool = None,
        mpg_lambda: float = None,
        mpg_num_projections: int = None,
        mpg_refinement_iters: int = None,
        mpg_gate_temperature: float = None,
        # Flow matching parameter override
        num_inference_timesteps: int = None,
        # Optional deployment safety clipping in normalized action space
        clip_normalized_action_keys: Optional[List[str]] = None,
        # RTC parameter
        enable_rtc: bool = True,
        # Lossless inference optimization flags
        enable_static_prefix_cache: bool = False,
        enable_npu_fusion_attention: bool = False,
        enable_cuda_gqa_attention: bool = False,
        enable_npu_fusion_attention_bsnd: bool = False,
        enable_npu_hybrid_attention_layout: bool = False,
        enable_npu_prefix_segment_route: bool = False,
        enable_npu_projection_fusion: bool = False,
        enable_npu_vectorized_mpg: bool = False,
        enable_npu_workspace_reuse: bool = False,
        enable_npu_kv_workspace: bool = False,
        enable_cuda_kv_workspace: bool = False,
        enable_npu_add_rms_norm: bool = False,
        enable_npu_fused_rotary: bool = False,
        enable_npu_fused_swiglu: bool = False,
        enable_cuda_fused_rotary: bool = False,
        enable_cuda_fused_swiglu: bool = False,
        enable_cuda_fused_only_projection_storage: bool = False,
        enable_fused_only_projection_storage: bool = False,
        enable_npu_static_tensor_cache: bool = False,
        enable_npu_dtype_fast_path: bool = False,
        enable_npu_euler_buffer_cache: bool = False,
        enable_npu_vision_state_overlap: bool = False,
        enable_npu_action_compile: bool = False,
        enable_npu_vision_compile: bool = False,
        enable_npu_linear_weight_prelayout: bool = False,
        enable_npu_persistent_compile_cache: bool = False,
        npu_compile_cache_dir: Optional[str] = None,
        enable_adaptive_flow_steps: bool = False,
        adaptive_flow_min_steps: int = 2,
        adaptive_flow_velocity_threshold: float = 0.0,
        enable_adaptive_mpg_refinement: bool = False,
        adaptive_mpg_gate_threshold: float = 0.0,
        enable_policy_prompt_cache: bool = False,
        npu_single_sample_fast_path: str = "off",
        enable_npu_capture_replay: bool = False,
        enable_npu_prefix_graph_replay: bool = False,
        npu_graph_cache_max_entries: int = 1,
        enable_cuda_capture_replay: bool = False,
        cuda_graph_cache_max_entries: Optional[int] = None,
        enable_npu_baseline_flow_graph_replay: bool = False,
        enable_npu_adaptive_flow_graph_replay: bool = False,
        # Metadata variant selection
        metadata_variant: str = None,
        stats_selection_mode: str = "auto",
    ):
        """
        Initialize BeingH policy.

        Args:
            model_path: Path to trained model checkpoint (self-contained directory)
            data_config_name: Data configuration name
            dataset_name: Dataset name for metadata loading
            embodiment_tag: Robot embodiment identifier
            instruction_template: Template for task instructions
            prop_pos: Proprioception position in sequence
            max_view_num: Maximum camera views (-1 = all)
            use_fixed_view: Use only ego view
            action_attn_mode: Attention mode for actions
            device: Inference device

            use_mpg: Override MPG enable flag
            mpg_lambda: Override MPG residual strength
            mpg_num_projections: Override Sliced Wasserstein projections
            mpg_refinement_iters: Override refinement iterations
            mpg_gate_temperature: Override MPG gate temperature

            num_inference_timesteps: Override diffusion steps
            clip_normalized_action_keys: Action modality keys to clamp to [-1, 1]
                before inverse normalization. Disabled by default so existing
                inference behavior is unchanged.
            enable_rtc: Enable Real-Time Chunking
            enable_static_prefix_cache: Enable the OPT-01 request-local causal
                prefix cache. The original forward path remains the default.
            enable_npu_fusion_attention: Enable OPT-02 NPU fused attention
            enable_cuda_gqa_attention: Use CUDA SDPA native GQA without
                materializing repeated K/V heads.
            enable_npu_fusion_attention_bsnd: Use the experimental BSND input
                layout for OPT-02 to avoid internal layout conversion.
                with native GQA. The original SDPA path remains the default.
            enable_npu_prefix_segment_route: Enable the OPT-05 validated
                segment route for the OPT-01 static prefix only.
            enable_npu_projection_fusion: Enable the OPT-06 action QKV and
                MLP gate/up projection fusion.
            enable_npu_vectorized_mpg: Enable the OPT-07 fixed-count
                vectorized MPG projection path.
            enable_npu_workspace_reuse: Enable the OPT-08 static-suffix
                clone/overwrite elimination path.
            enable_npu_kv_workspace: Reuse graph-owned full KV buffers and
                update only the dynamic suffix instead of concatenating the
                static prefix on every graph replay.
            enable_cuda_kv_workspace: CUDA alias for the same graph-owned KV
                workspace contract.
            enable_npu_add_rms_norm: Fuse the action-suffix attention residual
                add and following RMSNorm with the native NPU operator.
            enable_npu_fused_rotary: Fuse RoPE half-rotation, multiply and add
                with the native NPU rotary kernel.
            enable_cuda_fused_rotary: Fuse GPU RoPE with a CUDA/Triton kernel.
            enable_cuda_fused_swiglu: Fuse GPU SwiGLU with a CUDA/Triton kernel.
            enable_cuda_fused_only_projection_storage: Store Q/K/V and Gate/Up
                as row views of their fused inference weights on CUDA.
            enable_fused_only_projection_storage: Store Q/K/V and Gate/Up as
                row views of one shared fused inference storage on any device.
                This is inference-only and preserves the original state-dict
                parameter names while avoiding a second fused-weight copy.
            enable_npu_static_tensor_cache: Enable the OPT-10 bounded mask
                and timestep tensor cache.
            enable_npu_dtype_fast_path: Enable the OPT-09 no-op BF16 cast
                elimination in the action attention suffix.
            enable_npu_euler_buffer_cache: Enable the OPT-15 action timestep
                frequency cache.
            enable_npu_vision_state_overlap: Enable the OPT-14 NPU stream
                overlap for vision and proprioception encoders.
            enable_npu_action_compile: Enable the OPT-11 GE compilation of
                the fixed-shape action encoder and decoder.
            enable_npu_vision_compile: Enable the OPT-13 GE compilation of
                the fixed-shape vision backbone and connector.
            enable_npu_linear_weight_prelayout: Enable OPT-16 process-global
                internal MatMul formats and pre-layout 2-D Linear weights.
            enable_npu_persistent_compile_cache: Enable OPT-17 on-disk
                TorchAir cache for fixed vision compilation across restarts.
            npu_compile_cache_dir: Stable, process-owned OPT-17 cache path.
            enable_policy_prompt_cache: Enable the OPT-12 bounded tokenizer
                result cache for repeated task instructions.
            npu_single_sample_fast_path: OPT-03 routing mode: off, auto, or
                force. It remains off by default during validation.
            enable_npu_capture_replay: Enable the fixed-shape OPT-04 NPU graph
                path for the OPT-01 action suffix. It remains off by default.
            enable_npu_prefix_graph_replay: Replay the fixed-shape OPT-01
                Prefix prefill as one NPU graph. It remains off by default.
            npu_graph_cache_max_entries: Maximum number of shape-specialized
                OPT-04 graphs. One preserves the original fixed-shape route.
            enable_cuda_capture_replay: Enable the CUDA Graph implementation
                of the same bounded OPT-04 action-suffix contract.
            cuda_graph_cache_max_entries: CUDA graph capacity. Defaults to the
                NPU-compatible capacity when omitted.
            enable_npu_baseline_flow_graph_replay: Capture the fixed four-step
                baseline flow iteration as one graph. Default off.
            enable_npu_adaptive_flow_graph_replay: Allow experimental
                ADAPT-01 to reuse the OPT-04 single-forward graph for a
                variable number of replays. It remains off by default.
            metadata_variant: Specific metadata variant to use
            stats_selection_mode: Auto-selection mode ('auto', 'task', 'embodiment')
        """
        self.device = torch.device(device)
        if self.device.type == "npu":
            configure_npu_linear_weight_prelayout(
                enable_npu_linear_weight_prelayout
            )
        self.model_path = model_path
        self.data_config_name = data_config_name
        self.prop_pos = prop_pos
        self.action_attn_mode = action_attn_mode
        self.use_fixed_view = use_fixed_view
        self.max_view_num = max_view_num
        self.dataset_name = dataset_name
        self.metadata_variant = metadata_variant
        self.stats_selection_mode = stats_selection_mode
        self.instruction_template = instruction_template
        self.enable_rtc = enable_rtc
        self.clip_normalized_action_keys = set(
            clip_normalized_action_keys or []
        )
        self._npu_capture_unhealthy_reason: Optional[str] = None
        self.enable_policy_prompt_cache = enable_policy_prompt_cache
        self._prompt_token_cache = {}

        self.embodiment_tag = EmbodimentTag(embodiment_tag)

        # ===========================
        # Set data config
        # ===========================
        DataConfigClass = DATA_CONFIG_MAP[self.data_config_name]

        self.data_config = DataConfigClass(
            embodiment_tag=self.embodiment_tag,
            use_fixed_view=self.use_fixed_view,
            max_view_num=self.max_view_num,
            obs_indices=[0],
            action_indices=list(range(16)),
        )
        unnormalized_clip_keys = self.clip_normalized_action_keys.difference(
            self.data_config.action_normalization_modes
        )
        if unnormalized_clip_keys:
            raise ValueError(
                "Normalized action clipping was requested for modalities without "
                f"a configured normalizer: {sorted(unnormalized_clip_keys)}"
            )

        self.unified_mapping = self.data_config.UNIFIED_MAPPING
        self._modality_transform = self.data_config.get_transforms()
        self._modality_transform.eval()
        self._rotation_6d_action_keys = {
            key
            for transform in getattr(self._modality_transform, "transforms", [])
            for key, representation in getattr(
                transform, "target_rotations", {}
            ).items()
            if key.startswith("action.") and representation == "rotation_6d"
        }

        self.language_key = self.data_config.LANGUAGE_KEYS[0]
        self.num_images = len(self.data_config.VIDEO_KEYS)

        # =====================================
        # Load model components and assemble
        # =====================================
        print(f"\n=== Loading model from {self.model_path} ===")

        config = BeingHConfig.from_pretrained(self.model_path)

        llm_version = self._detect_llm_version(config.llm_config)
        print(f"Detected LLM version: {llm_version}")

        QwenConfigClass, LanguageModelClass, _ = VERSION_CONFIGS[llm_version]
        print(f"Using {LanguageModelClass.__name__}")

        # Convert configs
        llm_config_dict = config.llm_config.to_dict()
        llm_config = QwenConfigClass.from_dict(llm_config_dict)

        # Setup expert config
        expert_config_dict = llm_config_dict.get('expert_config')
        if expert_config_dict:
            if not isinstance(expert_config_dict, dict):
                expert_config_dict = expert_config_dict.to_dict()
            expert_config = QwenConfigClass.from_dict(expert_config_dict)
            llm_config.expert_config = expert_config

        # Setup vision config
        vit_config_dict = config.vit_config.to_dict()
        vit_config = InternVisionConfig.from_dict(vit_config_dict)

        # Update main config
        config.llm_config = llm_config
        config.vit_config = vit_config
        
        # Initialize components
        language_model = LanguageModelClass(config.llm_config)
        vit_model = InternVisionModel(config.vit_config)
        connector = InternVLConnector(
            llm_hidden_size=config.llm_config.hidden_size,
            vit_hidden_size=config.vit_config.hidden_size,
            downsample_ratio=config.downsample_ratio,
        )

        # Assemble model
        self.model = BeingH(language_model, vit_model, connector, config)

        # Load weights
        print("Loading state dict...")
        state_dict = load_safetensors(self.model_path)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device, dtype=torch.bfloat16)
        self.fused_projection_storage_report = None
        self.cuda_fused_projection_storage_report = None
        if (
            enable_fused_only_projection_storage
            or enable_cuda_fused_only_projection_storage
        ):
            self.fused_projection_storage_report = (
                prepare_fused_projection_storage(self.model)
            )
            # Keep the legacy attribute for existing GPU reports and callers.
            self.cuda_fused_projection_storage_report = (
                self.fused_projection_storage_report
            )
        self.npu_linear_weight_prelayout_count = 0
        if enable_npu_linear_weight_prelayout:
            self.npu_linear_weight_prelayout_count = (
                prelayout_npu_linear_weights(self.model)
            )
        self.model.eval()
        self.model.enable_static_prefix_cache = enable_static_prefix_cache
        self.model.enable_npu_fusion_attention = enable_npu_fusion_attention
        self.model.enable_cuda_gqa_attention = enable_cuda_gqa_attention
        self.model.npu_fusion_attention_input_layout = (
            "BSND" if enable_npu_fusion_attention_bsnd else "BNSD"
        )
        self.model.enable_npu_hybrid_attention_layout = (
            enable_npu_hybrid_attention_layout
        )
        self.model.enable_npu_prefix_segment_route = (
            enable_npu_prefix_segment_route
        )
        self.model.enable_npu_projection_fusion = (
            enable_npu_projection_fusion
        )
        self.model.enable_npu_vectorized_mpg = enable_npu_vectorized_mpg
        self.model.enable_npu_workspace_reuse = enable_npu_workspace_reuse
        self.model.enable_npu_kv_workspace = (
            enable_npu_kv_workspace or enable_cuda_kv_workspace
        )
        self.model.enable_npu_add_rms_norm = enable_npu_add_rms_norm
        self.model.enable_npu_fused_rotary = enable_npu_fused_rotary
        self.model.enable_npu_fused_swiglu = enable_npu_fused_swiglu
        self.model.enable_cuda_fused_rotary = enable_cuda_fused_rotary
        self.model.enable_cuda_fused_swiglu = enable_cuda_fused_swiglu
        self.model.enable_cuda_fused_only_projection_storage = (
            enable_cuda_fused_only_projection_storage
        )
        self.model.enable_npu_static_tensor_cache = (
            enable_npu_static_tensor_cache
        )
        self.model.enable_npu_dtype_fast_path = enable_npu_dtype_fast_path
        self.model.enable_npu_euler_buffer_cache = (
            enable_npu_euler_buffer_cache
        )
        self.model.enable_npu_vision_state_overlap = (
            enable_npu_vision_state_overlap
        )
        self.model.enable_npu_action_compile = enable_npu_action_compile
        self.model.enable_npu_vision_compile = enable_npu_vision_compile
        self.model.enable_npu_persistent_compile_cache = (
            enable_npu_persistent_compile_cache
        )
        self.model.npu_compile_cache_dir = npu_compile_cache_dir
        self.model.enable_adaptive_flow_steps = enable_adaptive_flow_steps
        self.model.adaptive_flow_min_steps = adaptive_flow_min_steps
        self.model.adaptive_flow_velocity_threshold = (
            adaptive_flow_velocity_threshold
        )
        self.model.enable_adaptive_mpg_refinement = (
            enable_adaptive_mpg_refinement
        )
        self.model.adaptive_mpg_gate_threshold = (
            adaptive_mpg_gate_threshold
        )
        if (
            npu_single_sample_fast_path
            not in VALID_NPU_SINGLE_SAMPLE_FAST_PATH_MODES
        ):
            choices = ", ".join(VALID_NPU_SINGLE_SAMPLE_FAST_PATH_MODES)
            raise ValueError(
                "invalid NPU single-sample fast-path mode "
                f"{npu_single_sample_fast_path!r}; expected one of: {choices}"
            )
        self.model.npu_single_sample_fast_path = npu_single_sample_fast_path
        self.model.enable_npu_capture_replay = (
            enable_npu_capture_replay or enable_cuda_capture_replay
        )
        self.model.enable_npu_prefix_graph_replay = (
            enable_npu_prefix_graph_replay
        )
        graph_cache_max_entries = (
            cuda_graph_cache_max_entries
            if self.device.type == "cuda"
            and cuda_graph_cache_max_entries is not None
            else npu_graph_cache_max_entries
        )
        if graph_cache_max_entries < 1:
            raise ValueError("graph_cache_max_entries must be positive")
        self.model.npu_graph_cache_max_entries = graph_cache_max_entries
        self.model.enable_npu_baseline_flow_graph_replay = (
            enable_npu_baseline_flow_graph_replay
        )
        self.model.enable_npu_adaptive_flow_graph_replay = (
            enable_npu_adaptive_flow_graph_replay
        )
        print(
            "OPT-01 static prefix cache: "
            f"{'enabled' if enable_static_prefix_cache else 'disabled'}"
        )
        print(
            "OPT-02 NPU fusion attention: "
            f"{'enabled' if enable_npu_fusion_attention else 'disabled'}"
        )
        print(
            "NPU-FG-ATTN-BSND fusion-attention layout: "
            f"{self.model.npu_fusion_attention_input_layout}"
        )
        print(
            "OPT-05 NPU prefix segment route: "
            f"{'enabled' if enable_npu_prefix_segment_route else 'disabled'}"
        )
        print(
            "OPT-06 NPU projection fusion: "
            f"{'enabled' if enable_npu_projection_fusion else 'disabled'}"
        )
        print(
            "OPT-07 NPU vectorized MPG: "
            f"{'enabled' if enable_npu_vectorized_mpg else 'disabled'}"
        )
        print(
            "OPT-08 NPU workspace reuse: "
            f"{'enabled' if enable_npu_workspace_reuse else 'disabled'}"
        )
        print(
            "NPU-MEM-01 full KV workspace: "
            f"{'enabled' if enable_npu_kv_workspace else 'disabled'}"
        )
        print(
            "NPU-OP-06A add RMSNorm: "
            f"{'enabled' if enable_npu_add_rms_norm else 'disabled'}"
        )
        print(
            "NPU-FG-ROPE fused rotary: "
            f"{'enabled' if enable_npu_fused_rotary else 'disabled'}"
        )
        print(
            "OPT-10 NPU static tensor cache: "
            f"{'enabled' if enable_npu_static_tensor_cache else 'disabled'}"
        )
        print(
            "OPT-09 NPU dtype fast path: "
            f"{'enabled' if enable_npu_dtype_fast_path else 'disabled'}"
        )
        print(
            "OPT-15 NPU Euler buffer cache: "
            f"{'enabled' if enable_npu_euler_buffer_cache else 'disabled'}"
        )
        print(
            "OPT-14 NPU vision/state overlap: "
            f"{'enabled' if enable_npu_vision_state_overlap else 'disabled'}"
        )
        print(
            "OPT-11 NPU action compile: "
            f"{'enabled' if enable_npu_action_compile else 'disabled'}"
        )
        print(
            "OPT-13 NPU vision compile: "
            f"{'enabled' if enable_npu_vision_compile else 'disabled'}"
        )
        print(
            "OPT-16 NPU Linear weight pre-layout: "
            f"{'enabled' if enable_npu_linear_weight_prelayout else 'disabled'}"
            f" ({self.npu_linear_weight_prelayout_count} weights)"
        )
        print(
            "OPT-17 NPU persistent compile cache: "
            f"{'enabled' if enable_npu_persistent_compile_cache else 'disabled'}"
            f" ({npu_compile_cache_dir or 'default path'})"
        )
        print(
            "ADAPT-01 flow steps: "
            f"{'enabled' if enable_adaptive_flow_steps else 'disabled'}"
            f" (min={adaptive_flow_min_steps}, "
            f"threshold={adaptive_flow_velocity_threshold})"
        )
        print(
            "ADAPT-02 MPG refinement: "
            f"{'enabled' if enable_adaptive_mpg_refinement else 'disabled'}"
            f" (gate threshold={adaptive_mpg_gate_threshold})"
        )
        print(
            "OPT-12 policy prompt cache: "
            f"{'enabled' if enable_policy_prompt_cache else 'disabled'}"
        )
        print(
            "OPT-03 NPU single-sample causal fast path: "
            f"{npu_single_sample_fast_path}"
        )
        print(
            "OPT-04 accelerator capture/replay: "
            f"{'enabled' if self.model.enable_npu_capture_replay else 'disabled'}"
            f" (shape cache entries={graph_cache_max_entries})"
        )
        print(
            "OPT-04P NPU Prefix graph replay: "
            f"{'enabled' if enable_npu_prefix_graph_replay else 'disabled'}"
        )
        print(
            "NPU-FG baseline flow graph: "
            f"{'enabled' if enable_npu_baseline_flow_graph_replay else 'disabled'}"
        )
        print(
            "ADAPT-01 NPU graph replay: "
            f"{'enabled' if enable_npu_adaptive_flow_graph_replay else 'disabled'}"
        )

        # Set core parameters
        # IMPORTANT: Set action_chunk_length from model config (not hardcoded)
        # This ensures inference uses the same chunk length as training
        self.action_chunk_length = self.model.config.action_chunk_length
        patch_size = self.model.config.vit_config.patch_size
        self.num_image_token = self.model.num_image_token = int(
            (self.model.config.force_image_size // patch_size) ** 2 * (0.5 ** 2)
        )

        self.gen_action_type = self.model.config.gen_action_type
        # For flow matching: action_token_num must equal action_chunk_length
        # (one token position per action step)
        self.action_token_num = self.action_chunk_length

        # =====================================
        # Setup and Apply Overrides
        # =====================================
        self._apply_mpg_overrides(
            use_mpg, mpg_lambda, mpg_num_projections, mpg_refinement_iters, mpg_gate_temperature
        )
        self._apply_flow_matching_overrides(num_inference_timesteps)
        self._setup_rtc()

        self._setup_tokenizer()

        _, self.image_transform = build_vit_transform_base(
            is_train=False,
            force_image_size=self.model.config.force_image_size,
            pad2square=False,
            normalize_type='imagenet'
        )

        # ===================
        # Load metadata
        # ===================
        # Load metadata from checkpoint directory (self-contained checkpoint)
        self._load_metadata(Path(self.model_path))

        print("✓ BeingHPolicy initialized successfully")

    def _detect_llm_version(self, llm_config) -> str:
        """Detect LLM version from config."""
        # Check layer module
        if hasattr(llm_config, 'layer_module') and llm_config.layer_module:
            if 'Qwen3' in llm_config.layer_module:
                version = "qwen3"
            elif 'Qwen2' in llm_config.layer_module:
                version = "qwen2.5"
            else:
                version = "qwen2.5"  # Default
        # Check architectures
        elif hasattr(llm_config, 'architectures') and llm_config.architectures:
            if 'Qwen3ForCausalLM' in llm_config.architectures:
                version = "qwen3"
            elif 'Qwen2ForCausalLM' in llm_config.architectures:
                version = "qwen2.5"
            else:
                version = "qwen2.5"


        return version
    
    def _apply_mpg_overrides(
        self,
        use_mpg: Optional[bool],
        mpg_lambda: Optional[float],
        mpg_num_projections: Optional[int],
        mpg_refinement_iters: Optional[int],
        mpg_gate_temperature: Optional[float],
    ):
        """Apply MPG parameter overrides."""
        if use_mpg is not None:
            print(f"Overriding use_mpg: {self.model.config.use_mpg} -> {use_mpg}")
            self.model.config.use_mpg = use_mpg
            self.model.use_mpg = use_mpg

        if mpg_lambda is not None:
            print(f"Overriding mpg_lambda: {self.model.config.mpg_lambda} -> {mpg_lambda}")
            self.model.config.mpg_lambda = mpg_lambda
            if hasattr(self.model, 'mpg') and self.model.mpg is not None:
                self.model.mpg.lambda_strength = mpg_lambda

        if mpg_num_projections is not None:
            print(f"Overriding mpg_num_projections: {self.model.config.mpg_num_projections} -> {mpg_num_projections}")
            self.model.config.mpg_num_projections = mpg_num_projections
            if hasattr(self.model, 'mpg') and self.model.mpg is not None:
                self.model.mpg.num_projections = mpg_num_projections
                self.model.mpg.sliced_wasserstein.num_projections = mpg_num_projections

        if mpg_refinement_iters is not None:
            print(f"Overriding mpg_refinement_iters: {self.model.config.mpg_refinement_iters} -> {mpg_refinement_iters}")
            self.model.config.mpg_refinement_iters = mpg_refinement_iters
            self.model.mpg_refinement_iters = mpg_refinement_iters

        if mpg_gate_temperature is not None:
            print(f"Overriding mpg_gate_temperature: {self.model.config.mpg_gate_temperature} -> {mpg_gate_temperature}")
            self.model.config.mpg_gate_temperature = mpg_gate_temperature
            if hasattr(self.model, 'mpg') and self.model.mpg is not None:
                self.model.mpg.gate_temperature = mpg_gate_temperature

        # Log final MPG configuration
        if hasattr(self.model.config, 'use_mpg') and self.model.config.use_mpg:
            print(f"MPG enabled: lambda={self.model.config.mpg_lambda}, "
                  f"projections={self.model.config.mpg_num_projections}, "
                  f"refinement_iters={self.model.config.mpg_refinement_iters}")

    def _apply_flow_matching_overrides(self, num_inference_timesteps: Optional[int]):
        """Apply flow matching parameter overrides."""
        if num_inference_timesteps is not None:
            print(f"Override num_inference_timesteps: "
                  f"{self.model.config.num_inference_timesteps} -> {num_inference_timesteps}")
            self.model.config.num_inference_timesteps = num_inference_timesteps
            self.model.num_inference_timesteps = num_inference_timesteps

    def _setup_rtc(self):
        """Setup Real-Time Chunking configuration."""
        if self.enable_rtc:
            # Enable inference-time prefix overwriting
            if not self.model.config.use_training_time_rtc:
                print(f"WARNING: Model was not trained with RTC, but RTC inference is enabled. May not work correctly.")

            self.model.config.use_inference_prefix_overwrite = True
            print(f"RTC enabled: Server returns full chunk_length={self.action_chunk_length} actions, "
                  f"trained_with_rtc={self.model.config.use_training_time_rtc}, "
                  f"prefix locking during denoising (client sends prev_chunk)")
        else:
            print("RTC disabled")

    def _setup_tokenizer(self):
        """Setup tokenizer and extract special token IDs."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=False,
            trust_remote_code=True
        )
        
        # Get special token IDs
        tokens = [
            '<|im_start|>', '<|im_end|>', '<img>', '</img>',
            '<|state_start|>', '<|state_end|>'
        ]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        (self.bos_token_id, self.eos_token_id, self.start_of_image,
         self.end_of_image, self.start_of_state, self.end_of_state) = token_ids
        
        newline_encoded = self.tokenizer.encode('\n')
        assert len(newline_encoded) == 1, "Newline should be single token"
        self.newline_token_id = newline_encoded[0]
   
    def _load_metadata(self, checkpoint_dir: Path):
        """
        Load dataset metadata with hierarchical stats and variant support.

        Args:
            checkpoint_dir: Checkpoint directory path (self-contained directory with metadata files)
        """
        metadata_filename = f"{self.dataset_name}_metadata.json"
        metadata_path = checkpoint_dir / metadata_filename

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found at {metadata_path}. "
                "This file is required for normalization statistics."
            )

        print(f"[Metadata] Loading from: {metadata_path}")

        with open(metadata_path, "r") as f:
            all_metadatas = json.load(f)

        metadata_dict = all_metadatas.get(self.dataset_name)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for dataset: '{self.dataset_name}' in {metadata_path}"
            )

        # Hierarchical Stats and Variant Selection 
        variants_key = f"{self.dataset_name}_variants"
        self.stats_level = "unknown"
        self.stats_source = "default"

        if variants_key in all_metadatas:
            """Select appropriate metadata variant."""

            available_variants = list(all_metadatas[variants_key].keys())
            print(f"\n[Metadata Variants] Available: {available_variants}")

            # Separate Level 1 (task) and Level 2 (embodiment) variants for display
            task_variants = [v for v in available_variants
                           if all_metadatas[variants_key][v].get('stats_level') == 'task']
            embodiment_variants = [v for v in available_variants
                                  if all_metadatas[variants_key][v].get('stats_level') == 'embodiment']

            if task_variants:
                print(f"  Level 1 (task): {task_variants}")
            if embodiment_variants:
                print(f"  Level 2 (embodiment): {embodiment_variants}")

            # Select metadata based on user preference
            if self.metadata_variant == "merged":
                # User requested merged statistics (use default top-level)
                print(f"  ✓ Using merged statistics from all variants")
                self.stats_level = metadata_dict.get('stats_level', 'merged')
                self.stats_source = "merged"

            elif self.metadata_variant and self.metadata_variant != "merged":
                # User requested specific variant
                if self.metadata_variant in all_metadatas[variants_key]:
                    metadata_dict = all_metadatas[variants_key][self.metadata_variant]
                    self.stats_level = metadata_dict.get('stats_level', 'unknown')
                    self.stats_source = f"variant:{self.metadata_variant}"
                    print(f"  ✓ Using variant: '{self.metadata_variant}' (level: {self.stats_level})")
                else:
                    print(f"  ⚠ Requested variant '{self.metadata_variant}' not found, using default")
                    # Fall through to auto-select

            if not self.metadata_variant or (self.metadata_variant and self.metadata_variant not in all_metadatas[variants_key] and self.metadata_variant != "merged"):
                # Auto-select (default behavior)
                # Priority: task-specific > embodiment-merged > default
                if self.stats_selection_mode == "task" and task_variants:
                    first_task = task_variants[0]
                    metadata_dict = all_metadatas[variants_key][first_task]
                    self.stats_level = "task"
                    self.stats_source = f"auto:task:{first_task}"
                    print(f"  → Auto-selected (task): '{first_task}'")

                elif self.stats_selection_mode == "embodiment" and embodiment_variants:
                    first_emb = embodiment_variants[0]
                    metadata_dict = all_metadatas[variants_key][first_emb]
                    self.stats_level = "embodiment"
                    self.stats_source = f"auto:embodiment:{first_emb}"
                    print(f"  → Auto-selected (embodiment): '{first_emb}'")

                elif self.stats_selection_mode == "auto":
                    # Default: use first available variant (usually task-specific)
                    first_variant = available_variants[0]
                    metadata_dict = all_metadatas[variants_key][first_variant]
                    self.stats_level = metadata_dict.get('stats_level', 'auto')
                    self.stats_source = f"auto:{first_variant}"
                    print(f"  → Auto-selected: '{first_variant}' (level: {self.stats_level})")

                else:
                    # Use default top-level
                    self.stats_level = metadata_dict.get('stats_level', 'default')
                    self.stats_source = "default"
                    print(f"  → Using default top-level metadata")

        else:
            # Legacy format - no variants
            if "statistics" not in metadata_dict and isinstance(metadata_dict, dict):
                print(f"Warning: Nested metadata detected, Keys: {list(metadata_dict.keys())}")
                first_key = next(iter(metadata_dict))
                print(f"Unalcking key: {first_key}")
                metadata_dict = metadata_dict[first_key]
                self.stats_source = f"legacy:{first_key}"
            else:
                self.stats_source = "legacy:direct"

        if "embodiment_tag" in metadata_dict:
            print(f"[Metadata] Embodiment tag: {metadata_dict['embodiment_tag']}")

        print(f"[Metadata] Stats level: {self.stats_level}, Source: {self.stats_source}")

        # Validate and set metadata
        metadata = DatasetMetadata.model_validate(metadata_dict)
        self._modality_transform.set_metadata(metadata)
        self.metadata = metadata

        print(f"✓ Metadata loaded for '{self.dataset_name}'")

    def _raise_if_npu_capture_unhealthy(self) -> None:
        if self._npu_capture_unhealthy_reason is not None:
            raise NPUCaptureProcessUnhealthyError(
                "NPU capture/replay process is unhealthy; restart the policy "
                f"worker: {self._npu_capture_unhealthy_reason}"
            )

    @torch.no_grad()
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get action with RTC support.
        
        For RTC mode, observations should contain:
        - 'prev_chunk': Previous action chunk (unified 200-dim, normalized)
        - 'inference_delay': Number of prefix actions
        
        Server returns full chunk; client handles composition/shifting.
        
        Args:
            observations: Environment observations
            
        Returns:
            Action dictionary with unnormalized actions
        """
        self._raise_if_npu_capture_unhealthy()

        # ===== 1. Extract RTC Parameters (sent by client) =====
        prev_chunk = observations.pop('prev_chunk', None)
        inference_delay = observations.pop('inference_delay', 0)

        # Prepare prev_chunk if RTC enabled
        if self.enable_rtc and prev_chunk is not None:
            # Client should send unified 200-dim actions (normalized)
            # Convert to tensor and move to device
            if isinstance(prev_chunk, np.ndarray):
                prev_chunk = torch.from_numpy(prev_chunk).to(dtype=torch.float32, device=self.device)
            elif isinstance(prev_chunk, torch.Tensor):
                prev_chunk = prev_chunk.to(dtype=torch.float32, device=self.device)

            # Ensure 3D shape: (B, chunk_length, action_dim)
            if prev_chunk.ndim == 2:
                prev_chunk = prev_chunk.unsqueeze(0)

            # Verify dimensions
            expected_dim = self.model.unified_action_dim  # Note: on model, not config
            if prev_chunk.shape[-1] != expected_dim:
                raise ValueError(
                    f"prev_chunk dimension mismatch: got {prev_chunk.shape[-1]}, "
                    f"expected {expected_dim}. Client must send unified actions, not raw actions!"
                )

        # ===== 2. Prepare Observations =====
        is_batch = self._check_is_batched(observations)
        if not is_batch:
            observations = unsqueeze_dict_values(observations)

        processed_obs = self._modality_transform(observations)

        # ===== 3. Prepare Packed Inputs =====
        packed_inputs = self._prepare_packed_inputs(
            processed_obs, observations[self.language_key]
        )
        for key, value in packed_inputs.items():
            if isinstance(value, torch.Tensor):
                packed_inputs[key] = value.to(self.device, non_blocking=True)

        # ===== 4. Add RTC Parameters to Model Input =====
        if self.enable_rtc and prev_chunk is not None and inference_delay > 0:
            packed_inputs['prev_chunk'] = prev_chunk
            packed_inputs['inference_delay'] = inference_delay

        # ===== 5. Generate Action Chunk =====
        try:
            model_pred = self.model.get_action(**packed_inputs)
        except NPUCaptureProcessUnhealthyError as error:
            self._npu_capture_unhealthy_reason = str(error)
            raise
        action_pred = model_pred["action_pred"]

        # Reshape to 3D for processing, from (B * chunk_length, action_dim) to (1, chunk_length, action_dim)
        action_chunk = action_pred.reshape(1, self.action_chunk_length, -1)  # (1, chunk_length, action_dim)

        # ===== 6. Unnormalize Actions =====
        action_pred_cpu = action_chunk.cpu().to(torch.float32)

        sliced_action_dict = {}
        for key, (start, end) in self.unified_mapping.items():
            if key.startswith('action.'):
                sliced_action_dict[key] = action_pred_cpu[:, :, start:end]

        unknown_clip_keys = self.clip_normalized_action_keys.difference(
            sliced_action_dict
        )
        if unknown_clip_keys:
            raise ValueError(
                "Cannot clip unknown normalized action keys: "
                f"{sorted(unknown_clip_keys)}"
            )
        for key in self.clip_normalized_action_keys:
            sliced_action_dict[key] = torch.clamp(
                sliced_action_dict[key], -1.0, 1.0
            )

        # PyTorch3D decodes Rot6D with Gram-Schmidt. Near-zero first vectors or
        # nearly collinear vector pairs are ill-conditioned; reject them before
        # inverse conversion so the deployment safety wrapper can hold/terminate.
        for key in self._rotation_6d_action_keys:
            rotation_6d = sliced_action_dict[key]
            first = rotation_6d[..., 0:3]
            second = rotation_6d[..., 3:6]
            first_norm = torch.linalg.vector_norm(first, dim=-1)
            safe_first_norm = torch.clamp(first_norm, min=1e-8)
            first_unit = first / safe_first_norm.unsqueeze(-1)
            second_orthogonal = second - (
                (first_unit * second).sum(dim=-1, keepdim=True) * first_unit
            )
            second_orthogonal_norm = torch.linalg.vector_norm(
                second_orthogonal, dim=-1
            )
            invalid = (first_norm < 1e-5) | (second_orthogonal_norm < 1e-5)
            if invalid.any():
                invalid_count = int(invalid.sum().item())
                raise ValueError(
                    f"Degenerate Rot6D prediction for {key}: "
                    f"{invalid_count} action(s) cannot define a stable rotation"
                )

        unnormalized_action_dict = self._modality_transform.unapply(sliced_action_dict)

        if not is_batch:
            unnormalized_action_dict = squeeze_dict_values(unnormalized_action_dict)

        # ===== 7. Return Actions =====
        result = {k: v.tolist() for k, v in unnormalized_action_dict.items()}

        # For RTC mode, also return unified actions for next query
        if self.enable_rtc:
            if not is_batch:
                result['action_unified'] = action_pred_cpu.squeeze(0).tolist()  # (chunk_length, action_dim)
            else:
                result['action_unified'] = action_pred_cpu.tolist()  # (batch, chunk_length, action_dim)

        return result

    def _encode_prompt_parts(self, instruction: str):
        cache_key = (
            self.model.system_message,
            self.instruction_template,
            self.action_chunk_length,
            instruction,
        )
        if self.enable_policy_prompt_cache:
            cached = self._prompt_token_cache.get(cache_key)
            if cached is not None:
                return cached

        system_ids = tuple(
            self.tokenizer.encode(f"system\n{self.model.system_message}")
        )
        user_ids = tuple(self.tokenizer.encode("user\n"))
        assistant_ids = tuple(self.tokenizer.encode("assistant\n"))
        formatted = self.instruction_template.format(
            task_description=instruction,
            k=self.action_chunk_length,
        )
        instruction_ids = tuple(self.tokenizer.encode(formatted))
        result = (system_ids, user_ids, assistant_ids, instruction_ids)
        if self.enable_policy_prompt_cache:
            if len(self._prompt_token_cache) >= 256:
                self._prompt_token_cache.clear()
            self._prompt_token_cache[cache_key] = result
        return result

    def _prepare_packed_inputs(self, processed_obs: Dict[str, Any], instructions: List[str]) -> Dict[str, Any]:
        """Prepare packed inputs for model."""
        first_state_key = next((k for k in processed_obs if k.startswith('state.')), None)
        N, _ = processed_obs[first_state_key].shape
        
        state_tensor = torch.zeros(N, self.model.unified_state_dim, dtype=torch.float32)

        # Fill state tensor from mapping
        for key, (start, end) in self.unified_mapping.items():
            if key.startswith('state.') and key in processed_obs:
                source_tensor = processed_obs[key]
                
                expected_dim = end - start
                if source_tensor.shape[-1] != expected_dim:
                    raise ValueError(
                        f"Dimension mismatch for '{key}': "
                        f"expected {expected_dim}, got {source_tensor.shape[-1]}"
                    )
                
                state_tensor[:, start:end] = source_tensor

        # Prepare vision inputs
        all_frames = []
        for key in self.data_config.VIDEO_KEYS:
            video_np = processed_obs[key]
            for f_idx in range(video_np.shape[0]):
                all_frames.append(Image.fromarray(video_np[f_idx]))
            
        pixel_values = torch.stack([self.image_transform(img) for img in all_frames])

        # Build prompt components.  OPT-12 caches immutable tokenizer results
        # by the full template/instruction key; packed tensors remain
        # request-local because images and state are dynamic.
        system_ids, user_ids, assistant_ids, inst_ids = (
            self._encode_prompt_parts(instructions[0])
        )
        
        # CORE: Build packed sequence
        packed_text_ids, packed_text_indexes = [], []
        packed_vit_token_indexes, packed_state_indexes, packed_action_indexes = [], [], []
        packed_position_ids = []
        split_lens, attn_modes = [], []
        
        curr = 0
        curr_rope_id = 0

        # === Block 1: System Turn ===
        block_ids = [
            self.bos_token_id,
            *system_ids,
            self.eos_token_id,
            self.newline_token_id,
        ]
        packed_text_ids.extend(block_ids)
        packed_text_indexes.extend(range(curr, curr + len(block_ids)))
        packed_position_ids.extend(range(curr_rope_id, curr_rope_id + len(block_ids)))
        curr += len(block_ids)
        curr_rope_id += len(block_ids)
        split_lens.append(len(block_ids))
        attn_modes.append("causal")

        # === Block 2: Content (user + vision + state + instruction) ===
        curr_split_start = curr

        # User prompt
        # is_bos: True, is_eos: False
        block_ids = [self.bos_token_id, *user_ids]
        packed_text_ids.extend(block_ids)
        packed_text_indexes.extend(range(curr, curr + len(block_ids)))
        curr += len(block_ids)
        
        # Vision
        num_total_images = len(all_frames)
        packed_text_ids.extend([self.start_of_image])
        packed_text_indexes.append(curr)
        curr += 1

        packed_vit_token_indexes.extend(range(curr, curr + self.num_image_token * num_total_images))
        curr += self.num_image_token * num_total_images

        packed_text_ids.extend([self.end_of_image])
        packed_text_indexes.append(curr)
        curr += 1
        
        # State
        packed_text_ids.extend([self.start_of_state])
        packed_text_indexes.append(curr)
        curr += 1

        packed_state_indexes.append(curr) # State occupies one token position
        curr += 1

        packed_text_ids.extend([self.end_of_state])
        packed_text_indexes.append(curr)
        curr += 1
        
        # Instruction
        # is_bos: False, is_eos: True
        block_ids = [*inst_ids, self.eos_token_id, self.newline_token_id]
        packed_text_ids.extend(block_ids)
        packed_text_indexes.extend(range(curr, curr + len(block_ids)))
        curr += len(block_ids)
        
        # Finalize Content Block
        curr_split_len = curr - curr_split_start
        packed_position_ids.extend(range(curr_rope_id, curr_rope_id + curr_split_len))
        curr_rope_id += curr_split_len
        split_lens.append(curr_split_len)
        attn_modes.append("causal")

        # === # Block 3: Action (assistant + action placeholders) ===
        curr_split_start = curr

        # Assistant prompt
        # is_bos: True, is_eos: False
        block_ids = [self.bos_token_id, *assistant_ids]
        packed_text_ids.extend(block_ids)
        packed_text_indexes.extend(range(curr, curr + len(block_ids)))
        curr += len(block_ids)
        
        # Action placeholders
        packed_action_indexes.extend(range(curr, curr + self.action_token_num))
        curr += self.action_token_num

        # Final EOS
        packed_text_ids.append(self.eos_token_id)
        packed_text_indexes.append(curr)
        curr += 1
        
        # Finalize Action Block
        curr_split_len = curr - curr_split_start
        packed_position_ids.extend(range(curr_rope_id, curr_rope_id + curr_split_len))
        # curr_rope_id += curr_split_len # No need to update rope_id for the last block
        split_lens.append(curr_split_len)
        attn_modes.append("causal")

        # Padding for block attention
        sequence_length = curr
        padding_len = (BLOCK_SIZE - (sequence_length % BLOCK_SIZE)) % BLOCK_SIZE
        sample_lens = [sequence_length]

        if padding_len > 0:
            split_lens.append(padding_len)
            attn_modes.append("causal")
            sample_lens.append(padding_len)

        # Assemble final dict
        packed_inputs = {
            "sequence_length": sequence_length,
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "sample_lens": sample_lens,
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "split_lens": split_lens,
            "attn_modes": attn_modes,
            "packed_vit_tokens": pixel_values.to(dtype=torch.bfloat16),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_action_indexes": torch.tensor(packed_action_indexes, dtype=torch.long),
            "padded_state": state_tensor.to(dtype=torch.bfloat16),
            "packed_state_indexes": torch.tensor(packed_state_indexes, dtype=torch.long),
            "embodiment_ids": torch.tensor([31], dtype=torch.long),
        }
        if (
            self.model.enable_npu_prefix_segment_route
            and self.model.enable_static_prefix_cache
        ):
            prefix_segment_route = None
            prefix_segment_route_fallback_reason = None
            try:
                if not packed_action_indexes:
                    raise ValueError("action indexes are empty")
                action_start = packed_action_indexes[0]
                prefix_und_indexes = [
                    index
                    for index in packed_text_indexes + packed_vit_token_indexes
                    if index < action_start
                ]
                prefix_gen_indexes = [
                    index
                    for index in packed_state_indexes
                    if index < action_start
                ]
                prefix_segment_route = build_prefix_segment_route(
                    und_global_indexes=prefix_und_indexes,
                    gen_global_indexes=prefix_gen_indexes,
                    prefix_length=action_start,
                )
            except ValueError as error:
                prefix_segment_route_fallback_reason = (
                    f"invalid_cpu_prefix_route:{error}"
                )
            packed_inputs["npu_prefix_segment_route"] = prefix_segment_route
            packed_inputs["npu_prefix_segment_route_fallback_reason"] = (
                prefix_segment_route_fallback_reason
            )
        return packed_inputs

    def _check_is_batched(self, obs: Dict[str, Any]) -> bool:
        first_val = next(iter(obs.values()))

        if isinstance(first_val, np.ndarray):
            return first_val.ndim > 1 and ("state" in next(iter(obs.keys())) or "video" in next(iter(obs.keys())))
        if isinstance(first_val, list):  # Applies to instruction lists
            return True
        return False
        
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        return self.data_config.modality_config()
