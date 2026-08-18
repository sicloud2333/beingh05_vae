# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""BeingH VLA Inference Server - Entry point for running the inference server."""

import argparse
import copy
import json
import torch
try:
    import torch_npu
except ImportError:
    torch_npu = None
import numpy as np
import random
import tyro
from pathlib import Path
from typing import Optional

from .beingh_policy import BeingHPolicy
from .beingh_service import BeingHInferenceServer

def set_seed(seed: int):
    """Set seed for all random number generators to ensure reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch_npu is not None and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"--- Random seed set to {seed} for reproducibility ---")

# --- 2. Define command line arguments using dataclass ---
from dataclasses import dataclass

@dataclass
class ServerArgs:
    """Command line arguments for the inference server."""
    model_path: str
    """Path to the trained model checkpoint (self-contained directory with model.safetensors and metadata)."""

    port: int = 5555
    """Port for the server to run on."""

    host: str = "0.0.0.0"
    """Host address for the server to bind to."""

    api_token: Optional[str] = None
    """Optional API token for authentication."""

    seed: int = 42
    """Random seed."""

    prompt_template: str = "long"
    prop_pos: str = "front"

    data_config_name: str = ""
    embodiment_tag: str = ""

    dataset_name: str = ""

    max_view_num: int = -1
    use_fixed_view: bool = False

    # MPG Parameter Overrides
    # =====================================================
    use_mpg: Optional[bool] = None
    """Override: Enable/disable MPG enhancement at inference."""

    mpg_lambda: Optional[float] = None
    """Override: MPG residual strength (e.g., 0.1)."""

    mpg_num_projections: Optional[int] = None
    """Override: Number of Sliced Wasserstein projections."""

    mpg_refinement_iters: Optional[int] = None
    """Override: MPG refinement iterations at inference."""

    mpg_gate_temperature: Optional[float] = None
    """Override: MPG gate temperature (higher = softer gating)."""

    # Flow Matching Parameter Override
    # =====================================================
    num_inference_timesteps: Optional[int] = None
    """Override: Number of flow matching denoising steps (default: use model config)."""

    # RTC (Real-Time Chunking) Parameter
    # =====================================================
    enable_rtc: bool = True
    """Enable Training-Time RTC support (requires model trained with RTC)."""

    enable_static_prefix_cache: bool = False
    """Enable the opt-in OPT-01 request-local causal prefix cache."""

    enable_npu_fusion_attention: bool = False
    """Enable the opt-in OPT-02 NPU fused attention with native GQA."""

    enable_cuda_gqa_attention: bool = False
    """Enable CUDA SDPA native GQA without materializing repeated K/V."""

    enable_npu_fusion_attention_bsnd: bool = False
    """Use the experimental BSND layout for OPT-02 fused attention."""

    enable_npu_hybrid_attention_layout: bool = False
    """Use BNSD for prefix prefill while keeping BSND for action suffix."""

    enable_npu_prefix_segment_route: bool = False
    """Enable the opt-in OPT-05 static-prefix segment route."""

    enable_npu_projection_fusion: bool = False
    """Enable the opt-in OPT-06 action projection fusion."""

    enable_npu_vectorized_mpg: bool = False
    """Enable the opt-in OPT-07 vectorized MPG projections."""

    enable_npu_workspace_reuse: bool = False
    """Enable the opt-in OPT-08 static-suffix workspace reuse path."""

    enable_npu_kv_workspace: bool = False
    """Reuse graph-owned full KV buffers instead of replay-time prefix cat."""

    enable_cuda_kv_workspace: bool = False
    """CUDA alias for graph-owned full KV buffers."""

    enable_npu_add_rms_norm: bool = False
    """Fuse the action-suffix attention residual add and RMSNorm."""

    enable_npu_fused_rotary: bool = False
    """Fuse query/key RoPE with the native NPU rotary kernel."""

    enable_npu_fused_swiglu: bool = False
    """Fuse SwiGLU activation and multiply after the gate/up projection."""

    enable_cuda_fused_rotary: bool = False
    """Fuse query/key RoPE with the CUDA/Triton kernel."""

    enable_cuda_fused_swiglu: bool = False
    """Fuse SwiGLU activation and multiply with the CUDA/Triton kernel."""

    enable_cuda_fused_only_projection_storage: bool = False
    """Use one shared CUDA storage for fused QKV and Gate/Up weights."""

    enable_fused_only_projection_storage: bool = False
    """Use one shared physical storage for fused QKV/Gate-Up weights."""

    enable_npu_static_tensor_cache: bool = False
    """Enable the opt-in OPT-10 mask/timestep tensor cache."""

    enable_npu_dtype_fast_path: bool = False
    """Enable the opt-in OPT-09 no-op BF16 cast elimination."""

    enable_npu_euler_buffer_cache: bool = False
    """Enable the opt-in OPT-15 action timestep frequency cache."""

    enable_npu_vision_state_overlap: bool = False
    """Enable the opt-in OPT-14 vision/proprioception stream overlap."""

    enable_npu_action_compile: bool = False
    """Enable the opt-in OPT-11 fixed-shape action subgraph compiler."""

    enable_npu_vision_compile: bool = False
    """Enable the opt-in OPT-13 fixed-shape vision compiler."""

    enable_npu_linear_weight_prelayout: bool = False
    """Enable the opt-in OPT-16 NPU Linear weight pre-layout."""

    enable_npu_persistent_compile_cache: bool = False
    """Enable the opt-in OPT-17 TorchAir compile cache across restarts."""

    npu_compile_cache_dir: Optional[str] = None
    """Stable process-owned directory for the OPT-17 compile cache."""

    enable_adaptive_flow_steps: bool = False
    """Enable experimental ADAPT-01 flow-step early termination."""

    adaptive_flow_min_steps: int = 2
    adaptive_flow_velocity_threshold: float = 0.0

    enable_adaptive_mpg_refinement: bool = False
    """Enable experimental ADAPT-02 MPG refinement early exit."""

    adaptive_mpg_gate_threshold: float = 0.0

    enable_policy_prompt_cache: bool = False
    """Enable the opt-in OPT-12 repeated-instruction tokenizer cache."""

    npu_single_sample_fast_path: str = "off"
    """OPT-03 routing mode: off, auto, or force."""

    enable_npu_capture_replay: bool = False
    """Enable the opt-in OPT-04 fixed-shape NPU capture/replay path."""

    enable_npu_prefix_graph_replay: bool = False
    """Replay the fixed-shape OPT-01 Prefix prefill as one NPU graph."""

    enable_cuda_capture_replay: bool = False
    """Enable bounded CUDA Graph capture/replay for the action suffix."""

    npu_graph_cache_max_entries: int = 1
    """Maximum number of OPT-04 shape-specialized graphs per worker."""

    cuda_graph_cache_max_entries: Optional[int] = None
    """CUDA graph capacity; defaults to npu_graph_cache_max_entries."""

    npu_graph_prewarm_instructions_file: Optional[str] = None
    """Optional JSON string list captured before the server accepts traffic."""

    freeze_npu_graph_cache_after_prewarm: bool = True
    """Forbid request-path graph capture after successful startup prewarm."""

    enable_npu_baseline_flow_graph_replay: bool = False
    """Capture the fixed baseline four-step flow iteration as one graph."""

    enable_npu_adaptive_flow_graph_replay: bool = False
    """Allow ADAPT-01 to use a variable number of OPT-04 graph replays."""

    # Metadata Variant Selection
    # =====================================================
    metadata_variant: Optional[str] = None
    """Metadata variant to use: None (auto), 'merged', or specific variant name like 'adamu_pick_simple' or 'PND_AdamU'"""

    stats_selection_mode: str = "auto"
    """Stats selection mode for hierarchical metadata: 'auto' (default), 'task', 'embodiment', 'total'"""


# --- 3. Main function ---
def prewarm_npu_graph_cache(
    policy: BeingHPolicy,
    instructions_file: str,
    *,
    seed: int,
    freeze_after: bool,
) -> dict:
    """Capture all configured instruction shapes before opening the server."""
    if not policy.model.enable_npu_capture_replay:
        raise ValueError("graph prewarm requires enable_npu_capture_replay")
    payload = json.loads(Path(instructions_file).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("prewarm instructions must be a non-empty JSON list")
    if not all(isinstance(item, str) and item.strip() for item in payload):
        raise ValueError("every prewarm instruction must be a non-empty string")

    rng = np.random.default_rng(seed)
    template = {
        # LIBERO/robosuite sends simulator state arrays as float64.  The
        # modality transform records each key's input dtype on first use, so
        # startup prewarm must use the same wire dtype as real requests.
        "state.state": np.zeros((1, 8), dtype=np.float64),
        "state.eef_position": np.zeros((1, 3), dtype=np.float64),
        "state.eef_rotation": np.zeros((1, 3), dtype=np.float64),
        "state.libero_gripper_position": np.zeros((1, 2), dtype=np.float64),
        "video.top_view": rng.integers(
            0, 256, size=(1, 256, 256, 3), dtype=np.uint8
        ),
        "video.wrist_view": rng.integers(
            0, 256, size=(1, 256, 256, 3), dtype=np.uint8
        ),
    }
    policy.model.unfreeze_npu_capture_replay_cache()
    for instruction in payload:
        observation = copy.deepcopy(template)
        observation["language.instruction"] = [instruction]
        policy.get_action(observation)
        if policy.device.type == "npu":
            torch.npu.synchronize()
        elif policy.device.type == "cuda":
            torch.cuda.synchronize(policy.device)

    runner = policy.model._get_npu_action_suffix_graph_runner()
    runners = {"action_suffix": runner}
    prefix_runner = getattr(policy.model, "_npu_prefix_graph_runner", None)
    if prefix_runner is not None:
        runners["prefix"] = prefix_runner
    baseline_runner = policy.model._npu_baseline_flow_graph_runner
    if baseline_runner is not None:
        runners["baseline_flow"] = baseline_runner
    prewarm_stats = {name: item.stats() for name, item in runners.items()}
    for name, item_stats in prewarm_stats.items():
        if (
            item_stats["cache_full_fallback_count"]
            or item_stats["failed_key_count"]
            or item_stats["unhealthy"]
        ):
            raise RuntimeError(
                f"{policy.device.type.upper()} {name} graph startup prewarm "
                "did not capture every "
                f"configured shape: {json.dumps(item_stats, sort_keys=True)}"
            )
    if freeze_after:
        policy.model.freeze_npu_capture_replay_cache()
    stats = {name: item.stats() for name, item in runners.items()}
    set_seed(seed)
    print(
        f"{policy.device.type.upper()} graph startup prewarm: "
        + json.dumps(stats, sort_keys=True)
    )
    return stats


def main(args: ServerArgs):

    set_seed(args.seed)

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch_npu is not None and torch.npu.is_available():
        device = "npu"
    else:
        device = "cpu"
    print(f"Running policy on device: {device}")

    if args.prompt_template == "short":
        instruction_template = "{task_description}"
    else:
        instruction_template = "According to the instruction '{task_description}', what's the micro-step actions in the next {k} steps?"

    # Initialize BeingHPolicy
    # Policy handles model loading, transform loading, metadata loading, etc.
    print("--- 2. Initializing BeingHPolicy ---")
    policy = BeingHPolicy(
        model_path=args.model_path,
        data_config_name=args.data_config_name,
        embodiment_tag=args.embodiment_tag,
        dataset_name=args.dataset_name,
        instruction_template=instruction_template,
        prop_pos=args.prop_pos,
        max_view_num=args.max_view_num,
        use_fixed_view=args.use_fixed_view,
        device=device,
        # MPG parameter overrides
        use_mpg=args.use_mpg,
        mpg_lambda=args.mpg_lambda,
        mpg_num_projections=args.mpg_num_projections,
        mpg_refinement_iters=args.mpg_refinement_iters,
        mpg_gate_temperature=args.mpg_gate_temperature,
        # Flow matching parameter override
        num_inference_timesteps=args.num_inference_timesteps,
        # RTC parameter
        enable_rtc=args.enable_rtc,
        # Lossless inference optimization flags
        enable_static_prefix_cache=args.enable_static_prefix_cache,
        enable_npu_fusion_attention=args.enable_npu_fusion_attention,
        enable_cuda_gqa_attention=args.enable_cuda_gqa_attention,
        enable_npu_fusion_attention_bsnd=(
            args.enable_npu_fusion_attention_bsnd
        ),
        enable_npu_hybrid_attention_layout=(
            args.enable_npu_hybrid_attention_layout
        ),
        enable_npu_prefix_segment_route=args.enable_npu_prefix_segment_route,
        enable_npu_projection_fusion=args.enable_npu_projection_fusion,
        enable_npu_vectorized_mpg=args.enable_npu_vectorized_mpg,
        enable_npu_workspace_reuse=args.enable_npu_workspace_reuse,
        enable_npu_kv_workspace=args.enable_npu_kv_workspace,
        enable_cuda_kv_workspace=args.enable_cuda_kv_workspace,
        enable_npu_add_rms_norm=args.enable_npu_add_rms_norm,
        enable_npu_fused_rotary=args.enable_npu_fused_rotary,
        enable_npu_fused_swiglu=args.enable_npu_fused_swiglu,
        enable_cuda_fused_rotary=args.enable_cuda_fused_rotary,
        enable_cuda_fused_swiglu=args.enable_cuda_fused_swiglu,
        enable_cuda_fused_only_projection_storage=(
            args.enable_cuda_fused_only_projection_storage
        ),
        enable_fused_only_projection_storage=(
            args.enable_fused_only_projection_storage
        ),
        enable_npu_static_tensor_cache=args.enable_npu_static_tensor_cache,
        enable_npu_dtype_fast_path=args.enable_npu_dtype_fast_path,
        enable_npu_euler_buffer_cache=args.enable_npu_euler_buffer_cache,
        enable_npu_vision_state_overlap=args.enable_npu_vision_state_overlap,
        enable_npu_action_compile=args.enable_npu_action_compile,
        enable_npu_vision_compile=args.enable_npu_vision_compile,
        enable_npu_linear_weight_prelayout=(
            args.enable_npu_linear_weight_prelayout
        ),
        enable_npu_persistent_compile_cache=(
            args.enable_npu_persistent_compile_cache
        ),
        npu_compile_cache_dir=args.npu_compile_cache_dir,
        enable_adaptive_flow_steps=args.enable_adaptive_flow_steps,
        adaptive_flow_min_steps=args.adaptive_flow_min_steps,
        adaptive_flow_velocity_threshold=(
            args.adaptive_flow_velocity_threshold
        ),
        enable_adaptive_mpg_refinement=(
            args.enable_adaptive_mpg_refinement
        ),
        adaptive_mpg_gate_threshold=args.adaptive_mpg_gate_threshold,
        enable_policy_prompt_cache=args.enable_policy_prompt_cache,
        npu_single_sample_fast_path=args.npu_single_sample_fast_path,
        enable_npu_capture_replay=args.enable_npu_capture_replay,
        enable_npu_prefix_graph_replay=args.enable_npu_prefix_graph_replay,
        npu_graph_cache_max_entries=args.npu_graph_cache_max_entries,
        enable_cuda_capture_replay=args.enable_cuda_capture_replay,
        cuda_graph_cache_max_entries=args.cuda_graph_cache_max_entries,
        enable_npu_baseline_flow_graph_replay=(
            args.enable_npu_baseline_flow_graph_replay
        ),
        enable_npu_adaptive_flow_graph_replay=(
            args.enable_npu_adaptive_flow_graph_replay
        ),
        # Metadata variant selection
        metadata_variant=args.metadata_variant,
        stats_selection_mode=args.stats_selection_mode,
    )

    if args.npu_graph_prewarm_instructions_file is not None:
        prewarm_npu_graph_cache(
            policy,
            args.npu_graph_prewarm_instructions_file,
            seed=args.seed,
            freeze_after=args.freeze_npu_graph_cache_after_prewarm,
        )

    # Create and run server
    # Server only needs a policy object that implements get_action method
    print(f"--- 3. Starting Inference Server on {args.host}:{args.port} ---")
    server = BeingHInferenceServer(
        policy=policy,
        port=args.port,
        host=args.host,
        api_token=args.api_token
    )
    server.run()

if __name__ == "__main__":
    # Use tyro to parse command line arguments
    args = tyro.cli(ServerArgs)
    main(args)
