#!/usr/bin/env python3
"""Numeric gate for NPU fused projection shared storage."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from BeingH.inference.beingh_policy import BeingHPolicy
from profile_npu_pipeline import build_observation


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    np.random.seed(seed)


def create_policy(args: argparse.Namespace, shared_storage: bool) -> BeingHPolicy:
    template = (
        "According to the instruction '{task_description}', what's the "
        "micro-step actions in the next {k} steps?"
    )
    return BeingHPolicy(
        model_path=args.model_path,
        data_config_name="libero_nonorm",
        dataset_name="libero_posttrain",
        embodiment_tag="libero",
        instruction_template=template,
        max_view_num=-1,
        use_fixed_view=False,
        action_attn_mode="causal",
        device=f"npu:{args.device}",
        enable_rtc=False,
        enable_static_prefix_cache=True,
        enable_npu_fusion_attention=True,
        enable_npu_fusion_attention_bsnd=True,
        enable_npu_prefix_segment_route=True,
        enable_npu_projection_fusion=True,
        enable_npu_vectorized_mpg=True,
        enable_npu_workspace_reuse=True,
        enable_npu_kv_workspace=True,
        enable_npu_euler_buffer_cache=True,
        enable_npu_fused_rotary=True,
        enable_npu_fused_swiglu=True,
        enable_npu_vision_compile=True,
        enable_npu_vision_state_overlap=True,
        enable_npu_persistent_compile_cache=True,
        npu_compile_cache_dir=args.cache_dir,
        enable_npu_capture_replay=True,
        npu_graph_cache_max_entries=8,
        enable_policy_prompt_cache=True,
        enable_fused_only_projection_storage=shared_storage,
    )


def emit(args: argparse.Namespace) -> None:
    torch.npu.set_device(args.device)
    policy = create_policy(args, args.variant == "shared")
    observation = build_observation()
    processed = policy._modality_transform(copy.deepcopy(observation))
    packed = policy._prepare_packed_inputs(
        processed, observation["language.instruction"]
    )
    packed = {
        key: value.to(policy.device) if isinstance(value, torch.Tensor) else value
        for key, value in packed.items()
    }

    # Warm the same fixed observation/shape before recording outputs.
    set_seed(args.seed_start - 1)
    with torch.no_grad():
        policy.model.get_action(**packed)
    torch.npu.synchronize()

    outputs = []
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        set_seed(seed)
        with torch.no_grad():
            action = policy.model.get_action(**packed)["action_pred"]
        torch.npu.synchronize()
        outputs.append(action.detach().cpu().float().numpy())
    array = np.stack(outputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "variant": args.variant,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "shape": list(array.shape),
        "all_finite": bool(np.isfinite(array).all()),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "fused_projection_storage_report": getattr(
            policy, "fused_projection_storage_report", None
        ),
    }
    np.savez_compressed(
        args.output,
        outputs=array,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    print("NPU_PROJECTION_NUMERIC " + json.dumps(metadata, sort_keys=True))


def compare(args: argparse.Namespace) -> None:
    baseline = np.load(args.baseline)["outputs"]
    shared = np.load(args.shared)["outputs"]
    if baseline.shape != shared.shape:
        raise ValueError(f"shape mismatch: {baseline.shape} != {shared.shape}")
    rows = []
    for index, (left, right) in enumerate(zip(baseline, shared, strict=True)):
        difference = np.abs(left - right)
        left_flat = left.reshape(-1).astype(np.float64)
        right_flat = right.reshape(-1).astype(np.float64)
        cosine = float(
            np.dot(left_flat, right_flat)
            / (np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
        )
        rows.append(
            {
                "seed": args.seed_start + index,
                "exact_equal": bool(np.array_equal(left, right)),
                "max_abs": float(difference.max()),
                "mean_abs": float(difference.mean()),
                "cosine": cosine,
                "finite": bool(np.isfinite(left).all() and np.isfinite(right).all()),
            }
        )
    summary = {
        "sample_count": len(rows),
        "all_finite": all(row["finite"] for row in rows),
        "all_exact_equal": all(row["exact_equal"] for row in rows),
        "max_abs": max(row["max_abs"] for row in rows),
        "mean_abs": float(np.mean([row["mean_abs"] for row in rows])),
        "min_cosine": min(row["cosine"] for row in rows),
    }
    result = {"summary": summary, "samples": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("NPU_PROJECTION_NUMERIC_COMPARE " + json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    emit_parser = subparsers.add_parser("emit")
    emit_parser.add_argument("--model-path", required=True)
    emit_parser.add_argument("--variant", choices=["baseline", "shared"], required=True)
    emit_parser.add_argument("--device", type=int, default=0)
    emit_parser.add_argument("--seed-start", type=int, default=50000)
    emit_parser.add_argument("--seed-count", type=int, default=20)
    emit_parser.add_argument("--cache-dir")
    emit_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--shared", type=Path, required=True)
    compare_parser.add_argument("--seed-start", type=int, default=50000)
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    emit(args) if args.command == "emit" else compare(args)


if __name__ == "__main__":
    main()
