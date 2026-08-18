"""Compare the current four-level lossless NPU optimization ladder."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch_npu

from BeingH.inference.beingh_policy import BeingHPolicy
from profile_npu_pipeline import build_observation


SEEDS = (41, 42, 43, 44, 45)
MIN_E1_COSINE_SIMILARITY = 0.99999


@dataclass(frozen=True)
class Variant:
    name: str
    static_prefix_cache: bool
    npu_fusion_attention: bool
    npu_capture_replay: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "static_prefix_cache": self.static_prefix_cache,
            "npu_fusion_attention": self.npu_fusion_attention,
            "fusion_sync": True if self.npu_fusion_attention else None,
            "npu_single_sample_fast_path": "off",
            "npu_capture_replay": self.npu_capture_replay,
        }


VARIANTS = (
    Variant("A", False, False, False),
    Variant("B", True, False, False),
    Variant("C", True, True, False),
    Variant("D", True, True, True),
)
COMPARISONS = (
    ("A-B", "A", "B"),
    ("B-C", "B", "C"),
    ("C-D", "C", "D"),
    ("A-D", "A", "D"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    np.random.seed(seed)


def bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().float().numpy()
    return bytes_hash(array.tobytes())


def numpy_rng_hash() -> str:
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    payload = bytearray(algorithm.encode("utf-8"))
    payload.extend(keys.tobytes())
    payload.extend(struct.pack("!q?", position, bool(has_gauss)))
    payload.extend(struct.pack("!d", float(cached_gaussian)))
    return bytes_hash(bytes(payload))


def rng_hashes() -> dict[str, str]:
    return {
        "cpu": bytes_hash(torch.get_rng_state().cpu().numpy().tobytes()),
        "npu": bytes_hash(torch.npu.get_rng_state().cpu().numpy().tobytes()),
        "numpy": numpy_rng_hash(),
    }


def compare_tensors(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left_fp32 = left.detach().cpu().float()
    right_fp32 = right.detach().cpu().float()
    difference = (left_fp32 - right_fp32).abs()
    denominator = left_fp32.abs().clamp_min(1e-8)
    relative = difference / denominator
    cosine = torch.nn.functional.cosine_similarity(
        left_fp32.flatten().unsqueeze(0),
        right_fp32.flatten().unsqueeze(0),
    ).item()
    return {
        "shape": list(left.shape),
        "left_finite": bool(torch.isfinite(left_fp32).all().item()),
        "right_finite": bool(torch.isfinite(right_fp32).all().item()),
        "exact_equal": bool(torch.equal(left_fp32, right_fp32)),
        "max_absolute_error": difference.max().item(),
        "mean_absolute_error": difference.mean().item(),
        "max_relative_error": relative.max().item(),
        "mean_relative_error": relative.mean().item(),
        "cosine_similarity": cosine,
        "left_hash": tensor_hash(left),
        "right_hash": tensor_hash(right),
    }


def create_policy(model_path: str, device: int) -> BeingHPolicy:
    template = (
        "According to the instruction '{task_description}', what's the "
        "micro-step actions in the next {k} steps?"
    )
    return BeingHPolicy(
        model_path=model_path,
        data_config_name="libero_nonorm",
        dataset_name="libero_posttrain",
        embodiment_tag="libero",
        instruction_template=template,
        max_view_num=-1,
        use_fixed_view=False,
        action_attn_mode="causal",
        device=f"npu:{device}",
        enable_rtc=False,
    )


def prepare_model_inputs(
    policy: BeingHPolicy,
    observation: dict[str, Any],
) -> dict[str, Any]:
    processed = policy._modality_transform(copy.deepcopy(observation))
    packed = policy._prepare_packed_inputs(
        processed,
        observation["language.instruction"],
    )
    return {
        key: value.to(policy.device) if isinstance(value, torch.Tensor) else value
        for key, value in packed.items()
    }


def install_graph_counters(runner: Any) -> dict[str, int]:
    counters = {"capture": 0, "bind": 0, "replay": 0}

    def wrap(name: str, function: Callable[..., Any]) -> Callable[..., Any]:
        def counted(*args: Any, **kwargs: Any) -> Any:
            counters[name] += 1
            return function(*args, **kwargs)

        return counted

    runner._capture = wrap("capture", runner._capture)
    runner._bind_prefix = wrap("bind", runner._bind_prefix)
    runner._replay = wrap("replay", runner._replay)
    return counters


def counter_delta(
    before: dict[str, int],
    after: dict[str, int],
) -> dict[str, int]:
    return {name: after[name] - before[name] for name in before}


def apply_variant(policy: BeingHPolicy, variant: Variant) -> None:
    policy.model.enable_static_prefix_cache = variant.static_prefix_cache
    policy.model.enable_npu_fusion_attention = variant.npu_fusion_attention
    policy.model.npu_single_sample_fast_path = "off"
    policy.model.enable_npu_capture_replay = variant.npu_capture_replay


def summarize_comparisons(
    samples: list[dict[str, Any]],
    comparison_name: str,
) -> dict[str, Any]:
    rows = [sample["comparisons"][comparison_name] for sample in samples]
    return {
        "all_finite": all(
            row["left_finite"] and row["right_finite"] for row in rows
        ),
        "all_exact_equal": all(row["exact_equal"] for row in rows),
        "all_rng_before_equal": all(row["rng_before_equal"] for row in rows),
        "all_rng_after_equal": all(row["rng_after_equal"] for row in rows),
        "max_absolute_error": max(row["max_absolute_error"] for row in rows),
        "mean_absolute_error": float(
            np.mean([row["mean_absolute_error"] for row in rows])
        ),
        "max_relative_error": max(row["max_relative_error"] for row in rows),
        "mean_relative_error": float(
            np.mean([row["mean_relative_error"] for row in rows])
        ),
        "min_cosine_similarity": min(
            row["cosine_similarity"] for row in rows
        ),
        "all_output_hash_equal": all(
            row["left_hash"] == row["right_hash"] for row in rows
        ),
    }


def git_value(repo_root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *arguments],
        text=True,
    ).strip()


def main() -> None:
    args = parse_args()
    torch.npu.set_device(args.device)
    set_seed(SEEDS[0])
    policy = create_policy(args.model_path, args.device)
    observation = build_observation()
    packed_inputs = prepare_model_inputs(policy, observation)

    runner = policy.model._get_npu_action_suffix_graph_runner()
    graph_counts = install_graph_counters(runner)
    samples: list[dict[str, Any]] = []

    for seed in SEEDS:
        variant_outputs: dict[str, torch.Tensor] = {}
        variant_rows: dict[str, dict[str, Any]] = {}
        for variant in VARIANTS:
            apply_variant(policy, variant)
            set_seed(seed)
            rng_before = rng_hashes()
            counts_before = dict(graph_counts)
            output = policy.model.get_action(**packed_inputs)["action_pred"]
            torch.npu.synchronize()
            rng_after = rng_hashes()
            counts_after = dict(graph_counts)

            variant_outputs[variant.name] = output.detach().clone()
            graph_delta = counter_delta(counts_before, counts_after)
            variant_rows[variant.name] = {
                "configuration": variant.as_dict(),
                "shape": list(output.shape),
                "finite": bool(torch.isfinite(output).all().item()),
                "hash": tensor_hash(output),
                "rng_before": rng_before,
                "rng_after": rng_after,
                "graph_counts": graph_delta,
                "fallback_reason": (
                    runner.last_fallback_reason
                    if variant.npu_capture_replay
                    else None
                ),
                "failure_count": len(runner.failure_reasons),
                "entry_count": runner.entry_count,
                "unhealthy": runner.unhealthy,
                "unhealthy_reason": runner.unhealthy_reason,
            }

        comparisons: dict[str, dict[str, Any]] = {}
        for name, left_name, right_name in COMPARISONS:
            comparison = compare_tensors(
                variant_outputs[left_name],
                variant_outputs[right_name],
            )
            comparison.update(
                {
                    "left": left_name,
                    "right": right_name,
                    "rng_before_equal": (
                        variant_rows[left_name]["rng_before"]
                        == variant_rows[right_name]["rng_before"]
                    ),
                    "rng_after_equal": (
                        variant_rows[left_name]["rng_after"]
                        == variant_rows[right_name]["rng_after"]
                    ),
                }
            )
            comparisons[name] = comparison

        sample = {
            "seed": seed,
            "variants": variant_rows,
            "comparisons": comparisons,
        }
        samples.append(sample)
        print("LADDER_NUMERIC_SAMPLE " + json.dumps(sample, sort_keys=True))

    comparison_summary = {
        name: summarize_comparisons(samples, name)
        for name, _, _ in COMPARISONS
    }
    expected_graph_counts = {"capture": 1, "bind": 4, "replay": 40}
    fallback_reasons = [
        sample["variants"]["D"]["fallback_reason"]
        for sample in samples
        if sample["variants"]["D"]["fallback_reason"] is not None
    ]
    validation_errors = []
    if graph_counts != expected_graph_counts:
        validation_errors.append(
            f"graph counts {graph_counts} != {expected_graph_counts}"
        )
    if fallback_reasons:
        validation_errors.append(f"graph fallbacks observed: {fallback_reasons}")
    if runner.failure_reasons:
        validation_errors.append(
            f"graph failures observed: {runner.failure_reasons}"
        )
    if runner.entry_count != 1:
        validation_errors.append(
            f"final graph entry count {runner.entry_count} != 1"
        )
    if runner.unhealthy:
        validation_errors.append(
            f"graph runner unhealthy: {runner.unhealthy_reason}"
        )
    for sample_index, sample in enumerate(samples):
        for variant_name in ("A", "B", "C"):
            counts = sample["variants"][variant_name]["graph_counts"]
            if counts != {"capture": 0, "bind": 0, "replay": 0}:
                validation_errors.append(
                    f"seed {sample['seed']} variant {variant_name} "
                    f"unexpected graph counts: {counts}"
                )
        expected_d_counts = (
            {"capture": 1, "bind": 0, "replay": 8}
            if sample_index == 0
            else {"capture": 0, "bind": 1, "replay": 8}
        )
        actual_d_counts = sample["variants"]["D"]["graph_counts"]
        if actual_d_counts != expected_d_counts:
            validation_errors.append(
                f"seed {sample['seed']} variant D graph counts "
                f"{actual_d_counts} != {expected_d_counts}"
            )
        if sample["variants"]["D"]["entry_count"] != 1:
            validation_errors.append(
                f"seed {sample['seed']} variant D did not retain one graph entry"
            )

    for name, summary in comparison_summary.items():
        if not summary["all_finite"]:
            validation_errors.append(f"{name} contains non-finite output")
        if not summary["all_rng_before_equal"]:
            validation_errors.append(f"{name} RNG-before states differ")
        if not summary["all_rng_after_equal"]:
            validation_errors.append(f"{name} RNG-after states differ")
    for exact_name in ("A-B", "C-D"):
        if not comparison_summary[exact_name]["all_exact_equal"]:
            validation_errors.append(
                f"{exact_name} expected exact output equality"
            )
    for e1_name in ("B-C", "A-D"):
        observed_cosine = comparison_summary[e1_name][
            "min_cosine_similarity"
        ]
        if observed_cosine < MIN_E1_COSINE_SIMILARITY:
            validation_errors.append(
                f"{e1_name} min cosine {observed_cosine} < "
                f"{MIN_E1_COSINE_SIMILARITY}"
            )

    repo_root = Path(__file__).resolve().parent.parent
    result = {
        "experiment": "current_head_unified_optimization_ladder",
        "model_path": str(Path(args.model_path).resolve()),
        "device": args.device,
        "seeds": list(SEEDS),
        "thresholds": {
            "B-C_min_cosine_similarity": MIN_E1_COSINE_SIMILARITY,
            "A-D_min_cosine_similarity": MIN_E1_COSINE_SIMILARITY,
        },
        "variants": {variant.name: variant.as_dict() for variant in VARIANTS},
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device_name": torch.npu.get_device_name(args.device),
            "commit": git_value(repo_root, "rev-parse", "HEAD"),
        },
        "samples": samples,
        "summary": {
            "comparisons": comparison_summary,
            "graph_counts": dict(graph_counts),
            "expected_graph_counts": expected_graph_counts,
            "fallback_reasons": fallback_reasons,
            "final_failure_count": len(runner.failure_reasons),
            "final_entry_count": runner.entry_count,
            "runner_unhealthy": runner.unhealthy,
            "validation_errors": validation_errors,
        },
    }
    numeric_metrics = [
        metric
        for summary in comparison_summary.values()
        for metric in (
            summary["max_absolute_error"],
            summary["mean_absolute_error"],
            summary["max_relative_error"],
            summary["mean_relative_error"],
            summary["min_cosine_similarity"],
        )
    ]
    if not all(math.isfinite(value) for value in numeric_metrics):
        validation_errors.append("non-finite comparison metric")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("LADDER_NUMERIC_SUMMARY " + json.dumps(result["summary"], sort_keys=True))
    print(f"LADDER_NUMERIC_OUTPUT {output_path}")
    if validation_errors:
        raise RuntimeError("; ".join(validation_errors))


if __name__ == "__main__":
    main()
