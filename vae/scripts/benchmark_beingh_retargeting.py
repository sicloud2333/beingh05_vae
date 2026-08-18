from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = REPO_ROOT / "vae/examples/beingh_shadow_grasp_eval.py"
DEFAULT_DATASET = (
    REPO_ROOT
    / "vae/evaluation/object_episodes/"
    "shadow_grasp_bottle22249179_aug100_2cam.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the VAE and geometry-retargeting Being-H zero-shot "
            "inference chains under matched closed-loop settings."
        )
    )
    parser.add_argument("--vae-model-path", type=Path, required=True)
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=REPO_ROOT / "vae/checkpoints/native_n2_epoch800_inference.pt",
        help=(
            "NativeVAE inference checkpoint used to decode the VAE chain. "
            "Pass the checkpoint paired with --vae-model-path for formal runs."
        ),
    )
    parser.add_argument("--geometry-model-path", type=Path, required=True)
    parser.add_argument(
        "--joint-model-path", type=Path, required=True,
        help="Matching-hand native-joint baseline checkpoint.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--hand",
        choices=("sharpa_hand_right", "gaia_hand_right"),
        default="sharpa_hand_right",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--observation-mode",
        choices=("encoded", "commanded"),
        default="commanded",
        help="Use the same hand-state feedback mode for both chains.",
    )
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmup-queries", type=int, default=2)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--order",
        choices=("vae-first", "geometry-first", "alternate"),
        default="alternate",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--geometry-max-iterations", type=int, default=12)
    parser.add_argument(
        "--geometry-retargeting-profile",
        choices=("raw", "stable"),
        default="raw",
    )
    parser.add_argument(
        "--native-joint-rate-limit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable the same target native-q limiter in both runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/benchmark_retargeting",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print matched evaluation commands without loading either model.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.episode < 0:
        raise ValueError("--episode must be non-negative")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.warmup_queries < 0:
        raise ValueError("--warmup-queries must be non-negative")
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.geometry_max_iterations <= 0:
        raise ValueError("--geometry-max-iterations must be positive")
    if not args.python.is_file():
        raise FileNotFoundError(f"Python executable not found: {args.python}")
    if not args.dataset.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {args.dataset}")
    if not args.vae_checkpoint.is_file():
        raise FileNotFoundError(
            f"NativeVAE checkpoint not found: {args.vae_checkpoint}"
        )
    for label, path in (
        ("VAE model", args.vae_model_path),
        ("geometry model", args.geometry_model_path),
        ("native-joint model", args.joint_model_path),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} checkpoint not found: {path}")


def percentile_summary(values: np.ndarray) -> dict[str, float | int | None]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data)]
    if not len(data):
        return {
            "count": 0,
            "total_s": 0.0,
            "mean_s": None,
            "p50_s": None,
            "p95_s": None,
            "min_s": None,
            "max_s": None,
        }
    return {
        "count": int(len(data)),
        "total_s": float(data.sum()),
        "mean_s": float(data.mean()),
        "p50_s": float(np.quantile(data, 0.50)),
        "p95_s": float(np.quantile(data, 0.95)),
        "min_s": float(data.min()),
        "max_s": float(data.max()),
    }


def build_eval_command(
    args: argparse.Namespace,
    chain: str,
    output_directory: Path,
) -> list[str]:
    if chain not in {"vae", "geometry", "joint"}:
        raise ValueError(f"Unknown chain: {chain}")
    model_path = {
        "vae": args.vae_model_path,
        "geometry": args.geometry_model_path,
        "joint": args.joint_model_path,
    }[chain]
    command = [
        # Preserve virtual-environment launcher symlinks. Resolving them can
        # escape the venv and run the base interpreter without its packages.
        str(args.python.absolute()),
        "-u",
        str(EVAL_SCRIPT),
        "--deployment-profile",
        "safe_smooth",
        "--model-path",
        str(model_path.resolve()),
        "--dataset",
        str(args.dataset.resolve()),
        "--episode",
        str(args.episode),
        "--hand",
        args.hand,
        "--latent-observation-mode",
        args.observation_mode,
        "--action-selection",
        "chunk",
        "--inference-mode",
        "sync",
        "--replan-every",
        "16",
        "--noise-mode",
        "fixed_per_query",
        "--max-steps",
        str(args.steps),
        "--warmup-queries",
        str(args.warmup_queries),
        "--safety-mode",
        "off",
        "--success-profile",
        "loose",
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--no-record-images",
        "--output",
        str(output_directory / "rollout.npz"),
        "--output-metadata",
        str(output_directory / "metadata.json"),
    ]
    command.extend(
        ["--native-joint-rate-limit"]
        if args.native_joint_rate_limit
        else ["--no-native-joint-rate-limit"]
    )
    if chain in {"vae", "joint"}:
        command.extend(["--joint-retargeting", "none"])
    else:
        command.extend(
            [
                "--joint-retargeting",
                "geometry",
                "--geometry-retargeting-profile",
                args.geometry_retargeting_profile,
                "--geometry-action-chunk-mode",
                "batch",
                "--geometry-max-iterations",
                str(args.geometry_max_iterations),
            ]
        )
    if chain == "vae":
        command.extend(
            ["--vae-checkpoint", str(args.vae_checkpoint.resolve())]
        )
    return command


def run_command(command: Sequence[str], cwd: Path) -> float:
    print("\n$ " + shlex.join(command), flush=True)
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    started_at = time.perf_counter()
    subprocess.run(command, cwd=cwd, env=environment, check=True)
    return time.perf_counter() - started_at


def load_run_metrics(
    chain: str,
    output_directory: Path,
    process_wall_s: float,
) -> dict[str, Any]:
    rollout_path = output_directory / "rollout.npz"
    metadata_path = output_directory / "metadata.json"
    if not rollout_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing benchmark outputs below {output_directory}"
        )
    rollout = np.load(rollout_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    query_latency = np.asarray(
        rollout["beingh_query_latency_s"], dtype=np.float64
    )
    policy_step_latency = np.asarray(
        rollout["closed_loop_policy_step_latency_s"], dtype=np.float64
    )
    query_indices = np.asarray(
        rollout["query_step_indices"], dtype=np.int64
    )
    valid_query_indices = query_indices[
        (query_indices >= 0) & (query_indices < len(policy_step_latency))
    ]
    query_step_latency = policy_step_latency[valid_query_indices]
    query_mask = np.zeros(len(policy_step_latency), dtype=bool)
    query_mask[valid_query_indices] = True
    non_query_step_latency = policy_step_latency[~query_mask]
    paired = min(len(query_step_latency), len(query_latency))
    query_step_non_model = (
        query_step_latency[:paired] - query_latency[:paired]
    )
    chain_non_model_total = float(
        policy_step_latency.sum() - query_latency.sum()
    )

    result: dict[str, Any] = {
        "chain": chain,
        "process_wall_s": float(process_wall_s),
        "steps": int(len(policy_step_latency)),
        "queries": int(len(query_latency)),
        "beingh_query": percentile_summary(query_latency),
        "end_to_end_policy_step": percentile_summary(policy_step_latency),
        "query_policy_step": percentile_summary(query_step_latency),
        "non_query_policy_step": percentile_summary(non_query_step_latency),
        "query_step_non_model_overhead": percentile_summary(
            query_step_non_model
        ),
        "non_model_chain_overhead_total_s": chain_non_model_total,
        "non_model_chain_overhead_per_step_s": (
            chain_non_model_total / len(policy_step_latency)
            if len(policy_step_latency)
            else None
        ),
        "amortized_policy_steps_per_s": (
            float(len(policy_step_latency) / policy_step_latency.sum())
            if policy_step_latency.sum() > 0
            else None
        ),
        "beingh_queries_per_s": (
            float(len(query_latency) / query_latency.sum())
            if query_latency.sum() > 0
            else None
        ),
        "success": bool(metadata.get("success", False)),
        "max_lift_m": float(metadata.get("max_lift_m", 0.0)),
        "rollout": str(rollout_path.resolve()),
        "metadata": str(metadata_path.resolve()),
    }
    if chain == "geometry":
        retarget = metadata.get("geometry_retargeting") or {}
        result["geometry_retargeting"] = {
            "observation": retarget.get("observation"),
            "action": retarget.get("action"),
            "action_chunk_mode": retarget.get("action_chunk_mode"),
            "observation_mode": retarget.get("observation_mode"),
        }
    return result


def aggregate_metric(
    runs: list[dict[str, Any]],
    section: str,
    key: str = "mean_s",
) -> float | None:
    values = [run.get(section, {}).get(key) for run in runs]
    finite = [float(value) for value in values if value is not None]
    return float(np.mean(finite)) if finite else None


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "trials": len(runs),
        "beingh_query_mean_s": aggregate_metric(runs, "beingh_query"),
        "query_policy_step_mean_s": aggregate_metric(
            runs, "query_policy_step"
        ),
        "non_query_policy_step_mean_s": aggregate_metric(
            runs, "non_query_policy_step"
        ),
        "end_to_end_policy_step_mean_s": aggregate_metric(
            runs, "end_to_end_policy_step"
        ),
        "query_step_non_model_overhead_mean_s": aggregate_metric(
            runs, "query_step_non_model_overhead"
        ),
        "non_model_chain_overhead_per_step_s": float(
            np.mean(
                [
                    run["non_model_chain_overhead_per_step_s"]
                    for run in runs
                    if run["non_model_chain_overhead_per_step_s"] is not None
                ]
            )
        ),
        "amortized_policy_steps_per_s": float(
            np.mean(
                [
                    run["amortized_policy_steps_per_s"]
                    for run in runs
                    if run["amortized_policy_steps_per_s"] is not None
                ]
            )
        ),
        "process_wall_s": float(
            np.mean([run["process_wall_s"] for run in runs])
        ),
    }


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator / denominator)


def comparison(aggregate: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    vae = aggregate["vae"]
    result: dict[str, float | None] = {}
    for chain in ("geometry", "joint"):
        other = aggregate[chain]
        label = "geometry" if chain == "geometry" else "joint"
        result[f"{label}_speedup_beingh_query_x"] = safe_ratio(
            vae["beingh_query_mean_s"], other["beingh_query_mean_s"]
        )
        result[f"{label}_speedup_query_policy_step_x"] = safe_ratio(
            vae["query_policy_step_mean_s"], other["query_policy_step_mean_s"]
        )
        result[f"{label}_speedup_non_query_policy_step_x"] = safe_ratio(
            vae["non_query_policy_step_mean_s"], other["non_query_policy_step_mean_s"]
        )
        result[f"{label}_speedup_amortized_policy_step_x"] = safe_ratio(
            vae["end_to_end_policy_step_mean_s"], other["end_to_end_policy_step_mean_s"]
        )
        result[f"{label}_vs_vae_policy_throughput_x"] = safe_ratio(
            other["amortized_policy_steps_per_s"], vae["amortized_policy_steps_per_s"]
        )
    return result


def milliseconds(value: float | None) -> str:
    return "n/a" if value is None else f"{1000.0 * value:.2f}"


def print_report(aggregate: dict[str, dict[str, Any]], ratios: dict[str, float | None]) -> None:
    rows = (
        ("Being-H query mean (ms)", "beingh_query_mean_s"),
        ("Query policy step mean (ms)", "query_policy_step_mean_s"),
        ("Non-query policy step mean (ms)", "non_query_policy_step_mean_s"),
        ("Amortized policy step mean (ms)", "end_to_end_policy_step_mean_s"),
        ("Non-model overhead / step (ms)", "non_model_chain_overhead_per_step_s"),
    )
    labels = (("vae", "VAE"), ("geometry", "geometry"), ("joint", "native-joint"))
    print("\n=== Being-H three-chain inference benchmark ===")
    print(f"{'metric':38s} {'VAE':>12s} {'geometry':>12s} {'native-joint':>14s}")
    for label, key in rows:
        values = [milliseconds(aggregate[name].get(key)) for name, _ in labels]
        print(f"{label:38s} {values[0]:>12s} {values[1]:>12s} {values[2]:>14s}")
    print(f"{'Amortized policy steps/s':38s} " + " ".join(
        f"{aggregate[name]['amortized_policy_steps_per_s']:>{12 if name != 'joint' else 14}.2f}"
        for name, _ in labels
    ))
    print("\nSpeedup ratios (>1 means the other chain is faster than VAE):")
    for key, value in ratios.items():
        print(f"  {key}: {'n/a' if value is None else f'{value:.3f}x'}")


def trial_order(args: argparse.Namespace, trial: int) -> tuple[str, ...]:
    orders = (
        ("vae", "geometry", "joint"),
        ("geometry", "joint", "vae"),
        ("joint", "vae", "geometry"),
    )
    if args.order == "vae-first":
        return orders[trial % len(orders)]
    if args.order == "geometry-first":
        return ("geometry", "vae", "joint") if trial % 2 == 0 else ("geometry", "joint", "vae")
    return orders[trial % len(orders)]


def main() -> None:
    args = parse_args()
    validate_args(args)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or (
        f"{timestamp}_{args.hand}_episode-{args.episode:06d}_"
        f"{args.observation_mode}"
    )
    benchmark_root = args.output_dir.expanduser().resolve() / run_name
    all_runs: dict[str, list[dict[str, Any]]] = {"vae": [], "geometry": [], "joint": []}
    commands: list[dict[str, Any]] = []

    for trial in range(args.trials):
        for chain in trial_order(args, trial):
            output_directory = benchmark_root / f"trial_{trial:02d}" / chain
            command = build_eval_command(args, chain, output_directory)
            commands.append(
                {"trial": trial, "chain": chain, "command": command}
            )
            if args.dry_run:
                print(f"\n[{chain} trial {trial}]\n$ {shlex.join(command)}")
                continue
            output_directory.mkdir(parents=True, exist_ok=True)
            process_wall_s = run_command(command, REPO_ROOT)
            all_runs[chain].append(
                load_run_metrics(chain, output_directory, process_wall_s)
            )

    if args.dry_run:
        return

    aggregate = {
        chain: aggregate_runs(runs) for chain, runs in all_runs.items()
    }
    ratios = comparison(aggregate)
    report = {
        "configuration": {
            "vae_model_path": str(args.vae_model_path.resolve()),
            "vae_checkpoint": str(args.vae_checkpoint.resolve()),
            "geometry_model_path": str(args.geometry_model_path.resolve()),
            "joint_model_path": str(args.joint_model_path.resolve()),
            "dataset": str(args.dataset.resolve()),
            "hand": args.hand,
            "episode": args.episode,
            "observation_mode": args.observation_mode,
            "steps": args.steps,
            "warmup_queries": args.warmup_queries,
            "trials": args.trials,
            "order": args.order,
            "device": args.device,
            "seed": args.seed,
            "native_joint_rate_limit": args.native_joint_rate_limit,
            "geometry_retargeting_profile": (
                args.geometry_retargeting_profile
            ),
            "geometry_max_iterations": args.geometry_max_iterations,
        },
        "commands": commands,
        "runs": all_runs,
        "aggregate": aggregate,
        "comparison": ratios,
    }
    benchmark_root.mkdir(parents=True, exist_ok=True)
    summary_path = benchmark_root / "summary.json"
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print_report(aggregate, ratios)
    print(f"\nSaved benchmark summary to {summary_path}")


if __name__ == "__main__":
    main()
