#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from eval_shadow_grasp_dataset_open_loop import (
    ROT6D_DATA_CONFIG_NAMES,
    compute_action_metrics,
    summarize_query_timing,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and independently aggregate a multi-episode open-loop run."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--plots-root", type=Path, required=True)
    parser.add_argument("--episode-start", type=int, required=True)
    parser.add_argument("--episode-end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require_file(path: Path, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"{label} is empty: {path}")
    return {"path": str(path), "size_bytes": int(size)}


def assert_equivalent(expected: Any, actual: Any, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise ValueError(f"{label} dictionary keys differ")
        for key in expected:
            assert_equivalent(expected[key], actual[key], f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"{label} list shape differs")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            assert_equivalent(left, right, f"{label}[{index}]")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        left = float(expected)
        right = float(actual)
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError(f"{label} is non-finite")
        if not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-10):
            raise ValueError(f"{label} differs: recomputed={left}, summary={right}")
        return
    if expected != actual:
        raise ValueError(f"{label} differs: {expected!r} != {actual!r}")


def main() -> None:
    args = parse_args()
    if args.episode_start < 0 or args.episode_end < args.episode_start:
        raise ValueError("Expected 0 <= episode-start <= episode-end")
    summary_path = args.summary.expanduser().resolve()
    require_file(summary_path, "open-loop summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_indices = list(range(args.episode_start, args.episode_end + 1))
    actual_indices = [int(value) for value in summary.get("episode_indices", [])]
    if actual_indices != expected_indices:
        raise ValueError(
            f"Episode indices mismatch: {actual_indices} != {expected_indices}"
        )
    if int(summary.get("num_episodes", -1)) != len(expected_indices):
        raise ValueError("num_episodes does not match requested episode range")
    summary_episodes = summary.get("episodes")
    if not isinstance(summary_episodes, list) or len(summary_episodes) != len(
        expected_indices
    ):
        raise ValueError("Summary episode metrics are missing or incomplete")

    predictions_root = args.predictions_root.expanduser().resolve()
    plots_root = args.plots_root.expanduser().resolve()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    all_latencies: list[float] = []
    total_predicted_actions = 0
    verified_episodes: list[dict[str, Any]] = []

    for episode, summary_metrics in zip(
        expected_indices, summary_episodes, strict=True
    ):
        directory = predictions_root / f"episode_{episode:06d}"
        predictions_path = directory / "predictions.npz"
        metrics_path = directory / "metrics.json"
        plot_path = plots_root / f"episode_{episode:06d}" / "curves.png"
        artifacts = {
            "predictions": require_file(
                predictions_path, f"episode {episode} predictions"
            ),
            "metrics": require_file(metrics_path, f"episode {episode} metrics"),
            "plot": require_file(plot_path, f"episode {episode} plot"),
        }
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert_equivalent(metrics, summary_metrics, f"episode {episode} metrics")
        if int(metrics.get("episode_index", -1)) != episode:
            raise ValueError(f"Episode {episode} metrics index is invalid")

        with np.load(predictions_path, allow_pickle=False) as payload:
            prediction = np.asarray(payload["predicted_actions"], dtype=np.float32)
            target = np.asarray(payload["ground_truth_actions"], dtype=np.float32)
            states = np.asarray(payload["states"], dtype=np.float32)
            frame_indices = np.asarray(payload["frame_indices"], dtype=np.int64)
            query_indices = np.asarray(payload["query_indices"], dtype=np.int64)
            chunks = np.asarray(payload["predicted_chunks"], dtype=np.float32)
        if (
            prediction.shape != target.shape
            or prediction.ndim != 2
            or prediction.shape[1] != 30
            or states.shape != prediction.shape
            or frame_indices.shape != (len(prediction),)
            or chunks.ndim != 3
            or chunks.shape[0] != len(query_indices)
            or chunks.shape[2] != 30
        ):
            raise ValueError(f"Episode {episode} prediction artifact shapes are invalid")
        if not all(
            np.isfinite(values).all()
            for values in (prediction, target, states, chunks)
        ):
            raise ValueError(f"Episode {episode} prediction artifact is non-finite")
        if int(metrics.get("num_evaluated_actions", -1)) != len(prediction):
            raise ValueError(f"Episode {episode} action count does not match metrics")
        if int(metrics.get("num_queries", -1)) != len(query_indices):
            raise ValueError(f"Episode {episode} query count does not match metrics")

        query_timing = metrics.get("query_timing")
        if query_timing is not None:
            latencies = [float(value) for value in query_timing["latencies_seconds"]]
            if len(latencies) != len(query_indices):
                raise ValueError(f"Episode {episode} timing/query counts differ")
            all_latencies.extend(latencies)
            total_predicted_actions += int(query_timing["num_predicted_actions"])
        predictions.append(prediction)
        targets.append(target)
        verified_episodes.append(
            {
                "episode": episode,
                "actions": int(len(prediction)),
                "queries": int(len(query_indices)),
                "artifacts": artifacts,
            }
        )

    all_predictions = np.concatenate(predictions, axis=0)
    all_targets = np.concatenate(targets, axis=0)
    if int(summary.get("num_evaluated_actions", -1)) != len(all_predictions):
        raise ValueError("Summary action count does not match artifacts")
    recomputed_metrics = compute_action_metrics(
        all_predictions,
        all_targets,
        include_xyz_equivalent=(
            summary_episodes[0].get("data_config_name")
            in ROT6D_DATA_CONFIG_NAMES
        ),
    )
    assert_equivalent(
        recomputed_metrics,
        summary.get("aggregate_metrics"),
        "aggregate_metrics",
    )

    recomputed_timing = None
    if all_latencies:
        recomputed_timing = summarize_query_timing(
            all_latencies,
            num_predicted_actions=total_predicted_actions,
            num_executed_actions=len(all_predictions),
            include_cold_warm_split=True,
        )
        assert_equivalent(
            recomputed_timing,
            summary.get("aggregate_query_timing"),
            "aggregate_query_timing",
        )

    report = {
        "summary": str(summary_path),
        "episode_indices": expected_indices,
        "num_episodes": len(expected_indices),
        "num_evaluated_actions": int(len(all_predictions)),
        "aggregate_metrics_recomputed": True,
        "aggregate_query_timing_recomputed": recomputed_timing is not None,
        "aggregate_metrics": recomputed_metrics,
        "aggregate_query_timing": recomputed_timing,
        "episodes": verified_episodes,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Verified open-loop batch report: {output}")


if __name__ == "__main__":
    main()
