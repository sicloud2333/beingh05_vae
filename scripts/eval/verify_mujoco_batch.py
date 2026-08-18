#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a multi-episode Being-H MuJoCo evaluation summary."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--episode-start", type=int, required=True)
    parser.add_argument("--episode-end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require_file(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is missing")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"{label} is empty: {path}")
    return {"path": str(path.resolve()), "size_bytes": int(size)}


def require_finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def verify_episode(item: dict[str, Any]) -> dict[str, Any]:
    episode = int(item["episode"])
    steps = int(item["steps"])
    queries = int(item["num_queries"])
    if steps <= 0 or queries <= 0:
        raise ValueError(
            f"Episode {episode} has invalid steps/queries: {steps}/{queries}"
        )

    artifacts = {
        "rollout": require_file(item.get("rollout"), f"episode {episode} rollout"),
        "metadata": require_file(item.get("metadata"), f"episode {episode} metadata"),
        "video": require_file(item.get("video"), f"episode {episode} primary video"),
    }
    videos = item.get("videos")
    if not isinstance(videos, dict):
        raise ValueError(f"Episode {episode} videos mapping is missing")
    required_views = ("ego_opposite", "wrist")
    artifacts["videos"] = {
        view: require_file(videos.get(view), f"episode {episode} {view} video")
        for view in required_views
    }

    require_finite(item.get("max_lift_m"), f"episode {episode} max_lift_m")
    timing = item.get("timing")
    if not isinstance(timing, dict):
        raise ValueError(f"Episode {episode} timing section is missing")
    beingh_query = timing.get("beingh_query")
    policy_step = timing.get("closed_loop_policy_step")
    if not isinstance(beingh_query, dict) or not isinstance(policy_step, dict):
        raise ValueError(f"Episode {episode} timing subsections are missing")
    if int(beingh_query.get("count", -1)) != queries:
        raise ValueError(
            f"Episode {episode} query timing count does not match num_queries"
        )
    if int(policy_step.get("count", -1)) != steps:
        raise ValueError(
            f"Episode {episode} policy-step timing count does not match steps"
        )
    for section_name, section in (
        ("beingh_query", beingh_query),
        ("closed_loop_policy_step", policy_step),
    ):
        for key in ("total_s", "mean_ms", "p50_ms", "p95_ms", "max_ms"):
            require_finite(
                section.get(key), f"episode {episode} {section_name}.{key}"
            )

    return {
        "episode": episode,
        "success": bool(item["success"]),
        "steps": steps,
        "queries": queries,
        "artifacts": artifacts,
    }


def main() -> None:
    args = parse_args()
    if args.episode_start < 0 or args.episode_end < args.episode_start:
        raise ValueError("Expected 0 <= episode-start <= episode-end")
    summary_path = args.summary.expanduser().resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    expected = list(range(args.episode_start, args.episode_end + 1))
    actual = [int(value) for value in summary.get("episode_indices", [])]
    if actual != expected:
        raise ValueError(f"Episode indices mismatch: expected {expected}, got {actual}")
    if int(summary.get("num_episodes", -1)) != len(expected):
        raise ValueError("num_episodes does not match the requested range")

    episode_items = summary.get("episodes")
    if not isinstance(episode_items, list) or len(episode_items) != len(expected):
        raise ValueError("Summary episodes list is missing or incomplete")
    verified = [verify_episode(item) for item in episode_items]
    if [item["episode"] for item in verified] != expected:
        raise ValueError("Embedded episode metadata is out of order or incomplete")

    num_successes = sum(item["success"] for item in verified)
    num_failures = len(verified) - num_successes
    total_steps = sum(item["steps"] for item in verified)
    total_queries = sum(item["queries"] for item in verified)
    expected_aggregates = {
        "num_successes": num_successes,
        "num_failures": num_failures,
        "total_steps": total_steps,
        "total_queries": total_queries,
    }
    for key, value in expected_aggregates.items():
        if int(summary.get(key, -1)) != value:
            raise ValueError(
                f"Aggregate {key} mismatch: summary={summary.get(key)!r}, "
                f"verified={value}"
            )
    expected_rate = num_successes / len(verified)
    if not math.isclose(
        require_finite(summary.get("success_rate"), "success_rate"),
        expected_rate,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("success_rate does not match per-episode results")

    report = {
        "summary": str(summary_path),
        "episode_indices": expected,
        **expected_aggregates,
        "success_rate": expected_rate,
        "episodes": verified,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Verified MuJoCo batch report: {output}")


if __name__ == "__main__":
    main()
