#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.object_episode import grounded_object_position  # noqa: E402


def episode_xy(payload: np.lib.npyio.NpzFile, index: int) -> np.ndarray:
    values = np.asarray(payload["object_world_xy"], dtype=np.float64)
    return values.copy() if values.shape == (2,) else values[index].copy()


def matrix_to_quaternion(rotation: np.ndarray) -> list[float]:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.asarray(rotation).reshape(9))
    return [float(value) for value in quaternion]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export portable object poses from a LeRobot dataset."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source_dataset", type=Path, default=None)
    args = parser.parse_args()

    metadata = json.loads(
        (args.dataset / "collection_metadata.json").read_text(encoding="utf-8")
    )
    source_dataset = args.source_dataset or Path(metadata["source_dataset"])
    if not source_dataset.is_absolute():
        source_dataset = args.dataset / source_dataset
    rows = list(
        csv.DictReader(
            (args.dataset / "episode_manifest.csv").open(
                "r",
                encoding="utf-8",
                newline="",
            )
        )
    )
    pose_cache: dict[tuple[str, float, bytes], tuple[float, list[float]]] = {}
    records: list[dict[str, object]] = []
    with np.load(source_dataset, allow_pickle=True) as payload:
        for row in rows:
            source_index = int(row["input_episode_index"])
            object_id = str(payload["object_id"][source_index])
            scale = float(payload["object_scale"][source_index])
            rotation = np.asarray(
                payload["object_rotmat"][source_index],
                dtype=np.float64,
            ).reshape(3, 3)
            xy = episode_xy(payload, source_index)
            key = (
                object_id,
                round(scale, 9),
                np.round(rotation, decimals=8).tobytes(),
            )
            if key not in pose_cache:
                base_position = grounded_object_position(
                    object_id,
                    scale,
                    rotation,
                    np.zeros(2),
                )
                pose_cache[key] = (
                    float(base_position[2]),
                    matrix_to_quaternion(rotation),
                )
            z, quaternion = pose_cache[key]
            records.append(
                {
                    "episode_index": int(row["lerobot_episode_index"]),
                    "source_episode_index": source_index,
                    "original_episode_index": int(row["original_episode_idx"]),
                    "scene_index": int(row["scene_index"]),
                    "object_id": object_id,
                    "category": row["category"],
                    "task": row["task"],
                    "scale": scale,
                    "position": [float(xy[0]), float(xy[1]), z],
                    "quaternion_wxyz": quaternion,
                    "frames": int(row["frames"]),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")
    print(
        f"[done] wrote {len(records)} episodes and {len(pose_cache)} base poses "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
