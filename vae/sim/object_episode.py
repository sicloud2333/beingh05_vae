from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from .scene import resolve_object_assets


@dataclass(frozen=True)
class DatasetObjectEpisode:
    object_id: str
    scale: float
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    source_dataset: Path
    source_episode_index: int
    lerobot_episode_index: int | None = None


def _episode_xy(payload: np.lib.npyio.NpzFile, index: int) -> np.ndarray:
    values = np.asarray(payload["object_world_xy"], dtype=np.float64)
    if values.shape == (2,):
        return values.copy()
    if values.ndim == 2 and values.shape[1] == 2:
        return values[index].copy()
    raise ValueError(f"Unsupported object_world_xy shape: {values.shape}")


def _obj_vertices(path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if not line.startswith("v "):
                continue
            values = line.split()
            if len(values) >= 4:
                vertices.append([float(values[1]), float(values[2]), float(values[3])])
    if not vertices:
        raise ValueError(f"No vertices found in {path}")
    return np.asarray(vertices, dtype=np.float64)


def grounded_object_position(
    object_id: str,
    scale: float,
    rotation: np.ndarray,
    xy: np.ndarray,
) -> np.ndarray:
    _, collision_meshes = resolve_object_assets(object_id)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    min_z = np.inf
    for mesh_path in collision_meshes:
        vertices = _obj_vertices(mesh_path) * float(scale)
        rotated = (rotation @ vertices.T).T
        min_z = min(min_z, float(rotated[:, 2].min()))
    if not np.isfinite(min_z):
        raise RuntimeError(f"Could not compute mesh bounds for {object_id}")
    xy = np.asarray(xy, dtype=np.float64).reshape(2)
    return np.asarray([xy[0], xy[1], -min_z], dtype=np.float64)


def _matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(
        quaternion,
        np.asarray(rotation, dtype=np.float64).reshape(9),
    )
    return quaternion


def _resolve_source_episode(
    dataset: Path,
    episode_index: int,
    source_dataset: Path | None,
) -> tuple[Path, int, int | None]:
    if dataset.is_file():
        if dataset.suffix != ".npz":
            raise ValueError("A dataset file must be an NPZ source dataset.")
        return dataset.resolve(), episode_index, None

    manifest_path = dataset / "episode_manifest.csv"
    metadata_path = dataset / "collection_metadata.json"
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"{dataset} is not a LeRobot dataset with episode_manifest.csv "
            "and collection_metadata.json"
        )

    selected: dict[str, str] | None = None
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if int(row["lerobot_episode_index"]) == episode_index:
                selected = row
                break
    if selected is None:
        raise IndexError(f"LeRobot episode {episode_index} is not in {manifest_path}")

    if source_dataset is None:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_dataset = Path(metadata["source_dataset"])
        if not source_dataset.is_absolute():
            source_dataset = dataset / source_dataset
    if not source_dataset.is_file():
        raise FileNotFoundError(
            f"Source NPZ not found: {source_dataset}. Pass source_dataset explicitly."
        )
    return (
        source_dataset.resolve(),
        int(selected["input_episode_index"]),
        episode_index,
    )


def load_dataset_object_episode(
    dataset: str | Path,
    episode_index: int,
    *,
    source_dataset: str | Path | None = None,
) -> DatasetObjectEpisode:
    dataset_path = Path(dataset).resolve()
    if dataset_path.is_file() and dataset_path.suffix == ".jsonl":
        with dataset_path.open("r", encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                if int(record["episode_index"]) != episode_index:
                    continue
                return DatasetObjectEpisode(
                    object_id=str(record["object_id"]),
                    scale=float(record["scale"]),
                    position=tuple(float(value) for value in record["position"]),
                    quaternion=tuple(
                        float(value) for value in record["quaternion_wxyz"]
                    ),
                    source_dataset=dataset_path,
                    source_episode_index=int(record["source_episode_index"]),
                    lerobot_episode_index=episode_index,
                )
        raise IndexError(f"Episode {episode_index} is not in {dataset_path}")

    source_path, source_index, lerobot_index = _resolve_source_episode(
        dataset_path,
        episode_index,
        None if source_dataset is None else Path(source_dataset).resolve(),
    )
    with np.load(source_path, allow_pickle=True) as payload:
        count = int(np.asarray(payload["object_id"]).shape[0])
        if not 0 <= source_index < count:
            raise IndexError(f"Source episode {source_index} is outside [0, {count})")
        object_id = str(payload["object_id"][source_index])
        scale = float(payload["object_scale"][source_index])
        rotation = np.asarray(
            payload["object_rotmat"][source_index],
            dtype=np.float64,
        ).reshape(3, 3)
        xy = _episode_xy(payload, source_index)

    position = grounded_object_position(object_id, scale, rotation, xy)
    quaternion = _matrix_to_quaternion(rotation)
    return DatasetObjectEpisode(
        object_id=object_id,
        scale=scale,
        position=tuple(float(value) for value in position),
        quaternion=tuple(float(value) for value in quaternion),
        source_dataset=source_path,
        source_episode_index=source_index,
        lerobot_episode_index=lerobot_index,
    )
