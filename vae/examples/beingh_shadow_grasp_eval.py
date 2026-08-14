from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import av
import numpy as np
import torch
from scipy.spatial.transform import Rotation


os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[2]
VAE_ROOT = REPO_ROOT / "vae"
for import_root in (REPO_ROOT, VAE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from BeingH.inference.beingh_policy import BeingHPolicy
from BeingH.inference.checkpoint_data_config import resolve_checkpoint_data_config
from native_vae import NativeVAE
from optimize import GeometryRetargeter, GeometryRetargeterConfig
from sim import (
    GesturePolicyAdapter,
    GraspEnv,
    GraspEnvConfig,
    POLICY_WRIST_EULER_OFFSETS,
    PolicyEvaluationClient,
    load_dataset_object_episode,
)


SUPPORTED_HANDS = (
    "shadow_hand_right",
    "gaia_hand_right",
    "sharpa_hand_right",
)
SUCCESS_PROFILES = {
    "strict": {"lift_m": 0.20, "frames": 10},
    "loose": {"lift_m": 0.10, "frames": 1},
}
RAW_DATA_CONFIG_NAME = "shadow_grasp_wrist_gesture"
Q99_DATA_CONFIG_NAME = "shadow_grasp_wrist_gesture_q99"
MINMAX_DATA_CONFIG_NAME = "shadow_grasp_wrist_gesture_minmax"
WRIST_MINMAX_ZRAW_DATA_CONFIG_NAME = "shadow_grasp_wrist_minmax_gesture_raw"
WRIST_EULER_MINMAX_ZRAW_DATA_CONFIG_NAME = (
    "shadow_grasp_wrist_euler_minmax_gesture_raw"
)
WRIST_ROT6D_MINMAX_ZRAW_DATA_CONFIG_NAME = (
    "shadow_grasp_wrist_rot6d_minmax_gesture_raw"
)
TWO_CAMERA_WRIST_EULER_MINMAX_ZRAW_DATA_CONFIG_NAME = (
    "shadow_grasp_2cam_wrist_euler_minmax_gesture_raw"
)
TWO_CAMERA_WRIST_ROT6D_MINMAX_ZRAW_DATA_CONFIG_NAME = (
    "shadow_grasp_2cam_wrist_rot6d_minmax_gesture_raw"
)
TWO_CAMERA_WRIST_ROT6D_MINMAX_JOINTS_DATA_CONFIG_NAME = (
    "shadow_grasp_2cam_wrist_rot6d_minmax_joints"
)
SHARPA_JOINT_DATA_CONFIG_NAME = "sharpa_grasp_2cam_wrist_rot6d_minmax_joints"
GAIA_JOINT_DATA_CONFIG_NAME = "gaia_grasp_2cam_wrist_rot6d_minmax_joints"
SUPPORTED_DATA_CONFIG_NAMES = (
    RAW_DATA_CONFIG_NAME,
    Q99_DATA_CONFIG_NAME,
    MINMAX_DATA_CONFIG_NAME,
    WRIST_MINMAX_ZRAW_DATA_CONFIG_NAME,
    WRIST_EULER_MINMAX_ZRAW_DATA_CONFIG_NAME,
    WRIST_ROT6D_MINMAX_ZRAW_DATA_CONFIG_NAME,
    TWO_CAMERA_WRIST_EULER_MINMAX_ZRAW_DATA_CONFIG_NAME,
    TWO_CAMERA_WRIST_ROT6D_MINMAX_ZRAW_DATA_CONFIG_NAME,
    TWO_CAMERA_WRIST_ROT6D_MINMAX_JOINTS_DATA_CONFIG_NAME,
    SHARPA_JOINT_DATA_CONFIG_NAME,
    GAIA_JOINT_DATA_CONFIG_NAME,
)
ROT6D_DATA_CONFIG_NAMES = (
    WRIST_ROT6D_MINMAX_ZRAW_DATA_CONFIG_NAME,
    TWO_CAMERA_WRIST_ROT6D_MINMAX_ZRAW_DATA_CONFIG_NAME,
    TWO_CAMERA_WRIST_ROT6D_MINMAX_JOINTS_DATA_CONFIG_NAME,
    SHARPA_JOINT_DATA_CONFIG_NAME,
    GAIA_JOINT_DATA_CONFIG_NAME,
)
TWO_CAMERA_DATA_CONFIG_NAMES = (
    TWO_CAMERA_WRIST_EULER_MINMAX_ZRAW_DATA_CONFIG_NAME,
    TWO_CAMERA_WRIST_ROT6D_MINMAX_ZRAW_DATA_CONFIG_NAME,
    TWO_CAMERA_WRIST_ROT6D_MINMAX_JOINTS_DATA_CONFIG_NAME,
    SHARPA_JOINT_DATA_CONFIG_NAME,
    GAIA_JOINT_DATA_CONFIG_NAME,
)
JOINT_ACTION_DATA_CONFIG_NAMES = (
    TWO_CAMERA_WRIST_ROT6D_MINMAX_JOINTS_DATA_CONFIG_NAME,
    SHARPA_JOINT_DATA_CONFIG_NAME,
    GAIA_JOINT_DATA_CONFIG_NAME,
)
DIRECT_JOINT_CONFIG_HANDS = {
    TWO_CAMERA_WRIST_ROT6D_MINMAX_JOINTS_DATA_CONFIG_NAME: "shadow_hand_right",
    SHARPA_JOINT_DATA_CONFIG_NAME: "sharpa_hand_right",
    GAIA_JOINT_DATA_CONFIG_NAME: "gaia_hand_right",
}
DIRECT_JOINT_CONFIG_DIMS = {
    TWO_CAMERA_WRIST_ROT6D_MINMAX_JOINTS_DATA_CONFIG_NAME: 22,
    SHARPA_JOINT_DATA_CONFIG_NAME: 22,
    GAIA_JOINT_DATA_CONFIG_NAME: 15,
}


def evaluation_policy_wrist_offset(args: argparse.Namespace) -> np.ndarray:
    """Return the wrist-frame offset used by the model being evaluated.

    Shadow-canonical VAE/joint models use the Being-H/Shadow policy frame and
    therefore need the target-hand offset. Target-native joint baselines were
    trained directly on the target hand's native wrist coordinates, so their
    model boundary must not apply the Shadow->target offset a second time.
    """
    if (
        args.data_config_name in JOINT_ACTION_DATA_CONFIG_NAMES
        and DIRECT_JOINT_CONFIG_HANDS[args.data_config_name] == args.hand
        and args.hand != "shadow_hand_right"
    ):
        return np.zeros(3, dtype=np.float32)
    return np.asarray(POLICY_WRIST_EULER_OFFSETS[args.hand], dtype=np.float32)
# The collection simulator uses the opposite positive axis for the final
# three thumb joints. Keep the model in its training-data coordinates and
# convert only at the MuJoCo boundary.
SHADOW_DATASET_NEGATED_JOINT_NAMES = frozenset({"THJ2", "THJ1", "THJ0"})
LEGACY_CONTROL_REFERENCE_DATASET = (
    REPO_ROOT / "data/shadow_grasp_0725_core_bottle_1071"
)
TWO_CAMERA_CONTROL_REFERENCE_DATASET = (
    REPO_ROOT / "data/shadow_grasp_bottle22249179_aug100_2cam"
)
SHARPA_CONTROL_REFERENCE_DATASET = (
    REPO_ROOT / "data/sharpa_grasp_bottle22249179_geo_visual100_2cam"
)
GAIA_CONTROL_REFERENCE_DATASET = (
    REPO_ROOT / "data/gaia_grasp_bottle22249179_geo_visual100_2cam"
)
LEGACY_EVALUATION_DATASET = (
    VAE_ROOT / "evaluation/object_episodes/shadow_grasp_0725.jsonl"
)
TWO_CAMERA_EVALUATION_DATASET = (
    VAE_ROOT
    / "evaluation/object_episodes/shadow_grasp_bottle22249179_aug100_2cam.jsonl"
)
INSTRUCTION_TEMPLATE = (
    "According to the instruction '{task_description}', "
    "what's the micro-step actions in the next {k} steps?"
)
DEFAULT_INSTRUCTION = "Grasp the object and lift it up."


def json_default(value: Any) -> Any:
    """Convert NumPy scalars/arrays and Paths for JSON metadata output."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def validate_shadow_joint_contract(
    dataset: Path,
    mujoco_joint_names: tuple[str, ...],
) -> dict[str, Any]:
    """Verify positional equivalence of dataset and MuJoCo Shadow joints.

    The collection metadata uses zero-based Shadow names (FFJ3..FFJ0), while
    the current MuJoCo/URDF assets use the conventional one-based names
    (FFJ4..FFJ1). Their positional semantics must match exactly.
    """
    info_path = dataset.expanduser().resolve() / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"Physical-joint evaluation requires LeRobot metadata: {info_path}"
        )
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    state_names = features.get("observation.state", {}).get("names")
    action_names = features.get("action", {}).get("names")
    if not isinstance(state_names, list) or not isinstance(action_names, list):
        raise ValueError(
            f"Physical-joint evaluation requires named state/action features in {info_path}"
        )
    dataset_state_joints = tuple(str(name) for name in state_names[6:28])
    dataset_action_joints = tuple(str(name) for name in action_names[6:28])
    if dataset_state_joints != dataset_action_joints:
        raise ValueError(
            "Dataset state/action Shadow joint orders differ: "
            f"{dataset_state_joints} vs {dataset_action_joints}"
        )

    zero_based_names: list[str] = []
    for name in mujoco_joint_names:
        prefix = name.rstrip("0123456789")
        suffix = name[len(prefix):]
        if not suffix or int(suffix) <= 0:
            raise ValueError(f"Cannot convert MuJoCo Shadow joint name {name!r}")
        zero_based_names.append(f"{prefix}{int(suffix) - 1}")
    expected_zero_based = tuple(zero_based_names)
    if dataset_state_joints == tuple(mujoco_joint_names):
        naming = "identical"
    elif dataset_state_joints == expected_zero_based:
        naming = "dataset_zero_based_mujoco_one_based"
    else:
        raise ValueError(
            "Dataset Shadow joints are not positionally compatible with MuJoCo. "
            f"dataset={dataset_state_joints}, mujoco={mujoco_joint_names}, "
            f"expected_zero_based={expected_zero_based}"
        )
    contract = {
        "naming": naming,
        "dataset_joint_names": list(dataset_state_joints),
        "mujoco_joint_names": list(mujoco_joint_names),
        "mapping": [
            {
                "dataset": source,
                "mujoco": target,
                "index": index,
                "dataset_to_mujoco_sign": (
                    -1.0
                    if source in SHADOW_DATASET_NEGATED_JOINT_NAMES
                    else 1.0
                ),
            }
            for index, (source, target) in enumerate(
                zip(dataset_state_joints, mujoco_joint_names)
            )
        ],
    }
    contract["dataset_to_mujoco_signs"] = [
        item["dataset_to_mujoco_sign"] for item in contract["mapping"]
    ]
    print(
        "[shadow joint contract] "
        f"naming={naming} dims={len(dataset_state_joints)} positional=true "
        "negated=THJ2->THJ3,THJ1->THJ2,THJ0->THJ1"
    )
    return contract


def mujoco_camera_name(source_column: str) -> str:
    prefix = "observation.images."
    if not source_column.startswith(prefix):
        raise ValueError(
            f"Unsupported simulation video source column {source_column!r}; "
            f"expected {prefix!r} prefix"
        )
    camera_name = source_column[len(prefix):]
    if not camera_name:
        raise ValueError(f"Empty MuJoCo camera name in {source_column!r}")
    return camera_name


def load_jsonl_metadata(dataset: Path, episode_index: int) -> dict[str, Any]:
    """Load episode metadata and preserve the training language annotation.

    Evaluation accepts both the object-episode JSONL manifests and complete
    LeRobot dataset directories. Previously directories returned an empty
    mapping, which silently selected the generic ``DEFAULT_INSTRUCTION`` even
    when ``meta/tasks.jsonl`` contained the exact training instruction.
    """
    dataset = dataset.expanduser().resolve()
    if dataset.is_file() and dataset.suffix == ".jsonl":
        with dataset.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                if int(record["episode_index"]) == episode_index:
                    metadata = dict(record)
                    if metadata.get("task"):
                        metadata["task_source"] = "evaluation_jsonl.task"
                    return metadata
        raise IndexError(f"Episode {episode_index} is not in {dataset}")

    if not dataset.is_dir():
        return {}

    episodes_path = dataset / "meta/episodes.jsonl"
    tasks_path = dataset / "meta/tasks.jsonl"
    if not episodes_path.is_file() or not tasks_path.is_file():
        return {}

    episode_record: dict[str, Any] | None = None
    with episodes_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record["episode_index"]) == episode_index:
                episode_record = dict(record)
                break
    if episode_record is None:
        raise IndexError(f"Episode {episode_index} is not in {episodes_path}")

    tasks_by_index: dict[int, str] = {}
    with tasks_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            task = record.get("task")
            if task:
                tasks_by_index[int(record["task_index"])] = str(task)

    task_index: int | None = None
    raw_task_index = episode_record.get("task_index")
    if raw_task_index is not None:
        task_index = int(raw_task_index)
    else:
        raw_task_indices = episode_record.get("task_indices")
        if raw_task_indices:
            task_index = int(raw_task_indices[0])

    task: str | None = None
    task_source: str | None = None
    if task_index is not None and task_index in tasks_by_index:
        task = tasks_by_index[task_index]
        task_source = "meta/tasks.jsonl[task_index]"

    episode_tasks = episode_record.get("tasks")
    if task is None and episode_tasks:
        task = str(episode_tasks[0])
        task_source = "meta/episodes.jsonl.tasks[0]"
        matching_indices = [
            index for index, text in tasks_by_index.items() if text == task
        ]
        if len(matching_indices) == 1:
            task_index = matching_indices[0]
            task_source = "meta/tasks.jsonl[matched episode task]"

    if task is None and episode_record.get("task"):
        task = str(episode_record["task"])
        task_source = "meta/episodes.jsonl.task"

    # A single-entry task table is unambiguous even in older LeRobot episode
    # metadata that omitted both task_index and tasks.
    if task is None and len(tasks_by_index) == 1:
        task_index, task = next(iter(tasks_by_index.items()))
        task_source = "meta/tasks.jsonl[single task]"

    metadata = dict(episode_record)
    if task is not None:
        metadata["task"] = task
        metadata["task_source"] = task_source
    if task_index is not None:
        metadata["task_index"] = task_index
    return metadata


def resolve_episode_instruction(
    *,
    instruction_override: str | None,
    episode_index: int,
    manifest_metadata: Mapping[str, Any],
    control_reference_dataset: Path,
) -> tuple[str, str, int | None]:
    """Resolve the exact language conditioning used by the training sample.

    The control-reference LeRobot dataset is authoritative. The evaluation
    manifest describes the object/scene and may be generated separately, so
    its task text is only a fallback. ``source_episode_index`` maps a generated
    evaluation episode back to the corresponding LeRobot episode.
    """
    if instruction_override is not None:
        return instruction_override, "cli:--instruction", None

    reference_candidates: list[int] = []
    for key in (
        "lerobot_episode_index",
        "reference_episode_index",
        "source_episode_index",
    ):
        value = manifest_metadata.get(key)
        if value is not None:
            candidate = int(value)
            if candidate not in reference_candidates:
                reference_candidates.append(candidate)
    if episode_index not in reference_candidates:
        reference_candidates.append(int(episode_index))

    reference_dataset = control_reference_dataset.expanduser().resolve()
    for reference_episode_index in reference_candidates:
        try:
            reference_metadata = load_jsonl_metadata(
                reference_dataset,
                reference_episode_index,
            )
        except IndexError:
            continue
        task = reference_metadata.get("task")
        if task:
            source = reference_metadata.get(
                "task_source", "control reference metadata"
            )
            return (
                str(task),
                f"control_reference:{source} "
                f"(episode={reference_episode_index})",
                reference_metadata.get("task_index"),
            )

    task = manifest_metadata.get("task")
    if task:
        return (
            str(task),
            str(manifest_metadata.get("task_source", "evaluation manifest")),
            manifest_metadata.get("task_index"),
        )
    return DEFAULT_INSTRUCTION, "built-in fallback", None


def build_cross_hand_initial_action(
    *,
    args: argparse.Namespace,
    episode_index: int,
    manifest_metadata: Mapping[str, Any],
    vae: NativeVAE | None,
    geometry_retargeter: GeometryRetargeter | None,
    policy_wrist_world_origin: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Initialize from the matching Shadow episode in physical world coordinates.

    Reference wrist xyz is relative to ``policy_wrist_world_origin``; reference
    Euler XYZ remains absolute and receives no translation transform. Legacy
    zero-origin Shadow checkpoints retain their historical environment default.
    """
    world_origin = np.asarray(policy_wrist_world_origin, dtype=np.float32)
    if world_origin.shape != (3,) or not np.all(np.isfinite(world_origin)):
        raise ValueError("policy_wrist_world_origin must be finite xyz [3]")

    if args.hand == "shadow_hand_right" and np.allclose(world_origin, 0.0):
        return None, {
            "mode": "environment_default",
            "reason": "legacy zero-origin Shadow evaluation",
            "policy_wrist_world_origin": world_origin.tolist(),
        }
    if args.hand != "shadow_hand_right" and not args.cross_hand_initialization:
        return None, {
            "mode": "environment_default",
            "reason": "cross-hand initialization disabled by CLI",
            "policy_wrist_world_origin": world_origin.tolist(),
        }

    import pandas as pd

    reference_dataset = args.control_reference_dataset.expanduser().resolve()
    candidate_indices = [int(episode_index)]
    for key in ("lerobot_episode_index", "reference_episode_index"):
        if key in manifest_metadata:
            candidate = int(manifest_metadata[key])
            if candidate not in candidate_indices:
                candidate_indices.append(candidate)

    reference_path: Path | None = None
    reference_episode_index: int | None = None
    for candidate in candidate_indices:
        matches = sorted(
            (reference_dataset / "data").glob(
                f"chunk-*/episode_{candidate:06d}.parquet"
            )
        )
        if matches:
            reference_path = matches[0]
            reference_episode_index = candidate
            break
    if reference_path is None or reference_episode_index is None:
        raise FileNotFoundError(
            "Could not find the matching Shadow reference episode for "
            f"evaluation episode {episode_index}; tried indices "
            f"{candidate_indices} below {reference_dataset / 'data'}"
        )

    frame = pd.read_parquet(reference_path, columns=["observation.state"])
    if len(frame) == 0:
        raise ValueError(f"Reference episode is empty: {reference_path}")
    reference_state = np.asarray(
        frame.iloc[0]["observation.state"], dtype=np.float32
    ).reshape(-1)
    direct_shadow_joints = args.data_config_name in JOINT_ACTION_DATA_CONFIG_NAMES
    required_state_dim = (
        6 + DIRECT_JOINT_CONFIG_DIMS[args.data_config_name]
        if direct_shadow_joints
        else 52
    )
    if reference_state.shape[0] < required_state_dim:
        raise ValueError(
            f"Expected at least {required_state_dim} state dimensions in {reference_path}, "
            f"got {reference_state.shape[0]}"
        )
    if not np.all(np.isfinite(reference_state[:required_state_dim])):
        raise ValueError(
            f"Reference first-frame state contains NaN/Inf: {reference_path}"
        )

    shadow_wrist_policy = reference_state[0:6].copy()
    if direct_shadow_joints:
        joint_dim = DIRECT_JOINT_CONFIG_DIMS[args.data_config_name]
        reference_q_dataset = reference_state[6 : 6 + joint_dim].copy()
        joint_signs = np.asarray(
            args.shadow_joint_contract["dataset_to_mujoco_signs"],
            dtype=np.float32,
        )
        reference_shadow_q_native = reference_q_dataset * joint_signs
        initialization_retarget = None
        model_hand = DIRECT_JOINT_CONFIG_HANDS[args.data_config_name]
        if model_hand == args.hand:
            target_q = reference_shadow_q_native.copy()
        else:
            if model_hand != "shadow_hand_right":
                raise ValueError(
                    f"Native-joint checkpoint for {model_hand} must be evaluated "
                    f"with --hand {model_hand}, not {args.hand}"
                )
            if geometry_retargeter is None:
                raise RuntimeError(
                    "Cross-hand physical-joint initialization requires the "
                    "geometry retargeter"
                )
            initialization_retarget = geometry_retargeter.retarget(
                reference_shadow_q_native,
                "shadow_hand_right",
                args.hand,
                stream="initialization",
                update_state=False,
            )
            target_q = initialization_retarget.target_q
        reference_z = None
    else:
        if vae is None:
            raise RuntimeError("z_gesture initialization requires NativeVAE")
        reference_q_dataset = None
        reference_shadow_q_native = None
        initialization_retarget = None
        reference_z = reference_state[28:52].copy()
        target_q = (
            vae.decode(reference_z, args.hand)[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    target_wrist_world = shadow_wrist_policy.copy()
    target_wrist_world[0:3] += world_origin
    # Only the cross-hand Euler convention is changed; RPY is never translated.
    target_wrist_world[3:6] -= evaluation_policy_wrist_offset(args)
    initial_action = np.concatenate(
        [target_wrist_world, target_q]
    ).astype(np.float32)
    metadata = {
        "mode": (
            "shadow_first_frame_relative_wrist_and_direct_joints"
            if direct_shadow_joints
            else "shadow_first_frame_relative_wrist_and_z_decode"
        ),
        "reference_dataset": str(reference_dataset),
        "reference_episode_index": reference_episode_index,
        "reference_episode_path": str(reference_path.resolve()),
        "reference_state_layout": (
            {"wrist": [0, 6], "shadow_joint_position": [6, 28]}
            if direct_shadow_joints
            else {"wrist": [0, 6], "z_gesture": [28, 52]}
        ),
        "policy_wrist_world_origin": world_origin.tolist(),
        "reference_shadow_wrist_policy_coordinates": (
            shadow_wrist_policy.tolist()
        ),
        "reference_shadow_joint_position": (
            reference_shadow_q_native.tolist()
            if direct_shadow_joints
            else None
        ),
        "reference_shadow_joint_position_dataset_coordinates": (
            None
            if reference_q_dataset is None
            else reference_q_dataset.tolist()
        ),
        "reference_z_gesture": (
            None if reference_z is None else reference_z.tolist()
        ),
        "target_q_requested": target_q.tolist(),
        "target_q_source": (
            (
                "signed_transform(observation.state[6:28])"
                if args.hand == "shadow_hand_right"
                else "geometry_retarget(shadow_hand_right->target_hand)"
            )
            if direct_shadow_joints
            else "VAE.decode(observation.state[28:52])"
        ),
        "initialization_geometry_retarget": (
            None
            if initialization_retarget is None
            else initialization_retarget.metadata()
        ),
        "decoded_target_q_requested": (
            None if direct_shadow_joints else target_q.tolist()
        ),
        "target_wrist_world_requested": target_wrist_world.tolist(),
    }
    print(
        "[dataset init] "
        f"hand={args.hand} reference_episode={reference_episode_index} "
        f"policy_xyz={shadow_wrist_policy[:3].tolist()} "
        f"world_xyz={target_wrist_world[:3].tolist()} "
        f"origin={world_origin.tolist()} source={reference_path}",
        flush=True,
    )
    return initial_action, metadata


def load_reference_initial_z_gesture(
    *,
    args: argparse.Namespace,
    episode_index: int,
    manifest_metadata: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the matching Shadow episode's first z_gesture without changing reset."""
    import pandas as pd

    reference_dataset = args.control_reference_dataset.expanduser().resolve()
    candidate_indices = [int(episode_index)]
    for key in ("lerobot_episode_index", "reference_episode_index"):
        if key in manifest_metadata:
            candidate = int(manifest_metadata[key])
            if candidate not in candidate_indices:
                candidate_indices.append(candidate)

    reference_path: Path | None = None
    reference_episode_index: int | None = None
    for candidate in candidate_indices:
        matches = sorted(
            (reference_dataset / "data").glob(
                f"chunk-*/episode_{candidate:06d}.parquet"
            )
        )
        if matches:
            reference_path = matches[0]
            reference_episode_index = candidate
            break
    if reference_path is None or reference_episode_index is None:
        raise FileNotFoundError(
            "Could not find initial commanded z_gesture for evaluation episode "
            f"{episode_index}; tried indices {candidate_indices} below "
            f"{reference_dataset / 'data'}"
        )

    frame = pd.read_parquet(reference_path, columns=["observation.state"])
    if len(frame) == 0:
        raise ValueError(f"Reference episode is empty: {reference_path}")
    reference_state = np.asarray(
        frame.iloc[0]["observation.state"], dtype=np.float32
    ).reshape(-1)
    if reference_state.shape[0] < 52:
        raise ValueError(
            f"Expected at least 52 state dimensions in {reference_path}, "
            f"got {reference_state.shape[0]}"
        )
    reference_z = reference_state[28:52].copy()
    if not np.all(np.isfinite(reference_z)):
        raise ValueError(
            f"Reference first-frame z_gesture contains NaN/Inf: {reference_path}"
        )
    return reference_z, {
        "source": "shadow_reference_first_observation",
        "reference_dataset": str(reference_dataset),
        "reference_episode_index": reference_episode_index,
        "reference_episode_path": str(reference_path.resolve()),
        "reference_state_slice": [28, 52],
    }


def load_dataset_wrist_action_trajectory(
    *,
    args: argparse.Namespace,
    episode_index: int,
    manifest_metadata: Mapping[str, Any],
    required_steps: int,
    policy_wrist_world_origin: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Load policy-coordinate wrist6 and convert xyz to MuJoCo world."""
    if args.wrist_action_source == "model":
        return None, {"source": "model"}

    import pandas as pd

    reference_dataset = args.control_reference_dataset.expanduser().resolve()
    candidate_indices = [int(episode_index)]
    for key in ("lerobot_episode_index", "reference_episode_index"):
        if key in manifest_metadata:
            candidate = int(manifest_metadata[key])
            if candidate not in candidate_indices:
                candidate_indices.append(candidate)

    reference_path: Path | None = None
    reference_episode_index: int | None = None
    for candidate in candidate_indices:
        matches = sorted(
            (reference_dataset / "data").glob(
                f"chunk-*/episode_{candidate:06d}.parquet"
            )
        )
        if matches:
            reference_path = matches[0]
            reference_episode_index = candidate
            break
    if reference_path is None or reference_episode_index is None:
        raise FileNotFoundError(
            "Could not find dataset wrist actions for evaluation episode "
            f"{episode_index}; tried indices {candidate_indices} below "
            f"{reference_dataset / 'data'}"
        )

    frame = pd.read_parquet(reference_path, columns=["action"])
    if len(frame) < required_steps:
        raise ValueError(
            f"Dataset wrist trajectory {reference_path} has {len(frame)} steps, "
            f"but evaluation requires {required_steps}"
        )
    actions = np.stack(frame["action"].to_numpy()).astype(np.float32)
    if actions.ndim != 2 or actions.shape[1] < 6:
        raise ValueError(
            f"Expected action [T,D>=6] in {reference_path}, got {actions.shape}"
        )
    wrist_actions = actions[:required_steps, :6].copy()
    if not np.all(np.isfinite(wrist_actions)):
        raise ValueError(f"Dataset wrist trajectory contains NaN/Inf: {reference_path}")

    world_origin = np.asarray(policy_wrist_world_origin, dtype=np.float32)
    if world_origin.shape != (3,) or not np.all(np.isfinite(world_origin)):
        raise ValueError("policy_wrist_world_origin must be finite xyz [3]")
    wrist_actions[:, 0:3] += world_origin
    # Dataset wrist Euler angles are absolute in the Shadow/Being-H convention.
    # Convert only their target-hand convention; do not translate RPY.
    wrist_actions[:, 3:6] -= np.asarray(
        evaluation_policy_wrist_offset(args)
    )
    metadata = {
        "source": "dataset",
        "reference_dataset": str(reference_dataset),
        "reference_episode_index": reference_episode_index,
        "reference_episode_path": str(reference_path.resolve()),
        "slice": [0, 6],
        "num_steps": int(len(wrist_actions)),
        "source_xyz_coordinates": "relative_to_wrist_world_origin",
        "executed_xyz_coordinates": "mujoco_world",
        "policy_wrist_world_origin": world_origin.tolist(),
        "target_hand_wrist_euler_offset_removed": list(
            evaluation_policy_wrist_offset(args)
        ),
    }
    print(
        "[wrist action] source=dataset "
        f"reference_episode={reference_episode_index} "
        f"steps={len(wrist_actions)} source={reference_path}",
        flush=True,
    )
    return wrist_actions, metadata


def write_mp4(path: Path, images: tuple[np.ndarray, ...], fps: int) -> None:
    if not images:
        raise ValueError("No RGB frames were rendered.")
    first = np.asarray(images[0])
    if first.ndim != 3 or first.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB frames, got {first.shape}")

    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = int(first.shape[1])
        stream.height = int(first.shape[0])
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        for image in images:
            frame_array = np.ascontiguousarray(image, dtype=np.uint8)
            if frame_array.shape != first.shape:
                raise ValueError(
                    f"All video frames must have shape {first.shape}, "
                    f"got {frame_array.shape}"
                )
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def align_euler_branch(
    target: np.ndarray,
    reference: np.ndarray,
    *,
    include_xyz_equivalent: bool = False,
) -> np.ndarray:
    """Choose the intrinsic-XYZ Euler solution nearest the current qpos.

    Legacy checkpoints retain component-wise 2*pi alignment. Rot6D checkpoints
    additionally consider the second equivalent XYZ solution
    ``(x + pi, pi - y, z + pi)`` before choosing the nearest joint command.
    """
    target = np.asarray(target)
    reference = np.asarray(reference)
    primary = reference + (target - reference + np.pi) % (2.0 * np.pi) - np.pi
    if not include_xyz_equivalent:
        return primary
    alternate_raw = np.asarray(
        [target[0] + np.pi, np.pi - target[1], target[2] + np.pi],
        dtype=target.dtype,
    )
    alternate = (
        reference
        + (alternate_raw - reference + np.pi) % (2.0 * np.pi)
        - np.pi
    )
    if np.linalg.norm(alternate - reference) < np.linalg.norm(primary - reference):
        return alternate
    return primary


def latency_summary_seconds(values: list[float]) -> dict[str, float | int]:
    samples = np.asarray(values, dtype=np.float64)
    if len(samples) == 0:
        return {
            "count": 0,
            "total_s": 0.0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "count": int(len(samples)),
        "total_s": float(samples.sum()),
        "mean_ms": float(samples.mean() * 1000.0),
        "p50_ms": float(np.quantile(samples, 0.50) * 1000.0),
        "p95_ms": float(np.quantile(samples, 0.95) * 1000.0),
        "max_ms": float(samples.max() * 1000.0),
    }


def wrist_motion_metrics(actions: np.ndarray) -> dict[str, float]:
    if len(actions) < 2:
        return {
            "position_step_mean_m": 0.0,
            "position_step_max_m": 0.0,
            "rotation_step_mean_deg": 0.0,
            "rotation_step_max_deg": 0.0,
        }
    position_steps = np.linalg.norm(np.diff(actions[:, 0:3], axis=0), axis=1)
    rotations = Rotation.from_euler("XYZ", actions[:, 3:6])
    rotation_steps = (rotations[:-1].inv() * rotations[1:]).magnitude()
    return {
        "position_step_mean_m": float(position_steps.mean()),
        "position_step_max_m": float(position_steps.max()),
        "rotation_step_mean_deg": float(np.degrees(rotation_steps).mean()),
        "rotation_step_max_deg": float(np.degrees(rotation_steps).max()),
    }


def _quadratic_temporal_smooth(
    values: np.ndarray,
    anchor: np.ndarray,
    velocity_weight: float,
    acceleration_weight: float,
) -> np.ndarray:
    """Smooth a complete chunk while anchoring its first derivatives to state."""
    values = np.asarray(values, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    if values.ndim != 2 or anchor.shape != (values.shape[1],):
        raise ValueError(
            f"Expected values [T,D] and anchor [D], got {values.shape}, {anchor.shape}"
        )
    horizon = len(values)
    if horizon <= 1 or (velocity_weight <= 0 and acceleration_weight <= 0):
        return values.astype(np.float32, copy=True)

    identity = np.eye(horizon, dtype=np.float64)
    system = identity.copy()
    rhs = values.copy()

    if velocity_weight > 0:
        first = np.eye(horizon, dtype=np.float64)
        if horizon > 1:
            first[1:, :-1] -= np.eye(horizon - 1, dtype=np.float64)
        first_target = np.zeros((horizon, values.shape[1]), dtype=np.float64)
        first_target[0] = anchor
        system += velocity_weight * (first.T @ first)
        rhs += velocity_weight * (first.T @ first_target)

    if acceleration_weight > 0:
        second = np.zeros((horizon, horizon), dtype=np.float64)
        second_target = np.zeros(
            (horizon, values.shape[1]), dtype=np.float64
        )
        second[0, 0] = 1.0
        second_target[0] = anchor
        if horizon > 1:
            second[1, 0:2] = (-2.0, 1.0)
            second_target[1] = -anchor
        for index in range(2, horizon):
            second[index, index - 2 : index + 1] = (1.0, -2.0, 1.0)
        system += acceleration_weight * (second.T @ second)
        rhs += acceleration_weight * (second.T @ second_target)

    return np.linalg.solve(system, rhs).astype(np.float32)


class ChunkTemporalSmoother:
    """Regularize first/second differences inside each predicted action chunk."""

    def __init__(
        self,
        velocity_weight: float,
        acceleration_weight: float,
        *,
        include_xyz_equivalent: bool = False,
    ) -> None:
        if velocity_weight < 0 or acceleration_weight < 0:
            raise ValueError("Chunk smoothing weights must be non-negative")
        self.velocity_weight = float(velocity_weight)
        self.acceleration_weight = float(acceleration_weight)
        self.include_xyz_equivalent = bool(include_xyz_equivalent)

    @property
    def enabled(self) -> bool:
        return self.velocity_weight > 0 or self.acceleration_weight > 0

    def apply(self, chunk: np.ndarray, state: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        if (
            chunk.ndim != 2
            or chunk.shape[1] < 6
            or state.shape != (chunk.shape[1],)
        ):
            raise ValueError(
                "Expected chunk [T,D>=6] and matching state [D], got "
                f"{chunk.shape}, {state.shape}"
            )
        if not self.enabled:
            return chunk.copy()

        aligned = chunk.copy()
        euler_reference = state[3:6].copy()
        for index in range(len(aligned)):
            aligned[index, 3:6] = align_euler_branch(
                aligned[index, 3:6],
                euler_reference,
                include_xyz_equivalent=self.include_xyz_equivalent,
            )
            euler_reference = aligned[index, 3:6]

        # Smooth wrist and the selected hand representation as separate blocks.
        # Downstream SO(3)/native-joint limiters remain hard constraints.
        output = aligned.copy()
        for block in (slice(0, 3), slice(3, 6), slice(6, chunk.shape[1])):
            output[:, block] = _quadratic_temporal_smooth(
                aligned[:, block],
                state[block],
                self.velocity_weight,
                self.acceleration_weight,
            )
        return output

    def metadata(self) -> dict[str, float | bool]:
        return {
            "enabled": self.enabled,
            "velocity_weight": self.velocity_weight,
            "acceleration_weight": self.acceleration_weight,
            "include_xyz_equivalent": self.include_xyz_equivalent,
        }


class TrainingDistributionRateLimiter:
    """Constrain wrist velocity/acceleration; rotation is limited on SO(3)."""

    def __init__(
        self,
        *,
        position_step_limits: np.ndarray,
        position_acceleration_limits: np.ndarray,
        rotation_step_limit: float,
        rotation_acceleration_limit: float,
        z_step_limits: np.ndarray | None,
        z_acceleration_limits: np.ndarray | None,
        limit_z_gesture: bool,
        quantile: float,
        source_dataset: Path,
        include_xyz_equivalent: bool = False,
    ) -> None:
        self.position_step_limits = np.asarray(position_step_limits, dtype=np.float32)
        self.position_acceleration_limits = np.asarray(
            position_acceleration_limits, dtype=np.float32
        )
        self.rotation_step_limit = float(rotation_step_limit)
        self.rotation_acceleration_limit = float(rotation_acceleration_limit)
        self.z_step_limits = (
            None if z_step_limits is None else np.asarray(z_step_limits, dtype=np.float32)
        )
        self.z_acceleration_limits = (
            None
            if z_acceleration_limits is None
            else np.asarray(z_acceleration_limits, dtype=np.float32)
        )
        self.limit_z_gesture = bool(limit_z_gesture)
        if self.limit_z_gesture and (
            self.z_step_limits is None or self.z_acceleration_limits is None
        ):
            raise ValueError("z velocity and acceleration limits are required")
        self.quantile = float(quantile)
        self.source_dataset = source_dataset
        self.include_xyz_equivalent = bool(include_xyz_equivalent)
        self._last_action: np.ndarray | None = None
        self._last_position_velocity: np.ndarray | None = None
        self._last_rotation_velocity: np.ndarray | None = None
        self._last_z_velocity: np.ndarray | None = None

    @classmethod
    def from_lerobot_dataset(
        cls,
        dataset: Path,
        quantile: float,
        *,
        limit_z_gesture: bool = False,
        include_xyz_equivalent: bool = False,
        direct_native_joints: bool = False,
    ) -> "TrainingDistributionRateLimiter":
        import pandas as pd

        if not 0.0 < quantile <= 1.0:
            raise ValueError("rate-limit-quantile must be in (0, 1]")
        parquet_paths = sorted(dataset.glob("data/chunk-*/episode_*.parquet"))
        if not parquet_paths:
            raise FileNotFoundError(f"No episode parquet files found under {dataset}")

        position_steps: list[np.ndarray] = []
        position_accelerations: list[np.ndarray] = []
        rotation_steps: list[np.ndarray] = []
        rotation_accelerations: list[np.ndarray] = []
        z_steps: list[np.ndarray] = []
        z_accelerations: list[np.ndarray] = []
        for path in parquet_paths:
            actions = np.stack(
                pd.read_parquet(path, columns=["action"])["action"].to_numpy()
            ).astype(np.float32)
            minimum_dim = 7 if direct_native_joints else 52
            if actions.ndim != 2 or actions.shape[1] < minimum_dim or len(actions) < 2:
                raise ValueError(f"Expected action [T,>={minimum_dim}] in {path}, got {actions.shape}")
            position_velocity = np.diff(actions[:, 0:3], axis=0)
            position_steps.append(np.abs(position_velocity))
            if len(position_velocity) >= 2:
                position_accelerations.append(np.abs(np.diff(position_velocity, axis=0)))

            rotations = Rotation.from_euler("XYZ", actions[:, 3:6])
            rotation_velocity = (
                rotations[:-1].inv() * rotations[1:]
            ).as_rotvec()
            rotation_steps.append(np.linalg.norm(rotation_velocity, axis=1))
            if len(rotation_velocity) >= 2:
                rotation_accelerations.append(
                    np.linalg.norm(np.diff(rotation_velocity, axis=0), axis=1)
                )

            if limit_z_gesture and not direct_native_joints:
                z_velocity = np.diff(actions[:, 28:52], axis=0)
                z_steps.append(np.abs(z_velocity))
                if len(z_velocity) >= 2:
                    z_accelerations.append(np.abs(np.diff(z_velocity, axis=0)))

        if not position_accelerations or not rotation_accelerations:
            raise ValueError("At least one three-frame episode is required for acceleration limits")
        position_limit = np.quantile(np.concatenate(position_steps), quantile, axis=0)
        position_acceleration_limit = np.quantile(
            np.concatenate(position_accelerations), quantile, axis=0
        )
        rotation_limit = float(np.quantile(np.concatenate(rotation_steps), quantile))
        rotation_acceleration_limit = float(
            np.quantile(np.concatenate(rotation_accelerations), quantile)
        )
        z_limit = (
            np.maximum(np.quantile(np.concatenate(z_steps), quantile, axis=0), 1e-5)
            if limit_z_gesture
            else None
        )
        z_acceleration_limit = (
            np.maximum(
                np.quantile(np.concatenate(z_accelerations), quantile, axis=0),
                1e-5,
            )
            if limit_z_gesture
            else None
        )
        return cls(
            position_step_limits=np.maximum(position_limit, 1e-5),
            position_acceleration_limits=np.maximum(
                position_acceleration_limit, 1e-5
            ),
            rotation_step_limit=max(rotation_limit, 1e-4),
            rotation_acceleration_limit=max(rotation_acceleration_limit, 1e-4),
            z_step_limits=z_limit,
            z_acceleration_limits=z_acceleration_limit,
            limit_z_gesture=limit_z_gesture,
            quantile=quantile,
            source_dataset=dataset.resolve(),
            include_xyz_equivalent=include_xyz_equivalent,
        )

    def reset(self) -> None:
        self._last_action = None
        self._last_position_velocity = None
        self._last_rotation_velocity = None
        self._last_z_velocity = None

    @staticmethod
    def _limit_vector_norm(value: np.ndarray, limit: float) -> np.ndarray:
        norm = float(np.linalg.norm(value))
        if norm > limit:
            return value * (limit / norm)
        return value

    def apply(self, target: np.ndarray, observation_state: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=np.float32)
        observation_state = np.asarray(observation_state, dtype=np.float32)
        reference = (
            observation_state.copy()
            if self._last_action is None
            else self._last_action.copy()
        )
        output = target.copy()

        position_velocity = np.clip(
            target[0:3] - reference[0:3],
            -self.position_step_limits,
            self.position_step_limits,
        )
        if self._last_position_velocity is not None:
            position_velocity = self._last_position_velocity + np.clip(
                position_velocity - self._last_position_velocity,
                -self.position_acceleration_limits,
                self.position_acceleration_limits,
            )
        output[0:3] = reference[0:3] + position_velocity

        reference_rotation = Rotation.from_euler("XYZ", reference[3:6])
        target_rotation = Rotation.from_euler("XYZ", target[3:6])
        rotation_velocity = (reference_rotation.inv() * target_rotation).as_rotvec()
        rotation_velocity = self._limit_vector_norm(
            rotation_velocity, self.rotation_step_limit
        )
        if self._last_rotation_velocity is not None:
            rotation_acceleration = self._limit_vector_norm(
                rotation_velocity - self._last_rotation_velocity,
                self.rotation_acceleration_limit,
            )
            rotation_velocity = self._last_rotation_velocity + rotation_acceleration
            rotation_velocity = self._limit_vector_norm(
                rotation_velocity, self.rotation_step_limit
            )
        limited_rotation = reference_rotation * Rotation.from_rotvec(rotation_velocity)
        output[3:6] = align_euler_branch(
            limited_rotation.as_euler("XYZ").astype(np.float32),
            reference[3:6],
            include_xyz_equivalent=self.include_xyz_equivalent,
        )

        if self.limit_z_gesture:
            assert self.z_step_limits is not None
            assert self.z_acceleration_limits is not None
            z_velocity = np.clip(
                target[6:30] - reference[6:30],
                -self.z_step_limits,
                self.z_step_limits,
            )
            if self._last_z_velocity is not None:
                z_velocity = self._last_z_velocity + np.clip(
                    z_velocity - self._last_z_velocity,
                    -self.z_acceleration_limits,
                    self.z_acceleration_limits,
                )
            output[6:30] = reference[6:30] + z_velocity
            self._last_z_velocity = z_velocity.copy()

        self._last_action = output.copy()
        self._last_position_velocity = position_velocity.copy()
        self._last_rotation_velocity = rotation_velocity.copy()
        return output

    def metadata(self) -> dict[str, Any]:
        return {
            "source_dataset": str(self.source_dataset),
            "quantile": self.quantile,
            "position_step_limits_m": self.position_step_limits.tolist(),
            "position_acceleration_limits_m_per_step2": (
                self.position_acceleration_limits.tolist()
            ),
            "rotation_step_limit_deg": float(np.degrees(self.rotation_step_limit)),
            "rotation_acceleration_limit_deg_per_step2": float(
                np.degrees(self.rotation_acceleration_limit)
            ),
            "limit_z_gesture": self.limit_z_gesture,
            "z_step_limits": (
                None if self.z_step_limits is None else self.z_step_limits.tolist()
            ),
            "z_acceleration_limits": (
                None
                if self.z_acceleration_limits is None
                else self.z_acceleration_limits.tolist()
            ),
            "decoded_native_joint_limiter": True,
        }


class AuditedNativePolicy:
    """Audit decoded actions, constrain native joints, and enforce safety faults."""

    def __init__(
        self,
        policy: GesturePolicyAdapter | ShadowJointPolicyAdapter,
        q_step_limits: np.ndarray | None,
        q_acceleration_limits: np.ndarray | None,
        *,
        safety_mode: str = "off",
        safety_max_wrist_position_error_m: float = 0.25,
        safety_max_wrist_rotation_error_deg: float = 90.0,
        safety_max_object_drop_m: float = 0.30,
        safety_max_workspace_radius_m: float = 0.60,
    ) -> None:
        if safety_mode not in {"off", "hold", "terminate"}:
            raise ValueError("safety_mode must be off, hold, or terminate")
        self.policy = policy
        self.q_step_limits = (
            None if q_step_limits is None else np.asarray(q_step_limits, dtype=np.float32)
        )
        self.q_acceleration_limits = (
            None
            if q_acceleration_limits is None
            else np.asarray(q_acceleration_limits, dtype=np.float32)
        )
        self.safety_mode = safety_mode
        self.safety_max_wrist_position_error_m = float(
            safety_max_wrist_position_error_m
        )
        self.safety_max_wrist_rotation_error_rad = float(
            np.radians(safety_max_wrist_rotation_error_deg)
        )
        self.safety_max_object_drop_m = float(safety_max_object_drop_m)
        self.safety_max_workspace_radius_m = float(safety_max_workspace_radius_m)
        self._last_action: np.ndarray | None = None
        self._last_q_velocity: np.ndarray | None = None
        self._initial_wrist_position: np.ndarray | None = None
        self._initial_object_z: float | None = None
        self._safety_latched = False
        self.termination_requested = False
        self.termination_reason: str | None = None
        self.safety_events: list[dict[str, Any]] = []
        self.raw_native_actions: list[np.ndarray] = []
        self.executed_native_actions: list[np.ndarray] = []
        self.model_native_actions: list[np.ndarray] = []
        self.dataset_wrist_actions_requested: list[np.ndarray] = []
        self.predict_latencies_s: list[float] = []
        self._dataset_wrist_trajectory: np.ndarray | None = None
        self._dataset_wrist_step = 0

    def set_initial_commanded_z(
        self, z_gesture: np.ndarray | None
    ) -> None:
        self.policy.set_initial_commanded_z(z_gesture)

    def set_wrist_action_trajectory(
        self, wrist_actions: np.ndarray | None
    ) -> None:
        if wrist_actions is None:
            self._dataset_wrist_trajectory = None
        else:
            values = np.asarray(wrist_actions, dtype=np.float32)
            if values.ndim != 2 or values.shape[1] != 6:
                raise ValueError(
                    f"Expected dataset wrist trajectory [T,6], got {values.shape}"
                )
            if len(values) == 0 or not np.all(np.isfinite(values)):
                raise ValueError(
                    "Dataset wrist trajectory must be non-empty and finite"
                )
            self._dataset_wrist_trajectory = values.copy()
        self._dataset_wrist_step = 0

    def _next_dataset_wrist(self) -> np.ndarray | None:
        if self._dataset_wrist_trajectory is None:
            return None
        if self._dataset_wrist_step >= len(self._dataset_wrist_trajectory):
            raise IndexError(
                "Dataset wrist trajectory was exhausted at control step "
                f"{self._dataset_wrist_step}"
            )
        wrist = self._dataset_wrist_trajectory[self._dataset_wrist_step].copy()
        self._dataset_wrist_step += 1
        self.dataset_wrist_actions_requested.append(wrist.copy())
        return wrist

    @classmethod
    def q_motion_limits_from_lerobot_dataset(
        cls,
        dataset: Path,
        vae: NativeVAE | None,
        target_hand: str,
        quantile: float,
        *,
        direct_shadow_joints: bool = False,
        geometry_retargeter: GeometryRetargeter | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        import pandas as pd

        parquet_paths = sorted(dataset.glob("data/chunk-*/episode_*.parquet"))
        if not parquet_paths:
            raise FileNotFoundError(f"No episode parquet files found under {dataset}")
        episode_hand_actions: list[np.ndarray] = []
        for path in parquet_paths:
            actions = np.stack(
                pd.read_parquet(path, columns=["action"])["action"].to_numpy()
            ).astype(np.float32)
            if actions.ndim != 2 or actions.shape[1] < 7:
                raise ValueError(
                    f"Expected action [T,>=7] in {path}, got {actions.shape}"
                )
            if direct_shadow_joints:
                # Canonical Shadow datasets contain [wrist6, joints22, z24],
                # while target-native joint datasets contain [wrist6, jointsN].
                hand_slice = actions[:, 6:28] if actions.shape[1] >= 52 else actions[:, 6:]
            else:
                hand_slice = actions[:, 28:52]
            episode_hand_actions.append(hand_slice)
        lengths = [len(values) for values in episode_hand_actions]
        if direct_shadow_joints:
            decoded = np.concatenate(episode_hand_actions, axis=0)
        else:
            if vae is None:
                raise RuntimeError("z_gesture motion limits require NativeVAE")
            decoded = vae.decode(
                np.concatenate(episode_hand_actions, axis=0), target_hand
            ).detach().cpu().numpy()
        q_steps: list[np.ndarray] = []
        q_accelerations: list[np.ndarray] = []
        offset = 0
        for length in lengths:
            episode_q = decoded[offset : offset + length]
            offset += length
            if length >= 2:
                velocity = np.diff(episode_q, axis=0)
                q_steps.append(np.abs(velocity))
                if len(velocity) >= 2:
                    q_accelerations.append(np.abs(np.diff(velocity, axis=0)))
        if not q_accelerations:
            raise ValueError("At least one three-frame episode is required for q acceleration limits")
        step_limits = np.maximum(
            np.quantile(np.concatenate(q_steps, axis=0), quantile, axis=0),
            1e-5,
        ).astype(np.float32)
        acceleration_limits = np.maximum(
            np.quantile(
                np.concatenate(q_accelerations, axis=0), quantile, axis=0
            ),
            1e-5,
        ).astype(np.float32)
        if (
            direct_shadow_joints
            and target_hand != "shadow_hand_right"
            and decoded.shape[1] == 22
        ):
            if geometry_retargeter is None:
                raise RuntimeError(
                    "Cross-hand physical-joint limits require GeometryRetargeter"
                )
            step_limits = geometry_retargeter.project_motion_limits(
                step_limits, "shadow_hand_right", target_hand
            )
            acceleration_limits = geometry_retargeter.project_motion_limits(
                acceleration_limits, "shadow_hand_right", target_hand
            )
        return step_limits, acceleration_limits

    def reset(self) -> None:
        self.policy.reset()
        self._last_action = None
        self._last_q_velocity = None
        self._initial_wrist_position = None
        self._initial_object_z = None
        self._safety_latched = False
        self.termination_requested = False
        self.termination_reason = None
        self.safety_events.clear()
        self.raw_native_actions.clear()
        self.executed_native_actions.clear()
        self.model_native_actions.clear()
        self.dataset_wrist_actions_requested.clear()
        self.predict_latencies_s.clear()
        self._dataset_wrist_step = 0

    def _record_fault(self, reason: str) -> None:
        if not self._safety_latched:
            self.safety_events.append(
                {"step": len(self.executed_native_actions), "reason": reason}
            )
            print(f"[safety] {self.safety_mode}: {reason}", flush=True)
        self._safety_latched = True
        self.termination_reason = reason
        self.termination_requested = self.safety_mode == "terminate"

    def _observation_fault(self, observation: Mapping[str, Any]) -> str | None:
        state = np.asarray(observation["state"], dtype=np.float32)
        object_pose = np.asarray(observation.get("object_pose", []), dtype=np.float32)
        if not np.isfinite(state).all():
            return "non-finite robot state"
        if object_pose.size and not np.isfinite(object_pose).all():
            return "non-finite object pose"
        if self._initial_wrist_position is None:
            self._initial_wrist_position = state[0:3].copy()
        if object_pose.size and self._initial_object_z is None:
            self._initial_object_z = float(object_pose[2])
        if (
            np.linalg.norm(state[0:3] - self._initial_wrist_position)
            > self.safety_max_workspace_radius_m
        ):
            return "wrist left configured workspace radius"
        if (
            object_pose.size
            and self._initial_object_z is not None
            and float(object_pose[2])
            < self._initial_object_z - self.safety_max_object_drop_m
        ):
            return "object dropped below configured safety floor"
        return None

    def _command_fault(
        self, target: np.ndarray, observation_state: np.ndarray
    ) -> str | None:
        if not np.isfinite(target).all():
            return "non-finite decoded action"
        position_error = float(np.linalg.norm(target[0:3] - observation_state[0:3]))
        if position_error > self.safety_max_wrist_position_error_m:
            return f"wrist command position error {position_error:.3f}m"
        current_rotation = Rotation.from_euler("XYZ", observation_state[3:6])
        target_rotation = Rotation.from_euler("XYZ", target[3:6])
        rotation_error = float((current_rotation.inv() * target_rotation).magnitude())
        if rotation_error > self.safety_max_wrist_rotation_error_rad:
            return f"wrist command rotation error {np.degrees(rotation_error):.1f}deg"
        return None

    def predict(self, observation: Mapping[str, Any]) -> np.ndarray:
        started_at = time.perf_counter()
        observation_state = np.asarray(observation["state"], dtype=np.float32)
        hold_action = observation_state.copy()
        dataset_wrist = self._next_dataset_wrist()
        model_target: np.ndarray | None = None
        fault = self._observation_fault(observation) if self.safety_mode != "off" else None

        if self._safety_latched or fault is not None:
            if fault is not None:
                self._record_fault(fault)
            target = hold_action
        else:
            try:
                target = np.asarray(self.policy.predict(observation), dtype=np.float32)
                model_target = target.copy()
                if dataset_wrist is not None:
                    target = target.copy()
                    target[:6] = dataset_wrist
            except Exception as error:
                if self.safety_mode == "off":
                    raise
                self._record_fault(f"policy exception: {type(error).__name__}: {error}")
                target = hold_action
            if self.safety_mode != "off" and not self._safety_latched:
                fault = self._command_fault(target, observation_state)
                if fault is not None:
                    self._record_fault(fault)
                    target = hold_action

        if model_target is None:
            model_target = target.copy()
        self.model_native_actions.append(model_target)
        self.raw_native_actions.append(target.copy())
        output = target.copy()
        if self.q_step_limits is not None and not self._safety_latched:
            reference = (
                observation_state if self._last_action is None else self._last_action
            )
            q_velocity = np.clip(
                target[6:] - reference[6:],
                -self.q_step_limits,
                self.q_step_limits,
            )
            if self._last_q_velocity is not None:
                if self.q_acceleration_limits is None:
                    raise RuntimeError("Native q acceleration limits are missing")
                q_velocity = self._last_q_velocity + np.clip(
                    q_velocity - self._last_q_velocity,
                    -self.q_acceleration_limits,
                    self.q_acceleration_limits,
                )
            output[6:] = reference[6:] + q_velocity
            self._last_q_velocity = q_velocity.copy()
        elif self._safety_latched:
            output = hold_action
            self._last_q_velocity = np.zeros_like(output[6:])

        self._last_action = output.copy()
        self.executed_native_actions.append(output.copy())
        self.predict_latencies_s.append(time.perf_counter() - started_at)
        return output


class ShadowJointPolicyAdapter:
    """Adapt native Shadow state/actions without passing joints through the VAE."""

    def __init__(
        self,
        policy: BeingHGesturePolicy,
        *,
        target_hand: str,
        joint_names: tuple[str, ...],
        dataset_to_mujoco_signs: np.ndarray,
        policy_wrist_euler_offset: np.ndarray,
        policy_wrist_world_origin: np.ndarray,
        observation_mode: str,
    ) -> None:
        if target_hand not in SUPPORTED_HANDS:
            raise ValueError(f"Unsupported target hand {target_hand!r}")
        if len(joint_names) <= 0:
            raise ValueError("Native-joint checkpoint must expose at least one joint")
        self.policy = policy
        self.target_hand = target_hand
        self.joint_names = tuple(joint_names)
        self.dataset_to_mujoco_signs = np.asarray(
            dataset_to_mujoco_signs, dtype=np.float32
        )
        if self.dataset_to_mujoco_signs.shape != (len(self.joint_names),) or not np.all(
            np.isin(self.dataset_to_mujoco_signs, (-1.0, 1.0))
        ):
            raise ValueError(
                "dataset_to_mujoco_signs must match native joint dimension and contain {-1,+1}"
            )
        self.policy_wrist_euler_offset = np.asarray(
            policy_wrist_euler_offset, dtype=np.float32
        )
        self.policy_wrist_world_origin = np.asarray(
            policy_wrist_world_origin, dtype=np.float32
        )
        if observation_mode not in {"encoded", "commanded"}:
            raise ValueError(
                "observation_mode must be encoded or commanded"
            )
        self.observation_mode = observation_mode
        self.z_dim = 0
        # Preserve the audit interface consumed by the shared result writer.
        self.encoded_latent_states: list[np.ndarray] = []
        self.policy_latent_states: list[np.ndarray] = []
        self._initial_commanded_shadow_q: np.ndarray | None = None
        self._commanded_shadow_q: np.ndarray | None = None

    def set_initial_commanded_z(self, shadow_q: np.ndarray | None) -> None:
        if shadow_q is None:
            self._initial_commanded_shadow_q = None
            self._commanded_shadow_q = None
            return
        values = np.asarray(shadow_q, dtype=np.float32)
        if values.shape != (len(self.joint_names),) or not np.all(np.isfinite(values)):
            raise ValueError(
                "Initial commanded native joints have the wrong dimension: "
                f"expected {len(self.joint_names)}, got {values.shape}"
            )
        self._initial_commanded_shadow_q = values.copy()
        self._commanded_shadow_q = values.copy()

    def reset(self) -> None:
        self.policy.reset()
        self.encoded_latent_states.clear()
        self.policy_latent_states.clear()
        self._commanded_shadow_q = (
            None
            if self._initial_commanded_shadow_q is None
            else self._initial_commanded_shadow_q.copy()
        )

    def predict(self, observation: Mapping[str, Any]) -> np.ndarray:
        native_state = np.asarray(observation["state"], dtype=np.float32)
        expected_dim = 6 + len(self.joint_names)
        if native_state.shape != (expected_dim,):
            raise ValueError(
                f"Expected native Shadow state [{expected_dim}], got "
                f"{native_state.shape}"
            )
        policy_wrist = native_state[:6].copy()
        policy_wrist[0:3] -= self.policy_wrist_world_origin
        policy_wrist[3:6] += self.policy_wrist_euler_offset
        observed_shadow_q = native_state[6:].copy()
        if self.observation_mode == "commanded":
            if self._commanded_shadow_q is None:
                raise RuntimeError(
                    "Commanded Shadow feedback has no initial joint state"
                )
            policy_shadow_q = self._commanded_shadow_q.copy()
        else:
            policy_shadow_q = observed_shadow_q.copy()
        self.encoded_latent_states.append(
            observed_shadow_q * self.dataset_to_mujoco_signs
        )
        self.policy_latent_states.append(
            policy_shadow_q * self.dataset_to_mujoco_signs
        )
        policy_observation = dict(observation)
        policy_observation["native_state"] = native_state
        policy_observation["state"] = np.concatenate(
            [
                policy_wrist,
                policy_shadow_q * self.dataset_to_mujoco_signs,
            ]
        ).astype(np.float32)

        policy_action = np.asarray(
            self.policy.predict(policy_observation), dtype=np.float32
        )
        if policy_action.shape != (expected_dim,):
            raise ValueError(
                f"Physical-joint policy must output [{expected_dim}], got "
                f"{policy_action.shape}"
            )
        if not np.all(np.isfinite(policy_action)):
            raise ValueError("Physical-joint policy output contains NaN or Inf")

        native_action = policy_action.copy()
        native_action[0:3] += self.policy_wrist_world_origin
        native_action[3:6] -= self.policy_wrist_euler_offset
        native_action[6:] *= self.dataset_to_mujoco_signs
        if self.observation_mode == "commanded":
            self._commanded_shadow_q = native_action[6:].copy()
        return native_action


class GeometricShadowJointPolicyAdapter:
    """Use native Shadow joints as the canonical cross-hand policy space."""

    def __init__(
        self,
        policy: BeingHGesturePolicy,
        *,
        retargeter: GeometryRetargeter,
        target_hand: str,
        dataset_to_mujoco_signs: np.ndarray,
        policy_wrist_euler_offset: np.ndarray,
        policy_wrist_world_origin: np.ndarray,
        action_chunk_mode: str,
        observation_mode: str,
    ) -> None:
        if target_hand == "shadow_hand_right":
            raise ValueError(
                "Use ShadowJointPolicyAdapter for native Shadow evaluation"
            )
        self.policy = policy
        self.retargeter = retargeter
        self.target_hand = target_hand
        self.target_joint_names = retargeter.joint_names(target_hand)
        self.shadow_joint_names = retargeter.joint_names("shadow_hand_right")
        self.dataset_to_mujoco_signs = np.asarray(
            dataset_to_mujoco_signs, dtype=np.float32
        )
        if self.dataset_to_mujoco_signs.shape != (22,) or not np.all(
            np.isin(self.dataset_to_mujoco_signs, (-1.0, 1.0))
        ):
            raise ValueError(
                "dataset_to_mujoco_signs must contain 22 values in {-1,+1}"
            )
        self.policy_wrist_euler_offset = np.asarray(
            policy_wrist_euler_offset, dtype=np.float32
        )
        self.policy_wrist_world_origin = np.asarray(
            policy_wrist_world_origin, dtype=np.float32
        )
        if action_chunk_mode not in {"batch", "sequential"}:
            raise ValueError(
                "action_chunk_mode must be 'batch' or 'sequential'"
            )
        self.action_chunk_mode = action_chunk_mode
        if observation_mode not in {"encoded", "commanded"}:
            raise ValueError(
                "observation_mode must be encoded or commanded"
            )
        self.observation_mode = observation_mode
        self.z_dim = 22
        self.encoded_latent_states: list[np.ndarray] = []
        self.policy_latent_states: list[np.ndarray] = []
        self.retargeted_shadow_observation_joints: list[np.ndarray] = []
        self.model_shadow_action_joints: list[np.ndarray] = []
        self.retargeted_target_action_joints: list[np.ndarray] = []
        self.observation_retarget_results: list[Any] = []
        self.action_retarget_results: list[Any] = []
        self._cached_shadow_action_chunk: np.ndarray | None = None
        self._cached_target_action_chunk: np.ndarray | None = None
        self._cached_chunk_index = 0
        self._cached_shadow_observation: np.ndarray | None = None
        self._initial_commanded_shadow_q: np.ndarray | None = None
        self._commanded_shadow_q: np.ndarray | None = None

    def set_initial_commanded_z(self, shadow_q: np.ndarray | None) -> None:
        if shadow_q is None:
            self._initial_commanded_shadow_q = None
            self._commanded_shadow_q = None
            return
        values = np.asarray(shadow_q, dtype=np.float32)
        if values.shape != (22,) or not np.all(np.isfinite(values)):
            raise ValueError(
                "Initial commanded Shadow joints must be finite [22], "
                f"got {values.shape}"
            )
        self._initial_commanded_shadow_q = values.copy()
        self._commanded_shadow_q = values.copy()

    def reset(self) -> None:
        self.policy.reset()
        self.retargeter.reset()
        self._commanded_shadow_q = (
            None
            if self._initial_commanded_shadow_q is None
            else self._initial_commanded_shadow_q.copy()
        )
        self.encoded_latent_states.clear()
        self.policy_latent_states.clear()
        self.retargeted_shadow_observation_joints.clear()
        self.model_shadow_action_joints.clear()
        self.retargeted_target_action_joints.clear()
        self.observation_retarget_results.clear()
        self.action_retarget_results.clear()
        self._cached_shadow_action_chunk = None
        self._cached_target_action_chunk = None
        self._cached_chunk_index = 0
        self._cached_shadow_observation = None

    @staticmethod
    def _result_summary(results: list[Any]) -> dict[str, Any]:
        if not results:
            return {
                "count": 0,
                "geometry_rmse_mean": None,
                "geometry_rmse_max": None,
                "latency_s_mean": None,
                "latency_s_max": None,
            }
        rmse = np.asarray(
            [result.geometry_rmse for result in results], dtype=np.float64
        )
        latency = np.asarray(
            [result.elapsed_s for result in results], dtype=np.float64
        )
        return {
            "count": len(results),
            "frames": int(
                sum(int(getattr(result, "batch_size", 1)) for result in results)
            ),
            "geometry_rmse_mean": float(rmse.mean()),
            "geometry_rmse_max": float(rmse.max()),
            "latency_s_mean": float(latency.mean()),
            "latency_s_max": float(latency.max()),
        }

    def retargeting_metadata(self) -> dict[str, Any]:
        return {
            **self.retargeter.metadata(),
            "canonical_hand": "shadow_hand_right",
            "action_chunk_mode": self.action_chunk_mode,
            "observation_mode": self.observation_mode,
            "observation_direction": (
                f"{self.target_hand}->shadow_hand_right"
            ),
            "action_direction": (
                f"shadow_hand_right->{self.target_hand}"
            ),
            "observation": self._result_summary(
                self.observation_retarget_results
            ),
            "action": self._result_summary(self.action_retarget_results),
        }

    def predict(self, observation: Mapping[str, Any]) -> np.ndarray:
        native_state = np.asarray(observation["state"], dtype=np.float32)
        expected_native_dim = 6 + len(self.target_joint_names)
        if native_state.shape != (expected_native_dim,):
            raise ValueError(
                f"Expected {self.target_hand} state [{expected_native_dim}], "
                f"got {native_state.shape}"
            )

        query_expected = self.policy.will_query_next()
        if (
            self.action_chunk_mode == "sequential"
            or query_expected
            or self._cached_shadow_observation is None
        ):
            observation_result = self.retargeter.retarget(
                native_state[6:],
                self.target_hand,
                "shadow_hand_right",
                stream="observation",
            )
            shadow_observation_native = observation_result.target_q
            self.observation_retarget_results.append(observation_result)
            self._cached_shadow_observation = (
                shadow_observation_native.copy()
            )
        else:
            shadow_observation_native = (
                self._cached_shadow_observation.copy()
            )
        self.retargeted_shadow_observation_joints.append(
            shadow_observation_native.copy()
        )
        if self.observation_mode == "commanded":
            if self._commanded_shadow_q is None:
                raise RuntimeError(
                    "Commanded geometry feedback has no initial Shadow joints"
                )
            policy_shadow_observation_native = (
                self._commanded_shadow_q.copy()
            )
        else:
            policy_shadow_observation_native = (
                shadow_observation_native.copy()
            )
        self.encoded_latent_states.append(
            shadow_observation_native * self.dataset_to_mujoco_signs
        )
        self.policy_latent_states.append(
            policy_shadow_observation_native * self.dataset_to_mujoco_signs
        )

        policy_wrist = native_state[:6].copy()
        policy_wrist[0:3] -= self.policy_wrist_world_origin
        policy_wrist[3:6] += self.policy_wrist_euler_offset
        policy_observation = dict(observation)
        policy_observation["native_state"] = native_state
        policy_observation["state"] = np.concatenate(
            [
                policy_wrist,
                policy_shadow_observation_native * self.dataset_to_mujoco_signs,
            ]
        ).astype(np.float32)

        query_count_before = len(self.policy.query_step_indices)
        policy_action = np.asarray(
            self.policy.predict(policy_observation), dtype=np.float32
        )
        query_count_after = len(self.policy.query_step_indices)
        if policy_action.shape != (28,):
            raise ValueError(
                f"Physical Shadow-joint policy must output [28], got "
                f"{policy_action.shape}"
            )
        if not np.all(np.isfinite(policy_action)):
            raise ValueError("Physical-joint policy output contains NaN or Inf")

        shadow_action_native = (
            policy_action[6:] * self.dataset_to_mujoco_signs
        )
        if self.observation_mode == "commanded":
            self._commanded_shadow_q = shadow_action_native.copy()
        if self.action_chunk_mode == "batch":
            if query_count_after > query_count_before:
                canonical_chunk = self.policy.current_action_chunk()
                shadow_action_chunk = (
                    canonical_chunk[:, 6:] * self.dataset_to_mujoco_signs
                )
                batch_result = self.retargeter.retarget_batch(
                    shadow_action_chunk,
                    "shadow_hand_right",
                    self.target_hand,
                    stream="action_chunk",
                )
                self.action_retarget_results.append(batch_result)
                self._cached_shadow_action_chunk = shadow_action_chunk.copy()
                self._cached_target_action_chunk = batch_result.target_q.copy()
                self._cached_chunk_index = 0
            if (
                self._cached_shadow_action_chunk is None
                or self._cached_target_action_chunk is None
                or self._cached_chunk_index >= len(
                    self._cached_target_action_chunk
                )
            ):
                raise RuntimeError("Batched target action chunk is unavailable")
            cached_shadow_action = self._cached_shadow_action_chunk[
                self._cached_chunk_index
            ]
            if not np.allclose(
                shadow_action_native,
                cached_shadow_action,
                atol=1e-5,
                rtol=1e-5,
            ):
                raise RuntimeError(
                    "Selected canonical hand action differs from the cached "
                    "batched chunk; use --geometry-action-chunk-mode sequential"
                )
            target_action_q = self._cached_target_action_chunk[
                self._cached_chunk_index
            ].copy()
            self._cached_chunk_index += 1
        else:
            action_result = self.retargeter.retarget(
                shadow_action_native,
                "shadow_hand_right",
                self.target_hand,
                stream="action",
            )
            self.action_retarget_results.append(action_result)
            target_action_q = action_result.target_q
        self.model_shadow_action_joints.append(shadow_action_native.copy())
        self.retargeted_target_action_joints.append(
            target_action_q.copy()
        )

        target_wrist = policy_action[:6].copy()
        target_wrist[0:3] += self.policy_wrist_world_origin
        target_wrist[3:6] -= self.policy_wrist_euler_offset
        return np.concatenate(
            [target_wrist, target_action_q]
        ).astype(np.float32)


class BeingHGesturePolicy:
    """Adapt wrist plus z_gesture or Shadow joints to BeingHPolicy's API."""

    def __init__(
        self,
        *,
        model_path: Path,
        data_config_name: str,
        instruction: str,
        device: str,
        replan_every: int,
        seed: int,
        noise_mode: str,
        action_selection: str,
        inference_mode: str,
        temporal_ensemble_decay: float,
        temporal_ensemble_max_history: int | None,
        rate_limiter: TrainingDistributionRateLimiter | None,
        chunk_smoother: ChunkTemporalSmoother | None,
        clip_normalized_wrist_action: bool,
        use_mpg: bool | None,
        mpg_refinement_iters: int | None,
        num_inference_timesteps: int | None,
    ) -> None:
        if replan_every <= 0:
            raise ValueError("replan_every must be positive")
        if noise_mode not in {"rollout", "fixed_per_query"}:
            raise ValueError("noise_mode must be 'rollout' or 'fixed_per_query'")
        if action_selection not in {"chunk", "temporal_ensemble"}:
            raise ValueError(
                "action_selection must be 'chunk' or 'temporal_ensemble'"
            )
        if inference_mode not in {"sync", "async"}:
            raise ValueError("inference_mode must be 'sync' or 'async'")
        if inference_mode == "async" and action_selection != "temporal_ensemble":
            raise ValueError(
                "async inference currently requires temporal_ensemble action selection"
            )
        if temporal_ensemble_decay < 0:
            raise ValueError("temporal_ensemble_decay must be non-negative")
        if (
            temporal_ensemble_max_history is not None
            and temporal_ensemble_max_history <= 0
        ):
            raise ValueError("temporal_ensemble_max_history must be positive")

        self.instruction = instruction
        self.hand_action_representation = (
            "shadow_joint_position"
            if data_config_name in JOINT_ACTION_DATA_CONFIG_NAMES
            else "z_gesture"
        )
        self.hand_action_dim = (
            DIRECT_JOINT_CONFIG_DIMS[data_config_name]
            if self.hand_action_representation == "shadow_joint_position"
            else 24
        )
        self.policy_action_dim = 6 + self.hand_action_dim
        self.replan_every = replan_every
        self.include_xyz_equivalent = (
            data_config_name in ROT6D_DATA_CONFIG_NAMES
        )
        self.seed = int(seed)
        self.noise_mode = noise_mode
        self.action_selection = action_selection
        self.inference_mode = inference_mode
        self.temporal_ensemble_decay = float(temporal_ensemble_decay)
        self.rate_limiter = rate_limiter
        self.chunk_smoother = chunk_smoother
        if (
            self.inference_mode == "async"
            and self.chunk_smoother is not None
            and self.chunk_smoother.enabled
        ):
            raise ValueError("Chunk smoothing currently requires synchronous inference")
        self.policy = BeingHPolicy(
            model_path=str(model_path),
            data_config_name=data_config_name,
            dataset_name="shadow_grasp_posttrain",
            embodiment_tag="new_embodiment",
            instruction_template=INSTRUCTION_TEMPLATE,
            device=device,
            enable_rtc=False,
            use_mpg=use_mpg,
            mpg_refinement_iters=mpg_refinement_iters,
            num_inference_timesteps=num_inference_timesteps,
            clip_normalized_action_keys=(
                [
                    "action.eef_position",
                    "action.eef_rotation",
                    *(
                        ["action.shadow_joint_position"]
                        if self.hand_action_representation
                        == "shadow_joint_position"
                        else []
                    ),
                ]
                if clip_normalized_wrist_action
                else None
            ),
        )
        columns = self.policy.data_config.VIDEO_SOURCE_COLUMNS
        self.video_source_columns = {
            video_key: str(columns[video_key])
            for video_key in self.policy.data_config.VIDEO_KEYS
        }
        self.video_camera_names = {
            video_key: mujoco_camera_name(source_column)
            for video_key, source_column in self.video_source_columns.items()
        }
        self.required_cameras = tuple(
            dict.fromkeys(self.video_camera_names.values())
        )
        self.policy_wrist_world_origin = np.asarray(
            getattr(
                self.policy.data_config,
                "WRIST_WORLD_ORIGIN",
                (0.0, 0.0, 0.0),
            ),
            dtype=np.float32,
        )
        if (
            self.policy_wrist_world_origin.shape != (3,)
            or not np.all(np.isfinite(self.policy_wrist_world_origin))
        ):
            raise ValueError(
                "Data config WRIST_WORLD_ORIGIN must be finite xyz [3]"
            )
        model_chunk_length = int(self.policy.action_chunk_length)
        requested_history = temporal_ensemble_max_history or model_chunk_length
        self.temporal_ensemble_max_history = min(
            int(requested_history),
            model_chunk_length,
        )
        if (
            self.action_selection == "temporal_ensemble"
            and self.replan_every > self.temporal_ensemble_max_history
        ):
            raise ValueError(
                "temporal ensemble query interval cannot exceed its action horizon"
            )
        self._chunk: np.ndarray | None = None
        self._chunk_index = 0
        self._step_index = 0
        self._ensemble_history: list[tuple[int, np.ndarray]] = []
        self.predicted_chunks: list[np.ndarray] = []
        self.smoothed_predicted_chunks: list[np.ndarray] = []
        self.query_step_indices: list[int] = []
        self.observed_latent_states: list[np.ndarray] = []
        self.latest_query_actions: list[np.ndarray] = []
        self.temporal_ensemble_actions: list[np.ndarray] = []
        self.temporal_ensemble_candidate_counts: list[int] = []
        self.raw_selected_actions: list[np.ndarray] = []
        self.branch_aligned_actions: list[np.ndarray] = []
        self.executed_latent_actions: list[np.ndarray] = []
        self.query_latencies_s: list[float] = []
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="beingh-inference")
            if self.inference_mode == "async"
            else None
        )
        self._pending_future: Future[np.ndarray] | None = None
        self._pending_query_step: int | None = None
        self._last_query_launch_step: int | None = None
        self.async_submitted_queries = 0
        self.async_blocking_waits = 0
        self.async_blocking_wait_seconds = 0.0

    def reset(self) -> None:
        if self._pending_future is not None:
            self._pending_future.cancel()
            if not self._pending_future.cancelled():
                self._pending_future.result()
            self._pending_future = None
            self._pending_query_step = None
        seed_everything(self.seed)
        self._chunk = None
        self._chunk_index = 0
        self._step_index = 0
        self._last_query_launch_step = None
        self.async_submitted_queries = 0
        self.async_blocking_waits = 0
        self.async_blocking_wait_seconds = 0.0
        self._ensemble_history.clear()
        self.predicted_chunks.clear()
        self.smoothed_predicted_chunks.clear()
        self.query_step_indices.clear()
        self.observed_latent_states.clear()
        self.latest_query_actions.clear()
        self.temporal_ensemble_actions.clear()
        self.temporal_ensemble_candidate_counts.clear()
        self.raw_selected_actions.clear()
        self.branch_aligned_actions.clear()
        self.executed_latent_actions.clear()
        self.query_latencies_s.clear()
        if self.rate_limiter is not None:
            self.rate_limiter.reset()

    def will_query_next(self) -> bool:
        """Return whether the next synchronous chunk-selection step queries Being-H."""
        if self.inference_mode != "sync" or self.action_selection != "chunk":
            return True
        return (
            self._chunk is None
            or self._chunk_index >= len(self._chunk)
            or self._chunk_index >= self.replan_every
        )

    def current_action_chunk(self) -> np.ndarray:
        """Return the current post-smoothing canonical action chunk."""
        if self._chunk is None:
            raise RuntimeError("Being-H has not produced an action chunk")
        return self._chunk.copy()

    def _query_model(self, observation: Mapping[str, Any]) -> np.ndarray:
        if self.noise_mode == "fixed_per_query":
            # Flow matching starts from random noise. Reusing one noise template makes
            # repeated receding-horizon queries a deterministic conditional policy.
            seed_everything(self.seed)

        state = np.asarray(observation["state"], dtype=np.float32)
        if state.shape != (self.policy_action_dim,):
            raise ValueError(
                f"Expected policy state [wrist6,{self.hand_action_representation}"
                f"{self.hand_action_dim}], got {state.shape}"
            )

        observation_images = observation.get("images", {})
        if observation_images is None:
            observation_images = {}
        if not isinstance(observation_images, Mapping):
            raise ValueError("observation['images'] must map camera names to RGB arrays")

        hand_state_key = f"state.{self.hand_action_representation}"
        hand_action_key = f"action.{self.hand_action_representation}"
        policy_observation: dict[str, Any] = {
            "state.eef_position": state[None, 0:3],
            "state.eef_rotation": state[None, 3:6],
            hand_state_key: state[None, 6:],
            "language.instruction": [self.instruction],
        }
        for video_key, camera_name in self.video_camera_names.items():
            image = observation_images.get(camera_name)
            if image is None and len(self.video_camera_names) == 1:
                # Preserve compatibility with callers that only populate legacy
                # observation['image'] for a single-view checkpoint.
                image = observation.get("image")
            if image is None:
                raise ValueError(
                    f"Being-H checkpoint requires MuJoCo camera {camera_name!r} "
                    f"for policy key {video_key!r}"
                )
            image = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    f"{camera_name}: expected HWC RGB image, got {image.shape}"
                )
            policy_observation[video_key] = image[None, ...]

        query_started_at = time.perf_counter()
        result = self.policy.get_action(policy_observation)
        values = []
        for key in (
            "action.eef_position",
            "action.eef_rotation",
            hand_action_key,
        ):
            value = np.asarray(result[key], dtype=np.float32)
            if value.ndim == 3:
                if value.shape[0] != 1:
                    raise ValueError(f"{key}: only batch size one is supported")
                value = value[0]
            values.append(value)
        chunk = np.concatenate(values, axis=-1)
        if chunk.ndim != 2 or chunk.shape[1] != self.policy_action_dim:
            raise ValueError(
                f"Expected Being-H action chunk [T,{self.policy_action_dim}], "
                f"got {chunk.shape}"
            )
        if not np.isfinite(chunk).all():
            raise ValueError("Being-H returned NaN or Inf in its action chunk")
        self.query_latencies_s.append(time.perf_counter() - query_started_at)
        return chunk

    def _register_query_result(
        self,
        query_step: int,
        chunk: np.ndarray,
    ) -> None:
        self.predicted_chunks.append(chunk.copy())
        self.query_step_indices.append(query_step)
        query_number = len(self.predicted_chunks)
        if query_number == 1 or query_number % 10 == 0:
            latency_ms = self.query_latencies_s[-1] * 1000.0
            print(
                f"[policy] query={query_number} env_step={query_step} "
                f"latency={latency_ms:.1f}ms",
                flush=True,
            )

    def _run_sync_query(
        self,
        observation: Mapping[str, Any],
        query_step: int,
    ) -> np.ndarray:
        chunk = self._query_model(observation)
        self._register_query_result(query_step, chunk)
        state = np.asarray(observation["state"], dtype=np.float32)
        smoothed = (
            self.chunk_smoother.apply(chunk, state)
            if self.chunk_smoother is not None
            else chunk.copy()
        )
        self.smoothed_predicted_chunks.append(smoothed.copy())
        return smoothed

    def _submit_async_query(
        self,
        observation: Mapping[str, Any],
        query_step: int,
    ) -> None:
        if self._executor is None:
            raise RuntimeError("Async inference executor is not initialized")
        if self._pending_future is not None:
            raise RuntimeError("Only one Being-H inference may be pending")
        snapshot = {
            "state": np.array(observation["state"], dtype=np.float32, copy=True),
            "image": (
                None
                if observation.get("image") is None
                else np.array(observation["image"], dtype=np.uint8, copy=True)
            ),
            "images": {
                camera_name: np.array(image, dtype=np.uint8, copy=True)
                for camera_name, image in observation.get("images", {}).items()
            },
        }
        self._pending_query_step = query_step
        self._pending_future = self._executor.submit(self._query_model, snapshot)
        self._last_query_launch_step = query_step
        self.async_submitted_queries += 1

    def _collect_async_query(self, *, block: bool) -> bool:
        future = self._pending_future
        if future is None:
            return False
        if not block and not future.done():
            return False
        wait_started = time.perf_counter()
        had_to_wait = block and not future.done()
        chunk = future.result()
        if had_to_wait:
            self.async_blocking_waits += 1
            self.async_blocking_wait_seconds += time.perf_counter() - wait_started
        query_step = self._pending_query_step
        if query_step is None:
            raise RuntimeError("Pending async query has no query step")
        self._pending_future = None
        self._pending_query_step = None
        self._register_query_result(query_step, chunk)
        self.smoothed_predicted_chunks.append(chunk.copy())
        self._chunk = chunk
        self._ensemble_history.append((query_step, chunk))
        return True

    def _has_valid_ensemble_candidate(self) -> bool:
        return any(
            0 <= self._step_index - query_step
            < min(len(query_chunk), self.temporal_ensemble_max_history)
            for query_step, query_chunk in self._ensemble_history
        )

    def finish_episode(self) -> None:
        """Finish any in-flight inference without destroying the reusable policy."""
        if self._pending_future is not None:
            self._collect_async_query(block=True)

    def close(self) -> None:
        self.finish_episode()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def _ensemble_action(
        self,
        state: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Aggregate queried chunks aligned to the current environment step."""
        current_step = self._step_index
        oldest_step = current_step - self.temporal_ensemble_max_history + 1
        self._ensemble_history = [
            (query_step, query_chunk)
            for query_step, query_chunk in self._ensemble_history
            if query_step >= oldest_step
            and 0 <= current_step - query_step < len(query_chunk)
        ]

        candidates: list[np.ndarray] = []
        ages: list[int] = []
        for query_step, query_chunk in self._ensemble_history:
            age = current_step - query_step
            if age >= self.temporal_ensemble_max_history:
                continue
            candidate = query_chunk[age].copy()
            candidate[3:6] = align_euler_branch(
                candidate[3:6],
                state[3:6],
                include_xyz_equivalent=self.include_xyz_equivalent,
            )
            candidates.append(candidate)
            ages.append(age)

        if not candidates:
            raise RuntimeError("Temporal ensemble has no prediction for current step")
        weights = np.exp(
            -self.temporal_ensemble_decay * np.asarray(ages, dtype=np.float64)
        )
        weights /= weights.sum()
        ensemble = np.sum(
            np.stack(candidates, axis=0).astype(np.float64)
            * weights[:, None],
            axis=0,
        ).astype(np.float32)
        return ensemble, len(candidates)

    def predict(self, observation: Mapping[str, Any]) -> np.ndarray:
        state = np.asarray(observation["state"], dtype=np.float32)
        self.observed_latent_states.append(state.copy())
        if self.action_selection == "temporal_ensemble":
            if self.inference_mode == "async":
                self._collect_async_query(block=False)
                query_due = (
                    self._last_query_launch_step is None
                    or self._step_index - self._last_query_launch_step
                    >= self.replan_every
                )
                if query_due:
                    if self._pending_future is not None:
                        self._collect_async_query(block=True)
                    self._submit_async_query(observation, self._step_index)
                if not self._has_valid_ensemble_candidate():
                    if self._pending_future is None:
                        raise RuntimeError(
                            "Action horizon expired with no pending async query"
                        )
                    self._collect_async_query(block=True)
            else:
                should_query = (
                    self._chunk is None
                    or self._step_index % self.replan_every == 0
                )
                if should_query:
                    self._chunk = self._run_sync_query(
                        observation,
                        self._step_index,
                    )
                    self._ensemble_history.append(
                        (self._step_index, self._chunk)
                    )
            if self._chunk is None:
                raise RuntimeError("Being-H has not produced an action chunk")
            latest_action = self._chunk[0].copy()
            branch_aligned, candidate_count = self._ensemble_action(state)
            raw_action = branch_aligned.copy()
        else:
            if (
                self._chunk is None
                or self._chunk_index >= len(self._chunk)
                or self._chunk_index >= self.replan_every
            ):
                self._chunk = self._run_sync_query(
                    observation,
                    self._step_index,
                )
                self._chunk_index = 0

            raw_action = self._chunk[self._chunk_index].copy()
            latest_action = self._chunk[0].copy()
            self._chunk_index += 1
            branch_aligned = raw_action.copy()
            branch_aligned[3:6] = align_euler_branch(
                raw_action[3:6],
                state[3:6],
                include_xyz_equivalent=self.include_xyz_equivalent,
            )
            candidate_count = 1

        self.latest_query_actions.append(latest_action.copy())
        self.raw_selected_actions.append(raw_action.copy())
        self.temporal_ensemble_actions.append(branch_aligned.copy())
        self.temporal_ensemble_candidate_counts.append(candidate_count)
        self.branch_aligned_actions.append(branch_aligned.copy())
        executed = (
            self.rate_limiter.apply(branch_aligned, state)
            if self.rate_limiter is not None
            else branch_aligned
        )
        self.executed_latent_actions.append(executed.copy())
        self._step_index += 1
        return executed

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a fine-tuned Being-H05 Shadow z_gesture or physical-joint "
            "policy in MuJoCo."
        )
    )
    parser.add_argument(
        "--deployment-profile",
        choices=("legacy", "safe_smooth"),
        default="safe_smooth",
        help=(
            "safe_smooth selects synchronous 16-step chunk execution, fixed query "
            "noise, chunk velocity/acceleration smoothing, SO(3)/native-q motion "
            "limits, normalized wrist clipping, and a safety guard. legacy keeps "
            "the historical defaults unless individual options are supplied."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs/shadow_grasp_bottle_1071"
            / "train-shadow_grasp_0725_core_bottle_1071_Being-H05-2B_"
              "freeze-mllm-True_chunk-16_tok-8192_norm-wrist_euler_minmax_"
              "zraw_wristw-1.0_tdelta-0.0_mpg-True_20260730_104141"
            / "0020000"
        ),
        help=(
            "Fine-tuned checkpoint directory containing config.json, tokenizer files, "
            "model safetensors, and shadow_grasp_posttrain_metadata.json."
        ),
    )
    parser.add_argument(
        "--data-config-name",
        choices=SUPPORTED_DATA_CONFIG_NAMES,
        default=None,
        help=(
            "Override the data transform stored with the checkpoint. By default "
            "it is detected from config.json or the legacy run_config YAML. Use "
            f"{Q99_DATA_CONFIG_NAME!r} for q01/q99 checkpoints, "
            f"{MINMAX_DATA_CONFIG_NAME!r} for full min-max checkpoints, and "
            f"{WRIST_MINMAX_ZRAW_DATA_CONFIG_NAME!r} for wrist-only min-max "
            "with axis-angle and raw z_gesture checkpoints, "
            f"{WRIST_EULER_MINMAX_ZRAW_DATA_CONFIG_NAME!r} for continuous "
            "Euler wrist min-max with raw z_gesture checkpoints, "
            f"{WRIST_ROT6D_MINMAX_ZRAW_DATA_CONFIG_NAME!r} for internal Rot6D "
            "wrist min-max with raw z_gesture checkpoints; the corresponding "
            "shadow_grasp_2cam_* names use ego_opposite + wrist; "
            f"{TWO_CAMERA_WRIST_ROT6D_MINMAX_JOINTS_DATA_CONFIG_NAME!r} is the "
            "22D Shadow-canonical physical-joint baseline; and "
            f"{RAW_DATA_CONFIG_NAME!r} for legacy raw checkpoints."
        ),
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help=(
            "Natural-language instruction override. When omitted, the episode "
            "task is read automatically from an evaluation JSONL or from the "
            "LeRobot meta/tasks.jsonl annotation."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--hand",
        choices=SUPPORTED_HANDS,
        default="shadow_hand_right",
        help="Target hand used by NativeVAE decoding and the MuJoCo environment.",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=VAE_ROOT / "checkpoints/native_n2_epoch800_inference.pt",
    )
    parser.add_argument(
        "--joint-retargeting",
        choices=("auto", "none", "geometry"),
        default="auto",
        help=(
            "For a physical Shadow-joint checkpoint, auto uses direct joints on "
            "Shadow and geometry optimization on Gaia/Sharpa. z_gesture "
            "checkpoints continue to use NativeVAE."
        ),
    )
    parser.add_argument(
        "--geometry-retargeting-profile",
        choices=("raw", "stable"),
        default="raw",
        help=(
            "raw minimizes only the NativeVAE-compatible 60D geometry; stable "
            "also regularizes normalized target-joint velocity/acceleration."
        ),
    )
    parser.add_argument(
        "--geometry-action-chunk-mode",
        choices=("auto", "batch", "sequential"),
        default="auto",
        help=(
            "auto uses one batched geometry solve per action chunk for "
            "synchronous chunk selection, and otherwise falls back to "
            "per-executed-step retargeting."
        ),
    )
    parser.add_argument("--geometry-max-iterations", type=int, default=12)
    parser.add_argument("--geometry-learning-rate", type=float, default=0.8)
    parser.add_argument("--geometry-tolerance", type=float, default=1e-7)
    parser.add_argument("--geometry-temporal-weight", type=float, default=2e-3)
    parser.add_argument(
        "--geometry-acceleration-weight", type=float, default=5e-4
    )
    parser.add_argument(
        "--scene-xml",
        type=Path,
        default=None,
        help="Optional generated hand/object scene. The selected hand template is used if omitted.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Portable JSONL manifest, LeRobot dataset directory, or source NPZ. "
            "When omitted, it is selected from the checkpoint data config."
        ),
    )
    episode_group = parser.add_mutually_exclusive_group()
    episode_group.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Evaluate one episode. Episode 4 is used when no episode option is given.",
    )
    episode_group.add_argument(
        "--episodes",
        "--episode-indices",
        dest="episode_indices",
        type=int,
        nargs="+",
        default=None,
        help="Evaluate multiple episodes while loading Being-H and the VAE only once.",
    )
    episode_group.add_argument(
        "--episode-range",
        dest="episode_range",
        type=int,
        nargs=2,
        metavar=("A", "B"),
        default=None,
        help=(
            "Evaluate every episode in the inclusive index range [A, B] while "
            "loading Being-H and the VAE only once."
        ),
    )
    episode_group.add_argument(
        "--all-episodes",
        action="store_true",
        help="Evaluate every episode available in the dataset or manifest.",
    )
    parser.add_argument("--source-dataset", type=Path, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Closed-loop policy steps. For a JSONL manifest this defaults to the "
            "episode frame count; otherwise it defaults to 300."
        ),
    )
    parser.add_argument(
        "--success-profile",
        choices=tuple(SUCCESS_PROFILES),
        default="strict",
        help=(
            "Success criterion: strict requires a 0.20 m lift for 10 consecutive "
            "control frames; loose succeeds once lift reaches 0.10 m."
        ),
    )
    parser.add_argument(
        "--replan-every",
        type=int,
        default=16,
        help=(
            "For chunk selection, actions executed before the next query. For temporal "
            "ensemble, the model query interval in environment steps."
        ),
    )
    parser.add_argument(
        "--action-selection",
        choices=("chunk", "temporal_ensemble"),
        default="chunk",
        help=(
            "chunk executes actions directly from one predicted chunk. "
            "temporal_ensemble queries every replan_every steps and averages "
            "overlapping chunk predictions aligned to the current execution time."
        ),
    )
    parser.add_argument(
        "--inference-mode",
        choices=("sync", "async"),
        default="sync",
        help=(
            "async runs Being-H in one background worker while buffered actions "
            "continue; it blocks only when a query slot or valid action is unavailable."
        ),
    )
    parser.add_argument(
        "--temporal-ensemble-decay",
        type=float,
        default=0.1,
        help=(
            "Exponential age decay k in weight=exp(-k*age); larger values "
            "favor newer predictions."
        ),
    )
    parser.add_argument(
        "--temporal-ensemble-max-history",
        type=int,
        default=None,
        help=(
            "Action horizon in environment steps for temporal aggregation; old chunks "
            "expire after this many steps. Defaults to the model chunk length."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--noise-mode",
        choices=("fixed_per_query", "rollout"),
        default="fixed_per_query",
        help=(
            "fixed_per_query reuses one flow-noise template for a deterministic "
            "conditional policy; rollout consumes a new noise sample per query."
        ),
    )
    parser.add_argument(
        "--execution-mode",
        choices=("raw", "rate_limited"),
        default="raw",
        help=(
            "raw evaluates model commands without hard motion limits. rate_limited "
            "constrains wrist and decoded-joint velocity/acceleration to the training "
            "distribution and logs interventions."
        ),
    )
    parser.add_argument(
        "--chunk-velocity-smoothing-weight",
        type=float,
        default=None,
        help="Quadratic first-difference penalty applied inside each predicted chunk.",
    )
    parser.add_argument(
        "--chunk-acceleration-smoothing-weight",
        type=float,
        default=None,
        help="Quadratic second-difference penalty applied inside each predicted chunk.",
    )
    parser.add_argument(
        "--clip-normalized-wrist-action",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Clamp normalized action.eef_position/action.eef_rotation to [-1,1] "
            "before inverse normalization; z_gesture is never clipped here."
        ),
    )
    parser.add_argument(
        "--bounded-wrist-euler",
        action="store_true",
        help=(
            "Restore the legacy [-pi,pi] MuJoCo wrist hinge limits. By default "
            "wrist Euler joints are continuous so nearest-branch commands can "
            "cross +/-pi without clipping or taking a full-turn path."
        ),
    )
    parser.add_argument(
        "--control-reference-dataset",
        type=Path,
        default=None,
        help=(
            "LeRobot dataset used for initialization and deployment limits. "
            "Defaults to the matching 2-camera dataset for shadow_grasp_2cam_* "
            "checkpoints and the legacy core-bottle dataset otherwise."
        ),
    )
    parser.add_argument(
        "--cross-hand-initialization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For Gaia/Sharpa, initialize wrist and native joints from the "
            "matching Shadow LeRobot episode's first wrist + z_gesture. "
            "Shadow evaluation is unchanged."
        ),
    )
    parser.add_argument(
        "--wrist-action-source",
        choices=("model", "dataset"),
        default="model",
        help=(
            "model executes Being-H wrist predictions. dataset replaces only "
            "the executed wrist6 with action[0:6] from the matching Shadow "
            "episode while retaining model-predicted z_gesture/finger actions."
        ),
    )
    parser.add_argument(
        "--latent-observation-mode",
        choices=("encoded", "commanded"),
        default="encoded",
        help=(
            "For z_gesture checkpoints, encoded re-encodes the target hand's "
            "actual joints and commanded feeds back the previous selected latent "
            "action. For cross-hand physical-joint geometry baselines, encoded "
            "retargets actual target joints back to Shadow and commanded feeds "
            "back the previous selected canonical Shadow-joint action."
        ),
    )
    parser.add_argument("--rate-limit-quantile", type=float, default=None)
    parser.add_argument(
        "--native-joint-rate-limit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply target-hand native-joint velocity/acceleration limits after "
            "VAE decoding or geometry retargeting. Use "
            "--no-native-joint-rate-limit to disable only this limiter while "
            "keeping wrist rate limits, chunk smoothing, wrist clipping, and "
            "safety unchanged."
        ),
    )
    parser.add_argument(
        "--limit-z-gesture",
        action="store_true",
        help=(
            "Also rate-limit z_gesture before VAE decoding. Disabled by default "
            "because decoded native joints are already rate-limited; enable only "
            "to reproduce the legacy double-limited controller."
        ),
    )
    parser.add_argument(
        "--safety-mode",
        choices=("off", "hold", "terminate"),
        default=None,
        help=(
            "On an abnormal observation/action, off raises normally, hold keeps the "
            "current physical command for the remaining rollout, and terminate executes "
            "one hold command then ends the rollout."
        ),
    )
    parser.add_argument(
        "--safety-max-wrist-position-error-m", type=float, default=0.25
    )
    parser.add_argument(
        "--safety-max-wrist-rotation-error-deg", type=float, default=90.0
    )
    parser.add_argument("--safety-max-object-drop-m", type=float, default=0.30)
    parser.add_argument(
        "--safety-max-workspace-radius-m", type=float, default=0.60
    )
    parser.add_argument("--num-inference-timesteps", type=int, default=None)
    parser.add_argument(
        "--warmup-queries",
        type=int,
        default=2,
        help=(
            "Discard this many full VAE+Being-H queries before the first episode. "
            "The policy and RNG are reset afterward, so rollout actions are unchanged."
        ),
    )
    parser.add_argument("--disable-mpg", action="store_true")
    parser.add_argument("--mpg-refinement-iters", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Single-episode NPZ path or multi-episode rollout directory.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="Single-episode MP4 path or multi-episode video directory.",
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=None,
        help=(
            "Single-episode metadata JSON path or multi-episode metadata directory."
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Multi-episode summary JSON path; an automatic path is used by default.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/sim_evaluation",
        help="Directory used for automatic output paths.",
    )
    parser.add_argument(
        "--no-record-images",
        action="store_true",
        help="Do not keep rendered images in memory during evaluation.",
    )
    return parser.parse_args()


def apply_deployment_profile(args: argparse.Namespace) -> None:
    if args.deployment_profile == "safe_smooth":
        args.inference_mode = "sync"
        args.action_selection = "chunk"
        args.replan_every = 16
        args.noise_mode = "fixed_per_query"
        args.execution_mode = "rate_limited"
        if args.chunk_velocity_smoothing_weight is None:
            args.chunk_velocity_smoothing_weight = 1.0
        if args.chunk_acceleration_smoothing_weight is None:
            args.chunk_acceleration_smoothing_weight = 4.0
        if args.clip_normalized_wrist_action is None:
            args.clip_normalized_wrist_action = True
        if args.rate_limit_quantile is None:
            args.rate_limit_quantile = 0.995
        if args.safety_mode is None:
            args.safety_mode = "terminate"
    else:
        if args.chunk_velocity_smoothing_weight is None:
            args.chunk_velocity_smoothing_weight = 0.0
        if args.chunk_acceleration_smoothing_weight is None:
            args.chunk_acceleration_smoothing_weight = 0.0
        if args.clip_normalized_wrist_action is None:
            args.clip_normalized_wrist_action = False
        if args.rate_limit_quantile is None:
            args.rate_limit_quantile = 0.999
        if args.safety_mode is None:
            args.safety_mode = "off"
    print(
        "[deployment] "
        f"profile={args.deployment_profile} infer={args.inference_mode} "
        f"selection={args.action_selection} h={args.replan_every} "
        f"noise={args.noise_mode} execution={args.execution_mode} "
        f"chunk_v={args.chunk_velocity_smoothing_weight:g} "
        f"chunk_a={args.chunk_acceleration_smoothing_weight:g} "
        f"clip_wrist={args.clip_normalized_wrist_action} "
        f"native_q_limit={args.native_joint_rate_limit} "
        f"safety={args.safety_mode}"
    )


def apply_success_profile(args: argparse.Namespace) -> None:
    criteria = SUCCESS_PROFILES[args.success_profile]
    args.success_lift_m = float(criteria["lift_m"])
    args.success_frames = int(criteria["frames"])
    print(
        "[success criterion] "
        f"profile={args.success_profile} lift>={args.success_lift_m:g}m "
        f"for {args.success_frames} consecutive control frames"
    )


def resolve_joint_retargeting_mode(
    args: argparse.Namespace,
    direct_shadow_joints: bool,
) -> None:
    if args.joint_retargeting == "auto":
        args.joint_retargeting = (
            "geometry"
            if direct_shadow_joints and args.hand != "shadow_hand_right"
            else "none"
        )
    if args.geometry_action_chunk_mode == "auto":
        args.geometry_action_chunk_mode = (
            "batch"
            if args.joint_retargeting == "geometry"
            and args.inference_mode == "sync"
            and args.action_selection == "chunk"
            else "sequential"
        )
    print(
        "[joint retargeting] "
        f"mode={args.joint_retargeting} "
        f"profile={args.geometry_retargeting_profile} "
        f"action_chunk={args.geometry_action_chunk_mode}"
    )


def validate_inputs(args: argparse.Namespace) -> None:
    direct_shadow_joints = (
        args.data_config_name in JOINT_ACTION_DATA_CONFIG_NAMES
    )
    required_model_files = (
        args.model_path / "config.json",
        args.model_path / "shadow_grasp_posttrain_metadata.json",
    )
    for path in required_model_files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required checkpoint file: {path}")
    if not list(args.model_path.glob("*.safetensors")):
        raise FileNotFoundError(f"No .safetensors file found in {args.model_path}")
    if not direct_shadow_joints and not args.vae_checkpoint.is_file():
        raise FileNotFoundError(f"VAE checkpoint not found: {args.vae_checkpoint}")
    if args.scene_xml is not None and not args.scene_xml.is_file():
        raise FileNotFoundError(f"Scene XML not found: {args.scene_xml}")
    if not args.dataset.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {args.dataset}")
    if args.replan_every <= 0:
        raise ValueError("--replan-every must be positive")
    if args.success_lift_m <= 0:
        raise ValueError("success lift threshold must be positive")
    if args.success_frames <= 0:
        raise ValueError("success frame count must be positive")
    if args.chunk_velocity_smoothing_weight < 0:
        raise ValueError("--chunk-velocity-smoothing-weight must be non-negative")
    if args.chunk_acceleration_smoothing_weight < 0:
        raise ValueError("--chunk-acceleration-smoothing-weight must be non-negative")
    if (
        args.inference_mode == "async"
        and (
            args.chunk_velocity_smoothing_weight > 0
            or args.chunk_acceleration_smoothing_weight > 0
        )
    ):
        raise ValueError("Chunk smoothing currently requires --inference-mode sync")
    if not 0.0 < args.rate_limit_quantile <= 1.0:
        raise ValueError("--rate-limit-quantile must be in (0,1]")
    for option in (
        "safety_max_wrist_position_error_m",
        "safety_max_wrist_rotation_error_deg",
        "safety_max_object_drop_m",
        "safety_max_workspace_radius_m",
    ):
        if getattr(args, option) <= 0:
            raise ValueError(f"--{option.replace('_', '-')} must be positive")
    if (
        args.action_selection == "temporal_ensemble"
        and args.temporal_ensemble_max_history is not None
        and args.replan_every > args.temporal_ensemble_max_history
    ):
        raise ValueError(
            "--replan-every cannot exceed --temporal-ensemble-max-history"
        )
    if args.inference_mode == "async" and args.action_selection != "temporal_ensemble":
        raise ValueError(
            "--inference-mode async requires --action-selection temporal_ensemble"
        )
    if args.temporal_ensemble_decay < 0:
        raise ValueError("--temporal-ensemble-decay must be non-negative")
    if (
        args.temporal_ensemble_max_history is not None
        and args.temporal_ensemble_max_history <= 0
    ):
        raise ValueError("--temporal-ensemble-max-history must be positive")
    if args.num_inference_timesteps is not None and args.num_inference_timesteps <= 0:
        raise ValueError("--num-inference-timesteps must be positive")
    if args.warmup_queries < 0:
        raise ValueError("--warmup-queries must be non-negative")
    if args.mpg_refinement_iters is not None and args.mpg_refinement_iters < 0:
        raise ValueError("--mpg-refinement-iters must be non-negative")
    if args.geometry_max_iterations <= 0:
        raise ValueError("--geometry-max-iterations must be positive")
    if args.geometry_learning_rate <= 0:
        raise ValueError("--geometry-learning-rate must be positive")
    if args.geometry_tolerance <= 0:
        raise ValueError("--geometry-tolerance must be positive")
    if args.geometry_temporal_weight < 0:
        raise ValueError("--geometry-temporal-weight must be non-negative")
    if args.geometry_acceleration_weight < 0:
        raise ValueError("--geometry-acceleration-weight must be non-negative")
    needs_control_reference = (
        args.execution_mode == "rate_limited"
        or args.wrist_action_source == "dataset"
        or args.latent_observation_mode == "commanded"
        or args.data_config_name in TWO_CAMERA_DATA_CONFIG_NAMES
        or (args.hand != "shadow_hand_right" and args.cross_hand_initialization)
    )
    if needs_control_reference and not args.control_reference_dataset.exists():
        raise FileNotFoundError(
            f"Control reference dataset not found: {args.control_reference_dataset}"
        )
    if args.limit_z_gesture and args.execution_mode != "rate_limited":
        raise ValueError("--limit-z-gesture requires --execution-mode rate_limited")
    if not direct_shadow_joints and args.joint_retargeting != "none":
        raise ValueError(
            "--joint-retargeting is only applicable to a physical-joint checkpoint"
        )
    if direct_shadow_joints:
        model_hand = DIRECT_JOINT_CONFIG_HANDS[args.data_config_name]
        if model_hand != "shadow_hand_right":
            if args.hand != model_hand:
                raise ValueError(
                    f"Target-native joint checkpoint {args.data_config_name} "
                    f"must use --hand {model_hand}"
                )
            if args.joint_retargeting != "none":
                raise ValueError(
                    "Target-native joint checkpoints do not use geometry "
                    "retargeting; pass --joint-retargeting none"
                )
        elif args.hand != "shadow_hand_right" and args.joint_retargeting != "geometry":
            raise ValueError(
                "Cross-hand evaluation of a physical Shadow-joint checkpoint "
                "requires --joint-retargeting geometry"
            )
    if (
        direct_shadow_joints
        and args.hand == "shadow_hand_right"
        and args.joint_retargeting != "none"
    ):
        raise ValueError(
            "Shadow evaluation uses direct physical joints and requires "
            "--joint-retargeting none"
        )
    if args.geometry_action_chunk_mode == "batch" and (
        args.joint_retargeting != "geometry"
        or args.inference_mode != "sync"
        or args.action_selection != "chunk"
    ):
        raise ValueError(
            "--geometry-action-chunk-mode batch requires geometry "
            "retargeting with --inference-mode sync and "
            "--action-selection chunk"
        )
    if (
        direct_shadow_joints
        and DIRECT_JOINT_CONFIG_HANDS[args.data_config_name] == "shadow_hand_right"
        and args.latent_observation_mode == "commanded"
        and args.hand != "shadow_hand_right"
        and args.joint_retargeting != "geometry"
    ):
        raise ValueError(
            "Cross-hand physical-joint commanded feedback requires "
            "geometry retargeting"
        )
    if (
        direct_shadow_joints
        and DIRECT_JOINT_CONFIG_HANDS[args.data_config_name] == "shadow_hand_right"
        and args.latent_observation_mode == "commanded"
        and args.hand != "shadow_hand_right"
        and not args.cross_hand_initialization
    ):
        raise ValueError(
            "Geometry commanded feedback requires "
            "--cross-hand-initialization"
        )
    if direct_shadow_joints and args.limit_z_gesture:
        raise ValueError(
            "--limit-z-gesture is not applicable to a physical-joint checkpoint"
        )


def available_episode_indices(dataset: Path) -> list[int]:
    dataset = dataset.expanduser().resolve()
    if dataset.is_file() and dataset.suffix == ".jsonl":
        indices: list[int] = []
        with dataset.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    indices.append(int(json.loads(line)["episode_index"]))
        return list(dict.fromkeys(indices))

    if dataset.is_file() and dataset.suffix == ".npz":
        with np.load(dataset, allow_pickle=True) as payload:
            count = int(np.asarray(payload["object_id"]).shape[0])
        return list(range(count))

    manifest_path = dataset / "episode_manifest.csv"
    if dataset.is_dir() and manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            return list(
                dict.fromkeys(
                    int(row["lerobot_episode_index"])
                    for row in csv.DictReader(handle)
                )
            )

    raise ValueError(
        f"Cannot enumerate episodes from {dataset}; expected JSONL, NPZ, "
        "or a LeRobot directory containing episode_manifest.csv"
    )


def resolve_episode_indices(
    args: argparse.Namespace,
    available: list[int],
) -> list[int]:
    if args.all_episodes:
        selected = list(available)
    elif args.episode_range is not None:
        start, end = args.episode_range
        if start > end:
            raise ValueError(
                f"--episode-range requires A <= B, but received {start} {end}"
            )
        selected = list(range(start, end + 1))
    elif args.episode_indices is not None:
        selected = list(dict.fromkeys(args.episode_indices))
    elif args.episode is not None:
        selected = [args.episode]
    else:
        selected = [4]

    available_set = set(available)
    invalid = [index for index in selected if index not in available_set]
    if invalid:
        invalid_preview = invalid[:20]
        invalid_suffix = (
            f" ... ({len(invalid)} unavailable in total)" if len(invalid) > 20 else ""
        )
        if available:
            availability = (
                f"available index range is {min(available)}..{max(available)} "
                f"({len(available)} episodes)"
            )
        else:
            availability = "the dataset contains no episodes"
        raise ValueError(
            f"Episodes {invalid_preview}{invalid_suffix} are unavailable; {availability}"
        )
    if not selected:
        raise ValueError("At least one episode must be selected")
    return selected


def _batch_directory(path: Path, option: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.suffix:
        raise ValueError(
            f"In multi-episode mode {option} must be a directory, not a file"
        )
    return resolved


def automatic_output_root(args: argparse.Namespace) -> Path:
    normalization = {
        Q99_DATA_CONFIG_NAME: "q99",
        MINMAX_DATA_CONFIG_NAME: "minmax",
        WRIST_MINMAX_ZRAW_DATA_CONFIG_NAME: "wrist-minmax_zraw",
        WRIST_EULER_MINMAX_ZRAW_DATA_CONFIG_NAME: "wrist-euler-minmax-zraw",
        WRIST_ROT6D_MINMAX_ZRAW_DATA_CONFIG_NAME: "wrist-rot6d-minmax-zraw",
        TWO_CAMERA_WRIST_EULER_MINMAX_ZRAW_DATA_CONFIG_NAME: (
            "2cam-wrist-euler-minmax-zraw"
        ),
        TWO_CAMERA_WRIST_ROT6D_MINMAX_ZRAW_DATA_CONFIG_NAME: (
            "2cam-wrist-rot6d-minmax-zraw"
        ),
        TWO_CAMERA_WRIST_ROT6D_MINMAX_JOINTS_DATA_CONFIG_NAME: (
            "2cam-wrist-rot6d-minmax-joints"
        ),
        SHARPA_JOINT_DATA_CONFIG_NAME: "sharpa-2cam-wrist-rot6d-minmax-joints",
        GAIA_JOINT_DATA_CONFIG_NAME: "gaia-2cam-wrist-rot6d-minmax-joints",
        RAW_DATA_CONFIG_NAME: "raw",
    }[args.data_config_name]
    mpg = "nompg" if args.disable_mpg else "mpg"
    denoise = (
        str(args.num_inference_timesteps)
        if args.num_inference_timesteps is not None
        else "model"
    )
    execution_label = args.execution_mode
    if args.execution_mode == "rate_limited":
        if args.data_config_name in JOINT_ACTION_DATA_CONFIG_NAMES:
            z_mode = "direct-joints"
        else:
            z_mode = "zlimited" if args.limit_z_gesture else "zraw"
        native_q_mode = (
            "nativeq" if args.native_joint_rate_limit else "nativeq-off"
        )
        execution_label = (
            f"rate_limited-q{args.rate_limit_quantile:g}-{z_mode}-"
            f"{native_q_mode}"
        )
    wrist_coordinate_label = (
        "wrist-bounded" if args.bounded_wrist_euler else "wrist-unwrapped"
    )
    smoothing_label = (
        f"smooth-v{args.chunk_velocity_smoothing_weight:g}"
        f"-a{args.chunk_acceleration_smoothing_weight:g}"
    )
    safety_label = f"safety-{args.safety_mode}"
    success_label = (
        f"success-{args.success_profile}-lift{args.success_lift_m:g}"
        f"-f{args.success_frames}"
    )
    wrist_source_label = (
        "" if args.wrist_action_source == "model" else "_wrist-dataset"
    )
    if args.latent_observation_mode == "encoded":
        latent_observation_label = ""
    elif args.data_config_name in JOINT_ACTION_DATA_CONFIG_NAMES:
        latent_observation_label = (
            f"_jobs-{args.latent_observation_mode}"
        )
    else:
        latent_observation_label = (
            f"_zobs-{args.latent_observation_mode}"
        )
    joint_retargeting_label = (
        ""
        if args.joint_retargeting == "none"
        else (
            f"_retarget-{args.joint_retargeting}"
            f"-{args.geometry_retargeting_profile}"
            f"-{args.geometry_action_chunk_mode}"
        )
    )
    evaluation_config = (
        f"{args.deployment_profile}{wrist_source_label}"
        f"{latent_observation_label}{joint_retargeting_label}_"
        f"{normalization}_{args.inference_mode}_"
        f"{args.action_selection}_h{args.replan_every}_{args.noise_mode}_"
        f"seed{args.seed}_{execution_label}_{smoothing_label}_"
        f"clipw-{int(args.clip_normalized_wrist_action)}_{safety_label}_"
        f"{success_label}_{wrist_coordinate_label}_{mpg}_d{denoise}"
    )
    return (
        args.output_dir.expanduser().resolve()
        / args.model_path.parent.name
        / args.model_path.name
        / args.hand
        / evaluation_config
    )


def automatic_output_paths(
    args: argparse.Namespace,
    episode_index: int,
    num_episodes: int,
) -> tuple[Path, Path, Path]:
    episode_directory_name = f"episode_{episode_index:06d}"
    default_episode_directory = (
        automatic_output_root(args) / episode_directory_name
    )

    if num_episodes == 1:
        output = (
            args.output.expanduser().resolve()
            if args.output is not None
            else default_episode_directory / "rollout.npz"
        )
        output_video = (
            args.output_video.expanduser().resolve()
            if args.output_video is not None
            else default_episode_directory / "rollout.mp4"
        )
        output_metadata = (
            args.output_metadata.expanduser().resolve()
            if args.output_metadata is not None
            else default_episode_directory / "metadata.json"
        )
    else:
        rollout_directory = (
            _batch_directory(args.output, "--output") / episode_directory_name
            if args.output is not None
            else default_episode_directory
        )
        video_directory = (
            _batch_directory(args.output_video, "--output-video")
            / episode_directory_name
            if args.output_video is not None
            else default_episode_directory
        )
        metadata_directory = (
            _batch_directory(args.output_metadata, "--output-metadata")
            / episode_directory_name
            if args.output_metadata is not None
            else default_episode_directory
        )
        output = rollout_directory / "rollout.npz"
        output_video = video_directory / "rollout.mp4"
        output_metadata = metadata_directory / "metadata.json"
    return output.resolve(), output_video.resolve(), output_metadata.resolve()


def warmup_closed_loop_policy(
    env: GraspEnv,
    policy: AuditedNativePolicy,
    num_queries: int,
    initial_action: np.ndarray | None = None,
) -> None:
    if num_queries <= 0:
        return
    observation, _ = env.reset(initial_action=initial_action)
    latencies: list[float] = []
    print(f"[warmup] running {num_queries} discarded full-policy queries...")
    for index in range(num_queries):
        # Reset before each request so chunk selection cannot serve a cached action.
        policy.reset()
        started_at = time.perf_counter()
        action = np.asarray(policy.predict(observation), dtype=np.float32)
        if not np.all(np.isfinite(action)):
            raise ValueError("Warm-up policy returned NaN or Inf")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started_at
        latencies.append(elapsed)
        print(
            f"[warmup] query={index + 1}/{num_queries} "
            f"latency={elapsed * 1000.0:.1f}ms",
            flush=True,
        )
    # Clear warm-up chunks, audits and RNG consumption before the formal rollout.
    policy.reset()
    print(f"[warmup] complete: {json.dumps(latency_summary_seconds(latencies))}")


def evaluate_episode(
    *,
    args: argparse.Namespace,
    episode_index: int,
    num_episodes: int,
    vae: NativeVAE | None,
    hand_model: Any,
    geometry_retargeter: GeometryRetargeter | None,
    latent_policy: BeingHGesturePolicy,
    policy: AuditedNativePolicy,
    rate_limiter: TrainingDistributionRateLimiter | None,
    q_step_limits: np.ndarray | None,
    q_acceleration_limits: np.ndarray | None,
    warmup_queries: int,
) -> dict[str, Any]:
    direct_shadow_joints = (
        args.data_config_name in JOINT_ACTION_DATA_CONFIG_NAMES
    )
    manifest_metadata = load_jsonl_metadata(args.dataset, episode_index)
    episode = load_dataset_object_episode(
        args.dataset,
        episode_index,
        source_dataset=args.source_dataset,
    )
    instruction, instruction_source, task_index = resolve_episode_instruction(
        instruction_override=args.instruction,
        episode_index=episode_index,
        manifest_metadata=manifest_metadata,
        control_reference_dataset=args.control_reference_dataset,
    )
    max_steps = args.max_steps
    if max_steps is None:
        max_steps = int(manifest_metadata.get("frames", 300))
    if max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    output, output_video, output_metadata = automatic_output_paths(
        args,
        episode_index,
        num_episodes,
    )
    initial_action, initialization_metadata = build_cross_hand_initial_action(
        args=args,
        episode_index=episode_index,
        manifest_metadata=manifest_metadata,
        vae=vae,
        geometry_retargeter=geometry_retargeter,
        policy_wrist_world_origin=latent_policy.policy_wrist_world_origin,
    )
    dataset_wrist_actions, wrist_action_metadata = (
        load_dataset_wrist_action_trajectory(
            args=args,
            episode_index=episode_index,
            manifest_metadata=manifest_metadata,
            required_steps=max_steps,
            policy_wrist_world_origin=latent_policy.policy_wrist_world_origin,
        )
    )
    policy.set_wrist_action_trajectory(dataset_wrist_actions)
    initial_commanded_z: np.ndarray | None = None
    if args.latent_observation_mode == "commanded":
        if direct_shadow_joints:
            reference_shadow_q = initialization_metadata.get(
                "reference_shadow_joint_position"
            )
            if reference_shadow_q is None:
                raise RuntimeError(
                    "Geometry commanded feedback requires reference "
                    "Shadow joints from cross-hand initialization"
                )
            initial_commanded_z = np.asarray(
                reference_shadow_q, dtype=np.float32
            )
            initialization_metadata[
                "joint_observation_reference"
            ] = "matching Shadow episode first observation.state[6:28]"
            print(
                "[joint observation] mode=commanded "
                "initial=shadow_reference_first_observation",
                flush=True,
            )
        else:
            reference_z_value = initialization_metadata.get(
                "reference_z_gesture"
            )
            if reference_z_value is None:
                initial_commanded_z, reference_metadata = (
                    load_reference_initial_z_gesture(
                        args=args,
                        episode_index=episode_index,
                        manifest_metadata=manifest_metadata,
                    )
                )
                initialization_metadata[
                    "latent_observation_reference"
                ] = reference_metadata
                initialization_metadata["reference_z_gesture"] = (
                    initial_commanded_z.tolist()
                )
            else:
                initial_commanded_z = np.asarray(
                    reference_z_value, dtype=np.float32
                )
            print(
                "[latent observation] mode=commanded "
                "initial=shadow_reference_first_observation",
                flush=True,
            )
    policy.set_initial_commanded_z(initial_commanded_z)

    latent_policy.instruction = instruction
    print("=" * 80)
    print(
        f"[episode] index={episode_index} object={episode.object_id} "
        f"scale={episode.scale:.6g} position={episode.position} "
        f"steps={max_steps}"
    )
    print(f"[instruction] {instruction} (source: {instruction_source})")

    primary_camera = (
        "ego_opposite"
        if "ego_opposite" in latent_policy.required_cameras
        else latent_policy.required_cameras[0]
    )
    env_config = GraspEnvConfig(
        hand=args.hand,
        scene_xml=args.scene_xml,
        camera=primary_camera,
        observation_cameras=latent_policy.required_cameras,
        width=320,
        height=240,
        fps=30,
        continuous_wrist_rotation=not args.bounded_wrist_euler,
        max_steps=max_steps,
        success_lift_m=args.success_lift_m,
        success_frames=args.success_frames,
        object_id=episode.object_id,
        object_scale=episode.scale,
        object_position=episode.position,
        object_quaternion=episode.quaternion,
    )
    with GraspEnv(env_config) as env:
        warmup_closed_loop_policy(
            env, policy, warmup_queries, initial_action=initial_action
        )
        result = PolicyEvaluationClient(env, policy).run(
            initial_action=initial_action,
            max_steps=max_steps,
            record_images=not args.no_record_images,
        )

    actual_initial_state = np.asarray(result.states[0], dtype=np.float32)
    actual_initial_z = np.empty((0,), dtype=np.float32)
    if initial_action is not None and not direct_shadow_joints:
        if vae is None:
            raise RuntimeError("z_gesture audit requires NativeVAE")
        actual_initial_z = (
            vae.encode(actual_initial_state[6:], args.hand)[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        reference_z = np.asarray(
            initialization_metadata["reference_z_gesture"], dtype=np.float32
        )
        target_policy_wrist = actual_initial_state[:6].copy()
        target_policy_wrist[0:3] -= latent_policy.policy_wrist_world_origin
        target_policy_wrist[3:6] += np.asarray(
            evaluation_policy_wrist_offset(args)
        )
        denominator = float(
            np.linalg.norm(reference_z) * np.linalg.norm(actual_initial_z)
        )
        initialization_metadata.update(
            {
                "initial_native_state_actual": actual_initial_state.tolist(),
                "initial_policy_wrist_actual": target_policy_wrist.tolist(),
                "initial_z_gesture_reencoded": actual_initial_z.tolist(),
                "z_cycle_l2": float(
                    np.linalg.norm(actual_initial_z - reference_z)
                ),
                "z_cycle_cosine": (
                    float(np.dot(reference_z, actual_initial_z) / denominator)
                    if denominator > 0
                    else 0.0
                ),
            }
        )
        print(
            "[cross-hand init] "
            f"actual_z_cycle_l2={initialization_metadata['z_cycle_l2']:.6f} "
            f"cosine={initialization_metadata['z_cycle_cosine']:.6f}",
            flush=True,
        )
    elif initial_action is not None:
        target_policy_wrist = actual_initial_state[:6].copy()
        target_policy_wrist[0:3] -= latent_policy.policy_wrist_world_origin
        target_policy_wrist[3:6] += np.asarray(
            evaluation_policy_wrist_offset(args)
        )
        reference_q = np.asarray(
            initialization_metadata["target_q_requested"],
            dtype=np.float32,
        )
        initialization_metadata.update(
            {
                "initial_native_state_actual": actual_initial_state.tolist(),
                "initial_policy_wrist_actual": target_policy_wrist.tolist(),
                "initial_target_joint_position_actual": (
                    actual_initial_state[6:].tolist()
                ),
                "joint_initialization_l2": float(
                    np.linalg.norm(actual_initial_state[6:] - reference_q)
                ),
            }
        )
        print(
            "[direct/retargeted joint init] "
            f"joint_l2={initialization_metadata['joint_initialization_l2']:.6f}",
            flush=True,
        )
    latent_policy.finish_episode()

    print(
        f"success={result.success}, steps={result.steps}, "
        f"max_lift={result.max_lift_m:.3f}m"
    )
    predicted_chunks = np.stack(latent_policy.predicted_chunks).astype(np.float32)
    smoothed_predicted_chunks = np.stack(
        latent_policy.smoothed_predicted_chunks
    ).astype(np.float32)
    observed_latent_states = np.stack(
        latent_policy.observed_latent_states
    ).astype(np.float32)
    encoded_native_latent_states = (
        np.stack(policy.policy.encoded_latent_states).astype(np.float32)
        if policy.policy.encoded_latent_states
        else np.empty((0, policy.policy.z_dim), dtype=np.float32)
    )
    policy_observation_latent_states = (
        np.stack(policy.policy.policy_latent_states).astype(np.float32)
        if policy.policy.policy_latent_states
        else np.empty((0, policy.policy.z_dim), dtype=np.float32)
    )
    if len(encoded_native_latent_states) != len(policy_observation_latent_states):
        raise RuntimeError(
            "Encoded and policy-observation latent audit lengths differ: "
            f"{len(encoded_native_latent_states)} != "
            f"{len(policy_observation_latent_states)}"
        )
    latent_observation_gap = (
        policy_observation_latent_states - encoded_native_latent_states
    )
    latent_observation_l2 = np.linalg.norm(latent_observation_gap, axis=1)
    if direct_shadow_joints:
        if args.hand == "shadow_hand_right":
            latent_observation_audit = {
                "mode": "direct_native_shadow_joints",
                "num_policy_steps": int(len(observed_latent_states)),
                "joint_state_source": "MuJoCo observation.state[6:28]",
            }
        else:
            latent_observation_audit = {
                "mode": args.latent_observation_mode,
                "representation": "canonical_shadow_joint_position",
                "num_policy_steps": int(
                    len(policy_observation_latent_states)
                ),
                "encoded_joint_state_source": (
                    f"geometry_retarget({args.hand} native joints"
                    "->shadow_hand_right native joints)"
                ),
                "policy_joint_state_source": (
                    "previous selected canonical Shadow-joint action"
                    if args.latent_observation_mode == "commanded"
                    else "encoded geometry-retargeted Shadow joints"
                ),
                "commanded_vs_encoded_l2_mean": (
                    float(latent_observation_l2.mean())
                    if len(latent_observation_l2)
                    else None
                ),
                "commanded_vs_encoded_l2_max": (
                    float(latent_observation_l2.max())
                    if len(latent_observation_l2)
                    else None
                ),
            }
    else:
        latent_observation_audit = {
            "mode": args.latent_observation_mode,
            "num_policy_steps": int(len(policy_observation_latent_states)),
            "commanded_vs_encoded_l2_mean": (
                float(latent_observation_l2.mean())
                if len(latent_observation_l2)
                else None
            ),
            "commanded_vs_encoded_l2_max": (
                float(latent_observation_l2.max())
                if len(latent_observation_l2)
                else None
            ),
        }
    latest_query_actions = np.stack(latent_policy.latest_query_actions).astype(np.float32)
    temporal_ensemble_actions = np.stack(
        latent_policy.temporal_ensemble_actions
    ).astype(np.float32)
    temporal_ensemble_candidate_counts = np.asarray(
        latent_policy.temporal_ensemble_candidate_counts,
        dtype=np.int64,
    )
    raw_selected_actions = np.stack(latent_policy.raw_selected_actions).astype(np.float32)
    branch_aligned_actions = np.stack(latent_policy.branch_aligned_actions).astype(np.float32)
    executed_latent_actions = np.stack(latent_policy.executed_latent_actions).astype(np.float32)
    filter_delta = executed_latent_actions - branch_aligned_actions
    intervention_steps = int(np.any(np.abs(filter_delta) > 1e-6, axis=1).sum())
    model_native_actions = np.stack(policy.model_native_actions).astype(np.float32)
    raw_native_actions = np.stack(policy.raw_native_actions).astype(np.float32)
    executed_native_actions = np.stack(policy.executed_native_actions).astype(np.float32)
    dataset_wrist_actions_requested = (
        np.stack(policy.dataset_wrist_actions_requested).astype(np.float32)
        if policy.dataset_wrist_actions_requested
        else np.empty((0, 6), dtype=np.float32)
    )
    native_filter_delta = executed_native_actions - raw_native_actions
    native_intervention_steps = int(
        np.any(np.abs(native_filter_delta) > 1e-6, axis=1).sum()
    )
    geometric_adapter = (
        policy.policy
        if isinstance(policy.policy, GeometricShadowJointPolicyAdapter)
        else None
    )
    retargeted_shadow_observation_joints = (
        np.stack(
            geometric_adapter.retargeted_shadow_observation_joints
        ).astype(np.float32)
        if geometric_adapter is not None
        and geometric_adapter.retargeted_shadow_observation_joints
        else np.empty((0, 22), dtype=np.float32)
    )
    model_shadow_action_joints = (
        np.stack(geometric_adapter.model_shadow_action_joints).astype(
            np.float32
        )
        if geometric_adapter is not None
        and geometric_adapter.model_shadow_action_joints
        else np.empty((0, 22), dtype=np.float32)
    )
    retargeted_target_action_joints = (
        np.stack(
            geometric_adapter.retargeted_target_action_joints
        ).astype(np.float32)
        if geometric_adapter is not None
        and geometric_adapter.retargeted_target_action_joints
        else np.empty(
            (0, len(hand_model.joint_names(args.hand))), dtype=np.float32
        )
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        states=result.states,
        actions=result.actions,
        object_poses=result.object_poses,
        observed_latent_states=observed_latent_states,
        observed_policy_states=observed_latent_states,
        encoded_native_latent_states=encoded_native_latent_states,
        policy_observation_latent_states=policy_observation_latent_states,
        policy_observation_minus_encoded_latent=latent_observation_gap,
        encoded_shadow_joint_states=(
            encoded_native_latent_states
            if direct_shadow_joints and args.hand != "shadow_hand_right"
            else np.empty((0, 22), dtype=np.float32)
        ),
        policy_observation_shadow_joint_states=(
            policy_observation_latent_states
            if direct_shadow_joints and args.hand != "shadow_hand_right"
            else np.empty((0, 22), dtype=np.float32)
        ),
        raw_predicted_chunks=predicted_chunks,
        smoothed_predicted_chunks=smoothed_predicted_chunks,
        query_step_indices=np.asarray(latent_policy.query_step_indices, dtype=np.int64),
        latest_query_latent_actions=latest_query_actions,
        temporal_ensemble_latent_actions=temporal_ensemble_actions,
        temporal_ensemble_candidate_counts=temporal_ensemble_candidate_counts,
        raw_selected_latent_actions=raw_selected_actions,
        branch_aligned_latent_actions=branch_aligned_actions,
        executed_latent_actions=executed_latent_actions,
        action_filter_delta=filter_delta,
        model_native_actions=model_native_actions,
        raw_native_actions=raw_native_actions,
        executed_native_actions=executed_native_actions,
        dataset_wrist_actions_requested=dataset_wrist_actions_requested,
        native_action_filter_delta=native_filter_delta,
        retargeted_shadow_observation_joints=(
            retargeted_shadow_observation_joints
        ),
        model_shadow_action_joints=model_shadow_action_joints,
        retargeted_target_action_joints=(
            retargeted_target_action_joints
        ),
        beingh_query_latency_s=np.asarray(
            latent_policy.query_latencies_s,
            dtype=np.float64,
        ),
        closed_loop_policy_step_latency_s=np.asarray(
            policy.predict_latencies_s,
            dtype=np.float64,
        ),
        initial_native_action_requested=np.asarray(
            initial_action if initial_action is not None else [],
            dtype=np.float32,
        ),
        initial_native_state_actual=actual_initial_state,
        initial_reference_z_gesture=np.asarray(
            initialization_metadata.get("reference_z_gesture") or [],
            dtype=np.float32,
        ),
        initial_observed_z_gesture=actual_initial_z,
        initial_reference_shadow_joint_position=np.asarray(
            initialization_metadata.get("reference_shadow_joint_position") or [],
            dtype=np.float32,
        ),
    )
    print(f"Saved rollout to {output}")

    saved_video_paths: dict[str, str] = {}
    if args.no_record_images:
        print("Video was not saved because --no-record-images was set.")
    else:
        images_by_camera = result.images_by_camera or {
            env_config.camera: result.images
        }
        for camera_name, camera_images in images_by_camera.items():
            camera_video = (
                output_video
                if camera_name == env_config.camera
                else output_video.with_name(
                    f"{output_video.stem}_{camera_name}{output_video.suffix}"
                )
            )
            write_mp4(camera_video, camera_images, env_config.fps)
            saved_video_paths[camera_name] = str(camera_video.resolve())
            print(f"Saved {camera_name} H.264 video to {camera_video}")

    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_path": str(args.model_path.resolve()),
        "deployment_profile": args.deployment_profile,
        "data_config_name": args.data_config_name,
        "data_config_source": args.data_config_source,
        "hand_action_representation": (
            latent_policy.hand_action_representation
        ),
        "shadow_joint_contract": getattr(
            args, "shadow_joint_contract", None
        ),
        "vae_checkpoint": (
            None if direct_shadow_joints else str(args.vae_checkpoint.resolve())
        ),
        "joint_retargeting": args.joint_retargeting,
        "geometry_retargeting": (
            None
            if geometric_adapter is None
            else geometric_adapter.retargeting_metadata()
        ),
        "dataset": str(args.dataset.resolve()),
        "episode": episode_index,
        "source_episode_index": episode.source_episode_index,
        "object_id": episode.object_id,
        "object_scale": episode.scale,
        "object_position": episode.position,
        "object_quaternion_wxyz": episode.quaternion,
        "instruction": instruction,
        "instruction_source": instruction_source,
        "task_index": task_index,
        "target_hand": args.hand,
        "target_hand_joint_names": list(hand_model.joint_names(args.hand)),
        "target_native_action_dim": 6 + len(
            hand_model.joint_names(args.hand)
        ),
        "policy_wrist_euler_offset": list(
            evaluation_policy_wrist_offset(args)
        ),
        "policy_wrist_world_origin": (
            latent_policy.policy_wrist_world_origin.tolist()
        ),
        "wrist_xyz_coordinate_contract": {
            "policy_to_world": "world_xyz = policy_xyz + wrist_world_origin",
            "world_to_policy": "policy_xyz = world_xyz - wrist_world_origin",
            "rpy": "absolute intrinsic-XYZ Euler; no translation",
        },
        "initialization": initialization_metadata,
        "latent_observation": latent_observation_audit,
        "wrist_action_source": args.wrist_action_source,
        "wrist_action_source_metadata": wrist_action_metadata,
        "camera": env_config.camera,
        "cameras": list(env_config.observation_cameras),
        "policy_video_source_columns": latent_policy.video_source_columns,
        "fps": env_config.fps,
        "max_steps": max_steps,
        "success_criterion": {
            "profile": args.success_profile,
            "lift_m": args.success_lift_m,
            "consecutive_control_frames": args.success_frames,
        },
        "warmup_queries_configured": args.warmup_queries,
        "warmup_queries_applied": warmup_queries,
        "replan_every": args.replan_every,
        "action_selection": args.action_selection,
        "inference_mode": args.inference_mode,
        "temporal_ensemble_query_interval": (
            args.replan_every
            if args.action_selection == "temporal_ensemble"
            else None
        ),
        "temporal_ensemble_decay": args.temporal_ensemble_decay,
        "temporal_ensemble_max_history": (
            latent_policy.temporal_ensemble_max_history
        ),
        "temporal_ensemble_candidate_count_max": int(
            temporal_ensemble_candidate_counts.max()
        ),
        "seed": args.seed,
        "noise_mode": args.noise_mode,
        "execution_mode": args.execution_mode,
        "chunk_smoothing": latent_policy.chunk_smoother.metadata(),
        "clip_normalized_wrist_action": args.clip_normalized_wrist_action,
        "continuous_wrist_rotation": not args.bounded_wrist_euler,
        "num_queries": len(predicted_chunks),
        "async_inference": {
            "submitted_queries": latent_policy.async_submitted_queries,
            "blocking_waits": latent_policy.async_blocking_waits,
            "blocking_wait_seconds": latent_policy.async_blocking_wait_seconds,
        },
        "timing": {
            "beingh_query": latency_summary_seconds(
                latent_policy.query_latencies_s
            ),
            "closed_loop_policy_step": latency_summary_seconds(
                policy.predict_latencies_s
            ),
        },
        "inference": {
            "num_inference_timesteps": int(
                latent_policy.policy.model.num_inference_timesteps
            ),
            "use_mpg": bool(latent_policy.policy.model.use_mpg),
            "mpg_refinement_iters": int(
                latent_policy.policy.model.mpg_refinement_iters
            ),
        },
        "rate_limiter": (
            None
            if rate_limiter is None
            else {
                **rate_limiter.metadata(),
                "decoded_native_joint_limiter": (
                    args.native_joint_rate_limit
                ),
            }
        ),
        "intervention_steps": intervention_steps,
        "native_joint_rate_limit": args.native_joint_rate_limit,
        "native_q_step_limits": None if q_step_limits is None else q_step_limits.tolist(),
        "native_q_acceleration_limits": (
            None
            if q_acceleration_limits is None
            else q_acceleration_limits.tolist()
        ),
        "native_intervention_steps": native_intervention_steps,
        "safety": {
            "mode": args.safety_mode,
            "events": policy.safety_events,
            "termination_requested": policy.termination_requested,
            "termination_reason": result.termination_reason,
            "max_wrist_position_error_m": (
                args.safety_max_wrist_position_error_m
            ),
            "max_wrist_rotation_error_deg": (
                args.safety_max_wrist_rotation_error_deg
            ),
            "max_object_drop_m": args.safety_max_object_drop_m,
            "max_workspace_radius_m": args.safety_max_workspace_radius_m,
        },
        "raw_selected_wrist_motion": wrist_motion_metrics(raw_selected_actions),
        "executed_wrist_motion": wrist_motion_metrics(executed_latent_actions),
        "model_native_wrist_motion": wrist_motion_metrics(model_native_actions),
        "dataset_wrist_motion": (
            wrist_motion_metrics(dataset_wrist_actions_requested)
            if len(dataset_wrist_actions_requested)
            else None
        ),
        "actual_wrist_motion": wrist_motion_metrics(result.states[:-1, :6]),
        "success": result.success,
        "steps": result.steps,
        "max_lift_m": result.max_lift_m,
        "rollout": str(output.resolve()),
        "video": (
            None
            if args.no_record_images
            else saved_video_paths.get(env_config.camera)
        ),
        "videos": saved_video_paths,
        "metadata": str(output_metadata.resolve()),
    }
    output_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    print(f"Saved metadata to {output_metadata}")
    return metadata


def main() -> None:
    args = parse_args()
    apply_deployment_profile(args)
    apply_success_profile(args)
    args.data_config_name, data_config_source = resolve_checkpoint_data_config(
        args.model_path,
        args.data_config_name,
        SUPPORTED_DATA_CONFIG_NAMES,
    )
    args.data_config_source = data_config_source
    direct_shadow_joints = (
        args.data_config_name in JOINT_ACTION_DATA_CONFIG_NAMES
    )
    resolve_joint_retargeting_mode(args, direct_shadow_joints)
    if args.dataset is None:
        args.dataset = (
            TWO_CAMERA_EVALUATION_DATASET
            if args.data_config_name in TWO_CAMERA_DATA_CONFIG_NAMES
            else LEGACY_EVALUATION_DATASET
        )
        print(f"[evaluation dataset] auto-selected {args.dataset}")
    if args.control_reference_dataset is None:
        args.control_reference_dataset = {
            SHARPA_JOINT_DATA_CONFIG_NAME: SHARPA_CONTROL_REFERENCE_DATASET,
            GAIA_JOINT_DATA_CONFIG_NAME: GAIA_CONTROL_REFERENCE_DATASET,
        }.get(
            args.data_config_name,
            TWO_CAMERA_CONTROL_REFERENCE_DATASET
            if args.data_config_name in TWO_CAMERA_DATA_CONFIG_NAMES
            else LEGACY_CONTROL_REFERENCE_DATASET,
        )
        print(
            "[control reference] auto-selected "
            f"{args.control_reference_dataset}"
        )
    validate_inputs(args)
    available = available_episode_indices(args.dataset)
    episode_indices = resolve_episode_indices(args, available)

    seed_everything(args.seed)
    include_xyz_equivalent = (
        args.data_config_name in ROT6D_DATA_CONFIG_NAMES
    )
    rate_limiter = None
    if args.execution_mode == "rate_limited":
        rate_limiter = TrainingDistributionRateLimiter.from_lerobot_dataset(
            args.control_reference_dataset,
            args.rate_limit_quantile,
            limit_z_gesture=args.limit_z_gesture,
            include_xyz_equivalent=include_xyz_equivalent,
            direct_native_joints=direct_shadow_joints,
        )
        print(f"[rate limiter] {json.dumps(rate_limiter.metadata())}")

    print(f"[episodes] {episode_indices}")
    geometry_retargeter: GeometryRetargeter | None = None
    vae: NativeVAE | None = None
    if direct_shadow_joints:
        print(
            "Loading geometry-only hand runtimes once for all selected "
            "episodes (NativeVAE weights are not loaded)..."
        )
        geometry_retargeter = GeometryRetargeter(
            device=args.device,
            config=GeometryRetargeterConfig(
                profile=args.geometry_retargeting_profile,
                max_iterations=args.geometry_max_iterations,
                learning_rate=args.geometry_learning_rate,
                tolerance=args.geometry_tolerance,
                temporal_weight=args.geometry_temporal_weight,
                acceleration_weight=args.geometry_acceleration_weight,
            ),
        )
        hand_model = geometry_retargeter
    else:
        print("Loading NativeVAE once for all selected episodes...")
        vae = NativeVAE.from_pretrained(
            checkpoint=args.vae_checkpoint,
            device=args.device,
        )
        hand_model = vae
    if args.hand not in hand_model.hand_names:
        raise ValueError(
            f"Hand runtime does not support {args.hand!r}; "
            f"available={hand_model.hand_names}"
        )
    print(
        f"[target hand] {args.hand}, "
        f"native joints={len(hand_model.joint_names(args.hand))}"
    )
    print(
        "[wrist frame] target Euler offset xyz="
        f"{evaluation_policy_wrist_offset(args).tolist()}; "
        "native->policy adds this offset, policy->native subtracts it",
        flush=True,
    )
    if direct_shadow_joints:
        model_hand = DIRECT_JOINT_CONFIG_HANDS[args.data_config_name]
        if model_hand == "shadow_hand_right":
            args.shadow_joint_contract = validate_shadow_joint_contract(
                args.control_reference_dataset,
                hand_model.joint_names("shadow_hand_right"),
            )
        else:
            joint_dim = DIRECT_JOINT_CONFIG_DIMS[args.data_config_name]
            args.shadow_joint_contract = {
                "dataset_to_mujoco_signs": [1.0] * joint_dim,
                "dataset_joint_names": list(hand_model.joint_names(model_hand)),
                "mujoco_joint_names": list(hand_model.joint_names(model_hand)),
                "naming": "target_native_identity",
            }
            print(
                f"[{model_hand} joint contract] native identity, dims={joint_dim}"
            )
    else:
        args.shadow_joint_contract = None

    first_manifest = load_jsonl_metadata(args.dataset, episode_indices[0])
    (
        initial_instruction,
        initial_instruction_source,
        _,
    ) = resolve_episode_instruction(
        instruction_override=args.instruction,
        episode_index=episode_indices[0],
        manifest_metadata=first_manifest,
        control_reference_dataset=args.control_reference_dataset,
    )
    print("Loading BeingHPolicy once for all selected episodes...")
    print(
        f"[initial instruction] {initial_instruction} "
        f"(source: {initial_instruction_source})"
    )
    print(
        f"[data config] {args.data_config_name} "
        f"(source: {data_config_source})"
    )
    chunk_smoother = ChunkTemporalSmoother(
        args.chunk_velocity_smoothing_weight,
        args.chunk_acceleration_smoothing_weight,
        include_xyz_equivalent=include_xyz_equivalent,
    )
    latent_policy = BeingHGesturePolicy(
        model_path=args.model_path,
        data_config_name=args.data_config_name,
        instruction=initial_instruction,
        device=args.device,
        replan_every=args.replan_every,
        seed=args.seed,
        noise_mode=args.noise_mode,
        action_selection=args.action_selection,
        inference_mode=args.inference_mode,
        temporal_ensemble_decay=args.temporal_ensemble_decay,
        temporal_ensemble_max_history=args.temporal_ensemble_max_history,
        rate_limiter=rate_limiter,
        chunk_smoother=chunk_smoother,
        clip_normalized_wrist_action=args.clip_normalized_wrist_action,
        use_mpg=False if args.disable_mpg else None,
        mpg_refinement_iters=args.mpg_refinement_iters,
        num_inference_timesteps=args.num_inference_timesteps,
    )
    print(
        f"[policy cameras] {latent_policy.video_source_columns} -> "
        f"MuJoCo {latent_policy.required_cameras}"
    )
    print(
        "[wrist coordinates] "
        f"world_origin={latent_policy.policy_wrist_world_origin.tolist()} "
        "world_xyz=policy_xyz+origin; RPY unchanged"
    )
    print(
        "[hand representation] "
        f"{latent_policy.hand_action_representation}"
    )
    print(
        "[latent observation] mode="
        + (
            (
                (
                    "commanded_shadow_joints"
                    if args.latent_observation_mode == "commanded"
                    else "direct_native_shadow_joints"
                )
                if args.hand == "shadow_hand_right"
                else (
                    "commanded_shadow_joints"
                    if args.latent_observation_mode == "commanded"
                    else "geometry_retargeted_target_to_shadow_joints"
                )
            )
            if direct_shadow_joints
            else args.latent_observation_mode
        )
    )
    print("Models loaded. Starting MuJoCo episode loop.")

    if direct_shadow_joints and (
        args.hand == "shadow_hand_right"
        or DIRECT_JOINT_CONFIG_HANDS[args.data_config_name] == args.hand
    ):
        gesture_policy = ShadowJointPolicyAdapter(
            latent_policy,
            target_hand=args.hand,
            joint_names=hand_model.joint_names(args.hand),
            dataset_to_mujoco_signs=np.asarray(
                args.shadow_joint_contract["dataset_to_mujoco_signs"],
                dtype=np.float32,
            ),
            policy_wrist_euler_offset=np.asarray(
                evaluation_policy_wrist_offset(args)
            ),
            policy_wrist_world_origin=(
                latent_policy.policy_wrist_world_origin
            ),
            observation_mode=args.latent_observation_mode,
        )
    elif direct_shadow_joints:
        if geometry_retargeter is None:
            raise RuntimeError("Geometry retargeter was not initialized")
        gesture_policy = GeometricShadowJointPolicyAdapter(
            latent_policy,
            retargeter=geometry_retargeter,
            target_hand=args.hand,
            dataset_to_mujoco_signs=np.asarray(
                args.shadow_joint_contract["dataset_to_mujoco_signs"],
                dtype=np.float32,
            ),
            policy_wrist_euler_offset=np.asarray(
                evaluation_policy_wrist_offset(args)
            ),
            policy_wrist_world_origin=(
                latent_policy.policy_wrist_world_origin
            ),
            action_chunk_mode=args.geometry_action_chunk_mode,
            observation_mode=args.latent_observation_mode,
        )
    else:
        if vae is None:
            raise RuntimeError("NativeVAE was not initialized")
        gesture_policy = GesturePolicyAdapter(
            latent_policy,
            vae=vae,
            target_hand=args.hand,
            encode_observation=True,
            latent_observation_mode=args.latent_observation_mode,
            policy_wrist_euler_offset=evaluation_policy_wrist_offset(args),
            policy_wrist_world_origin=latent_policy.policy_wrist_world_origin,
        )
    q_step_limits = None
    q_acceleration_limits = None
    if args.execution_mode == "rate_limited" and args.native_joint_rate_limit:
        (
            q_step_limits,
            q_acceleration_limits,
        ) = AuditedNativePolicy.q_motion_limits_from_lerobot_dataset(
            args.control_reference_dataset,
            vae,
            args.hand,
            args.rate_limit_quantile,
            direct_shadow_joints=direct_shadow_joints,
            geometry_retargeter=geometry_retargeter,
        )
        print(
            f"[native-q limiter] step_limits={q_step_limits.tolist()} "
            f"acceleration_limits={q_acceleration_limits.tolist()}"
        )
    elif args.execution_mode == "rate_limited":
        print(
            "[native-q limiter] disabled; wrist rate limiting remains enabled"
        )
    policy = AuditedNativePolicy(
        gesture_policy,
        q_step_limits,
        q_acceleration_limits,
        safety_mode=args.safety_mode,
        safety_max_wrist_position_error_m=(
            args.safety_max_wrist_position_error_m
        ),
        safety_max_wrist_rotation_error_deg=(
            args.safety_max_wrist_rotation_error_deg
        ),
        safety_max_object_drop_m=args.safety_max_object_drop_m,
        safety_max_workspace_radius_m=args.safety_max_workspace_radius_m,
    )

    episode_metadata: list[dict[str, Any]] = []
    try:
        for episode_index in episode_indices:
            episode_metadata.append(
                evaluate_episode(
                    args=args,
                    episode_index=episode_index,
                    num_episodes=len(episode_indices),
                    vae=vae,
                    hand_model=hand_model,
                    geometry_retargeter=geometry_retargeter,
                    latent_policy=latent_policy,
                    policy=policy,
                    rate_limiter=rate_limiter,
                    q_step_limits=q_step_limits,
                    q_acceleration_limits=q_acceleration_limits,
                    warmup_queries=(
                        args.warmup_queries if not episode_metadata else 0
                    ),
                )
            )
    finally:
        latent_policy.close()

    if len(episode_metadata) > 1:
        successes = [bool(item["success"]) for item in episode_metadata]
        successful_episode_indices = [
            int(item["episode"]) for item in episode_metadata if bool(item["success"])
        ]
        failed_episode_indices = [
            int(item["episode"]) for item in episode_metadata if not bool(item["success"])
        ]
        lifts = np.asarray(
            [float(item["max_lift_m"]) for item in episode_metadata],
            dtype=np.float64,
        )
        summary = {
            "model_path": str(args.model_path.resolve()),
            "data_config_name": args.data_config_name,
            "data_config_source": args.data_config_source,
            "hand_action_representation": (
                latent_policy.hand_action_representation
            ),
            "shadow_joint_contract": args.shadow_joint_contract,
            "vae_checkpoint": str(args.vae_checkpoint.resolve()),
            "dataset": str(args.dataset.resolve()),
            "episode_indices": episode_indices,
            "num_episodes": len(episode_metadata),
            "num_successes": len(successful_episode_indices),
            "num_failures": len(failed_episode_indices),
            "success_rate": float(np.mean(successes)),
            "successful_episode_indices": successful_episode_indices,
            "failed_episode_indices": failed_episode_indices,
            "mean_max_lift_m": float(lifts.mean()),
            "max_lift_m": float(lifts.max()),
            "total_steps": int(sum(int(item["steps"]) for item in episode_metadata)),
            "total_queries": int(
                sum(int(item["num_queries"]) for item in episode_metadata)
            ),
            "configuration": {
                "hand": args.hand,
                "replan_every": args.replan_every,
                "action_selection": args.action_selection,
                "inference_mode": args.inference_mode,
                "noise_mode": args.noise_mode,
                "execution_mode": args.execution_mode,
                "success_criterion": {
                    "profile": args.success_profile,
                    "lift_m": args.success_lift_m,
                    "consecutive_control_frames": args.success_frames,
                },
                "continuous_wrist_rotation": not args.bounded_wrist_euler,
                "wrist_action_source": args.wrist_action_source,
                "latent_observation_mode": args.latent_observation_mode,
                "cross_hand_initialization": args.cross_hand_initialization,
                "control_reference_dataset": str(
                    args.control_reference_dataset.resolve()
                ),
                "rate_limit_quantile": args.rate_limit_quantile,
                "limit_z_gesture": args.limit_z_gesture,
                "native_joint_rate_limit": args.native_joint_rate_limit,
                "seed": args.seed,
            },
            "episodes": episode_metadata,
        }
        if args.summary_output is None:
            summary_path = automatic_output_root(args) / (
                f"summary_episodes_{episode_indices[0]:06d}_"
                f"{episode_indices[-1]:06d}_n{len(episode_indices)}.json"
            )
        else:
            summary_path = args.summary_output.expanduser().resolve()
        if summary_path.suffix.lower() != ".json":
            summary_path = summary_path.with_suffix(".json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
            encoding="utf-8",
        )
        print("=" * 80)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))
        print(
            f"Successful episodes ({len(successful_episode_indices)}): "
            f"{successful_episode_indices}"
        )
        print(
            f"Failed episodes ({len(failed_episode_indices)}): "
            f"{failed_episode_indices}"
        )
        print(f"Saved batch summary to {summary_path}")


if __name__ == "__main__":
    main()
