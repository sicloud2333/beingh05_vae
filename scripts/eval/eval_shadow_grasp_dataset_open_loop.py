#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
SUPPORTED_DATA_CONFIG_NAMES = (
    RAW_DATA_CONFIG_NAME,
    Q99_DATA_CONFIG_NAME,
    MINMAX_DATA_CONFIG_NAME,
    WRIST_MINMAX_ZRAW_DATA_CONFIG_NAME,
    WRIST_EULER_MINMAX_ZRAW_DATA_CONFIG_NAME,
    WRIST_ROT6D_MINMAX_ZRAW_DATA_CONFIG_NAME,
    TWO_CAMERA_WRIST_EULER_MINMAX_ZRAW_DATA_CONFIG_NAME,
    TWO_CAMERA_WRIST_ROT6D_MINMAX_ZRAW_DATA_CONFIG_NAME,
)
ROT6D_DATA_CONFIG_NAMES = (
    WRIST_ROT6D_MINMAX_ZRAW_DATA_CONFIG_NAME,
    TWO_CAMERA_WRIST_ROT6D_MINMAX_ZRAW_DATA_CONFIG_NAME,
)
DATASET_GROUP_NAME = "shadow_grasp_posttrain"
EMBODIMENT_TAG = "new_embodiment"
ACTION_DIM_NAMES = (
    "wrist_x",
    "wrist_y",
    "wrist_z",
    "wrist_rx",
    "wrist_ry",
    "wrist_rz",
    *(f"z_gesture_{index:02d}" for index in range(24)),
)
INSTRUCTION_TEMPLATE = (
    "According to the instruction '{task_description}', "
    "what's the micro-step actions in the next {k} steps?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fine-tuned Being-H Shadow policy against a recorded LeRobot "
            "trajectory without stepping a simulator."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help=(
            "Self-contained numeric checkpoint directory, or its parent training "
            "run directory (the latest numeric checkpoint is selected)."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=REPO_ROOT / "data/shadow_grasp_0725_core_bottle_1071",
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
            "shadow_grasp_2cam_* names select ego_opposite + wrist inputs; and "
            f"{RAW_DATA_CONFIG_NAME!r} for legacy raw checkpoints."
        ),
    )
    episode_group = parser.add_mutually_exclusive_group()
    episode_group.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help="Evaluate one episode. Episode 0 is used when no episode option is given.",
    )
    episode_group.add_argument(
        "--episode-indices",
        type=int,
        nargs="+",
        default=None,
        help="Evaluate multiple episode indices while loading the model only once.",
    )
    episode_group.add_argument(
        "--all-episodes",
        action="store_true",
        help="Evaluate every episode declared in meta/info.json.",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("teacher_forced_chunks", "first_chunk"),
        default="teacher_forced_chunks",
        help=(
            "teacher_forced_chunks re-queries from recorded observations every "
            "exec-horizon steps; first_chunk predicts only from the start frame."
        ),
    )
    parser.add_argument(
        "--exec-horizon",
        type=int,
        default=16,
        help="Number of predicted actions retained before the next query.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional query limit for a quick test.",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="Override the instruction stored in meta/tasks.jsonl.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--noise-mode",
        choices=("rollout", "fixed_per_query"),
        default="rollout",
        help=(
            "rollout consumes a new Flow/MPG noise sample for every query. "
            "fixed_per_query resets all RNGs before every query so observation "
            "changes, rather than noise changes, drive temporal differences."
        ),
    )
    parser.add_argument(
        "--num-inference-timesteps",
        type=int,
        default=None,
        help="Optional flow-matching inference step override.",
    )
    parser.add_argument(
        "--disable-mpg",
        action="store_true",
        help="Disable MPG at inference for a baseline diagnostic.",
    )
    parser.add_argument(
        "--mpg-refinement-iters",
        type=int,
        default=None,
        help="Optional MPG refinement-iteration override.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Single-episode .npz path or multi-episode output directory; an automatic path is used by default.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Single-episode PNG path or multi-episode plot directory; defaults alongside --output.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help=(
            "Batch summary JSON path. In multi-episode mode an automatic path "
            "under results/offline_inference is used by default."
        ),
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=180,
        help="DPI of the overview PNG.",
    )
    parser.add_argument(
        "--plot-scale-mode",
        choices=("dataset", "auto"),
        default="dataset",
        help=(
            "dataset keeps each dimension on a fixed dataset min/max scale "
            "across episodes and checkpoints; auto reproduces per-panel autoscaling."
        ),
    )
    parser.add_argument(
        "--plot-scale-padding",
        type=float,
        default=0.07,
        help="Fractional padding added to fixed dataset plot ranges.",
    )
    return parser.parse_args()


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "config.json").is_file() and (path / "model.safetensors").is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Model path does not exist: {path}")

    candidates = [
        child
        for child in path.iterdir()
        if child.is_dir()
        and child.name.isdigit()
        and (child / "config.json").is_file()
        and (child / "model.safetensors").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No self-contained checkpoint or numeric checkpoint directory found in {path}"
        )
    checkpoint = max(candidates, key=lambda item: int(item.name))
    print(f"Selected latest checkpoint: {checkpoint}")
    return checkpoint


def validate_checkpoint(checkpoint: Path) -> None:
    required = (
        checkpoint / "config.json",
        checkpoint / "model.safetensors",
        checkpoint / f"{DATASET_GROUP_NAME}_metadata.json",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required checkpoint file: {path}")


def load_info(dataset_path: Path) -> dict[str, Any]:
    info_path = dataset_path / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset info not found: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def episode_paths(
    dataset_path: Path,
    info: dict[str, Any],
    episode_index: int,
    video_source_columns: dict[str, str],
) -> tuple[Path, dict[str, Path]]:
    if episode_index < 0 or episode_index >= int(info["total_episodes"]):
        raise ValueError(
            f"episode-index must be in [0, {int(info['total_episodes']) - 1}]"
        )
    episode_chunk = episode_index // int(info["chunks_size"])
    format_values = {
        "episode_chunk": episode_chunk,
        "episode_index": episode_index,
    }
    parquet_path = dataset_path / info["data_path"].format(**format_values)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")

    video_paths: dict[str, Path] = {}
    features = info.get("features", {})
    for policy_key, source_column in video_source_columns.items():
        if source_column not in features:
            raise KeyError(
                f"Dataset does not provide required camera {source_column!r}; "
                f"checkpoint policy key={policy_key!r}"
            )
        video_values = dict(format_values, video_key=source_column)
        video_path = dataset_path / info["video_path"].format(**video_values)
        if not video_path.is_file():
            raise FileNotFoundError(
                f"Episode video for {policy_key!r} not found: {video_path}"
            )
        video_paths[policy_key] = video_path
    return parquet_path, video_paths


def policy_video_source_columns(policy: Any) -> dict[str, str]:
    keys = tuple(policy.data_config.VIDEO_KEYS)
    columns = policy.data_config.VIDEO_SOURCE_COLUMNS
    missing = [key for key in keys if key not in columns]
    if missing:
        raise KeyError(f"Data config has no source columns for video keys {missing}")
    return {key: str(columns[key]) for key in keys}


def load_instruction(
    dataset_path: Path,
    task_index: int,
    override: str | None,
) -> str:
    if override:
        return override

    tasks_path = dataset_path / "meta/tasks.jsonl"
    tasks: dict[int, str] = {}
    with tasks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            tasks[int(record["task_index"])] = str(record["task"])
    if task_index not in tasks:
        raise KeyError(f"task_index={task_index} not found in {tasks_path}")
    return tasks[task_index]


def source_state(raw_state: np.ndarray) -> np.ndarray:
    """Select raw wrist6 and z_gesture24 from the recorded 52-D state."""
    if raw_state.shape != (52,):
        raise ValueError(f"Expected recorded state [52], got {raw_state.shape}")
    return np.concatenate([raw_state[0:6], raw_state[28:52]]).astype(np.float32)


def source_actions(raw_actions: np.ndarray) -> np.ndarray:
    """Select raw wrist6 and z_gesture24 from recorded 52-D actions."""
    if raw_actions.ndim != 2 or raw_actions.shape[1] != 52:
        raise ValueError(f"Expected recorded actions [T,52], got {raw_actions.shape}")
    return np.concatenate(
        [raw_actions[:, 0:6], raw_actions[:, 28:52]],
        axis=-1,
    ).astype(np.float32)


def source_states(raw_states: np.ndarray) -> np.ndarray:
    """Select raw wrist6 and z_gesture24 from recorded 52-D states."""
    if raw_states.ndim != 2 or raw_states.shape[1] != 52:
        raise ValueError(f"Expected recorded states [T,52], got {raw_states.shape}")
    return np.concatenate(
        [raw_states[:, 0:6], raw_states[:, 28:52]],
        axis=-1,
    ).astype(np.float32)


def resolve_plot_value_ranges(
    dataset_path: Path,
    mode: str,
    padding_fraction: float,
) -> tuple[np.ndarray | None, str]:
    if mode == "auto":
        return None, "Per-panel autoscale (state + action GT + action pred)"

    stats_path = dataset_path / "meta/stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"Dataset plot scaling requires statistics at {stats_path}"
        )
    statistics = json.loads(stats_path.read_text(encoding="utf-8"))
    try:
        state_min = source_state(
            np.asarray(statistics["observation.state"]["min"], dtype=np.float32)
        )
        state_max = source_state(
            np.asarray(statistics["observation.state"]["max"], dtype=np.float32)
        )
        action_min = source_state(
            np.asarray(statistics["action"]["min"], dtype=np.float32)
        )
        action_max = source_state(
            np.asarray(statistics["action"]["max"], dtype=np.float32)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid plot statistics in {stats_path}: {error}") from error

    lower = np.minimum(state_min, action_min).astype(np.float64)
    upper = np.maximum(state_max, action_max).astype(np.float64)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError(f"Non-finite plot statistics in {stats_path}")

    span = upper - lower
    degenerate = span <= 0
    if degenerate.any():
        fallback = np.maximum(np.abs(lower) * 0.05, 1e-3)
        lower[degenerate] -= fallback[degenerate]
        upper[degenerate] += fallback[degenerate]
        span = upper - lower
    lower -= padding_fraction * span
    upper += padding_fraction * span
    ranges = np.stack([lower, upper], axis=-1)
    if ranges.shape != (len(ACTION_DIM_NAMES), 2):
        raise ValueError(f"Expected plot ranges [30,2], got {ranges.shape}")
    note = (
        "Fixed y-axis per dimension from dataset state/action min-max "
        f"(+{padding_fraction:.0%} padding); clipped predictions are counted"
    )
    return ranges, note


def remove_single_batch(value: Any, key: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(f"{key}: only batch size 1 is supported, got {array.shape}")
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{key}: expected [chunk,D], got {array.shape}")
    return array


def predict_chunk(
    policy: BeingHPolicy,
    raw_state: np.ndarray,
    images: dict[str, np.ndarray],
    instruction: str,
) -> np.ndarray:
    state = source_state(raw_state)
    observation = {
        "state.eef_position": state[None, 0:3],
        "state.eef_rotation": state[None, 3:6],
        "state.z_gesture": state[None, 6:30],
        "language.instruction": [instruction],
    }
    for video_key in policy.data_config.VIDEO_KEYS:
        if video_key not in images:
            raise KeyError(f"Missing required policy camera {video_key!r}")
        image = np.ascontiguousarray(
            np.asarray(images[video_key], dtype=np.uint8)
        )
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"{video_key}: expected HWC RGB image, got {image.shape}"
            )
        observation[video_key] = image[None, ...]
    result = policy.get_action(observation)
    position = remove_single_batch(result["action.eef_position"], "action.eef_position")
    rotation = remove_single_batch(result["action.eef_rotation"], "action.eef_rotation")
    gesture = remove_single_batch(result["action.z_gesture"], "action.z_gesture")
    chunk = np.concatenate([position, rotation, gesture], axis=-1)
    if chunk.shape[1] != 30:
        raise ValueError(f"Expected predicted action [chunk,30], got {chunk.shape}")
    if not np.isfinite(chunk).all():
        raise ValueError("Prediction contains NaN or Inf")
    return chunk.astype(np.float32)


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - target))))


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - target)))


def wrap_to_pi(values: np.ndarray) -> np.ndarray:
    """Wrap angular differences to [-pi, pi)."""
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def align_euler_to_reference(
    euler: np.ndarray,
    reference: np.ndarray,
    *,
    include_xyz_equivalent: bool = False,
) -> np.ndarray:
    """Choose the nearest intrinsic-XYZ branch for scalar plots/metrics."""
    if euler.shape != reference.shape or euler.shape[-1] != 3:
        raise ValueError(
            f"Euler/reference shapes must match and end in 3, got "
            f"{euler.shape} and {reference.shape}"
        )
    reference_unwrapped = np.unwrap(reference, axis=0)
    unwrap_offset = reference_unwrapped - reference
    primary = reference + wrap_to_pi(euler - reference) + unwrap_offset
    if not include_xyz_equivalent:
        return primary

    alternate_raw = euler.copy()
    alternate_raw[..., 0] += np.pi
    alternate_raw[..., 1] = np.pi - alternate_raw[..., 1]
    alternate_raw[..., 2] += np.pi
    alternate = (
        reference + wrap_to_pi(alternate_raw - reference) + unwrap_offset
    )
    use_alternate = np.linalg.norm(
        alternate - reference_unwrapped, axis=-1
    ) < np.linalg.norm(primary - reference_unwrapped, axis=-1)
    return np.where(use_alternate[..., None], alternate, primary)


def rotation_geodesic_radians(
    prediction_euler: np.ndarray,
    target_euler: np.ndarray,
) -> np.ndarray:
    """Intrinsic-XYZ orientation error, independent of Euler angle branches."""
    from scipy.spatial.transform import Rotation

    prediction_rotation = Rotation.from_euler("XYZ", prediction_euler)
    target_rotation = Rotation.from_euler("XYZ", target_euler)
    return (prediction_rotation.inv() * target_rotation).magnitude()


def compute_action_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    include_xyz_equivalent: bool = False,
) -> dict[str, Any]:
    expected_dim = len(ACTION_DIM_NAMES)
    if (
        prediction.shape != target.shape
        or prediction.ndim != 2
        or prediction.shape[1] != expected_dim
    ):
        raise ValueError(
            f"Expected matching action arrays [T,{expected_dim}], got "
            f"{prediction.shape} and {target.shape}"
        )

    # Euler coordinates are periodic. This aligned copy is useful for scalar
    # summaries, while the SO(3) geodesic below is the authoritative metric.
    aligned_prediction = prediction.copy()
    aligned_prediction[:, 3:6] = align_euler_to_reference(
        prediction[:, 3:6],
        target[:, 3:6],
        include_xyz_equivalent=include_xyz_equivalent,
    )
    aligned_target = target.copy()
    aligned_target[:, 3:6] = np.unwrap(target[:, 3:6], axis=0)

    rotation_error = rotation_geodesic_radians(
        prediction[:, 3:6], target[:, 3:6]
    )
    rotation_error_degrees = np.degrees(rotation_error)

    prediction_gesture = prediction[:, 6:30]
    target_gesture = target[:, 6:30]
    cosine_denominator = np.linalg.norm(prediction_gesture, axis=1) * np.linalg.norm(
        target_gesture, axis=1
    )
    valid_cosine = cosine_denominator > 1e-8
    if valid_cosine.any():
        gesture_cosine_mean = float(
            np.mean(
                np.sum(
                    prediction_gesture[valid_cosine] * target_gesture[valid_cosine],
                    axis=1,
                )
                / cosine_denominator[valid_cosine]
            )
        )
    else:
        gesture_cosine_mean = None

    per_dimension_rmse: dict[str, float] = {}
    per_dimension_mae: dict[str, float] = {}
    for dimension, name in enumerate(ACTION_DIM_NAMES):
        per_dimension_rmse[name] = rmse(
            aligned_prediction[:, dimension], aligned_target[:, dimension]
        )
        per_dimension_mae[name] = mae(
            aligned_prediction[:, dimension], aligned_target[:, dimension]
        )

    return {
        # Convenient scalar after resolving the common +/-pi branch issue.
        # The SO(3) geodesic remains the authoritative rotation metric.
        "rmse_all": rmse(aligned_prediction, aligned_target),
        "rmse_all_raw_euler": rmse(prediction, target),
        "mae_all": mae(aligned_prediction, aligned_target),
        "rmse_wrist_position": rmse(prediction[:, 0:3], target[:, 0:3]),
        "mae_wrist_position": mae(prediction[:, 0:3], target[:, 0:3]),
        "rmse_wrist_euler_wrapped": rmse(
            aligned_prediction[:, 3:6], aligned_target[:, 3:6]
        ),
        "rmse_wrist_euler_raw": rmse(prediction[:, 3:6], target[:, 3:6]),
        "rotation_geodesic_rad_mean": float(rotation_error.mean()),
        "rotation_geodesic_deg_mean": float(rotation_error_degrees.mean()),
        "rotation_geodesic_deg_median": float(np.median(rotation_error_degrees)),
        "rotation_geodesic_deg_p95": float(
            np.percentile(rotation_error_degrees, 95)
        ),
        "rotation_geodesic_deg_max": float(rotation_error_degrees.max()),
        "rmse_z_gesture": rmse(prediction_gesture, target_gesture),
        "mae_z_gesture": mae(prediction_gesture, target_gesture),
        "z_gesture_cosine_similarity_mean": gesture_cosine_mean,
        "per_dimension_rmse": per_dimension_rmse,
        "per_dimension_mae": per_dimension_mae,
        "metric_notes": {
            "rotation": (
                "SO(3) geodesic error is authoritative; raw Euler RMSE is "
                "retained only to expose +/-pi branch artifacts."
            ),
            "z_gesture": (
                "Latent-space error is a proxy. Decode predicted and target "
                "latents to physical joints for end-to-end hand evaluation."
            ),
        },
    }


def save_overview_plot(
    path: Path,
    frame_indices: np.ndarray,
    state: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    query_indices: list[int],
    episode_index: int,
    mode: str,
    dpi: int,
    value_ranges: np.ndarray | None,
    scale_note: str,
    include_xyz_equivalent: bool = False,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    expected_shape = (len(frame_indices), len(ACTION_DIM_NAMES))
    for key, value in (
        ("state", state),
        ("action GT", target),
        ("action pred", prediction),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"{key}: expected {expected_shape}, got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{key} contains NaN or Inf")
    if value_ranges is not None:
        value_ranges = np.asarray(value_ranges, dtype=np.float64)
        if value_ranges.shape != (len(ACTION_DIM_NAMES), 2):
            raise ValueError(
                f"Expected fixed plot ranges [30,2], got {value_ranges.shape}"
            )
        if not np.isfinite(value_ranges).all() or np.any(
            value_ranges[:, 1] <= value_ranges[:, 0]
        ):
            raise ValueError("Fixed plot ranges must be finite with upper > lower")

    # Plot Euler coordinates on the branch nearest to action GT. The physical
    # rotation quality is summarized with SO(3) geodesic error in the header.
    plot_state = state.copy()
    plot_target = target.copy()
    plot_prediction = prediction.copy()
    plot_state[:, 3:6] = align_euler_to_reference(
        state[:, 3:6],
        target[:, 3:6],
        include_xyz_equivalent=include_xyz_equivalent,
    )
    plot_prediction[:, 3:6] = align_euler_to_reference(
        prediction[:, 3:6],
        target[:, 3:6],
        include_xyz_equivalent=include_xyz_equivalent,
    )
    plot_target[:, 3:6] = np.unwrap(target[:, 3:6], axis=0)
    action_metrics = compute_action_metrics(
        prediction,
        target,
        include_xyz_equivalent=include_xyz_equivalent,
    )

    columns, rows = 6, 5
    cell_width, cell_height = 600, 500
    header_height = 185
    image = Image.new(
        "RGB",
        (columns * cell_width, header_height + rows * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(image)

    def load_font(size: int) -> ImageFont.ImageFont:
        candidates = (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        )
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                pass
        return ImageFont.load_default()

    title_font = load_font(34)
    summary_font = load_font(23)
    legend_font = load_font(25)
    subplot_font = load_font(20)
    tick_font = load_font(18)
    axis_font = load_font(17)
    colors = {
        "state": "#2ca02c",
        "action GT": "#1f77b4",
        "action pred": "#d62728",
    }

    def format_tick(value: float) -> str:
        """Keep dense subplot tick labels compact but numerically unambiguous."""
        if abs(value) < 5e-12:
            return "0"
        magnitude = abs(value)
        if magnitude >= 1e4 or magnitude < 1e-3:
            return f"{value:.2e}"
        return f"{value:.3g}"

    title = f"Shadow grasp offline inference | episode {episode_index} | {mode}"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(
        ((image.width - (title_box[2] - title_box[0])) / 2, 12),
        title,
        fill="black",
        font=title_font,
    )
    summary = (
        f"position RMSE={action_metrics['rmse_wrist_position']:.4g}  |  "
        f"rotation geodesic mean/p95/max="
        f"{action_metrics['rotation_geodesic_deg_mean']:.2f}/"
        f"{action_metrics['rotation_geodesic_deg_p95']:.2f}/"
        f"{action_metrics['rotation_geodesic_deg_max']:.2f} deg  |  "
        f"z RMSE={action_metrics['rmse_z_gesture']:.4g}"
    )
    summary_box = draw.textbbox((0, 0), summary, font=summary_font)
    draw.text(
        ((image.width - (summary_box[2] - summary_box[0])) / 2, 61),
        summary,
        fill="#333333",
        font=summary_font,
    )
    note = f"Euler branches aligned to action GT; {scale_note}"
    note_box = draw.textbbox((0, 0), note, font=summary_font)
    draw.text(
        ((image.width - (note_box[2] - note_box[0])) / 2, 96),
        note,
        fill="#666666",
        font=summary_font,
    )
    legend_y = 137
    legend_x = image.width // 2 - 360
    for label in ("state", "action GT", "action pred"):
        draw.line(
            (legend_x, legend_y + 12, legend_x + 65, legend_y + 12),
            fill=colors[label],
            width=5,
        )
        draw.text((legend_x + 78, legend_y), label, fill="black", font=legend_font)
        legend_x += 245

    frame_min = float(frame_indices[0])
    frame_max = float(frame_indices[-1])
    frame_span = max(frame_max - frame_min, 1.0)
    query_boundaries = set(query_indices[1:])

    for dimension, name in enumerate(ACTION_DIM_NAMES):
        row, column = divmod(dimension, columns)
        cell_left = column * cell_width
        cell_top = header_height + row * cell_height
        plot_left = cell_left + 82
        plot_right = cell_left + cell_width - 22
        plot_top = cell_top + 52
        plot_bottom = cell_top + cell_height - 58

        if value_ranges is None:
            values = np.concatenate(
                [
                    plot_state[:, dimension],
                    plot_target[:, dimension],
                    plot_prediction[:, dimension],
                ]
            )
            value_min = float(values.min())
            value_max = float(values.max())
            if value_max == value_min:
                padding = max(abs(value_min) * 0.05, 1e-3)
            else:
                padding = (value_max - value_min) * 0.07
            value_min -= padding
            value_max += padding
        else:
            value_min = float(value_ranges[dimension, 0])
            value_max = float(value_ranges[dimension, 1])
        value_span = value_max - value_min
        prediction_low_clips = int(
            np.count_nonzero(plot_prediction[:, dimension] < value_min)
        )
        prediction_high_clips = int(
            np.count_nonzero(plot_prediction[:, dimension] > value_max)
        )

        dimension_rmse = action_metrics["per_dimension_rmse"][name]
        metric_label = "RMSEwrap" if 3 <= dimension < 6 else "RMSE"
        unit = "m" if dimension < 3 else "rad" if dimension < 6 else "latent"
        draw.text(
            (cell_left + 12, cell_top + 12),
            (
                f"{dimension:02d}  {name}  "
                f"{metric_label}={dimension_rmse:.4g} [{unit}]"
            ),
            fill="black",
            font=subplot_font,
        )
        if prediction_low_clips or prediction_high_clips:
            draw.text(
                (cell_left + 12, cell_top + 32),
                f"pred clip: down={prediction_low_clips}, up={prediction_high_clips}",
                fill=colors["action pred"],
                font=tick_font,
            )
        draw.rectangle(
            (plot_left, plot_top, plot_right, plot_bottom),
            outline="#9a9a9a",
            width=1,
        )

        def x_coordinate(frame: float) -> float:
            return plot_left + (frame - frame_min) / frame_span * (
                plot_right - plot_left
            )

        def y_coordinate(value: float) -> float:
            clipped_value = min(max(value, value_min), value_max)
            return plot_bottom - (clipped_value - value_min) / value_span * (
                plot_bottom - plot_top
            )

        # Five labelled y ticks make the physical/latent scale readable on
        # every subplot. The labels share the exact limits used for clipping.
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            grid_y = plot_top + fraction * (plot_bottom - plot_top)
            if 0.0 < fraction < 1.0:
                draw.line(
                    (plot_left, grid_y, plot_right, grid_y),
                    fill="#e7e7e7",
                    width=1,
                )
            draw.line(
                (plot_left - 6, grid_y, plot_left, grid_y),
                fill="#777777",
                width=1,
            )
            tick_value = value_max - fraction * value_span
            tick_label = format_tick(tick_value)
            tick_box = draw.textbbox((0, 0), tick_label, font=tick_font)
            tick_width = tick_box[2] - tick_box[0]
            tick_height = tick_box[3] - tick_box[1]
            draw.text(
                (
                    plot_left - tick_width - 10,
                    grid_y - tick_height / 2 - tick_box[1],
                ),
                tick_label,
                fill="#444444",
                font=tick_font,
            )

        # Use actual recorded frame indices and avoid duplicate labels for
        # very short trajectories.
        x_tick_frames = sorted(
            {
                int(round(value))
                for value in np.linspace(frame_min, frame_max, num=5)
            }
        )
        for tick_frame in x_tick_frames:
            tick_x = x_coordinate(float(tick_frame))
            if frame_min < tick_frame < frame_max:
                draw.line(
                    (tick_x, plot_top, tick_x, plot_bottom),
                    fill="#eeeeee",
                    width=1,
                )
            draw.line(
                (tick_x, plot_bottom, tick_x, plot_bottom + 6),
                fill="#777777",
                width=1,
            )
            tick_label = str(tick_frame)
            tick_box = draw.textbbox((0, 0), tick_label, font=tick_font)
            tick_width = tick_box[2] - tick_box[0]
            label_x = min(
                max(tick_x - tick_width / 2, cell_left + 2),
                cell_left + cell_width - tick_width - 2,
            )
            draw.text(
                (label_x, plot_bottom + 9),
                tick_label,
                fill="#444444",
                font=tick_font,
            )
        for boundary in query_boundaries:
            if frame_min <= boundary <= frame_max:
                boundary_x = x_coordinate(float(boundary))
                draw.line(
                    (boundary_x, plot_top, boundary_x, plot_bottom),
                    fill="#c7c7c7",
                    width=1,
                )

        for label, series in (
            ("state", plot_state[:, dimension]),
            ("action GT", plot_target[:, dimension]),
            ("action pred", plot_prediction[:, dimension]),
        ):
            points = [
                (x_coordinate(float(frame)), y_coordinate(float(value)))
                for frame, value in zip(frame_indices, series, strict=True)
            ]
            if len(points) == 1:
                x, y = points[0]
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=colors[label])
            else:
                draw.line(points, fill=colors[label], width=3, joint="curve")

        axis_label = "frame index"
        axis_box = draw.textbbox((0, 0), axis_label, font=axis_font)
        draw.text(
            (
                (plot_left + plot_right - (axis_box[2] - axis_box[0])) / 2,
                plot_bottom + 34,
            ),
            axis_label,
            fill="#555555",
            font=axis_font,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", dpi=(dpi, dpi), optimize=True)

def resolve_episode_indices(
    args: argparse.Namespace,
    total_episodes: int,
) -> list[int]:
    if args.all_episodes:
        episode_indices = list(range(total_episodes))
    elif args.episode_indices is not None:
        episode_indices = list(dict.fromkeys(args.episode_indices))
    elif args.episode_index is not None:
        episode_indices = [args.episode_index]
    else:
        episode_indices = [0]

    invalid = [
        index for index in episode_indices
        if index < 0 or index >= total_episodes
    ]
    if invalid:
        raise ValueError(
            f"Episode indices {invalid} are outside [0, {total_episodes - 1}]"
        )
    if not episode_indices:
        raise ValueError("At least one episode must be selected")
    return episode_indices


def normalization_label(data_config_name: str) -> str:
    return {
        RAW_DATA_CONFIG_NAME: "raw",
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
    }[data_config_name]


def automatic_output_root(
    args: argparse.Namespace,
    checkpoint: Path,
    dataset_path: Path,
) -> Path:
    denoise = (
        str(args.num_inference_timesteps)
        if args.num_inference_timesteps is not None
        else "model"
    )
    refinement = (
        str(args.mpg_refinement_iters)
        if args.mpg_refinement_iters is not None
        else "model"
    )
    query_limit = "all" if args.max_queries is None else str(args.max_queries)
    mpg = "nompg" if args.disable_mpg else "mpg"
    evaluation_config = (
        f"{normalization_label(args.data_config_name)}_{args.mode}_"
        f"h{args.exec_horizon}_s{args.start_frame}_q{query_limit}_"
        f"{args.noise_mode}_seed{args.seed}_{mpg}_d{denoise}_r{refinement}_"
        f"scale-{args.plot_scale_mode}"
    )
    return (
        REPO_ROOT
        / "results/offline_inference"
        / checkpoint.parent.name
        / checkpoint.name
        / dataset_path.name
        / evaluation_config
    )


def episode_artifact_paths(
    args: argparse.Namespace,
    checkpoint: Path,
    dataset_path: Path,
    episode_index: int,
    num_episodes: int,
) -> tuple[Path, Path, Path]:
    episode_dir_name = f"episode_{episode_index:06d}"

    if args.output is None:
        episode_dir = (
            automatic_output_root(args, checkpoint, dataset_path)
            / episode_dir_name
        )
        output = episode_dir / "predictions.npz"
        metrics_output = episode_dir / "metrics.json"
    elif num_episodes == 1:
        # Preserve the historical single-episode explicit-file behavior.
        output = args.output.expanduser().resolve()
        if output.suffix != ".npz":
            output = output.with_suffix(".npz")
        metrics_output = output.with_suffix(".json")
    else:
        output_root = args.output.expanduser().resolve()
        if output_root.suffix:
            raise ValueError(
                "In multi-episode mode --output must be a directory, not a file"
            )
        episode_dir = output_root / episode_dir_name
        output = episode_dir / "predictions.npz"
        metrics_output = episode_dir / "metrics.json"

    if args.plot_output is None:
        plot_output = output.parent / "curves.png"
    elif num_episodes == 1:
        # Preserve the historical single-episode explicit-file behavior.
        plot_output = args.plot_output.expanduser().resolve()
        if plot_output.suffix.lower() != ".png":
            plot_output = plot_output.with_suffix(".png")
    else:
        plot_root = args.plot_output.expanduser().resolve()
        if plot_root.suffix:
            raise ValueError(
                "In multi-episode mode --plot-output must be a directory, not a file"
            )
        plot_output = plot_root / episode_dir_name / "curves.png"

    return (
        output.resolve(),
        metrics_output.resolve(),
        plot_output.resolve(),
    )


def evaluate_episode(
    *,
    args: argparse.Namespace,
    policy: Any,
    checkpoint: Path,
    dataset_path: Path,
    info: dict[str, Any],
    episode_index: int,
    num_episodes: int,
    pd: Any,
    torch: Any,
    get_all_frames: Any,
    plot_value_ranges: np.ndarray | None,
    plot_scale_note: str,
) -> dict[str, Any]:
    video_source_columns = policy_video_source_columns(policy)
    parquet_path, video_paths = episode_paths(
        dataset_path,
        info,
        episode_index,
        video_source_columns,
    )
    dataframe = pd.read_parquet(parquet_path)
    states = np.stack(dataframe["observation.state"].to_numpy()).astype(np.float32)
    selected_states = source_states(states)
    raw_actions = np.stack(dataframe["action"].to_numpy()).astype(np.float32)
    ground_truth = source_actions(raw_actions)
    frames_by_video_key = {
        video_key: get_all_frames(
            str(video_path),
            video_backend="torchvision_av",
        )
        for video_key, video_path in video_paths.items()
    }
    for video_key, frames in frames_by_video_key.items():
        if len(frames) < len(dataframe):
            raise ValueError(
                f"Episode {episode_index}: {video_key} video has "
                f"{len(frames)} frames, but parquet has {len(dataframe)} rows"
            )
    if args.start_frame < 0 or args.start_frame >= len(dataframe):
        raise ValueError(
            f"Episode {episode_index}: start-frame must be in "
            f"[0, {len(dataframe) - 1}]"
        )

    task_index = int(dataframe["task_index"].iloc[args.start_frame])
    instruction = load_instruction(dataset_path, task_index, args.instruction)
    print("=" * 80)
    print(f"Episode: {episode_index}, frames={len(dataframe)}")
    print(f"Instruction: {instruction}")
    print(f"Mode: {args.mode}, exec_horizon={args.exec_horizon}")
    print(f"Policy cameras: {list(video_source_columns.values())}")
    wrist_world_origin = np.asarray(
        getattr(policy.data_config, "WRIST_WORLD_ORIGIN", (0.0, 0.0, 0.0)),
        dtype=np.float32,
    )
    print(
        "Wrist xyz: recorded policy coordinates "
        f"relative to world origin {wrist_world_origin.tolist()} "
        "(open-loop performs no world translation)"
    )

    if args.mode == "first_chunk":
        query_indices = [args.start_frame]
    else:
        query_indices = list(
            range(args.start_frame, len(dataframe), args.exec_horizon)
        )
    if args.max_queries is not None:
        query_indices = query_indices[: args.max_queries]

    executed_predictions: list[np.ndarray] = []
    executed_targets: list[np.ndarray] = []
    executed_frame_indices: list[np.ndarray] = []
    predicted_chunks: list[np.ndarray] = []

    for query_number, frame_index in enumerate(query_indices, start=1):
        if args.noise_mode == "fixed_per_query":
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
        chunk = predict_chunk(
            policy,
            states[frame_index],
            {
                video_key: frames[frame_index]
                for video_key, frames in frames_by_video_key.items()
            },
            instruction,
        )
        predicted_chunks.append(chunk)
        execute_length = min(
            args.exec_horizon,
            len(chunk),
            len(dataframe) - frame_index,
        )
        executed_predictions.append(chunk[:execute_length])
        executed_targets.append(
            ground_truth[frame_index : frame_index + execute_length]
        )
        executed_frame_indices.append(
            np.arange(frame_index, frame_index + execute_length, dtype=np.int64)
        )
        print(
            f"episode {episode_index} query {query_number}/{len(query_indices)}: "
            f"frame={frame_index}, executed={execute_length}"
        )

    prediction = np.concatenate(executed_predictions, axis=0)
    target = np.concatenate(executed_targets, axis=0)
    frame_indices = np.concatenate(executed_frame_indices, axis=0)
    aligned_states = selected_states[frame_indices]
    chunks = np.stack(predicted_chunks, axis=0)

    metrics = {
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_path),
        "data_config_name": args.data_config_name,
        "data_config_source": args.data_config_source,
        "episode_index": episode_index,
        "start_frame": args.start_frame,
        "mode": args.mode,
        "exec_horizon": args.exec_horizon,
        "num_queries": len(query_indices),
        "num_evaluated_actions": len(prediction),
        "instruction": instruction,
        "video_source_columns": video_source_columns,
        "video_paths": {
            key: str(value) for key, value in video_paths.items()
        },
        "policy_wrist_world_origin": wrist_world_origin.tolist(),
        "wrist_xyz_coordinates": (
            "recorded policy coordinates; world translation is intentionally "
            "not applied during offline open-loop evaluation"
        ),
        "plot_scale": {
            "mode": args.plot_scale_mode,
            "padding_fraction": args.plot_scale_padding,
            "ranges": (
                None if plot_value_ranges is None else plot_value_ranges.tolist()
            ),
        },
        "inference_config": {
            "num_inference_timesteps": int(policy.model.num_inference_timesteps),
            "use_mpg": bool(policy.model.use_mpg),
            "mpg_refinement_iters": int(policy.model.mpg_refinement_iters),
            "seed": args.seed,
            "noise_mode": args.noise_mode,
        },
    }
    metrics.update(
        compute_action_metrics(
            prediction,
            target,
            include_xyz_equivalent=args.include_xyz_equivalent,
        )
    )

    output, metrics_path, plot_output = episode_artifact_paths(
        args,
        checkpoint,
        dataset_path,
        episode_index,
        num_episodes,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        predicted_actions=prediction,
        ground_truth_actions=target,
        states=aligned_states,
        frame_indices=frame_indices,
        query_indices=np.asarray(query_indices, dtype=np.int64),
        predicted_chunks=chunks,
        dimension_names=np.asarray(ACTION_DIM_NAMES),
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    save_overview_plot(
        path=plot_output,
        frame_indices=frame_indices,
        state=aligned_states,
        target=target,
        prediction=prediction,
        query_indices=query_indices,
        episode_index=episode_index,
        mode=args.mode,
        dpi=args.plot_dpi,
        value_ranges=plot_value_ranges,
        scale_note=plot_scale_note,
        include_xyz_equivalent=args.include_xyz_equivalent,
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved predictions: {output}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved overview plot: {plot_output}")
    return {
        "metrics": metrics,
        "prediction": prediction,
        "target": target,
        "output": output,
        "metrics_path": metrics_path,
        "plot_output": plot_output,
    }


def main() -> None:
    args = parse_args()
    if args.exec_horizon <= 0:
        raise ValueError("exec-horizon must be positive")
    if args.max_queries is not None and args.max_queries <= 0:
        raise ValueError("max-queries must be positive")
    if args.plot_dpi <= 0:
        raise ValueError("plot-dpi must be positive")
    if args.plot_scale_padding < 0:
        raise ValueError("plot-scale-padding must be non-negative")
    if args.num_inference_timesteps is not None and args.num_inference_timesteps <= 0:
        raise ValueError("num-inference-timesteps must be positive")
    if args.mpg_refinement_iters is not None and args.mpg_refinement_iters < 0:
        raise ValueError("mpg-refinement-iters must be non-negative")

    import pandas as pd
    import torch

    from BeingH.inference.beingh_policy import BeingHPolicy
    from BeingH.inference.checkpoint_data_config import (
        resolve_checkpoint_data_config,
    )
    from BeingH.utils.video_utils import get_all_frames

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint = resolve_checkpoint(args.model_path)
    validate_checkpoint(checkpoint)
    args.data_config_name, data_config_source = resolve_checkpoint_data_config(
        checkpoint,
        args.data_config_name,
        SUPPORTED_DATA_CONFIG_NAMES,
    )
    args.data_config_source = data_config_source
    args.include_xyz_equivalent = (
        args.data_config_name in ROT6D_DATA_CONFIG_NAMES
    )
    dataset_path = args.dataset_path.expanduser().resolve()
    info = load_info(dataset_path)
    episode_indices = resolve_episode_indices(args, int(info["total_episodes"]))
    plot_value_ranges, plot_scale_note = resolve_plot_value_ranges(
        dataset_path,
        args.plot_scale_mode,
        args.plot_scale_padding,
    )

    print(f"Checkpoint: {checkpoint}")
    print(f"Episodes: {episode_indices}")
    print(f"Data config: {args.data_config_name} (source: {data_config_source})")
    print(f"Plot scale: {plot_scale_note}")
    print("Loading BeingHPolicy once for all selected episodes...")
    policy = BeingHPolicy(
        model_path=str(checkpoint),
        data_config_name=args.data_config_name,
        dataset_name=DATASET_GROUP_NAME,
        embodiment_tag=EMBODIMENT_TAG,
        instruction_template=INSTRUCTION_TEMPLATE,
        device=args.device,
        enable_rtc=False,
        use_mpg=False if args.disable_mpg else None,
        mpg_refinement_iters=args.mpg_refinement_iters,
        num_inference_timesteps=args.num_inference_timesteps,
    )
    print("Model loaded. Starting episode loop.")
    if args.exec_horizon > policy.action_chunk_length:
        raise ValueError(
            f"exec-horizon={args.exec_horizon} exceeds model chunk length "
            f"{policy.action_chunk_length}"
        )

    results: list[dict[str, Any]] = []
    for episode_index in episode_indices:
        results.append(
            evaluate_episode(
                args=args,
                policy=policy,
                checkpoint=checkpoint,
                dataset_path=dataset_path,
                info=info,
                episode_index=episode_index,
                num_episodes=len(episode_indices),
                pd=pd,
                torch=torch,
                get_all_frames=get_all_frames,
                plot_value_ranges=plot_value_ranges,
                plot_scale_note=plot_scale_note,
            )
        )

    if len(results) > 1:
        all_predictions = np.concatenate(
            [result["prediction"] for result in results], axis=0
        )
        all_targets = np.concatenate(
            [result["target"] for result in results], axis=0
        )
        aggregate_metrics = compute_action_metrics(
            all_predictions,
            all_targets,
            include_xyz_equivalent=args.include_xyz_equivalent,
        )
        summary = {
            "checkpoint": str(checkpoint),
            "dataset": str(dataset_path),
            "episode_indices": episode_indices,
            "num_episodes": len(results),
            "num_evaluated_actions": int(len(all_predictions)),
            "mode": args.mode,
            "exec_horizon": args.exec_horizon,
            "inference_config": results[0]["metrics"]["inference_config"],
            "aggregate_metrics": aggregate_metrics,
            "episodes": [result["metrics"] for result in results],
        }
        if args.summary_output is None:
            summary_path = (
                automatic_output_root(args, checkpoint, dataset_path)
                / "summary.json"
            )
        else:
            summary_path = args.summary_output.expanduser().resolve()
        if summary_path.suffix.lower() != ".json":
            summary_path = summary_path.with_suffix(".json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("=" * 80)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Saved batch summary: {summary_path}")


if __name__ == "__main__":
    main()
