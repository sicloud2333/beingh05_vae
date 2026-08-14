#!/usr/bin/env python3
"""Render identical policy states across hands and audit camera alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import mujoco
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


VAE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VAE_ROOT.parent
if str(VAE_ROOT) not in sys.path:
    sys.path.insert(0, str(VAE_ROOT))

from native_vae import NativeVAE  # noqa: E402
from sim import (  # noqa: E402
    GraspEnv,
    GraspEnvConfig,
    POLICY_WRIST_EULER_OFFSETS,
    load_dataset_object_episode,
)


HANDS = ("shadow_hand_right", "gaia_hand_right", "sharpa_hand_right")
CAMERAS = ("ego_opposite", "wrist")
HAND_COLORS = {
    "shadow_hand_right": (43, 131, 186),
    "gaia_hand_right": (230, 85, 13),
    "sharpa_hand_right": (35, 139, 69),
}
OBJECT_COLOR = (255, 215, 0)


def _episode_parquet(dataset: Path, episode: int) -> Path:
    meta_path = dataset / "meta/info.json"
    if meta_path.is_file():
        info = json.loads(meta_path.read_text(encoding="utf-8"))
        pattern = info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
        chunk_size = int(info.get("chunks_size", 1000))
    else:
        pattern = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        chunk_size = 1000
    return dataset / pattern.format(
        episode_chunk=episode // chunk_size, episode_index=episode
    )


def load_states(dataset: Path, episode: int) -> np.ndarray:
    path = _episode_parquet(dataset, episode)
    if not path.is_file():
        raise FileNotFoundError(path)
    column = pq.read_table(path, columns=["observation.state"])["observation.state"]
    values = np.asarray(column.combine_chunks().values, dtype=np.float32)
    states = values.reshape(-1, column.type.list_size)
    if states.shape[1] < 52:
        raise ValueError(f"Expected state width >=52, got {states.shape} from {path}")
    return states


def descendant_body_ids(model: mujoco.MjModel, root_name: str) -> set[int]:
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
    if root < 0:
        raise KeyError(f"Body {root_name!r} is absent")
    result = {int(root)}
    changed = True
    while changed:
        changed = False
        for body_id in range(1, model.nbody):
            if body_id not in result and int(model.body_parentid[body_id]) in result:
                result.add(body_id)
                changed = True
    return result


def geom_ids_for_bodies(model: mujoco.MjModel, body_ids: Iterable[int]) -> set[int]:
    body_ids = set(body_ids)
    return {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in body_ids
    }


def segmentation_mask(segmentation: np.ndarray, geom_ids: set[int]) -> np.ndarray:
    if segmentation.ndim != 3 or segmentation.shape[-1] != 2:
        raise ValueError(f"Expected segmentation [H,W,2], got {segmentation.shape}")
    return (
        (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
        & np.isin(segmentation[..., 0], np.asarray(sorted(geom_ids), dtype=np.int32))
    )


def mask_statistics(mask: np.ndarray) -> dict[str, Any]:
    y, x = np.nonzero(mask)
    if x.size == 0:
        return {
            "pixels": 0,
            "fraction": 0.0,
            "center_xy": None,
            "bbox_xyxy": None,
        }
    return {
        "pixels": int(x.size),
        "fraction": float(mask.mean()),
        "center_xy": [float(x.mean()), float(y.mean())],
        "bbox_xyxy": [int(x.min()), int(y.min()), int(x.max()), int(y.max())],
    }


def rotation_error_degrees(a: np.ndarray, b: np.ndarray) -> float:
    relative = np.asarray(a).reshape(3, 3).T @ np.asarray(b).reshape(3, 3)
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _render_rgb(env: GraspEnv, camera: str) -> np.ndarray:
    renderer = env._renderer
    if renderer is None:
        raise RuntimeError("Renderer is disabled")
    renderer.disable_segmentation_rendering()
    renderer.update_scene(env.data, camera=camera)
    return renderer.render().copy()


def _render_segmentation(env: GraspEnv, camera: str) -> np.ndarray:
    renderer = env._renderer
    if renderer is None:
        raise RuntimeError("Renderer is disabled")
    renderer.enable_segmentation_rendering()
    renderer.update_scene(env.data, camera=camera)
    value = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    return value


def _render_without_hand(
    env: GraspEnv, camera: str, hand_geom_ids: set[int]
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(sorted(hand_geom_ids), dtype=np.int32)
    original = env.model.geom_rgba[ids].copy()
    try:
        env.model.geom_rgba[ids, 3] = 0.0
        return _render_rgb(env, camera), _render_segmentation(env, camera)
    finally:
        env.model.geom_rgba[ids] = original


def _save_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


def _mask_image(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[mask] = color
    return image


def _overlay(
    rgb: np.ndarray, hand_mask: np.ndarray, object_mask: np.ndarray
) -> np.ndarray:
    out = rgb.astype(np.float32)
    for mask, color in ((hand_mask, (255, 0, 255)), (object_mask, OBJECT_COLOR)):
        out[mask] = 0.55 * out[mask] + 0.45 * np.asarray(color)
    return np.clip(out, 0, 255).astype(np.uint8)


def _camera_record(env: GraspEnv, camera: str) -> dict[str, Any]:
    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    return {
        "world_position": env.data.cam_xpos[camera_id].astype(float).tolist(),
        "world_rotation": env.data.cam_xmat[camera_id]
        .reshape(3, 3)
        .astype(float)
        .tolist(),
        "local_position": env.model.cam_pos[camera_id].astype(float).tolist(),
        "local_quaternion_wxyz": env.model.cam_quat[camera_id]
        .astype(float)
        .tolist(),
        "fovy_degrees": float(env.model.cam_fovy[camera_id]),
    }


def _native_action(
    state: np.ndarray,
    hand: str,
    vae: NativeVAE,
    wrist_world_origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    native_wrist = np.asarray(state[:6], dtype=np.float32).copy()
    native_wrist[:3] += wrist_world_origin
    native_wrist[3:6] -= np.asarray(
        POLICY_WRIST_EULER_OFFSETS[hand], dtype=np.float32
    )
    q = (
        vae.decode(np.array(state[28:52], dtype=np.float32, copy=True), hand)[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    return np.concatenate([native_wrist, q]), q


def _center_delta(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    if a["center_xy"] is None or b["center_xy"] is None:
        return None
    return float(
        np.linalg.norm(np.asarray(a["center_xy"]) - np.asarray(b["center_xy"]))
    )


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def _contact_sheet(
    path: Path,
    images: dict[str, np.ndarray],
    hands: tuple[str, ...],
    title: str,
) -> None:
    rows = ("rgb", "overlay", "rgb_no_hand")
    sample = next(iter(images.values()))
    height, width = sample.shape[:2]
    header, label_width = 42, 120
    canvas = Image.new(
        "RGB", (label_width + len(hands) * width, header + len(rows) * height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="black")
    for col, hand in enumerate(hands):
        draw.text((label_width + col * width + 5, 24), hand, fill="black")
    for row, kind in enumerate(rows):
        y = header + row * height
        draw.text((8, y + 8), kind, fill="black")
        for col, hand in enumerate(hands):
            canvas.paste(
                Image.fromarray(images[f"{hand}:{kind}"]),
                (label_width + col * width, y),
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the same dataset wrist/z_gesture with Shadow, Gaia and Sharpa, "
            "then compare camera geometry, masks, occlusion and RGB."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/shadow_grasp_bottle22249179_aug100_2cam",
    )
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 40, 80, 120])
    parser.add_argument("--hands", choices=HANDS, nargs="+", default=list(HANDS))
    parser.add_argument("--cameras", nargs="+", default=list(CAMERAS))
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=VAE_ROOT / "checkpoints/native_n2_epoch800_inference.pt",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument(
        "--wrist-world-origin", type=float, nargs=3, default=[0.0, 0.0, 0.4]
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/cross_hand_camera_diagnostics",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset = args.dataset.resolve()
    states = load_states(dataset, args.episode)
    frame_indices = sorted(set(args.frames))
    invalid = [index for index in frame_indices if not 0 <= index < len(states)]
    if invalid:
        raise IndexError(f"Frames {invalid} outside episode length {len(states)}")
    hands = tuple(args.hands)
    cameras = tuple(args.cameras)
    if "shadow_hand_right" not in hands:
        raise ValueError("Include shadow_hand_right as the comparison reference")

    episode = load_dataset_object_episode(dataset, args.episode)
    out = args.output_dir.resolve() / f"episode_{args.episode:06d}"
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[episode] {args.episode} frames={len(states)} object={episode.object_id} "
        f"scale={episode.scale:.6g} pos={episode.position}"
    )
    print(f"[VAE] loading once from {args.vae_checkpoint}")
    vae = NativeVAE.from_pretrained(args.vae_checkpoint, device=args.device)
    origin = np.asarray(args.wrist_world_origin, dtype=np.float32)
    report: dict[str, Any] = {
        "dataset": str(dataset),
        "episode": args.episode,
        "episode_frames": int(len(states)),
        "frames": {},
    }

    for frame in frame_indices:
        print(f"[frame {frame}] rendering {', '.join(hands)}")
        frame_record: dict[str, Any] = {
            "policy_wrist_xyz_euler": states[frame, :6].astype(float).tolist(),
            "z_gesture": states[frame, 28:52].astype(float).tolist(),
            "hands": {},
            "comparisons_to_shadow": {},
        }
        arrays: dict[str, dict[str, np.ndarray]] = {}
        sheets: dict[str, dict[str, np.ndarray]] = {camera: {} for camera in cameras}
        for hand in hands:
            action, decoded_q = _native_action(states[frame], hand, vae, origin)
            config = GraspEnvConfig(
                hand=hand,
                camera=cameras[0],
                observation_cameras=cameras,
                width=args.width,
                height=args.height,
                drive_mode="qpos",
                continuous_wrist_rotation=True,
                object_id=episode.object_id,
                object_scale=episode.scale,
                object_position=episode.position,
                object_quaternion=episode.quaternion,
            )
            hand_record: dict[str, Any] = {
                "native_wrist_xyz_euler": action[:6].astype(float).tolist(),
                "decoded_native_joints": decoded_q.astype(float).tolist(),
                "cameras": {},
            }
            arrays[hand] = {}
            with GraspEnv(config) as env:
                env.reset(initial_action=action)
                hand_geoms = geom_ids_for_bodies(
                    env.model, descendant_body_ids(env.model, "wrist_x_link")
                )
                visible_hand_geoms = {
                    geom_id
                    for geom_id in hand_geoms
                    if float(env.model.geom_rgba[geom_id, 3]) > 0.0
                }
                object_geoms = geom_ids_for_bodies(
                    env.model, descendant_body_ids(env.model, "object")
                )
                for camera in cameras:
                    rgb = _render_rgb(env, camera)
                    segmentation = _render_segmentation(env, camera)
                    hand_mask = segmentation_mask(segmentation, visible_hand_geoms)
                    object_mask = segmentation_mask(segmentation, object_geoms)
                    rgb_no_hand, segmentation_no_hand = _render_without_hand(
                        env, camera, hand_geoms
                    )
                    object_no_hand = segmentation_mask(
                        segmentation_no_hand, object_geoms
                    )
                    hand_stats = mask_statistics(hand_mask)
                    object_stats = mask_statistics(object_mask)
                    object_no_hand_stats = mask_statistics(object_no_hand)
                    no_hand_pixels = object_no_hand_stats["pixels"]
                    occlusion = (
                        max(
                            0.0,
                            1.0
                            - float(object_stats["pixels"]) / float(no_hand_pixels),
                        )
                        if no_hand_pixels
                        else None
                    )
                    camera_record = _camera_record(env, camera)
                    camera_record.update(
                        {
                            "hand_mask": hand_stats,
                            "object_mask": object_stats,
                            "object_mask_without_hand": object_no_hand_stats,
                            "object_occlusion_fraction": occlusion,
                        }
                    )
                    hand_record["cameras"][camera] = camera_record
                    arrays[hand][f"{camera}:rgb"] = rgb
                    arrays[hand][f"{camera}:rgb_no_hand"] = rgb_no_hand
                    arrays[hand][f"{camera}:hand_mask"] = hand_mask
                    arrays[hand][f"{camera}:object_mask"] = object_mask
                    overlay = _overlay(rgb, hand_mask, object_mask)
                    frame_dir = out / f"frame_{frame:06d}" / camera / hand
                    _save_image(frame_dir / "rgb.png", rgb)
                    _save_image(frame_dir / "rgb_no_hand.png", rgb_no_hand)
                    _save_image(frame_dir / "overlay.png", overlay)
                    _save_image(
                        frame_dir / "hand_mask.png",
                        _mask_image(hand_mask, HAND_COLORS[hand]),
                    )
                    _save_image(
                        frame_dir / "object_mask.png",
                        _mask_image(object_mask, OBJECT_COLOR),
                    )
                    sheets[camera][f"{hand}:rgb"] = rgb
                    sheets[camera][f"{hand}:overlay"] = overlay
                    sheets[camera][f"{hand}:rgb_no_hand"] = rgb_no_hand
            frame_record["hands"][hand] = hand_record

        shadow = frame_record["hands"]["shadow_hand_right"]
        for hand in hands:
            if hand == "shadow_hand_right":
                continue
            comparisons: dict[str, Any] = {}
            for camera in cameras:
                reference = shadow["cameras"][camera]
                current = frame_record["hands"][hand]["cameras"][camera]
                ref_rgb = arrays["shadow_hand_right"][f"{camera}:rgb"]
                rgb = arrays[hand][f"{camera}:rgb"]
                ref_no_hand = arrays["shadow_hand_right"][f"{camera}:rgb_no_hand"]
                no_hand = arrays[hand][f"{camera}:rgb_no_hand"]
                comparisons[camera] = {
                    "camera_position_error_mm": float(
                        1000.0
                        * np.linalg.norm(
                            np.asarray(current["world_position"])
                            - np.asarray(reference["world_position"])
                        )
                    ),
                    "camera_rotation_error_deg": rotation_error_degrees(
                        np.asarray(reference["world_rotation"]),
                        np.asarray(current["world_rotation"]),
                    ),
                    "fovy_error_deg": float(
                        current["fovy_degrees"] - reference["fovy_degrees"]
                    ),
                    "visible_object_center_error_px": _center_delta(
                        reference["object_mask"], current["object_mask"]
                    ),
                    "object_center_without_hand_error_px": _center_delta(
                        reference["object_mask_without_hand"],
                        current["object_mask_without_hand"],
                    ),
                    "hand_mask_iou": _mask_iou(
                        arrays["shadow_hand_right"][f"{camera}:hand_mask"],
                        arrays[hand][f"{camera}:hand_mask"],
                    ),
                    "rgb_mae_0_255": float(
                        np.abs(rgb.astype(np.float32) - ref_rgb.astype(np.float32)).mean()
                    ),
                    "rgb_without_hand_mae_0_255": float(
                        np.abs(
                            no_hand.astype(np.float32) - ref_no_hand.astype(np.float32)
                        ).mean()
                    ),
                }
            frame_record["comparisons_to_shadow"][hand] = comparisons

        for camera in cameras:
            _contact_sheet(
                out / f"frame_{frame:06d}" / f"{camera}_comparison.png",
                sheets[camera],
                hands,
                f"episode {args.episode} frame {frame} | {camera}",
            )
        report["frames"][str(frame)] = frame_record

    aggregates: dict[str, Any] = {}
    for hand in hands:
        if hand == "shadow_hand_right":
            continue
        aggregates[hand] = {}
        for camera in cameras:
            rows = [
                report["frames"][str(frame)]["comparisons_to_shadow"][hand][camera]
                for frame in frame_indices
            ]
            aggregates[hand][camera] = {}
            for key in rows[0]:
                values = [row[key] for row in rows if row[key] is not None]
                aggregates[hand][camera][key] = (
                    {
                        "mean": float(np.mean(values)),
                        "max": float(np.max(values)),
                    }
                    if values
                    else None
                )
    report["aggregate_comparisons_to_shadow"] = aggregates
    report_path = out / "diagnostics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nAggregate comparison to Shadow:")
    for hand, by_camera in aggregates.items():
        for camera, metrics in by_camera.items():
            object_center = metrics["object_center_without_hand_error_px"]
            object_text = (
                f"{object_center['mean']:.3f} px"
                if object_center is not None
                else "not visible"
            )
            print(
                f"  {hand:18s} {camera:12s} "
                f"camera={metrics['camera_position_error_mm']['max']:.3f} mm/"
                f"{metrics['camera_rotation_error_deg']['max']:.4f} deg, "
                f"object(no hand)={object_text}, "
                f"hand IoU={metrics['hand_mask_iou']['mean']:.3f}, "
                f"RGB(no hand) MAE={metrics['rgb_without_hand_mae_0_255']['mean']:.3f}"
            )
    print(f"Saved diagnostics to {report_path}")


if __name__ == "__main__":
    main()
