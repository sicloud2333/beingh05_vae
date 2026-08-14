#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim import GraspEnv, GraspEnvConfig, list_object_ids  # noqa: E402


HANDS = ("shadow_hand_right", "gaia_hand_right", "sharpa_hand_right")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open the standalone grasp environment in MuJoCo GUI."
    )
    parser.add_argument("--hand", choices=HANDS, default="shadow_hand_right")
    parser.add_argument("--scene_xml", type=Path, default=None)
    parser.add_argument("--base_scene", type=Path, default=None)
    parser.add_argument(
        "--camera",
        choices=["free", "ego", "ego_opposite", "wrist"],
        default="free",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--drive_mode", choices=["ctrl", "qpos"], default="ctrl")
    parser.add_argument("--home_wrist", type=float, nargs=6, default=None)
    object_group = parser.add_mutually_exclusive_group()
    object_group.add_argument("--object_id", type=str, default=None)
    object_group.add_argument("--object_mesh", type=Path, nargs="+", default=None)
    parser.add_argument("--list_objects", action="store_true")
    parser.add_argument("--object_scale", type=float, default=1.0)
    parser.add_argument("--object_position", type=float, nargs=3, default=[0, 0, 0.035])
    parser.add_argument(
        "--object_quaternion",
        type=float,
        nargs=4,
        default=[1, 0, 0, 0],
    )
    args = parser.parse_args()

    if args.list_objects:
        print("\n".join(list_object_ids()))
        return

    config_kwargs = dict(
        hand=args.hand,
        scene_xml=args.scene_xml,
        fps=args.fps,
        drive_mode=args.drive_mode,
        render_images=False,
        home_wrist=None if args.home_wrist is None else tuple(args.home_wrist),
        object_id=args.object_id,
        object_meshes=tuple(args.object_mesh or ()),
        object_scale=args.object_scale,
        object_position=tuple(args.object_position),
        object_quaternion=tuple(args.object_quaternion),
    )
    if args.base_scene is not None:
        config_kwargs["base_scene"] = args.base_scene

    with GraspEnv(GraspEnvConfig(**config_kwargs)) as env:
        observation, _ = env.reset()
        hold_action = observation["state"].copy()
        viewer = env.launch_gui()
        if args.camera != "free":
            camera_id = mujoco.mj_name2id(
                env.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                args.camera,
            )
            if camera_id < 0:
                raise KeyError(f"Camera {args.camera!r} is missing from the scene.")
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = camera_id

        print(
            f"[gui] hand={args.hand} action_dim={env.action_dim} "
            f"scene={env.scene_path}"
        )
        while env.gui_is_running:
            started = time.perf_counter()
            observation, _, _, _, _ = env.step(hold_action)
            env.sync_gui()
            time.sleep(max(0.0, env.dt - (time.perf_counter() - started)))


if __name__ == "__main__":
    main()
