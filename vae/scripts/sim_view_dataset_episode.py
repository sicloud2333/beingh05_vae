#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import mujoco


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim import GraspEnv, GraspEnvConfig, load_dataset_object_episode  # noqa: E402


HANDS = ("shadow_hand_right", "gaia_hand_right", "sharpa_hand_right")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open an exact LeRobot/source-NPZ object episode in MuJoCo."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--source_dataset", type=Path, default=None)
    parser.add_argument("--hand", choices=HANDS, default="shadow_hand_right")
    parser.add_argument("--camera", choices=["free", "ego", "ego_opposite", "wrist"], default="free")
    parser.add_argument("--base_scene", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--drive_mode", choices=["ctrl", "qpos"], default="ctrl")
    args = parser.parse_args()

    episode = load_dataset_object_episode(
        args.dataset,
        args.episode,
        source_dataset=args.source_dataset,
    )
    config_kwargs = dict(
        hand=args.hand,
        fps=args.fps,
        drive_mode=args.drive_mode,
        render_images=False,
        object_id=episode.object_id,
        object_scale=episode.scale,
        object_position=episode.position,
        object_quaternion=episode.quaternion,
    )
    if args.base_scene is not None:
        config_kwargs["base_scene"] = args.base_scene

    print(
        f"[episode] lerobot={episode.lerobot_episode_index} "
        f"source={episode.source_episode_index} object={episode.object_id} "
        f"scale={episode.scale:.6g} pos={episode.position} "
        f"quat_wxyz={episode.quaternion}"
    )
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

        while env.gui_is_running:
            started = time.perf_counter()
            observation, _, _, _, _ = env.step(hold_action)
            env.sync_gui()
            time.sleep(max(0.0, env.dt - (time.perf_counter() - started)))


if __name__ == "__main__":
    main()
