#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim import GraspEnv, GraspEnvConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset, step and render all grasp hands.")
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()

    for hand in ("shadow_hand_right", "gaia_hand_right", "sharpa_hand_right"):
        config = GraspEnvConfig(
            hand=hand,
            render_images=not args.no_render,
            max_steps=max(1, args.steps),
        )
        with GraspEnv(config) as env:
            observation, _ = env.reset()
            for _ in range(args.steps):
                observation, _, _, _, info = env.step(observation["state"])
            image_shape = None if observation["image"] is None else observation["image"].shape
            print(
                f"{hand}: action={env.action_dim}, state={observation['state'].shape}, "
                f"image={image_shape}, dt={info['dt']:.4f}s"
            )


if __name__ == "__main__":
    main()
