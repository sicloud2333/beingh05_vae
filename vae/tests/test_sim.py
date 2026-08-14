from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np

from sim import (
    GraspEnv,
    GraspEnvConfig,
    GesturePolicyAdapter,
    PolicyEvaluationClient,
    ReplayPolicy,
    load_dataset_object_episode,
)


HANDS = {
    "shadow_hand_right": 28,
    "gaia_hand_right": 21,
    "sharpa_hand_right": 28,
}


class GraspEnvTests(unittest.TestCase):
    def test_reset_and_step(self) -> None:
        for hand, action_dim in HANDS.items():
            with self.subTest(hand=hand):
                with GraspEnv(
                    GraspEnvConfig(hand=hand, render_images=False, max_steps=2)
                ) as env:
                    observation, info = env.reset()
                    self.assertEqual(observation["state"].shape, (action_dim,))
                    self.assertEqual(observation["object_pose"].shape, (7,))
                    self.assertEqual(observation["object_velocity"].shape, (6,))
                    self.assertIsNone(observation["image"])
                    self.assertEqual(observation["images"], {})
                    observation, _, _, _, info = env.step(observation["state"])
                    self.assertTrue(np.isfinite(observation["state"]).all())
                    self.assertEqual(info["action_dim"], action_dim)

    def test_gesture_policy_wrist_world_origin_round_trip(self) -> None:
        import torch
        from types import SimpleNamespace

        class FakeVAE:
            model = SimpleNamespace(
                gesture_encoder=SimpleNamespace(latent_dim=24)
            )

            @staticmethod
            def joint_names(hand: str) -> tuple[str, ...]:
                del hand
                return tuple(f"joint_{index}" for index in range(22))

            @staticmethod
            def encode(q: np.ndarray, hand: str) -> torch.Tensor:
                del hand
                return torch.zeros((len(q), 24), dtype=torch.float32)

            @staticmethod
            def decode(z: np.ndarray, hand: str) -> torch.Tensor:
                del hand
                return torch.zeros((len(z), 22), dtype=torch.float32)

        class EchoWristPolicy:
            def __init__(self) -> None:
                self.last_observation = None

            def reset(self) -> None:
                return None

            def predict(self, observation):
                self.last_observation = observation
                return np.concatenate(
                    [
                        np.asarray(observation["state"][:6], dtype=np.float32),
                        np.zeros(24, dtype=np.float32),
                    ]
                )

        policy = EchoWristPolicy()
        adapter = GesturePolicyAdapter(
            policy,
            vae=FakeVAE(),
            target_hand="shadow_hand_right",
            policy_wrist_world_origin=(0.0, 0.0, 0.4),
        )
        native_state = np.zeros(28, dtype=np.float32)
        native_state[:6] = [0.0, -0.3, 0.45, -1.2, 0.2, 2.8]
        action = adapter.predict({"state": native_state})

        np.testing.assert_allclose(
            policy.last_observation["state"][:6],
            [0.0, -0.3, 0.05, -1.2, 0.2, 2.8],
            atol=1e-6,
        )
        np.testing.assert_allclose(action[:6], native_state[:6], atol=1e-6)

    def test_commanded_latent_observation_uses_previous_latent_action(self) -> None:
        import torch
        from types import SimpleNamespace

        class FakeVAE:
            model = SimpleNamespace(
                gesture_encoder=SimpleNamespace(latent_dim=24)
            )

            @staticmethod
            def joint_names(hand: str) -> tuple[str, ...]:
                del hand
                return tuple(f"joint_{index}" for index in range(22))

            @staticmethod
            def encode(q: np.ndarray, hand: str) -> torch.Tensor:
                del hand
                return torch.full((len(q), 24), 9.0, dtype=torch.float32)

            @staticmethod
            def decode(z: np.ndarray, hand: str) -> torch.Tensor:
                del hand
                return torch.zeros((len(z), 22), dtype=torch.float32)

        class ConstantLatentPolicy:
            def __init__(self) -> None:
                self.observations: list[np.ndarray] = []

            def reset(self) -> None:
                return None

            def predict(self, observation):
                state = np.asarray(observation["state"], dtype=np.float32)
                self.observations.append(state.copy())
                return np.concatenate(
                    [state[:6], np.full(24, 3.0, dtype=np.float32)]
                )

        latent_policy = ConstantLatentPolicy()
        adapter = GesturePolicyAdapter(
            latent_policy,
            vae=FakeVAE(),
            target_hand="shadow_hand_right",
            latent_observation_mode="commanded",
        )
        initial_z = np.linspace(-1.0, 1.0, 24, dtype=np.float32)
        adapter.set_initial_commanded_z(initial_z)
        native_state = np.zeros(28, dtype=np.float32)

        adapter.predict({"state": native_state})
        adapter.predict({"state": native_state})
        np.testing.assert_allclose(
            latent_policy.observations[0][6:], initial_z, atol=1e-6
        )
        np.testing.assert_allclose(
            latent_policy.observations[1][6:], 3.0, atol=1e-6
        )

        adapter.reset()
        adapter.predict({"state": native_state})
        np.testing.assert_allclose(
            latent_policy.observations[2][6:], initial_z, atol=1e-6
        )
        self.assertEqual(len(adapter.encoded_latent_states), 1)
        self.assertEqual(len(adapter.policy_latent_states), 1)
        np.testing.assert_allclose(
            adapter.encoded_latent_states[0], 9.0, atol=1e-6
        )
        np.testing.assert_allclose(
            adapter.policy_latent_states[0], initial_z, atol=1e-6
        )

    def test_replay_policy_client(self) -> None:
        with GraspEnv(
            GraspEnvConfig(
                hand="shadow_hand_right",
                render_images=False,
                max_steps=2,
            )
        ) as env:
            initial, _ = env.reset()
            actions = np.repeat(initial["state"][None], 2, axis=0)
            result = PolicyEvaluationClient(env, ReplayPolicy(actions)).run(
                initial_action=actions[0],
                max_steps=2,
                record_images=False,
            )
        self.assertEqual(result.states.shape, (3, 28))
        self.assertEqual(result.actions.shape, (2, 28))
        self.assertEqual(result.object_poses.shape, (3, 7))

    def test_wrist_camera_is_attached_to_moving_wrist(self) -> None:
        import mujoco

        for hand in HANDS:
            with self.subTest(hand=hand):
                with GraspEnv(
                    GraspEnvConfig(hand=hand, render_images=False)
                ) as env:
                    camera_id = mujoco.mj_name2id(
                        env.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist"
                    )
                    wrist_body_id = mujoco.mj_name2id(
                        env.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_rz_link"
                    )
                    self.assertGreaterEqual(camera_id, 0)
                    self.assertGreaterEqual(wrist_body_id, 0)
                    self.assertEqual(
                        int(env.model.cam_bodyid[camera_id]), wrist_body_id
                    )

    def test_sharpa_wrist_camera_matches_shadow_policy_view(self) -> None:
        import mujoco

        policy_wrist = np.asarray(
            [0.0, -0.3, 0.45, -np.pi / 2.0, 0.0, np.pi],
            dtype=np.float64,
        )
        camera_poses = {}
        for hand, yaw_offset in (
            ("shadow_hand_right", 0.0),
            ("sharpa_hand_right", np.pi / 2.0),
        ):
            with GraspEnv(
                GraspEnvConfig(hand=hand, render_images=False)
            ) as env:
                native_wrist = policy_wrist.copy()
                native_wrist[5] -= yaw_offset
                env.data.qpos[env._qpos_adrs[:6]] = native_wrist
                mujoco.mj_forward(env.model, env.data)
                camera_id = mujoco.mj_name2id(
                    env.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist"
                )
                camera_poses[hand] = (
                    env.data.cam_xpos[camera_id].copy(),
                    env.data.cam_xmat[camera_id].copy(),
                )

        np.testing.assert_allclose(
            camera_poses["sharpa_hand_right"][0],
            camera_poses["shadow_hand_right"][0],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            camera_poses["sharpa_hand_right"][1],
            camera_poses["shadow_hand_right"][1],
            atol=1e-7,
        )

    def test_packaged_object_assets(self) -> None:
        with GraspEnv(
            GraspEnvConfig(
                hand="shadow_hand_right",
                object_id="core_mug_1038e4eac0e18dcce02ae6d2a21d494a",
                object_scale=0.06,
                object_position=(0.0, 0.0, 0.041),
                render_images=False,
                max_steps=1,
            )
        ) as env:
            self.assertGreater(env.model.nmesh, 1)
            observation, _ = env.reset()
            self.assertTrue(np.isfinite(observation["object_pose"]).all())

    def test_source_npz_object_episode(self) -> None:
        with TemporaryDirectory() as directory:
            dataset = Path(directory) / "episode.npz"
            np.savez(
                dataset,
                object_id=np.asarray(
                    ["core_bottle_1071fa4cddb2da2fc8724d5673a063a6"]
                ),
                object_scale=np.asarray([0.06], dtype=np.float32),
                object_rotmat=np.eye(3, dtype=np.float32)[None],
                object_world_xy=np.asarray([[0.01, -0.02]], dtype=np.float32),
            )
            episode = load_dataset_object_episode(dataset, 0)
        self.assertEqual(
            episode.object_id,
            "core_bottle_1071fa4cddb2da2fc8724d5673a063a6",
        )
        self.assertAlmostEqual(episode.scale, 0.06, places=5)
        self.assertEqual(episode.source_episode_index, 0)
        self.assertGreater(episode.position[2], 0.0)

    def test_portable_object_episode_manifest(self) -> None:
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / "episodes.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "episode_index": 7,
                        "source_episode_index": 11,
                        "object_id": "core_bottle_1071fa4cddb2da2fc8724d5673a063a6",
                        "scale": 0.08,
                        "position": [0.01, -0.02, 0.035],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            episode = load_dataset_object_episode(manifest, 7)
        self.assertEqual(episode.source_episode_index, 11)
        self.assertEqual(episode.position, (0.01, -0.02, 0.035))

    @unittest.skipUnless(
        os.environ.get("RUN_MUJOCO_RENDER_TESTS") == "1",
        "Set RUN_MUJOCO_RENDER_TESTS=1 with a working GL backend.",
    )
    def test_rgb_observation(self) -> None:
        for hand in HANDS:
            with self.subTest(hand=hand):
                with GraspEnv(
                    GraspEnvConfig(
                        hand=hand,
                        observation_cameras=("ego_opposite", "wrist"),
                        width=320,
                        height=240,
                    )
                ) as env:
                    observation, _ = env.reset()
                self.assertEqual(
                    tuple(observation["images"]), ("ego_opposite", "wrist")
                )
                self.assertTrue(
                    np.array_equal(
                        observation["image"],
                        observation["images"]["ego_opposite"],
                    )
                )
                for image in observation["images"].values():
                    self.assertEqual(image.shape, (240, 320, 3))
                    self.assertEqual(image.dtype, np.uint8)
                    self.assertGreater(float(image.std()), 1.0)


if __name__ == "__main__":
    unittest.main()
