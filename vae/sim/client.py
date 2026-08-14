from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from native_vae import NativeVAE

from .env import GraspEnv


class Policy(Protocol):
    def reset(self) -> None: ...

    def predict(self, observation: Mapping[str, Any]) -> np.ndarray: ...


class CallablePolicy:
    def __init__(self, function: Callable[[Mapping[str, Any]], Sequence[float]]) -> None:
        self.function = function

    def reset(self) -> None:
        return None

    def predict(self, observation: Mapping[str, Any]) -> np.ndarray:
        return np.asarray(self.function(observation), dtype=np.float32)


class ReplayPolicy:
    """Open-loop action replay using the same client interface as a learned policy."""

    def __init__(self, actions: Sequence[Sequence[float]] | np.ndarray) -> None:
        value = np.asarray(actions, dtype=np.float32)
        if value.ndim != 2:
            raise ValueError(f"Expected actions [T,D], got {value.shape}")
        if len(value) == 0:
            raise ValueError("Replay actions cannot be empty.")
        self.actions = value
        self._index = 0

    def reset(self) -> None:
        self._index = 0

    def predict(self, observation: Mapping[str, Any]) -> np.ndarray:
        del observation
        index = min(self._index, len(self.actions) - 1)
        self._index += 1
        return self.actions[index].copy()


class GesturePolicyAdapter:
    """Decode policy output ``[wrist 6D, z_gesture]`` to a target hand action."""

    def __init__(
        self,
        policy: Policy,
        vae: NativeVAE,
        target_hand: str,
        encode_observation: bool = True,
        policy_wrist_euler_offset: Sequence[float] | np.ndarray | None = None,
        policy_wrist_world_origin: Sequence[float] | np.ndarray | None = None,
        latent_observation_mode: str = "encoded",
    ) -> None:
        self.policy = policy
        self.vae = vae
        self.target_hand = target_hand
        self.encode_observation = encode_observation
        if latent_observation_mode not in {"encoded", "commanded"}:
            raise ValueError(
                "latent_observation_mode must be 'encoded' or 'commanded'"
            )
        self.latent_observation_mode = latent_observation_mode
        offset = (
            np.zeros(3, dtype=np.float32)
            if policy_wrist_euler_offset is None
            else np.asarray(policy_wrist_euler_offset, dtype=np.float32)
        )
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError(
                "policy_wrist_euler_offset must be finite Euler XYZ [3]"
            )
        self.policy_wrist_euler_offset = offset
        world_origin = (
            np.zeros(3, dtype=np.float32)
            if policy_wrist_world_origin is None
            else np.asarray(policy_wrist_world_origin, dtype=np.float32)
        )
        if world_origin.shape != (3,) or not np.all(np.isfinite(world_origin)):
            raise ValueError(
                "policy_wrist_world_origin must be a finite xyz vector [3]"
            )
        self.policy_wrist_world_origin = world_origin
        self.z_dim = int(vae.model.gesture_encoder.latent_dim)
        self._initial_commanded_z: np.ndarray | None = None
        self._commanded_z: np.ndarray | None = None
        self.encoded_latent_states: list[np.ndarray] = []
        self.policy_latent_states: list[np.ndarray] = []

    def set_initial_commanded_z(
        self, z_gesture: Sequence[float] | np.ndarray | None
    ) -> None:
        if z_gesture is None:
            self._initial_commanded_z = None
            self._commanded_z = None
            return
        values = np.asarray(z_gesture, dtype=np.float32)
        if values.shape != (self.z_dim,) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"Initial commanded z_gesture must be finite [{self.z_dim}], "
                f"got {values.shape}"
            )
        self._initial_commanded_z = values.copy()
        self._commanded_z = values.copy()

    def reset(self) -> None:
        self.policy.reset()
        self._commanded_z = (
            None
            if self._initial_commanded_z is None
            else self._initial_commanded_z.copy()
        )
        self.encoded_latent_states.clear()
        self.policy_latent_states.clear()

    def predict(self, observation: Mapping[str, Any]) -> np.ndarray:
        policy_observation = observation
        if self.encode_observation:
            native_state = np.asarray(observation["state"], dtype=np.float32)
            expected_native_dim = 6 + len(self.vae.joint_names(self.target_hand))
            if native_state.shape != (expected_native_dim,):
                raise ValueError(
                    f"Expected target-hand state [{expected_native_dim}], "
                    f"got {native_state.shape}"
                )
            encoded_z = (
                self.vae.encode(native_state[None, 6:], self.target_hand)[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            if encoded_z.shape != (self.z_dim,) or not np.all(
                np.isfinite(encoded_z)
            ):
                raise ValueError(
                    f"VAE encoded z_gesture must be finite [{self.z_dim}], "
                    f"got {encoded_z.shape}"
                )
            self.encoded_latent_states.append(encoded_z.copy())
            z_state = (
                self._commanded_z.copy()
                if self.latent_observation_mode == "commanded"
                and self._commanded_z is not None
                else encoded_z
            )
            self.policy_latent_states.append(z_state.copy())
            policy_wrist = native_state[:6].copy()
            policy_wrist[0:3] -= self.policy_wrist_world_origin
            policy_wrist[3:6] += self.policy_wrist_euler_offset
            policy_observation = dict(observation)
            policy_observation["native_state"] = native_state
            policy_observation["state"] = np.concatenate(
                [policy_wrist, z_state]
            )

        wrist_z = np.asarray(self.policy.predict(policy_observation), dtype=np.float32)
        expected = 6 + self.z_dim
        if wrist_z.shape != (expected,):
            raise ValueError(
                f"Gesture policy must output [wrist6,z{self.z_dim}] ({expected} values), "
                f"got {wrist_z.shape}"
            )
        if not np.all(np.isfinite(wrist_z)):
            raise ValueError("Gesture policy output contains NaN or Inf")
        if self.latent_observation_mode == "commanded":
            self._commanded_z = wrist_z[6:].copy()
        target_q = self.vae.decode(wrist_z[None, 6:], self.target_hand)[0]
        target_wrist = wrist_z[:6].copy()
        target_wrist[0:3] += self.policy_wrist_world_origin
        target_wrist[3:6] -= self.policy_wrist_euler_offset
        return np.concatenate(
            [target_wrist, target_q.detach().cpu().numpy().astype(np.float32)]
        )


@dataclass(frozen=True)
class EvaluationResult:
    success: bool
    steps: int
    max_lift_m: float
    states: np.ndarray
    actions: np.ndarray
    object_poses: np.ndarray
    images: tuple[np.ndarray, ...]
    termination_reason: str | None = None
    images_by_camera: Mapping[str, tuple[np.ndarray, ...]] | None = None


class PolicyEvaluationClient:
    """Run a policy against :class:`GraspEnv` and collect rollout observations."""

    def __init__(self, env: GraspEnv, policy: Policy) -> None:
        self.env = env
        self.policy = policy

    def run(
        self,
        *,
        initial_action: Sequence[float] | np.ndarray | None = None,
        object_pose: Sequence[float] | np.ndarray | None = None,
        max_steps: int | None = None,
        record_images: bool = True,
    ) -> EvaluationResult:
        self.policy.reset()
        observation, info = self.env.reset(
            initial_action=initial_action,
            object_pose=object_pose,
        )
        states: list[np.ndarray] = [observation["state"]]
        actions: list[np.ndarray] = []
        object_poses: list[np.ndarray] = [observation["object_pose"]]
        images: list[np.ndarray] = []
        images_by_camera: dict[str, list[np.ndarray]] = {}

        def record_observation_images(current: Mapping[str, Any]) -> None:
            if not record_images:
                return
            if current["image"] is not None:
                images.append(current["image"])
            for camera_name, frame in current.get("images", {}).items():
                images_by_camera.setdefault(camera_name, []).append(frame)

        record_observation_images(observation)

        budget = int(max_steps or self.env.config.max_steps)
        for _ in range(budget):
            action = np.asarray(self.policy.predict(observation), dtype=np.float32)
            observation, _, terminated, truncated, info = self.env.step(action)
            # Record the actual target after GraspEnv validation/limit coercion,
            # rather than the pre-environment request.
            actions.append(self.env.last_action)
            states.append(observation["state"])
            object_poses.append(observation["object_pose"])
            record_observation_images(observation)
            safety_termination = bool(
                getattr(self.policy, "termination_requested", False)
            )
            if terminated or truncated or safety_termination:
                break

        action_array = np.stack(actions) if actions else np.empty((0, self.env.action_dim))
        return EvaluationResult(
            success=bool(info["success"]),
            steps=len(actions),
            max_lift_m=float(info["max_lift_m"]),
            states=np.stack(states),
            actions=action_array.astype(np.float32),
            object_poses=np.stack(object_poses),
            images=tuple(images),
            termination_reason=getattr(
                self.policy, "termination_reason", None
            ),
            images_by_camera={
                camera_name: tuple(frames)
                for camera_name, frames in images_by_camera.items()
            },
        )
