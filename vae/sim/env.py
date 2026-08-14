from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np
import yaml

from .scene import DEFAULT_BASE_SCENE, PACKAGE_ROOT, compose_grasp_scene


WRIST_JOINTS = ("wrist_x", "wrist_y", "wrist_z", "wrist_rx", "wrist_ry", "wrist_rz")
WRIST_ACTUATORS = ("ctrl_x", "ctrl_y", "ctrl_z", "ctrl_rx", "ctrl_ry", "ctrl_rz")
DEFAULT_HOME_WRISTS = {
    "shadow_hand_right": (0.0, -0.30, 0.30, -np.pi / 2.0, 0.0, np.pi),
    "gaia_hand_right": (0.0, -0.30, 0.30, -np.pi / 2.0, 0.0, np.pi),
    "sharpa_hand_right": (0.0, -0.30, 0.30, -np.pi / 2.0, 0.0, np.pi / 2.0),
}


@dataclass(frozen=True)
class GraspEnvConfig:
    hand: str
    scene_xml: str | Path | None = None
    base_scene: str | Path = DEFAULT_BASE_SCENE
    camera: str = "ego_opposite"
    observation_cameras: tuple[str, ...] = ()
    width: int = 320
    height: int = 240
    fps: int = 30
    drive_mode: str = "ctrl"
    physics_substep_multiplier: int = 1
    continuous_wrist_rotation: bool = False
    max_steps: int = 300
    success_lift_m: float = 0.20
    success_frames: int = 10
    terminate_on_success: bool = False
    render_images: bool = True
    disable_hand_self_collision: bool | None = None
    finger_armature: float = 0.001
    finger_damping: float = 0.1
    home_wrist: tuple[float, float, float, float, float, float] | None = None
    object_id: str | None = None
    object_meshes: tuple[str | Path, ...] = ()
    object_scale: float | tuple[float, float, float] = 1.0
    object_position: tuple[float, float, float] = (0.0, 0.0, 0.035)
    object_quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


def _load_joint_names(hand: str) -> tuple[str, ...]:
    config_path = PACKAGE_ROOT / "configs/right_hands.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        return tuple(config["hands"][hand]["active_joint_names"])
    except KeyError as exc:
        raise KeyError(f"Unknown hand {hand!r} in {config_path}") from exc


class GraspEnv:
    """Unified MuJoCo evaluation environment for Shadow, Gaia and Sharpa."""

    def __init__(self, config: GraspEnvConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            config = GraspEnvConfig(**kwargs)
        elif kwargs:
            raise TypeError("Pass either GraspEnvConfig or keyword arguments, not both.")
        if config.drive_mode not in {"ctrl", "qpos"}:
            raise ValueError("drive_mode must be 'ctrl' or 'qpos'.")
        if config.fps <= 0 or config.physics_substep_multiplier <= 0:
            raise ValueError("fps and physics_substep_multiplier must be positive.")

        self.config = config
        self.hand = config.hand
        self.joint_names = _load_joint_names(config.hand)
        self.action_names = WRIST_JOINTS + self.joint_names
        self.action_dim = len(self.action_names)

        self._temp_dir = TemporaryDirectory(prefix=f"native_vae_{self.hand}_")
        scene_output = Path(self._temp_dir.name) / "grasp_scene.xml"
        self.scene_path = compose_grasp_scene(
            hand=self.hand,
            generated_scene=config.scene_xml,
            base_scene=config.base_scene,
            output_scene=scene_output,
            object_id=config.object_id,
            object_meshes=config.object_meshes,
            object_scale=config.object_scale,
            object_position=config.object_position,
            object_quaternion=config.object_quaternion,
        )
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self._stabilize_model()
        self.data = mujoco.MjData(self.model)

        requested_cameras = config.observation_cameras or (config.camera,)
        self.observation_cameras = tuple(
            dict.fromkeys((config.camera, *requested_cameras))
        )
        for camera_name in self.observation_cameras:
            self._name_id(mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

        self._joint_ids = np.asarray(
            [self._name_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.action_names],
            dtype=np.int32,
        )
        if config.continuous_wrist_rotation:
            # Euler coordinates are periodic. Keeping artificial [-pi, pi]
            # hinge limits conflicts with nearest-branch unwrapping: a short
            # +pi -> +pi+epsilon command would otherwise be clipped at +pi.
            # The position actuators and SO(3) rate limiter still govern motion.
            self.model.jnt_limited[self._joint_ids[3:6]] = 0
        self._qpos_adrs = self.model.jnt_qposadr[self._joint_ids].copy()
        self._dof_adrs = self.model.jnt_dofadr[self._joint_ids].copy()
        actuator_names = WRIST_ACTUATORS + tuple(f"ctrl_{name}" for name in self.joint_names)
        self._actuator_ids = np.asarray(
            [self._name_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names],
            dtype=np.int32,
        )
        self._object_joint_id = self._name_id(mujoco.mjtObj.mjOBJ_JOINT, "object_joint")
        self._object_qpos_adr = int(self.model.jnt_qposadr[self._object_joint_id])
        self._object_dof_adr = int(self.model.jnt_dofadr[self._object_joint_id])
        self._joint_lower, self._joint_upper = self._controlled_joint_limits()
        self._substeps_exact = max(
            1.0,
            (1.0 / config.fps)
            / float(self.model.opt.timestep)
            * config.physics_substep_multiplier,
        )
        self._substep_residual = 0.0
        self._last_substeps = int(round(self._substeps_exact))

        self._renderer: mujoco.Renderer | None = None
        if config.render_images:
            self._renderer = mujoco.Renderer(
                self.model,
                height=config.height,
                width=config.width,
            )

        self._step_count = 0
        self._lift_streak = 0
        self._success = False
        self._object_initial_z = 0.0
        self._max_lift = 0.0
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)

    def _stabilize_model(self) -> None:
        disable_self_collision = self.config.disable_hand_self_collision
        if disable_self_collision is None:
            disable_self_collision = self.hand == "gaia_hand_right"
        if disable_self_collision:
            for geom_id in range(self.model.ngeom):
                if (
                    self.model.geom_contype[geom_id] == 0
                    and self.model.geom_conaffinity[geom_id] == 0
                ):
                    continue
                body_id = int(self.model.geom_bodyid[geom_id])
                body_name = (
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                    or ""
                )
                if body_name.startswith(("world", "floor", "table", "object")):
                    continue
                self.model.geom_contype[geom_id] = 2
                self.model.geom_conaffinity[geom_id] = 1

        if self.hand == "gaia_hand_right":
            for joint_name in self.joint_names:
                joint_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_name,
                )
                if joint_id < 0:
                    continue
                dof_adr = int(self.model.jnt_dofadr[joint_id])
                self.model.dof_armature[dof_adr] = max(
                    self.model.dof_armature[dof_adr],
                    float(self.config.finger_armature),
                )
                self.model.dof_damping[dof_adr] = max(
                    self.model.dof_damping[dof_adr],
                    float(self.config.finger_damping),
                )

    def _name_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        value = mujoco.mj_name2id(self.model, object_type, name)
        if value < 0:
            raise KeyError(f"{self.hand}: MJCF is missing {object_type.name} {name!r}")
        return int(value)

    def _controlled_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(self.action_dim, -np.inf, dtype=np.float64)
        upper = np.full(self.action_dim, np.inf, dtype=np.float64)
        limited = self.model.jnt_limited[self._joint_ids].astype(bool)
        lower[limited] = self.model.jnt_range[self._joint_ids[limited], 0]
        upper[limited] = self.model.jnt_range[self._joint_ids[limited], 1]
        return lower, upper

    @property
    def dt(self) -> float:
        return float(self.config.physics_substep_multiplier) / float(self.config.fps)

    @property
    def last_action(self) -> np.ndarray:
        """Return the finite, joint-limit-coerced target sent to MuJoCo."""
        return self._last_action.copy()

    def _next_substeps(self) -> int:
        self._substep_residual += self._substeps_exact
        count = max(1, int(np.floor(self._substep_residual + 1e-12)))
        self._substep_residual -= count
        self._last_substeps = count
        return count

    def _coerce_action(self, action: Sequence[float] | np.ndarray) -> np.ndarray:
        value = np.asarray(action, dtype=np.float64)
        if value.shape != (self.action_dim,):
            raise ValueError(
                f"{self.hand}: expected action [{self.action_dim}], got {value.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError("Action contains NaN or Inf.")
        return np.clip(value, self._joint_lower, self._joint_upper)

    def _set_qpos(self, action: np.ndarray, *, zero_velocity: bool) -> None:
        self.data.qpos[self._qpos_adrs] = action
        if zero_velocity:
            self.data.qvel[self._dof_adrs] = 0.0

    def _set_ctrl(self, action: np.ndarray) -> None:
        self.data.ctrl[self._actuator_ids] = action

    def reset(
        self,
        *,
        initial_action: Sequence[float] | np.ndarray | None = None,
        object_pose: Sequence[float] | np.ndarray | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        mujoco.mj_resetData(self.model, self.data)
        if object_pose is not None:
            pose = np.asarray(object_pose, dtype=np.float64)
            if pose.shape != (7,):
                raise ValueError("object_pose must be [xyz, quaternion(wxyz)] with shape [7].")
            self.data.qpos[self._object_qpos_adr : self._object_qpos_adr + 7] = pose
            self.data.qvel[self._object_dof_adr : self._object_dof_adr + 6] = 0.0

        if initial_action is None:
            home_wrist = self.config.home_wrist or DEFAULT_HOME_WRISTS[self.hand]
            if len(home_wrist) != 6:
                raise ValueError("home_wrist must contain xyz + Euler RPY (6 values).")
            action = np.concatenate(
                [
                    np.asarray(home_wrist, dtype=np.float64),
                    np.zeros(len(self.joint_names), dtype=np.float64),
                ]
            )
            action = self._coerce_action(action)
            self._set_qpos(action, zero_velocity=True)
        else:
            action = self._coerce_action(initial_action)
            self._set_qpos(action, zero_velocity=True)
        self._set_ctrl(action)
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._substep_residual = 0.0
        self._last_substeps = int(round(self._substeps_exact))
        self._lift_streak = 0
        self._success = False
        self._object_initial_z = float(self.object_pose[2])
        self._max_lift = 0.0
        self._last_action = action.astype(np.float32)
        info = self._info()
        return self._observation(), info

    def step(
        self,
        action: Sequence[float] | np.ndarray,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        target = self._coerce_action(action)
        substeps = self._next_substeps()
        if self.config.drive_mode == "ctrl":
            self._set_ctrl(target)
            for _ in range(substeps):
                mujoco.mj_step(self.model, self.data)
        else:
            self._set_ctrl(target)
            for _ in range(substeps):
                self._set_qpos(target, zero_velocity=True)
                mujoco.mj_step(self.model, self.data)
            self._set_qpos(target, zero_velocity=True)
            mujoco.mj_forward(self.model, self.data)

        self._step_count += 1
        self._last_action = target.astype(np.float32)
        lift = float(self.object_pose[2] - self._object_initial_z)
        self._max_lift = max(self._max_lift, lift)
        self._lift_streak = self._lift_streak + 1 if lift >= self.config.success_lift_m else 0
        self._success = self._success or (
            self._lift_streak >= self.config.success_frames
        )
        terminated = bool(self._success and self.config.terminate_on_success)
        truncated = self._step_count >= self.config.max_steps
        info = self._info()
        return self._observation(), lift, terminated, truncated, info

    @property
    def state(self) -> np.ndarray:
        return self.data.qpos[self._qpos_adrs].astype(np.float32, copy=True)

    @property
    def object_pose(self) -> np.ndarray:
        start = self._object_qpos_adr
        return self.data.qpos[start : start + 7].astype(np.float32, copy=True)

    @property
    def object_velocity(self) -> np.ndarray:
        start = self._object_dof_adr
        return self.data.qvel[start : start + 6].astype(np.float32, copy=True)

    def render(self, camera: str | None = None) -> np.ndarray:
        if self._renderer is None:
            raise RuntimeError("Image rendering is disabled for this environment.")
        camera_name = self.config.camera if camera is None else camera
        if camera_name not in self.observation_cameras:
            self._name_id(mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self._renderer.update_scene(self.data, camera=camera_name)
        return self._renderer.render().copy()

    def render_cameras(self) -> dict[str, np.ndarray]:
        """Render every requested policy camera exactly once."""
        return {
            camera_name: self.render(camera_name)
            for camera_name in self.observation_cameras
        }

    def _observation(self) -> dict[str, Any]:
        images = self.render_cameras() if self._renderer is not None else {}
        image = images.get(self.config.camera)
        return {
            "state": self.state,
            "image": image,
            "images": images,
            "object_pose": self.object_pose,
            "object_velocity": self.object_velocity,
            "sim_time": np.float32(self.data.time),
        }

    def _info(self) -> dict[str, Any]:
        current_lift = float(self.object_pose[2] - self._object_initial_z)
        return {
            "hand": self.hand,
            "step": self._step_count,
            "current_lift_m": current_lift,
            "max_lift_m": self._max_lift,
            "lift_streak": self._lift_streak,
            "success": self._success,
            "action_dim": self.action_dim,
            "joint_names": self.joint_names,
            "dt": self.dt,
            "physics_substeps": self._last_substeps,
        }

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._temp_dir.cleanup()

    def __enter__(self) -> "GraspEnv":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
