from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
import yaml


FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")


@dataclass(frozen=True)
class NativePadPoint:
    link: str
    offset: tuple[float, float, float]


@dataclass(frozen=True)
class NativeFingerSpec:
    alignment_link: str
    chain_points: tuple[str, str, str]
    pad: NativePadPoint


@dataclass(frozen=True)
class NativeJointSpec:
    finger: str
    slot: int


@dataclass(frozen=True)
class NativeHandSpec:
    name: str
    urdf_path: Path
    palm_link: str
    active_joint_names: tuple[str, ...]
    fixed_fk_joints: Mapping[str, float]
    joint_semantics: Mapping[str, NativeJointSpec]
    fingers: Mapping[str, NativeFingerSpec]
    mujoco_scene_globs: tuple[str, ...]

    def full_q_from_active(
        self,
        active_q: torch.Tensor,
        fk_joint_names: list[str] | tuple[str, ...],
    ) -> torch.Tensor:
        """Expand native decoder q into the complete URDF FK joint order."""
        if active_q.shape[-1] != len(self.active_joint_names):
            raise ValueError(
                f"{self.name}: expected active q dim={len(self.active_joint_names)}, "
                f"got {active_q.shape[-1]}."
            )
        out = active_q.new_zeros((*active_q.shape[:-1], len(fk_joint_names)))
        active_index = {name: idx for idx, name in enumerate(self.active_joint_names)}
        for fk_idx, name in enumerate(fk_joint_names):
            if name in active_index:
                out[..., fk_idx] = active_q[..., active_index[name]]
            elif name in self.fixed_fk_joints:
                out[..., fk_idx] = float(self.fixed_fk_joints[name])
            else:
                raise KeyError(f"{self.name}: FK joint {name!r} is neither active nor fixed.")
        return out


def load_native_hand_specs(path: str | Path) -> dict[str, NativeHandSpec]:
    path = Path(path)
    payload = yaml.safe_load(path.read_text())
    specs: dict[str, NativeHandSpec] = {}
    for name, raw in payload["hands"].items():
        fingers = {}
        for finger_name in FINGER_ORDER:
            finger_raw = raw["fingers"][finger_name]
            chain_points = tuple(str(item) for item in finger_raw["chain_points"])
            if len(chain_points) != 3:
                raise ValueError(f"{name}/{finger_name}: chain_points must contain 3 links.")
            pad_raw = finger_raw["pad"]
            offset = tuple(float(item) for item in pad_raw["offset"])
            if len(offset) != 3:
                raise ValueError(f"{name}/{finger_name}: pad offset must contain 3 values.")
            fingers[finger_name] = NativeFingerSpec(
                alignment_link=str(finger_raw.get("alignment_link", chain_points[0])),
                chain_points=chain_points,
                pad=NativePadPoint(link=str(pad_raw["link"]), offset=offset),
            )
        specs[name] = NativeHandSpec(
            name=str(name),
            urdf_path=Path(raw["urdf_path"]),
            palm_link=str(raw["palm_link"]),
            active_joint_names=tuple(str(item) for item in raw["active_joint_names"]),
            fixed_fk_joints={str(key): float(value) for key, value in raw.get("fixed_fk_joints", {}).items()},
            joint_semantics={
                str(joint_name): NativeJointSpec(
                    finger=str(joint_raw["finger"]),
                    slot=int(joint_raw["slot"]),
                )
                for joint_name, joint_raw in raw["joint_semantics"].items()
            },
            fingers=fingers,
            mujoco_scene_globs=tuple(str(item) for item in raw.get("mujoco_scene_globs", [])),
        )
        missing_semantics = set(specs[name].active_joint_names) - set(specs[name].joint_semantics)
        extra_semantics = set(specs[name].joint_semantics) - set(specs[name].active_joint_names)
        if missing_semantics or extra_semantics:
            raise ValueError(
                f"{name}: joint_semantics must exactly cover active_joint_names; "
                f"missing={sorted(missing_semantics)}, extra={sorted(extra_semantics)}"
            )
        used_slots: set[tuple[str, int]] = set()
        for joint_name, joint_spec in specs[name].joint_semantics.items():
            if joint_spec.finger not in FINGER_ORDER or not 0 <= joint_spec.slot < 5:
                raise ValueError(f"{name}/{joint_name}: invalid semantic joint mapping {joint_spec}.")
            key = (joint_spec.finger, joint_spec.slot)
            if key in used_slots:
                raise ValueError(f"{name}: duplicate semantic joint slot {key}.")
            used_slots.add(key)
    return specs
