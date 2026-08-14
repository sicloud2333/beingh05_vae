from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "orthohand-native-vae-matplotlib"),
)

import pytorch_kinematics as pk
import torch

from .morphology import (
    FINGER_ORDER,
    build_joint_queries_from_normalized_morphology,
    build_raw_morphology_descriptor,
    normalize_morphology_descriptor,
)

from .hand_spec import NativeHandSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_OUTPUT_JOINTS = 22


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_joint_limits(urdf_path: str | Path) -> dict[str, tuple[float, float]]:
    root = ET.parse(project_path(urdf_path)).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        if joint.attrib.get("type", "fixed") == "fixed":
            continue
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"Joint {joint.attrib['name']} has no limits.")
        limits[joint.attrib["name"]] = (
            float(limit.attrib.get("lower", 0.0)),
            float(limit.attrib.get("upper", 0.0)),
        )
    return limits


def transform_point(matrix: torch.Tensor, offset: tuple[float, float, float]) -> torch.Tensor:
    local = torch.as_tensor(offset, dtype=matrix.dtype, device=matrix.device)
    return (matrix[..., :3, :3] @ local.reshape(3, 1)).squeeze(-1) + matrix[..., :3, 3]


@dataclass
class NativeHandRuntime:
    spec: NativeHandSpec
    chain: object
    fk_joint_names: tuple[str, ...]
    limits: dict[str, tuple[float, float]]
    world_to_semantic_rotation: torch.Tensor
    world_to_semantic_translation: torch.Tensor
    morphology_raw: torch.Tensor
    morphology_vec: torch.Tensor
    joint_queries: torch.Tensor
    q_lower: torch.Tensor
    q_upper: torch.Tensor
    q_mask: torch.Tensor
    palm_radius: float

    @classmethod
    def build(cls, spec: NativeHandSpec, device: str | torch.device = "cpu") -> "NativeHandRuntime":
        urdf_path = project_path(spec.urdf_path)
        with urdf_path.open("rb") as handle:
            chain = pk.build_chain_from_urdf(handle.read()).to(device=device, dtype=torch.float32)
        fk_joint_names = tuple(chain.get_joint_parameter_names())
        limits = parse_joint_limits(urdf_path)

        zero_active = torch.zeros((1, len(spec.active_joint_names)), dtype=torch.float32, device=device)
        zero_full = spec.full_q_from_active(zero_active, fk_joint_names)
        fk_zero = {name: tf.get_matrix() for name, tf in chain.forward_kinematics(zero_full).items()}
        rotation, translation = _semantic_transform(spec, fk_zero)
        descriptor = _native_morphology_descriptor(spec, fk_zero, rotation, translation, limits)
        raw = build_raw_morphology_descriptor(descriptor)
        morphology_vec = normalize_morphology_descriptor(raw)["full_features"]
        all_queries = build_joint_queries_from_normalized_morphology(raw)

        queries = torch.zeros((MAX_OUTPUT_JOINTS, all_queries.shape[-1]), dtype=torch.float32)
        q_lower = torch.zeros(MAX_OUTPUT_JOINTS, dtype=torch.float32)
        q_upper = torch.zeros(MAX_OUTPUT_JOINTS, dtype=torch.float32)
        q_mask = torch.zeros(MAX_OUTPUT_JOINTS, dtype=torch.float32)
        if len(spec.active_joint_names) > MAX_OUTPUT_JOINTS:
            raise ValueError(f"{spec.name}: {len(spec.active_joint_names)} active joints exceed {MAX_OUTPUT_JOINTS}.")
        for output_idx, joint_name in enumerate(spec.active_joint_names):
            semantic = spec.joint_semantics[joint_name]
            finger_idx = FINGER_ORDER.index(semantic.finger)
            queries[output_idx] = all_queries[finger_idx * 5 + semantic.slot]
            lower, upper = limits[joint_name]
            q_lower[output_idx] = lower
            q_upper[output_idx] = upper
            q_mask[output_idx] = float(abs(upper - lower) > 1e-8)

        return cls(
            spec=spec,
            chain=chain,
            fk_joint_names=fk_joint_names,
            limits=limits,
            world_to_semantic_rotation=rotation,
            world_to_semantic_translation=translation,
            morphology_raw=raw,
            morphology_vec=morphology_vec,
            joint_queries=queries,
            q_lower=q_lower,
            q_upper=q_upper,
            q_mask=q_mask,
            palm_radius=float(raw[0]),
        )

    def to(self, device: str | torch.device) -> "NativeHandRuntime":
        return NativeHandRuntime.build(self.spec, device=device)

    def full_q(self, active_q: torch.Tensor) -> torch.Tensor:
        return self.spec.full_q_from_active(active_q, self.fk_joint_names)

    def sample_active_q(
        self,
        count: int,
        rng: np.random.Generator,
        *,
        limit_shrink_ratio: float = 0.95,
    ) -> torch.Tensor:
        """50% independent, 25% open/half/closed, 25% finger-correlated samples."""
        count = int(count)
        values = np.empty((count, len(self.spec.active_joint_names)), dtype=np.float32)
        mode = rng.choice(3, size=count, p=(0.50, 0.25, 0.25))
        finger_progress = {
            finger: rng.uniform(0.05, 0.95, size=count).astype(np.float32)
            for finger in FINGER_ORDER
        }
        strata = rng.choice(np.asarray([0.08, 0.50, 0.92], dtype=np.float32), size=count)
        strata = np.clip(strata + rng.normal(0.0, 0.035, size=count), 0.025, 0.975)
        for joint_idx, joint_name in enumerate(self.spec.active_joint_names):
            lower, upper = self.limits[joint_name]
            center = 0.5 * (lower + upper)
            half = 0.5 * (upper - lower) * limit_shrink_ratio
            lo, hi = center - half, center + half
            independent = rng.uniform(lo, hi, size=count)
            stratified = lo + strata * (hi - lo)
            finger = self.spec.joint_semantics[joint_name].finger
            correlated_u = np.clip(
                finger_progress[finger] + rng.normal(0.0, 0.06, size=count),
                0.0,
                1.0,
            )
            correlated = lo + correlated_u * (hi - lo)
            values[:, joint_idx] = np.where(
                mode == 0,
                independent,
                np.where(mode == 1, stratified, correlated),
            ).astype(np.float32)
        return torch.from_numpy(values)

    def padded_joint_tensors(self, active_q: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = active_q.shape[0]
        q = torch.zeros((batch, MAX_OUTPUT_JOINTS), dtype=torch.float32)
        dim = len(self.spec.active_joint_names)
        q[:, :dim] = active_q.cpu()
        lower = self.q_lower.unsqueeze(0).expand(batch, -1).clone()
        upper = self.q_upper.unsqueeze(0).expand(batch, -1).clone()
        mask = self.q_mask.unsqueeze(0).expand(batch, -1).clone()
        q_norm = torch.zeros_like(q)
        ranges = upper - lower
        q_norm = torch.where(mask > 0.5, 2.0 * (q - lower) / ranges.clamp_min(1e-8) - 1.0, q_norm)
        return {"q": q, "q_norm": q_norm, "q_mask": mask, "q_lower": lower, "q_upper": upper}

    def kinematic_chain_gesture(self, active_q: torch.Tensor) -> dict[str, torch.Tensor]:
        full_q = self.full_q(active_q)
        fk = {name: tf.get_matrix() for name, tf in self.chain.forward_kinematics(full_q).items()}
        batch = active_q.shape[0]
        points = []
        for finger in FINGER_ORDER:
            finger_spec = self.spec.fingers[finger]
            chain_points = [fk[link][..., :3, 3] for link in finger_spec.chain_points]
            pad = transform_point(fk[finger_spec.pad.link], finger_spec.pad.offset)
            points.append(chain_points + [pad])
        # [B, finger, chain-point, xyz], transformed by one fixed semantic frame.
        stacked = torch.stack([torch.stack(item, dim=1) for item in points], dim=1)
        rotation = self.world_to_semantic_rotation.to(stacked)
        translation = self.world_to_semantic_translation.to(stacked)
        semantic = torch.einsum("ij,bfkj->bfki", rotation, stacked) + translation
        vectors = torch.cat(
            [semantic[:, :, 0], semantic[:, :, 1] - semantic[:, :, 0],
             semantic[:, :, 2] - semantic[:, :, 1], semantic[:, :, 3] - semantic[:, :, 2]],
            dim=1,
        )
        x_gesture = vectors.reshape(batch, 60)
        valid_mask = torch.ones_like(x_gesture)
        return {"x_gesture": x_gesture, "valid_mask": valid_mask, "tips": semantic[:, :, 3]}

    def tips_from_padded_q(self, padded_q: torch.Tensor) -> torch.Tensor:
        active = padded_q[:, : len(self.spec.active_joint_names)]
        return self.kinematic_chain_gesture(active)["tips"]


def _semantic_transform(
    spec: NativeHandSpec,
    fk_zero: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    palm = fk_zero[spec.palm_link][0]
    palm_r, palm_t = palm[:3, :3], palm[:3, 3]
    roots_local: dict[str, torch.Tensor] = {}
    for finger, finger_spec in spec.fingers.items():
        root_world = fk_zero[finger_spec.alignment_link][0, :3, 3]
        roots_local[finger] = palm_r.T @ (root_world - palm_t)
    origin_local = torch.stack([roots_local[f] for f in ("index", "middle", "ring", "little")]).mean(0)
    directions = []
    for finger in ("index", "middle", "ring", "little"):
        finger_spec = spec.fingers[finger]
        pad_world = transform_point(fk_zero[finger_spec.pad.link][0], finger_spec.pad.offset)
        pad_local = palm_r.T @ (pad_world - palm_t)
        directions.append(torch.nn.functional.normalize(pad_local - roots_local[finger], dim=0))
    y_axis = torch.nn.functional.normalize(torch.stack(directions).mean(0), dim=0)
    lateral_raw = roots_local["index"] - roots_local["little"]
    x_axis = torch.nn.functional.normalize(lateral_raw - y_axis * torch.dot(y_axis, lateral_raw), dim=0)
    z_axis = torch.nn.functional.normalize(torch.linalg.cross(x_axis, y_axis), dim=0)
    y_axis = torch.nn.functional.normalize(torch.linalg.cross(z_axis, x_axis), dim=0)
    rotation = torch.stack([x_axis, y_axis, z_axis]) @ palm_r.T
    origin_world = palm_r @ origin_local + palm_t
    translation = -rotation @ origin_world
    return rotation, translation


def _native_morphology_descriptor(
    spec: NativeHandSpec,
    fk_zero: dict[str, torch.Tensor],
    rotation: torch.Tensor,
    translation: torch.Tensor,
    limits: dict[str, tuple[float, float]],
) -> dict:
    finger_xyz = []
    finger_lengths = []
    pad_radii = []
    for finger in FINGER_ORDER:
        finger_spec = spec.fingers[finger]
        chain_world = [fk_zero[link][0, :3, 3] for link in finger_spec.chain_points]
        pad_world = transform_point(fk_zero[finger_spec.pad.link][0], finger_spec.pad.offset)
        semantic = [rotation @ point + translation for point in chain_world + [pad_world]]
        finger_xyz.append(semantic[0].tolist())
        finger_lengths.append([float(torch.linalg.norm(semantic[i + 1] - semantic[i])) for i in range(3)])
        pad_radii.append(float(torch.linalg.norm(torch.as_tensor(finger_spec.pad.offset))))
    palm_radius = max(float(torch.linalg.norm(torch.tensor(xyz))) for xyz in finger_xyz)
    joint_lowers = torch.zeros((5, 5), dtype=torch.float32)
    joint_uppers = torch.zeros((5, 5), dtype=torch.float32)
    for joint_name in spec.active_joint_names:
        semantic = spec.joint_semantics[joint_name]
        finger_idx = FINGER_ORDER.index(semantic.finger)
        lower, upper = limits[joint_name]
        joint_lowers[finger_idx, semantic.slot] = lower
        joint_uppers[finger_idx, semantic.slot] = upper
    return {
        "palm_radius": max(palm_radius, 1e-4),
        "finger_radius": float(np.median(pad_radii)),
        "finger_lengths": finger_lengths,
        "finger_xyz": finger_xyz,
        "little_extra_origin": [0.0] * 6,
        "thumb_rpy": [0.0] * 3,
        "thumb_axes": [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        "joint_lowers": joint_lowers.tolist(),
        "joint_uppers": joint_uppers.tolist(),
    }
