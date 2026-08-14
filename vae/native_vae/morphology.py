from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

import torch

NUM_FINGERS = 5
MAX_JOINTS_PER_FINGER = 5
FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")
JOINT_QUERY_DIM = 22
OHRA_MORPHOLOGY_DATA_DIM = 66


@dataclass(frozen=True)
class MorphologyDescriptorLayout:
    palm_radius: slice = field(default_factory=lambda: slice(0, 1))
    finger_radius: slice = field(default_factory=lambda: slice(1, 2))
    finger_presence: slice = field(default_factory=lambda: slice(2, 7))
    finger_lengths: slice = field(default_factory=lambda: slice(7, 22))
    finger_xyz: slice = field(default_factory=lambda: slice(22, 37))
    little_extra_origin: slice = field(default_factory=lambda: slice(37, 43))
    thumb_rpy: slice = field(default_factory=lambda: slice(43, 46))
    thumb_axes: slice = field(default_factory=lambda: slice(46, 52))
    joint_presence: slice = field(default_factory=lambda: slice(52, 77))
    joint_lowers: slice = field(default_factory=lambda: slice(77, 102))
    joint_uppers: slice = field(default_factory=lambda: slice(102, 127))

    @property
    def total_dim(self) -> int:
        return self.joint_uppers.stop


LAYOUT = MorphologyDescriptorLayout()
MORPHOLOGY_DIM = LAYOUT.total_dim
DEFAULT_ANGLE_SCALE = math.pi


def _to_tensor(data: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(data, dtype=dtype)


def _pad_nested_sequence(
    data: Any,
    target_shape: tuple[int, ...],
    *,
    fill_value: float = 0.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    out = torch.full(target_shape, float(fill_value), dtype=dtype)

    if len(target_shape) == 1:
        if data is None:
            return out
        values = data if isinstance(data, list) else [data]
        limit = min(len(values), target_shape[0])
        for idx in range(limit):
            out[idx] = float(values[idx])
        return out

    if data is None:
        return out

    rows = min(len(data), target_shape[0]) if isinstance(data, list) else 0
    for row_idx in range(rows):
        out[row_idx] = _pad_nested_sequence(
            data[row_idx],
            target_shape[1:],
            fill_value=fill_value,
            dtype=dtype,
        )
    return out


def _expand_finger_lengths(raw_lengths: torch.Tensor) -> torch.Tensor:
    """
    Normalize OHRA-style finger length descriptors to [5, 3].

    Supported inputs:
    - [2, 3]: thumb + shared-other-fingers
    - [5, 3]: per-finger lengths
    - anything else: copy as much as possible into a [5, 3] zero-padded tensor
    """
    raw_lengths = raw_lengths.to(dtype=torch.float32)

    if raw_lengths.ndim == 1:
        raw_lengths = raw_lengths.unsqueeze(0)

    out = torch.zeros((NUM_FINGERS, 3), dtype=torch.float32)

    if raw_lengths.shape == (2, 3):
        out[0] = raw_lengths[0]
        out[1:] = raw_lengths[1].unsqueeze(0).expand(NUM_FINGERS - 1, -1)
        return out

    rows = min(NUM_FINGERS, raw_lengths.shape[0])
    cols = min(3, raw_lengths.shape[1]) if raw_lengths.ndim >= 2 else 0
    if rows > 0 and cols > 0:
        out[:rows, :cols] = raw_lengths[:rows, :cols]
    return out


def _infer_finger_presence(finger_xyz: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (torch.linalg.norm(finger_xyz, dim=1) > eps).to(dtype=torch.float32)


def _infer_joint_presence(
    joint_lowers: torch.Tensor,
    joint_uppers: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    return ((joint_lowers.abs() + joint_uppers.abs()) > eps).to(dtype=torch.float32)


def _combine_finger_presence(
    finger_xyz: torch.Tensor,
    joint_presence: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    xyz_presence = _infer_finger_presence(finger_xyz, eps=eps)
    joint_any = (joint_presence.sum(dim=1) > 0.0).to(dtype=torch.float32)
    return torch.maximum(xyz_presence, joint_any)


def load_morphology_descriptor(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_raw_morphology_descriptor(
    descriptor: Mapping[str, Any],
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Build the raw fixed-length descriptor before any normalization.

    Raw layout:
        [ palm_radius(1),
          finger_radius(1),
          finger_presence(5),
          finger_lengths(5,3),
          finger_xyz(5,3),
          little_extra_origin(6),
          thumb_rpy(3),
          thumb_axes(2,3),
          joint_presence(5,5),
          joint_lowers(5,5),
          joint_uppers(5,5) ]
    """
    palm_radius = _to_tensor([descriptor["palm_radius"]])
    finger_radius = _to_tensor([descriptor["finger_radius"]])
    finger_lengths = _expand_finger_lengths(_to_tensor(descriptor["finger_lengths"]))
    finger_xyz = _pad_nested_sequence(descriptor["finger_xyz"], (NUM_FINGERS, 3))
    little_extra_origin = _pad_nested_sequence(descriptor["little_extra_origin"], (6,))
    thumb_rpy = _pad_nested_sequence(descriptor["thumb_rpy"], (3,))
    thumb_axes = _pad_nested_sequence(descriptor["thumb_axes"], (2, 3))
    joint_lowers = _pad_nested_sequence(descriptor["joint_lowers"], (NUM_FINGERS, MAX_JOINTS_PER_FINGER))
    joint_uppers = _pad_nested_sequence(descriptor["joint_uppers"], (NUM_FINGERS, MAX_JOINTS_PER_FINGER))

    joint_presence = _infer_joint_presence(joint_lowers, joint_uppers, eps=eps)
    finger_presence = _combine_finger_presence(finger_xyz, joint_presence, eps=eps)

    parts = [
        palm_radius.reshape(-1),
        finger_radius.reshape(-1),
        finger_presence.reshape(-1),
        finger_lengths.reshape(-1),
        finger_xyz.reshape(-1),
        little_extra_origin.reshape(-1),
        thumb_rpy.reshape(-1),
        thumb_axes.reshape(-1),
        joint_presence.reshape(-1),
        joint_lowers.reshape(-1),
        joint_uppers.reshape(-1),
    ]
    out = torch.cat(parts, dim=0)
    if out.numel() != MORPHOLOGY_DIM:
        raise RuntimeError(f"Expected morphology dim {MORPHOLOGY_DIM}, got {out.numel()}")
    return out


def normalize_morphology_descriptor(
    descriptor: Mapping[str, Any] | torch.Tensor,
    *,
    angle_scale: float = DEFAULT_ANGLE_SCALE,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Morphology normalization v1.

    Rules:
    - scale feature: log(palm_radius)
    - length / position fields: divide by palm_radius
    - angular fields: divide by pi
    - axes and binary masks: keep as-is
    - no dataset-level statistics
    """
    raw = descriptor if torch.is_tensor(descriptor) else build_raw_morphology_descriptor(descriptor, eps=eps)
    raw = raw.to(dtype=torch.float32)
    parts = split_morphology_descriptor(raw)

    palm_radius = parts["palm_radius"].clamp_min(eps)
    scale_features = torch.log(palm_radius)

    shape_parts = {
        "finger_radius": parts["finger_radius"] / palm_radius,
        "finger_presence": parts["finger_presence"],
        "finger_lengths": parts["finger_lengths"] / palm_radius[..., None, None],
        "finger_xyz": parts["finger_xyz"] / palm_radius[..., None, None],
        "little_extra_origin": parts["little_extra_origin"].clone(),
        "thumb_rpy": parts["thumb_rpy"] / float(angle_scale),
        "thumb_axes": parts["thumb_axes"],
        "joint_presence": parts["joint_presence"],
        "joint_lowers": parts["joint_lowers"] / float(angle_scale),
        "joint_uppers": parts["joint_uppers"] / float(angle_scale),
    }
    shape_parts["little_extra_origin"][..., :3] = (
        shape_parts["little_extra_origin"][..., :3] / palm_radius
    )
    shape_parts["little_extra_origin"][..., 3:] = (
        shape_parts["little_extra_origin"][..., 3:] / float(angle_scale)
    )

    shape_features = torch.cat(
        [
            shape_parts["finger_radius"].reshape(*raw.shape[:-1], -1),
            shape_parts["finger_presence"].reshape(*raw.shape[:-1], -1),
            shape_parts["finger_lengths"].reshape(*raw.shape[:-1], -1),
            shape_parts["finger_xyz"].reshape(*raw.shape[:-1], -1),
            shape_parts["little_extra_origin"].reshape(*raw.shape[:-1], -1),
            shape_parts["thumb_rpy"].reshape(*raw.shape[:-1], -1),
            shape_parts["thumb_axes"].reshape(*raw.shape[:-1], -1),
            shape_parts["joint_presence"].reshape(*raw.shape[:-1], -1),
            shape_parts["joint_lowers"].reshape(*raw.shape[:-1], -1),
            shape_parts["joint_uppers"].reshape(*raw.shape[:-1], -1),
        ],
        dim=-1,
    )
    full_features = torch.cat([scale_features, shape_features], dim=-1)

    return {
        "raw": raw,
        "scale_features": scale_features,
        "shape_features": shape_features,
        "full_features": full_features,
    }


def build_joint_queries_from_normalized_morphology(
    descriptor: Mapping[str, Any] | torch.Tensor,
    *,
    angle_scale: float = DEFAULT_ANGLE_SCALE,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Build 22D joint queries for the universal decoder from the standardized hand JSON.

    Query layout per joint slot:
        finger_onehot(5)
        joint_slot_onehot(5)
        joint_presence(1)
        joint_lower/pi(1)
        joint_upper/pi(1)
        joint_center/pi(1)
        joint_range/pi(1)
        finger_xyz_local(3)
        finger_lengths_local(3)
        hierarchy(1)
    """
    if torch.is_tensor(descriptor):
        parts = split_morphology_descriptor(descriptor.to(dtype=torch.float32))
        palm_radius = parts["palm_radius"].clamp_min(eps)
        scale = palm_radius.reshape(-1)[0]
        finger_lengths = parts["finger_lengths"] / scale
        finger_xyz = parts["finger_xyz"] / scale
        joint_presence = parts["joint_presence"]
        joint_lowers = parts["joint_lowers"] / float(angle_scale)
        joint_uppers = parts["joint_uppers"] / float(angle_scale)
    else:
        norm = normalize_morphology_descriptor(descriptor, angle_scale=angle_scale, eps=eps)
        parts = split_morphology_descriptor(norm["raw"])
        palm_radius = parts["palm_radius"].clamp_min(eps)
        scale = palm_radius.reshape(-1)[0]
        finger_lengths = parts["finger_lengths"] / scale
        finger_xyz = parts["finger_xyz"] / scale
        joint_presence = parts["joint_presence"]
        joint_lowers = parts["joint_lowers"] / float(angle_scale)
        joint_uppers = parts["joint_uppers"] / float(angle_scale)

    if palm_radius.dim() != 1:
        raise ValueError("Expected unbatched descriptor when building joint queries.")

    queries = []
    eye_finger = torch.eye(NUM_FINGERS, dtype=torch.float32)
    eye_slot = torch.eye(MAX_JOINTS_PER_FINGER, dtype=torch.float32)
    for finger_idx in range(NUM_FINGERS):
        for slot_idx in range(MAX_JOINTS_PER_FINGER):
            lower = joint_lowers[finger_idx, slot_idx]
            upper = joint_uppers[finger_idx, slot_idx]
            presence = joint_presence[finger_idx, slot_idx]
            center = 0.5 * (lower + upper)
            joint_range = upper - lower
            hierarchy = torch.tensor([slot_idx / max(MAX_JOINTS_PER_FINGER - 1, 1)], dtype=torch.float32)
            query = torch.cat(
                [
                    eye_finger[finger_idx],
                    eye_slot[slot_idx],
                    presence.view(1),
                    lower.view(1),
                    upper.view(1),
                    center.view(1),
                    joint_range.view(1),
                    finger_xyz[finger_idx],
                    finger_lengths[finger_idx],
                    hierarchy,
                ],
                dim=0,
            )
            queries.append(query)
    out = torch.stack(queries, dim=0)
    if out.shape != (NUM_FINGERS * MAX_JOINTS_PER_FINGER, JOINT_QUERY_DIM):
        raise RuntimeError(f"Expected joint query shape (25, {JOINT_QUERY_DIM}), got {tuple(out.shape)}")
    return out


def flatten_morphology_descriptor(
    descriptor: Mapping[str, Any],
    *,
    normalize: bool = False,
    angle_scale: float = DEFAULT_ANGLE_SCALE,
    eps: float = 1e-8,
) -> torch.Tensor:
    if normalize:
        return normalize_morphology_descriptor(
            descriptor,
            angle_scale=angle_scale,
            eps=eps,
        )["full_features"]
    return build_raw_morphology_descriptor(descriptor, eps=eps)


def load_and_flatten_morphology_descriptor(
    path: str | Path,
    *,
    normalize: bool = False,
    angle_scale: float = DEFAULT_ANGLE_SCALE,
    eps: float = 1e-8,
) -> torch.Tensor:
    descriptor = load_morphology_descriptor(path)
    return flatten_morphology_descriptor(
        descriptor,
        normalize=normalize,
        angle_scale=angle_scale,
        eps=eps,
    )


def load_and_normalize_morphology_descriptor(
    path: str | Path,
    *,
    angle_scale: float = DEFAULT_ANGLE_SCALE,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    descriptor = load_morphology_descriptor(path)
    return normalize_morphology_descriptor(
        descriptor,
        angle_scale=angle_scale,
        eps=eps,
    )


def load_and_build_joint_queries(
    path: str | Path,
    *,
    angle_scale: float = DEFAULT_ANGLE_SCALE,
    eps: float = 1e-8,
) -> torch.Tensor:
    descriptor = load_morphology_descriptor(path)
    return build_joint_queries_from_normalized_morphology(
        descriptor,
        angle_scale=angle_scale,
        eps=eps,
    )


def build_ohra_morphology_data(
    descriptor: Mapping[str, Any] | torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Build the 66D hand-parameter representation used by OHRA's morphology VAE.

    Layout:
        continuous geometry/rpy params: 32
        thumb axis1 one-hot:            6
        thumb axis2 one-hot:            6
        joint existence flags:          22

    This intentionally mirrors OHRA ``sample_utils.params_list_to_data`` while
    accepting the same JSON/tensor descriptors used by this project.
    """
    if torch.is_tensor(descriptor):
        raw = descriptor.to(dtype=torch.float32)
    else:
        raw = build_raw_morphology_descriptor(descriptor, eps=eps)
    parts = split_morphology_descriptor(raw)

    finger_lengths = parts["finger_lengths"]
    if finger_lengths.dim() != 2:
        raise ValueError("Expected an unbatched morphology descriptor.")

    # OHRA stores thumb lengths plus one shared non-thumb length triplet.
    non_thumb_presence = parts["finger_presence"][1:].view(-1, 1)
    non_thumb_lengths = finger_lengths[1:]
    denom = non_thumb_presence.sum().clamp_min(1.0)
    shared_non_thumb = (non_thumb_lengths * non_thumb_presence).sum(dim=0) / denom

    continuous = torch.cat(
        [
            parts["palm_radius"].reshape(-1),
            parts["finger_radius"].reshape(-1),
            finger_lengths[0].reshape(-1),
            shared_non_thumb.reshape(-1),
            parts["finger_xyz"].reshape(-1),
            parts["little_extra_origin"].reshape(-1),
            parts["thumb_rpy"].reshape(-1),
        ],
        dim=0,
    )
    if continuous.numel() != 32:
        raise RuntimeError(f"Expected 32 continuous OHRA params, got {continuous.numel()}")

    axis_options = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
        device=raw.device,
    )

    axis_data = []
    for axis in parts["thumb_axes"].reshape(2, 3):
        distances = torch.linalg.norm(axis_options - axis.view(1, 3), dim=1)
        one_hot = torch.zeros(6, dtype=torch.float32, device=raw.device)
        one_hot[int(torch.argmin(distances).item())] = 1.0
        axis_data.append(one_hot)

    joint_lowers = parts["joint_lowers"]
    joint_uppers = parts["joint_uppers"]
    joint_presence = ((joint_lowers - joint_uppers).abs() > eps).to(dtype=torch.float32)
    joint_disc = torch.cat(
        [
            joint_presence[0, :5],
            joint_presence[1, :4],
            joint_presence[2, :4],
            joint_presence[3, :4],
            joint_presence[4, :5],
        ],
        dim=0,
    )

    out = torch.cat([continuous, axis_data[0], axis_data[1], joint_disc], dim=0)
    if out.numel() != OHRA_MORPHOLOGY_DATA_DIM:
        raise RuntimeError(f"Expected OHRA morphology dim 66, got {out.numel()}")
    return out


def load_and_build_ohra_morphology_data(
    path: str | Path,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    descriptor = load_morphology_descriptor(path)
    return build_ohra_morphology_data(descriptor, eps=eps)


def split_morphology_descriptor(h: torch.Tensor) -> Dict[str, torch.Tensor]:
    if h.shape[-1] != MORPHOLOGY_DIM:
        raise ValueError(f"Expected last dim {MORPHOLOGY_DIM}, got {h.shape[-1]}")

    return {
        "palm_radius": h[..., LAYOUT.palm_radius],
        "finger_radius": h[..., LAYOUT.finger_radius],
        "finger_presence": h[..., LAYOUT.finger_presence],
        "finger_lengths": h[..., LAYOUT.finger_lengths].reshape(*h.shape[:-1], NUM_FINGERS, 3),
        "finger_xyz": h[..., LAYOUT.finger_xyz].reshape(*h.shape[:-1], NUM_FINGERS, 3),
        "little_extra_origin": h[..., LAYOUT.little_extra_origin],
        "thumb_rpy": h[..., LAYOUT.thumb_rpy],
        "thumb_axes": h[..., LAYOUT.thumb_axes].reshape(*h.shape[:-1], 2, 3),
        "joint_presence": h[..., LAYOUT.joint_presence].reshape(
            *h.shape[:-1], NUM_FINGERS, MAX_JOINTS_PER_FINGER
        ),
        "joint_lowers": h[..., LAYOUT.joint_lowers].reshape(
            *h.shape[:-1], NUM_FINGERS, MAX_JOINTS_PER_FINGER
        ),
        "joint_uppers": h[..., LAYOUT.joint_uppers].reshape(
            *h.shape[:-1], NUM_FINGERS, MAX_JOINTS_PER_FINGER
        ),
    }
