from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from .hand_runtime import NativeHandRuntime
from .hand_spec import load_native_hand_specs
from .morphology import FINGER_ORDER


def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    log_var = log_var.clamp(-12.0, 12.0)
    return (-0.5 * (1.0 + log_var - mu.square() - log_var.exp())).sum(dim=1).mean()


def denormalize_q(
    q_norm: torch.Tensor,
    q_lower: torch.Tensor,
    q_upper: torch.Tensor,
    q_mask: torch.Tensor,
) -> torch.Tensor:
    q = 0.5 * (q_norm.clamp(-1.0, 1.0) + 1.0) * (q_upper - q_lower) + q_lower
    return torch.where(q_mask > 0.5, q, torch.zeros_like(q))


def tips_from_kinematic_chain(x_gesture: torch.Tensor) -> torch.Tensor:
    if x_gesture.ndim != 2 or x_gesture.shape[-1] != 60:
        raise ValueError(f"Expected x_gesture [B,60], got {tuple(x_gesture.shape)}")
    batch = x_gesture.shape[0]
    palm_to_root = x_gesture[:, 0:15].reshape(batch, 5, 3)
    root_to_joint1 = x_gesture[:, 15:30].reshape(batch, 5, 3)
    tail = x_gesture[:, 30:60].reshape(batch, 10, 3)
    return palm_to_root + root_to_joint1 + tail[:, 0:5] + tail[:, 5:10]


@dataclass
class NativeHandBank:
    runtimes: dict[str, NativeHandRuntime]

    @classmethod
    def build(
        cls,
        hand_config: str | Path,
        device: str | torch.device,
        hand_names: Sequence[str] | None = None,
    ) -> "NativeHandBank":
        specs = load_native_hand_specs(hand_config)
        selected = tuple(hand_names or specs.keys())
        return cls(
            {
                name: NativeHandRuntime.build(specs[name], device=device)
                for name in selected
            }
        )

    def target_batch(
        self,
        hand_name: str,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        runtime = self.runtimes[hand_name]
        return {
            "morphology_vec": runtime.morphology_vec.to(device).unsqueeze(0).expand(batch_size, -1),
            "joint_queries": runtime.joint_queries.to(device).unsqueeze(0).expand(batch_size, -1, -1),
            "q_lower": runtime.q_lower.to(device).unsqueeze(0).expand(batch_size, -1),
            "q_upper": runtime.q_upper.to(device).unsqueeze(0).expand(batch_size, -1),
            "q_mask": runtime.q_mask.to(device).unsqueeze(0).expand(batch_size, -1),
        }


def _weighted_tip_l1(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    error = (prediction - reference).abs().sum(dim=-1)
    return (error * weights.view(1, -1)).sum() / (weights.sum() * prediction.shape[0]).clamp_min(1.0)


TIP_PAIRS = tuple(combinations(range(len(FINGER_ORDER)), 2))


def _pair_vector_loss(prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    losses = [
        ((prediction[:, left] - prediction[:, right]) - (reference[:, left] - reference[:, right]))
        .abs()
        .sum(dim=-1)
        for left, right in TIP_PAIRS
    ]
    return torch.stack(losses, dim=1).mean()


def _pair_distance_loss(prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    losses = [
        (
            torch.linalg.norm(prediction[:, left] - prediction[:, right], dim=-1)
            - torch.linalg.norm(reference[:, left] - reference[:, right], dim=-1)
        ).abs()
        for left, right in TIP_PAIRS
    ]
    return torch.stack(losses, dim=1).mean()


def compute_native_vae_loss(
    *,
    model,
    output,
    batch: Mapping[str, torch.Tensor | Sequence[str]],
    hand_bank: NativeHandBank,
    lambda_q: float,
    lambda_tip: float,
    lambda_cross_abs: float,
    lambda_cross_pair_vector: float,
    lambda_cross_pair_distance: float,
    beta_action: float,
    beta_morphology: float,
    finger_weights_cross_abs: Sequence[float],
) -> dict[str, torch.Tensor]:
    device = output.q_norm_hat.device
    q_norm = torch.as_tensor(batch["q_norm"], device=device)
    q_mask = torch.as_tensor(batch["q_mask"], device=device)
    q_lower = torch.as_tensor(batch["q_lower"], device=device)
    q_upper = torch.as_tensor(batch["q_upper"], device=device)
    x_gesture = torch.as_tensor(batch["x_gesture"], device=device)
    hand_names = list(batch["hand_name"])

    q_error = F.smooth_l1_loss(output.q_norm_hat, q_norm, reduction="none")
    loss_q = (q_error * q_mask).sum() / q_mask.sum().clamp_min(1.0)
    q_prediction = denormalize_q(output.q_norm_hat, q_lower, q_upper, q_mask)
    reference_tips_all = tips_from_kinematic_chain(x_gesture)

    unit_weights = torch.ones(len(FINGER_ORDER), dtype=q_prediction.dtype, device=device)
    cross_weights = torch.as_tensor(
        finger_weights_cross_abs,
        dtype=q_prediction.dtype,
        device=device,
    )
    loss_tip = q_prediction.new_zeros(())
    loss_cross_abs = q_prediction.new_zeros(())
    loss_pair_vector = q_prediction.new_zeros(())
    loss_pair_distance = q_prediction.new_zeros(())
    unique_hands = sorted(set(hand_names))
    cross_groups = 0

    use_cross = any(
        weight != 0.0
        for weight in (lambda_cross_abs, lambda_cross_pair_vector, lambda_cross_pair_distance)
    )
    for source_hand in unique_hands:
        indices = torch.tensor(
            [index for index, name in enumerate(hand_names) if name == source_hand],
            device=device,
            dtype=torch.long,
        )
        source_runtime = hand_bank.runtimes[source_hand]
        source_q = q_prediction.index_select(0, indices)
        reference_tips = reference_tips_all.index_select(0, indices)
        predicted_tips = source_runtime.tips_from_padded_q(source_q)
        loss_tip = loss_tip + _weighted_tip_l1(predicted_tips, reference_tips, unit_weights)

        if not use_cross:
            continue
        source_z = output.z_action.index_select(0, indices)
        for target_hand, target_runtime in hand_bank.runtimes.items():
            if target_hand == source_hand:
                continue
            target = hand_bank.target_batch(target_hand, len(indices), device)
            target_morphology, _ = model.encode_morphology(target["morphology_vec"])
            target_q_norm = model.decode_joints(
                source_z,
                target_morphology,
                joint_queries=target["joint_queries"],
                q_mask=target["q_mask"],
            )
            target_q = denormalize_q(
                target_q_norm,
                target["q_lower"],
                target["q_upper"],
                target["q_mask"],
            )
            target_tips = target_runtime.tips_from_padded_q(target_q)
            if lambda_cross_abs != 0.0:
                loss_cross_abs = loss_cross_abs + _weighted_tip_l1(
                    target_tips,
                    reference_tips,
                    cross_weights,
                )
            if lambda_cross_pair_vector != 0.0:
                loss_pair_vector = loss_pair_vector + _pair_vector_loss(target_tips, reference_tips)
            if lambda_cross_pair_distance != 0.0:
                loss_pair_distance = loss_pair_distance + _pair_distance_loss(target_tips, reference_tips)
            cross_groups += 1

    loss_tip = loss_tip / max(len(unique_hands), 1)
    loss_cross_abs = loss_cross_abs / max(cross_groups, 1)
    loss_pair_vector = loss_pair_vector / max(cross_groups, 1)
    loss_pair_distance = loss_pair_distance / max(cross_groups, 1)
    loss_kl_action = kl_divergence(output.action_mu, output.action_log_var)
    loss_kl_morphology = kl_divergence(output.morphology_mu, output.morphology_log_var)

    total = (
        lambda_q * loss_q
        + lambda_tip * loss_tip
        + lambda_cross_abs * loss_cross_abs
        + lambda_cross_pair_vector * loss_pair_vector
        + lambda_cross_pair_distance * loss_pair_distance
        + beta_action * loss_kl_action
        + beta_morphology * loss_kl_morphology
    )
    return {
        "loss": total,
        "loss_q": loss_q,
        "loss_tip": loss_tip,
        "loss_cross_tip_abs": loss_cross_abs,
        "loss_cross_pair_vector": loss_pair_vector,
        "loss_cross_pair_distance": loss_pair_distance,
        "loss_kl_action": loss_kl_action,
        "loss_kl_morphology": loss_kl_morphology,
    }
