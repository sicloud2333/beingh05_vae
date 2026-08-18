"""Decision helpers for experimental adaptive Being-H inference."""

from __future__ import annotations

import torch


def relative_velocity_residual(
    actions: torch.Tensor, velocity: torch.Tensor
) -> torch.Tensor:
    """Return a scale-normalized mean absolute Euler velocity."""
    numerator = velocity.float().abs().mean()
    denominator = actions.float().abs().mean().clamp_min(1e-6)
    return numerator / denominator


def should_terminate_flow(
    *,
    completed_steps: int,
    min_steps: int,
    residual: float,
    threshold: float,
) -> bool:
    if min_steps < 1:
        raise ValueError("min_steps must be at least one")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    return completed_steps >= min_steps and residual <= threshold


def euler_extrapolate_remaining(
    actions: torch.Tensor,
    velocity: torch.Tensor,
    *,
    dt: float,
    remaining_steps: int,
) -> torch.Tensor:
    """Finish the remaining integration interval with the latest velocity."""
    if remaining_steps < 0:
        raise ValueError("remaining_steps must be non-negative")
    return actions + (dt * remaining_steps) * velocity


def should_skip_mpg_refinement(*, gate: float, threshold: float) -> bool:
    """Skip when MPG's alignment gate says its residual would be weak."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("MPG gate threshold must be in [0, 1]")
    return gate <= threshold
