from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch

from native_vae.hand_runtime import NativeHandRuntime
from native_vae.hand_spec import load_native_hand_specs


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HAND_CONFIG = PACKAGE_ROOT / "configs/right_hands.yaml"


@dataclass(frozen=True)
class GeometryRetargeterConfig:
    """Numerical settings for the geometry-only retargeting baseline.

    ``raw`` minimizes only the NativeVAE-compatible 60D geometric error.
    ``stable`` adds normalized joint-space temporal regularization while keeping
    the same geometry, semantic frame, and hard URDF joint limits.
    """

    profile: str = "raw"
    max_iterations: int = 12
    learning_rate: float = 0.8
    tolerance: float = 1e-7
    temporal_weight: float = 2e-3
    acceleration_weight: float = 5e-4

    def __post_init__(self) -> None:
        if self.profile not in {"raw", "stable"}:
            raise ValueError("profile must be 'raw' or 'stable'")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.temporal_weight < 0 or self.acceleration_weight < 0:
            raise ValueError("temporal weights must be non-negative")


@dataclass(frozen=True)
class GeometryRetargetResult:
    source_hand: str
    target_hand: str
    target_q: np.ndarray
    source_geometry: np.ndarray
    target_geometry: np.ndarray
    geometry_rmse: float
    geometry_max_error: float
    objective: float
    iterations: int
    converged: bool
    elapsed_s: float

    def metadata(self) -> dict[str, Any]:
        return {
            "source_hand": self.source_hand,
            "target_hand": self.target_hand,
            "geometry_rmse": self.geometry_rmse,
            "geometry_max_error": self.geometry_max_error,
            "objective": self.objective,
            "iterations": self.iterations,
            "converged": self.converged,
            "elapsed_s": self.elapsed_s,
        }


@dataclass(frozen=True)
class GeometryRetargetBatchResult:
    source_hand: str
    target_hand: str
    target_q: np.ndarray
    source_geometry: np.ndarray
    target_geometry: np.ndarray
    per_frame_geometry_rmse: np.ndarray
    per_frame_geometry_max_error: np.ndarray
    geometry_rmse: float
    geometry_max_error: float
    objective: float
    iterations: int
    converged: bool
    elapsed_s: float

    @property
    def batch_size(self) -> int:
        return int(self.target_q.shape[0])

    def metadata(self) -> dict[str, Any]:
        return {
            "source_hand": self.source_hand,
            "target_hand": self.target_hand,
            "batch_size": self.batch_size,
            "geometry_rmse": self.geometry_rmse,
            "geometry_max_error": self.geometry_max_error,
            "per_frame_geometry_rmse": (
                self.per_frame_geometry_rmse.tolist()
            ),
            "objective": self.objective,
            "iterations": self.iterations,
            "converged": self.converged,
            "elapsed_s": self.elapsed_s,
        }


@dataclass
class _StreamState:
    previous_q: torch.Tensor
    previous_previous_q: torch.Tensor | None = None


class GeometryRetargeter:
    """Retarget native joint angles by optimizing NativeVAE's hand geometry.

    The objective uses the exact normalized 60D ``x_gesture`` representation
    used before NativeVAE encoding: five finger roots followed by three chain
    vectors per finger, all expressed in the fixed semantic palm frame and
    divided by that hand's palm radius. No learned VAE weights are loaded.
    """

    def __init__(
        self,
        *,
        hand_config: str | Path = DEFAULT_HAND_CONFIG,
        device: str | torch.device = "cpu",
        config: GeometryRetargeterConfig | None = None,
    ) -> None:
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            requested = torch.device("cpu")
        self.device = requested
        self.config = config or GeometryRetargeterConfig()
        self.hand_config = Path(hand_config).expanduser().resolve()
        self.specs = load_native_hand_specs(self.hand_config)
        self.runtimes = {
            name: NativeHandRuntime.build(spec, device=requested)
            for name, spec in self.specs.items()
        }
        self._streams: dict[tuple[str, str, str], _StreamState] = {}

    @property
    def hand_names(self) -> tuple[str, ...]:
        return tuple(self.runtimes)

    def joint_names(self, hand: str) -> tuple[str, ...]:
        return self._runtime(hand).spec.active_joint_names

    def reset(self, stream: str | None = None) -> None:
        if stream is None:
            self._streams.clear()
            return
        for key in [key for key in self._streams if key[2] == stream]:
            del self._streams[key]

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "geometry_optimization",
            "geometry": (
                "NativeVAE-compatible normalized 60D semantic chain vectors"
            ),
            "hand_config": str(self.hand_config),
            "device": str(self.device),
            "config": asdict(self.config),
            "hands": {
                name: {
                    "joint_names": list(runtime.spec.active_joint_names),
                    "palm_radius": runtime.palm_radius,
                }
                for name, runtime in self.runtimes.items()
            },
        }

    def _runtime(self, hand: str) -> NativeHandRuntime:
        if hand not in self.runtimes:
            raise KeyError(f"Unknown hand {hand!r}; available={self.hand_names}")
        return self.runtimes[hand]

    def _q_tensor(self, q: np.ndarray | torch.Tensor | Sequence[float], hand: str) -> torch.Tensor:
        value = torch.as_tensor(q, dtype=torch.float32, device=self.device)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        expected = len(self.joint_names(hand))
        if value.ndim != 2 or value.shape[1] != expected:
            raise ValueError(
                f"{hand}: expected q [B,{expected}], got {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"{hand}: q contains NaN or Inf")
        return value

    def geometry(self, q: np.ndarray | torch.Tensor | Sequence[float], hand: str) -> torch.Tensor:
        runtime = self._runtime(hand)
        values = self._q_tensor(q, hand)
        return runtime.kinematic_chain_gesture(values)["x_gesture"] / float(
            runtime.palm_radius
        )

    def _limits(self, runtime: NativeHandRuntime) -> tuple[torch.Tensor, torch.Tensor]:
        count = len(runtime.spec.active_joint_names)
        return (
            runtime.q_lower[:count].to(self.device).unsqueeze(0),
            runtime.q_upper[:count].to(self.device).unsqueeze(0),
        )

    @staticmethod
    def _normalize_q(
        q: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> torch.Tensor:
        return 2.0 * (q - lower) / (upper - lower).clamp_min(1e-8) - 1.0

    @staticmethod
    def _bounded_q(
        unconstrained: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> torch.Tensor:
        return lower + 0.5 * (torch.tanh(unconstrained) + 1.0) * (upper - lower)

    @staticmethod
    def _unconstrained_q(
        q: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> torch.Tensor:
        normalized = GeometryRetargeter._normalize_q(q, lower, upper)
        return torch.atanh(normalized.clamp(-0.999, 0.999))

    def _semantic_initial_guess(
        self,
        source_q: torch.Tensor,
        source_hand: str,
        target_hand: str,
    ) -> torch.Tensor:
        source_runtime = self._runtime(source_hand)
        target_runtime = self._runtime(target_hand)
        source_lower, source_upper = self._limits(source_runtime)
        target_lower, target_upper = self._limits(target_runtime)
        source_norm = self._normalize_q(
            source_q.clamp(source_lower, source_upper), source_lower, source_upper
        )
        by_semantic = {
            (
                source_runtime.spec.joint_semantics[name].finger,
                source_runtime.spec.joint_semantics[name].slot,
            ): source_norm[:, index]
            for index, name in enumerate(source_runtime.spec.active_joint_names)
        }
        by_finger: dict[str, list[torch.Tensor]] = {}
        for (finger, _), value in by_semantic.items():
            by_finger.setdefault(finger, []).append(value)
        target_norm_values = []
        for name in target_runtime.spec.active_joint_names:
            semantic = target_runtime.spec.joint_semantics[name]
            key = (semantic.finger, semantic.slot)
            if key in by_semantic:
                value = by_semantic[key]
            else:
                values = by_finger.get(semantic.finger, [])
                value = (
                    torch.stack(values, dim=0).mean(dim=0)
                    if values
                    else source_norm.new_zeros(source_norm.shape[0])
                )
            target_norm_values.append(value)
        target_norm = torch.stack(target_norm_values, dim=1)
        return target_lower + 0.5 * (target_norm + 1.0) * (
            target_upper - target_lower
        )

    def retarget(
        self,
        q: np.ndarray | torch.Tensor | Sequence[float],
        source_hand: str,
        target_hand: str,
        *,
        stream: str = "default",
        warm_start: np.ndarray | torch.Tensor | Sequence[float] | None = None,
        update_state: bool = True,
    ) -> GeometryRetargetResult:
        started_at = time.perf_counter()
        source_q = self._q_tensor(q, source_hand)
        if source_q.shape[0] != 1:
            raise ValueError(
                "retarget() accepts one pose; use retarget_batch() for [T,D]"
            )
        source_geometry = self.geometry(source_q, source_hand).detach()
        if source_hand == target_hand:
            target_q = source_q.detach().clone()
            elapsed = time.perf_counter() - started_at
            values = target_q[0].cpu().numpy().astype(np.float32)
            geometry = source_geometry[0].cpu().numpy().astype(np.float32)
            return GeometryRetargetResult(
                source_hand=source_hand,
                target_hand=target_hand,
                target_q=values,
                source_geometry=geometry,
                target_geometry=geometry.copy(),
                geometry_rmse=0.0,
                geometry_max_error=0.0,
                objective=0.0,
                iterations=0,
                converged=True,
                elapsed_s=elapsed,
            )

        target_runtime = self._runtime(target_hand)
        target_lower, target_upper = self._limits(target_runtime)
        key = (source_hand, target_hand, str(stream))
        state = self._streams.get(key)
        if warm_start is not None:
            initial_q = self._q_tensor(warm_start, target_hand)
        elif state is not None:
            initial_q = state.previous_q.detach().clone()
        else:
            initial_q = self._semantic_initial_guess(
                source_q, source_hand, target_hand
            )
        initial_q = initial_q.clamp(target_lower, target_upper)
        unconstrained = torch.nn.Parameter(
            self._unconstrained_q(initial_q, target_lower, target_upper)
        )
        optimizer = torch.optim.LBFGS(
            [unconstrained],
            lr=self.config.learning_rate,
            max_iter=self.config.max_iterations,
            tolerance_grad=self.config.tolerance,
            tolerance_change=self.config.tolerance,
            line_search_fn="strong_wolfe",
        )
        closure_calls = 0
        last_objective = float("inf")

        def closure() -> torch.Tensor:
            nonlocal closure_calls, last_objective
            optimizer.zero_grad(set_to_none=True)
            target_q = self._bounded_q(
                unconstrained, target_lower, target_upper
            )
            target_geometry = self.geometry(target_q, target_hand)
            loss = torch.mean((target_geometry - source_geometry) ** 2)
            if self.config.profile == "stable" and state is not None:
                target_norm = self._normalize_q(
                    target_q, target_lower, target_upper
                )
                previous_norm = self._normalize_q(
                    state.previous_q, target_lower, target_upper
                )
                loss = loss + self.config.temporal_weight * torch.mean(
                    (target_norm - previous_norm) ** 2
                )
                if state.previous_previous_q is not None:
                    previous_previous_norm = self._normalize_q(
                        state.previous_previous_q,
                        target_lower,
                        target_upper,
                    )
                    acceleration = (
                        target_norm
                        - 2.0 * previous_norm
                        + previous_previous_norm
                    )
                    loss = loss + self.config.acceleration_weight * torch.mean(
                        acceleration**2
                    )
            loss.backward()
            closure_calls += 1
            last_objective = float(loss.detach().cpu())
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            target_q = self._bounded_q(
                unconstrained, target_lower, target_upper
            )
            target_geometry = self.geometry(target_q, target_hand)
            difference = target_geometry - source_geometry
            rmse = float(torch.sqrt(torch.mean(difference**2)).cpu())
            max_error = float(torch.max(torch.abs(difference)).cpu())
            if update_state:
                previous_previous = (
                    None if state is None else state.previous_q.detach().clone()
                )
                self._streams[key] = _StreamState(
                    previous_q=target_q.detach().clone(),
                    previous_previous_q=previous_previous,
                )
        elapsed = time.perf_counter() - started_at
        return GeometryRetargetResult(
            source_hand=source_hand,
            target_hand=target_hand,
            target_q=target_q[0].detach().cpu().numpy().astype(np.float32),
            source_geometry=source_geometry[0].cpu().numpy().astype(np.float32),
            target_geometry=target_geometry[0].detach().cpu().numpy().astype(np.float32),
            geometry_rmse=rmse,
            geometry_max_error=max_error,
            objective=last_objective,
            iterations=closure_calls,
            converged=bool(np.isfinite(last_objective)),
            elapsed_s=elapsed,
        )

    def retarget_batch(
        self,
        q: np.ndarray | torch.Tensor,
        source_hand: str,
        target_hand: str,
        *,
        stream: str = "action_chunk",
        warm_start: np.ndarray | torch.Tensor | None = None,
        update_state: bool = True,
    ) -> GeometryRetargetBatchResult:
        """Jointly optimize an action chunk using batched differentiable FK."""
        started_at = time.perf_counter()
        source_q = self._q_tensor(q, source_hand)
        batch_size = int(source_q.shape[0])
        if batch_size <= 0:
            raise ValueError("retarget_batch requires at least one pose")
        source_geometry = self.geometry(source_q, source_hand).detach()
        if source_hand == target_hand:
            values = source_q.detach().cpu().numpy().astype(np.float32)
            geometry = source_geometry.cpu().numpy().astype(np.float32)
            zeros = np.zeros(batch_size, dtype=np.float32)
            return GeometryRetargetBatchResult(
                source_hand=source_hand,
                target_hand=target_hand,
                target_q=values,
                source_geometry=geometry,
                target_geometry=geometry.copy(),
                per_frame_geometry_rmse=zeros,
                per_frame_geometry_max_error=zeros.copy(),
                geometry_rmse=0.0,
                geometry_max_error=0.0,
                objective=0.0,
                iterations=0,
                converged=True,
                elapsed_s=time.perf_counter() - started_at,
            )

        target_runtime = self._runtime(target_hand)
        target_lower, target_upper = self._limits(target_runtime)
        key = (source_hand, target_hand, str(stream))
        state = self._streams.get(key)
        if warm_start is not None:
            initial_q = self._q_tensor(warm_start, target_hand)
            if initial_q.shape[0] != batch_size:
                raise ValueError(
                    "Batch warm_start must have the same number of frames"
                )
        else:
            initial_q = self._semantic_initial_guess(
                source_q, source_hand, target_hand
            )
            if state is not None:
                # Anchor the new chunk at the last executed/optimized target pose.
                initial_q[0] = state.previous_q[0]
        initial_q = initial_q.clamp(target_lower, target_upper)
        unconstrained = torch.nn.Parameter(
            self._unconstrained_q(initial_q, target_lower, target_upper)
        )
        optimizer = torch.optim.LBFGS(
            [unconstrained],
            lr=self.config.learning_rate,
            max_iter=self.config.max_iterations,
            tolerance_grad=self.config.tolerance,
            tolerance_change=self.config.tolerance,
            line_search_fn="strong_wolfe",
        )
        closure_calls = 0
        last_objective = float("inf")

        def closure() -> torch.Tensor:
            nonlocal closure_calls, last_objective
            optimizer.zero_grad(set_to_none=True)
            target_q = self._bounded_q(
                unconstrained, target_lower, target_upper
            )
            target_geometry = self.geometry(target_q, target_hand)
            loss = torch.mean((target_geometry - source_geometry) ** 2)
            if self.config.profile == "stable":
                target_norm = self._normalize_q(
                    target_q, target_lower, target_upper
                )
                if state is None:
                    velocity = target_norm[1:] - target_norm[:-1]
                    acceleration = (
                        target_norm[2:]
                        - 2.0 * target_norm[1:-1]
                        + target_norm[:-2]
                    )
                else:
                    previous_norm = self._normalize_q(
                        state.previous_q, target_lower, target_upper
                    )
                    extended = torch.cat([previous_norm, target_norm], dim=0)
                    velocity = extended[1:] - extended[:-1]
                    if state.previous_previous_q is None:
                        acceleration = (
                            extended[2:]
                            - 2.0 * extended[1:-1]
                            + extended[:-2]
                        )
                    else:
                        previous_previous_norm = self._normalize_q(
                            state.previous_previous_q,
                            target_lower,
                            target_upper,
                        )
                        extended = torch.cat(
                            [previous_previous_norm, previous_norm, target_norm],
                            dim=0,
                        )
                        acceleration = (
                            extended[2:]
                            - 2.0 * extended[1:-1]
                            + extended[:-2]
                        )
                if len(velocity):
                    loss = loss + self.config.temporal_weight * torch.mean(
                        velocity**2
                    )
                if len(acceleration):
                    loss = loss + self.config.acceleration_weight * torch.mean(
                        acceleration**2
                    )
            loss.backward()
            closure_calls += 1
            last_objective = float(loss.detach().cpu())
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            target_q = self._bounded_q(
                unconstrained, target_lower, target_upper
            )
            target_geometry = self.geometry(target_q, target_hand)
            difference = target_geometry - source_geometry
            per_frame_rmse = torch.sqrt(torch.mean(difference**2, dim=1))
            per_frame_max = torch.max(torch.abs(difference), dim=1).values
            rmse = float(torch.sqrt(torch.mean(difference**2)).cpu())
            max_error = float(torch.max(torch.abs(difference)).cpu())
            if update_state:
                previous_q = target_q[-1:].detach().clone()
                previous_previous_q = (
                    target_q[-2:-1].detach().clone()
                    if batch_size >= 2
                    else (
                        None
                        if state is None
                        else state.previous_q.detach().clone()
                    )
                )
                self._streams[key] = _StreamState(
                    previous_q=previous_q,
                    previous_previous_q=previous_previous_q,
                )
        return GeometryRetargetBatchResult(
            source_hand=source_hand,
            target_hand=target_hand,
            target_q=target_q.detach().cpu().numpy().astype(np.float32),
            source_geometry=source_geometry.cpu().numpy().astype(np.float32),
            target_geometry=target_geometry.detach().cpu().numpy().astype(np.float32),
            per_frame_geometry_rmse=(
                per_frame_rmse.cpu().numpy().astype(np.float32)
            ),
            per_frame_geometry_max_error=(
                per_frame_max.cpu().numpy().astype(np.float32)
            ),
            geometry_rmse=rmse,
            geometry_max_error=max_error,
            objective=last_objective,
            iterations=closure_calls,
            converged=bool(np.isfinite(last_objective)),
            elapsed_s=time.perf_counter() - started_at,
        )

    def retarget_chunk(
        self,
        q: np.ndarray | torch.Tensor,
        source_hand: str,
        target_hand: str,
        *,
        stream: str = "action",
        update_state: bool = True,
    ) -> list[GeometryRetargetResult]:
        values = np.asarray(q, dtype=np.float32)
        if values.ndim == 1:
            values = values[None]
        expected = len(self.joint_names(source_hand))
        if values.ndim != 2 or values.shape[1] != expected:
            raise ValueError(
                f"{source_hand}: expected chunk [T,{expected}], got {values.shape}"
            )
        return [
            self.retarget(
                frame,
                source_hand,
                target_hand,
                stream=stream,
                update_state=update_state,
            )
            for frame in values
        ]

    def project_motion_limits(
        self,
        source_limits: np.ndarray,
        source_hand: str,
        target_hand: str,
    ) -> np.ndarray:
        """Project absolute joint-step limits by semantic slot and range ratio."""
        values = np.asarray(source_limits, dtype=np.float32)
        source_runtime = self._runtime(source_hand)
        target_runtime = self._runtime(target_hand)
        if values.shape != (len(source_runtime.spec.active_joint_names),):
            raise ValueError(
                f"{source_hand}: expected limits [{len(source_runtime.spec.active_joint_names)}], "
                f"got {values.shape}"
            )
        source_lower, source_upper = self._limits(source_runtime)
        target_lower, target_upper = self._limits(target_runtime)
        source_ranges = (source_upper - source_lower)[0].cpu().numpy()
        target_ranges = (target_upper - target_lower)[0].cpu().numpy()
        fractions_by_semantic: dict[tuple[str, int], float] = {}
        fractions_by_finger: dict[str, list[float]] = {}
        for index, name in enumerate(source_runtime.spec.active_joint_names):
            semantic = source_runtime.spec.joint_semantics[name]
            fraction = float(values[index] / max(float(source_ranges[index]), 1e-8))
            fractions_by_semantic[(semantic.finger, semantic.slot)] = fraction
            fractions_by_finger.setdefault(semantic.finger, []).append(fraction)
        projected = np.empty(len(target_runtime.spec.active_joint_names), dtype=np.float32)
        for index, name in enumerate(target_runtime.spec.active_joint_names):
            semantic = target_runtime.spec.joint_semantics[name]
            fraction = fractions_by_semantic.get((semantic.finger, semantic.slot))
            if fraction is None:
                fraction = float(np.median(fractions_by_finger[semantic.finger]))
            projected[index] = max(fraction * float(target_ranges[index]), 1e-5)
        return projected
