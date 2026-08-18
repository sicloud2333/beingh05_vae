"""Thread-safe action buffering for SYS-01 and CTRL-01 experiments."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RingBufferMetrics:
    commits: int
    consumed: int
    underflows: int
    prefix_violations: int
    max_depth: int


class ActionRingBuffer:
    """Lock committed actions while allowing an inference thread to stitch a postfix."""

    def __init__(self, capacity: int, action_dim: int):
        if capacity < 1 or action_dim < 1:
            raise ValueError("capacity and action_dim must be positive")
        self.capacity = capacity
        self.action_dim = action_dim
        self._actions: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()
        self._last_action = np.zeros(action_dim, dtype=np.float32)
        self._committed_tick = -1
        self._commits = 0
        self._consumed = 0
        self._underflows = 0
        self._prefix_violations = 0
        self._max_depth = 0

    def stitch(
        self,
        *,
        start_tick: int,
        actions: np.ndarray,
        committed_prefix_end: int,
    ) -> int:
        """Write only the postfix after both consumed and explicitly locked ticks."""
        array = np.asarray(actions, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.action_dim:
            raise ValueError(
                f"actions must have shape (T, {self.action_dim}), got {array.shape}"
            )
        written = 0
        with self._lock:
            hard_lock_end = max(self._committed_tick, committed_prefix_end)
            for offset, action in enumerate(array):
                tick = start_tick + offset
                if tick <= hard_lock_end:
                    previous = self._actions.get(tick)
                    if previous is not None and not np.array_equal(previous, action):
                        self._prefix_violations += 1
                    continue
                self._actions[tick] = action.copy()
                written += 1
            self._commits += 1
            while len(self._actions) > self.capacity:
                # Preserve the nearest executable actions; discard the
                # farthest speculative postfix first.
                farthest = max(self._actions)
                del self._actions[farthest]
            self._max_depth = max(self._max_depth, len(self._actions))
        return written

    def consume(self, tick: int) -> tuple[np.ndarray, bool]:
        """Consume one control tick; repeat the last action on underflow."""
        with self._lock:
            if tick <= self._committed_tick:
                raise ValueError("control ticks must be strictly increasing")
            action = self._actions.pop(tick, None)
            underflow = action is None
            if underflow:
                action = self._last_action.copy()
                self._underflows += 1
            else:
                self._last_action = action.copy()
            self._committed_tick = tick
            self._consumed += 1
            return action, underflow

    def depth_after(self, tick: int) -> int:
        with self._lock:
            return sum(index > tick for index in self._actions)

    def metrics(self) -> RingBufferMetrics:
        with self._lock:
            return RingBufferMetrics(
                commits=self._commits,
                consumed=self._consumed,
                underflows=self._underflows,
                prefix_violations=self._prefix_violations,
                max_depth=self._max_depth,
            )


@dataclass(frozen=True)
class ChunkConsumptionPlan:
    control_hz: float
    model_chunk_length: int
    consume_length: int
    p95_latency_ms: float
    safety_ticks: int = 1

    def __post_init__(self):
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if not 1 <= self.consume_length <= self.model_chunk_length:
            raise ValueError("consume_length must be within the model chunk")
        if self.p95_latency_ms < 0 or self.safety_ticks < 0:
            raise ValueError("latency and safety_ticks must be non-negative")

    @property
    def latency_commitment_ticks(self) -> int:
        tick_ms = 1000.0 / self.control_hz
        return math.ceil(self.p95_latency_ms / tick_ms) + self.safety_ticks

    @property
    def request_rate_hz(self) -> float:
        return self.control_hz / self.consume_length

    @property
    def amortized_inference_ms_per_action(self) -> float:
        return self.p95_latency_ms / self.consume_length

    @property
    def maximum_observation_age_ms(self) -> float:
        return self.p95_latency_ms + (
            (self.consume_length - 1) * 1000.0 / self.control_hz
        )

    def should_request(self, remaining_actions: int) -> bool:
        return remaining_actions <= self.latency_commitment_ticks


def build_consumption_grid(
    *,
    control_rates_hz: Iterable[float],
    consume_lengths: Iterable[int],
    model_chunk_length: int,
    p95_latency_ms: float,
    safety_ticks: int = 1,
) -> list[ChunkConsumptionPlan]:
    return [
        ChunkConsumptionPlan(
            control_hz=rate,
            model_chunk_length=model_chunk_length,
            consume_length=length,
            p95_latency_ms=p95_latency_ms,
            safety_ticks=safety_ticks,
        )
        for rate in control_rates_hz
        for length in consume_lengths
    ]
