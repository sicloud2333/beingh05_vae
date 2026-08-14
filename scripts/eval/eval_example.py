#!/usr/bin/env python3
"""Run remote GR00T inference for dual Z1 and dual Revo3."""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from unitree_deploy.policy.gr00t_revo3_z1.action_adapter import (
    CartesianActionTarget,
    Gr00tRevo3Z1ActionAdapter,
    RobotActionTarget,
)
from unitree_deploy.policy.gr00t_revo3_z1.chunk_buffer import (
    ActionChunk,
    LatestActionChunkBuffer,
    RTCActionQueue,
    TemporalActionEnsembler,
)
from unitree_deploy.policy.gr00t_revo3_z1.modality import (
    Gr00tPolicyContract,
    RTCServerContract,
)
from unitree_deploy.policy.gr00t_revo3_z1.observation_adapter import (
    Gr00tRevo3Z1ObservationAdapter,
    RobotObservationSnapshot,
)
from unitree_deploy.policy.gr00t_revo3_z1.performance import (
    WindowedPerformanceProfiler,
    format_performance_report,
)
from unitree_deploy.policy.gr00t_revo3_z1.safety import (
    Gr00tRevo3Z1Safety,
    SafetyLimits,
    SafetyViolation,
)
from unitree_deploy.policy.gr00t_revo3_z1.rtc import (
    InferenceDelayEstimator,
    RTCClientConfig,
    validate_rtc_response_info,
)
from unitree_deploy.robot_devices.robots_devices_utils import precise_wait
from unitree_deploy.utils.rich_logger import (
    log_error,
    log_info,
    log_success,
    log_warning,
)
from unitree_deploy.utils.weighted_moving_filter import EmaFilter


ROBOT_TYPE = "z1_dual_revo3_policy_realsense"
SHADOW_TOKEN = "CONNECT_Z1_REVO3_GR00T"
LIVE_TOKEN = "ENABLE_Z1_REVO3_GR00T"
AUTO_TOKEN = "ENABLE_Z1_REVO3_GR00T_AUTO"
REVO3_FLEXION_INDICES = np.asarray(
    [index for index in range(21) if index not in (5, 9, 13, 17)],
    dtype=np.int64,
)


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    rollout_id: int
    sequence_id: int
    target_step: int
    observation_at: float
    submitted_at: float
    observation: dict[str, Any]
    options: dict[str, Any] | None = None


class InferenceWorker:
    """Run the blocking ZMQ REQ client outside the 30 Hz control loop."""

    def __init__(
        self,
        host: str,
        port: int,
        timeout_ms: int,
        api_token: str | None,
        contract: Gr00tPolicyContract,
        chunk_buffer: (
            LatestActionChunkBuffer | TemporalActionEnsembler | RTCActionQueue
        ),
        profiler: WindowedPerformanceProfiler,
        delay_estimator: InferenceDelayEstimator | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self.contract = contract
        self.chunk_buffer = chunk_buffer
        self.profiler = profiler
        self.delay_estimator = delay_estimator
        self._requests: queue.Queue[InferenceRequest] = queue.Queue(maxsize=1)
        self._events: queue.Queue[str] = queue.Queue(maxsize=16)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="gr00t-z1-revo3-inference",
            daemon=True,
        )
        self._thread.start()

    def submit(self, request: InferenceRequest) -> None:
        replaced = 0
        while True:
            try:
                self._requests.get_nowait()
                replaced += 1
            except queue.Empty:
                break
        self.profiler.increment("inference_submitted")
        if replaced:
            self.profiler.increment("inference_requests_replaced", replaced)
        try:
            self._requests.put_nowait(request)
        except queue.Full:
            self.profiler.increment("inference_requests_replaced")

    def drain_events(self) -> list[str]:
        events: list[str] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def close(self, join_timeout_s: float | None = None) -> None:
        self._stop_event.set()
        if join_timeout_s is None:
            join_timeout_s = max(2.0, self.timeout_ms / 1000.0 + 0.5)
        self._thread.join(timeout=max(0.0, join_timeout_s))

    def _put_event(self, message: str) -> None:
        try:
            self._events.put_nowait(message)
        except queue.Full:
            try:
                self._events.get_nowait()
            except queue.Empty:
                pass
            try:
                self._events.put_nowait(message)
            except queue.Full:
                pass

    def _run(self) -> None:
        from unitree_deploy.policy.gr00t_revo3_z1.lightweight_client import (
            PolicyClient,
        )

        active_rollout = None
        client = PolicyClient(
            host=self.host,
            port=self.port,
            timeout_ms=self.timeout_ms,
            api_token=self.api_token,
        )
        try:
            while not self._stop_event.is_set():
                try:
                    request = self._requests.get(timeout=0.1)
                except queue.Empty:
                    continue
                dequeued_at = time.monotonic()
                self.profiler.record(
                    "inference_queue_wait",
                    dequeued_at - request.submitted_at,
                )
                try:
                    if request.rollout_id != active_rollout:
                        reset_started_at = time.monotonic()
                        reset_options = (
                            {
                                "rtc": {
                                    "protocol_version": 1,
                                    "rollout_id": request.rollout_id,
                                }
                            }
                            if request.options is not None
                            else None
                        )
                        client.reset(reset_options)
                        self.profiler.record(
                            "policy_reset",
                            time.monotonic() - reset_started_at,
                        )
                        active_rollout = request.rollout_id
                    policy_started_at = time.monotonic()
                    action, info = client.get_action(
                        request.observation,
                        options=request.options,
                    )
                    policy_finished_at = time.monotonic()
                    self.profiler.record(
                        "policy_roundtrip",
                        policy_finished_at - policy_started_at,
                    )
                    call_timing = client.get_last_call_timing()
                    if call_timing is not None:
                        self.profiler.increment(
                            "inference_request_bytes",
                            call_timing.request_bytes,
                        )
                        self.profiler.increment(
                            "inference_response_bytes",
                            call_timing.response_bytes,
                        )
                        self.profiler.record(
                            "client_serialize",
                            call_timing.serialize_s,
                        )
                        self.profiler.record("client_send", call_timing.send_s)
                        self.profiler.record(
                            "client_recv_wait",
                            call_timing.receive_s,
                        )
                        self.profiler.record(
                            "client_deserialize",
                            call_timing.deserialize_s,
                        )
                    validation_started_at = time.monotonic()
                    action = self.contract.validate_action(action)
                    metadata = (
                        validate_rtc_response_info(
                            info,
                            request_options=request.options,
                        )
                        if request.options is not None
                        else (dict(info) if isinstance(info, dict) else {})
                    )
                    if request.options is not None:
                        rtc_options = request.options["rtc"]
                        prefix = rtc_options["prev_chunk_left_over"]
                        metadata["requested_inference_delay"] = int(
                            rtc_options["inference_delay"]
                        )
                        metadata["prefix_horizon"] = (
                            0
                            if prefix is None
                            else int(next(iter(prefix.values())).shape[1])
                        )
                    self.profiler.record(
                        "action_validation",
                        time.monotonic() - validation_started_at,
                    )
                    received_at = time.monotonic()
                    if self.delay_estimator is not None:
                        self.delay_estimator.add_latency(
                            received_at - request.submitted_at
                        )
                    self.profiler.record(
                        "inference_end_to_end",
                        received_at - request.submitted_at,
                    )
                    self.profiler.increment("inference_completed")
                    self.chunk_buffer.publish(
                        ActionChunk(
                            rollout_id=request.rollout_id,
                            sequence_id=request.sequence_id,
                            observation_at=request.observation_at,
                            submitted_at=request.submitted_at,
                            received_at=received_at,
                            actions=action,
                            target_step=request.target_step,
                            metadata=metadata,
                        )
                    )
                except Exception as exc:
                    self.profiler.record(
                        "inference_end_to_end",
                        time.monotonic() - request.submitted_at,
                    )
                    self.profiler.increment("inference_failures")
                    self._put_event(
                        f"GR00T inference failed for sequence "
                        f"{request.sequence_id}: {exc}"
                    )
        finally:
            client.close()


def _first_hand_motion_step(
    action: np.ndarray,
    current_q: np.ndarray,
    threshold: float,
) -> tuple[int | None, float]:
    """Find the first predicted flexion-motion step relative to current feedback."""
    values = np.asarray(action, dtype=np.float64)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[2] != 21:
        raise ValueError(f"Hand action must have shape (1, T, 21), got {values.shape}.")
    reference = np.asarray(current_q, dtype=np.float64).reshape(-1)
    if reference.shape != (21,):
        raise ValueError(f"Hand feedback must have shape (21,), got {reference.shape}.")
    deltas = np.max(
        np.abs(
            values[0][:, REVO3_FLEXION_INDICES]
            - reference[REVO3_FLEXION_INDICES]
        ),
        axis=1,
    )
    matching = np.flatnonzero(deltas >= threshold)
    first_step = int(matching[0]) if matching.size else None
    return first_step, float(np.max(deltas))


def _shadow_hand_order_diagnostic(
    action: dict[str, np.ndarray],
    left_current_q: np.ndarray,
    right_current_q: np.ndarray,
    threshold: float,
) -> str:
    """Summarize which predicted hand first departs from current feedback."""
    left_step, left_peak = _first_hand_motion_step(
        action["left_hand"],
        left_current_q,
        threshold,
    )
    right_step, right_peak = _first_hand_motion_step(
        action["right_hand"],
        right_current_q,
        threshold,
    )
    if left_step is None and right_step is None:
        first = "none"
    elif right_step is None or (
        left_step is not None and left_step < right_step
    ):
        first = "left"
    elif left_step is None or right_step < left_step:
        first = "right"
    else:
        first = "same"
    return (
        f"hand_motion_first={first}, "
        f"left_first_step={left_step}, right_first_step={right_step}, "
        f"left_peak_delta={left_peak:.3f}rad, "
        f"right_peak_delta={right_peak:.3f}rad, "
        f"threshold={threshold:.3f}rad"
    )


def _snapshot_fallback_action(
    snapshot: RobotObservationSnapshot,
) -> dict[str, np.ndarray]:
    """Build a one-step hold action from the latest measured robot state."""
    return {
        "left_revo_pose_in_head": np.asarray(
            snapshot.left_revo_pose_in_head,
            dtype=np.float64,
        ).reshape(1, 1, 9),
        "right_revo_pose_in_head": np.asarray(
            snapshot.right_revo_pose_in_head,
            dtype=np.float64,
        ).reshape(1, 1, 9),
        "left_hand": np.asarray(
            snapshot.left_hand_q,
            dtype=np.float64,
        ).reshape(1, 1, 21),
        "right_hand": np.asarray(
            snapshot.right_hand_q,
            dtype=np.float64,
        ).reshape(1, 1, 21),
    }


def _reset_ik_q_filter(
    ik_q_filter: EmaFilter | None,
    arm_q: np.ndarray,
) -> None:
    """Reset and seed the IK joint filter from current arm feedback."""
    if ik_q_filter is None:
        return
    current_q = np.asarray(arm_q, dtype=np.float64).reshape(-1)
    if current_q.shape != (12,) or not np.isfinite(current_q).all():
        raise ValueError(
            f"IK EMA seed must be a finite 12D arm vector, got {current_q.shape}."
        )
    ik_q_filter.reset()
    ik_q_filter.add_data(current_q)


def _filter_ik_target(
    ik_q_filter: EmaFilter | None,
    target: RobotActionTarget,
) -> RobotActionTarget:
    """Apply optional EMA to a successful IK solution."""
    if ik_q_filter is None:
        return target
    filtered_arm_q = ik_q_filter.add_data(target.arm_q)
    if filtered_arm_q.shape != (12,) or not np.isfinite(filtered_arm_q).all():
        raise ValueError("IK EMA produced an invalid dual-arm target.")
    return replace(target, arm_q=filtered_arm_q)


def _record_capture_timing(
    profiler: WindowedPerformanceProfiler,
    observation_adapter: Gr00tRevo3Z1ObservationAdapter,
) -> None:
    """Copy the latest observation timing breakdown into the profiler."""
    for name, duration_s in observation_adapter.last_capture_timing_s.items():
        profiler.record(name, duration_s)


def _finish_profiled_cycle(
    profiler: WindowedPerformanceProfiler,
    *,
    cycle_started_at: float,
    cycle_end: float,
    mode: str,
    control_dt: float,
    report_enabled: bool,
) -> None:
    """Finish one fixed-rate cycle and emit a report when its window is due."""
    active_finished_at = time.monotonic()
    profiler.record("active_work", active_finished_at - cycle_started_at)
    overrun_s = max(0.0, active_finished_at - cycle_end)
    if overrun_s > 0:
        profiler.increment("cycle_overruns")
        profiler.record("cycle_overrun", overrun_s)
    else:
        precise_wait(cycle_end)
    profiler.record("cycle_total", time.monotonic() - cycle_started_at)
    profiler.increment("cycles")
    if not report_enabled:
        return
    report = profiler.report_if_due()
    if report is not None:
        log_info(
            format_performance_report(
                report,
                mode=mode,
                control_period_s=control_dt,
            )
        )


class HoldToRunControl:
    """Thread-safe manual/automatic execution and emergency-stop state."""

    def __init__(self, *, automatic: bool = False) -> None:
        self._lock = threading.Lock()
        self._automatic = automatic
        self._enabled = False
        self._resume_requested = False
        self._origin_reset_requested = False
        self._origin_reset_key_down = False
        self._exit_requested = False
        self._emergency_stop_requested = False

    def arm_automatic_rollout(self) -> bool:
        """Start one continuous rollout unless stop has already been requested."""
        with self._lock:
            if (
                not self._automatic
                or self._exit_requested
                or self._emergency_stop_requested
            ):
                return False
            self._enabled = True
            self._resume_requested = True
            return True

    def request_emergency_stop(self, reason: str) -> None:
        """Latch emergency stop; it cannot be cleared during this process."""
        with self._lock:
            if self._emergency_stop_requested:
                return
            self._enabled = False
            self._exit_requested = True
            self._emergency_stop_requested = True
        print(f"EMERGENCY STOP latched ({reason})")

    def on_press(self, key) -> None:
        key_name = getattr(key, "name", None)
        if key_name in {"space", "esc"}:
            self.request_emergency_stop(key_name)
            return
        try:
            char = key.char.lower()
        except (AttributeError, TypeError):
            return
        if char == "x":
            self.request_emergency_stop("x")
            return
        with self._lock:
            if char == "c" and not self._automatic and not self._enabled:
                self._enabled = True
                self._resume_requested = True
                print("'c' pressed: start a fresh GR00T rollout")
            elif (
                char == "r"
                and not self._origin_reset_key_down
                and not self._exit_requested
                and not self._emergency_stop_requested
            ):
                self._enabled = False
                self._origin_reset_requested = True
                self._origin_reset_key_down = True
                print(
                    "'r' pressed: return to inference origin and prepare "
                    "the next rollout"
                )
            elif char == "e":
                self._enabled = False
                self._exit_requested = True
                print("'e' pressed: safe shutdown requested")

    def on_release(self, key) -> None:
        try:
            char = key.char.lower()
        except (AttributeError, TypeError):
            return
        if char == "r":
            with self._lock:
                self._origin_reset_key_down = False
            return
        if char != "c":
            return
        with self._lock:
            if self._automatic:
                return
            self._enabled = False
            print("'c' released: policy execution paused")

    def snapshot(self) -> tuple[bool, bool, bool, bool]:
        with self._lock:
            resume = self._resume_requested
            self._resume_requested = False
            return (
                self._enabled,
                resume,
                self._exit_requested,
                self._emergency_stop_requested,
            )

    def emergency_stop_requested(self) -> bool:
        with self._lock:
            return self._emergency_stop_requested

    def consume_origin_reset_request(self) -> bool:
        """Consume one operator request to return to the inference origin."""
        with self._lock:
            requested = self._origin_reset_requested
            self._origin_reset_requested = False
            return requested

    def stop_requested(self) -> bool:
        """Return whether reset motion must stop immediately."""
        with self._lock:
            return self._exit_requested or self._emergency_stop_requested


def _wait_for_automatic_start(
    keyboard_control: HoldToRunControl,
    delay_s: float,
) -> bool:
    """Wait for the armed countdown while continuing to accept stop keys."""
    deadline = time.monotonic() + delay_s
    last_reported_second = None
    while True:
        _, _, exit_requested, emergency_stop_requested = (
            keyboard_control.snapshot()
        )
        if exit_requested or emergency_stop_requested:
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return keyboard_control.arm_automatic_rollout()
        second = max(1, int(np.ceil(remaining)))
        if second != last_reported_second:
            log_warning(
                f"Automatic model execution starts in {second}s. "
                "Press Space, Esc, or x for EMERGENCY STOP; press e for safe exit."
            )
            last_reported_second = second
        time.sleep(min(0.05, remaining))


def _server_preflight(
    args: argparse.Namespace,
) -> tuple[Gr00tPolicyContract, RTCServerContract | None]:
    from unitree_deploy.policy.gr00t_revo3_z1.lightweight_client import (
        PolicyClient,
    )

    with PolicyClient(
        host=args.policy_host,
        port=args.policy_port,
        timeout_ms=args.timeout_ms,
        api_token=args.api_token,
    ) as client:
        if not client.ping():
            raise ConnectionError(
                f"GR00T PolicyServer did not respond at "
                f"{args.policy_host}:{args.policy_port}."
            )
        contract = Gr00tPolicyContract.from_server_response(
            client.get_modality_config()
        )
        rtc_contract = (
            RTCServerContract.from_server_response(
                client.get_rtc_contract(),
                policy_contract=contract,
            )
            if args.chunk_strategy == "rtc"
            else None
        )
    log_success(
        "GR00T server contract validated: "
        f"video_horizon={len(contract.video.delta_indices)}, "
        f"state_horizon={len(contract.state.delta_indices)}, "
        f"action_horizon={contract.action_horizon}."
    )
    if rtc_contract is not None:
        log_success(
            "GR00T RTC contract validated: "
            f"protocol={rtc_contract.protocol_version}, "
            f"action_horizon={rtc_contract.action_horizon}, "
            f"schedules={rtc_contract.supported_schedules}, "
            f"eef_format={rtc_contract.eef_format}."
        )
    return contract, rtc_contract


def _get_controllers(robot):
    if len(robot.arm) != 1:
        raise RuntimeError(f"Expected one dual-Z1 controller, got {len(robot.arm)}.")
    if set(robot.endeffector) != {"left", "right"}:
        raise RuntimeError(
            "Expected Revo3 controllers named left and right, got "
            f"{tuple(robot.endeffector)}."
        )
    return (
        next(iter(robot.arm.values())),
        robot.endeffector["left"],
        robot.endeffector["right"],
    )


def _initialize_hardware(
    arm,
    left_hand,
    right_hand,
    control_dt: float,
    duration_s: float,
) -> None:
    """Initialize hands and synchronize Mink after the arm connect start pose."""
    _drive_revo3_to_zero(
        left_hand,
        right_hand,
        control_dt=control_dt,
        duration_s=duration_s,
    )

    arm_q = np.asarray(arm.read_current_arm_q(), dtype=np.float64)
    arm.z1_mink.set_dual_arm_q(arm_q)
    arm.write_arm(
        q_target=arm_q,
        time_target=time.monotonic() + control_dt,
        cmd_target="schedule_waypoint",
    )
    log_success("Z1/Revo3 policy hardware initialization completed.")


def _drive_revo3_to_zero(
    left_hand,
    right_hand,
    *,
    control_dt: float,
    duration_s: float,
) -> None:
    """Continuously command all joints of both Revo3 hands to zero."""
    deadline = time.monotonic() + duration_s
    zero = np.zeros(21, dtype=np.float64)
    while time.monotonic() < deadline:
        cycle_end = min(deadline, time.monotonic() + control_dt)
        left_hand.write_endeffector(zero)
        right_hand.write_endeffector(zero)
        precise_wait(cycle_end)


def _return_to_inference_origin(
    arm,
    left_hand,
    right_hand,
    *,
    arm_q: np.ndarray,
    left_hand_q: np.ndarray,
    right_hand_q: np.ndarray,
    control_dt: float,
    duration_s: float,
    stop_requested: Callable[[], bool] | None = None,
    arm_tolerance_rad: float = 0.08,
    hand_tolerance_rad: float = 0.20,
    settle_timeout_s: float = 2.0,
    arm_abort_tolerance_rad: float = 0.20,
    hand_abort_tolerance_rad: float = 0.50,
) -> None:
    """Move all controlled joints back to the captured policy origin."""
    target_arm_q = np.asarray(arm_q, dtype=np.float64).reshape(-1)
    target_left_q = np.asarray(left_hand_q, dtype=np.float64).reshape(-1)
    target_right_q = np.asarray(right_hand_q, dtype=np.float64).reshape(-1)
    if target_arm_q.shape != (12,):
        raise ValueError(
            f"Inference-origin arm target must have shape (12,), "
            f"got {target_arm_q.shape}."
        )
    if target_left_q.shape != (21,) or target_right_q.shape != (21,):
        raise ValueError(
            "Inference-origin Revo3 targets must have shape (21,) per hand."
        )
    if not all(
        np.isfinite(value).all()
        for value in (target_arm_q, target_left_q, target_right_q)
    ):
        raise ValueError("Inference-origin targets contain NaN or Inf.")
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Inference-origin reset duration must be positive.")
    if not np.isfinite(settle_timeout_s) or settle_timeout_s < 0:
        raise ValueError("Inference-origin settle timeout must be non-negative.")

    started_at = time.monotonic()
    deadline = started_at + duration_s
    arm.write_arm(
        q_target=target_arm_q,
        time_target=deadline,
        cmd_target="schedule_waypoint",
    )
    while time.monotonic() < deadline:
        if stop_requested is not None and stop_requested():
            raise InterruptedError("Inference-origin reset was interrupted.")
        cycle_end = min(deadline, time.monotonic() + control_dt)
        left_hand.write_endeffector(target_left_q)
        right_hand.write_endeffector(target_right_q)
        precise_wait(cycle_end)

    settle_deadline = time.monotonic() + settle_timeout_s
    while True:
        if stop_requested is not None and stop_requested():
            raise InterruptedError("Inference-origin reset was interrupted.")
        current_arm_q = np.asarray(arm.read_current_arm_q(), dtype=np.float64)
        current_left_q = np.asarray(
            left_hand.read_current_endeffector_q(),
            dtype=np.float64,
        )
        current_right_q = np.asarray(
            right_hand.read_current_endeffector_q(),
            dtype=np.float64,
        )
        if (
            current_arm_q.shape != (12,)
            or current_left_q.shape != (21,)
            or current_right_q.shape != (21,)
        ):
            raise RuntimeError("Invalid hardware feedback shape after origin reset.")
        arm_error = float(np.max(np.abs(current_arm_q - target_arm_q)))
        hand_error = float(
            max(
                np.max(np.abs(current_left_q - target_left_q)),
                np.max(np.abs(current_right_q - target_right_q)),
            )
        )
        if arm_error <= arm_tolerance_rad and hand_error <= hand_tolerance_rad:
            origin_reached = True
            break
        if time.monotonic() >= settle_deadline:
            origin_reached = False
            break
        cycle_end = min(settle_deadline, time.monotonic() + control_dt)
        left_hand.write_endeffector(target_left_q)
        right_hand.write_endeffector(target_right_q)
        precise_wait(cycle_end)

    if (
        arm_error > arm_abort_tolerance_rad
        or hand_error > hand_abort_tolerance_rad
    ):
        raise RuntimeError(
            "Inference-origin reset remains far outside the safe tolerance: "
            f"arm_max_abs_err={arm_error:.4f}rad "
            f"(abort={arm_abort_tolerance_rad:.4f}), "
            f"hand_max_abs_err={hand_error:.4f}rad "
            f"(abort={hand_abort_tolerance_rad:.4f})."
        )

    hold_arm_q = target_arm_q if origin_reached else current_arm_q
    arm.write_arm(
        q_target=hold_arm_q,
        time_target=time.monotonic() + control_dt,
        cmd_target="schedule_waypoint",
    )
    left_hand.write_endeffector(target_left_q)
    right_hand.write_endeffector(target_right_q)
    arm.z1_mink.set_dual_arm_q(current_arm_q)
    if origin_reached:
        log_success(
            "Returned to inference origin: "
            f"arm_max_abs_err={arm_error:.4f}rad, "
            f"hand_max_abs_err={hand_error:.4f}rad."
        )
    else:
        log_warning(
            "Inference-origin reset reached only the soft fallback tolerance; "
            "the next rollout will use measured feedback as its actual origin: "
            f"arm_max_abs_err={arm_error:.4f}rad "
            f"(preferred={arm_tolerance_rad:.4f}), "
            f"hand_max_abs_err={hand_error:.4f}rad "
            f"(preferred={hand_tolerance_rad:.4f})."
        )


def _warmup_policy(
    args: argparse.Namespace,
    contract: Gr00tPolicyContract,
    observation: dict[str, Any],
) -> None:
    from unitree_deploy.policy.gr00t_revo3_z1.lightweight_client import (
        PolicyClient,
    )

    if args.warmup_requests <= 0:
        return
    with PolicyClient(
        host=args.policy_host,
        port=args.policy_port,
        timeout_ms=args.timeout_ms,
        api_token=args.api_token,
    ) as client:
        client.reset()
        for index in range(args.warmup_requests):
            started_at = time.monotonic()
            action, _ = client.get_action(observation)
            contract.validate_action(action)
            latency = time.monotonic() - started_at
            log_info(
                f"Shadow warmup {index + 1}/{args.warmup_requests}: "
                f"{latency * 1000.0:.1f}ms."
            )
    log_success("GR00T warmup complete; all warmup actions were discarded.")


def _write_target(
    arm,
    left_hand,
    right_hand,
    target,
    command_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    arm.write_arm(
        q_target=target.arm_q,
        time_target=command_time,
        cmd_target="schedule_waypoint",
    )
    left_sent = left_hand.write_endeffector(target.left_hand_q)
    right_sent = right_hand.write_endeffector(target.right_hand_q)
    if left_sent is None:
        left_sent = target.left_hand_q
    if right_sent is None:
        right_sent = target.right_hand_q
    return (
        np.asarray(left_sent, dtype=np.float64),
        np.asarray(right_sent, dtype=np.float64),
    )


def _safe_exit_reset(
    arm,
    left_hand,
    right_hand,
    control_dt: float,
    duration_s: float,
) -> None:
    log_info("Resetting dual Z1 to start pose and Revo3 joints to zero.")
    _drive_revo3_to_zero(
        left_hand,
        right_hand,
        control_dt=control_dt,
        duration_s=duration_s,
    )
    stop_zero_stream = threading.Event()
    zero_stream_errors: list[Exception] = []

    def stream_zero() -> None:
        zero = np.zeros(21, dtype=np.float64)
        while not stop_zero_stream.is_set():
            cycle_end = time.monotonic() + control_dt
            try:
                left_hand.write_endeffector(zero)
                right_hand.write_endeffector(zero)
            except Exception as exc:
                zero_stream_errors.append(exc)
                stop_zero_stream.set()
                return
            precise_wait(cycle_end)

    zero_thread = threading.Thread(
        target=stream_zero,
        name="revo3-safe-reset-zero",
        daemon=True,
    )
    zero_thread.start()
    arm_reset_error: Exception | None = None
    try:
        arm.go_start()
    except Exception as exc:
        arm_reset_error = exc
    finally:
        stop_zero_stream.set()
        zero_thread.join(timeout=max(1.0, 2.0 * control_dt))

    _drive_revo3_to_zero(
        left_hand,
        right_hand,
        control_dt=control_dt,
        duration_s=max(0.25, 2.0 * control_dt),
    )
    _verify_revo3_zero_feedback(
        left_hand,
        right_hand,
        control_dt=control_dt,
        timeout_s=max(1.0, duration_s),
    )
    if zero_stream_errors:
        raise RuntimeError(
            f"Revo3 zero stream failed during Z1 reset: {zero_stream_errors[0]}"
        )
    if arm_reset_error is not None:
        raise RuntimeError(f"Dual-Z1 go_start failed: {arm_reset_error}")
    log_success("Dual Z1 returned to start pose; all Revo3 joints commanded to zero.")


def _verify_revo3_zero_feedback(
    left_hand,
    right_hand,
    *,
    control_dt: float,
    timeout_s: float,
    tolerance_rad: float = 0.20,
) -> None:
    """Keep commanding zero until calibrated Revo3 feedback is near zero."""
    deadline = time.monotonic() + timeout_s
    zero = np.zeros(21, dtype=np.float64)
    last_error = float("inf")
    while time.monotonic() < deadline:
        cycle_end = min(deadline, time.monotonic() + control_dt)
        left_hand.write_endeffector(zero)
        right_hand.write_endeffector(zero)
        left_q = np.asarray(
            left_hand.read_current_endeffector_q(),
            dtype=np.float64,
        )
        right_q = np.asarray(
            right_hand.read_current_endeffector_q(),
            dtype=np.float64,
        )
        if left_q.shape != (21,) or right_q.shape != (21,):
            raise RuntimeError(
                "Revo3 feedback must contain 21 joints per hand during reset."
            )
        last_error = float(
            max(np.max(np.abs(left_q)), np.max(np.abs(right_q)))
        )
        if last_error <= tolerance_rad:
            log_success(
                "Revo3 zero feedback verified: "
                f"max_abs_err={last_error:.4f}rad."
            )
            return
        precise_wait(cycle_end)
    raise RuntimeError(
        "Revo3 failed to reach zero before reset timeout: "
        f"max_abs_err={last_error:.4f}rad, tolerance={tolerance_rad:.4f}rad."
    )


def _emergency_hold_current(
    arm,
    left_hand,
    right_hand,
    *,
    control_dt: float,
) -> None:
    """Send one best-effort hold command using current hardware feedback."""
    errors: list[str] = []
    try:
        arm_q = np.asarray(arm.read_current_arm_q(), dtype=np.float64)
        if arm_q.shape != (12,) or not np.all(np.isfinite(arm_q)):
            raise ValueError(f"invalid arm feedback shape/value: {arm_q.shape}")
        arm.write_arm(
            q_target=arm_q,
            time_target=time.monotonic() + control_dt,
            cmd_target="schedule_waypoint",
        )
    except Exception as exc:
        errors.append(f"Z1 hold failed: {exc}")

    for name, hand in (("left Revo3", left_hand), ("right Revo3", right_hand)):
        try:
            hand_q = np.asarray(
                hand.read_current_endeffector_q(),
                dtype=np.float64,
            )
            if hand_q.shape != (21,) or not np.all(np.isfinite(hand_q)):
                raise ValueError(
                    f"invalid hand feedback shape/value: {hand_q.shape}"
                )
            hand.write_endeffector(hand_q)
        except Exception as exc:
            errors.append(f"{name} hold failed: {exc}")

    # Z1 write_arm updates the command consumed by its control thread. Leave one
    # control period for that thread to transmit the feedback-based hold target.
    precise_wait(time.monotonic() + control_dt)
    if errors:
        raise RuntimeError("; ".join(errors))
    log_error(
        "Emergency hold command sent from current feedback. "
        "Automatic reset is disabled; hardware emergency stop remains authoritative."
    )


def _disconnect_after_emergency(env) -> None:
    """Disconnect this deployment stack without invoking Z1 go_home."""
    arm, left_hand, right_hand = _get_controllers(env.robot)
    errors: list[str] = []
    disconnect_without_motion = getattr(arm, "disconnect_without_motion", None)
    if disconnect_without_motion is None:
        errors.append(
            "Dual-Z1 controller does not provide disconnect_without_motion()."
        )
    else:
        try:
            disconnect_without_motion()
        except Exception as exc:
            errors.append(f"Dual-Z1 disconnect failed: {exc}")

    for name, hand in (("left Revo3", left_hand), ("right Revo3", right_hand)):
        try:
            if getattr(hand, "is_connected", True):
                hand.disconnect()
        except Exception as exc:
            errors.append(f"{name} disconnect failed: {exc}")
    for name, camera in env.robot.cameras.items():
        try:
            if getattr(camera, "is_connected", True):
                camera.disconnect()
        except Exception as exc:
            errors.append(f"{name} camera disconnect failed: {exc}")
    if errors:
        raise RuntimeError("; ".join(errors))
    log_success("Emergency disconnect completed without Z1 go_home.")


def _run_hardware(
    args: argparse.Namespace,
    contract: Gr00tPolicyContract,
    rtc_contract: RTCServerContract | None,
) -> None:
    from pynput import keyboard

    from unitree_deploy.real_unitree_env import make_real_env

    control_dt = 1.0 / args.control_freq
    command_lookahead_s = args.command_lookahead_pico_frames / 90.0
    live = args.mode == "live"
    profiler = WindowedPerformanceProfiler(args.perf_log_interval)
    env = None
    listener = None
    worker = None
    reset_on_exit = False
    rtc_config: RTCClientConfig | None = None
    delay_estimator: InferenceDelayEstimator | None = None
    if args.chunk_strategy == "rtc":
        if rtc_contract is None:
            raise RuntimeError("RTC strategy requires a validated RTC contract.")
        chunk_buffer = RTCActionQueue(
            action_chunk_horizon=args.action_chunk_horizon,
        )
        rtc_config = RTCClientConfig(
            execution_horizon=args.rtc_execution_horizon,
            max_guidance_weight=args.rtc_max_guidance_weight,
            prefix_attention_schedule=args.rtc_prefix_attention_schedule,
            initial_inference_delay=args.rtc_inference_delay,
            min_inference_delay=args.rtc_min_inference_delay,
            max_inference_delay=args.rtc_max_inference_delay,
        )
        delay_estimator = InferenceDelayEstimator(
            control_freq=args.control_freq,
            initial_steps=args.rtc_inference_delay,
            min_steps=args.rtc_min_inference_delay,
            max_steps=args.rtc_max_inference_delay,
        )
    else:
        chunk_buffer = TemporalActionEnsembler(
            temporal_ensemble_coeff=args.temporal_ensemble_coeff,
            action_chunk_horizon=args.action_chunk_horizon,
            max_chunks=args.max_temporal_chunks,
            max_sequence_lag=args.max_temporal_sequence_lag,
        )
    keyboard_control = HoldToRunControl(automatic=args.auto_start)
    emergency_stop_triggered = False

    try:
        env = make_real_env(ROBOT_TYPE, dt=control_dt)
        env.connect()
        reset_on_exit = True
        arm, left_hand, right_hand = _get_controllers(env.robot)
        _initialize_hardware(
            arm,
            left_hand,
            right_hand,
            control_dt,
            args.init_duration,
        )

        observation_adapter = Gr00tRevo3Z1ObservationAdapter(
            env.robot,
            contract,
            max_capture_span_s=args.max_capture_span,
            max_hand_feedback_age_s=args.max_hand_feedback_age,
        )
        initial_snapshot = observation_adapter.capture()
        inference_origin_arm_q = initial_snapshot.arm_q.copy()
        inference_origin_left_hand_q = initial_snapshot.left_hand_q.copy()
        inference_origin_right_hand_q = initial_snapshot.right_hand_q.copy()
        observation_adapter.reset(initial_snapshot)
        initial_observation = observation_adapter.build(args.instruction)
        _warmup_policy(args, contract, initial_observation)

        action_adapter = Gr00tRevo3Z1ActionAdapter(
            observation_adapter.head_position,
            observation_adapter.head_quat_wxyz,
        )
        safety_limits = SafetyLimits(
            max_eef_translation_step_m=args.max_eef_translation_step,
            max_eef_rotation_step_rad=np.deg2rad(
                args.max_eef_rotation_step_deg
            ),
            max_eef_translation_reject_m=args.max_eef_translation_reject,
            max_eef_rotation_reject_rad=np.deg2rad(
                args.max_eef_rotation_reject_deg
            ),
            max_eef_linear_speed_m_s=args.max_eef_linear_speed,
            max_eef_angular_speed_rad_s=args.max_eef_angular_speed,
            max_arm_tracking_error_rad=args.max_arm_tracking_error,
            max_hand_tracking_error_rad=args.max_hand_tracking_error,
            max_hand_feedback_age_s=args.max_hand_feedback_age,
        )
        safety = Gr00tRevo3Z1Safety(
            arm,
            left_hand,
            right_hand,
            control_dt=control_dt,
            limits=safety_limits,
        )
        ik_q_filter = (
            None
            if args.ik_ema_alpha == 0.0
            else EmaFilter(alpha=args.ik_ema_alpha, data_size=12)
        )
        if ik_q_filter is None:
            log_info("IK joint EMA disabled.")
        else:
            log_info(f"IK joint EMA enabled: alpha={args.ik_ema_alpha:.3f}.")
        translation_jump_limit = safety_limits.max_eef_translation_step_m
        if safety_limits.max_eef_linear_speed_m_s is not None:
            translation_jump_limit = (
                safety_limits.max_eef_linear_speed_m_s * control_dt
            )
        rotation_jump_limit = safety_limits.max_eef_rotation_step_rad
        if safety_limits.max_eef_angular_speed_rad_s is not None:
            rotation_jump_limit = (
                safety_limits.max_eef_angular_speed_rad_s * control_dt
            )
        log_info(
            "EEF safety limits: "
            f"translation_shape={translation_jump_limit:.3f}m, "
            f"translation_reject={safety_limits.max_eef_translation_reject_m:.3f}m, "
            f"rotation_shape={np.rad2deg(rotation_jump_limit):.1f}deg, "
            "rotation_reject="
            f"{np.rad2deg(safety_limits.max_eef_rotation_reject_rad):.1f}deg."
        )
        safety.reset(
            initial_snapshot.arm_q,
            initial_snapshot.left_hand_q,
            initial_snapshot.right_hand_q,
        )
        _reset_ik_q_filter(ik_q_filter, initial_snapshot.arm_q)
        worker = InferenceWorker(
            host=args.policy_host,
            port=args.policy_port,
            timeout_ms=args.timeout_ms,
            api_token=args.api_token,
            contract=contract,
            chunk_buffer=chunk_buffer,
            profiler=profiler,
            delay_estimator=delay_estimator,
        )
        listener = keyboard.Listener(
            on_press=keyboard_control.on_press,
            on_release=keyboard_control.on_release,
        )
        listener.start()
        mode_text = "LIVE CONTROL" if live else "SHADOW MODE"
        if args.auto_start:
            log_warning(
                f"{mode_text} AUTOMATIC: after a {args.auto_start_delay:.1f}s "
                "countdown, inference and action execution continue until stopped."
            )
            log_warning(
                "EMERGENCY STOP keys: Space, Esc, or x. "
                "Emergency stop is latched and does not run go_start. "
                "Press r to return to the inference origin and automatically "
                "start the next rollout; press e for normal safe exit."
            )
        else:
            log_info(
                f"{mode_text}: hold 'c' to run a rollout, release 'c' to pause, "
                "press 'r' to return to the inference origin, "
                "press Space/Esc/x for emergency stop, or press 'e' to exit."
            )
        if rtc_config is None:
            log_info(
                "Temporal ensemble action selection enabled: "
                f"inference_interval_steps={args.inference_interval_steps}, "
                f"action_chunk_horizon={args.action_chunk_horizon}, "
                f"coefficient={args.temporal_ensemble_coeff:.4f}, "
                f"max_chunks={args.max_temporal_chunks}, "
                f"max_sequence_lag={args.max_temporal_sequence_lag}."
            )
        else:
            log_info(
                "Server-guided RTC action selection enabled: "
                f"inference_interval_steps={args.inference_interval_steps}, "
                f"action_chunk_horizon={args.action_chunk_horizon}, "
                f"execution_horizon={rtc_config.execution_horizon}, "
                f"inference_delay={rtc_config.initial_inference_delay}"
                f"[{rtc_config.min_inference_delay},"
                f"{rtc_config.max_inference_delay}], "
                f"max_guidance_weight={rtc_config.max_guidance_weight:.3f}, "
                f"schedule={rtc_config.prefix_attention_schedule}."
            )
        log_info(
            f"Stage performance profiling enabled: "
            f"report_interval={args.perf_log_interval:.1f}s."
        )
        log_info(
            "Versioned asynchronous dual-Z1 IK enabled; the control loop "
            "does not block while Mink computes the current target."
        )
        if args.auto_start and not _wait_for_automatic_start(
            keyboard_control,
            args.auto_start_delay,
        ):
            emergency_stop_triggered = (
                keyboard_control.emergency_stop_requested()
            )

        rollout_id = 0
        sequence_id = 0
        control_step = 0
        pending_ik_targets: dict[int, CartesianActionTarget] = {}
        initial_ik_result = arm.latest_arm_ik_result()
        last_polled_ik_step = (
            0 if initial_ik_result is None else initial_ik_result.step_index
        )
        fallback_action = _snapshot_fallback_action(initial_snapshot)
        have_policy_action = False
        was_enabled = False
        last_event_log = 0.0
        last_shadow_log = 0.0
        last_live_timing_log = 0.0
        while True:
            cycle_started_at = time.monotonic()
            cycle_end = cycle_started_at + control_dt
            (
                enabled,
                resume_requested,
                exit_requested,
                emergency_stop_requested,
            ) = keyboard_control.snapshot()
            if emergency_stop_requested:
                emergency_stop_triggered = True
                chunk_buffer.clear()
                log_error(
                    "Emergency stop accepted. Rejecting all pending and future "
                    "policy actions."
                )
                break
            if exit_requested:
                break
            if keyboard_control.consume_origin_reset_request():
                chunk_buffer.clear()
                pending_ik_targets.clear()
                have_policy_action = False
                was_enabled = False
                log_info(
                    "Returning to the captured inference origin without "
                    "closing the deployment process."
                )
                try:
                    _return_to_inference_origin(
                        arm,
                        left_hand,
                        right_hand,
                        arm_q=inference_origin_arm_q,
                        left_hand_q=inference_origin_left_hand_q,
                        right_hand_q=inference_origin_right_hand_q,
                        control_dt=control_dt,
                        duration_s=args.rollout_reset_duration,
                        stop_requested=keyboard_control.stop_requested,
                    )
                except InterruptedError:
                    if keyboard_control.emergency_stop_requested():
                        emergency_stop_triggered = True
                        log_error(
                            "Emergency stop interrupted inference-origin reset."
                        )
                    break
                except Exception as exc:
                    log_error(
                        "Inference-origin reset failed; ending this deployment "
                        f"for safety: {exc}"
                    )
                    break

                reset_snapshot = observation_adapter.capture()
                observation_adapter.reset(reset_snapshot)
                fallback_action = _snapshot_fallback_action(reset_snapshot)
                safety.reset(
                    reset_snapshot.arm_q,
                    reset_snapshot.left_hand_q,
                    reset_snapshot.right_hand_q,
                )
                _reset_ik_q_filter(ik_q_filter, reset_snapshot.arm_q)
                latest_ik_result = arm.latest_arm_ik_result()
                if latest_ik_result is not None:
                    last_polled_ik_step = latest_ik_result.step_index
                profiler.reset()
                if args.auto_start:
                    if not keyboard_control.arm_automatic_rollout():
                        break
                    log_success(
                        "Inference origin restored. Automatic mode will start "
                        "a fresh rollout."
                    )
                else:
                    log_success(
                        "Inference origin restored. Press and hold 'c' to "
                        "start the next rollout."
                    )
                continue

            capture_started_at = time.monotonic()
            try:
                snapshot = observation_adapter.capture()
                _record_capture_timing(profiler, observation_adapter)
            except Exception as exc:
                profiler.record(
                    "capture_total",
                    time.monotonic() - capture_started_at,
                )
                profiler.increment("observation_failures")
                chunk_buffer.clear()
                pending_ik_targets.clear()
                have_policy_action = False
                now = time.monotonic()
                if now - last_event_log >= 1.0:
                    log_warning(
                        f"Observation unavailable; holding last command: {exc}"
                    )
                    last_event_log = now
                _finish_profiled_cycle(
                    profiler,
                    cycle_started_at=cycle_started_at,
                    cycle_end=cycle_end,
                    mode=args.mode,
                    control_dt=control_dt,
                    report_enabled=enabled,
                )
                continue

            if resume_requested:
                rollout_id += 1
                sequence_id = 0
                control_step = 0
                chunk_buffer.clear()
                if delay_estimator is not None:
                    delay_estimator.reset()
                profiler.reset()
                observation_adapter.reset(snapshot)
                fallback_action = _snapshot_fallback_action(snapshot)
                have_policy_action = False
                pending_ik_targets.clear()
                latest_ik_result = arm.latest_arm_ik_result()
                if latest_ik_result is not None:
                    last_polled_ik_step = latest_ik_result.step_index
                # Synchronize only at rollout boundaries. During execution the
                # Mink process must keep its previous solution as the next seed.
                arm.z1_mink.set_dual_arm_q(snapshot.arm_q)
                safety.reset(
                    snapshot.arm_q,
                    snapshot.left_hand_q,
                    snapshot.right_hand_q,
                )
                _reset_ik_q_filter(ik_q_filter, snapshot.arm_q)
                log_success(f"Started fresh GR00T rollout {rollout_id}.")
            elif enabled:
                observation_adapter.append(snapshot)

            if not enabled:
                if was_enabled:
                    chunk_buffer.clear()
                    if delay_estimator is not None:
                        delay_estimator.reset()
                    safety.reset(
                        snapshot.arm_q,
                        snapshot.left_hand_q,
                        snapshot.right_hand_q,
                    )
                    _reset_ik_q_filter(ik_q_filter, snapshot.arm_q)
                    pending_ik_targets.clear()
                    latest_ik_result = arm.latest_arm_ik_result()
                    if latest_ik_result is not None:
                        last_polled_ik_step = latest_ik_result.step_index
                was_enabled = False
                _finish_profiled_cycle(
                    profiler,
                    cycle_started_at=cycle_started_at,
                    cycle_end=cycle_end,
                    mode=args.mode,
                    control_dt=control_dt,
                    report_enabled=False,
                )
                continue

            was_enabled = True
            if control_step % args.inference_interval_steps == 0:
                build_started_at = time.monotonic()
                observation = observation_adapter.build(args.instruction)
                profiler.record(
                    "observation_build",
                    time.monotonic() - build_started_at,
                )
                request_options = None
                if rtc_config is not None:
                    inference_delay = delay_estimator.estimate_steps()
                    prefix = chunk_buffer.build_prefix(
                        rollout_id=rollout_id,
                        target_step=control_step,
                        now=time.monotonic(),
                        max_chunk_age_s=args.max_chunk_age,
                        execution_horizon=rtc_config.execution_horizon,
                    )
                    request_options = rtc_config.build_options(
                        rollout_id=rollout_id,
                        sequence_id=sequence_id,
                        target_step=control_step,
                        inference_delay=inference_delay,
                        prefix=prefix,
                    )
                    profiler.increment(
                        "rtc_requests_with_prefix"
                        if prefix is not None
                        else "rtc_requests_without_prefix"
                    )
                worker.submit(
                    InferenceRequest(
                        rollout_id=rollout_id,
                        sequence_id=sequence_id,
                        target_step=control_step,
                        observation_at=snapshot.captured_at,
                        submitted_at=time.monotonic(),
                        observation=observation,
                        options=request_options,
                    )
                )
                sequence_id += 1

            for event in worker.drain_events():
                now = time.monotonic()
                if now - last_event_log >= 1.0:
                    log_warning(event)
                    last_event_log = now

            if not live:
                latest = chunk_buffer.latest()
                now = time.monotonic()
                if latest is not None and now - last_shadow_log >= 1.0:
                    latency = latest.received_at - latest.submitted_at
                    age = now - latest.observation_at
                    hand_diagnostic = _shadow_hand_order_diagnostic(
                        latest.actions,
                        snapshot.left_hand_q,
                        snapshot.right_hand_q,
                        args.shadow_hand_motion_threshold,
                    )
                    rtc_diagnostic = ""
                    if rtc_config is not None:
                        metadata = latest.metadata or {}
                        rtc_diagnostic = (
                            f", rtc_applied={metadata.get('rtc_applied')}, "
                            f"rtc_reason={metadata.get('rtc_reason')}, "
                            f"rtc_inference_delay="
                            f"{metadata.get('requested_inference_delay')}, "
                            f"rtc_prefix_horizon="
                            f"{metadata.get('prefix_horizon')}"
                        )
                    log_info(
                        "Shadow prediction ready: "
                        f"sequence={latest.sequence_id}, "
                        f"target_step={latest.target_step}, "
                        f"aligned_step={control_step - latest.target_step}, "
                        f"latency={latency * 1000.0:.1f}ms, "
                        f"observation_age={age * 1000.0:.1f}ms, "
                        f"{hand_diagnostic}{rtc_diagnostic}."
                    )
                    last_shadow_log = now
                _finish_profiled_cycle(
                    profiler,
                    cycle_started_at=cycle_started_at,
                    cycle_end=cycle_end,
                    mode=args.mode,
                    control_dt=control_dt,
                    report_enabled=True,
                )
                control_step += 1
                continue

            temporal_started_at = time.monotonic()
            action_selection = chunk_buffer.get_action(
                rollout_id=rollout_id,
                control_step=control_step,
                now=time.monotonic(),
                max_chunk_age_s=args.max_chunk_age,
                fallback_actions=fallback_action,
            )
            profiler.record(
                "temporal_select",
                time.monotonic() - temporal_started_at,
            )
            if action_selection.candidate_count == 0:
                profiler.increment("no_action_candidates")
            now = time.monotonic()
            latest = chunk_buffer.latest()
            if latest is not None:
                profiler.record(
                    "action_observation_age",
                    max(0.0, now - latest.observation_at),
                )
                profiler.record(
                    "action_result_age",
                    max(0.0, now - latest.received_at),
                )
                profiler.record(
                    "action_alignment_lag",
                    max(0, control_step - latest.target_step) * control_dt,
                )
            if live and now - last_live_timing_log >= 1.0:
                if latest is not None:
                    strategy_name = (
                        "RTC" if rtc_config is not None else "Temporal"
                    )
                    rtc_timing = ""
                    if rtc_config is not None:
                        metadata = latest.metadata or {}
                        rtc_timing = (
                            f", rtc_applied={metadata.get('rtc_applied')}, "
                            f"rtc_inference_delay="
                            f"{metadata.get('requested_inference_delay')}, "
                            f"rtc_prefix_horizon="
                            f"{metadata.get('prefix_horizon')}, "
                            f"next_delay_estimate="
                            f"{delay_estimator.estimate_steps()}"
                        )
                    log_info(
                        f"{strategy_name} control timing: "
                        f"control_step={control_step}, "
                        f"candidates={action_selection.candidate_count}, "
                        f"selected_sequences={action_selection.sequence_ids}, "
                        f"newest_weight="
                        f"{action_selection.weights[-1] if action_selection.weights else 0.0:.3f}, "
                        f"latest_submitted_sequence={sequence_id - 1}, "
                        f"latest_sequence={latest.sequence_id}, "
                        f"inflight_sequence_gap="
                        f"{max(0, sequence_id - 1 - latest.sequence_id)}, "
                        f"latest_aligned_step={control_step - latest.target_step}, "
                        f"inference_latency="
                        f"{(latest.received_at - latest.submitted_at) * 1000.0:.1f}ms, "
                        f"observation_age="
                        f"{(now - latest.observation_at) * 1000.0:.1f}ms"
                        f"{rtc_timing}."
                    )
                    last_live_timing_log = now
            if action_selection.candidate_count > 0:
                fallback_action = {
                    key: value.copy()
                    for key, value in action_selection.actions.items()
                }
                have_policy_action = True
            if not have_policy_action:
                safety.hold_last_target()
                _finish_profiled_cycle(
                    profiler,
                    cycle_started_at=cycle_started_at,
                    cycle_end=cycle_end,
                    mode=args.mode,
                    control_dt=control_dt,
                    report_enabled=True,
                )
                control_step += 1
                continue

            try:
                now = time.monotonic()
                stage_started_at = time.monotonic()
                safety.validate_feedback_freshness(now)
                safety.validate_tracking(
                    snapshot.arm_q,
                    snapshot.left_hand_q,
                    snapshot.right_hand_q,
                )
                profiler.record(
                    "feedback_safety",
                    time.monotonic() - stage_started_at,
                )
                stage_started_at = time.monotonic()
                cartesian = action_adapter.decode_step(
                    action_selection.actions,
                    0,
                )
                profiler.record(
                    "action_decode",
                    time.monotonic() - stage_started_at,
                )
                stage_started_at = time.monotonic()
                cartesian = safety.shape_cartesian(
                    cartesian,
                    snapshot.arm_q,
                    current_wrist_poses=(
                        (
                            snapshot.left_wrist_position,
                            snapshot.left_wrist_quat_wxyz,
                        ),
                        (
                            snapshot.right_wrist_position,
                            snapshot.right_wrist_quat_wxyz,
                        ),
                    ),
                )
                profiler.record(
                    "cartesian_safety",
                    time.monotonic() - stage_started_at,
                )
                ik_started_at = time.monotonic()
                submitted_ik_version = action_adapter.submit_ik(
                    arm,
                    cartesian,
                )
                pending_ik_targets[submitted_ik_version] = cartesian
                if len(pending_ik_targets) > 64:
                    oldest_version = min(pending_ik_targets)
                    pending_ik_targets.pop(oldest_version, None)

                ik_result = arm.latest_arm_ik_result()
                profiler.record(
                    "dual_arm_ik",
                    time.monotonic() - ik_started_at,
                )
                if (
                    ik_result is None
                    or ik_result.step_index <= last_polled_ik_step
                ):
                    profiler.increment("ik_results_pending")
                    _finish_profiled_cycle(
                        profiler,
                        cycle_started_at=cycle_started_at,
                        cycle_end=cycle_end,
                        mode=args.mode,
                        control_dt=control_dt,
                        report_enabled=True,
                    )
                    control_step += 1
                    continue

                last_polled_ik_step = ik_result.step_index
                completed_ik_version = ik_result.target_version
                profiler.record(
                    "ik_target_lag",
                    max(0, submitted_ik_version - completed_ik_version)
                    * control_dt,
                )
                completed_cartesian = pending_ik_targets.get(
                    completed_ik_version
                )
                for version in tuple(pending_ik_targets):
                    if version < completed_ik_version:
                        pending_ik_targets.pop(version, None)

                if completed_cartesian is None:
                    profiler.increment("ik_results_unmatched")
                    _finish_profiled_cycle(
                        profiler,
                        cycle_started_at=cycle_started_at,
                        cycle_end=cycle_end,
                        mode=args.mode,
                        control_dt=control_dt,
                        report_enabled=True,
                    )
                    control_step += 1
                    continue

                try:
                    target = action_adapter.target_from_ik_result(
                        arm,
                        completed_cartesian,
                        ik_result,
                    )
                except RuntimeError as exc:
                    profiler.increment("ik_results_unachieved")
                    now = time.monotonic()
                    if now - last_event_log >= 1.0:
                        log_warning(
                            "Dual-Z1 IK result did not achieve its versioned "
                            f"target; holding last command: {exc}"
                        )
                        last_event_log = now
                    _finish_profiled_cycle(
                        profiler,
                        cycle_started_at=cycle_started_at,
                        cycle_end=cycle_end,
                        mode=args.mode,
                        control_dt=control_dt,
                        report_enabled=True,
                    )
                    control_step += 1
                    continue
                pending_ik_targets.pop(completed_ik_version, None)
                stage_started_at = time.monotonic()
                target = _filter_ik_target(ik_q_filter, target)
                profiler.record(
                    "ik_ema",
                    time.monotonic() - stage_started_at,
                )
                stage_started_at = time.monotonic()
                target = safety.shape_target(target)
                profiler.record(
                    "target_shape",
                    time.monotonic() - stage_started_at,
                )
                stage_started_at = time.monotonic()
                if keyboard_control.emergency_stop_requested():
                    emergency_stop_triggered = True
                    chunk_buffer.clear()
                    break
                left_sent, right_sent = _write_target(
                    arm,
                    left_hand,
                    right_hand,
                    target,
                    cycle_end + command_lookahead_s,
                )
                safety.commit_sent_hand_targets(left_sent, right_sent)
                profiler.record(
                    "command_write",
                    time.monotonic() - stage_started_at,
                )
            except TimeoutError as exc:
                profiler.increment("ik_timeouts")
                # Keep the last transmitted target and velocity state. Resetting
                # velocity here creates a stop-and-reaccelerate jerk next cycle.
                now = time.monotonic()
                if now - last_event_log >= 1.0:
                    log_warning(f"Dual-Z1 IK skipped for this cycle: {exc}")
                    last_event_log = now
            except (SafetyViolation, RuntimeError, ValueError) as exc:
                profiler.increment("safety_rejections")
                chunk_buffer.clear()
                pending_ik_targets.clear()
                fallback_action = _snapshot_fallback_action(snapshot)
                have_policy_action = False
                safety.reset(
                    snapshot.arm_q,
                    snapshot.left_hand_q,
                    snapshot.right_hand_q,
                )
                _reset_ik_q_filter(ik_q_filter, snapshot.arm_q)
                now = time.monotonic()
                if now - last_event_log >= 1.0:
                    log_warning(
                        f"Policy target rejected; holding last command: {exc}"
                    )
                    last_event_log = now

            _finish_profiled_cycle(
                profiler,
                cycle_started_at=cycle_started_at,
                cycle_end=cycle_end,
                mode=args.mode,
                control_dt=control_dt,
                report_enabled=True,
            )
            control_step += 1

    except KeyboardInterrupt:
        log_warning(
            "Ctrl+C detected; stopping policy execution and starting safe reset."
        )
    finally:
        if listener is not None:
            listener.stop()
        if emergency_stop_triggered and env is not None and reset_on_exit:
            try:
                arm, left_hand, right_hand = _get_controllers(env.robot)
                _emergency_hold_current(
                    arm,
                    left_hand,
                    right_hand,
                    control_dt=control_dt,
                )
            except Exception as exc:
                log_error(
                    f"Emergency software hold was incomplete: {exc}. "
                    "Use the hardware emergency stop."
                )
        if worker is not None:
            # Inference never touches hardware. Do not delay a hardware reset
            # while waiting for a blocked network request to finish.
            worker.close(join_timeout_s=0.1)
        if env is not None:
            if reset_on_exit and not emergency_stop_triggered:
                try:
                    arm, left_hand, right_hand = _get_controllers(env.robot)
                    _safe_exit_reset(
                        arm,
                        left_hand,
                        right_hand,
                        control_dt,
                        args.init_duration,
                    )
                except Exception as exc:
                    log_error(f"Safe exit reset failed: {exc}")
            elif emergency_stop_triggered:
                log_warning(
                    "Emergency stop remains latched. Skipping go_start and hand-zero "
                    "reset to avoid any automatic post-stop motion."
                )
            if emergency_stop_triggered:
                try:
                    _disconnect_after_emergency(env)
                except Exception as exc:
                    log_error(
                        f"Emergency no-motion disconnect was incomplete: {exc}. "
                        "Use the hardware emergency stop."
                    )
            else:
                env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remote GR00T deployment for dual Z1 and dual Revo3. "
            "The default server-check mode never connects hardware."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("server-check", "shadow", "live"),
        default="server-check",
    )
    parser.add_argument("--policy-host", required=True)
    parser.add_argument("--policy-port", type=int, default=5555)
    parser.add_argument("--timeout-ms", type=int, default=3000)
    parser.add_argument("--api-token", default=None)
    parser.add_argument(
        "--instruction",
        default="pick up the lemons into the basket",
    )
    parser.add_argument("--control-freq", type=float, default=30.0)
    parser.add_argument(
        "--chunk-strategy",
        choices=("temporal", "rtc"),
        default="temporal",
        help=(
            "Use the existing client-side temporal ensemble or server-guided "
            "real-time chunking. The default preserves the existing behavior."
        ),
    )
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=None,
        help=(
            "Deprecated compatibility alias for --action-chunk-horizon. "
            "Temporal ensemble is used by default."
        ),
    )
    parser.add_argument("--inference-interval-steps", type=int, default=4)
    parser.add_argument("--action-chunk-horizon", type=int, default=None)
    parser.add_argument("--temporal-ensemble-coeff", type=float, default=0.5)
    parser.add_argument("--max-temporal-chunks", type=int, default=16)
    parser.add_argument(
        "--max-temporal-sequence-lag",
        type=int,
        default=2,
        help=(
            "Maximum older sequence gap allowed in temporal ensemble. "
            "The default 2 blends at most the latest three aligned chunks."
        ),
    )
    parser.add_argument("--max-chunk-age", type=float, default=0.30)
    parser.add_argument("--rtc-inference-delay", type=int, default=4)
    parser.add_argument("--rtc-min-inference-delay", type=int, default=3)
    parser.add_argument("--rtc-max-inference-delay", type=int, default=6)
    parser.add_argument("--rtc-execution-horizon", type=int, default=8)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=5.0)
    parser.add_argument(
        "--rtc-prefix-attention-schedule",
        choices=("zeros", "ones", "linear", "exp"),
        default="exp",
    )
    parser.add_argument("--perf-log-interval", type=float, default=2.0)
    parser.add_argument(
        "--ik-timeout",
        type=float,
        default=0.20,
        help=(
            "Compatibility option for synchronous IK callers. Live GR00T "
            "control uses versioned asynchronous IK and does not block on it."
        ),
    )
    parser.add_argument(
        "--ik-ema-alpha",
        type=float,
        default=0.20,
        help=(
            "EMA coefficient applied to successful 12D IK joint targets before "
            "joint velocity/acceleration shaping. Set to 0 to disable."
        ),
    )
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--init-duration", type=float, default=2.0)
    parser.add_argument(
        "--rollout-reset-duration",
        type=float,
        default=2.0,
        help=(
            "Seconds used to return Z1 and Revo3 to the captured inference "
            "origin after the operator presses 'r'."
        ),
    )
    parser.add_argument("--max-capture-span", type=float, default=0.10)
    parser.add_argument("--max-hand-feedback-age", type=float, default=0.20)
    parser.add_argument(
        "--shadow-hand-motion-threshold",
        type=float,
        default=0.20,
        help=(
            "Flexion-joint delta in radians used to report which hand moves "
            "first in each shadow action chunk."
        ),
    )
    parser.add_argument(
        "--max-eef-translation-step",
        type=float,
        default=0.05,
        help="Soft translation shaping distance from current EEF feedback.",
    )
    parser.add_argument(
        "--max-eef-rotation-step-deg",
        type=float,
        default=45.0,
        help="Soft rotation shaping angle from current EEF feedback.",
    )
    parser.add_argument(
        "--max-eef-translation-reject",
        type=float,
        default=0.15,
        help="Hard translation anomaly threshold that rejects a policy target.",
    )
    parser.add_argument(
        "--max-eef-rotation-reject-deg",
        type=float,
        default=90.0,
        help="Hard rotation anomaly threshold that rejects a policy target.",
    )
    parser.add_argument(
        "--max-eef-linear-speed",
        type=float,
        default=None,
        help=(
            "Legacy compatibility override. When set, translation shaping is "
            "limited to this speed multiplied by the control period."
        ),
    )
    parser.add_argument(
        "--max-eef-angular-speed",
        type=float,
        default=None,
        help=(
            "Legacy compatibility override. When set, rotation shaping is "
            "limited to this speed multiplied by the control period."
        ),
    )
    parser.add_argument("--max-arm-tracking-error", type=float, default=0.35)
    parser.add_argument("--max-hand-tracking-error", type=float, default=0.70)
    parser.add_argument("--command-lookahead-pico-frames", type=int, default=2)
    parser.add_argument("--confirm-hardware-access", default="")
    parser.add_argument("--confirm-live-control", default="")
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help=(
            "Automatically start one continuous live rollout after initialization. "
            "Requires the independent --confirm-auto-control token."
        ),
    )
    parser.add_argument(
        "--auto-start-delay",
        type=float,
        default=5.0,
        help="Countdown in seconds before automatic live execution starts.",
    )
    parser.add_argument("--confirm-auto-control", default="")
    args = parser.parse_args()

    positive_names = (
        "policy_port",
        "timeout_ms",
        "control_freq",
        "inference_interval_steps",
        "max_temporal_chunks",
        "max_chunk_age",
        "rtc_execution_horizon",
        "rtc_max_guidance_weight",
        "perf_log_interval",
        "ik_timeout",
        "init_duration",
        "rollout_reset_duration",
        "max_capture_span",
        "max_hand_feedback_age",
        "shadow_hand_motion_threshold",
        "max_eef_translation_step",
        "max_eef_rotation_step_deg",
        "max_eef_translation_reject",
        "max_eef_rotation_reject_deg",
        "max_arm_tracking_error",
        "max_hand_tracking_error",
    )
    for name in positive_names:
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    for name in ("max_eef_linear_speed", "max_eef_angular_speed"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    translation_shape_limit = args.max_eef_translation_step
    if args.max_eef_linear_speed is not None:
        translation_shape_limit = (
            args.max_eef_linear_speed / args.control_freq
        )
    if args.max_eef_translation_reject < translation_shape_limit:
        parser.error(
            "--max-eef-translation-reject must be greater than or equal to "
            "the effective translation shaping limit."
        )
    rotation_shape_limit_deg = args.max_eef_rotation_step_deg
    if args.max_eef_angular_speed is not None:
        rotation_shape_limit_deg = np.rad2deg(
            args.max_eef_angular_speed / args.control_freq
        )
    if args.max_eef_rotation_reject_deg < rotation_shape_limit_deg:
        parser.error(
            "--max-eef-rotation-reject-deg must be greater than or equal to "
            "the effective rotation shaping limit."
        )
    for name in ("execution_horizon", "action_chunk_horizon"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if (
        not np.isfinite(args.temporal_ensemble_coeff)
        or args.temporal_ensemble_coeff < 0
    ):
        parser.error("--temporal-ensemble-coeff must be finite and non-negative.")
    if args.warmup_requests < 0:
        parser.error("--warmup-requests must be non-negative.")
    if args.max_temporal_sequence_lag < 0:
        parser.error("--max-temporal-sequence-lag must be non-negative.")
    if args.rtc_min_inference_delay < 0:
        parser.error("--rtc-min-inference-delay must be non-negative.")
    if args.rtc_max_inference_delay < args.rtc_min_inference_delay:
        parser.error(
            "--rtc-max-inference-delay must be greater than or equal to "
            "--rtc-min-inference-delay."
        )
    if not (
        args.rtc_min_inference_delay
        <= args.rtc_inference_delay
        <= args.rtc_max_inference_delay
    ):
        parser.error(
            "--rtc-inference-delay must be inside the configured RTC delay range."
        )
    if args.rtc_max_inference_delay >= args.rtc_execution_horizon:
        parser.error(
            "--rtc-max-inference-delay must be smaller than "
            "--rtc-execution-horizon."
        )
    if (
        not np.isfinite(args.ik_ema_alpha)
        or not 0.0 <= args.ik_ema_alpha <= 1.0
    ):
        parser.error("--ik-ema-alpha must be finite and in [0, 1].")
    if args.command_lookahead_pico_frames not in (0, 1, 2, 3):
        parser.error("--command-lookahead-pico-frames must be 0, 1, 2, or 3.")
    if not np.isfinite(args.auto_start_delay) or args.auto_start_delay < 0:
        parser.error("--auto-start-delay must be finite and non-negative.")
    if not args.instruction.strip():
        parser.error("--instruction must not be empty.")
    return args


def main() -> int:
    args = parse_args()
    contract, rtc_contract = _server_preflight(args)
    if args.action_chunk_horizon is None:
        args.action_chunk_horizon = (
            args.execution_horizon
            if args.execution_horizon is not None
            else contract.action_horizon
        )
    elif args.execution_horizon is not None:
        log_warning(
            "--execution-horizon is deprecated and ignored because "
            "--action-chunk-horizon was provided."
        )
    if args.action_chunk_horizon > contract.action_horizon:
        raise ValueError(
            f"action_chunk_horizon={args.action_chunk_horizon} exceeds server action "
            f"horizon={contract.action_horizon}."
        )
    if args.inference_interval_steps > args.action_chunk_horizon:
        raise ValueError(
            "inference_interval_steps must not exceed action_chunk_horizon: "
            f"{args.inference_interval_steps} > {args.action_chunk_horizon}."
        )
    if args.chunk_strategy == "rtc":
        if rtc_contract is None:
            raise RuntimeError("RTC server contract was not validated.")
        if args.rtc_execution_horizon > args.action_chunk_horizon:
            raise ValueError(
                "rtc_execution_horizon must not exceed action_chunk_horizon: "
                f"{args.rtc_execution_horizon} > {args.action_chunk_horizon}."
            )
        if (
            args.rtc_prefix_attention_schedule
            not in rtc_contract.supported_schedules
        ):
            raise ValueError(
                "RTC prefix schedule is not supported by the server: "
                f"{args.rtc_prefix_attention_schedule!r} not in "
                f"{rtc_contract.supported_schedules}."
            )
    if args.mode == "server-check":
        log_success("Server check complete. No hardware was connected.")
        return 0
    if args.mode == "shadow" and args.confirm_hardware_access != SHADOW_TOKEN:
        raise PermissionError(
            f"Shadow mode connects and initializes real hardware. Pass "
            f"--confirm-hardware-access {SHADOW_TOKEN} to continue."
        )
    if args.mode == "live" and args.confirm_live_control != LIVE_TOKEN:
        raise PermissionError(
            f"Live mode controls real hardware. Pass "
            f"--confirm-live-control {LIVE_TOKEN} to continue."
        )
    if args.auto_start:
        if args.mode != "live":
            raise ValueError("--auto-start is only supported with --mode live.")
        if args.confirm_auto_control != AUTO_TOKEN:
            raise PermissionError(
                "Automatic live control requires an independent confirmation. "
                f"Pass --confirm-auto-control {AUTO_TOKEN} to continue."
            )
    _run_hardware(args, contract, rtc_contract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())