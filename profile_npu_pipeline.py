"""Measure Being-H0.5 inference latency on Ascend NPU.

This profiler produces two complementary measurements:

1. End-to-end policy latency with only request-boundary synchronization.
2. Semantic stage latency using either synchronized wall timing or low-intrusion
   device events.

The legacy synchronized mode intentionally serializes asynchronous NPU work.
The event mode records device events around stages and synchronizes once at the
request boundary, substantially reducing profiler-induced serialization. Raw
samples, summaries, an operator table, and a Chrome trace are written to the
output directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
try:
    import torch_npu
except ImportError:
    torch_npu = None

import BeingH.model.beingvla as beingvla_module
from BeingH.inference.beingh_policy import BeingHPolicy


ACTIVE_DEVICE_TYPE = ""

STAGE_TIMING_SYNC = "sync"
STAGE_TIMING_EVENT = "event"
STAGE_TIMING_MODES = (STAGE_TIMING_SYNC, STAGE_TIMING_EVENT)

# These stages are dominated by host work. Device events around them would
# intentionally omit tokenizer/Python time, so event mode keeps wall time as
# their primary stage metric and reports device time separately when present.
HOST_PRIMARY_STAGES = frozenset({"policy_pack_inputs"})


def synchronize() -> None:
    if ACTIVE_DEVICE_TYPE == "cuda":
        torch.cuda.synchronize()
    elif ACTIVE_DEVICE_TYPE == "npu":
        torch.npu.synchronize()


def create_timing_event() -> Any:
    """Create a backend timing event without synchronizing the device."""
    if ACTIVE_DEVICE_TYPE == "cuda":
        return torch.cuda.Event(enable_timing=True)
    if ACTIVE_DEVICE_TYPE == "npu":
        return torch.npu.Event(enable_timing=True)
    raise RuntimeError(
        f"Device-event stage timing requires CUDA or NPU, got {ACTIVE_DEVICE_TYPE!r}"
    )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def describe(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# FINE-GRAINED DENOISE INSTRUMENTATION (added 2026-07-31) -- BEGIN
#
# Stage names emitted for the two denoise loop bodies.  Both are *nested*
# timers: `DENOISE_OUTER_STAGE` contains `DENOISE_INNER_STAGE`, which in turn
# contains action_encoder / qwen / action_decoder / mpg_* .  Nested totals
# overlap and must never be summed with their children.
DENOISE_OUTER_STAGE = "denoise_outer_iteration"   # 1 sample per MPG iteration
DENOISE_INNER_STAGE = "denoise_inner_flow_step"   # 1 sample per flow timestep

# Explicit nesting map, emitted into the JSON so the reader never has to guess.
STAGE_NESTING = {
    "instrumented_policy_total_ms": [
        "policy_pack_inputs",
        "model_get_action_total",
    ],
    "model_get_action_total": [
        "vision_pipeline",
        "proprioception_encoder",
        "gpu_sparse_attention_mask",
        "gpu_block_attention_mask",
        "npu_dense_attention_mask",
        "qwen_prefix_prefill",
        DENOISE_OUTER_STAGE,
    ],
    "vision_pipeline": ["vision_backbone", "vision_connector"],
    DENOISE_OUTER_STAGE: [
        DENOISE_INNER_STAGE,
        "action_encoder@outside_flow_step",
        "baseline_flow_graph_replay",
    ],
    DENOISE_INNER_STAGE: [
        "action_encoder@in_flow_step",
        "qwen_mot_forward",
        "qwen_action_suffix_forward",
        "mpg_action_to_vlm_projection",
        "mpg_enhancement",
        "mpg_vlm_to_action_projection",
        "action_decoder",
    ],
}


def iteration_prefix(mpg_iteration: int) -> str:
    """`0` -> baseline, `1` -> mpg_refine, `n>1` -> mpg_refine<n>."""
    if mpg_iteration == 0:
        return "baseline"
    if mpg_iteration == 1:
        return "mpg_refine"
    return f"mpg_refine{mpg_iteration}"


def position_label(mpg_iteration: int | None, flow_step: int | None) -> str:
    """Human-readable label for a call's position in the denoise schedule."""
    if mpg_iteration is None:
        return "outside_denoise"
    prefix = iteration_prefix(mpg_iteration)
    if flow_step is None:
        return f"{prefix}_iteration_tail"
    return f"{prefix}_step{flow_step}"


# FINE-GRAINED DENOISE INSTRUMENTATION -- END


class StageRecorder:
    def __init__(self, timing_mode: str = STAGE_TIMING_SYNC) -> None:
        if timing_mode not in STAGE_TIMING_MODES:
            raise ValueError(
                f"stage timing mode must be one of {STAGE_TIMING_MODES}, "
                f"got {timing_mode!r}"
            )
        self.timing_mode = timing_mode
        self.enabled = False
        self.measure_model_boundary = False
        self.current: dict[str, list[float]] = defaultdict(list)
        self.current_host_launch: dict[str, list[float]] = defaultdict(list)
        self.current_device: dict[str, list[float]] = defaultdict(list)
        self.model_boundary_samples: list[float] = []
        # --- fine-grained additions ---
        # Ordered, labelled per-call records for the current request.
        self.call_records: list[dict[str, Any]] = []
        # Live denoise-loop position, maintained by the loop hook below.
        self.loop_context: dict[str, int | None] = {
            "mpg_iteration": None,
            "flow_step": None,
        }
        self._pending_events: list[dict[str, Any]] = []
        self._next_order = 0

    def reset(self) -> None:
        self.current = defaultdict(list)
        self.current_host_launch = defaultdict(list)
        self.current_device = defaultdict(list)
        self.call_records = []
        self.loop_context = {"mpg_iteration": None, "flow_step": None}
        self._pending_events = []
        self._next_order = 0

    def _record(
        self,
        stage: str,
        elapsed_ms: float,
        label: str | None = None,
        *,
        timing_source: str = "synchronized_wall",
        host_launch_ms: float | None = None,
        device_ms: float | None = None,
        mpg_iteration: int | None = None,
        flow_step: int | None = None,
        order: int | None = None,
    ) -> None:
        self.current[stage].append(elapsed_ms)
        if host_launch_ms is not None:
            self.current_host_launch[stage].append(host_launch_ms)
        if device_ms is not None:
            self.current_device[stage].append(device_ms)
        if mpg_iteration is None:
            mpg_iteration = self.loop_context["mpg_iteration"]
        if flow_step is None:
            flow_step = self.loop_context["flow_step"]
        if order is None:
            order = self._next_order
            self._next_order += 1
        self.call_records.append(
            {
                "order": order,
                "stage": stage,
                "ms": elapsed_ms,
                "timing_source": timing_source,
                "host_launch_ms": host_launch_ms,
                "device_ms": device_ms,
                "mpg_iteration": mpg_iteration,
                "flow_step": flow_step,
                "label": (
                    label
                    if label is not None
                    else position_label(mpg_iteration, flow_step)
                ),
            }
        )

    def _queue_event(
        self,
        *,
        stage: str,
        start_event: Any,
        end_event: Any,
        host_launch_ms: float,
        label: str | None,
        mpg_iteration: int | None,
        flow_step: int | None,
    ) -> None:
        order = self._next_order
        self._next_order += 1
        self._pending_events.append(
            {
                "order": order,
                "stage": stage,
                "start_event": start_event,
                "end_event": end_event,
                "host_launch_ms": host_launch_ms,
                "label": label,
                "mpg_iteration": mpg_iteration,
                "flow_step": flow_step,
            }
        )

    def finalize_request(self) -> None:
        """Resolve queued device events after the caller synchronizes once."""
        if self.timing_mode != STAGE_TIMING_EVENT:
            return
        pending = self._pending_events
        self._pending_events = []
        for item in sorted(pending, key=lambda entry: entry["order"]):
            device_ms = float(
                item["start_event"].elapsed_time(item["end_event"])
            )
            host_launch_ms = float(item["host_launch_ms"])
            primary_ms = (
                host_launch_ms
                if item["stage"] in HOST_PRIMARY_STAGES
                else device_ms
            )
            self._record(
                item["stage"],
                primary_ms,
                item["label"],
                timing_source=(
                    "host_wall"
                    if item["stage"] in HOST_PRIMARY_STAGES
                    else "device_event"
                ),
                host_launch_ms=host_launch_ms,
                device_ms=device_ms,
                mpg_iteration=item["mpg_iteration"],
                flow_step=item["flow_step"],
                order=item["order"],
            )

    def timed_call(
        self, stage: str, function: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        if not self.enabled:
            return function(*args, **kwargs)
        if self.timing_mode == STAGE_TIMING_EVENT:
            mpg_iteration = self.loop_context["mpg_iteration"]
            flow_step = self.loop_context["flow_step"]
            label = position_label(mpg_iteration, flow_step)
            start_event = create_timing_event()
            end_event = create_timing_event()
            start_event.record()
            host_start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                host_launch_ms = (time.perf_counter() - host_start) * 1000.0
                end_event.record()
                self._queue_event(
                    stage=stage,
                    start_event=start_event,
                    end_event=end_event,
                    host_launch_ms=host_launch_ms,
                    label=label,
                    mpg_iteration=mpg_iteration,
                    flow_step=flow_step,
                )
        synchronize()
        start = time.perf_counter()
        result = function(*args, **kwargs)
        synchronize()
        self._record(stage, (time.perf_counter() - start) * 1000.0)
        return result

    def wrap_method(self, target: Any, method_name: str, stage: str) -> None:
        original = getattr(target, method_name)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.timed_call(stage, original, *args, **kwargs)

        setattr(target, method_name, wrapper)

    # --- fine-grained additions: denoise loop hook -------------------------
    def loop_hook(self, loop_name: str, iterable: Any) -> Any:
        """Installed into `beingvla.PROFILE_LOOP_HOOK`.

        Returns the original iterable untouched unless stage-isolated
        profiling is active, so service-mode measurements and production
        behaviour are unaffected.
        """
        if not self.enabled:
            return iterable
        return self._timed_loop(loop_name, iterable)

    def _timed_loop(self, loop_name: str, iterable: Any) -> Any:
        stage = (
            DENOISE_OUTER_STAGE
            if loop_name == "mpg_iteration"
            else DENOISE_INNER_STAGE
        )
        for index, item in enumerate(iterable):
            self.loop_context[loop_name] = index
            if loop_name == "mpg_iteration":
                self.loop_context["flow_step"] = None
                label = f"{iteration_prefix(index)}_iteration"
            else:
                label = position_label(
                    self.loop_context["mpg_iteration"], index
                )
            if self.timing_mode == STAGE_TIMING_EVENT:
                start_event = create_timing_event()
                end_event = create_timing_event()
                start_event.record()
            else:
                synchronize()
            start = time.perf_counter()
            try:
                yield item
            finally:
                # Also runs on `break` (GeneratorExit), so a partially
                # executed body is still attributed instead of vanishing.
                host_launch_ms = (time.perf_counter() - start) * 1000.0
                mpg_iteration = self.loop_context["mpg_iteration"]
                flow_step = index if loop_name == "flow_step" else None
                if self.timing_mode == STAGE_TIMING_EVENT:
                    end_event.record()
                    self._queue_event(
                        stage=stage,
                        start_event=start_event,
                        end_event=end_event,
                        host_launch_ms=host_launch_ms,
                        label=label,
                        mpg_iteration=mpg_iteration,
                        flow_step=flow_step,
                    )
                else:
                    synchronize()
                    self._record(
                        stage,
                        (time.perf_counter() - start) * 1000.0,
                        label,
                        mpg_iteration=mpg_iteration,
                        flow_step=flow_step,
                    )
                self.loop_context[loop_name] = None


DEFAULT_INSTRUCTION = (
    "Pick up the black bowl between the plate and the ramekin and place it "
    "on the plate."
)


def build_observation(
    image_size: int = 256,
    instruction: str = DEFAULT_INSTRUCTION,
) -> dict[str, Any]:
    """Create a deterministic LIBERO-shaped request without simulator overhead."""
    rng = np.random.default_rng(41)
    return {
        "state.state": np.zeros((1, 8), dtype=np.float32),
        "state.eef_position": np.zeros((1, 3), dtype=np.float32),
        "state.eef_rotation": np.zeros((1, 3), dtype=np.float32),
        "state.libero_gripper_position": np.zeros((1, 2), dtype=np.float32),
        "video.top_view": rng.integers(
            0, 256, size=(1, image_size, image_size, 3), dtype=np.uint8
        ),
        "video.wrist_view": rng.integers(
            0, 256, size=(1, image_size, image_size, 3), dtype=np.uint8
        ),
        "language.instruction": [instruction],
    }


def load_instructions(path: Optional[str]) -> list[str]:
    """Load a JSON string list for deterministic mixed-prompt profiling."""
    if path is None:
        return [DEFAULT_INSTRUCTION]
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("instructions file must contain a non-empty JSON list")
    if not all(isinstance(item, str) and item.strip() for item in payload):
        raise ValueError("every instruction must be a non-empty string")
    return payload


def add_stage_instrumentation(policy: BeingHPolicy, recorder: StageRecorder) -> None:
    model = policy.model
    recorder.wrap_method(policy, "_prepare_packed_inputs", "policy_pack_inputs")
    recorder.wrap_method(model, "extract_feature", "vision_pipeline")
    if not model.enable_npu_vision_compile:
        recorder.wrap_method(model.vit_model, "forward", "vision_backbone")
        recorder.wrap_method(model.connector, "forward", "vision_connector")
    recorder.wrap_method(
        model.proprio_encoder_robot, "forward", "proprioception_encoder"
    )
    # OPT-11 compiles these module callables with fullgraph=True. Wrapping
    # their forward methods with a profiler-side synchronize would insert a
    # graph break into the compiled graph, so use the model boundary timing
    # for this candidate instead of per-module stage instrumentation.
    if not model.enable_npu_action_compile:
        recorder.wrap_method(model.action_encoder, "forward", "action_encoder")
    recorder.wrap_method(model.language_model, "forward_train", "qwen_mot_forward")
    recorder.wrap_method(
        model.language_model,
        "build_static_prefix_cache",
        "qwen_prefix_prefill",
    )
    recorder.wrap_method(
        model.language_model,
        "forward_action_with_prefix_cache",
        "qwen_action_suffix_forward",
    )
    if model.enable_npu_capture_replay:
        recorder.wrap_method(
            model._get_npu_action_suffix_graph_runner(),
            "_replay",
            "qwen_action_suffix_forward",
        )
    if model.enable_npu_prefix_graph_replay:
        recorder.wrap_method(
            model._get_npu_prefix_graph_runner(),
            "forward",
            "qwen_prefix_prefill",
        )
    if model.enable_npu_baseline_flow_graph_replay:
        recorder.wrap_method(
            model._get_npu_baseline_flow_graph_runner(),
            "_replay",
            "baseline_flow_graph_replay",
        )
    if not model.enable_npu_action_compile:
        recorder.wrap_method(model.action_decoder, "forward", "action_decoder")
    if model.mpg is not None:
        recorder.wrap_method(model.mpg, "forward", "mpg_enhancement")
    if model.action_to_vlm_proj is not None:
        recorder.wrap_method(
            model.action_to_vlm_proj, "forward", "mpg_action_to_vlm_projection"
        )
    if model.vlm_to_action_proj is not None:
        recorder.wrap_method(
            model.vlm_to_action_proj, "forward", "mpg_vlm_to_action_projection"
        )

    if ACTIVE_DEVICE_TYPE == "npu":
        original_mask_builder = beingvla_module.create_npu_causal_masks

        def timed_mask_builder(*args: Any, **kwargs: Any) -> Any:
            return recorder.timed_call(
                "npu_dense_attention_mask", original_mask_builder, *args, **kwargs
            )

        beingvla_module.create_npu_causal_masks = timed_mask_builder
    else:
        original_sparse_mask_builder = beingvla_module.create_sparse_mask
        original_block_mask_builder = beingvla_module.create_block_mask

        def timed_sparse_mask_builder(*args: Any, **kwargs: Any) -> Any:
            return recorder.timed_call(
                "gpu_sparse_attention_mask",
                original_sparse_mask_builder,
                *args,
                **kwargs,
            )

        def timed_block_mask_builder(*args: Any, **kwargs: Any) -> Any:
            return recorder.timed_call(
                "gpu_block_attention_mask",
                original_block_mask_builder,
                *args,
                **kwargs,
            )

        beingvla_module.create_sparse_mask = timed_sparse_mask_builder
        beingvla_module.create_block_mask = timed_block_mask_builder

    original_model_get_action = model.get_action

    def timed_model_get_action(*args: Any, **kwargs: Any) -> Any:
        if recorder.measure_model_boundary and not recorder.enabled:
            synchronize()
            start = time.perf_counter()
            result = original_model_get_action(*args, **kwargs)
            synchronize()
            recorder.model_boundary_samples.append(
                (time.perf_counter() - start) * 1000.0
            )
            return result
        return recorder.timed_call(
            "model_get_action_total", original_model_get_action, *args, **kwargs
        )

    model.get_action = timed_model_get_action

    # --- fine-grained additions: attach the denoise loop hook. The model-side
    # `profiled_loop` is an identity function until this assignment happens,
    # and the hook itself is a pass-through while `recorder.enabled` is False.
    beingvla_module.PROFILE_LOOP_HOOK = recorder.loop_hook


def add_prefix_layer_instrumentation(
    policy: BeingHPolicy, recorder: StageRecorder
) -> None:
    """Optionally time each decoder layer during prefix prefill.

    This is intentionally opt-in because one event pair per layer adds
    profiler bookkeeping.  It is used only for a diagnostic run after the
    regular end-to-end baseline has already been collected.
    """
    language_model = policy.model.language_model
    layers = getattr(language_model, "layers", None)
    if layers is None:
        layers = language_model.model.layers
    for index, layer in enumerate(layers):
        recorder.wrap_method(layer, "forward", f"prefix_layer_{index:02d}")
        # A focused attention-only mode lets us measure the dominant sub-path
        # across all 28 layers without adding the much larger bookkeeping cost
        # of wrapping every layer submodule.  It is diagnostic-only and does
        # not affect the default profiling or serving path.
        if (
            os.environ.get("BEING_PROFILE_PREFIX_ATTENTION") == "1"
            and getattr(layer, "self_attn", None) is not None
        ):
            recorder.wrap_method(
                layer.self_attn, "forward", f"prefix_layer_{index:02d}_self_attn"
            )
        if os.environ.get("BEING_PROFILE_PREFIX_DETAIL") == "1" and index < 2:
            for name in (
                "input_layernorm",
                "input_layernorm_mot_gen",
                "self_attn",
                "post_attention_layernorm",
                "post_attention_layernorm_mot_gen",
                "mlp",
                "mlp_mot_gen",
            ):
                module = getattr(layer, name, None)
                if module is not None:
                    recorder.wrap_method(
                        module, "forward", f"prefix_layer_{index:02d}_{name}"
                    )


# ---------------------------------------------------------------------------
# FINE-GRAINED DENOISE INSTRUMENTATION: analysis helpers -- BEGIN
# ---------------------------------------------------------------------------


def _sum_calls(
    calls: list[dict[str, Any]], stage: str, where: str = "any"
) -> tuple[float, int]:
    total = 0.0
    count = 0
    for call in calls:
        if call["stage"] != stage:
            continue
        if where == "in_flow_step" and call["flow_step"] is None:
            continue
        if where == "outside_flow_step" and call["flow_step"] is not None:
            continue
        total += call["ms"]
        count += 1
    return total, count


def _resolve_child(
    calls: list[dict[str, Any]], child: str
) -> tuple[float, int]:
    if "@" in child:
        stage, where = child.split("@", 1)
        return _sum_calls(calls, stage, where)
    return _sum_calls(calls, child)


def build_reconciliation(
    calls: list[dict[str, Any]], policy_total_ms: float
) -> dict[str, Any]:
    """Explicit `total - sum(mutually exclusive children)` per nesting level."""
    levels: dict[str, Any] = {}
    for parent, children in STAGE_NESTING.items():
        if parent == "instrumented_policy_total_ms":
            parent_total = policy_total_ms
        else:
            parent_total, parent_calls = _sum_calls(calls, parent)
            if parent_calls == 0:
                continue
        child_ms: dict[str, float] = {}
        for child in children:
            value, count = _resolve_child(calls, child)
            if count == 0:
                continue
            child_ms[child] = value
        accounted = sum(child_ms.values())
        levels[parent] = {
            "total_ms": parent_total,
            "children_ms": child_ms,
            "accounted_ms": accounted,
            "residual_ms": parent_total - accounted,
            "residual_pct_of_parent": (
                100.0 * (parent_total - accounted) / parent_total
                if parent_total
                else 0.0
            ),
        }
    return levels


def summarize_reconciliation(
    per_request: list[dict[str, Any]]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    parents = {parent for entry in per_request for parent in entry}
    for parent in parents:
        entries = [entry[parent] for entry in per_request if parent in entry]
        child_names = {name for entry in entries for name in entry["children_ms"]}
        summary[parent] = {
            "total_ms": describe([e["total_ms"] for e in entries]),
            "accounted_ms": describe([e["accounted_ms"] for e in entries]),
            "residual_ms": describe([e["residual_ms"] for e in entries]),
            "residual_pct_of_parent": describe(
                [e["residual_pct_of_parent"] for e in entries]
            ),
            "children_ms": {
                name: describe(
                    [e["children_ms"].get(name, 0.0) for e in entries]
                )
                for name in sorted(child_names)
            },
        }
    return summary


def build_denoise_decomposition(
    stage_samples: list[dict[str, Any]]
) -> dict[str, Any]:
    """Per-(mpg_iteration, flow_step) breakdown of the denoise loops."""
    step_totals: dict[str, list[float]] = defaultdict(list)
    step_stage_ms: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    step_stage_calls: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    iter_totals: dict[str, list[float]] = defaultdict(list)
    iter_step_sums: dict[str, list[float]] = defaultdict(list)

    for sample in stage_samples:
        calls = sample["calls"]
        per_label_stage: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        per_iter_step_sum: dict[str, float] = defaultdict(float)
        for call in calls:
            stage = call["stage"]
            label = call["label"]
            if stage == DENOISE_INNER_STAGE:
                step_totals[label].append(call["ms"])
                iteration = call["mpg_iteration"]
                if iteration is not None:
                    per_iter_step_sum[
                        f"{iteration_prefix(iteration)}_iteration"
                    ] += call["ms"]
                continue
            if stage == DENOISE_OUTER_STAGE:
                iter_totals[label].append(call["ms"])
                continue
            if call["flow_step"] is not None:
                per_label_stage[label][stage].append(call["ms"])
        for label, stages in per_label_stage.items():
            for stage, values in stages.items():
                step_stage_ms[label][stage].append(sum(values))
                step_stage_calls[label][stage].append(len(values))
        for label, value in per_iter_step_sum.items():
            iter_step_sums[label].append(value)

    per_flow_step: dict[str, Any] = {}
    for label in sorted(step_totals):
        totals = step_totals[label]
        stages = {
            stage: {
                "calls_per_step": (
                    statistics.mean(step_stage_calls[label][stage])
                ),
                "ms": describe(values),
            }
            for stage, values in sorted(step_stage_ms[label].items())
        }
        accounted_per_sample = [
            sum(
                values[index]
                for values in step_stage_ms[label].values()
                if index < len(values)
            )
            for index in range(len(totals))
        ]
        glue = [
            total - accounted
            for total, accounted in zip(totals, accounted_per_sample)
        ]
        per_flow_step[label] = {
            "flow_step_total_ms": describe(totals),
            "stages": stages,
            "accounted_ms": describe(accounted_per_sample),
            "glue_ms": describe(glue),
            "glue_pct_of_step": describe(
                [
                    100.0 * g / t if t else 0.0
                    for g, t in zip(glue, totals)
                ]
            ),
        }

    per_outer_iteration: dict[str, Any] = {}
    for label in sorted(iter_totals):
        totals = iter_totals[label]
        step_sums = iter_step_sums.get(label, [])
        glue = [
            total - steps for total, steps in zip(totals, step_sums)
        ]
        per_outer_iteration[label] = {
            "iteration_total_ms": describe(totals),
            "sum_of_flow_steps_ms": describe(step_sums),
            "glue_ms": describe(glue),
        }

    return {
        "note": (
            f"'{DENOISE_INNER_STAGE}' is nested inside "
            f"'{DENOISE_OUTER_STAGE}', which is nested inside "
            "'model_get_action_total'. 'glue_ms' is the loop-body time not "
            "covered by the named child stages (noise sampling, Euler "
            "update, clone/index_put, reshape/cat, RTC masking, Python "
            "overhead, plus profiler synchronization overhead)."
        ),
        "per_flow_step": per_flow_step,
        "per_outer_iteration": per_outer_iteration,
    }


# FINE-GRAINED DENOISE INSTRUMENTATION: analysis helpers -- END


def create_policy(args: argparse.Namespace) -> BeingHPolicy:
    template = (
        "According to the instruction '{task_description}', what's the micro-step "
        "actions in the next {k} steps?"
    )
    return BeingHPolicy(
        model_path=args.model_path,
        data_config_name="libero_nonorm",
        dataset_name="libero_posttrain",
        embodiment_tag="libero",
        instruction_template=template,
        max_view_num=-1,
        use_fixed_view=False,
        action_attn_mode="causal",
        device=f"{ACTIVE_DEVICE_TYPE}:{args.device}",
        enable_rtc=False,
        enable_static_prefix_cache=args.enable_static_prefix_cache,
        enable_npu_fusion_attention=args.enable_npu_fusion_attention,
        enable_npu_fusion_attention_bsnd=(
            args.enable_npu_fusion_attention_bsnd
        ),
        enable_npu_hybrid_attention_layout=(
            args.enable_npu_hybrid_attention_layout
        ),
        enable_npu_prefix_segment_route=(
            args.enable_npu_prefix_segment_route
        ),
        enable_npu_projection_fusion=args.enable_npu_projection_fusion,
        enable_npu_vectorized_mpg=args.enable_npu_vectorized_mpg,
        enable_npu_workspace_reuse=args.enable_npu_workspace_reuse,
        enable_npu_kv_workspace=args.enable_npu_kv_workspace,
        enable_cuda_kv_workspace=args.enable_cuda_kv_workspace,
        enable_npu_add_rms_norm=args.enable_npu_add_rms_norm,
        enable_npu_fused_rotary=args.enable_npu_fused_rotary,
        enable_npu_fused_swiglu=args.enable_npu_fused_swiglu,
        enable_cuda_fused_rotary=args.enable_cuda_fused_rotary,
        enable_cuda_fused_swiglu=args.enable_cuda_fused_swiglu,
        enable_cuda_fused_only_projection_storage=(
            args.enable_cuda_fused_only_projection_storage
        ),
        enable_fused_only_projection_storage=(
            args.enable_fused_only_projection_storage
        ),
        enable_npu_static_tensor_cache=args.enable_npu_static_tensor_cache,
        enable_npu_dtype_fast_path=args.enable_npu_dtype_fast_path,
        enable_npu_euler_buffer_cache=args.enable_npu_euler_buffer_cache,
        enable_npu_vision_state_overlap=args.enable_npu_vision_state_overlap,
        enable_npu_action_compile=args.enable_npu_action_compile,
        enable_npu_vision_compile=args.enable_npu_vision_compile,
        enable_npu_linear_weight_prelayout=(
            args.enable_npu_linear_weight_prelayout
        ),
        enable_npu_persistent_compile_cache=(
            args.enable_npu_persistent_compile_cache
        ),
        npu_compile_cache_dir=args.npu_compile_cache_dir,
        enable_adaptive_flow_steps=args.enable_adaptive_flow_steps,
        adaptive_flow_min_steps=args.adaptive_flow_min_steps,
        adaptive_flow_velocity_threshold=(
            args.adaptive_flow_velocity_threshold
        ),
        enable_adaptive_mpg_refinement=(
            args.enable_adaptive_mpg_refinement
        ),
        adaptive_mpg_gate_threshold=args.adaptive_mpg_gate_threshold,
        enable_policy_prompt_cache=args.enable_policy_prompt_cache,
        npu_single_sample_fast_path=args.npu_single_sample_fast_path,
        enable_npu_capture_replay=getattr(
            args, "enable_npu_capture_replay", False
        ),
        enable_npu_prefix_graph_replay=getattr(
            args, "enable_npu_prefix_graph_replay", False
        ),
        npu_graph_cache_max_entries=getattr(
            args, "npu_graph_cache_max_entries", 1
        ),
        enable_cuda_capture_replay=getattr(
            args, "enable_cuda_capture_replay", False
        ),
        cuda_graph_cache_max_entries=getattr(
            args, "cuda_graph_cache_max_entries", None
        ),
        enable_npu_baseline_flow_graph_replay=getattr(
            args, "enable_npu_baseline_flow_graph_replay", False
        ),
    )


def run_policy(policy: BeingHPolicy, observation: dict[str, Any]) -> dict[str, Any]:
    return policy.get_action(copy.deepcopy(observation))


def capture_operator_profile(
    policy: BeingHPolicy,
    observation: dict[str, Any],
    output_dir: Path,
    *,
    npu_profiler_level: str = "level0",
    with_stack: bool = False,
) -> dict[str, Any]:
    if ACTIVE_DEVICE_TYPE == "cuda":
        trace_path = output_dir / "gpu_operator_trace.json"
        table_path = output_dir / "gpu_operator_top.txt"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=with_stack,
        ) as profile:
            run_policy(policy, observation)
            synchronize()
        profile.export_chrome_trace(str(trace_path))
        table_path.write_text(
            profile.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=80
            ),
            encoding="utf-8",
        )
        return {
            "captured": True,
            "parsed": True,
            "trace": str(trace_path),
            "operator_table": str(table_path),
        }

    if torch_npu is None:
        return {
            "captured": False,
            "parsed": False,
            "reason": "torch_npu is unavailable",
        }

    # torch-npu launches the CANN exporter as a subprocess.  Calling this
    # script via an absolute virtualenv interpreter does not necessarily put
    # that virtualenv on PATH; in that case msprof may pick an ABI-incompatible
    # system Python and silently fail to emit operator/kernel CSVs.  Pin the
    # subprocess lookup to the interpreter that is running this profiler.
    interpreter_dir = str(Path(sys.executable).parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if not path_entries or path_entries[0] != interpreter_dir:
        os.environ["PATH"] = os.pathsep.join(
            [interpreter_dir, *[entry for entry in path_entries if entry]]
        )

    profile_dir = output_dir / "npu_operator_profile"
    trace_path = output_dir / "npu_operator_trace.json"
    handler = torch_npu.profiler.tensorboard_trace_handler(
        str(profile_dir), analyse_flag=True, async_mode=False
    )
    schedule = torch_npu.profiler.schedule(
        wait=0, warmup=0, active=1, repeat=1, skip_first=0
    )
    profiler_level = (
        torch_npu.profiler.ProfilerLevel.Level1
        if npu_profiler_level == "level1"
        else torch_npu.profiler.ProfilerLevel.Level0
    )
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=profiler_level,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        export_type="text",
    )
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=schedule,
        on_trace_ready=handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=with_stack,
        experimental_config=experimental_config,
    ) as profile:
        run_policy(policy, observation)
        synchronize()
        profile.step()
    try:
        profile.export_chrome_trace(str(trace_path))
    except Exception as error:
        return {
            "captured": True,
            "parsed": False,
            "profile_directory": str(profile_dir),
            "reason": f"Chrome trace export failed: {error}",
        }

    operator_files = list(profile_dir.rglob("operator_details.csv"))
    kernel_files = list(profile_dir.rglob("kernel_details.csv"))
    return {
        "captured": True,
        "parsed": bool(operator_files),
        "trace": str(trace_path),
        "profile_directory": str(profile_dir),
        "operator_details": [str(path) for path in operator_files],
        "kernel_details": [str(path) for path in kernel_files],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--backend", choices=["auto", "cuda", "npu"], default="auto"
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--stage-iterations", type=int, default=5)
    parser.add_argument("--memory-snapshot-interval", type=int, default=0)
    parser.add_argument(
        "--stage-timing-mode",
        choices=STAGE_TIMING_MODES,
        default=STAGE_TIMING_SYNC,
        help=(
            "'sync' preserves the legacy per-stage synchronization protocol; "
            "'event' records device events and synchronizes once per request"
        ),
    )
    parser.add_argument("--action-chunk-length", type=int)
    parser.add_argument("--enable-static-prefix-cache", action="store_true")
    parser.add_argument("--enable-npu-fusion-attention", action="store_true")
    parser.add_argument("--enable-cuda-gqa-attention", action="store_true")
    parser.add_argument(
        "--enable-npu-fusion-attention-bsnd", action="store_true"
    )
    parser.add_argument(
        "--enable-npu-hybrid-attention-layout", action="store_true"
    )
    parser.add_argument(
        "--enable-npu-prefix-segment-route",
        action="store_true",
    )
    parser.add_argument("--enable-npu-projection-fusion", action="store_true")
    parser.add_argument("--enable-npu-vectorized-mpg", action="store_true")
    parser.add_argument("--enable-npu-workspace-reuse", action="store_true")
    parser.add_argument("--enable-npu-kv-workspace", action="store_true")
    parser.add_argument("--enable-cuda-kv-workspace", action="store_true")
    parser.add_argument("--enable-npu-add-rms-norm", action="store_true")
    parser.add_argument("--enable-npu-fused-rotary", action="store_true")
    parser.add_argument("--enable-npu-fused-swiglu", action="store_true")
    parser.add_argument("--enable-cuda-fused-rotary", action="store_true")
    parser.add_argument("--enable-cuda-fused-swiglu", action="store_true")
    parser.add_argument(
        "--enable-cuda-fused-only-projection-storage", action="store_true"
    )
    parser.add_argument(
        "--enable-fused-only-projection-storage",
        action="store_true",
        help=(
            "Use one shared physical storage for fused QKV/Gate-Up weights "
            "on the selected device (inference only)."
        ),
    )
    parser.add_argument("--enable-npu-static-tensor-cache", action="store_true")
    parser.add_argument("--enable-npu-dtype-fast-path", action="store_true")
    parser.add_argument("--enable-npu-euler-buffer-cache", action="store_true")
    parser.add_argument(
        "--enable-npu-vision-state-overlap", action="store_true"
    )
    parser.add_argument("--enable-npu-action-compile", action="store_true")
    parser.add_argument("--enable-npu-vision-compile", action="store_true")
    parser.add_argument(
        "--enable-npu-linear-weight-prelayout", action="store_true"
    )
    parser.add_argument(
        "--enable-npu-persistent-compile-cache", action="store_true"
    )
    parser.add_argument("--npu-compile-cache-dir")
    parser.add_argument("--enable-adaptive-flow-steps", action="store_true")
    parser.add_argument("--adaptive-flow-min-steps", type=int, default=2)
    parser.add_argument(
        "--adaptive-flow-velocity-threshold", type=float, default=0.0
    )
    parser.add_argument(
        "--enable-adaptive-mpg-refinement", action="store_true"
    )
    parser.add_argument(
        "--adaptive-mpg-gate-threshold", type=float, default=0.0
    )
    parser.add_argument("--enable-policy-prompt-cache", action="store_true")
    parser.add_argument(
        "--npu-single-sample-fast-path",
        choices=["off", "auto", "force"],
        default="off",
    )
    parser.add_argument("--enable-npu-capture-replay", action="store_true")
    parser.add_argument(
        "--enable-npu-prefix-graph-replay",
        action="store_true",
        help="Replay the fixed-shape OPT-01 Prefix prefill as one NPU graph",
    )
    parser.add_argument("--enable-cuda-capture-replay", action="store_true")
    parser.add_argument("--npu-graph-cache-max-entries", type=int, default=1)
    parser.add_argument("--cuda-graph-cache-max-entries", type=int)
    parser.add_argument(
        "--enable-npu-baseline-flow-graph-replay", action="store_true"
    )
    parser.add_argument(
        "--freeze-npu-graph-cache-after-warmup",
        action="store_true",
        help="Prohibit new request-path graph capture after warmup",
    )
    parser.add_argument(
        "--instructions-file",
        help="JSON list of task instructions cycled across all request phases",
    )
    parser.add_argument("--skip-operator-profile", action="store_true")
    parser.add_argument(
        "--npu-operator-profiler-level",
        choices=["level0", "level1"],
        default="level0",
        help="Use level1 when AI Core pipeline metrics are required.",
    )
    parser.add_argument(
        "--operator-profile-with-stack",
        action="store_true",
        help="Capture Python call stacks for source attribution.",
    )
    return parser.parse_args()


def main() -> None:
    global ACTIVE_DEVICE_TYPE
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "auto":
        if torch.cuda.is_available():
            ACTIVE_DEVICE_TYPE = "cuda"
        elif torch_npu is not None and torch.npu.is_available():
            ACTIVE_DEVICE_TYPE = "npu"
        else:
            raise RuntimeError("Neither CUDA nor NPU is available")
    else:
        ACTIVE_DEVICE_TYPE = args.backend

    torch.manual_seed(41)
    np.random.seed(41)
    if ACTIVE_DEVICE_TYPE == "cuda":
        torch.cuda.set_device(args.device)
        torch.cuda.manual_seed_all(41)
    else:
        if torch_npu is None:
            raise RuntimeError("NPU backend requested but torch_npu is unavailable")
        if args.enable_npu_linear_weight_prelayout:
            torch.npu.config.allow_internal_format = True
            torch.npu.set_mm_bmm_format_nd(False)
        torch.npu.set_device(args.device)
        torch.npu.manual_seed_all(41)

    load_start = time.perf_counter()
    policy = create_policy(args)
    if args.action_chunk_length is not None:
        policy.action_chunk_length = args.action_chunk_length
        policy.action_token_num = args.action_chunk_length
        policy.model.action_chunk_length = args.action_chunk_length
        policy.model.config.action_chunk_length = args.action_chunk_length
        policy.model.config.action_token_num = args.action_chunk_length
    policy.model.enable_static_prefix_cache = args.enable_static_prefix_cache
    policy.model.enable_npu_fusion_attention = args.enable_npu_fusion_attention
    policy.model.enable_cuda_gqa_attention = args.enable_cuda_gqa_attention
    policy.model.npu_fusion_attention_input_layout = (
        "BSND" if args.enable_npu_fusion_attention_bsnd else "BNSD"
    )
    policy.model.enable_npu_hybrid_attention_layout = (
        args.enable_npu_hybrid_attention_layout
    )
    policy.model.enable_npu_prefix_segment_route = (
        args.enable_npu_prefix_segment_route
    )
    policy.model.enable_npu_projection_fusion = (
        args.enable_npu_projection_fusion
    )
    policy.model.enable_npu_vectorized_mpg = args.enable_npu_vectorized_mpg
    policy.model.enable_npu_workspace_reuse = args.enable_npu_workspace_reuse
    policy.model.enable_npu_kv_workspace = (
        args.enable_npu_kv_workspace or args.enable_cuda_kv_workspace
    )
    policy.model.enable_npu_add_rms_norm = args.enable_npu_add_rms_norm
    policy.model.enable_npu_fused_rotary = args.enable_npu_fused_rotary
    policy.model.enable_npu_fused_swiglu = args.enable_npu_fused_swiglu
    policy.model.enable_cuda_fused_rotary = args.enable_cuda_fused_rotary
    policy.model.enable_cuda_fused_swiglu = args.enable_cuda_fused_swiglu
    policy.model.enable_npu_static_tensor_cache = (
        args.enable_npu_static_tensor_cache
    )
    policy.model.enable_npu_dtype_fast_path = args.enable_npu_dtype_fast_path
    policy.model.enable_npu_euler_buffer_cache = (
        args.enable_npu_euler_buffer_cache
    )
    policy.model.enable_npu_vision_state_overlap = (
        args.enable_npu_vision_state_overlap
    )
    policy.model.enable_npu_action_compile = args.enable_npu_action_compile
    policy.model.enable_npu_vision_compile = args.enable_npu_vision_compile
    policy.model.npu_single_sample_fast_path = (
        args.npu_single_sample_fast_path
    )
    policy.model.enable_npu_capture_replay = (
        args.enable_npu_capture_replay or args.enable_cuda_capture_replay
    )
    policy.model.enable_npu_prefix_graph_replay = (
        args.enable_npu_prefix_graph_replay
    )
    policy.model.npu_graph_cache_max_entries = (
        args.cuda_graph_cache_max_entries
        if ACTIVE_DEVICE_TYPE == "cuda"
        and args.cuda_graph_cache_max_entries is not None
        else args.npu_graph_cache_max_entries
    )
    policy.model.enable_npu_baseline_flow_graph_replay = (
        args.enable_npu_baseline_flow_graph_replay
    )
    synchronize()
    model_load_s = time.perf_counter() - load_start
    instructions = load_instructions(args.instructions_file)
    observations = [build_observation(instruction=item) for item in instructions]
    observation = observations[0]

    recorder = StageRecorder(args.stage_timing_mode)
    add_stage_instrumentation(policy, recorder)
    if os.environ.get("BEING_PROFILE_PREFIX_LAYERS") == "1":
        add_prefix_layer_instrumentation(policy, recorder)

    device_api = torch.cuda if ACTIVE_DEVICE_TYPE == "cuda" else torch.npu
    device_api.reset_peak_memory_stats(args.device)
    first_start = time.perf_counter()
    run_policy(policy, observation)
    synchronize()
    first_request_ms = (time.perf_counter() - first_start) * 1000.0

    for request_index in range(args.warmup):
        run_policy(policy, observations[request_index % len(observations)])
    synchronize()
    if (
        (
            args.enable_npu_capture_replay
            or args.enable_cuda_capture_replay
            or args.enable_npu_prefix_graph_replay
        )
        and args.freeze_npu_graph_cache_after_warmup
    ):
        policy.model.freeze_npu_capture_replay_cache()
    graph_runner_after_warmup = getattr(
        policy.model, "_npu_action_suffix_graph_runner", None
    )
    graph_cache_after_warmup = (
        graph_runner_after_warmup.stats()
        if graph_runner_after_warmup is not None
        else None
    )
    prefix_graph_runner_after_warmup = getattr(
        policy.model, "_npu_prefix_graph_runner", None
    )
    prefix_graph_cache_after_warmup = (
        prefix_graph_runner_after_warmup.stats()
        if prefix_graph_runner_after_warmup is not None
        else None
    )
    baseline_flow_runner_after_warmup = getattr(
        policy.model, "_npu_baseline_flow_graph_runner", None
    )
    baseline_flow_graph_cache_after_warmup = (
        baseline_flow_runner_after_warmup.stats()
        if baseline_flow_runner_after_warmup is not None
        else None
    )

    e2e_samples: list[float] = []
    memory_snapshots: list[dict[str, int]] = []
    adaptive_traces: list[dict[str, Any]] = []
    recorder.enabled = False
    recorder.measure_model_boundary = True
    for request_index in range(args.iterations):
        synchronize()
        start = time.perf_counter()
        run_policy(policy, observations[request_index % len(observations)])
        synchronize()
        e2e_samples.append((time.perf_counter() - start) * 1000.0)
        if args.memory_snapshot_interval > 0 and (
            request_index == 0
            or (request_index + 1) % args.memory_snapshot_interval == 0
            or request_index + 1 == args.iterations
        ):
            memory_snapshots.append(
                {
                    "completed_requests": request_index + 1,
                    "allocated_bytes": device_api.memory_allocated(args.device),
                    "reserved_bytes": device_api.memory_reserved(args.device),
                    "max_allocated_bytes": device_api.max_memory_allocated(
                        args.device
                    ),
                    "max_reserved_bytes": device_api.max_memory_reserved(
                        args.device
                    ),
                }
            )
        if (
            args.enable_adaptive_flow_steps
            or args.enable_adaptive_mpg_refinement
        ):
            adaptive_traces.append(
                {
                    "flow_steps": list(
                        policy.model.last_adaptive_flow_steps
                    ),
                    "flow_residuals": list(
                        policy.model.last_adaptive_flow_residuals
                    ),
                    "flow_residual_traces": [
                        list(trace)
                        for trace in (
                            policy.model.last_adaptive_flow_residual_traces
                        )
                    ],
                    "mpg_gate": policy.model.last_adaptive_mpg_gate,
                    "mpg_skipped": (
                        policy.model.last_adaptive_mpg_skipped
                    ),
                }
            )
    recorder.measure_model_boundary = False
    e2e_model_samples = recorder.model_boundary_samples
    e2e_pre_post_samples = [
        total - model
        for total, model in zip(e2e_samples, e2e_model_samples, strict=True)
    ]

    stage_samples: list[dict[str, Any]] = []
    recorder.enabled = True
    for request_index in range(args.stage_iterations):
        recorder.reset()
        synchronize()
        start = time.perf_counter()
        result = run_policy(
            policy, observations[request_index % len(observations)]
        )
        synchronize()
        total_ms = (time.perf_counter() - start) * 1000.0
        recorder.finalize_request()
        stage_samples.append(
            {
                "request_index": request_index,
                "instruction_index": request_index % len(observations),
                "instrumented_policy_total_ms": total_ms,
                "stages_ms": dict(recorder.current),
                "stage_host_launch_ms": dict(recorder.current_host_launch),
                "stage_device_ms": dict(recorder.current_device),
                # --- fine-grained addition: ordered, labelled per-call log ---
                "calls": list(recorder.call_records),
                "output_keys": sorted(result),
            }
        )
    recorder.enabled = False

    stage_values: dict[str, list[float]] = defaultdict(list)
    stage_call_values: dict[str, list[float]] = defaultdict(list)
    stage_call_counts: dict[str, list[int]] = defaultdict(list)
    stage_host_values: dict[str, list[float]] = defaultdict(list)
    stage_host_call_values: dict[str, list[float]] = defaultdict(list)
    stage_device_values: dict[str, list[float]] = defaultdict(list)
    stage_device_call_values: dict[str, list[float]] = defaultdict(list)
    for sample in stage_samples:
        for stage, calls in sample["stages_ms"].items():
            stage_values[stage].append(sum(calls))
            stage_call_values[stage].extend(calls)
            stage_call_counts[stage].append(len(calls))
        for stage, calls in sample["stage_host_launch_ms"].items():
            stage_host_values[stage].append(sum(calls))
            stage_host_call_values[stage].extend(calls)
        for stage, calls in sample["stage_device_ms"].items():
            stage_device_values[stage].append(sum(calls))
            stage_device_call_values[stage].extend(calls)

    # --- fine-grained additions: labelled aggregation + reconciliation ------
    labelled_values: dict[str, list[float]] = defaultdict(list)
    labelled_counts: dict[str, list[int]] = defaultdict(list)
    for sample in stage_samples:
        per_key: dict[str, list[float]] = defaultdict(list)
        for call in sample["calls"]:
            per_key[f"{call['stage']}::{call['label']}"].append(call["ms"])
        for key, values in per_key.items():
            labelled_values[key].append(sum(values))
            labelled_counts[key].append(len(values))
    stage_label_summary = {
        key: describe(values) for key, values in sorted(labelled_values.items())
    }
    reconciliation_per_request = [
        build_reconciliation(
            sample["calls"], sample["instrumented_policy_total_ms"]
        )
        for sample in stage_samples
    ]
    residual_reconciliation = summarize_reconciliation(
        reconciliation_per_request
    )
    denoise_decomposition = build_denoise_decomposition(stage_samples)

    model_config = policy.model.config
    result = {
        "measurement_note": {
            "e2e": "Request-boundary synchronization only; use for serving latency.",
            "stages": (
                (
                    "Synchronization surrounds each semantic stage; use for attribution."
                    if args.stage_timing_mode == STAGE_TIMING_SYNC
                    else "Device events surround each semantic stage and resolve after one "
                    "request-boundary synchronization; policy_pack_inputs uses host wall time."
                )
                + " Nested stage totals overlap and must not be summed directly."
            ),
            "denoise": (
                f"'{DENOISE_OUTER_STAGE}' times one MPG-iteration loop body "
                f"(1 sample per iteration); '{DENOISE_INNER_STAGE}' times one "
                "flow-timestep loop body and is nested inside it. Every call "
                "is labelled with its (mpg_iteration, flow_step) position; see "
                "'stage_summary_per_label' and 'denoise_decomposition'."
            ),
            "residual": (
                "'residual_reconciliation' reports, per nesting level, "
                "total - sum(mutually exclusive children) so the "
                "unattributed block is an explicit number."
            ),
        },
        "stage_nesting": STAGE_NESTING,
        "stage_timing_mode": args.stage_timing_mode,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__ if torch_npu is not None else None,
            "backend": ACTIVE_DEVICE_TYPE,
            "device_index": args.device,
            "device_name": device_api.get_device_name(args.device),
        },
        "configuration": {
            "model_path": str(Path(args.model_path).resolve()),
            "data_config_name": "libero_nonorm",
            "dataset_name": "libero_posttrain",
            "embodiment_tag": "libero",
            "image_views": 2,
            "image_size": 256,
            "action_chunk_length": policy.action_chunk_length,
            "action_chunk_length_override": args.action_chunk_length,
            "unified_action_dim": policy.model.unified_action_dim,
            "num_inference_timesteps": policy.model.num_inference_timesteps,
            "use_mpg": policy.model.use_mpg,
            "mpg_refinement_iters": getattr(
                policy.model, "mpg_refinement_iters", 0
            ),
            "mpg_num_projections": getattr(
                model_config, "mpg_num_projections", None
            ),
            "attention_mode": policy.action_attn_mode,
            "rtc_enabled": False,
            "static_prefix_cache": args.enable_static_prefix_cache,
            "npu_fusion_attention": args.enable_npu_fusion_attention,
            "cuda_gqa_attention": args.enable_cuda_gqa_attention,
            "npu_fusion_attention_input_layout": (
                policy.model.npu_fusion_attention_input_layout
            ),
            "npu_hybrid_attention_layout": (
                policy.model.enable_npu_hybrid_attention_layout
            ),
            "npu_prefix_segment_route": (
                args.enable_npu_prefix_segment_route
            ),
            "npu_projection_fusion": args.enable_npu_projection_fusion,
            "npu_vectorized_mpg": args.enable_npu_vectorized_mpg,
            "npu_workspace_reuse": args.enable_npu_workspace_reuse,
            "npu_kv_workspace": args.enable_npu_kv_workspace,
            "cuda_kv_workspace": args.enable_cuda_kv_workspace,
            "npu_add_rms_norm": args.enable_npu_add_rms_norm,
            "npu_fused_rotary": args.enable_npu_fused_rotary,
            "npu_fused_swiglu": args.enable_npu_fused_swiglu,
            "cuda_fused_rotary": args.enable_cuda_fused_rotary,
            "cuda_fused_swiglu": args.enable_cuda_fused_swiglu,
            "cuda_fused_only_projection_storage": (
                args.enable_cuda_fused_only_projection_storage
            ),
            "fused_only_projection_storage": (
                args.enable_fused_only_projection_storage
            ),
            "cuda_fused_projection_storage_report": (
                getattr(policy, "cuda_fused_projection_storage_report", None)
                if ACTIVE_DEVICE_TYPE == "cuda"
                else None
            ),
            "fused_projection_storage_report": getattr(
                policy, "fused_projection_storage_report", None
            ),
            "npu_operator_profiler_level": args.npu_operator_profiler_level,
            "operator_profile_with_stack": args.operator_profile_with_stack,
            "npu_static_tensor_cache": args.enable_npu_static_tensor_cache,
            "npu_dtype_fast_path": args.enable_npu_dtype_fast_path,
            "npu_euler_buffer_cache": args.enable_npu_euler_buffer_cache,
            "npu_vision_state_overlap": args.enable_npu_vision_state_overlap,
            "npu_action_compile": args.enable_npu_action_compile,
            "npu_vision_compile": args.enable_npu_vision_compile,
            "npu_linear_weight_prelayout": (
                args.enable_npu_linear_weight_prelayout
            ),
            "npu_linear_weight_prelayout_count": (
                policy.npu_linear_weight_prelayout_count
            ),
            "npu_persistent_compile_cache": (
                args.enable_npu_persistent_compile_cache
            ),
            "npu_compile_cache_dir": args.npu_compile_cache_dir,
            "policy_prompt_cache": args.enable_policy_prompt_cache,
            "npu_prefix_segment_route_last_reason": (
                policy.model.npu_prefix_segment_route_last_reason
            ),
            "npu_single_sample_fast_path": (
                args.npu_single_sample_fast_path
            ),
            "npu_capture_replay": args.enable_npu_capture_replay,
            "npu_prefix_graph_replay": args.enable_npu_prefix_graph_replay,
            "cuda_capture_replay": args.enable_cuda_capture_replay,
            "npu_graph_cache_max_entries": (
                args.npu_graph_cache_max_entries
            ),
            "cuda_graph_cache_max_entries": (
                args.cuda_graph_cache_max_entries
            ),
            "freeze_npu_graph_cache_after_warmup": (
                args.freeze_npu_graph_cache_after_warmup
            ),
            "npu_baseline_flow_graph_replay": (
                args.enable_npu_baseline_flow_graph_replay
            ),
            "instruction_count": len(instructions),
            "instruction_lengths": [len(item) for item in instructions],
            "pytorch_npu_alloc_conf": os.getenv(
                "PYTORCH_NPU_ALLOC_CONF", ""
            ),
            "adaptive_flow_steps": args.enable_adaptive_flow_steps,
            "adaptive_flow_min_steps": args.adaptive_flow_min_steps,
            "adaptive_flow_velocity_threshold": (
                args.adaptive_flow_velocity_threshold
            ),
            "adaptive_mpg_refinement": (
                args.enable_adaptive_mpg_refinement
            ),
            "adaptive_mpg_gate_threshold": (
                args.adaptive_mpg_gate_threshold
            ),
        },
        "model_load_s": model_load_s,
        "first_request_ms": first_request_ms,
        "warmup_iterations": args.warmup,
        "e2e_samples_ms": e2e_samples,
        "memory_snapshots": memory_snapshots,
        "e2e_summary": describe(e2e_samples),
        "e2e_model_samples_ms": e2e_model_samples,
        "e2e_model_summary": describe(e2e_model_samples),
        "e2e_pre_post_samples_ms": e2e_pre_post_samples,
        "e2e_pre_post_summary": describe(e2e_pre_post_samples),
        "adaptive_traces": adaptive_traces,
        "stage_samples": stage_samples,
        "stage_summary_per_request": {
            stage: describe(values) for stage, values in stage_values.items()
        },
        "stage_summary_per_call": {
            stage: describe(values) for stage, values in stage_call_values.items()
        },
        "stage_call_counts_per_request": stage_call_counts,
        "stage_host_launch_summary_per_request": {
            stage: describe(values) for stage, values in stage_host_values.items()
        },
        "stage_host_launch_summary_per_call": {
            stage: describe(values)
            for stage, values in stage_host_call_values.items()
        },
        "stage_device_summary_per_request": {
            stage: describe(values) for stage, values in stage_device_values.items()
        },
        "stage_device_summary_per_call": {
            stage: describe(values)
            for stage, values in stage_device_call_values.items()
        },
        # --- fine-grained additions ---
        "stage_summary_per_label": stage_label_summary,
        "stage_label_call_counts_per_request": labelled_counts,
        "residual_reconciliation": residual_reconciliation,
        "residual_reconciliation_per_request": reconciliation_per_request,
        "denoise_decomposition": denoise_decomposition,
        "memory": {
            "peak_allocated_bytes": device_api.max_memory_allocated(args.device),
            "peak_reserved_bytes": device_api.max_memory_reserved(args.device),
            "allocator_backend": (
                torch.npu.get_allocator_backend()
                if ACTIVE_DEVICE_TYPE == "npu"
                else None
            ),
            "inactive_split_bytes_current": (
                torch.npu.memory_stats().get(
                    "inactive_split_bytes.all.current", 0
                )
                if ACTIVE_DEVICE_TYPE == "npu"
                else None
            ),
            "num_alloc_retries": (
                torch.npu.memory_stats().get("num_alloc_retries", 0)
                if ACTIVE_DEVICE_TYPE == "npu"
                else None
            ),
        },
    }

    graph_runner = getattr(policy.model, "_npu_action_suffix_graph_runner", None)
    result["npu_graph_cache"] = (
        graph_runner.stats() if graph_runner is not None else None
    )
    result["npu_graph_cache_after_warmup"] = graph_cache_after_warmup
    prefix_graph_runner = getattr(
        policy.model, "_npu_prefix_graph_runner", None
    )
    result["npu_prefix_graph_cache"] = (
        prefix_graph_runner.stats()
        if prefix_graph_runner is not None
        else None
    )
    result["npu_prefix_graph_cache_after_warmup"] = (
        prefix_graph_cache_after_warmup
    )
    baseline_flow_runner = getattr(
        policy.model, "_npu_baseline_flow_graph_runner", None
    )
    result["npu_baseline_flow_graph_cache"] = (
        baseline_flow_runner.stats()
        if baseline_flow_runner is not None
        else None
    )
    result["npu_baseline_flow_graph_cache_after_warmup"] = (
        baseline_flow_graph_cache_after_warmup
    )

    raw_path = output_dir / "npu_pipeline_profile.json"
    raw_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    operator_result = {"captured": False, "reason": "disabled"}
    if not args.skip_operator_profile:
        operator_result = capture_operator_profile(
            policy,
            observation,
            output_dir,
            npu_profiler_level=args.npu_operator_profiler_level,
            with_stack=args.operator_profile_with_stack,
        )
    result["operator_profile"] = operator_result
    raw_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("PROFILE_RESULT " + json.dumps(result), flush=True)
    print(f"PROFILE_OUTPUT {raw_path}", flush=True)


if __name__ == "__main__":
    main()
