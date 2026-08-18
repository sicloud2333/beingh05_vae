"""Pure routing policy for the opt-in OPT-04 accelerator graph path."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CaptureReplayRoute:
    eligible: bool
    reason: str


def resolve_npu_capture_replay_route(
    *,
    enabled: bool,
    device_type: str,
    training: bool,
    grad_enabled: bool,
    static_prefix_cache: bool,
    has_static_prefix_context: bool,
    parallel_inference: bool,
    use_rtc: bool,
    attention_mode: str,
    use_expert: bool,
    flow_steps: int,
    use_mpg: bool,
    mpg_refinement_iters: int,
    adaptive_flow_steps: bool = False,
    adaptive_mpg_refinement: bool = False,
    allow_adaptive_flow_replay: bool = False,
) -> CaptureReplayRoute:
    """Return whether the narrowly scoped OPT-04 graph path is safe."""
    checks = (
        (enabled, "disabled"),
        (device_type in {"npu", "cuda"}, "requires_accelerator"),
        (not training, "requires_eval"),
        (not grad_enabled, "requires_no_grad"),
        (static_prefix_cache, "requires_opt01"),
        (has_static_prefix_context, "missing_static_prefix_context"),
        (not parallel_inference, "parallel_inference"),
        (not use_rtc, "rtc_enabled"),
        (attention_mode == "causal", "requires_causal_attention"),
        (use_expert, "requires_mot_expert"),
        (
            not adaptive_mpg_refinement,
            "adaptive_mpg_refinement",
        ),
        (
            not adaptive_flow_steps or allow_adaptive_flow_replay,
            "adaptive_flow_replay_disabled",
        ),
        (flow_steps == 4, "requires_four_flow_steps"),
        (use_mpg, "requires_mpg"),
        (mpg_refinement_iters == 1, "requires_one_mpg_refinement"),
    )
    for passed, reason in checks:
        if not passed:
            return CaptureReplayRoute(False, reason)
    return CaptureReplayRoute(True, "eligible")
