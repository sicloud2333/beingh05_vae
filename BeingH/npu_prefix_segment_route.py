"""Validated CPU routing for the opt-in OPT-05 static-prefix path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence


PrefixBranch = Literal["und", "gen"]


@dataclass(frozen=True)
class PrefixSegment:
    """One contiguous mapping between branch-major and global token order."""

    branch: PrefixBranch
    source_start: int
    source_end: int
    global_start: int
    global_end: int

    @property
    def length(self) -> int:
        return self.source_end - self.source_start


@dataclass(frozen=True)
class PrefixSegmentRoute:
    """Bidirectional contiguous-segment route for one static prefix."""

    prefix_length: int
    und_length: int
    gen_length: int
    global_segments: tuple[PrefixSegment, ...]
    und_segments: tuple[PrefixSegment, ...]
    gen_segments: tuple[PrefixSegment, ...]

    def matches(self, *, prefix_length: int, und_length: int, gen_length: int) -> bool:
        return (
            self.prefix_length == prefix_length
            and self.und_length == und_length
            and self.gen_length == gen_length
        )


@dataclass(frozen=True)
class PrefixSegmentRouteDecision:
    eligible: bool
    reason: str


def _validate_branch_source_coverage(
    branch: PrefixBranch,
    segments: Sequence[PrefixSegment],
    expected_length: int,
) -> None:
    next_source = 0
    for segment in segments:
        if segment.branch != branch:
            raise ValueError(f"{branch} route contains a {segment.branch} segment")
        if segment.source_start != next_source:
            raise ValueError(
                f"{branch} source coverage has a gap or overlap at {next_source}"
            )
        if segment.length <= 0 or segment.length != (
            segment.global_end - segment.global_start
        ):
            raise ValueError(f"{branch} route contains an invalid segment")
        next_source = segment.source_end
    if next_source != expected_length:
        raise ValueError(
            f"{branch} source coverage ended at {next_source}, "
            f"expected {expected_length}"
        )


def build_prefix_segment_route(
    *,
    und_global_indexes: Sequence[int],
    gen_global_indexes: Sequence[int],
    prefix_length: int,
) -> PrefixSegmentRoute:
    """Build a complete, non-overlapping route without device tensor reads."""
    if prefix_length <= 0:
        raise ValueError("prefix length must be positive")

    owners: list[Optional[tuple[PrefixBranch, int]]] = [None] * prefix_length
    branch_indexes = (
        ("und", und_global_indexes),
        ("gen", gen_global_indexes),
    )
    for branch, global_indexes in branch_indexes:
        for source_index, raw_global_index in enumerate(global_indexes):
            global_index = int(raw_global_index)
            if global_index < 0 or global_index >= prefix_length:
                raise ValueError(
                    f"{branch} global index {global_index} is outside "
                    f"[0, {prefix_length})"
                )
            if owners[global_index] is not None:
                previous_branch, previous_source = owners[global_index]
                raise ValueError(
                    f"global index {global_index} is assigned by both "
                    f"{previous_branch}[{previous_source}] and "
                    f"{branch}[{source_index}]"
                )
            owners[global_index] = (branch, source_index)

    missing = [index for index, owner in enumerate(owners) if owner is None]
    if missing:
        preview = ", ".join(str(index) for index in missing[:8])
        raise ValueError(f"prefix route does not cover global indexes: {preview}")

    global_segments: list[PrefixSegment] = []
    segment_branch, segment_source_start = owners[0]  # type: ignore[misc]
    segment_global_start = 0
    previous_source = segment_source_start
    for global_index in range(1, prefix_length):
        branch, source_index = owners[global_index]  # type: ignore[misc]
        if branch == segment_branch and source_index == previous_source + 1:
            previous_source = source_index
            continue
        global_segments.append(
            PrefixSegment(
                branch=segment_branch,
                source_start=segment_source_start,
                source_end=previous_source + 1,
                global_start=segment_global_start,
                global_end=global_index,
            )
        )
        segment_branch = branch
        segment_source_start = source_index
        segment_global_start = global_index
        previous_source = source_index
    global_segments.append(
        PrefixSegment(
            branch=segment_branch,
            source_start=segment_source_start,
            source_end=previous_source + 1,
            global_start=segment_global_start,
            global_end=prefix_length,
        )
    )

    und_segments = tuple(
        sorted(
            (
                segment
                for segment in global_segments
                if segment.branch == "und"
            ),
            key=lambda segment: segment.source_start,
        )
    )
    gen_segments = tuple(
        sorted(
            (
                segment
                for segment in global_segments
                if segment.branch == "gen"
            ),
            key=lambda segment: segment.source_start,
        )
    )
    _validate_branch_source_coverage("und", und_segments, len(und_global_indexes))
    _validate_branch_source_coverage("gen", gen_segments, len(gen_global_indexes))

    return PrefixSegmentRoute(
        prefix_length=prefix_length,
        und_length=len(und_global_indexes),
        gen_length=len(gen_global_indexes),
        global_segments=tuple(global_segments),
        und_segments=und_segments,
        gen_segments=gen_segments,
    )


def resolve_npu_prefix_segment_route(
    *,
    enabled: bool,
    device_type: str,
    training: bool,
    grad_enabled: bool,
    static_prefix_cache: bool,
    single_sample: bool,
    parallel_inference: bool,
    use_rtc: bool,
    attention_mode: str,
    use_expert: bool,
    route: Optional[PrefixSegmentRoute],
) -> PrefixSegmentRouteDecision:
    """Return whether the narrowly scoped OPT-05 prefix route is safe."""
    checks = (
        (enabled, "disabled"),
        (device_type in {"npu", "cuda"}, "requires_accelerator"),
        (not training, "requires_eval"),
        (not grad_enabled, "requires_no_grad"),
        (static_prefix_cache, "requires_opt01"),
        (single_sample, "requires_single_sample"),
        (not parallel_inference, "parallel_inference"),
        (not use_rtc, "rtc_enabled"),
        (attention_mode == "causal", "requires_causal_attention"),
        (use_expert, "requires_mot_expert"),
        (route is not None, "route_unavailable"),
    )
    for passed, reason in checks:
        if not passed:
            return PrefixSegmentRouteDecision(False, reason)
    return PrefixSegmentRouteDecision(True, "eligible")
