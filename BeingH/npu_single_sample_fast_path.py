"""Routing helpers for the OPT-03 NPU single-sample causal fast path."""

from __future__ import annotations

from typing import Iterable


VALID_NPU_SINGLE_SAMPLE_FAST_PATH_MODES = ("off", "auto", "force")


def real_sample_lens(
    sample_lens: Iterable[int], packed_seq_len: int
) -> list[int]:
    """Return real packed samples, ignoring trailing padding-only samples."""
    real_lens = []
    real_total = 0
    for sample_len in sample_lens:
        if real_total == packed_seq_len:
            break
        if sample_len <= 0:
            raise ValueError(f"sample length must be positive, got {sample_len}")
        if real_total + sample_len > packed_seq_len:
            raise ValueError(
                "sample lengths cross packed sequence boundary: "
                f"{real_total} + {sample_len} > {packed_seq_len}"
            )
        real_lens.append(sample_len)
        real_total += sample_len
    if real_total != packed_seq_len:
        raise ValueError(
            f"sample lengths sum to {real_total}, expected {packed_seq_len}"
        )
    return real_lens


def resolve_npu_single_sample_fast_path(
    mode: str,
    sample_lens: Iterable[int],
    packed_seq_len: int,
    *,
    parallel_inference: bool,
) -> bool:
    """Resolve OPT-03 routing without changing the packed fallback path."""
    if mode not in VALID_NPU_SINGLE_SAMPLE_FAST_PATH_MODES:
        choices = ", ".join(VALID_NPU_SINGLE_SAMPLE_FAST_PATH_MODES)
        raise ValueError(
            f"invalid NPU single-sample fast-path mode {mode!r}; "
            f"expected one of: {choices}"
        )
    if mode == "off":
        return False

    real_count = len(real_sample_lens(sample_lens, packed_seq_len))
    eligible = real_count == 1 and not parallel_inference
    if mode == "auto":
        return eligible
    if not eligible:
        raise RuntimeError(
            "OPT-03 force mode requires exactly one real sample and "
            "non-parallel inference; "
            f"real_sample_count={real_count}, "
            f"parallel_inference={parallel_inference}"
        )
    return True
