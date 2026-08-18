"""Safety contract for a future cross-request system-prefix KV cache.

The current Qwen3 MoT implementation can cache a *complete request-local*
prefix and evaluate only the action suffix.  Reusing only the system turn
across requests additionally requires a decoder kernel that extends cached
per-layer K/V with the dynamic user/vision/state/instruction tokens.  This
module deliberately refuses to advertise that unavailable kernel while
providing deterministic cache keys and layout eligibility checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence


MIXED_PREFIX_EXTENSION_REQUIRED = "mixed_prefix_extension_kernel_required"


@dataclass(frozen=True)
class CrossRequestPrefixKey:
    """All inputs that may affect an immutable system-prefix KV."""

    model_fingerprint: str
    tokenizer_fingerprint: str
    system_token_ids: tuple[int, ...]
    device: str
    dtype: str
    software_fingerprint: str

    @property
    def digest(self) -> str:
        payload = {
            "model": self.model_fingerprint,
            "tokenizer": self.tokenizer_fingerprint,
            "tokens": self.system_token_ids,
            "device": self.device,
            "dtype": self.dtype,
            "software": self.software_fingerprint,
        }
        serialized = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()


@dataclass(frozen=True)
class CrossRequestPrefixPlan:
    """Result of checking whether the current decoder can safely reuse KV."""

    system_length: int
    dynamic_prefix_length: int
    action_length: int
    system_is_leading_contiguous: bool
    supported_by_current_decoder: bool
    reason: str


def build_prefix_key(
    *,
    model_fingerprint: str,
    tokenizer_fingerprint: str,
    system_token_ids: Sequence[int],
    device: str,
    dtype: str,
    software_fingerprint: str,
) -> CrossRequestPrefixKey:
    if not system_token_ids:
        raise ValueError("system_token_ids must not be empty")
    return CrossRequestPrefixKey(
        model_fingerprint=model_fingerprint,
        tokenizer_fingerprint=tokenizer_fingerprint,
        system_token_ids=tuple(int(token) for token in system_token_ids),
        device=device,
        dtype=dtype,
        software_fingerprint=software_fingerprint,
    )


def analyze_packed_layout(
    *,
    system_positions: Sequence[int],
    dynamic_prefix_positions: Sequence[int],
    action_positions: Sequence[int],
) -> CrossRequestPrefixPlan:
    """Check layout and return an explicit no-false-reuse decision.

    Being-H0.5 packs ``system -> dynamic content -> action``.  A leading
    contiguous system block is necessary for cross-request caching, but it is
    not sufficient: each decoder layer must extend the system K/V with the
    dynamic mixed MoT prefix before the action suffix is evaluated.
    """

    system = tuple(int(position) for position in system_positions)
    dynamic = tuple(int(position) for position in dynamic_prefix_positions)
    action = tuple(int(position) for position in action_positions)
    leading = bool(system) and system == tuple(range(len(system)))
    ordered = (
        leading
        and (not dynamic or min(dynamic) >= len(system))
        and (
            not action
            or min(action) >= len(system) + len(dynamic)
        )
    )
    if not ordered:
        return CrossRequestPrefixPlan(
            system_length=len(system),
            dynamic_prefix_length=len(dynamic),
            action_length=len(action),
            system_is_leading_contiguous=False,
            supported_by_current_decoder=False,
            reason="packed_layout_not_system_dynamic_action",
        )
    return CrossRequestPrefixPlan(
        system_length=len(system),
        dynamic_prefix_length=len(dynamic),
        action_length=len(action),
        system_is_leading_contiguous=True,
        supported_by_current_decoder=False,
        reason=MIXED_PREFIX_EXTENSION_REQUIRED,
    )


def require_supported(plan: CrossRequestPrefixPlan) -> None:
    """Fail closed instead of silently reusing numerically invalid K/V."""

    if not plan.supported_by_current_decoder:
        raise NotImplementedError(
            "Cross-request system KV reuse is unavailable: " + plan.reason
        )
