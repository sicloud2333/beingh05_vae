"""Shape-keyed NPU capture/replay for the OPT-01 action suffix."""

from __future__ import annotations

import logging
import threading
import weakref
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import torch

from BeingH.npu_capture_replay_route import (
    CaptureReplayRoute,
    resolve_npu_capture_replay_route,
)

logger = logging.getLogger(__name__)

# Keep failed graph handles alive until interpreter teardown.  Releasing a
# handle can invoke runtime cleanup, which is unsafe after capture corruption.
_UNSAFE_NPU_GRAPH_QUARANTINE: list[Any] = []


class NPUCaptureProcessUnhealthyError(RuntimeError):
    """The current process must not issue more NPU work after a graph failure."""


@dataclass(frozen=True)
class TensorSpec:
    """Hashable tensor metadata that affects an NPU graph input surface."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device_type: str
    device_index: Optional[int]

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "TensorSpec":
        return cls(
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
            device_type=tensor.device.type,
            device_index=tensor.device.index,
        )


@dataclass(frozen=True)
class ActionSuffixGraphKey:
    """All state that must remain stable across graph replays."""

    inputs: tuple[TensorSpec, ...]
    feature_flags: tuple[tuple[str, Any], ...]
    module_state: tuple[tuple[int, int], ...]


def flatten_prefix_cache(prefix_cache: Mapping[str, Any]) -> tuple[torch.Tensor, ...]:
    """Flatten the OPT-01 per-layer ``(key, value)`` cache into tensor args."""
    layers = prefix_cache.get("layers")
    if not isinstance(layers, (list, tuple)) or not layers:
        raise ValueError("prefix cache must contain a non-empty layers sequence")

    flattened: list[torch.Tensor] = []
    for layer_index, layer_cache in enumerate(layers):
        if not isinstance(layer_cache, (list, tuple)) or len(layer_cache) != 2:
            raise ValueError(
                f"prefix cache layer {layer_index} must contain key and value"
            )
        for tensor in layer_cache:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"prefix cache layer {layer_index} contains a non-tensor"
                )
            flattened.append(tensor)
    return tuple(flattened)


def _feature_flag_tuple(flags: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    normalized = tuple(sorted(flags.items()))
    try:
        hash(normalized)
    except TypeError as error:
        raise TypeError("capture/replay feature flags must be hashable") from error
    return normalized


def _module_state(module: torch.nn.Module) -> tuple[tuple[int, int], ...]:
    """Track parameter addresses and in-place versions captured by the graph."""
    return tuple(
        (parameter.data_ptr(), parameter._version)
        for parameter in module.parameters()
    )


class _GraphEntry:
    def __init__(
        self,
        *,
        key: ActionSuffixGraphKey,
        graph: Any,
        static_action: torch.Tensor,
        static_position_ids: torch.Tensor,
        static_attention_mask: torch.Tensor,
        static_prefix_kv: tuple[torch.Tensor, ...],
        static_full_kv: tuple[torch.Tensor, ...],
        static_output: torch.Tensor,
    ) -> None:
        self.key = key
        self.graph = graph
        self.static_action = static_action
        self.static_position_ids = static_position_ids
        self.static_attention_mask = static_attention_mask
        self.static_prefix_kv = static_prefix_kv
        self.static_full_kv = static_full_kv
        self.static_output = static_output

    def reset(self) -> None:
        reset = getattr(self.graph, "reset", None)
        if reset is not None:
            reset()


class NPUFixedBaselineFlowModule(torch.nn.Module):
    """Four-step baseline flow body used as an NPUGraph capture target."""

    def __init__(
        self,
        *,
        action_encoder: torch.nn.Module,
        language_model: torch.nn.Module,
        action_decoder: torch.nn.Module,
        action_chunk_length: int,
        num_steps: int,
        num_timestep_buckets: int,
    ) -> None:
        super().__init__()
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        self.action_encoder = action_encoder
        self.language_model = language_model
        self.action_decoder = action_decoder
        self.action_chunk_length = action_chunk_length
        self.num_steps = num_steps
        self.num_timestep_buckets = num_timestep_buckets
        self.dt = 1.0 / float(num_steps)

    def forward_action_with_prefix_cache(
        self,
        *,
        action_sequence: torch.Tensor,
        packed_position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_cache: Mapping[str, Any],
    ) -> torch.Tensor:
        actions = action_sequence
        batch_size = actions.shape[0]
        for step in range(self.num_steps):
            timestep = int(
                step / float(self.num_steps) * self.num_timestep_buckets
            )
            timesteps = torch.full(
                (batch_size,), timestep, device=actions.device
            )
            action_features = self.action_encoder(actions, timesteps)
            flat_features = action_features.reshape(
                batch_size * self.action_chunk_length, -1
            ).to(prefix_cache["layers"][0][0].dtype)
            hidden = self.language_model.forward_action_with_prefix_cache(
                action_sequence=flat_features,
                packed_position_ids=packed_position_ids,
                attention_mask=attention_mask,
                prefix_cache=prefix_cache,
            )
            velocity = self.action_decoder(
                hidden.reshape(batch_size, self.action_chunk_length, -1)
            )
            actions = actions + self.dt * velocity
        return actions


class NPUActionSuffixGraphSession:
    """One serial request bound to a persistent graph entry or eager fallback."""

    def __init__(
        self,
        runner: "NPUActionSuffixGraphRunner",
        *,
        acquired: bool,
        prefix_cache: Mapping[str, Any],
        packed_position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        feature_flags: Mapping[str, Any],
        fallback_reason: Optional[str] = None,
    ) -> None:
        self._runner = runner
        self._acquired = acquired
        self._prefix_cache: Optional[Mapping[str, Any]] = prefix_cache
        self._packed_position_ids: Optional[torch.Tensor] = packed_position_ids
        self._attention_mask: Optional[torch.Tensor] = attention_mask
        self._feature_flags = dict(feature_flags)
        self._entry: Optional[_GraphEntry] = None
        self._initialized = False
        self._closed = False
        self.fallback_reason = fallback_reason

    @property
    def using_graph(self) -> bool:
        return self._entry is not None

    @property
    def unhealthy(self) -> bool:
        return self._runner.unhealthy

    def forward(self, action_sequence: torch.Tensor) -> torch.Tensor:
        self._runner.raise_if_unhealthy()
        if self._closed:
            raise RuntimeError("capture/replay session is already closed")
        if not self._initialized:
            self._runner._initialize_session(self, action_sequence)
            self._initialized = True

        if self._entry is None:
            return self._runner._eager_forward(
                action_sequence,
                self._require_position_ids(),
                self._require_attention_mask(),
                self._require_prefix_cache(),
            )

        try:
            return self._runner._replay(self._entry, action_sequence)
        except NPUCaptureProcessUnhealthyError:
            raise
        except Exception as error:
            entry = self._entry
            self._entry = None
            self._runner._quarantine_graph(
                getattr(entry, "graph", None)
            )
            raise self._runner._mark_process_unhealthy(
                "replay", error
            ) from error

    def _require_prefix_cache(self) -> Mapping[str, Any]:
        if self._prefix_cache is None:
            raise RuntimeError("capture/replay prefix cache was released")
        return self._prefix_cache

    def _require_position_ids(self) -> torch.Tensor:
        if self._packed_position_ids is None:
            raise RuntimeError("capture/replay position ids were released")
        return self._packed_position_ids

    def _require_attention_mask(self) -> torch.Tensor:
        if self._attention_mask is None:
            raise RuntimeError("capture/replay attention mask was released")
        return self._attention_mask

    def _release_request_inputs(self) -> None:
        self._prefix_cache = None
        self._packed_position_ids = None
        self._attention_mask = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._entry = None
        self._release_request_inputs()
        if self._acquired:
            self._runner._forget_active_request(self)
            self._runner._request_lock.release()
            self._acquired = False

    def __enter__(self) -> "NPUActionSuffixGraphSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class NPUActionSuffixGraphRunner:
    """Persistent bounded NPUGraph cache with per-request prefix binding.

    ``max_entries=1`` preserves the original fixed-shape behavior.  Larger
    values allow a serial single-environment worker to retain several shapes.
    A full cache falls back to eager execution instead of evicting/resetting a
    live graph on the request path.
    """

    def __init__(
        self,
        target_module: torch.nn.Module,
        *,
        warmup_iters: int = 2,
        max_entries: int = 1,
        enable_kv_workspace: bool = False,
    ) -> None:
        if warmup_iters < 1:
            raise ValueError("warmup_iters must be positive")
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._target_ref = weakref.ref(target_module)
        self._warmup_iters = warmup_iters
        self._max_entries = max_entries
        self._enable_kv_workspace = enable_kv_workspace
        self._entries: dict[ActionSuffixGraphKey, _GraphEntry] = {}
        self._failed_keys: dict[ActionSuffixGraphKey, str] = {}
        self._request_lock = threading.Lock()
        self._active_request = threading.local()
        self.last_fallback_reason: Optional[str] = None
        self._unhealthy_reason: Optional[str] = None
        self._capture_on_miss = True
        self.capture_count = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.cache_full_fallback_count = 0
        self.cache_frozen_fallback_count = 0
        self.bind_count = 0
        self.replay_count = 0
        self.eager_fallback_count = 0

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def enable_kv_workspace(self) -> bool:
        return self._enable_kv_workspace

    def stats(self) -> dict[str, Any]:
        """Return JSON-serializable cache/replay counters."""
        return {
            "max_entries": self._max_entries,
            "enable_kv_workspace": self._enable_kv_workspace,
            "entry_count": self.entry_count,
            "capture_count": self.capture_count,
            "cache_hit_count": self.cache_hit_count,
            "cache_miss_count": self.cache_miss_count,
            "cache_full_fallback_count": self.cache_full_fallback_count,
            "cache_frozen_fallback_count": self.cache_frozen_fallback_count,
            "capture_on_miss": self._capture_on_miss,
            "bind_count": self.bind_count,
            "replay_count": self.replay_count,
            "eager_fallback_count": self.eager_fallback_count,
            "failed_key_count": len(self._failed_keys),
            "last_fallback_reason": self.last_fallback_reason,
            "unhealthy": self.unhealthy,
            "unhealthy_reason": self.unhealthy_reason,
        }

    @property
    def failure_reasons(self) -> dict[ActionSuffixGraphKey, str]:
        return dict(self._failed_keys)

    @property
    def unhealthy(self) -> bool:
        return self._unhealthy_reason is not None

    @property
    def unhealthy_reason(self) -> Optional[str]:
        return self._unhealthy_reason

    def raise_if_unhealthy(self) -> None:
        if self._unhealthy_reason is not None:
            raise NPUCaptureProcessUnhealthyError(
                "NPU capture/replay process is unhealthy; restart the worker "
                f"before issuing more NPU work: {self._unhealthy_reason}"
            )

    def freeze(self) -> None:
        """Disable request-path capture after startup prewarming."""
        self.raise_if_unhealthy()
        with self._request_lock:
            self._capture_on_miss = False

    def unfreeze(self) -> None:
        """Allow controlled startup capture before serving requests."""
        self.raise_if_unhealthy()
        with self._request_lock:
            self._capture_on_miss = True

    def _mark_process_unhealthy(
        self,
        phase: str,
        error: Exception,
    ) -> NPUCaptureProcessUnhealthyError:
        reason = f"{phase}:{type(error).__name__}: {error}"
        if self._unhealthy_reason is None:
            self._unhealthy_reason = reason
            logger.critical(
                "OPT-04 NPU action-suffix %s failed after graph-related NPU "
                "work started; same-process NPU fallback is unsafe and the "
                "worker must restart: %s",
                phase,
                reason,
            )
        return NPUCaptureProcessUnhealthyError(
            "NPU capture/replay process is unhealthy; restart the worker "
            f"before issuing more NPU work: {self._unhealthy_reason}"
        )

    @staticmethod
    def _quarantine_graph(graph: Any) -> None:
        if graph is not None:
            _UNSAFE_NPU_GRAPH_QUARANTINE.append(graph)

    def try_open_request(
        self,
        *,
        prefix_cache: Mapping[str, Any],
        packed_position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        feature_flags: Mapping[str, Any],
    ) -> NPUActionSuffixGraphSession:
        self.raise_if_unhealthy()
        acquired = self._request_lock.acquire(blocking=False)
        reason = None if acquired else "concurrent_request"
        self.last_fallback_reason = reason
        session = NPUActionSuffixGraphSession(
            self,
            acquired=acquired,
            prefix_cache=prefix_cache,
            packed_position_ids=packed_position_ids,
            attention_mask=attention_mask,
            feature_flags=feature_flags,
            fallback_reason=reason,
        )
        if acquired:
            self._active_request.session = session
        return session

    def close_active_request(self) -> None:
        """Close the graph session owned by the calling request thread."""
        session = getattr(self._active_request, "session", None)
        if session is not None:
            session.close()

    def _forget_active_request(
        self, session: NPUActionSuffixGraphSession
    ) -> None:
        if getattr(self._active_request, "session", None) is session:
            del self._active_request.session

    def _target(self) -> torch.nn.Module:
        target = self._target_ref()
        if target is None:
            raise RuntimeError("capture/replay target module no longer exists")
        return target

    def _make_key(
        self,
        action_sequence: torch.Tensor,
        packed_position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_kv: Iterable[torch.Tensor],
        feature_flags: Mapping[str, Any],
    ) -> ActionSuffixGraphKey:
        target = self._target()
        inputs = (
            action_sequence,
            packed_position_ids,
            attention_mask,
            *tuple(prefix_kv),
        )
        return ActionSuffixGraphKey(
            inputs=tuple(TensorSpec.from_tensor(tensor) for tensor in inputs),
            feature_flags=_feature_flag_tuple(feature_flags),
            module_state=_module_state(target),
        )

    def _initialize_session(
        self,
        session: NPUActionSuffixGraphSession,
        action_sequence: torch.Tensor,
    ) -> None:
        if not session._acquired:
            return

        prefix_cache = session._require_prefix_cache()
        packed_position_ids = session._require_position_ids()
        attention_mask = session._require_attention_mask()
        prefix_kv = flatten_prefix_cache(prefix_cache)
        key = self._make_key(
            action_sequence,
            packed_position_ids,
            attention_mask,
            prefix_kv,
            session._feature_flags,
        )

        stale_keys = [
            entry_key
            for entry_key in self._entries
            if entry_key.module_state != key.module_state
        ]
        if stale_keys:
            self._clear_entries()

        if key in self._failed_keys:
            reason = self._failed_keys[key]
            session.fallback_reason = f"capture_failed:{reason}"
            self.last_fallback_reason = session.fallback_reason
            return

        entry = self._entries.get(key)
        if entry is None:
            self.cache_miss_count += 1
        else:
            self.cache_hit_count += 1

        if entry is None and not self._capture_on_miss:
            self.cache_frozen_fallback_count += 1
            session.fallback_reason = "graph_cache_frozen_miss"
            self.last_fallback_reason = session.fallback_reason
            return

        if entry is None and len(self._entries) >= self._max_entries:
            self.cache_full_fallback_count += 1
            session.fallback_reason = "graph_cache_full"
            self.last_fallback_reason = session.fallback_reason
            return

        if entry is None:
            try:
                entry = self._capture(
                    key=key,
                    action_sequence=action_sequence,
                    packed_position_ids=packed_position_ids,
                    attention_mask=attention_mask,
                    prefix_kv=prefix_kv,
                )
            except NPUCaptureProcessUnhealthyError:
                raise
            except Exception as error:
                reason = self._record_failure(key, "capture", error)
                session.fallback_reason = f"capture_failed:{reason}"
                self.last_fallback_reason = session.fallback_reason
                return
            self._entries[key] = entry
        else:
            try:
                self._bind_prefix(
                    entry,
                    packed_position_ids=packed_position_ids,
                    attention_mask=attention_mask,
                    prefix_kv=prefix_kv,
                )
            except NPUCaptureProcessUnhealthyError:
                raise
            except Exception as error:
                self._quarantine_graph(entry.graph)
                raise self._mark_process_unhealthy(
                    "prefix_bind", error
                ) from error

        session._entry = entry
        session.fallback_reason = None
        self.last_fallback_reason = None

    def _capture(
        self,
        *,
        key: ActionSuffixGraphKey,
        action_sequence: torch.Tensor,
        packed_position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_kv: tuple[torch.Tensor, ...],
    ) -> _GraphEntry:
        if action_sequence.device.type != "npu":
            raise RuntimeError("NPUGraph capture requires NPU tensors")

        # A policy request may run under torch.inference_mode().  Static graph
        # inputs must remain mutable across later requests, so create them as
        # normal no-grad tensors explicitly.
        graph = None
        graph_npu_work_started = False
        try:
            with torch.inference_mode(False), torch.no_grad():
                # Static input cloning is the first graph-specific NPU work.
                # Any failure from here through warmup, capture, or the final
                # synchronization makes same-process fallback unsafe.
                graph_npu_work_started = True
                static_action = action_sequence.clone()
                static_position_ids = packed_position_ids.clone()
                static_attention_mask = attention_mask.clone()
                if self._enable_kv_workspace:
                    static_prefix_kv_list = []
                    static_full_kv_list = []
                    suffix_length = action_sequence.shape[0]
                    for tensor in prefix_kv:
                        full_tensor = torch.empty(
                            (tensor.shape[0] + suffix_length, *tensor.shape[1:]),
                            dtype=tensor.dtype,
                            device=tensor.device,
                        )
                        prefix_view = full_tensor[: tensor.shape[0]]
                        prefix_view.copy_(tensor)
                        static_prefix_kv_list.append(prefix_view)
                        static_full_kv_list.append(full_tensor)
                    static_prefix_kv = tuple(static_prefix_kv_list)
                    static_full_kv = tuple(static_full_kv_list)
                else:
                    static_prefix_kv = tuple(
                        tensor.clone() for tensor in prefix_kv
                    )
                    static_full_kv = ()

                warmup_stream = torch.npu.Stream()
                warmup_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(warmup_stream):
                    for _ in range(self._warmup_iters):
                        self._flat_forward(
                            static_action,
                            static_position_ids,
                            static_attention_mask,
                            static_prefix_kv,
                            static_full_kv,
                        )
                torch.npu.current_stream().wait_stream(warmup_stream)
                torch.npu.synchronize()

                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    static_output = self._flat_forward(
                        static_action,
                        static_position_ids,
                        static_attention_mask,
                        static_prefix_kv,
                        static_full_kv,
                    )
                torch.npu.synchronize()
        except Exception as error:
            if graph_npu_work_started:
                self._quarantine_graph(graph)
                raise self._mark_process_unhealthy(
                    "graph_setup", error
                ) from error
            raise

        self.capture_count += 1
        return _GraphEntry(
            key=key,
            graph=graph,
            static_action=static_action,
            static_position_ids=static_position_ids,
            static_attention_mask=static_attention_mask,
            static_prefix_kv=static_prefix_kv,
            static_full_kv=static_full_kv,
            static_output=static_output,
        )

    def _flat_forward(
        self,
        action_sequence: torch.Tensor,
        packed_position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_kv: tuple[torch.Tensor, ...],
        full_kv: tuple[torch.Tensor, ...] = (),
    ) -> torch.Tensor:
        layers = [
            (prefix_kv[index], prefix_kv[index + 1])
            for index in range(0, len(prefix_kv), 2)
        ]
        prefix_cache: dict[str, Any] = {"layers": layers}
        if full_kv:
            prefix_cache["full_layers"] = [
                (full_kv[index], full_kv[index + 1])
                for index in range(0, len(full_kv), 2)
            ]
        return self._target().forward_action_with_prefix_cache(
            action_sequence=action_sequence,
            packed_position_ids=packed_position_ids,
            attention_mask=attention_mask,
            prefix_cache=prefix_cache,
        )

    def _eager_forward(
        self,
        action_sequence: torch.Tensor,
        packed_position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_cache: Mapping[str, Any],
    ) -> torch.Tensor:
        self.eager_fallback_count += 1
        return self._target().forward_action_with_prefix_cache(
            action_sequence=action_sequence,
            packed_position_ids=packed_position_ids,
            attention_mask=attention_mask,
            prefix_cache=prefix_cache,
        )

    @staticmethod
    def _copy_(destination: torch.Tensor, source: torch.Tensor) -> None:
        with torch.inference_mode(False), torch.no_grad():
            destination.copy_(source)

    def _bind_prefix(
        self,
        entry: _GraphEntry,
        *,
        packed_position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_kv: tuple[torch.Tensor, ...],
    ) -> None:
        with torch.inference_mode(False), torch.no_grad():
            entry.static_position_ids.copy_(packed_position_ids)
            entry.static_attention_mask.copy_(attention_mask)
            for destination, source in zip(
                entry.static_prefix_kv, prefix_kv, strict=True
            ):
                destination.copy_(source)
        self.bind_count += 1

    def _replay(
        self, entry: _GraphEntry, action_sequence: torch.Tensor
    ) -> torch.Tensor:
        self._copy_(entry.static_action, action_sequence)
        with torch.inference_mode(False), torch.no_grad():
            entry.graph.replay()
            self.replay_count += 1
            # NPUGraph outputs alias persistent graph memory.  Never expose that
            # storage to callers because the next replay overwrites it.
            return entry.static_output.clone()

    def _clear_entries(self) -> None:
        if self._entries:
            torch.npu.synchronize()
        for entry in self._entries.values():
            entry.reset()
        self._entries.clear()

    def _record_failure(
        self,
        key: ActionSuffixGraphKey,
        phase: str,
        error: Exception,
    ) -> str:
        reason = f"{phase}:{type(error).__name__}: {error}"
        if key not in self._failed_keys:
            self._failed_keys[key] = reason
            logger.warning(
                "OPT-04 NPU action-suffix %s failed; this graph key will "
                "use eager fallback: %s",
                phase,
                reason,
            )
        return self._failed_keys[key]

    def clear(self) -> None:
        """Release captured graphs and failure sentinels."""
        self.raise_if_unhealthy()
        with self._request_lock:
            self._clear_entries()
            self._failed_keys.clear()
            self.last_fallback_reason = None

    def __del__(self) -> None:
        if getattr(self, "_unhealthy_reason", None) is not None:
            # Graph/runtime cleanup can itself issue NPU work.  A worker in
            # this state is intentionally recoverable only by process restart.
            return
        try:
            self._clear_entries()
        except Exception:
            # Interpreter shutdown may already have torn down torch_npu.
            pass


class CUDAActionSuffixGraphRunner(NPUActionSuffixGraphRunner):
    """CUDA Graph equivalent of the bounded NPU action-suffix runner.

    Cache keys, startup-only capture, request serialization, prefix rebinding,
    graph-owned KV workspaces, and fail-closed handling are intentionally
    shared with the validated NPU implementation.  Only stream and graph
    primitives differ between the backends.
    """

    def _capture(
        self,
        *,
        key: ActionSuffixGraphKey,
        action_sequence: torch.Tensor,
        packed_position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_kv: tuple[torch.Tensor, ...],
    ) -> _GraphEntry:
        if action_sequence.device.type != "cuda":
            raise RuntimeError("CUDA Graph capture requires CUDA tensors")

        graph = None
        graph_cuda_work_started = False
        try:
            with torch.inference_mode(False), torch.no_grad():
                graph_cuda_work_started = True
                static_action = action_sequence.clone()
                static_position_ids = packed_position_ids.clone()
                static_attention_mask = attention_mask.clone()
                if self._enable_kv_workspace:
                    static_prefix_kv_list = []
                    static_full_kv_list = []
                    suffix_length = action_sequence.shape[0]
                    for tensor in prefix_kv:
                        full_tensor = torch.empty(
                            (tensor.shape[0] + suffix_length, *tensor.shape[1:]),
                            dtype=tensor.dtype,
                            device=tensor.device,
                        )
                        prefix_view = full_tensor[: tensor.shape[0]]
                        prefix_view.copy_(tensor)
                        static_prefix_kv_list.append(prefix_view)
                        static_full_kv_list.append(full_tensor)
                    static_prefix_kv = tuple(static_prefix_kv_list)
                    static_full_kv = tuple(static_full_kv_list)
                else:
                    static_prefix_kv = tuple(
                        tensor.clone() for tensor in prefix_kv
                    )
                    static_full_kv = ()

                warmup_stream = torch.cuda.Stream(device=action_sequence.device)
                current_stream = torch.cuda.current_stream(action_sequence.device)
                warmup_stream.wait_stream(current_stream)
                with torch.cuda.stream(warmup_stream):
                    for _ in range(self._warmup_iters):
                        self._flat_forward(
                            static_action,
                            static_position_ids,
                            static_attention_mask,
                            static_prefix_kv,
                            static_full_kv,
                        )
                current_stream.wait_stream(warmup_stream)
                torch.cuda.synchronize(action_sequence.device)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_output = self._flat_forward(
                        static_action,
                        static_position_ids,
                        static_attention_mask,
                        static_prefix_kv,
                        static_full_kv,
                    )
                torch.cuda.synchronize(action_sequence.device)
        except Exception as error:
            if graph_cuda_work_started:
                self._quarantine_graph(graph)
                raise self._mark_process_unhealthy(
                    "cuda_graph_setup", error
                ) from error
            raise

        self.capture_count += 1
        return _GraphEntry(
            key=key,
            graph=graph,
            static_action=static_action,
            static_position_ids=static_position_ids,
            static_attention_mask=static_attention_mask,
            static_prefix_kv=static_prefix_kv,
            static_full_kv=static_full_kv,
            static_output=static_output,
        )

    def _clear_entries(self) -> None:
        if self._entries:
            torch.cuda.synchronize()
        for entry in self._entries.values():
            entry.reset()
        self._entries.clear()
