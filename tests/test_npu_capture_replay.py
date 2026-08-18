from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from BeingH.npu_capture_replay import (
    NPUCaptureProcessUnhealthyError,
    NPUActionSuffixGraphRunner,
    NPUFixedBaselineFlowModule,
    TensorSpec,
    flatten_prefix_cache,
    resolve_npu_capture_replay_route,
)


class _DummyActionSuffix(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.forward_calls = 0

    def forward_action_with_prefix_cache(
        self,
        *,
        action_sequence: torch.Tensor,
        packed_position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_cache: dict,
    ) -> torch.Tensor:
        del packed_position_ids, attention_mask
        self.forward_calls += 1
        return action_sequence * self.scale + prefix_cache["layers"][0][0].mean()


class _WorkspaceAwareSuffix(torch.nn.Module):
    def forward_action_with_prefix_cache(
        self,
        *,
        action_sequence: torch.Tensor,
        packed_position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_cache: dict,
    ) -> torch.Tensor:
        del packed_position_ids, attention_mask
        prefix_key = prefix_cache["layers"][0][0]
        full_key = prefix_cache["full_layers"][0][0]
        self.saw_shared_prefix_storage = (
            prefix_key.data_ptr() == full_key.data_ptr()
        )
        return action_sequence + full_key.sum()


class _IdentityActionEncoder(torch.nn.Module):
    def forward(self, actions, timesteps):
        del timesteps
        return actions


class _IdentityActionDecoder(torch.nn.Module):
    def forward(self, hidden):
        return hidden


class _FakeNpuTensor:
    device = SimpleNamespace(type="npu")

    def clone(self):
        return self


class _FakeGraph:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class _FakeStream:
    def wait_stream(self, stream) -> None:
        del stream


class _FailingCaptureContext:
    def __enter__(self):
        raise RuntimeError("capture_end failed")

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback


class _FakeNpuRuntime:
    def __init__(self) -> None:
        self.graph_instance = _FakeGraph()
        self.synchronize_calls = 0
        self.npu_graph_calls = 0

    def Stream(self):
        return _FakeStream()

    def current_stream(self):
        return _FakeStream()

    def stream(self, stream):
        del stream
        return nullcontext()

    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def NPUGraph(self):
        self.npu_graph_calls += 1
        return self.graph_instance

    def graph(self, graph):
        del graph
        return _FailingCaptureContext()


class _CaptureHarnessRunner(NPUActionSuffixGraphRunner):
    def _flat_forward(
        self,
        action_sequence,
        packed_position_ids,
        attention_mask,
        prefix_kv,
        full_kv=(),
    ):
        del packed_position_ids, attention_mask, prefix_kv, full_kv
        return action_sequence


class _WarmupFailureRunner(_CaptureHarnessRunner):
    def _flat_forward(
        self,
        action_sequence,
        packed_position_ids,
        attention_mask,
        prefix_kv,
        full_kv=(),
    ):
        del action_sequence, packed_position_ids, attention_mask, prefix_kv, full_kv
        raise RuntimeError("graph warmup failed")


class _PostCaptureFailureRunner(NPUActionSuffixGraphRunner):
    def _capture(self, **kwargs):
        del kwargs
        raise self._mark_process_unhealthy(
            "capture", RuntimeError("capture failed after begin")
        )


class _ReplayFailureRunner(NPUActionSuffixGraphRunner):
    def _initialize_session(self, session, action_sequence) -> None:
        del action_sequence
        session._entry = object()

    def _replay(self, entry, action_sequence):
        del entry, action_sequence
        raise RuntimeError("graph replay failed")


class _BindFailureRunner(NPUActionSuffixGraphRunner):
    def _bind_prefix(self, entry, **kwargs) -> None:
        del entry, kwargs
        raise RuntimeError("static prefix bind failed")


class _BoundedCacheHarnessRunner(NPUActionSuffixGraphRunner):
    def _capture(self, **kwargs):
        self.capture_count += 1
        return SimpleNamespace(key=kwargs["key"], graph=_FakeGraph())

    def _bind_prefix(self, entry, **kwargs) -> None:
        del entry, kwargs
        self.bind_count += 1

    def _replay(self, entry, action_sequence):
        del entry
        self.replay_count += 1
        return action_sequence


def _prefix_cache(value: float = 1.0) -> dict:
    key = torch.full((3, 2), value)
    value_tensor = torch.full((3, 2), value + 1)
    return {"layers": [(key, value_tensor)], "prefix_length": 3}


class NpuCaptureReplayRoutingTest(unittest.TestCase):
    def eligible_route(self, **overrides):
        arguments = {
            "enabled": True,
            "device_type": "npu",
            "training": False,
            "grad_enabled": False,
            "static_prefix_cache": True,
            "has_static_prefix_context": True,
            "parallel_inference": False,
            "use_rtc": False,
            "attention_mode": "causal",
            "use_expert": True,
            "flow_steps": 4,
            "use_mpg": True,
            "mpg_refinement_iters": 1,
        }
        arguments.update(overrides)
        return resolve_npu_capture_replay_route(**arguments)

    def test_fixed_single_request_route_is_eligible(self) -> None:
        route = self.eligible_route()
        self.assertTrue(route.eligible)
        self.assertEqual(route.reason, "eligible")

    def test_default_off_routes_to_eager(self) -> None:
        route = self.eligible_route(enabled=False)
        self.assertFalse(route.eligible)
        self.assertEqual(route.reason, "disabled")

    def test_parallel_and_rtc_routes_stay_eager(self) -> None:
        self.assertEqual(
            self.eligible_route(parallel_inference=True).reason,
            "parallel_inference",
        )
        self.assertEqual(self.eligible_route(use_rtc=True).reason, "rtc_enabled")

    def test_fixed_flow_and_mpg_configuration_is_required(self) -> None:
        self.assertEqual(
            self.eligible_route(flow_steps=3).reason,
            "requires_four_flow_steps",
        )
        self.assertEqual(
            self.eligible_route(mpg_refinement_iters=2).reason,
            "requires_one_mpg_refinement",
        )

    def test_adaptive_flow_graph_replay_is_explicit(self) -> None:
        self.assertEqual(
            self.eligible_route(adaptive_flow_steps=True).reason,
            "adaptive_flow_replay_disabled",
        )
        route = self.eligible_route(
            adaptive_flow_steps=True,
            allow_adaptive_flow_replay=True,
        )
        self.assertTrue(route.eligible)

    def test_tensor_spec_distinguishes_shape_and_stride(self) -> None:
        contiguous = torch.zeros(2, 3)
        transposed = torch.zeros(3, 2).transpose(0, 1)
        self.assertEqual(contiguous.shape, transposed.shape)
        self.assertNotEqual(
            TensorSpec.from_tensor(contiguous),
            TensorSpec.from_tensor(transposed),
        )

    def test_prefix_cache_flattens_key_value_pairs_only(self) -> None:
        cache = _prefix_cache()
        flattened = flatten_prefix_cache(cache)
        self.assertEqual(len(flattened), 2)
        self.assertIs(flattened[0], cache["layers"][0][0])
        self.assertIs(flattened[1], cache["layers"][0][1])

    def test_fixed_baseline_flow_module_matches_manual_euler_steps(self) -> None:
        language = _DummyActionSuffix()
        module = NPUFixedBaselineFlowModule(
            action_encoder=_IdentityActionEncoder(),
            language_model=language,
            action_decoder=_IdentityActionDecoder(),
            action_chunk_length=4,
            num_steps=4,
            num_timestep_buckets=1000,
        )
        actions = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        cache = _prefix_cache(1.0)
        result = module.forward_action_with_prefix_cache(
            action_sequence=actions,
            packed_position_ids=torch.arange(4),
            attention_mask=torch.zeros(4, 8),
            prefix_cache=cache,
        )
        expected = actions
        for _ in range(4):
            velocity = expected * language.scale + 1.0
            expected = expected + 0.25 * velocity
        self.assertTrue(torch.equal(result, expected))

    def test_graph_cache_requires_positive_capacity(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_entries must be positive"):
            NPUActionSuffixGraphRunner(_DummyActionSuffix(), max_entries=0)

    def test_kv_workspace_is_explicit_and_reported(self) -> None:
        baseline = NPUActionSuffixGraphRunner(_DummyActionSuffix())
        candidate = NPUActionSuffixGraphRunner(
            _DummyActionSuffix(), enable_kv_workspace=True
        )
        self.assertFalse(baseline.enable_kv_workspace)
        self.assertFalse(baseline.stats()["enable_kv_workspace"])
        self.assertTrue(candidate.enable_kv_workspace)
        self.assertTrue(candidate.stats()["enable_kv_workspace"])

    def test_flat_forward_exposes_graph_owned_full_kv_layers(self) -> None:
        target = _WorkspaceAwareSuffix()
        runner = NPUActionSuffixGraphRunner(
            target, enable_kv_workspace=True
        )
        full_key = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        full_value = full_key + 100
        prefix_key = full_key[:3]
        prefix_value = full_value[:3]
        result = runner._flat_forward(
            torch.ones(2, 2),
            torch.arange(2),
            torch.zeros(2, 5),
            (prefix_key, prefix_value),
            (full_key, full_value),
        )
        self.assertTrue(target.saw_shared_prefix_storage)
        self.assertTrue(torch.equal(result, torch.ones(2, 2) + full_key.sum()))

    def test_bounded_cache_retains_multiple_shapes_and_falls_back_when_full(
        self,
    ) -> None:
        target = _DummyActionSuffix()
        runner = _BoundedCacheHarnessRunner(target, max_entries=2)
        action = torch.ones(2, 4)

        def invoke(position_length: int) -> torch.Tensor:
            request = {
                "prefix_cache": _prefix_cache(),
                "packed_position_ids": torch.arange(position_length),
                "attention_mask": torch.zeros(position_length, 5),
                "feature_flags": {"opt04": True},
            }
            with runner.try_open_request(**request) as session:
                return session.forward(action)

        invoke(2)
        invoke(3)
        invoke(2)
        invoke(4)

        self.assertEqual(runner.entry_count, 2)
        self.assertEqual(runner.capture_count, 2)
        self.assertEqual(runner.cache_miss_count, 3)
        self.assertEqual(runner.cache_hit_count, 1)
        self.assertEqual(runner.cache_full_fallback_count, 1)
        self.assertEqual(runner.replay_count, 3)
        self.assertEqual(runner.eager_fallback_count, 1)
        self.assertEqual(runner.last_fallback_reason, "graph_cache_full")
        self.assertEqual(runner.stats()["max_entries"], 2)

        # Avoid issuing a real torch.npu synchronization from test teardown.
        runner._entries.clear()

    def test_frozen_cache_never_captures_a_new_shape(self) -> None:
        target = _DummyActionSuffix()
        runner = _BoundedCacheHarnessRunner(target, max_entries=2)
        action = torch.ones(2, 4)

        def invoke(position_length: int) -> None:
            request = {
                "prefix_cache": _prefix_cache(),
                "packed_position_ids": torch.arange(position_length),
                "attention_mask": torch.zeros(position_length, 5),
                "feature_flags": {"opt04": True},
            }
            with runner.try_open_request(**request) as session:
                session.forward(action)

        invoke(2)
        runner.freeze()
        invoke(3)

        self.assertEqual(runner.capture_count, 1)
        self.assertEqual(runner.entry_count, 1)
        self.assertEqual(runner.cache_frozen_fallback_count, 1)
        self.assertEqual(runner.eager_fallback_count, 1)
        self.assertEqual(
            runner.last_fallback_reason, "graph_cache_frozen_miss"
        )
        self.assertFalse(runner.stats()["capture_on_miss"])
        runner._entries.clear()

    def test_concurrent_request_falls_back_without_stealing_lock(self) -> None:
        runner = NPUActionSuffixGraphRunner(_DummyActionSuffix())
        request = {
            "prefix_cache": _prefix_cache(),
            "packed_position_ids": torch.arange(2),
            "attention_mask": torch.zeros(2, 5),
            "feature_flags": {"opt04": True},
        }
        first = runner.try_open_request(**request)
        second = runner.try_open_request(**request)
        self.assertIsNone(first.fallback_reason)
        self.assertEqual(second.fallback_reason, "concurrent_request")
        second.close()
        runner.close_active_request()
        self.assertTrue(first._closed)

        third = runner.try_open_request(**request)
        self.assertIsNone(third.fallback_reason)
        third.close()

    def test_pre_capture_failure_falls_back_and_is_logged_once(self) -> None:
        target = _DummyActionSuffix()
        runner = NPUActionSuffixGraphRunner(target)
        request = {
            "prefix_cache": _prefix_cache(),
            "packed_position_ids": torch.arange(2),
            "attention_mask": torch.zeros(2, 5),
            "feature_flags": {"opt04": True},
        }
        action = torch.ones(2, 4)

        with self.assertLogs("BeingH.npu_capture_replay", level="WARNING") as logs:
            with runner.try_open_request(**request) as session:
                first = session.forward(action)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("requires NPU tensors", logs.output[0])
        self.assertEqual(len(runner.failure_reasons), 1)
        self.assertFalse(runner.unhealthy)

        with self.assertNoLogs("BeingH.npu_capture_replay", level="WARNING"):
            with runner.try_open_request(**request) as session:
                second = session.forward(action)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(target.forward_calls, 2)

    def test_post_capture_failure_is_fail_closed(self) -> None:
        target = _DummyActionSuffix()
        runner = _CaptureHarnessRunner(target)
        runtime = _FakeNpuRuntime()
        request = {
            "prefix_cache": _prefix_cache(),
            "packed_position_ids": torch.arange(2),
            "attention_mask": torch.zeros(2, 5),
            "feature_flags": {"opt04": True},
        }

        with self.assertLogs(
            "BeingH.npu_capture_replay", level="CRITICAL"
        ) as logs:
            session = runner.try_open_request(**request)
            with patch.object(torch, "npu", runtime, create=True):
                with self.assertRaisesRegex(
                    NPUCaptureProcessUnhealthyError,
                    "restart the worker",
                ):
                    runner._capture(
                        key=None,
                        action_sequence=_FakeNpuTensor(),
                        packed_position_ids=_FakeNpuTensor(),
                        attention_mask=_FakeNpuTensor(),
                        prefix_kv=(_FakeNpuTensor(),),
                    )
            self.assertTrue(session.unhealthy)
            with self.assertRaises(NPUCaptureProcessUnhealthyError):
                session.forward(torch.ones(2, 4))
            session.close()

        self.assertEqual(len(logs.output), 1)
        self.assertTrue(runner.unhealthy)
        self.assertIn("capture_end failed", runner.unhealthy_reason)
        self.assertEqual(target.forward_calls, 0)
        self.assertEqual(runtime.synchronize_calls, 1)
        self.assertEqual(runtime.graph_instance.reset_calls, 0)

        with self.assertRaisesRegex(
            NPUCaptureProcessUnhealthyError,
            "restart the worker",
        ):
            runner.try_open_request(**request)

    def test_graph_warmup_failure_is_fail_closed(self) -> None:
        target = _DummyActionSuffix()
        runner = _WarmupFailureRunner(target)
        runtime = _FakeNpuRuntime()
        request = {
            "prefix_cache": _prefix_cache(),
            "packed_position_ids": torch.arange(2),
            "attention_mask": torch.zeros(2, 5),
            "feature_flags": {"opt04": True},
        }
        session = runner.try_open_request(**request)

        with self.assertLogs(
            "BeingH.npu_capture_replay", level="CRITICAL"
        ):
            with patch.object(torch, "npu", runtime, create=True):
                with self.assertRaisesRegex(
                    NPUCaptureProcessUnhealthyError,
                    "restart the worker",
                ):
                    runner._capture(
                        key=None,
                        action_sequence=_FakeNpuTensor(),
                        packed_position_ids=_FakeNpuTensor(),
                        attention_mask=_FakeNpuTensor(),
                        prefix_kv=(_FakeNpuTensor(),),
                    )

        self.assertTrue(runner.unhealthy)
        self.assertTrue(session.unhealthy)
        self.assertIn("graph warmup failed", runner.unhealthy_reason)
        self.assertEqual(runtime.synchronize_calls, 0)
        self.assertEqual(runtime.npu_graph_calls, 0)
        self.assertEqual(runtime.graph_instance.reset_calls, 0)
        self.assertEqual(target.forward_calls, 0)
        with self.assertRaises(NPUCaptureProcessUnhealthyError):
            session.forward(torch.ones(2, 4))
        session.close()

        with self.assertRaises(NPUCaptureProcessUnhealthyError):
            runner.try_open_request(**request)

    def test_session_does_not_swallow_post_capture_error(self) -> None:
        target = _DummyActionSuffix()
        runner = _PostCaptureFailureRunner(target)
        request = {
            "prefix_cache": _prefix_cache(),
            "packed_position_ids": torch.arange(2),
            "attention_mask": torch.zeros(2, 5),
            "feature_flags": {"opt04": True},
        }

        with self.assertLogs(
            "BeingH.npu_capture_replay", level="CRITICAL"
        ):
            with runner.try_open_request(**request) as session:
                with self.assertRaisesRegex(
                    NPUCaptureProcessUnhealthyError,
                    "restart the worker",
                ):
                    session.forward(torch.ones(2, 4))

        self.assertTrue(runner.unhealthy)
        self.assertEqual(target.forward_calls, 0)

    def test_replay_failure_is_fail_closed_and_rejects_later_requests(
        self,
    ) -> None:
        target = _DummyActionSuffix()
        runner = _ReplayFailureRunner(target)
        request = {
            "prefix_cache": _prefix_cache(),
            "packed_position_ids": torch.arange(2),
            "attention_mask": torch.zeros(2, 5),
            "feature_flags": {"opt04": True},
        }

        with self.assertLogs(
            "BeingH.npu_capture_replay", level="CRITICAL"
        ):
            with runner.try_open_request(**request) as session:
                with self.assertRaisesRegex(
                    NPUCaptureProcessUnhealthyError,
                    "restart the worker",
                ):
                    session.forward(torch.ones(2, 4))

        self.assertTrue(runner.unhealthy)
        self.assertIn("graph replay failed", runner.unhealthy_reason)
        self.assertEqual(target.forward_calls, 0)

        with self.assertRaises(NPUCaptureProcessUnhealthyError):
            runner.try_open_request(**request)

    def test_prefix_bind_failure_is_fail_closed_without_graph_reset(
        self,
    ) -> None:
        target = _DummyActionSuffix()
        runner = _BindFailureRunner(target)
        prefix_cache = _prefix_cache()
        position_ids = torch.arange(2)
        attention_mask = torch.zeros(2, 5)
        feature_flags = {"opt04": True}
        action = torch.ones(2, 4)
        key = runner._make_key(
            action,
            position_ids,
            attention_mask,
            flatten_prefix_cache(prefix_cache),
            feature_flags,
        )
        graph = _FakeGraph()
        runner._entries[key] = SimpleNamespace(key=key, graph=graph)
        request = {
            "prefix_cache": prefix_cache,
            "packed_position_ids": position_ids,
            "attention_mask": attention_mask,
            "feature_flags": feature_flags,
        }

        with self.assertLogs(
            "BeingH.npu_capture_replay", level="CRITICAL"
        ):
            with runner.try_open_request(**request) as session:
                with self.assertRaisesRegex(
                    NPUCaptureProcessUnhealthyError,
                    "restart the worker",
                ):
                    session.forward(action)

        self.assertTrue(runner.unhealthy)
        self.assertIn("static prefix bind failed", runner.unhealthy_reason)
        self.assertEqual(graph.reset_calls, 0)
        self.assertEqual(target.forward_calls, 0)
        self.assertEqual(runner.entry_count, 1)

        with self.assertRaises(NPUCaptureProcessUnhealthyError):
            runner.try_open_request(**request)


if __name__ == "__main__":
    unittest.main()
