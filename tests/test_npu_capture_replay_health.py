from __future__ import annotations

import unittest

from BeingH.inference.beingh_service import BeingHInferenceServer
from BeingH.npu_capture_replay import NPUCaptureProcessUnhealthyError


class _FailingPolicy:
    def __init__(self) -> None:
        self.calls = 0

    def get_action(self, observations):
        del observations
        self.calls += 1
        raise NPUCaptureProcessUnhealthyError("capture runtime is poisoned")


class _StatsRunner:
    def stats(self):
        return {"entry_count": 5, "unhealthy": False}


class NpuCaptureReplayHealthTest(unittest.TestCase):
    def test_server_exposes_read_only_capture_stats(self) -> None:
        policy = _FailingPolicy()
        policy.model = type(
            "Model",
            (),
            {
                "_npu_action_suffix_graph_runner": _StatsRunner(),
                "_npu_baseline_flow_graph_runner": None,
            },
        )()
        server = BeingHInferenceServer.__new__(BeingHInferenceServer)
        server.policy = policy
        server._npu_capture_unhealthy_reason = None

        stats = server._get_inference_stats()
        self.assertEqual(stats["npu_graph_cache"]["entry_count"], 5)
        self.assertIsNone(stats["npu_baseline_flow_graph_cache"])
        self.assertIsNone(stats["capture_unhealthy_reason"])

    def test_server_stops_and_rejects_after_fatal_capture_error(self) -> None:
        policy = _FailingPolicy()
        server = BeingHInferenceServer.__new__(BeingHInferenceServer)
        server.policy = policy
        server.running = True
        server._npu_capture_unhealthy_reason = None

        with self.assertRaisesRegex(
            NPUCaptureProcessUnhealthyError,
            "capture runtime is poisoned",
        ):
            server._get_action({})

        self.assertFalse(server.running)
        self.assertEqual(policy.calls, 1)

        with self.assertRaisesRegex(
            NPUCaptureProcessUnhealthyError,
            "restart the server worker",
        ):
            server._get_action({})
        self.assertEqual(policy.calls, 1)


if __name__ == "__main__":
    unittest.main()
