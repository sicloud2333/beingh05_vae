from __future__ import annotations

import unittest

from BeingH.npu_capture_replay_route import resolve_npu_capture_replay_route


class NpuCaptureReplayRoutingTest(unittest.TestCase):
    def resolve(self, **overrides):
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
        route = self.resolve()
        self.assertTrue(route.eligible)
        self.assertEqual(route.reason, "eligible")

    def test_cuda_uses_the_same_safety_contract(self) -> None:
        route = self.resolve(device_type="cuda")
        self.assertTrue(route.eligible)
        self.assertEqual(route.reason, "eligible")

    def test_cpu_is_rejected(self) -> None:
        route = self.resolve(device_type="cpu")
        self.assertFalse(route.eligible)
        self.assertEqual(route.reason, "requires_accelerator")

    def test_default_off_routes_to_eager(self) -> None:
        route = self.resolve(enabled=False)
        self.assertFalse(route.eligible)
        self.assertEqual(route.reason, "disabled")

    def test_parallel_and_rtc_routes_stay_eager(self) -> None:
        self.assertEqual(
            self.resolve(parallel_inference=True).reason,
            "parallel_inference",
        )
        self.assertEqual(self.resolve(use_rtc=True).reason, "rtc_enabled")

    def test_fixed_flow_and_mpg_configuration_is_required(self) -> None:
        self.assertEqual(
            self.resolve(flow_steps=3).reason,
            "requires_four_flow_steps",
        )
        self.assertEqual(
            self.resolve(mpg_refinement_iters=2).reason,
            "requires_one_mpg_refinement",
        )

    def test_adaptive_flow_requires_explicit_graph_replay_opt_in(self) -> None:
        self.assertEqual(
            self.resolve(adaptive_flow_steps=True).reason,
            "adaptive_flow_replay_disabled",
        )
        route = self.resolve(
            adaptive_flow_steps=True,
            allow_adaptive_flow_replay=True,
        )
        self.assertTrue(route.eligible)

    def test_adaptive_mpg_stays_ineligible(self) -> None:
        self.assertEqual(
            self.resolve(adaptive_mpg_refinement=True).reason,
            "adaptive_mpg_refinement",
        )


if __name__ == "__main__":
    unittest.main()
