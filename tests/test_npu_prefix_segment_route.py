from __future__ import annotations

import unittest

from BeingH.npu_prefix_segment_route import (
    PrefixSegmentRoute,
    build_prefix_segment_route,
    resolve_npu_prefix_segment_route,
)


def reconstruct_global(
    route: PrefixSegmentRoute,
    und_values: list[str],
    gen_values: list[str],
) -> list[str]:
    result: list[str] = []
    for segment in route.global_segments:
        source = und_values if segment.branch == "und" else gen_values
        result.extend(source[segment.source_start : segment.source_end])
    return result


def restore_branch(
    global_values: list[str],
    route_segments,
) -> list[str]:
    result: list[str] = []
    for segment in route_segments:
        result.extend(global_values[segment.global_start : segment.global_end])
    return result


class NpuPrefixSegmentRouteTest(unittest.TestCase):
    def test_artificial_interleaving_round_trips(self) -> None:
        und_indexes = [0, 2, 3, 6]
        gen_indexes = [1, 4, 5]
        route = build_prefix_segment_route(
            und_global_indexes=und_indexes,
            gen_global_indexes=gen_indexes,
            prefix_length=7,
        )
        und_values = [f"u{index}" for index in range(len(und_indexes))]
        gen_values = [f"g{index}" for index in range(len(gen_indexes))]
        global_values = reconstruct_global(route, und_values, gen_values)

        self.assertEqual(
            global_values,
            ["u0", "g0", "u1", "u2", "g1", "g2", "u3"],
        )
        self.assertEqual(
            restore_branch(global_values, route.und_segments),
            und_values,
        )
        self.assertEqual(
            restore_branch(global_values, route.gen_segments),
            gen_values,
        )

    def test_policy_five_segment_layout_with_variable_instruction(self) -> None:
        for instruction_length in (1, 7, 31):
            with self.subTest(instruction_length=instruction_length):
                text_before_vision = list(range(0, 3))
                vision_indexes = list(range(3, 7))
                text_before_state = [7, 8]
                state_indexes = [9]
                text_after_state = list(
                    range(10, 10 + instruction_length + 3)
                )
                text_indexes = (
                    text_before_vision
                    + text_before_state
                    + text_after_state
                )
                action_start = text_after_state[-1] + 1
                und_indexes = text_indexes + vision_indexes
                route = build_prefix_segment_route(
                    und_global_indexes=und_indexes,
                    gen_global_indexes=state_indexes,
                    prefix_length=action_start,
                )

                self.assertEqual(len(route.global_segments), 5)
                self.assertEqual(route.prefix_length, action_start)
                self.assertEqual(route.und_length, len(und_indexes))
                self.assertEqual(route.gen_length, 1)
                self.assertEqual(
                    reconstruct_global(
                        route,
                        [f"u{index}" for index in range(len(und_indexes))],
                        ["state"],
                    )[state_indexes[0]],
                    "state",
                )

    def test_invalid_routes_are_rejected(self) -> None:
        cases = (
            {
                "und_global_indexes": [0, 1],
                "gen_global_indexes": [1, 2],
                "prefix_length": 3,
            },
            {
                "und_global_indexes": [0],
                "gen_global_indexes": [2],
                "prefix_length": 3,
            },
            {
                "und_global_indexes": [0, 3],
                "gen_global_indexes": [1],
                "prefix_length": 3,
            },
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    build_prefix_segment_route(**arguments)

    def resolve(self, **overrides):
        route = build_prefix_segment_route(
            und_global_indexes=[0, 2],
            gen_global_indexes=[1],
            prefix_length=3,
        )
        arguments = {
            "enabled": True,
            "device_type": "npu",
            "training": False,
            "grad_enabled": False,
            "static_prefix_cache": True,
            "single_sample": True,
            "parallel_inference": False,
            "use_rtc": False,
            "attention_mode": "causal",
            "use_expert": True,
            "route": route,
        }
        arguments.update(overrides)
        return resolve_npu_prefix_segment_route(**arguments)

    def test_fixed_static_prefix_route_is_eligible(self) -> None:
        decision = self.resolve()
        self.assertTrue(decision.eligible)
        self.assertEqual(decision.reason, "eligible")

    def test_cuda_static_prefix_route_is_eligible(self) -> None:
        decision = self.resolve(device_type="cuda")
        self.assertTrue(decision.eligible)
        self.assertEqual(decision.reason, "eligible")

    def test_off_training_parallel_and_rtc_fall_back(self) -> None:
        self.assertEqual(self.resolve(enabled=False).reason, "disabled")
        self.assertEqual(
            self.resolve(device_type="cpu").reason,
            "requires_accelerator",
        )
        self.assertEqual(self.resolve(training=True).reason, "requires_eval")
        self.assertEqual(
            self.resolve(grad_enabled=True).reason,
            "requires_no_grad",
        )
        self.assertEqual(
            self.resolve(parallel_inference=True).reason,
            "parallel_inference",
        )
        self.assertEqual(self.resolve(use_rtc=True).reason, "rtc_enabled")

    def test_invalid_operating_modes_fall_back(self) -> None:
        self.assertEqual(
            self.resolve(static_prefix_cache=False).reason,
            "requires_opt01",
        )
        self.assertEqual(
            self.resolve(single_sample=False).reason,
            "requires_single_sample",
        )
        self.assertEqual(
            self.resolve(attention_mode="full").reason,
            "requires_causal_attention",
        )
        self.assertEqual(
            self.resolve(use_expert=False).reason,
            "requires_mot_expert",
        )
        self.assertEqual(self.resolve(route=None).reason, "route_unavailable")


if __name__ == "__main__":
    unittest.main()
