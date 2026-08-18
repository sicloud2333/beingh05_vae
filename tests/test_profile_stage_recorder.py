import unittest
from unittest import mock

import profile_npu_pipeline as profiler


class FakeEvent:
    clock = 0

    def __init__(self) -> None:
        self.timestamp = None

    def record(self) -> None:
        self.timestamp = FakeEvent.clock
        FakeEvent.clock += 1

    def elapsed_time(self, other: "FakeEvent") -> float:
        return float(other.timestamp - self.timestamp)


class StageRecorderTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeEvent.clock = 0

    def test_describe_reports_p99(self) -> None:
        summary = profiler.describe([float(value) for value in range(1, 101)])

        self.assertEqual(summary["p95_ms"], 95.0)
        self.assertEqual(summary["p99_ms"], 99.0)

    def test_event_mode_does_not_synchronize_inside_stage(self) -> None:
        recorder = profiler.StageRecorder(profiler.STAGE_TIMING_EVENT)
        recorder.enabled = True
        recorder.loop_context = {"mpg_iteration": 1, "flow_step": 2}

        with (
            mock.patch.object(
                profiler, "create_timing_event", side_effect=FakeEvent
            ),
            mock.patch.object(
                profiler,
                "synchronize",
                side_effect=AssertionError("unexpected stage synchronize"),
            ),
        ):
            self.assertEqual(recorder.timed_call("qwen", lambda: 7), 7)

        self.assertEqual(recorder.call_records, [])
        recorder.finalize_request()

        self.assertEqual(recorder.current["qwen"], [1.0])
        self.assertEqual(recorder.current_device["qwen"], [1.0])
        self.assertEqual(len(recorder.current_host_launch["qwen"]), 1)
        self.assertEqual(recorder.call_records[0]["label"], "mpg_refine_step2")
        self.assertEqual(
            recorder.call_records[0]["timing_source"], "device_event"
        )

    def test_event_mode_uses_host_time_for_pack_inputs(self) -> None:
        recorder = profiler.StageRecorder(profiler.STAGE_TIMING_EVENT)
        recorder.enabled = True
        with mock.patch.object(
            profiler, "create_timing_event", side_effect=FakeEvent
        ):
            recorder.timed_call("policy_pack_inputs", lambda: None)
        recorder.finalize_request()

        call = recorder.call_records[0]
        self.assertEqual(call["timing_source"], "host_wall")
        self.assertEqual(call["device_ms"], 1.0)
        self.assertEqual(call["ms"], call["host_launch_ms"])

    def test_event_loop_preserves_step_labels(self) -> None:
        recorder = profiler.StageRecorder(profiler.STAGE_TIMING_EVENT)
        recorder.enabled = True
        recorder.loop_context["mpg_iteration"] = 0

        with mock.patch.object(
            profiler, "create_timing_event", side_effect=FakeEvent
        ):
            self.assertEqual(
                list(recorder.loop_hook("flow_step", range(2))), [0, 1]
            )
        recorder.finalize_request()

        self.assertEqual(
            [call["label"] for call in recorder.call_records],
            ["baseline_step0", "baseline_step1"],
        )
        self.assertEqual(recorder.current[profiler.DENOISE_INNER_STAGE], [1.0, 1.0])

    def test_sync_mode_preserves_legacy_synchronization(self) -> None:
        recorder = profiler.StageRecorder(profiler.STAGE_TIMING_SYNC)
        recorder.enabled = True
        calls = []
        with mock.patch.object(
            profiler, "synchronize", side_effect=lambda: calls.append(1)
        ):
            self.assertEqual(recorder.timed_call("stage", lambda: 3), 3)

        self.assertEqual(len(calls), 2)
        self.assertEqual(
            recorder.call_records[0]["timing_source"], "synchronized_wall"
        )


if __name__ == "__main__":
    unittest.main()
