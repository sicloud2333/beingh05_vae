import unittest

import numpy as np

from BeingH.inference.action_ring_buffer import (
    ActionRingBuffer,
    ChunkConsumptionPlan,
    build_consumption_grid,
)


class ActionRingBufferTest(unittest.TestCase):
    def test_hard_prefix_locking_and_postfix_stitch(self):
        buffer = ActionRingBuffer(capacity=16, action_dim=2)
        first = np.arange(12, dtype=np.float32).reshape(6, 2)
        self.assertEqual(
            buffer.stitch(
                start_tick=0, actions=first, committed_prefix_end=-1
            ),
            6,
        )
        action, underflow = buffer.consume(0)
        np.testing.assert_array_equal(action, first[0])
        self.assertFalse(underflow)

        replacement = np.full((6, 2), 99, dtype=np.float32)
        written = buffer.stitch(
            start_tick=0, actions=replacement, committed_prefix_end=2
        )
        self.assertEqual(written, 3)
        np.testing.assert_array_equal(buffer.consume(1)[0], first[1])
        np.testing.assert_array_equal(buffer.consume(2)[0], first[2])
        np.testing.assert_array_equal(buffer.consume(3)[0], replacement[3])
        self.assertGreaterEqual(buffer.metrics().prefix_violations, 2)

    def test_underflow_repeats_last_safe_action(self):
        buffer = ActionRingBuffer(capacity=4, action_dim=1)
        buffer.stitch(
            start_tick=0,
            actions=np.array([[3.0]], dtype=np.float32),
            committed_prefix_end=-1,
        )
        self.assertEqual(buffer.consume(0)[0].item(), 3.0)
        action, underflow = buffer.consume(1)
        self.assertTrue(underflow)
        self.assertEqual(action.item(), 3.0)

    def test_capacity_preserves_nearest_actions(self):
        buffer = ActionRingBuffer(capacity=3, action_dim=1)
        buffer.stitch(
            start_tick=0,
            actions=np.arange(6, dtype=np.float32).reshape(6, 1),
            committed_prefix_end=-1,
        )
        for tick in range(3):
            action, underflow = buffer.consume(tick)
            self.assertFalse(underflow)
            self.assertEqual(action.item(), float(tick))

    def test_chunk_consumption_plan_keeps_model_shape_fixed(self):
        plan = ChunkConsumptionPlan(
            control_hz=20,
            model_chunk_length=16,
            consume_length=8,
            p95_latency_ms=140,
            safety_ticks=1,
        )
        self.assertEqual(plan.latency_commitment_ticks, 4)
        self.assertEqual(plan.request_rate_hz, 2.5)
        self.assertTrue(plan.should_request(4))
        self.assertFalse(plan.should_request(5))

    def test_grid(self):
        grid = build_consumption_grid(
            control_rates_hz=[10, 20],
            consume_lengths=[4, 8],
            model_chunk_length=16,
            p95_latency_ms=140,
        )
        self.assertEqual(len(grid), 4)


if __name__ == "__main__":
    unittest.main()
