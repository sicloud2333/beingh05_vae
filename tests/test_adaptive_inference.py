import unittest

import torch

from BeingH.adaptive_inference import (
    euler_extrapolate_remaining,
    relative_velocity_residual,
    should_skip_mpg_refinement,
    should_terminate_flow,
)


class AdaptiveInferenceTest(unittest.TestCase):
    def test_relative_velocity_residual(self):
        actions = torch.tensor([2.0, -2.0])
        velocity = torch.tensor([0.5, -0.5])
        self.assertAlmostEqual(
            relative_velocity_residual(actions, velocity).item(), 0.25
        )

    def test_flow_requires_minimum_steps_and_threshold(self):
        self.assertFalse(
            should_terminate_flow(
                completed_steps=1,
                min_steps=2,
                residual=0.01,
                threshold=0.02,
            )
        )
        self.assertTrue(
            should_terminate_flow(
                completed_steps=2,
                min_steps=2,
                residual=0.01,
                threshold=0.02,
            )
        )

    def test_euler_extrapolation_finishes_remaining_interval(self):
        actions = torch.tensor([1.0])
        velocity = torch.tensor([2.0])
        result = euler_extrapolate_remaining(
            actions, velocity, dt=0.25, remaining_steps=2
        )
        torch.testing.assert_close(result, torch.tensor([2.0]))

    def test_mpg_skip_threshold(self):
        self.assertTrue(should_skip_mpg_refinement(gate=0.1, threshold=0.2))
        self.assertFalse(should_skip_mpg_refinement(gate=0.3, threshold=0.2))


if __name__ == "__main__":
    unittest.main()
