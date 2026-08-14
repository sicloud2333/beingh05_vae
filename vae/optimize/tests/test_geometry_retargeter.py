from __future__ import annotations

import unittest

import numpy as np

from optimize import GeometryRetargeter, GeometryRetargeterConfig


class GeometryRetargeterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.retargeter = GeometryRetargeter(
            device="cpu",
            config=GeometryRetargeterConfig(max_iterations=12),
        )

    def test_identity_is_exact(self) -> None:
        q = np.linspace(-0.1, 0.1, 22, dtype=np.float32)
        runtime = self.retargeter.runtimes["shadow_hand_right"]
        lower = runtime.q_lower[:22].cpu().numpy()
        upper = runtime.q_upper[:22].cpu().numpy()
        q = np.clip(q, lower, upper)
        result = self.retargeter.retarget(
            q,
            "shadow_hand_right",
            "shadow_hand_right",
        )
        np.testing.assert_allclose(result.target_q, q, atol=1e-7)
        self.assertEqual(result.geometry_rmse, 0.0)

    def test_cross_hand_is_finite_and_bounded(self) -> None:
        q = np.zeros(22, dtype=np.float32)
        result = self.retargeter.retarget(
            q,
            "shadow_hand_right",
            "gaia_hand_right",
            update_state=False,
        )
        runtime = self.retargeter.runtimes["gaia_hand_right"]
        lower = runtime.q_lower[:15].cpu().numpy()
        upper = runtime.q_upper[:15].cpu().numpy()
        self.assertEqual(result.target_q.shape, (15,))
        self.assertTrue(np.isfinite(result.target_q).all())
        self.assertTrue(np.all(result.target_q >= lower - 1e-6))
        self.assertTrue(np.all(result.target_q <= upper + 1e-6))
        self.assertTrue(np.isfinite(result.geometry_rmse))

    def test_projected_limits_match_target_dimension(self) -> None:
        source = np.full(22, 0.01, dtype=np.float32)
        target = self.retargeter.project_motion_limits(
            source,
            "shadow_hand_right",
            "gaia_hand_right",
        )
        self.assertEqual(target.shape, (15,))
        self.assertTrue(np.all(target > 0))

    def test_action_chunk_is_optimized_as_one_batch(self) -> None:
        chunk = np.zeros((16, 22), dtype=np.float32)
        chunk[:, 0] = np.linspace(0.0, 0.2, 16, dtype=np.float32)
        result = self.retargeter.retarget_batch(
            chunk,
            "shadow_hand_right",
            "sharpa_hand_right",
            update_state=False,
        )
        runtime = self.retargeter.runtimes["sharpa_hand_right"]
        lower = runtime.q_lower[:22].cpu().numpy()
        upper = runtime.q_upper[:22].cpu().numpy()
        self.assertEqual(result.target_q.shape, (16, 22))
        self.assertEqual(result.per_frame_geometry_rmse.shape, (16,))
        self.assertEqual(result.batch_size, 16)
        self.assertTrue(np.isfinite(result.target_q).all())
        self.assertTrue(np.all(result.target_q >= lower[None] - 1e-6))
        self.assertTrue(np.all(result.target_q <= upper[None] + 1e-6))

    def test_underactuated_gaia_batch_maps_back_to_shadow(self) -> None:
        chunk = np.zeros((3, 15), dtype=np.float32)
        chunk[:, 0] = np.linspace(0.0, 0.1, 3, dtype=np.float32)
        result = self.retargeter.retarget_batch(
            chunk,
            "gaia_hand_right",
            "shadow_hand_right",
            update_state=False,
        )
        self.assertEqual(result.target_q.shape, (3, 22))
        self.assertTrue(np.isfinite(result.target_q).all())

    def test_stable_batch_accepts_consecutive_chunks(self) -> None:
        retargeter = GeometryRetargeter(
            device="cpu",
            config=GeometryRetargeterConfig(
                profile="stable", max_iterations=4
            ),
        )
        first = np.zeros((4, 22), dtype=np.float32)
        second = first.copy()
        second[:, 1] = 0.05
        result_a = retargeter.retarget_batch(
            first,
            "shadow_hand_right",
            "gaia_hand_right",
            stream="action_chunk",
        )
        result_b = retargeter.retarget_batch(
            second,
            "shadow_hand_right",
            "gaia_hand_right",
            stream="action_chunk",
        )
        self.assertEqual(result_a.target_q.shape, (4, 15))
        self.assertEqual(result_b.target_q.shape, (4, 15))
        self.assertTrue(np.isfinite(result_b.objective))


if __name__ == "__main__":
    unittest.main()
