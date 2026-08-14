from __future__ import annotations

import unittest

import torch

from native_vae import NativeVAE


class NativeVAESmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vae = NativeVAE.from_pretrained(device="cpu")

    def test_all_hands_load_and_decode(self) -> None:
        expected_dims = {
            "shadow_hand_right": 22,
            "gaia_hand_right": 15,
            "sharpa_hand_right": 22,
        }
        self.assertEqual(set(self.vae.hand_names), set(expected_dims))
        for hand, dim in expected_dims.items():
            q = torch.zeros(4, dim)
            z = self.vae.encode(q, hand)
            decoded = self.vae.decode(z, hand)
            tips = self.vae.fingerpads(decoded, hand)
            self.assertEqual(tuple(z.shape), (4, 24))
            self.assertEqual(tuple(decoded.shape), (4, dim))
            self.assertEqual(tuple(tips.shape), (4, 5, 3))
            self.assertTrue(torch.isfinite(decoded).all())
            runtime = self.vae.runtimes[hand]
            lower = runtime.q_lower[:dim]
            upper = runtime.q_upper[:dim]
            self.assertTrue((decoded.cpu() >= lower - 1e-6).all())
            self.assertTrue((decoded.cpu() <= upper + 1e-6).all())

    def test_cross_hand_shapes(self) -> None:
        q = torch.zeros(3, 22)
        result = self.vae.retarget(q, "shadow_hand_right", "gaia_hand_right")
        self.assertEqual(tuple(result.z_gesture.shape), (3, 24))
        self.assertEqual(tuple(result.target_q.shape), (3, 15))
        self.assertEqual(tuple(result.fingerpad_error.shape), (3, 5))


if __name__ == "__main__":
    unittest.main()
