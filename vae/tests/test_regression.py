from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch

from native_vae import NativeVAE


REFERENCE = Path(__file__).resolve().parent / "reference_outputs.npz"


class NativeVAERegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vae = NativeVAE.from_pretrained(device="cpu")
        cls.reference = np.load(REFERENCE)

    def test_against_original_project(self) -> None:
        source = "shadow_hand_right"
        q = self.reference["shadow_q"]
        z = self.vae.encode(q, source)
        np.testing.assert_allclose(
            z.detach().cpu().numpy(),
            self.reference["shadow_z"],
            rtol=1e-6,
            atol=1e-6,
        )
        for target in self.vae.hand_names:
            result = self.vae.retarget(q, source, target)
            np.testing.assert_allclose(
                result.target_q.detach().cpu().numpy(),
                self.reference[f"{target}_q"],
                rtol=1e-6,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                result.target_fingerpads.detach().cpu().numpy(),
                self.reference[f"{target}_tips"],
                rtol=1e-6,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
