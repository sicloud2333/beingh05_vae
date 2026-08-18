from __future__ import annotations

import unittest

from BeingH.npu_single_sample_fast_path import (
    resolve_npu_single_sample_fast_path,
)


class NpuSingleSampleFastPathRoutingTest(unittest.TestCase):
    def resolve(
        self,
        mode: str,
        sample_lens: list[int],
        packed_seq_len: int,
        parallel_inference: bool = False,
    ) -> bool:
        return resolve_npu_single_sample_fast_path(
            mode,
            sample_lens,
            packed_seq_len,
            parallel_inference=parallel_inference,
        )

    def test_auto_routes_single_real_sample_to_fast_path(self) -> None:
        self.assertTrue(self.resolve("auto", [16], 16))

    def test_auto_ignores_trailing_dummy_padding(self) -> None:
        self.assertTrue(self.resolve("auto", [16, 112], 16))

    def test_auto_routes_multiple_real_samples_to_packed_path(self) -> None:
        self.assertFalse(self.resolve("auto", [7, 9], 16))

    def test_auto_routes_parallel_inference_to_packed_path(self) -> None:
        self.assertFalse(
            self.resolve("auto", [16], 16, parallel_inference=True)
        )

    def test_off_always_routes_to_packed_path(self) -> None:
        self.assertFalse(self.resolve("off", [16], 16))

    def test_force_accepts_one_non_parallel_real_sample(self) -> None:
        self.assertTrue(self.resolve("force", [16, 112], 16))

    def test_force_rejects_multiple_real_samples(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "real_sample_count=2"):
            self.resolve("force", [7, 9], 16)

    def test_force_rejects_parallel_inference(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "parallel_inference=True"):
            self.resolve("force", [16], 16, parallel_inference=True)

    def test_invalid_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid"):
            self.resolve("enabled", [16], 16)


if __name__ == "__main__":
    unittest.main()
