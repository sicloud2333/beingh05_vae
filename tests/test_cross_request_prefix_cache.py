import unittest

from BeingH.inference.cross_request_prefix_cache import (
    MIXED_PREFIX_EXTENSION_REQUIRED,
    analyze_packed_layout,
    build_prefix_key,
    require_supported,
)


class CrossRequestPrefixCacheTest(unittest.TestCase):
    def test_key_is_deterministic_and_version_scoped(self):
        common = {
            "model_fingerprint": "checkpoint-sha",
            "tokenizer_fingerprint": "tokenizer-sha",
            "system_token_ids": [1, 2, 3],
            "device": "npu:0",
            "dtype": "bfloat16",
            "software_fingerprint": "torch-npu+cann",
        }
        first = build_prefix_key(**common)
        second = build_prefix_key(**common)
        self.assertEqual(first.digest, second.digest)
        changed = build_prefix_key(
            **{**common, "software_fingerprint": "different-cann"}
        )
        self.assertNotEqual(first.digest, changed.digest)

    def test_current_mixed_prefix_fails_closed(self):
        plan = analyze_packed_layout(
            system_positions=range(4),
            dynamic_prefix_positions=range(4, 11),
            action_positions=range(11, 27),
        )
        self.assertTrue(plan.system_is_leading_contiguous)
        self.assertFalse(plan.supported_by_current_decoder)
        self.assertEqual(plan.reason, MIXED_PREFIX_EXTENSION_REQUIRED)
        with self.assertRaises(NotImplementedError):
            require_supported(plan)

    def test_non_leading_system_is_rejected(self):
        plan = analyze_packed_layout(
            system_positions=[1, 2],
            dynamic_prefix_positions=[0, 3],
            action_positions=[4],
        )
        self.assertFalse(plan.system_is_leading_contiguous)
        self.assertEqual(
            plan.reason, "packed_layout_not_system_dynamic_action"
        )


if __name__ == "__main__":
    unittest.main()
