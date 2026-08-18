from __future__ import annotations

import contextlib
import unittest
from unittest import mock

from BeingH.npu_prefix_segment_route import build_prefix_segment_route

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    import BeingH.model.llm.qwen3_navit as qwen3_navit
    from BeingH.model.llm.qwen3_navit import (
        PackedAttentionMoT,
        Qwen3Config,
    )


@contextlib.contextmanager
def fake_sdpa_kernel(*args, **kwargs):
    del args, kwargs
    yield


def fake_sdpa(query, key, value, attention_mask, *, enable_gqa=False):
    if enable_gqa:
        groups = query.shape[-3] // key.shape[-3]
        key = key.repeat_interleave(groups, dim=-3)
        value = value.repeat_interleave(groups, dim=-3)
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
    scores = scores * (query.shape[-1] ** -0.5)
    scores = scores + attention_mask.float()
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, value.float()).to(query.dtype)


def fake_fusion_attention(
    query_states,
    key_states,
    value_states,
    attention_mask,
    head_num,
    scale,
    input_layout="BNSD",
):
    del head_num, scale, input_layout
    groups = query_states.shape[0] // key_states.shape[0]
    key_states = key_states[:, None, :, :].repeat(
        1, groups, 1, 1
    ).reshape(query_states.shape[0], key_states.shape[1], key_states.shape[2])
    value_states = value_states[:, None, :, :].repeat(
        1, groups, 1, 1
    ).reshape(
        query_states.shape[0],
        value_states.shape[1],
        value_states.shape[2],
    )
    return fake_sdpa(
        query_states.unsqueeze(0),
        key_states.unsqueeze(0),
        value_states.unsqueeze(0),
        attention_mask.unsqueeze(0).unsqueeze(0),
    ).squeeze(0)


@unittest.skipIf(torch is None, "PyTorch is not installed in the local CPU Python")
class NpuPrefixSegmentAttentionTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(123)
        expert_config = Qwen3Config(
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=32,
        )
        config = Qwen3Config(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=32,
            qk_norm=True,
        )
        config.expert_config = expert_config
        self.attention = PackedAttentionMoT(config, layer_idx=0).to(
            dtype=torch.bfloat16
        )
        self.und_indexes = torch.tensor([0, 2, 3, 6], dtype=torch.long)
        self.gen_indexes = torch.tensor([1, 4, 5], dtype=torch.long)
        self.route = build_prefix_segment_route(
            und_global_indexes=self.und_indexes.tolist(),
            gen_global_indexes=self.gen_indexes.tolist(),
            prefix_length=7,
        )
        self.und = torch.randn(4, 32, dtype=torch.bfloat16)
        self.gen = torch.randn(3, 24, dtype=torch.bfloat16)
        self.position_embeddings = (
            torch.ones(7, 8, dtype=torch.bfloat16),
            torch.zeros(7, 8, dtype=torch.bfloat16),
        )
        self.attention_mask = torch.full(
            (7, 7),
            float("-inf"),
            dtype=torch.bfloat16,
        )
        self.attention_mask.triu_(diagonal=1)

    def test_projected_qkv_and_output_routing_match_scatter_gather(self) -> None:
        query, key, value = self.attention._project_static_prefix_qkv(
            self.und,
            self.gen,
            self.route,
        )

        expected_query = torch.zeros_like(query)
        expected_key = torch.zeros_like(key)
        expected_value = torch.zeros_like(value)
        und_query = self.attention.q_norm(
            self.attention.q_proj(self.und).view(4, 4, 8)
        )
        gen_query = self.attention.q_norm_mot_gen(
            self.attention.q_proj_mot_gen(self.gen).view(3, 4, 8)
        )
        und_key = self.attention.k_norm(
            self.attention.k_proj(self.und).view(4, 2, 8)
        )
        gen_key = self.attention.k_norm_mot_gen(
            self.attention.k_proj_mot_gen(self.gen).view(3, 2, 8)
        )
        expected_query[self.und_indexes] = und_query
        expected_query[self.gen_indexes] = gen_query
        expected_key[self.und_indexes] = und_key
        expected_key[self.gen_indexes] = gen_key
        expected_value[self.und_indexes] = self.attention.v_proj(
            self.und
        ).view(4, 2, 8)
        expected_value[self.gen_indexes] = self.attention.v_proj_mot_gen(
            self.gen
        ).view(3, 2, 8)

        self.assertTrue(torch.equal(query, expected_query))
        self.assertTrue(torch.equal(key, expected_key))
        self.assertTrue(torch.equal(value, expected_value))
        self.assertEqual(key.shape, (7, 2, 8))
        self.assertEqual(value.shape, (7, 2, 8))

        global_output = torch.randn(7, 32, dtype=torch.bfloat16)
        restored_und = self.attention._restore_prefix_branch_order(
            global_output,
            self.route.und_segments,
        )
        restored_gen = self.attention._restore_prefix_branch_order(
            global_output,
            self.route.gen_segments,
        )
        self.assertTrue(
            torch.equal(restored_und, global_output[self.und_indexes])
        )
        self.assertTrue(
            torch.equal(restored_gen, global_output[self.gen_indexes])
        )

    def test_full_prefix_attention_matches_with_fusion_on_and_off(self) -> None:
        patches = (
            mock.patch.object(qwen3_navit, "sdpa_kernel", fake_sdpa_kernel),
            mock.patch.object(
                qwen3_navit,
                "scaled_dot_product_attention",
                fake_sdpa,
            ),
            mock.patch.object(
                qwen3_navit,
                "_npu_fusion_attention",
                fake_fusion_attention,
            ),
        )
        with patches[0], patches[1], patches[2]:
            for fusion_enabled in (False, True):
                with self.subTest(fusion_enabled=fusion_enabled):
                    self.attention.enable_npu_fusion_attention = fusion_enabled
                    self.attention.use_npu_single_sample_fast_path = True
                    baseline = self.attention.forward_train(
                        packed_sequence_und=self.und,
                        packed_sequence_gen=self.gen,
                        sample_lens=[7],
                        attention_mask=[self.attention_mask],
                        packed_position_embeddings=self.position_embeddings,
                        packed_und_token_indexes=self.und_indexes,
                        packed_gen_token_indexes=self.gen_indexes,
                        return_kv_cache=True,
                    )
                    candidate = self.attention.forward_train(
                        packed_sequence_und=self.und,
                        packed_sequence_gen=self.gen,
                        sample_lens=[7],
                        attention_mask=[self.attention_mask],
                        packed_position_embeddings=self.position_embeddings,
                        packed_und_token_indexes=self.und_indexes,
                        packed_gen_token_indexes=self.gen_indexes,
                        return_kv_cache=True,
                        prefix_segment_route=self.route,
                    )

                    self.assertTrue(torch.equal(baseline[0], candidate[0]))
                    self.assertTrue(torch.equal(baseline[1], candidate[1]))
                    self.assertTrue(
                        torch.equal(baseline[2][0], candidate[2][0])
                    )
                    self.assertTrue(
                        torch.equal(baseline[2][1], candidate[2][1])
                    )
                    expected_heads = 2 if fusion_enabled else 4
                    self.assertEqual(
                        candidate[2][0].shape,
                        (7, expected_heads, 8),
                    )


if __name__ == "__main__":
    unittest.main()
