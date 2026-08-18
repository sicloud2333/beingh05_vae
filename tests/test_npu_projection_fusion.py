from types import SimpleNamespace
import unittest

import torch
from torch import nn

from BeingH.model.llm.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm
from BeingH.model.llm.qwen3_navit import PackedAttentionMoT


class NpuProjectionFusionTest(unittest.TestCase):
    def test_gate_up_fusion_preserves_output_and_state_dict(self):
        torch.manual_seed(7)
        config = SimpleNamespace(
            hidden_size=8,
            intermediate_size=12,
            hidden_act="silu",
        )
        module = Qwen3MLP(config).eval()
        inputs = torch.randn(2, 3, config.hidden_size)

        baseline = module(inputs)
        module.enable_npu_gate_up_fusion = True
        candidate = module(inputs)

        torch.testing.assert_close(candidate, baseline, rtol=1e-6, atol=1e-6)
        self.assertNotIn("_npu_fused_gate_up_weight", module.state_dict())

    def test_action_qkv_fusion_preserves_projection_order_and_state_dict(self):
        torch.manual_seed(11)
        module = PackedAttentionMoT.__new__(PackedAttentionMoT)
        nn.Module.__init__(module)
        module.num_heads = 4
        module.num_key_value_heads = 2
        module.head_dim = 3
        input_size = 5
        module.q_proj_mot_gen = nn.Linear(input_size, 12, bias=False)
        module.k_proj_mot_gen = nn.Linear(input_size, 6, bias=False)
        module.v_proj_mot_gen = nn.Linear(input_size, 6, bias=False)
        module._npu_fused_qkv_mot_gen_weight = None
        module._npu_fused_qkv_mot_gen_bias = None
        inputs = torch.randn(2, 7, input_size)

        module.enable_npu_qkv_fusion = False
        baseline = module._project_action_qkv(inputs)
        module.enable_npu_qkv_fusion = True
        candidate = module._project_action_qkv(inputs)

        for actual, expected in zip(candidate, baseline, strict=True):
            torch.testing.assert_close(
                actual, expected, rtol=1e-6, atol=1e-6
            )
        self.assertNotIn(
            "_npu_fused_qkv_mot_gen_weight", module.state_dict()
        )

    def test_fused_swiglu_flag_keeps_cpu_fallback_exact(self):
        torch.manual_seed(9)
        config = SimpleNamespace(
            hidden_size=8,
            intermediate_size=12,
            hidden_act="silu",
        )
        module = Qwen3MLP(config).eval()
        module.enable_npu_gate_up_fusion = True
        inputs = torch.randn(2, 3, config.hidden_size)

        baseline = module(inputs)
        module.enable_npu_fused_swiglu = True
        candidate = module(inputs)

        torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)

    def test_npu_rms_norm_flag_keeps_cpu_fallback_exact(self):
        torch.manual_seed(13)
        module = Qwen3RMSNorm(16).eval()
        inputs = torch.randn(2, 3, 16)

        baseline = module(inputs)
        module.enable_npu_fused_rms_norm = True
        candidate = module(inputs)

        torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)

    def test_dtype_fast_path_only_skips_proven_noop_bfloat16_cast(self):
        module = PackedAttentionMoT.__new__(PackedAttentionMoT)
        nn.Module.__init__(module)
        module.enable_npu_dtype_fast_path = True
        bfloat16 = torch.randn(2, 3, dtype=torch.bfloat16)
        float32 = bfloat16.float()

        self.assertIs(module._to_attention_bfloat16(bfloat16), bfloat16)
        converted = module._to_attention_bfloat16(float32)
        self.assertEqual(converted.dtype, torch.bfloat16)
        torch.testing.assert_close(converted, bfloat16, rtol=0, atol=0)
