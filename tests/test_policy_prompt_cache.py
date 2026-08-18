import unittest
from types import SimpleNamespace

from BeingH.inference.beingh_policy import BeingHPolicy


class _CountingTokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, text):
        self.calls.append(text)
        return [ord(character) for character in text]


class PolicyPromptCacheTest(unittest.TestCase):
    def setUp(self):
        self.policy = BeingHPolicy.__new__(BeingHPolicy)
        self.policy.model = SimpleNamespace(system_message="safe robot")
        self.policy.instruction_template = "{task_description} / {k}"
        self.policy.action_chunk_length = 16
        self.policy.tokenizer = _CountingTokenizer()
        self.policy.enable_policy_prompt_cache = True
        self.policy._prompt_token_cache = {}

    def test_repeated_instruction_reuses_all_tokenizer_results(self):
        first = self.policy._encode_prompt_parts("pick up the cup")
        second = self.policy._encode_prompt_parts("pick up the cup")
        self.assertIs(first, second)
        self.assertEqual(len(self.policy.tokenizer.calls), 4)

    def test_instruction_and_template_are_part_of_cache_key(self):
        first = self.policy._encode_prompt_parts("pick up the cup")
        second = self.policy._encode_prompt_parts("open the drawer")
        self.assertNotEqual(first[-1], second[-1])
        self.assertEqual(len(self.policy.tokenizer.calls), 8)

        self.policy.instruction_template = "new {task_description} / {k}"
        self.policy._encode_prompt_parts("pick up the cup")
        self.assertEqual(len(self.policy.tokenizer.calls), 12)
