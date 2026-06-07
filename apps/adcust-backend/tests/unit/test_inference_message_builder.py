from __future__ import annotations

import unittest

try:
    from _bootstrap import ROOT  # noqa: F401
except ModuleNotFoundError:
    from tests._bootstrap import ROOT  # noqa: F401

from adcust_logic.services.engines.client.inference_engines.huggingface_engine import HuggingfaceEngine


class InferenceMessageBuilderTest(unittest.TestCase):
    def test_build_messages_preserves_system_history_and_current_turn(self):
        messages = HuggingfaceEngine()._build_messages(
            "我刚才说我叫什么？",
            system_prompt="Use history.",
            history=[
                {"role": "user", "content": "你好，我叫 Neil。"},
                {"role": "assistant", "content": "你好 Neil。"},
                {"role": "tool", "content": "ignored"},
                {"role": "assistant", "content": ""},
            ],
        )

        self.assertEqual([item["role"] for item in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[-1]["content"], "我刚才说我叫什么？")

    def test_plain_prompt_fallback_contains_role_labels(self):
        engine = HuggingfaceEngine()
        prompt = engine._messages_to_plain_prompt([
            {"role": "system", "content": "Use history."},
            {"role": "user", "content": "Hi"},
        ])

        self.assertIn("System: Use history.", prompt)
        self.assertTrue(prompt.endswith("Assistant:"))
