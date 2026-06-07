from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

try:
    from _bootstrap import ROOT
except ModuleNotFoundError:
    from tests._bootstrap import ROOT


class _Tokenizer:
    chat_template = None
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


class _BatchEncodingTokenizer(_Tokenizer):
    chat_template = "available"

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        token_count = 12 if add_generation_prompt else 18
        return {
            "input_ids": list(range(token_count)),
            "attention_mask": [1] * token_count,
        }


class KnowledgeTrainingHelpersTest(unittest.TestCase):
    """验证自动知识监督可以解析问答，并且只监督 assistant 答案。"""

    @classmethod
    def setUpClass(cls):
        path = Path(ROOT) / "packages" / "adcust-logic" / "adcust_logic" / "remote" / "train_entry.py"
        spec = importlib.util.spec_from_file_location("adcust_train_entry_test", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_extracts_generated_question_answer_json(self):
        result = self.module.extract_json_array(
            '```json\n[{"question":"Who is Neil?","answer":"Neil is the founder."}]\n```'
        )
        self.assertEqual(result[0]["answer"], "Neil is the founder.")

    def test_masks_prompt_tokens_from_answer_loss(self):
        example = self.module.build_qa_training_example(
            _Tokenizer(),
            "Who is Neil?",
            "Neil is the founder.",
            128,
        )
        self.assertIsNotNone(example)
        self.assertIn(-100, example["labels"])
        self.assertTrue(any(label != -100 for label in example["labels"]))

    def test_extracts_plain_text_question_answer_format(self):
        result = self.module.extract_qa_pairs(
            "Question: Who founded the city?\nAnswer: Neil founded the city."
        )
        self.assertEqual(result[0]["question"], "Who founded the city?")
        self.assertEqual(result[0]["answer"], "Neil founded the city.")

    def test_builds_deterministic_fallback_for_chinese_story(self):
        result = self.module.build_fallback_qa_pairs(
            "尼尔在雨夜抵达银港。随后他在旧钟楼找到了失踪的地图。"
        )
        self.assertGreaterEqual(len(result), 2)
        self.assertTrue(any("尼尔" in item["answer"] for item in result))

    def test_builds_deterministic_fallback_for_english_story(self):
        result = self.module.build_fallback_qa_pairs(
            "Neil arrived at Silver Harbor during the storm. He found the missing map in the old tower."
        )
        self.assertGreaterEqual(len(result), 2)
        self.assertTrue(any("Silver Harbor" in item["answer"] for item in result))

    def test_accepts_batch_encoding_from_qwen_chat_template(self):
        example = self.module.build_qa_training_example(
            _BatchEncodingTokenizer(),
            "Who found the map?",
            "Neil found the map.",
            128,
        )
        self.assertIsNotNone(example)
        self.assertEqual(len(example["input_ids"]), 18)
        self.assertEqual(sum(label != -100 for label in example["labels"]), 6)
