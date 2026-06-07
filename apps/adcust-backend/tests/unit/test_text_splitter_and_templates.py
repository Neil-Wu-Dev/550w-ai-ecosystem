from __future__ import annotations

import unittest

try:
    from _bootstrap import ROOT  # noqa: F401
except ModuleNotFoundError:
    from tests._bootstrap import ROOT  # noqa: F401

from adcust_logic.services.engines.client.data_engines.knowledge_engines import (
    KnowledgeInjectionTemplate,
    SmokeTestTemplate,
)
from adcust_logic.services.engines.client.data_engines.pdf_engines import SimpleTextSplitter


class TextSplitterAndTemplatesTest(unittest.TestCase):
    def test_splitter_filters_empty_chunks_and_enforces_min_size(self):
        chunks = SimpleTextSplitter().split("abc   def   ghi", 1)
        self.assertEqual(chunks, ["abc   def", "ghi"])

    def test_templates_wrap_training_chunks(self):
        knowledge = KnowledgeInjectionTemplate().wrap(["fact"])
        smoke = SmokeTestTemplate().wrap(["ignored"])

        self.assertEqual(knowledge[0]["output"], "fact")
        self.assertIn("ground-truth", knowledge[0]["instruction"])
        self.assertEqual(smoke[0]["output"], "Pass")
