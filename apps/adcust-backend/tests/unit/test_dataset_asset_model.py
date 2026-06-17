from __future__ import annotations

import unittest
import contextlib
import io

try:
    from _bootstrap import ROOT  # noqa: F401
except ModuleNotFoundError:
    from tests._bootstrap import ROOT  # noqa: F401

from adcust_logic.exceptions import ValidationException
from adcust_logic.models.dataset_asset import DatasetAsset


class DatasetAssetModelTest(unittest.TestCase):
    def test_dataset_asset_validates_and_destroys_chunks(self):
        asset = DatasetAsset(
            source_name="dataset.json",
            strategy="SMOKE_TEST",
            chunks=[{"instruction": "x", "input": "", "output": "y"}],
            chunk_size=128,
        )

        self.assertEqual(len(asset.chunks), 1)
        with contextlib.redirect_stdout(io.StringIO()):
            asset.destroy()
        self.assertIsNone(asset.chunks)

    def test_dataset_asset_rejects_empty_chunks_and_invalid_size(self):
        with self.assertRaises(ValidationException):
            DatasetAsset("dataset.json", "SMOKE_TEST", [], 128)
        with self.assertRaises(ValidationException):
            DatasetAsset("dataset.json", "SMOKE_TEST", [{"output": "x"}], 9000)
