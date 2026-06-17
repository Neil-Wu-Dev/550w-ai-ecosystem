from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from _bootstrap import ROOT  # noqa: F401
except ModuleNotFoundError:
    from tests._bootstrap import ROOT  # noqa: F401

from adcust_logic.services.client.model_manifest_service import ModelManifestService

from tests.fake_server import write_fake_model


class ModelManifestServiceTest(unittest.TestCase):
    def test_manifest_hash_is_stable_for_same_model_files(self):
        with tempfile.TemporaryDirectory() as temp:
            model_dir = Path(temp) / "model"
            write_fake_model(model_dir)
            service = ModelManifestService()

            first = service.build_local_manifest(str(model_dir))
            second = service.build_local_manifest(str(model_dir))

            self.assertEqual(first["combined_hash"], second["combined_hash"])
            self.assertEqual(first["source_type"], "local_huggingface_directory")
            self.assertTrue(any(item["path"] == "config.json" for item in first["files"]))

    def test_manifest_rejects_missing_model_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                ModelManifestService().build_local_manifest(str(Path(temp) / "missing"))
