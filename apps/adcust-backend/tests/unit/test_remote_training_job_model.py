from __future__ import annotations

import unittest

try:
    from _bootstrap import ROOT  # noqa: F401
except ModuleNotFoundError:
    from tests._bootstrap import ROOT  # noqa: F401

from adcust_logic.exceptions import ValidationException
from adcust_logic.models.remote_training_job import RemoteTrainingJob


class RemoteTrainingJobModelTest(unittest.TestCase):
    def test_job_rejects_relative_remote_workspace(self):
        with self.assertRaises(ValidationException):
            RemoteTrainingJob(
                job_id="job_bad",
                provider_id="fake",
                dataset_path="dataset.json",
                base_model_name_or_path="/models/fake",
                adapter_name="adapter_ok",
                local_output_dir="out",
                remote_workspace_path="relative/path",
                training_params={"epochs": 1},
                base_model_manifest={"combined_hash": "abc"},
            )
