from __future__ import annotations

import json
from pathlib import Path

from adcust_logic.defs import SIG_FILENAME

from tests.fake_server import remote_training_request
from tests.unit.training_service_test_case import TrainingServiceTestCase


class RemoteTrainingPipelineTest(TrainingServiceTestCase):
    def test_remote_training_pipeline_runs_through_fake_server(self):
        events = list(self.service.start_remote_adapter_job(remote_training_request(self.root)))

        statuses = [event["status"] for event in events]
        self.assertIn("uploading", statuses)
        self.assertIn("running", statuses)
        self.assertIn("collecting", statuses)
        self.assertEqual(events[-1]["status"], "completed")
        self.assertEqual(events[-1]["percentage"], 100)
        self.assertTrue(events[-1]["requires_manual_shutdown"])

        uploaded_names = "\n".join(self.server.uploaded_files)
        self.assertIn("dataset.json", uploaded_names)
        self.assertIn("train_config.json", uploaded_names)
        self.assertIn("expected_base_model_manifest.json", uploaded_names)
        self.assertIn("train_entry.py", uploaded_names)
        self.assertTrue(any("train_entry.py" in command and "--config" in command for command in self.server.executed_commands))
        self.assertFalse(any("timeout " in command for command in self.server.executed_commands))
        self.assertTrue(self.driver.disconnected)

        output_dir = Path(events[-1]["local_output_dir"])
        self.assertTrue((output_dir / SIG_FILENAME).is_file())
        signature = json.loads((output_dir / SIG_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(signature["adapter_name"], "unit_test_adapter")
        self.assertEqual(signature["execution"], "REMOTE_SSH")
        self.assertEqual(signature["adapter_format"], "peft_lora")
        self.assertIn("combined_hash", signature["base_model_manifest"])
