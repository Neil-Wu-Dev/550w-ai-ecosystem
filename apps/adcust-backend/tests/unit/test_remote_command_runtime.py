from __future__ import annotations

from tests.unit.training_service_test_case import TrainingServiceTestCase


class RemoteCommandRuntimeTest(TrainingServiceTestCase):
    def test_remote_command_only_runs_training_script_without_pod_lifecycle_control(self):
        command = self.service._build_remote_command(
            "/workspace/job",
            "/workspace/job/train_entry.py",
            "/workspace/job/train_config.json",
        )

        self.assertNotIn("timeout ", command)
        self.assertNotIn("shutdown", command.lower())
        self.assertNotIn("terminate", command.lower())
        self.assertIn("train.pid", command)
        self.assertIn("python3", command)
