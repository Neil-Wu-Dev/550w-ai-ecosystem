from __future__ import annotations

import json

from tests.fake_server import remote_training_request
from tests.integration.backend_integration_test_case import BackendIntegrationTestCase


class RemoteAbortEndpointIntegrationTest(BackendIntegrationTestCase):
    def test_remote_abort_endpoint_sends_fake_server_termination_signal(self):
        start = self.client.post("/api/v1/training/remote/start", json=remote_training_request(self.root))
        events = [json.loads(line) for line in start.text.splitlines() if line.strip()]
        job_id = events[-1]["job_id"]

        abort = self.client.post(f"/api/v1/training/remote/{job_id}/abort")

        self.assertEqual(abort.status_code, 200)
        self.assertEqual(abort.json()["status"], "aborted")
        self.assertTrue(any("kill -TERM" in command for command in self.server.executed_commands))
