from __future__ import annotations

import json

from tests.fake_server import remote_training_request
from tests.integration.backend_integration_test_case import BackendIntegrationTestCase


class RemoteStatusEndpointIntegrationTest(BackendIntegrationTestCase):
    def test_remote_status_endpoint_returns_job_snapshot_after_stream(self):
        start = self.client.post("/api/v1/training/remote/start", json=remote_training_request(self.root))
        events = [json.loads(line) for line in start.text.splitlines() if line.strip()]
        job_id = events[-1]["job_id"]

        status = self.client.get(f"/api/v1/training/remote/{job_id}")

        self.assertEqual(status.status_code, 200)
        payload = status.json()
        self.assertEqual(payload["job_id"], job_id)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["training_params"]["epochs"], 1)
