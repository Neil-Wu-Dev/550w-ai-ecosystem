from __future__ import annotations

import json

from tests.fake_server import remote_training_request
from tests.integration.backend_integration_test_case import BackendIntegrationTestCase


class RemoteStartEndpointIntegrationTest(BackendIntegrationTestCase):
    def test_remote_start_endpoint_streams_complete_pipeline(self):
        response = self.client.post("/api/v1/training/remote/start", json=remote_training_request(self.root))

        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
        self.assertGreaterEqual(len(events), 5)
        self.assertEqual(events[-1]["status"], "completed")
        self.assertEqual(events[-1]["percentage"], 100)
        self.assertTrue(events[-1]["requires_manual_shutdown"])
        self.assertTrue(any(event["status"] == "running" for event in events))
        self.assertTrue(any("Remote training" in event["message"] for event in events))
