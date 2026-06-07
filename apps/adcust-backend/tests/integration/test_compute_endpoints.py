from __future__ import annotations

try:
    from _bootstrap import ROOT  # noqa: F401
except ModuleNotFoundError:
    from tests._bootstrap import ROOT  # noqa: F401

from tests.fake_server import FakeComputeDriver, FakeInfraService, FakeRemoteServer, make_provider
from tests.integration.backend_integration_test_case import BackendIntegrationTestCase


class ComputeEndpointsIntegrationTest(BackendIntegrationTestCase):
    def setUp(self):
        super().setUp()
        from app.config.dependencies import get_infra_service

        self.compute_server = FakeRemoteServer(self.root / "fake_compute_remote")
        self.compute_driver = FakeComputeDriver(self.compute_server)
        self.compute_provider = make_provider()
        self.compute_service = FakeInfraService(self.compute_provider, self.compute_driver)
        self.app.dependency_overrides[get_infra_service] = lambda: self.compute_service

    def test_compute_provider_lifecycle_endpoints(self):
        create_payload = {
            "id": "fake-ssh-node",
            "name": "Fake SSH Node",
            "provider_type": "SSH",
            "connection_info": {
                "host": "fake.local",
                "port": 22,
                "username": "tester",
                "remote_workspace_path": "/adcust-test-workspace",
                "hourly_rate_usd": 1.25,
            },
        }

        created = self.client.post("/api/v1/compute/", json=create_payload)
        self.assertEqual(created.status_code, 200)
        self.assertEqual(created.json()["id"], "fake-ssh-node")

        listed = self.client.get("/api/v1/compute/")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()[0]["id"], "fake-ssh-node")

        verified = self.client.post("/api/v1/compute/fake-ssh-node/verify")
        self.assertEqual(verified.status_code, 200)
        self.assertEqual(verified.json()["status"], "connected")

        synced = self.client.post("/api/v1/compute/fake-ssh-node/sync")
        self.assertEqual(synced.status_code, 200)

        disconnected = self.client.post("/api/v1/compute/fake-ssh-node/disconnect")
        self.assertEqual(disconnected.status_code, 200)
        self.assertFalse(disconnected.json()["is_active"])

        removed = self.client.delete("/api/v1/compute/fake-ssh-node")
        self.assertEqual(removed.status_code, 200)
        self.assertEqual(removed.json()["status"], "success")
