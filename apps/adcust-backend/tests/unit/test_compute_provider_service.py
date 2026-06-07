from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from _bootstrap import ROOT  # noqa: F401
except ModuleNotFoundError:
    from tests._bootstrap import ROOT  # noqa: F401

from adcust_logic.services.infra.compute_provider_service import ComputeProviderService

from tests.fake_server import FakeComputeDriver, FakeRemoteServer, make_provider


class FakeDriverComputeProviderService(ComputeProviderService):
    def __init__(self, storage_path: str, driver: FakeComputeDriver):
        self.fake_driver = driver
        super().__init__(storage_path)

    def get_driver_for(self, provider):
        return self.fake_driver


class ComputeProviderServiceTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.server = FakeRemoteServer(self.root / "fake_remote")
        self.driver = FakeComputeDriver(self.server)
        self.service = FakeDriverComputeProviderService(str(self.root / "providers.json"), self.driver)
        self.provider = make_provider()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_load_list_and_remove_provider(self):
        self.service.save_provider(self.provider)
        self.assertEqual(self.service.get_provider(self.provider.id).name, "Fake SSH Node")
        self.assertEqual(len(self.service.list_all_providers()), 1)

        reloaded = FakeDriverComputeProviderService(str(self.root / "providers.json"), self.driver)
        self.assertEqual(reloaded.get_provider(self.provider.id).id, self.provider.id)

        self.assertTrue(reloaded.remove_provider(self.provider.id))
        self.assertIsNone(reloaded.get_provider(self.provider.id))

    def test_verify_sync_disconnect_and_cleanup(self):
        self.service.save_provider(self.provider)

        self.assertTrue(self.service.verify_connection(self.provider))
        synced = self.service.sync_resource_status(self.provider.id)
        self.assertTrue(synced.telemetry_data["env_probe"]["has_gpu"])
        self.assertEqual(synced.telemetry_data["env_probe"]["gpus"][0]["name"], "Fake GPU")

        disconnected = self.service.mark_provider_disconnected(self.provider.id)
        self.assertFalse(disconnected.is_active)
        self.assertEqual(disconnected.telemetry_data["session"]["state"], "released")

        self.assertTrue(self.service.remove_provider(self.provider.id, cleanup_remote=True))
        self.assertTrue(any(command.startswith("rm -rf") for command in self.server.executed_commands))
