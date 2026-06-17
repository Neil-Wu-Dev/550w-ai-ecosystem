"""FastAPI 集成测试基类。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

try:
    from _bootstrap import ROOT  # noqa: F401
except ModuleNotFoundError:
    from tests._bootstrap import ROOT  # noqa: F401

try:
    from fastapi.testclient import TestClient
    from app.main import create_app
    from app.config.dependencies import get_training_service
    from adcust_logic.services.client.training_service import TrainingService
except ModuleNotFoundError as exc:
    TestClient = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests.fake_server import (
    FakeComputeDriver,
    FakeInfraService,
    FakeRemoteServer,
    NullDatasetService,
    NullModelManager,
    make_provider,
)


class BackendIntegrationTestCase(unittest.TestCase):
    def setUp(self):
        if TestClient is None:
            self.skipTest(f"Backend dependency missing; FastAPI integration test skipped: {IMPORT_ERROR}")

        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.old_cwd = os.getcwd()
        os.chdir(self.root)

        self.server = FakeRemoteServer(self.root / "fake_remote")
        self.driver = FakeComputeDriver(self.server)
        self.training_service = TrainingService(
            model_manager=NullModelManager(),
            dataset_service=NullDatasetService(),
            infra_service=FakeInfraService(make_provider(), self.driver),
        )

        self.app = create_app()
        self.app.dependency_overrides[get_training_service] = lambda: self.training_service
        self.client = TestClient(self.app)

    def tearDown(self):
        if hasattr(self, "app"):
            self.app.dependency_overrides.clear()
        if hasattr(self, "old_cwd"):
            os.chdir(self.old_cwd)
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()
