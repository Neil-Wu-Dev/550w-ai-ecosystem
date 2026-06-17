"""训练服务单元测试基类。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

try:
    from _bootstrap import ROOT  # noqa: F401
except ModuleNotFoundError:
    from tests._bootstrap import ROOT  # noqa: F401

from adcust_logic.services.client.training_service import TrainingService

from tests.fake_server import (
    FakeComputeDriver,
    FakeInfraService,
    FakeRemoteServer,
    NullDatasetService,
    NullModelManager,
    make_provider,
)


class TrainingServiceTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.old_cwd = os.getcwd()
        os.chdir(self.root)
        self.server = FakeRemoteServer(self.root / "fake_remote")
        self.driver = FakeComputeDriver(self.server)
        self.provider = make_provider()
        self.service = TrainingService(
            model_manager=NullModelManager(),
            dataset_service=NullDatasetService(),
            infra_service=FakeInfraService(self.provider, self.driver),
        )

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.temp_dir.cleanup()
