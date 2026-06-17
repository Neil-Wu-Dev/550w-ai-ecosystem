from __future__ import annotations

import unittest

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from tests import _bootstrap  # noqa: F401

from adcust_logic.services.client.training_service import TrainingService

from tests.fake_server import FakeComputeDriver, FakeInfraService, FakeRemoteServer, NullDatasetService, NullModelManager, make_provider


class RemoteTrainingLossEventTest(unittest.TestCase):
    """验证带有 PTY 控制字符的远端 loss 事件仍能被后端识别。"""

    def test_prefixed_loss_event_is_parsed(self):
        server = FakeRemoteServer.__new__(FakeRemoteServer)
        driver = FakeComputeDriver(server)
        service = TrainingService(
            model_manager=NullModelManager(),
            dataset_service=NullDatasetService(),
            infra_service=FakeInfraService(make_provider(), driver),
        )

        event = service._parse_remote_training_event(
            '\x1b[2KADCUST_EVENT {"adcust_event":"train_metric","loss":1.234,'
            '"step":2,"total_steps":10,"train_progress":20}'
        )

        self.assertIsNotNone(event)
        self.assertEqual(event["loss"], 1.234)
        self.assertEqual(event["step"], 2)
        self.assertGreater(event["percentage"], 60)
