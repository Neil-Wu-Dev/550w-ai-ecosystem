from __future__ import annotations

from adcust_logic.exceptions import ValidationException

from tests.fake_server import remote_training_request
from tests.unit.training_service_test_case import TrainingServiceTestCase


class TrainingParamsValidationTest(TrainingServiceTestCase):
    def test_training_params_require_explicit_lora_settings(self):
        request = remote_training_request(self.root)
        request.pop("target_modules")

        with self.assertRaises(ValidationException):
            self.service._build_training_params(request)
