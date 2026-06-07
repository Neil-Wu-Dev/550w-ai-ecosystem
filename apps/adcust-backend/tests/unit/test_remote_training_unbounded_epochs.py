from __future__ import annotations

import unittest

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from tests import _bootstrap  # noqa: F401

from adcust_logic.schemas.remote_training_schema import RemoteTrainingStartRequest


class RemoteTrainingUnboundedEpochsTest(unittest.TestCase):
    """验证训练轮数不再被人为限制为 20。"""

    def test_epochs_above_twenty_are_accepted(self):
        request = RemoteTrainingStartRequest(
            provider_id="provider",
            dataset_path="dataset.pdf",
            local_base_model_path="local-model",
            base_model_name_or_path="organization/model",
            adapter_name="knowledge-adapter",
            local_output_root="adapters",
            epochs=40,
            learning_rate=0.0002,
            max_seq_length=521,
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            use_qlora=True,
            cleanup_remote=False,
        )

        self.assertEqual(request.epochs, 40)
