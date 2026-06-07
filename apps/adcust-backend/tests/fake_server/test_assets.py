"""测试资产工厂。

所有测试文件都写入调用方传入的临时目录，测试结束后由测试框架清理。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

try:
    from _bootstrap import ROOT  # noqa: F401
except ModuleNotFoundError:
    from tests._bootstrap import ROOT  # noqa: F401

from adcust_logic.models.infra.compute_provider_asset import ComputeProviderAsset


class NullModelManager:
    pass


class NullDatasetService:
    pass


def make_provider() -> ComputeProviderAsset:
    return ComputeProviderAsset(
        id="fake-ssh-node",
        name="Fake SSH Node",
        provider_type="SSH",
        connection_info={
            "host": "fake.local",
            "port": 22,
            "username": "tester",
            "remote_workspace_path": "/adcust-test-workspace",
            "hourly_rate_usd": 1.25,
        },
        is_active=True,
    )


def write_fake_model(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text('{"model_type":"qwen2","architectures":["FakeCausalLM"]}', encoding="utf-8")
    (model_dir / "tokenizer.json").write_text('{"version":"1.0","model":{"type":"fake"}}', encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"fake-base-model")
    metadata_dir = model_dir / ".cache" / "huggingface" / "download"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "config.json.metadata").write_text(
        "0123456789abcdef0123456789abcdef01234567\n",
        encoding="utf-8",
    )


def write_fake_dataset(dataset_path: Path) -> None:
    dataset_path.write_text(
        json.dumps([{"instruction": "Say hello", "output": "Hello"}], ensure_ascii=False),
        encoding="utf-8",
    )


def remote_training_request(tmp_root: Path, provider_id: str = "fake-ssh-node") -> Dict[str, Any]:
    model_dir = tmp_root / "local_model"
    dataset_path = tmp_root / "dataset.json"
    output_root = tmp_root / "adapters"
    write_fake_model(model_dir)
    write_fake_dataset(dataset_path)
    return {
        "provider_id": provider_id,
        "dataset_path": str(dataset_path),
        "local_base_model_path": str(model_dir),
        "base_model_name_or_path": "fake-org/fake-qwen",
        "adapter_name": "unit_test_adapter",
        "local_output_root": str(output_root),
        "epochs": 1,
        "learning_rate": 0.0002,
        "max_seq_length": 128,
        "lora_rank": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "use_qlora": True,
        "cleanup_remote": True,
    }
