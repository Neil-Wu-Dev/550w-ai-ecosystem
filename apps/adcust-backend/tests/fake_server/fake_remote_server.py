"""模拟远端 Linux GPU 主机。

这个 fake server 只服务测试：它在临时目录里模拟远端工作区、命令执行、
日志流和 artifact 下载，不连接真实 SSH，也不产生真实 adapter 产品。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Generator, List

try:
    from _bootstrap import ROOT  # noqa: F401
except ModuleNotFoundError:
    from tests._bootstrap import ROOT  # noqa: F401

class FakeRemoteServer:
    def __init__(self, root: Path):
        self.root = root
        self.connected = False
        self.uploaded_files: List[str] = []
        self.downloaded_dirs: List[str] = []
        self.executed_commands: List[str] = []

    def remote_to_local(self, remote_path: str) -> Path:
        normalized = remote_path.strip("/")
        return self.root / normalized

    def ensure_dir(self, remote_path: str) -> None:
        self.remote_to_local(remote_path).mkdir(parents=True, exist_ok=True)

    def push_file(self, local_path: str, remote_path: str) -> None:
        target = self.remote_to_local(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, target)
        self.uploaded_files.append(remote_path)

    def pull_dir(self, remote_path: str, local_path: str) -> None:
        source = self.remote_to_local(remote_path)
        if not source.is_dir():
            raise AssertionError(f"Fake remote output does not exist: {remote_path}")
        target = Path(local_path)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        self.downloaded_dirs.append(remote_path)

    def exec_command(self, command: str) -> Generator[str, None, None]:
        self.executed_commands.append(command)
        if "nvidia-smi" in command and "disk_free_gb" in command:
            yield (
                '{"has_python": true, "has_gpu": true, '
                '"gpus": [{"name": "Fake GPU", "memory_total_mb": 24576, "memory_free_mb": 23000, '
                '"temperature_c": 42, "utilization_percent": 3}], '
                '"has_pytorch": true, "torch_version": "fake", "cuda_available": true, "disk_free_gb": 100.0}'
            )
            return

        if command.startswith("rm -rf "):
            remote_path = command.split("'", 2)[1]
            shutil.rmtree(self.remote_to_local(remote_path), ignore_errors=True)
            yield "[fake-server] remote workspace cleaned"
            return

        if "kill -TERM" in command:
            yield "AdCust remote termination signal sent"
            return

        if "[AdCust] Checking remote Python runtime" in command:
            for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "datasets", "huggingface_hub", "safetensors"):
                yield (
                    "ADCUST_EVENT "
                    + json.dumps(
                        {
                            "adcust_event": "dependency_status",
                            "message": f"{name}: ready",
                            "resource": {
                                "id": f"dependency:{name}",
                                "kind": "python_dependency",
                                "name": name,
                                "status": "ready",
                            },
                        }
                    )
                )
            yield 'ADCUST_EVENT {"adcust_event":"dependency_ready","message":"Remote dependencies ready"}'
            return

        config_path = self._extract_quoted_arg_after(command, "--config")
        config = json.loads(self.remote_to_local(config_path).read_text(encoding="utf-8"))
        output_dir = self.remote_to_local(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "adapter_config.json").write_text(
            json.dumps({"fake": True, "adapter_name": config["adapter_name"]}, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "adapter_model.safetensors").write_bytes(b"fake-adapter-bytes")
        yield "[fake-server] Remote training config loaded"
        yield "[fake-server] LoRA/QLoRA SFT training started"
        yield "[fake-server] Training completed"

    def _extract_quoted_arg_after(self, command: str, flag: str) -> str:
        marker = f"{flag} \\\""
        if marker in command:
            return command.split(marker, 1)[1].split("\\\"", 1)[0]
        marker = f'{flag} "'
        if marker in command:
            return command.split(marker, 1)[1].split('"', 1)[0]
        raise AssertionError(f"Cannot find {flag} in command: {command}")
