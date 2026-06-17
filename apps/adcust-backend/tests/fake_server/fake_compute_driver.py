"""实现 AdCust IComputeDriver 契约的 fake SSH driver。"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict

from .fake_remote_server import FakeRemoteServer


class FakeComputeDriver:
    def __init__(self, server: FakeRemoteServer):
        self.server = server
        self.disconnected = False

    def connect(self, config: Dict[str, Any]):
        self.server.connected = True

    def test_connection(self, config: Dict[str, Any]):
        self.server.connected = True
        return True

    def ensure_dir(self, remote_path: str):
        self.server.ensure_dir(remote_path)

    def push_file(self, local_path: str, remote_path: str):
        self.server.push_file(local_path, remote_path)

    def exec_command(self, command: str):
        yield from self.server.exec_command(command)

    def pull_file(self, remote_path: str, local_path: str):
        source = self.server.remote_to_local(remote_path)
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    def pull_dir(self, remote_path: str, local_path: str):
        self.server.pull_dir(remote_path, local_path)

    def disconnect(self):
        self.disconnected = True
        self.server.connected = False
