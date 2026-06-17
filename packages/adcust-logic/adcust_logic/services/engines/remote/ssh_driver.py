# -*- coding: utf-8 -*-
import os
import posixpath
import socket
import stat
from io import StringIO
from typing import Dict, Any, Generator, Optional

import paramiko

from adcust_logic.exceptions import BusinessException
from adcust_logic.interfaces.services.infra.i_compute_driver import IComputeDriver

class SSHComputeDriver(IComputeDriver):
    """
    通用 SSH 算力驱动。
    AdCust 只假设存在一台可 SSH 访问的 Linux + NVIDIA GPU 主机，不关心它来自哪个云厂商。
    """

    def __init__(self):
        self.client = self._create_client()
        self._transport = None
        self._last_channel = None

    def _create_client(self):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        return client

    def connect(self, config: Dict[str, Any]):
        """
        建立 SSH 连接。
        支持 password、key_path、private_key 三种凭据形态，方便桌面端和 headless 包复用。
        """
        try:
            # 每次任务都创建新的 SSH client，避免复用上次训练后已关闭的 transport。
            self.disconnect()
            self.client = self._create_client()
            self._transport = None
            self._last_channel = None
            pkey = self._load_private_key(config)
            self.client.connect(
                hostname=config.get("host"),
                port=int(config.get("port", 22)),
                username=config.get("username") or config.get("user"),
                password=config.get("password") or None,
                pkey=pkey,
                timeout=20.0,
                banner_timeout=20.0,
                auth_timeout=20.0,
            )
            self._transport = self.client.get_transport()
        except (paramiko.SSHException, OSError, socket.error) as exc:
            raise BusinessException("ERR_REMOTE_CONNECT_FAILED", reason=str(exc))

    def _load_private_key(self, config: Dict[str, Any]) -> Optional[paramiko.PKey]:
        key_path = config.get("key_path")
        private_key = config.get("private_key")
        passphrase = config.get("passphrase") or config.get("password")
        if key_path:
            return paramiko.RSAKey.from_private_key_file(key_path, password=passphrase)
        if private_key:
            return paramiko.RSAKey.from_private_key(StringIO(private_key), password=passphrase)
        return None

    def test_connection(self, config: Dict[str, Any]) -> bool:
        self.connect(config)
        try:
            for _ in self.exec_command("printf adcust-ok"):
                pass
            return True
        finally:
            self.disconnect()

    def ensure_dir(self, remote_path: str):
        safe_path = self._quote(remote_path)
        for _ in self.exec_command(f"mkdir -p {safe_path}"):
            pass

    def push_file(self, local_path: str, remote_path: str):
        """
        上传单个文件。上层服务负责决定哪些资产可以上传，驱动只处理传输。
        """
        sftp = self.client.open_sftp()
        try:
            self.ensure_dir(posixpath.dirname(remote_path))
            sftp.put(local_path, remote_path)
        finally:
            sftp.close()

    def push_dir(self, local_dir: str, remote_dir: str):
        self.ensure_dir(remote_dir)
        for root, _, files in os.walk(local_dir):
            relative_root = os.path.relpath(root, local_dir)
            remote_root = remote_dir if relative_root == "." else posixpath.join(remote_dir, relative_root.replace("\\", "/"))
            self.ensure_dir(remote_root)
            for file_name in files:
                self.push_file(os.path.join(root, file_name), posixpath.join(remote_root, file_name))

    def exec_command(self, command: str) -> Generator[str, None, None]:
        """
        执行远端命令，并把 stdout/stderr 合并为实时日志流。
        """
        # 训练事件必须保持逐行 JSON 完整；PTY 的进度控制字符会污染事件行。
        stdin, stdout, stderr = self.client.exec_command(command, get_pty=False)
        stdout.channel.set_combine_stderr(True)
        self._last_channel = stdout.channel
        for line in stdout:
            yield line
        code = stdout.channel.recv_exit_status()
        if code != 0:
            raise BusinessException("ERR_REMOTE_COMMAND_FAILED", command=command, exit_code=code)

    def pull_file(self, remote_path: str, local_path: str):
        """
        下载单个文件。
        """
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        sftp = self.client.open_sftp()
        try:
            sftp.get(remote_path, local_path)
        finally:
            sftp.close()

    def pull_dir(self, remote_dir: str, local_dir: str):
        os.makedirs(local_dir, exist_ok=True)
        sftp = self.client.open_sftp()
        try:
            for item in sftp.listdir_attr(remote_dir):
                remote_child = posixpath.join(remote_dir, item.filename)
                local_child = os.path.join(local_dir, item.filename)
                if stat.S_ISDIR(item.st_mode):
                    self.pull_dir(remote_child, local_child)
                else:
                    sftp.get(remote_child, local_child)
        finally:
            sftp.close()

    def abort_current_command(self):
        if self._last_channel:
            self._last_channel.close()

    def disconnect(self):
        """
        断开 SSH 连接。
        """
        if self.client:
            self.client.close()
        self._transport = None
        self._last_channel = None

    def _quote(self, value: str) -> str:
        return "'" + value.replace("'", "'\"'\"'") + "'"
