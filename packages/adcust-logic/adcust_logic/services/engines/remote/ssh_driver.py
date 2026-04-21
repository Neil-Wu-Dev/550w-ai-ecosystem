# -*- coding: utf-8 -*-
import paramiko
from adcust_logic.interfaces.services.infra.i_compute_driver import IComputeDriver
from typing import Dict, Any, Generator

class SSHComputeDriver(IComputeDriver):
    """
    [AdCust 核心组件] 通用的 SSH 算力驱动
    功能：实现本地与远程 GPU 服务器的通信、文件传输及指令调度。
    """

    def __init__(self):
        """
        动作 1：初始化
        在内存中创建一个 SSH 客户端对象，并设置自动信任所有远程主机。
        """
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def connect(self, config: Dict[str, Any]):
        """
        动作 2：连接
        输入：包含 host, port, user, password 的字典。
        执行：建立加密的 TCP 隧道。
        """
        self.client.connect(
            hostname=config.get('host'),
            port=config.get('port', 22),
            username=config.get('user'),
            password=config.get('password'),
            pkey=config.get('ssh_key'),
            timeout=30.0  # 增加超时控制，防止程序无限期卡死
        )

    def push_file(self, local_path: str, remote_path: str):
        """
        动作 3：上传 (Push)
        执行：将本地训练脚本或数据发送到云端。
        """
        sftp = self.client.open_sftp()
        try:
            sftp.put(local_path, remote_path)
        finally:
            # 无论成功失败，必须销毁 SFTP 对象以释放内存
            sftp.close()

    def exec_command(self, command: str) -> Generator[str, None, None]:
        """
        动作 4：执行命令 (Execute)
        执行：在云端触发训练指令。
        传出：实时传回每一行训练日志。
        """
        stdin, stdout, stderr = self.client.exec_command(command, get_pty=True)
        # 实时监听远程服务器的输出缓冲区
        for line in stdout:
            yield line

    def pull_file(self, remote_path: str, local_path: str):
        """
        动作 5：下载 (Pull) - [新增补全]
        执行：将训练完成的模型权重文件（.pth）从云端拿回本地。
        """
        sftp = self.client.open_sftp()
        try:
            sftp.get(remote_path, local_path)
        finally:
            # 销毁对象，释放文件句柄
            sftp.close()

    def disconnect(self):
        """
        动作 6：断开 (Disconnect) - [新增补全]
        执行：彻底销毁 SSH 客户端对象，归还系统所有网络资源。
        """
        if self.client:
            self.client.close()