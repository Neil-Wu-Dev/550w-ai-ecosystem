from abc import ABC, abstractmethod
from typing import Dict, Any, Generator

class IComputeDriver(ABC):
    """
    云端物理驱动契约。
    任何云端（SSH, Kaggle, GPUCloud）只要实现这四个方法，就能接入 AdCust。
    """
    @abstractmethod
    def connect(self, config: Dict[str, Any]):
        """建立物理连接（握手）"""
        pass

    @abstractmethod
    def push_file(self, local_path: str, remote_path: str):
        """物理上传（对应第 3, 5 步）"""
        pass

    @abstractmethod
    def exec_command(self, command: str) -> Generator[str, None, None]:
        """执行远端指令并流式回传日志（对应第 2, 4, 6 步）"""
        pass

    @abstractmethod
    def pull_file(self, remote_path: str, local_path: str):
        """物理下载（对应第 6 步结果回收）"""
        pass

    def disconnect(self):
        """释放底层连接资源"""
        pass
