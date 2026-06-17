from abc import ABC, abstractmethod
from typing import List, Optional, Generator, Dict, Any

from adcust_logic.defs import TrainingMode
from adcust_logic.models.remote_training_job import RemoteTrainingJob


class ITrainingService(ABC):
    @abstractmethod
    def customize_new_adapter(
        self,
        target_dir: str,
        trainable_layers: List[str],
        mode: TrainingMode = TrainingMode.SEQUENTIAL,
        custom_name: Optional[str] = None,
        epochs: int = 1
    ) -> Generator[Dict[str, Any], None, None]:
        """兼容旧入口。当前推荐使用显式远程训练入口。"""
        pass

    @abstractmethod
    def abort_training(self, mode: TrainingMode = TrainingMode.SEQUENTIAL):
        """尝试中断训练进程；不表示关闭或销毁云服务器。"""
        pass

    @abstractmethod
    def start_remote_adapter_job(self, request_data: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """通过 SSH 连接用户已经手动开启的远程机器，并启动 adapter 训练。"""
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> RemoteTrainingJob:
        """获取训练任务状态。"""
        pass

    @abstractmethod
    def abort_remote_job(self, job_id: str) -> RemoteTrainingJob:
        """尝试向远程训练进程下发终止信号；不表示关闭或销毁云服务器。"""
        pass

    @abstractmethod
    def _run_remote_training(self, provider, model, dataset, layers, epochs) -> Generator[Dict[str, Any], None, None]:
        """保留内部规范入口，实际远程训练使用 start_remote_adapter_job。"""
        pass
