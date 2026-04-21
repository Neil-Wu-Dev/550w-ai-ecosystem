from abc import ABC, abstractmethod
from typing import List, Optional, Generator, Dict, Any
from adcust_logic.defs import TrainingMode


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
        """
        启动训练任务的核心入口。
        实现类应通过依赖注入获取 ModelManager 和 DatasetService，
        并根据当前活跃的 ComputeProvider 自动决策是【本地训练】还是【云端调度】。
        """
        pass

    @abstractmethod
    def abort_training(self, mode: TrainingMode = TrainingMode.SEQUENTIAL):
        """
        强制中断当前训练。
        如果是本地训练，应向 Engine 下发中断信号；
        如果是远端训练，应通过 Driver 向远程节点发送终止指令（如 SIGTERM）。
        """
        pass

    # --- 这里是关键：为了对齐你的实现类，必须在接口中显式约束私有逻辑的规范 ---

    @abstractmethod
    def _run_remote_training(self, provider, model, dataset, layers, epochs) -> Generator[Dict[str, Any], None, None]:
        """
        [内部规范] 定义远端训练的调度逻辑规范。
        虽然是私有方法的前缀，但在复杂调度接口中，这有助于子类化时保持逻辑一致。
        """
        pass

