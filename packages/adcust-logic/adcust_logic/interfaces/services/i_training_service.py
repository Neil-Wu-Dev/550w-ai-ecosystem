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
        【修改】启动训练任务。
        不再接收 base_model 和 dataset 参数，由实现类内部通过服务注入获取。
        """
        pass

    @abstractmethod
    def abort_training(self, mode: TrainingMode = TrainingMode.SEQUENTIAL):
        """强制中断当前训练"""
        pass