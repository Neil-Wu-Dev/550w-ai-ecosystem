from abc import ABC, abstractmethod
from typing import List, Optional, Generator, Dict, Any
from app.models.model_asset import ModelAsset
from app.models.dataset_asset import DatasetAsset
from app.core.constants import TrainingMode


class ITrainingService(ABC):
    """
    模型定制化服务接口
    负责编排算力引擎、管理适配器存储逻辑以及状态转发。
    """

    @abstractmethod
    def customize_new_adapter(
            self,
            base_model: ModelAsset,
            dataset: DatasetAsset,
            target_dir: str,
            trainable_layers: List[str],
            mode: TrainingMode = TrainingMode.SEQUENTIAL,
            custom_name: Optional[str] = None,
            epochs: int = 1  # <--- 【核心修改】增加轮数参数，确保与实现类一致
    ) -> Generator[Dict[str, Any], None, None]:
        """
        开始一个新的适配器定制任务（训练）。

        Args:
            base_model: 底座模型资产。
            dataset: 已解析好的数据资产。
            target_dir: 适配器导出的根目录。
            trainable_layers: 用户勾选的可训练层列表（例如 ["q_proj", "v_proj"]）。
            mode: 训练模式（如 SEQUENTIAL）。
            custom_name: 用户自定义的适配器名称（可选）。
            epochs: 训练迭代轮数，默认为 1。

        Yields:
            Dict[str, Any]: 实时进度数据，包含以下字段：
                - status: "start" | "running" | "completed" | "error" | "aborted"
                - message: 当前状态描述。
                - percentage: 0-100 的总体进度浮点数。
                - epoch: 当前所在轮次（可选）。
                - step: 当前轮次内的步骤（可选）。
        """
        pass

    @abstractmethod
    def abort_training(self, mode: TrainingMode = TrainingMode.SEQUENTIAL):
        """
        强制中断当前正在进行的训练任务。
        应触发对应算力引擎的终止开关并清理资源。

        Args:
            mode: 指定要中断的引擎模式。
        """
        pass