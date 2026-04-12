from abc import ABC, abstractmethod
from typing import Generator, List, Dict, Any
from app.models.model_asset import ModelAsset
from app.models.dataset_asset import DatasetAsset


class ITrainingEngine(ABC):
    """
    算力引擎核心接口 (Interface)
    负责底座模型的加载、LoRA 层注入以及真实的梯度更新。
    """

    @abstractmethod
    def train(
            self,
            base_model: ModelAsset,
            dataset: DatasetAsset,
            save_path: str,
            target_modules: List[str],
            epochs: int = 1
    ) -> Generator[Dict[str, Any], None, None]:
        """
        执行模型训练任务。

        Args:
            base_model: 底座模型资产信息（包含本地路径、架构等）
            dataset: 经过 Orchestrator 处理后的数据集资产（包含 chunks 列表）
            save_path: 适配器（Adapter）最终导出的绝对路径
            target_modules: 允许被 LoRA 注入的可训练层名称列表（如 ["q_proj", "v_proj"]）
            epochs: 训练迭代轮数，决定了数据集被完整遍历的次数

        Yields:
            Dict[str, Any]: 包含以下字段的状态字典，用于前端实时显示进度：
                - status: "start" | "running" | "completed" | "error" | "aborted"
                - message: 当前执行步骤的具体描述文字
                - percentage: 0-100 的浮点数，代表整体进度（包含加载、训练各阶段）
                - epoch (可选): 当前正在进行的轮次索引 (1..N)
                - step (可选): 当前轮次内已处理的 chunk 索引 (1..M)
                - total_steps (可选): 本次任务预估的总迭代步数 (chunks * epochs)
        """
        pass

    @abstractmethod
    def abort(self):
        """
        外部控制信号入口：强制中断当前正在进行的训练任务。
        实现类在捕获此信号后，应立即停止 GPU 计算迭代，并尝试优雅地释放显存资源。
        """
        pass