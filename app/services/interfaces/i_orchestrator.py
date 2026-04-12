from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Generator
from app.models.model_asset import ModelAsset
from app.models.adapter_asset import AdapterAsset
from app.core.constants import InferenceDispatchStrategy


class IOrchestrator(ABC):
    """
    全流程编排器接口 (Interface)
    作为系统的最高层逻辑抽象，负责协调模型管理、数据准备、模型训练（定制化）及推理分发。
    """

    @abstractmethod
    def set_base_model(self, local_path: str) -> ModelAsset:
        """
        选定并加载底座模型资产。

        Args:
            local_path: 模型在本地磁盘的绝对路径。
        Returns:
            ModelAsset: 包含模型元数据的对象。
        """
        pass

    @abstractmethod
    def get_static_options(self) -> Dict[str, Any]:
        """
        获取前端所需的静态配置选项（如训练策略、引擎模式、切片参数限制等）。
        """
        pass

    @abstractmethod
    def prepare_training_data(self, file_path: str, chunk_size: int, strategy: str) -> Dict[str, Any]:
        """
        准备并解析训练所需的数据集资产。

        Args:
            file_path: 源文件路径（PDF/TXT等）。
            chunk_size: 文本切片大小。
            strategy: 训练策略（知识注入/冒烟测试等）。
        """
        pass

    @abstractmethod
    def execute_customization(
            self,
            target_dir: str,
            trainable_layers: List[str],
            mode: str,
            custom_name: Optional[str] = None,
            epochs: int = 1  # <--- 【关键修改】增加训练轮数参数，确保与实现类一致
    ) -> Generator[Dict[str, Any], None, None]:
        """
        执行定制化训练流（微调）。

        Args:
            target_dir: 适配器保存的根目录。
            trainable_layers: 允许被训练的层名称列表。
            mode: 训练模式字符串（如 "sequential"）。
            custom_name: 用户自定义的适配器文件夹名称。
            epochs: 训练迭代轮数。

        Yields:
            Dict[str, Any]: 实时推送的训练状态，包含：
                - status: 状态标识 ("start", "running", "completed", "error", "aborted")
                - message: 步骤描述文字。
                - percentage: 总体进度百分比 (0-100)。
                - epoch: 当前轮次 (可选)。
                - step: 当前步骤 (可选)。
        """
        pass

    @abstractmethod
    def stop_customization(self, mode: str = "sequential") -> Dict[str, str]:
        """
        强制停止当前正在进行的定制化训练任务。

        Args:
            mode: 需要停止的引擎模式。
        """
        pass

    @abstractmethod
    def boot_inference_engine(self) -> str:
        """启动推理引擎，准备接受对话请求。"""
        pass

    @abstractmethod
    def shutdown_inference_engine(self) -> None:
        """关闭推理引擎并清理所有已挂载的适配器。"""
        pass

    @abstractmethod
    def load_adapter_asset(self, adapter_path: str, slot_id: str) -> AdapterAsset:
        """
        加载适配器资产并映射到指定的推理槽位（Slot）。
        实现类应在此处进行基础的架构兼容性校验。
        """
        pass

    @abstractmethod
    def stream_chat(
            self,
            targets: List[Dict[str, Any]],
            prompt: str,
            strategy: InferenceDispatchStrategy,
            max_tokens: int = 512
    ) -> Generator[str, None, None]:
        """
        多路径流式推理分发。

        Args:
            targets: 目标槽位列表（包含 ID 等信息）。
            prompt: 用户输入。
            strategy: 分发策略（串行 SERIAL 或 交替 INTERLEAVED）。
            max_tokens: 最大生成长度。
        """
        pass