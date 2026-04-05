from abc import ABC, abstractmethod
from app.models.model_asset import ModelAsset
from app.models.adapter_asset import AdapterAsset # 别忘了导入这个

class IModelManagerService(ABC):
    @abstractmethod
    def select_model_by_path(self, local_path: str) -> ModelAsset:
        """根据路径加载并验证模型元数据"""
        pass

    @abstractmethod
    def get_adapter_asset(self, adapter_path: str) -> AdapterAsset:
        """【新增】解析适配器目录并返回适配器实体"""
        pass