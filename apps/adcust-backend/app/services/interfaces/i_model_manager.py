from abc import ABC, abstractmethod
from typing import Optional
from app.models.model_asset import ModelAsset
from app.models.adapter_asset import AdapterAsset

class IModelManagerService(ABC):
    @abstractmethod
    def select_model_by_path(self, local_path: str) -> ModelAsset:
        """根据路径加载并验证模型元数据，并将其设为当前激活模型"""
        pass

    @abstractmethod
    def get_adapter_asset(self, adapter_path: str) -> AdapterAsset:
        """解析适配器目录并返回适配器实体"""
        pass

    @abstractmethod
    def get_active_model(self) -> Optional[ModelAsset]:
        """【新增】获取当前系统中已选定并激活的模型资产"""
        pass