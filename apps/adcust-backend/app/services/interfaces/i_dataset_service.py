from abc import ABC, abstractmethod
from typing import Optional
from app.models.dataset_asset import DatasetAsset

class IDatasetService(ABC):
    @abstractmethod
    def prepare_dataset_asset(
        self,
        file_path: str,
        chunk_size: int,
        strategy: str
    ) -> DatasetAsset:
        """解析文件并生成数据集资产"""
        pass

    @abstractmethod
    def get_active_dataset(self) -> Optional[DatasetAsset]:
        """【新增】获取当前已就绪的数据集资产"""
        pass

    @abstractmethod
    def release_dataset(self):
        """【修改】释放当前激活的数据集内存占用（无需再传入 asset 参数）"""
        pass