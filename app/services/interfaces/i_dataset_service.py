from abc import ABC, abstractmethod
from app.models.dataset_asset import DatasetAsset

class IDatasetService(ABC):
    @abstractmethod
    def prepare_dataset_asset(
        self,
        file_path: str,
        chunk_size: int,
        strategy: str
    ) -> DatasetAsset:
        pass

    @abstractmethod
    def release_dataset(self, asset: DatasetAsset):
        pass