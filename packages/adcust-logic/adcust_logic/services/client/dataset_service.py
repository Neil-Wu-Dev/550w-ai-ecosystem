from typing import Optional
from adcust_logic.interfaces.services.client.i_dataset_service import IDatasetService
from adcust_logic.models.dataset_asset import DatasetAsset
from adcust_logic.defs import ChunkConfig, TrainingStrategy
from adcust_logic.services.engines.client.data_engines.pdf_engines import PdfParser, SimpleTextSplitter
from adcust_logic.services.engines.client.data_engines.knowledge_engines import KnowledgeInjectionTemplate, SmokeTestTemplate


class DatasetService(IDatasetService):
    def __init__(self):
        self._parser = PdfParser()
        self._splitter = SimpleTextSplitter()
        # 【状态下沉】管理当前加载的数据集资产
        self._active_dataset: Optional[DatasetAsset] = None

        self._templates = {
            TrainingStrategy.KNOWLEDGE_INJECTION: KnowledgeInjectionTemplate(),
            TrainingStrategy.SMOKE_TEST: SmokeTestTemplate()
        }

    def prepare_dataset_asset(self, file_path: str, chunk_size: int, strategy: str) -> DatasetAsset:
        """
        功能：处理文件并生成数据集，同时负责旧资源的自动清理
        参数：file_path (str)-文件路径, chunk_size (int)-切片大小, strategy (str)-策略名
        返回：DatasetAsset 对象
        """
        # 【逻辑下沉】如果当前已有数据集，在加载新数据前自动执行销毁释放 RAM
        if self._active_dataset:
            self.release_dataset()

        actual_size = min(chunk_size, ChunkConfig.HARD_LIMIT)
        raw_text = self._parser.extract_text(file_path)
        text_chunks = self._splitter.split(raw_text, actual_size)

        template_worker = self._templates.get(strategy)
        if not template_worker:
            raise NotImplementedError(f"策略 {strategy} 尚未注册。")

        formatted_chunks = template_worker.wrap(text_chunks)

        asset = DatasetAsset(
            source_name=file_path.split("/")[-1],
            strategy=strategy,
            chunks=formatted_chunks,
            chunk_size=actual_size
        )

        # 记录到当前状态
        self._active_dataset = asset
        return asset

    def get_active_dataset(self) -> Optional[DatasetAsset]:
        """获取当前就绪的数据集资产"""
        return self._active_dataset

    def release_dataset(self):
        """
        功能：显式销毁当前数据集并清空引用，释放内存
        """
        if self._active_dataset:
            self._active_dataset.destroy()
            self._active_dataset = None