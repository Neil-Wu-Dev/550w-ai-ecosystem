from app.services.interfaces.i_dataset_service import IDatasetService
from app.models.dataset_asset import DatasetAsset
from app.core.constants import ChunkConfig, TrainingStrategy
from app.core.data_engines.pdf_engines import PdfParser, SimpleTextSplitter
from app.core.data_engines.knowledge_engines import KnowledgeInjectionTemplate, SmokeTestTemplate


class DatasetService(IDatasetService):
    def __init__(self):
        # 组装底层引擎工具
        self._parser = PdfParser()
        self._splitter = SimpleTextSplitter()

        # 策略分发字典：支持未来横向扩展更多策略
        self._templates = {
            TrainingStrategy.KNOWLEDGE_INJECTION: KnowledgeInjectionTemplate(),
            TrainingStrategy.SMOKE_TEST: SmokeTestTemplate()
        }

    def prepare_dataset_asset(
            self,
            file_path: str,
            chunk_size: int,
            strategy: str
    ) -> DatasetAsset:

        # 1. 拦截风险：校验分块长度上限
        actual_size = min(chunk_size, ChunkConfig.HARD_LIMIT)

        # 2. 调用 Parser 引擎提取文本
        raw_text = self._parser.extract_text(file_path)

        # 3. 调用 Splitter 引擎执行 Chunk 处理
        text_chunks = self._splitter.split(raw_text, actual_size)

        # 4. 根据 Strategy 路由到对应的 Template 引擎进行自动化组织语言
        template_worker = self._templates.get(strategy)
        if not template_worker:
            raise NotImplementedError(f"策略 {strategy} 对应的模板引擎尚未注册。")

        formatted_chunks = template_worker.wrap(text_chunks)

        # 5. 生成标准 DatasetAsset 对象
        return DatasetAsset(
            source_name=file_path.split("/")[-1],
            strategy=strategy,
            chunks=formatted_chunks,
            chunk_size=actual_size
        )

    def release_dataset(self, asset: DatasetAsset):
        """执行销毁逻辑，释放 RAM 占用"""
        if asset:
            asset.destroy()