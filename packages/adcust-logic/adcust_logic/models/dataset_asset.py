from typing import List, Dict, Any
# 严格通过封装后的异常包导入，确保 main.py 拦截器能够捕获
from adcust_logic.exceptions import ValidationException

class DatasetAsset:
    """
    语料资产：PDF 解析后的标准化产物。
    当前设计为内存驻留，支持手动销毁以确保显存安全。
    """
    def __init__(
        self,
        source_name: str,
        strategy: str,
        chunks: List[Dict[str, Any]],
        chunk_size: int
    ):
        # --- [Rich Model Validation Layer] ---
        # 挂载点：在赋值前执行强制校验，确保非法数据无法进入内存
        self._validate(source_name, strategy, chunks, chunk_size)

        # 属性与赋值：一个字都没动，完全保留原始结构
        self.source_name = source_name
        self.strategy = strategy
        self.chunks = chunks
        self.chunk_size = chunk_size

    def _validate(self, name: str, strategy: str, chunks: List[Dict[str, Any]], size: int):
        """
        内部校验：严禁手写英文报错，只传递信号 Key 和上下文数据
        """
        # 1. Source Name 校验
        if not name or not (2 <= len(name) <= 128):
            raise ValidationException("ERR_DATASET_NAME_INVALID", name=name)

        # 2. Strategy 校验
        if not strategy:
            raise ValidationException("ERR_STRATEGY_MISSING")

        # 3. Chunks 校验 (语料资产核心，必须存在)
        if not chunks or len(chunks) == 0:
            raise ValidationException("ERR_CHUNKS_EMPTY", source=name)

        # 4. Chunk Size 校验 (工业标准 1-8192)
        if size <= 0 or size > 8192:
            raise ValidationException("ERR_CHUNK_SIZE_OUT_OF_RANGE", size=size)

    def destroy(self):
        """
        内存回收器：显式释放引用。
        在训练完成或程序销毁时由 Service 调用。
        """
        if self.chunks is not None:
            source_name_snapshot = self.source_name # 保持原逻辑
            self.chunks.clear() # 清空列表元素
            self.chunks = None  # 切断引用，告知垃圾回收器 (GC)
            print(f"--- [AdCust Asset Guard] 已彻底销毁语料资产内存: {source_name_snapshot} ---")

    def __repr__(self):
        chunk_count = len(self.chunks) if self.chunks is not None else 0
        return f"<DatasetAsset {self.source_name} | Strategy: {self.strategy} | Chunks: {chunk_count}>"