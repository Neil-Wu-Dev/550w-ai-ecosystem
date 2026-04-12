from typing import List, Dict, Any
from app.core.constants import TrainingStrategy

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
        self.source_name = source_name
        self.strategy = strategy
        self.chunks = chunks
        self.chunk_size = chunk_size

    def destroy(self):
        """
        内存回收器：显式释放引用。
        在训练完成或程序销毁时由 Service 调用。
        """
        if self.chunks is not None:
            self.chunks.clear() # 清空列表元素
            self.chunks = None  # 切断引用，告知垃圾回收器 (GC)
            print(f"--- [AdCust Asset Guard] 已彻底销毁语料资产内存: {self.source_name} ---")