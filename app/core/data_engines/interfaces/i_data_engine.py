from abc import ABC, abstractmethod
from typing import List, Dict, Any

class IDataParser(ABC):
    """文本提取标准接口"""
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        pass

class ITextSplitter(ABC):
    """文本切分标准接口"""
    @abstractmethod
    def split(self, text: str, chunk_size: int) -> List[str]:
        pass

class ITemplateEngine(ABC):
    """策略包装标准接口"""
    @abstractmethod
    def wrap(self, chunks: List[str]) -> List[Dict[str, Any]]:
        pass