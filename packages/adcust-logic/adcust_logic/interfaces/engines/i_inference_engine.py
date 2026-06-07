from abc import ABC, abstractmethod
from typing import Generator, Optional, List, Dict
from adcust_logic.models.adapter_asset import AdapterAsset

class IInferenceEngine(ABC):
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """加载底座模型，返回是否成功"""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """物理卸载模型，释放显存"""
        pass

    @abstractmethod
    def switch_adapter(self, adapter: Optional[AdapterAsset]) -> None:
        """切换适配器"""
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Generator[str, None, None]:
        """流式生成文本块"""
        pass
