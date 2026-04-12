from abc import ABC, abstractmethod
from typing import Generator, Optional, Dict, Any
from app.models.adapter_asset import AdapterAsset

class IInferenceService(ABC):
    @abstractmethod
    def run_engine(self, model_path: str) -> bool:
        """启动推理引擎，将底座模型加载至显存"""
        pass

    @abstractmethod
    def stop_engine(self) -> None:
        """停止推理引擎，彻底释放显存资源"""
        pass

    @abstractmethod
    def generate_single_path(
        self,
        prompt: str,
        adapter: Optional[AdapterAsset],
        max_tokens: int
    ) -> Generator[str, None, None]:
        """执行单路推理任务（由显存锁保护，支持 AdapterAsset 实体）"""
        pass

    @abstractmethod
    def get_engine_status(self) -> Dict[str, Any]:
        """获取当前引擎运行状态及挂载的适配器元数据"""
        pass