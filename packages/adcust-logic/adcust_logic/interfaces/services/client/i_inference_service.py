from abc import ABC, abstractmethod
from typing import Generator, Dict, Any, Optional, List
from adcust_logic.models.adapter_asset import AdapterAsset

class IInferenceService(ABC):
    @abstractmethod
    def run_engine(self, model_path: str) -> bool:
        """加载底座模型至显存"""
        pass

    @abstractmethod
    def stop_engine(self) -> None:
        """停止引擎并释放显存"""
        pass

    @abstractmethod
    def request_stop_generation(self) -> None:
        """请求中断当前生成，但不卸载已加载的底座模型。"""
        pass

    @abstractmethod
    def mount_adapter(
        self,
        adapter_path: str,
        slot_id: str,
        base_model_path: Optional[str] = None,
    ) -> AdapterAsset:
        """【新增】将指定路径的适配器挂载到逻辑槽位，并执行架构兼容性检查"""
        pass

    @abstractmethod
    def unmount_adapter(self, slot_id: str) -> None:
        """卸载指定槽位的适配器"""
        pass

    @abstractmethod
    def generate_single_path(
        self,
        prompt: str,
        slot_id: str, # 【修改】通过 slot_id 调用，而非直接传对象
        max_tokens: int,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Generator[str, None, None]:
        """执行单路推理"""
        pass

    @abstractmethod
    def get_engine_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        pass
