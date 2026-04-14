from threading import Lock
from typing import Generator, Dict, Any
from domain_logic.interfaces.services.i_inference_service import IInferenceService
from domain_logic.interfaces.services.i_model_manager import IModelManagerService
from domain_logic.services.engines.inference_engines.huggingface_engine import HuggingfaceEngine
from domain_logic.models.adapter_asset import AdapterAsset


class InferenceService(IInferenceService):
    def __init__(self, model_manager: IModelManagerService):
        self._model_manager = model_manager
        self._engine = HuggingfaceEngine()
        self._gpu_lock = Lock()
        self._current_model_path = None
        # 【状态下沉】管理槽位与适配器的映射关系
        self._active_adapters: Dict[str, AdapterAsset] = {}

    def mount_adapter(self, adapter_path: str, slot_id: str) -> AdapterAsset:
        """
        功能：【逻辑下沉】适配器挂载业务。包含从路径加载资产、架构兼容性软校验、记录挂载状态。
        参数：adapter_path (str)-物理路径, slot_id (str)-槽位ID
        返回：AdapterAsset 对象
        """
        # 1. 检查底座状态
        active_model = self._model_manager.get_active_model()
        if not active_model:
            raise RuntimeError("拒绝挂载：必须先加载底座模型。")

        # 2. 调用 ModelManager 获取元数据
        adapter_entity = self._model_manager.get_adapter_asset(adapter_path)

        # 3. 【逻辑下沉】原 Orchestrator 里的软校验逻辑
        cur_arch = active_model.architecture.lower()
        tgt_base = adapter_entity.base_model_name.lower()
        if cur_arch not in tgt_base and tgt_base not in cur_arch:
            raise ValueError(f"架构不兼容: {cur_arch} 不能挂载为 {tgt_base} 的适配器")

        # 4. 记录状态
        self._active_adapters[slot_id] = adapter_entity
        return adapter_entity

    def run_engine(self, model_path: str) -> bool:
        """点火加载底座显存"""
        with self._gpu_lock:
            if self._current_model_path == model_path: return True
            success = self._engine.load_model(model_path)
            if success: self._current_model_path = model_path
            return success

    def stop_engine(self):
        """熄火释放显存"""
        with self._gpu_lock:
            self._engine.unload_model()
            self._current_model_path = None
            self._active_adapters.clear()

    def generate_single_path(self, prompt: str, slot_id: str, max_tokens: int) -> Generator[str, None, None]:
        """执行特定槽位的推理逻辑"""
        if not self._current_model_path: raise RuntimeError("引擎未启动")

        # 从内部映射中获取适配器实体
        adapter = self._active_adapters.get(slot_id)
        with self._gpu_lock:
            self._engine.switch_adapter(adapter)
            yield from self._engine.generate_stream(prompt, max_tokens)

    def get_engine_status(self) -> Dict[str, Any]:
        """导出引擎实时状态"""
        active_adapter = self._engine.active_adapter
        return {
            "status": "running" if self._current_model_path else "stopped",
            "active_slots": list(self._active_adapters.keys())
        }