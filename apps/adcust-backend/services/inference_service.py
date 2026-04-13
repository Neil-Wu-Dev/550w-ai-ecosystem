from threading import Lock
from typing import Generator, Optional, Dict, Any
from app.services.interfaces.i_inference_service import IInferenceService
from app.core.inference_engines.huggingface_engine import HuggingfaceEngine
from app.models.adapter_asset import AdapterAsset


class InferenceService(IInferenceService):
    def __init__(self):
        # 初始化阶段引擎处于“熄火”状态
        self._engine = HuggingfaceEngine()
        self._gpu_lock = Lock()
        self._current_model_path = None

    def run_engine(self, model_path: str) -> bool:
        """启动引擎：执行底座模型的显存点火"""
        with self._gpu_lock:
            # 规避频繁加载导致的碎片：如果路径一致则跳过物理加载
            if self._current_model_path == model_path:
                print(f"--- [Inference] 引擎已在运行中: {model_path} ---")
                return True

            print(f"--- [Inference] 正在启动引擎，加载底座: {model_path} ---")
            try:
                success = self._engine.load_model(model_path)
                if success:
                    self._current_model_path = model_path
                    return True
            except Exception as e:
                print(f"--- [Inference] 引擎启动失败! 错误: {str(e)} ---")
                raise RuntimeError(f"Engine Start Failed: {str(e)}")
            return False

    def stop_engine(self):
        """关闭引擎：彻底卸载模型，清理显存碎片"""
        with self._gpu_lock:
            if self._current_model_path is None:
                print("--- [Inference] 引擎原本就处于关闭状态 ---")
                return

            self._engine.unload_model()
            self._current_model_path = None
            print("--- [Inference] 引擎已成功关闭，显存已释放 ---")

    def generate_single_path(
            self,
            prompt: str,
            adapter: Optional[AdapterAsset],
            max_tokens: int
    ) -> Generator[str, None, None]:
        """受保护的推理任务：必须在 run_engine 之后执行"""

        # 确保引擎已点火
        if not self._current_model_path:
            raise RuntimeError("拒绝推理：推理引擎尚未启动，请先调用 run_engine。")

        with self._gpu_lock:
            # 1. 切换适配器（传入领域实体对象）
            self._engine.switch_adapter(adapter)

            # 2. 状态反馈：让后台知道当前是谁在说话
            if adapter:
                print(f"--- [Inference] 挂载适配器推理: {adapter.name} [{adapter.strategy}] ---")
            else:
                print(f"--- [Inference] 使用原始底座进行推理 ---")

            # 3. 执行流式生成
            yield from self._engine.generate_stream(prompt, max_tokens)

    def get_engine_status(self) -> Dict[str, Any]:
        """状态查询：导出当前适配器的所有元数据给前端"""
        active_adapter = self._engine.active_adapter
        return {
            "status": "running" if self._current_model_path else "stopped",
            "current_model": self._current_model_path,
            "active_adapter": {
                "name": active_adapter.name,
                "strategy": active_adapter.strategy,
                "base_model": active_adapter.base_model_name,
                "config": active_adapter.config_params,
                "is_active": active_adapter.is_active
            } if active_adapter else None
        }