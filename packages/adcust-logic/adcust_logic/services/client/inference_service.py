# -*- coding: utf-8 -*-
import json
import os
from threading import Lock
from typing import Any, Dict, Generator, List, Optional

from adcust_logic.defs import SIG_FILENAME
from adcust_logic.exceptions import BusinessException
from adcust_logic.interfaces.services.client.i_inference_service import IInferenceService
from adcust_logic.interfaces.services.client.i_model_manager import IModelManagerService
from adcust_logic.models.adapter_asset import AdapterAsset
from adcust_logic.services.client.model_manifest_service import ModelManifestService
from adcust_logic.services.engines.client.inference_engines.huggingface_engine import HuggingfaceEngine


class InferenceService(IInferenceService):
    """管理本地底座模型、adapter 槽位和流式推理生命周期。"""

    def __init__(self, model_manager: IModelManagerService):
        self._model_manager = model_manager
        self._engine = HuggingfaceEngine()
        self._gpu_lock = Lock()
        self._current_model_path: Optional[str] = None
        self._manifest_service = ModelManifestService()
        self._active_adapters: Dict[str, AdapterAsset] = {}

    def mount_adapter(
        self,
        adapter_path: str,
        slot_id: str,
        base_model_path: Optional[str] = None,
    ) -> AdapterAsset:
        """校验并挂载 adapter。

        模型未启动时使用显式底座路径完成预挂载；模型已启动时立即执行
        PEFT 加载，确保接口返回成功时 adapter 已经真正可用。
        """
        validation_model_path = self._current_model_path or base_model_path
        if not validation_model_path:
            raise RuntimeError("Select a local base model folder before mounting an adapter.")

        adapter = self._model_manager.get_adapter_asset(adapter_path)
        signature = self._assert_adapter_signature_matches_model(adapter_path, validation_model_path)
        adapter.base_model_name = (
            signature.get("base_model")
            or signature.get("base_model_name_or_path")
            or signature.get("base_model_manifest", {}).get("repository_id")
            or adapter.base_model_name
        )

        if self._current_model_path:
            with self._gpu_lock:
                self._engine.switch_adapter(adapter)
        self._active_adapters[slot_id] = adapter
        return adapter

    def _assert_adapter_signature_matches_model(
        self,
        adapter_path: str,
        model_path: str,
    ) -> Dict[str, Any]:
        signature_path = os.path.join(adapter_path, SIG_FILENAME)
        if not os.path.exists(signature_path):
            raise BusinessException("ERR_ADAPTER_SIGNATURE_MISSING", path=signature_path)
        with open(signature_path, "r", encoding="utf-8") as file:
            signature = json.load(file)

        expected_manifest = signature.get("base_model_manifest") or signature.get("expected_base_model_manifest")
        if not expected_manifest:
            raise BusinessException("ERR_ADAPTER_BASE_MANIFEST_MISSING", path=signature_path)

        active_manifest = self._manifest_service.build_local_manifest(
            model_path,
            str(expected_manifest.get("repository_id") or signature.get("base_model") or ""),
        )
        if active_manifest.get("combined_hash") != expected_manifest.get("combined_hash"):
            raise BusinessException(
                "ERR_ADAPTER_BASE_MODEL_MISMATCH",
                adapter_hash=expected_manifest.get("combined_hash"),
                active_hash=active_manifest.get("combined_hash"),
            )
        return signature

    def unmount_adapter(self, slot_id: str) -> None:
        """卸载指定槽位；模型关闭时也允许清除预挂载状态。"""
        removed = self._active_adapters.pop(slot_id, None)
        if not self._current_model_path:
            return
        with self._gpu_lock:
            if not self._active_adapters:
                self._engine.switch_adapter(None)
            elif removed and self._engine.active_adapter is removed:
                self._engine.switch_adapter(next(iter(self._active_adapters.values())))

    def run_engine(self, model_path: str) -> bool:
        """加载底座模型，并激活已经完成校验的预挂载 adapter。"""
        with self._gpu_lock:
            if self._current_model_path == model_path:
                return True

            for adapter in self._active_adapters.values():
                self._assert_adapter_signature_matches_model(adapter.local_path, model_path)

            self._model_manager.select_model_by_path(model_path)
            success = self._engine.load_model(model_path)
            if success:
                try:
                    if self._active_adapters:
                        self._engine.switch_adapter(next(iter(self._active_adapters.values())))
                    self._current_model_path = model_path
                except Exception:
                    self._engine.unload_model()
                    self._current_model_path = None
                    raise
            return success

    def stop_engine(self) -> None:
        """停止生成、卸载模型并清除所有 adapter 槽位。"""
        if hasattr(self._engine, "request_stop"):
            self._engine.request_stop()
        with self._gpu_lock:
            self._engine.unload_model()
            self._current_model_path = None
            self._active_adapters.clear()

    def request_stop_generation(self) -> None:
        """请求当前生成尽快停止，不卸载底座模型或 adapter。"""
        if hasattr(self._engine, "request_stop"):
            self._engine.request_stop()

    def generate_single_path(
        self,
        prompt: str,
        slot_id: str,
        max_tokens: int,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Generator[str, None, None]:
        """切换到指定槽位的 adapter 后执行流式推理。"""
        if not self._current_model_path:
            raise RuntimeError("Inference engine is not running.")

        adapter = self._active_adapters.get(slot_id)
        with self._gpu_lock:
            self._engine.switch_adapter(adapter)
            yield from self._engine.generate_stream(
                prompt,
                max_tokens,
                system_prompt=system_prompt,
                history=history,
            )

    def get_engine_status(self) -> Dict[str, Any]:
        """返回模型状态以及各槽位真实挂载信息。"""
        return {
            "status": "running" if self._current_model_path else "stopped",
            "active_slots": list(self._active_adapters.keys()),
            "adapters": {
                slot_id: {
                    "name": adapter.name,
                    "path": adapter.local_path,
                    "base_model": adapter.base_model_name,
                }
                for slot_id, adapter in self._active_adapters.items()
            },
        }
