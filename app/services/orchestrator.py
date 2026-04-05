import logging
import json
from typing import List, Optional, Dict, Any, Generator
from app.models.model_asset import ModelAsset
from app.models.dataset_asset import DatasetAsset
from app.models.adapter_asset import AdapterAsset
from app.services.interfaces.i_orchestrator import IOrchestrator
from app.services.interfaces.i_model_manager import IModelManagerService
from app.services.interfaces.i_dataset_service import IDatasetService
from app.services.interfaces.i_training_service import ITrainingService
from app.services.interfaces.i_inference_service import IInferenceService
from app.core.constants import TrainingMode, TrainingStrategy, ChunkConfig, InferenceDispatchStrategy

logger = logging.getLogger(__name__)


class Orchestrator(IOrchestrator):
    def __init__(self, model_manager: IModelManagerService, dataset_service: IDatasetService,
                 training_service: ITrainingService, inference_service: IInferenceService):
        self._model_manager = model_manager
        self._dataset_service = dataset_service
        self._training_service = training_service
        self._inference_service = inference_service
        self._current_model: Optional[ModelAsset] = None
        self._current_dataset: Optional[DatasetAsset] = None
        self._active_adapters: Dict[str, AdapterAsset] = {}

    def get_static_options(self) -> Dict[str, Any]:
        return {
            "strategies": [s.value for s in TrainingStrategy],
            "modes": [m.value for m in TrainingMode],
            "chunk_options": {
                "recommended": ChunkConfig.RECOMMENDED,
                "default": ChunkConfig.DEFAULT,
                "hard_limit": ChunkConfig.HARD_LIMIT
            }
        }

    def set_base_model(self, local_path: str) -> ModelAsset:
        model_asset = self._model_manager.select_model_by_path(local_path)
        self._current_model = model_asset
        return model_asset

    def prepare_training_data(self, file_path: str, chunk_size: int, strategy: str) -> Dict[str, Any]:
        if self._current_dataset:
            self._dataset_service.release_dataset(self._current_dataset)

        dataset_asset = self._dataset_service.prepare_dataset_asset(file_path, chunk_size, strategy)
        self._current_dataset = dataset_asset
        return {
            "source_name": dataset_asset.source_name,
            "chunk_count": len(dataset_asset.chunks),
            "status": "ready"
        }

    # --- 核心重构：训练执行流（支持 Epochs） ---
    def execute_customization(
            self,
            target_dir: str,
            trainable_layers: List[str],
            mode: str,
            custom_name: Optional[str] = None,
            epochs: int = 1  # <--- 【关键修改】新增参数，接收来自 Controller 的训练轮数
    ) -> Generator[Dict[str, Any], None, None]:
        """
        编排层：实时消费并转发 TrainingService 的训练状态，支持多轮训练参数传递。
        """
        if not self._current_model or not self._current_dataset:
            yield {"status": "error", "message": "资产（模型或数据集）未就绪，请检查是否已选择底座并上传数据",
                   "percentage": 0}
            return

        try:
            logger.info(f"--- [Orchestrator] 发起定制任务: {custom_name} | Epochs: {epochs} ---")

            # 建立引擎流，透传 epochs 参数
            training_stream = self._training_service.customize_new_adapter(
                base_model=self._current_model,
                dataset=self._current_dataset,
                target_dir=target_dir,
                trainable_layers=trainable_layers,
                mode=TrainingMode(mode),
                custom_name=custom_name,
                epochs=epochs  # <--- 【关键修改】向下透传给 TrainingService
            )

            # 实时转发引擎传来的每一个状态字典 (包含 status, message, percentage, epoch 等)
            for progress in training_stream:
                yield progress

        except Exception as e:
            logger.error(f"Orchestrator 捕获训练流程异常: {str(e)}")
            yield {"status": "error", "message": f"编排调度异常: {str(e)}", "percentage": 0}

        finally:
            # 训练结束（无论成功、失败或中止），清理数据集资源以释放内存
            if self._current_dataset:
                self._dataset_service.release_dataset(self._current_dataset)
                self._current_dataset = None
                logger.info("--- [Orchestrator] 训练会话资源已释放 ---")

    def stop_customization(self, mode: str = "sequential") -> Dict[str, str]:
        """
        从编排层直接切断算力引擎。
        """
        try:
            self._training_service.abort_training(mode=TrainingMode(mode))
            return {"status": "success", "message": "终止指令已成功发送至算力引擎"}
        except Exception as e:
            return {"status": "error", "message": f"终止操作失败: {str(e)}"}

    # --- 适配器挂载逻辑（软校验架构兼容性） ---
    def load_adapter_asset(self, adapter_path: str, slot_id: str) -> AdapterAsset:
        if not self._current_model:
            raise RuntimeError("拒绝挂载：推理引擎底座尚未选定，请先加载底座模型。")

        # 1. 获取适配器实体
        adapter_entity = self._model_manager.get_adapter_asset(adapter_path)

        # 2. 架构兼容性软校验
        current_arch = self._current_model.architecture.lower()
        target_base = adapter_entity.base_model_name.lower()

        # 只要架构大类一致（如都包含 'llama'），则允许挂载，解决路径不一致问题
        is_compatible = current_arch in target_base or target_base in current_arch

        if not is_compatible:
            logger.error(f"不兼容：当前底座为 {current_arch}，适配器对应 {target_base}")
            raise ValueError(f"适配器不兼容: 架构冲突 ({current_arch} vs {target_base})")

        # 3. 记录并激活适配器
        logger.info(f"--- [Orchestrator] 适配器已就绪并挂载至 Slot: {slot_id} ---")
        self._active_adapters[slot_id] = adapter_entity
        return adapter_entity

    # --- 推理逻辑 ---
    def boot_inference_engine(self) -> str:
        if not self._current_model: raise ValueError("未选定底座")
        success = self._inference_service.run_engine(self._current_model.local_path)
        return f"Model {self._current_model.name} running." if success else "Boot failed."

    def shutdown_inference_engine(self) -> None:
        self._inference_service.stop_engine()
        self._active_adapters.clear()

    def stream_chat(self, targets: List[Dict[str, Any]], prompt: str, strategy: InferenceDispatchStrategy,
                    max_tokens: int = 512) -> Generator[str, None, None]:
        status = self._inference_service.get_engine_status()
        if status["status"] != "running": raise RuntimeError("推理引擎未运行")

        if strategy == InferenceDispatchStrategy.SERIAL:
            for t in targets:
                adapter_obj = self._active_adapters.get(t["id"])
                for token in self._inference_service.generate_single_path(prompt, adapter_obj, max_tokens):
                    yield json.dumps({"source": t["id"], "token": token, "is_final": False}) + "\n"
                yield json.dumps({"source": t["id"], "token": "", "is_final": True}) + "\n"

        elif strategy == InferenceDispatchStrategy.INTERLEAVED:
            gens = []
            for t in targets:
                adapter_obj = self._active_adapters.get(t["id"])
                gens.append({"id": t["id"], "active": True,
                             "gen": self._inference_service.generate_single_path(prompt, adapter_obj, max_tokens)})

            while any(g["active"] for g in gens):
                for g in gens:
                    if not g["active"]: continue
                    try:
                        for _ in range(2):
                            token = next(g["gen"])
                            yield json.dumps({"source": g["id"], "token": token, "is_final": False}) + "\n"
                    except StopIteration:
                        g["active"] = False
                        yield json.dumps({"source": g["id"], "token": "", "is_final": True}) + "\n"