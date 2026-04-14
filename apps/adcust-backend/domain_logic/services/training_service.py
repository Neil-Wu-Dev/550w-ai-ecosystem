import os
import json
import logging
from datetime import datetime
from typing import List, Optional, Generator, Dict, Any

from domain_logic.interfaces.services.i_training_service import ITrainingService
from domain_logic.interfaces.services.i_model_manager import IModelManagerService
from domain_logic.interfaces.services.i_dataset_service import IDatasetService
from domain_logic.services.engines.training_engines.sequential_engine import SequentialEngine
from domain_logic.defs import TrainingMode, SIG_FILENAME

logger = logging.getLogger(__name__)


class TrainingService(ITrainingService):
    def __init__(self, model_manager: IModelManagerService, dataset_service: IDatasetService):
        # 【接口注入】直接依赖其他服务获取必要资产，不再通过 Orchestrator 传参
        self._model_manager = model_manager
        self._dataset_service = dataset_service
        self._engines = {
            TrainingMode.SEQUENTIAL: SequentialEngine(),
        }

    def customize_new_adapter(
            self,
            target_dir: str,
            trainable_layers: List[str],
            mode: TrainingMode = TrainingMode.SEQUENTIAL,
            custom_name: Optional[str] = None,
            epochs: int = 1
    ) -> Generator[Dict[str, Any], None, None]:
        """
        功能：核心训练流。包含：前置资产检查、存储环境准备、流式训练、以及【核心下沉】的最终清理。
        参数：target_dir (str)-保存目录, trainable_layers (List)-层级, epochs (int)-轮数等
        返回：生成器，逐条返回训练进度字典
        """
        # 【逻辑归位】从注入的服务中动态获取资产状态
        base_model = self._model_manager.get_active_model()
        dataset = self._dataset_service.get_active_dataset()

        # 前置业务检查
        if not base_model or not dataset:
            yield {"status": "error", "message": "底座模型或数据集未加载", "percentage": 0}
            return

        try:
            # 准备路径逻辑
            self._validate_support(dataset.strategy, mode)
            folder_name = custom_name or f"{base_model.name}_{dataset.strategy}_e{epochs}_{datetime.now().strftime('%H%M%S')}"
            adapter_path = os.path.abspath(os.path.join(target_dir, folder_name))

            engine = self._engines.get(mode)

            # 执行流式训练
            for progress in engine.train(
                    base_model=base_model,
                    dataset=dataset,
                    save_path=adapter_path,
                    target_modules=trainable_layers,
                    epochs=epochs
            ):
                yield progress
                if progress.get("status") == "completed":
                    self._save_signature(adapter_path, base_model, dataset, trainable_layers, mode, epochs)

        except Exception as e:
            logger.error(f"训练失败: {str(e)}")
            yield {"status": "error", "message": str(e), "percentage": 0}

        finally:
            # 【逻辑下沉】原 Orchestrator 里的清理逻辑，现在由 TrainingService 结束后自我闭环
            self._dataset_service.release_dataset()
            logger.info("--- [TrainingService] 训练结束，已自动触发数据集资源回收 ---")

    def abort_training(self, mode: TrainingMode = TrainingMode.SEQUENTIAL):
        """中止指令下发"""
        engine = self._engines.get(mode)
        if engine: engine.abort()

    def _validate_support(self, strategy: str, mode: TrainingMode):
        if mode != TrainingMode.SEQUENTIAL: raise NotImplementedError("仅支持顺序模式")

    def _save_signature(self, path, base_model, dataset, layers, mode, epochs):
        # 签名持久化逻辑（保持原样）
        sig = {"base": base_model.name, "strategy": dataset.strategy, "epochs": epochs}
        with open(os.path.join(path, SIG_FILENAME), "w") as f: json.dump(sig, f)