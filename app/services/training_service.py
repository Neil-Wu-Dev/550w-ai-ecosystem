import os
import json
import logging
from datetime import datetime
from typing import List, Optional, Generator, Dict, Any

from app.models.model_asset import ModelAsset
from app.models.dataset_asset import DatasetAsset
from app.services.interfaces.i_training_service import ITrainingService
from app.core.training_engines.sequential_engine import SequentialEngine
from app.core.constants import (
    TrainingMode,
    TrainingStrategy,
    AdapterSigKey,
    SIG_FILENAME
)

logger = logging.getLogger(__name__)

class TrainingService(ITrainingService):
    def __init__(self):
        # 预注册算力引擎，保持单例引用以便后续执行 abort
        # 这里的实现类 SequentialEngine 已经支持了 epochs 参数
        self._engines = {
            TrainingMode.SEQUENTIAL: SequentialEngine(),
        }

    def abort_training(self, mode: TrainingMode = TrainingMode.SEQUENTIAL):
        """
        供外部调用的终止入口。
        通过找到对应的引擎实例并触发其 abort() 方法。
        """
        engine = self._engines.get(mode)
        if engine:
            engine.abort()
            logger.info(f"--- [TrainingService] 已向引擎 {mode} 发送终止信号 ---")

    def customize_new_adapter(
            self,
            base_model: ModelAsset,
            dataset: DatasetAsset,
            target_dir: str,
            trainable_layers: List[str],
            mode: TrainingMode = TrainingMode.SEQUENTIAL,
            custom_name: Optional[str] = None,
            epochs: int = 1  # <--- 【关键修改】新增 epochs 参数，默认值为 1
    ) -> Generator[Dict[str, Any], None, None]:
        """
        重构为生成器函数，实时转发引擎状态，并支持多轮训练控制。
        """

        # 1. 验证模式支持情况
        try:
            self._validate_support(dataset.strategy, mode)
        except Exception as e:
            yield {"status": "error", "message": str(e), "percentage": 0}
            return

        # 2. 准备物理存储环境
        if not custom_name or custom_name.strip() == "":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            source_tag = dataset.source_name.replace(".", "_")
            # 命名中加入 epoch 信息便于识别
            folder_name = f"{base_model.name}_{dataset.strategy}_e{epochs}_{timestamp}"
        else:
            folder_name = custom_name

        # 组合最终的绝对路径
        adapter_path = os.path.abspath(os.path.join(target_dir, folder_name))

        # 3. 获取算力引擎
        engine = self._engines.get(mode)
        if not engine:
            yield {"status": "error", "message": f"模式 {mode} 无可用引擎", "percentage": 0}
            return

        # 4. 执行流式训练
        logger.info(f"--- [AdCust Service] 启动算力引擎: {mode} | 预设轮数: {epochs} ---")

        # 迭代引擎生成的每一个状态字典
        # 【关键修改】将 epochs 参数透传给 engine.train
        for progress in engine.train(
                base_model=base_model,
                dataset=dataset,
                save_path=adapter_path,
                target_modules=trainable_layers,
                epochs=epochs  # <--- 透传参数
        ):
            # 转发状态给上层 (Orchestrator -> Controller)
            yield progress

            # 如果引擎反馈已完成，则在此时持久化签名文件
            if progress.get("status") == "completed":
                self._save_signature(
                    adapter_path,
                    base_model,
                    dataset,
                    trainable_layers,
                    mode,
                    epochs  # 将轮数存入签名
                )

    def _validate_support(self, strategy: str, mode: TrainingMode):
        """逻辑拦截器"""
        # 这里的枚举值判断应与你 app.core.constants 中的定义匹配
        supported_strategies = [TrainingStrategy.KNOWLEDGE_INJECTION, TrainingStrategy.SMOKE_TEST]
        if strategy not in supported_strategies:
            raise NotImplementedError(f"策略 {strategy} 尚未在当前算力引擎中激活。")
        if mode != TrainingMode.SEQUENTIAL:
            raise NotImplementedError(f"目前仅支持 SEQUENTIAL 训练模式。")

    def _save_signature(self, path, base_model, dataset, layers, mode, epochs):
        """持久化 AdCust 签名，记录本次训练的元数据（包含轮数）"""
        signature_content = {
            AdapterSigKey.SIG_VERSION: "1.0",
            AdapterSigKey.BASE_MODEL_NAME: base_model.name,
            AdapterSigKey.BASE_MODEL_PATH: base_model.local_path,
            AdapterSigKey.ARCH: base_model.architecture,
            AdapterSigKey.LAYERS: layers,
            AdapterSigKey.STRATEGY: dataset.strategy,
            AdapterSigKey.DATASET_SOURCE: dataset.source_name,
            AdapterSigKey.CHUNK_SIZE: dataset.chunk_size,
            AdapterSigKey.MODE: mode.value,
            "epochs": epochs,  # 记录实际训练轮数
            AdapterSigKey.CREATED_AT: datetime.now().isoformat(),
            "export_path": path
        }

        # 确保路径存在后再写入签名
        if os.path.exists(path):
            try:
                sig_full_path = os.path.join(path, SIG_FILENAME)
                with open(sig_full_path, "w", encoding="utf-8") as f:
                    json.dump(signature_content, f, indent=4, ensure_ascii=False)
                logger.info(f"--- [TrainingService] 签名文件已写入: {sig_full_path} ---")
            except Exception as e:
                logger.error(f"写入签名文件失败: {str(e)}")