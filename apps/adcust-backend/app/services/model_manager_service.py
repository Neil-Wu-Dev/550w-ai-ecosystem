import os
import json
from threading import Lock
from app.models.model_asset import ModelAsset
from app.services.interfaces.i_model_manager import IModelManagerService
from app.core.constants import MODEL_LAYER_MAP
from app.models.adapter_asset import AdapterAsset
from typing import Optional


class ModelManagerService(IModelManagerService):
    def __init__(self):
        # 【状态下沉】在服务内部维护当前激活的模型实体，实现单例状态化
        self._current_model: Optional[ModelAsset] = None
        self._lock = Lock()

    def select_model_by_path(self, local_path: str) -> ModelAsset:
        """
        功能：解析物理路径并将其设为系统当前操作的底座模型
        参数：local_path (str) - 模型文件夹的绝对路径
        返回：ModelAsset - 实例化的模型资产对象
        """
        if not os.path.isdir(local_path):
            raise ValueError(f"路径不存在: {local_path}")

        config_path = os.path.join(local_path, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"无效的模型目录，缺少 config.json")

        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        arch = config_data.get("model_type", "unknown").lower()
        prec_raw = config_data.get("torch_dtype", "fp32")
        precision = str(prec_raw).split('.')[-1]
        layers = MODEL_LAYER_MAP.get(arch, [])

        # 实例化对象
        asset = ModelAsset(
            name=os.path.basename(os.path.normpath(local_path)),
            local_path=os.path.abspath(local_path),
            architecture=arch,
            precision=precision,
            trainable_layers=layers
        )

        # 【逻辑归位】直接在内部更新当前激活的模型状态，不再依赖外部 Orchestrator 记录
        with self._lock:
            self._current_model = asset

        return asset

    def get_active_model(self) -> Optional[ModelAsset]:
        """
        功能：供其他 Service（如 Training/Inference）调用的接口，获取当前激活的模型
        返回：ModelAsset 或 None
        """
        return self._current_model

    def get_adapter_asset(self, adapter_path: str) -> AdapterAsset:
        """
        功能：解析适配器（Adapter）的元数据
        参数：adapter_path (str) - 适配器文件夹路径
        返回：AdapterAsset 对象
        """
        if not os.path.isdir(adapter_path):
            raise ValueError(f"适配器路径无效: {adapter_path}")

        config_path = os.path.join(adapter_path, "adapter_config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"缺少 adapter_config.json")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        return AdapterAsset(
            name=os.path.basename(os.path.normpath(adapter_path)),
            local_path=os.path.abspath(adapter_path),
            base_model_name=config.get("base_model_name_or_path", "unknown"),
            strategy=config.get("peft_type", "LoRA"),
            config_params=config
        )