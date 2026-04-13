import os
import json
from app.models.model_asset import ModelAsset
from app.services.interfaces.i_model_manager import IModelManagerService
# 引入我们刚才定义的字典
from app.core.constants import MODEL_LAYER_MAP
from app.models.adapter_asset import AdapterAsset

class ModelManagerService(IModelManagerService):
    def select_model_by_path(self, local_path: str) -> ModelAsset:
        # 1. 物理检查：确保路径存在且是一个目录
        if not os.path.isdir(local_path):
            raise ValueError(f"Path not found or is not a directory: {local_path}")

        # 2. 识别 config.json
        config_path = os.path.join(local_path, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Not a valid model directory: config.json missing at {local_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"config.json is corrupted: {config_path}")

        # 3. 提取核心元数据
        # 模型架构 (例如: "llama")
        arch = config_data.get("model_type", "unknown").lower()

        # 提取精度 (处理 torch_dtype 字段，如 "float16")
        prec_raw = config_data.get("torch_dtype", "fp32")
        # 简单清洗：把 "torch.float16" 变成 "float16"
        precision = str(prec_raw).split('.')[-1]

        # 4. 【核心逻辑】根据 Constants 里的字典进行匹配
        # 如果模型架构在字典里，就拿走它所有的黄金层级名；
        # 如果不在字典里，返回空列表，由前端提醒用户手动输入或不支持
        layers = MODEL_LAYER_MAP.get(arch, [])

        # 5. 实例化并返回对象
        # 使用 os.path.normpath 确保不同系统的路径分隔符一致
        return ModelAsset(
            name=os.path.basename(os.path.normpath(local_path)),
            local_path=os.path.abspath(local_path),  # 存绝对路径最稳
            architecture=arch,
            precision=precision,
            trainable_layers=layers
        )

    def get_adapter_asset(self, adapter_path: str) -> AdapterAsset:
        """实现具体的适配器元数据解析"""
        if not os.path.isdir(adapter_path):
            raise ValueError(f"适配器路径无效: {adapter_path}")

        config_path = os.path.join(adapter_path, "adapter_config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"不是合法的适配器目录: 缺少 adapter_config.json")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 假设 AdapterAsset 接收这些参数，请根据你的 AdapterAsset 定义调整
        return AdapterAsset(
            name=os.path.basename(os.path.normpath(adapter_path)),
            local_path=os.path.abspath(adapter_path),
            base_model_name=config.get("base_model_name_or_path", "unknown"),
            strategy=config.get("peft_type", "LoRA"),
            config_params=config
        )