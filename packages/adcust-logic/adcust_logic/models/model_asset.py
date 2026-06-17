from datetime import datetime
from typing import List, Optional
# 从你修好的 __init__.py 导入，确保能被 main.py 拦截
from adcust_logic.exceptions import ValidationException

class ModelAsset:
    """
    模型资产实体（纯内存/业务逻辑对象）
    用于在 AdCust 微服务之间传递底座模型的核心物理属性
    """
    def __init__(
        self,
        name: str,                # 目录名或模型名
        local_path: str,          # 硬盘绝对路径
        architecture: str,        # 官方标准化架构名 (如 "gpt2", "llama")
        precision: str,           # 量化/精度信息 (如 "fp32", "fp16", "int4", "int8")
        trainable_layers: List[str] # 该模型可用于调用或注入训练的层级列表 (如 ["c_attn", "c_proj"])
    ):
        # --- 3. 注入 Validator 调用 (点火) ---
        self._validate(name, local_path, architecture, precision, trainable_layers)

        # --- 2. 绝对不准动赋值 (原样保留) ---
        self.name = name
        self.local_path = local_path
        self.architecture = architecture
        self.precision = precision
        self.trainable_layers = trainable_layers

    # --- 1. 属性校验逻辑 (绝对不含手写报错信息) ---
    def _validate(self, name: str, path: str, arch: str, prec: str, layers: List[str]):
        """
        内部校验：只抛出 Key 信号。
        """
        if not name or not (3 <= len(name) <= 64):
            raise ValidationException("ERR_MODEL_NAME_INVALID", name=name)

        if not path or not (3 <= len(path) <= 255):
            raise ValidationException("ERR_PATH_INVALID", path=path)

        if not arch or not (2 <= len(arch) <= 32):
            raise ValidationException("ERR_ARCH_UNSUPPORTED", arch=arch)

        if not prec:
            raise ValidationException("ERR_PRECISION_MISSING")

        if not layers or len(layers) == 0:
            raise ValidationException("ERR_LAYERS_EMPTY")

    def __repr__(self):
        return f"<ModelAsset {self.name} ({self.architecture}) @ {self.precision}>"