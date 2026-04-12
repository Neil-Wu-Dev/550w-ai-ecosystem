from datetime import datetime
from typing import List, Optional

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
        self.name = name
        self.local_path = local_path
        self.architecture = architecture
        self.precision = precision
        self.trainable_layers = trainable_layers

    def __repr__(self):
        return f"<ModelAsset {self.name} ({self.architecture}) @ {self.precision}>"