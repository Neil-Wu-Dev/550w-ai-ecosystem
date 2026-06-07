from typing import Dict, Optional
import os
# 通过 __init__.py 封装后的异常包导入，确保全局拦截生效
from adcust_logic.exceptions import ValidationException

class AdapterAsset:
    """
    适配器资产实体 (Adapter Asset Domain Entity)
    核心逻辑：Base Model 是静止的资产，Adapter 是定制化的“灵魂”。
    """

    def __init__(
            self,
            name: str,  # 适配器的显示名称 (如: gpt2-legal-knowledge-v1)
            local_path: str,  # 适配器文件夹的物理绝对路径
            base_model_name: str,  # 关联的底座模型名称/标识
            strategy: str,  # 训练意图 (如: TONE, KNOWLEDGE, SMOKE_TEST)
            config_params: Dict,  # LoRA 核心参数 (rank, alpha, target_modules)
            is_active: bool = True  # 标记该适配器目前是否可用/物理存在
            # parent_id: Optional[str] = None, # [未来扩展] 用于 SEQUENTIAL 模式，记录父代适配器
            # torch_version: str = "unknown"  # [未来扩展] 记录训练时的 PyTorch 版本，用于兼容性审计
    ):
        # --- [Rich Model Validation Layer] ---
        # 仅仅添加了这一行调用逻辑
        self._validate(name, local_path, base_model_name, strategy)

        # 下面所有赋值一个字都没动
        self.name = name
        self.local_path = local_path
        self.base_model_name = base_model_name
        self.strategy = strategy
        self.config_params = config_params
        self.is_active = is_active

        # 预留字段初始化（保持注释状态，不准动）
        # self.parent_id = parent_id
        # self.torch_version = torch_version

    def _validate(self, name: str, path: str, base_name: str, strategy: str):
        """
        内部校验：严禁手动传异常信息，只传信号 Key 和数据 Context
        """
        # 1. Name 校验
        if not name or not (3 <= len(name) <= 64):
            raise ValidationException("ERR_ADAPTER_NAME_INVALID", name=name)

        # 2. Local Path 校验
        if not path or not (3 <= len(path) <= 255):
            raise ValidationException("ERR_PATH_INVALID", path=path)

        # 3. Base Model Name 校验
        if not base_name or not base_name.strip():
            raise ValidationException("ERR_BASE_MODEL_REQUIRED")

        # 4. Strategy 校验
        valid_strategies = {"TONE", "KNOWLEDGE", "SMOKE_TEST", "LORA", "QLORA"}
        if strategy.upper() not in valid_strategies:
            raise ValidationException("ERR_STRATEGY_UNKNOWN", strategy=strategy)

    def __repr__(self):
        return f"<AdapterAsset {self.name} (Strategy: {self.strategy}) -> Base: {self.base_model_name}>"
