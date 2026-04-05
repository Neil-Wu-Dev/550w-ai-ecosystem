from typing import Dict, Optional


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
        self.name = name
        self.local_path = local_path
        self.base_model_name = base_model_name
        self.strategy = strategy
        self.config_params = config_params
        self.is_active = is_active

        # 预留字段初始化（当前处于注释/隐藏状态）
        # self.parent_id = parent_id
        # self.torch_version = torch_version

    def __repr__(self):
        return f"<AdapterAsset {self.name} (Strategy: {self.strategy}) -> Base: {self.base_model_name}>"