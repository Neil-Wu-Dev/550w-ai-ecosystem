"""
本文件定义 AI 训练与推理的业务逻辑协议。
包含：训练组织模式、意图策略、任务分发模式以及数据处理的相关配置。
这是领域逻辑的“开关”和“规格说明书”。
"""
from enum import Enum

class TrainingMode(str, Enum):
    """训练模式：决定任务的物理组织结构"""
    SEQUENTIAL = "sequel"  # 路径 A: 叠罗汉 (串行) - [PoC 阶段首选]

    # 路径 B：全家桶模式（混合训练）
    JOINT_MIXED = "joint_mixed"

    # 路径 C：分布式插件模式（多适配器）
    MULTI_ADAPTER = "multi_adapter"

class TrainingStrategy(str, Enum):
    """训练策略：意图驱动，决定语料格式与算法"""
    SMOKE_TEST = "smoke"              # 链路验证
    KNOWLEDGE_INJECTION = "knowledge" # 知识注入

class ChunkConfig:
    """数据分片配置：影响显存占用与训练粒度"""
    RECOMMENDED = [256, 512, 1024]
    HARD_LIMIT = 2048
    DEFAULT = 512

class InferenceTaskMode(str, Enum):
    """推理任务模式"""
    SINGLE = "single"
    COMPARE = "compare"

class InferenceDispatchStrategy(str, Enum):
    """推理分发策略"""
    SERIAL = "serial"
    INTERLEAVED = "interleaved"