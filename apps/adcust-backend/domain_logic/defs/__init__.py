"""
领域定义包的统一入口。
将分散在不同物理文件中的定义汇总并向外暴露。
调用者可以统一通过 'from domain_logic.defs import ...' 进行引用。
"""
from .training import (
    TrainingMode,
    TrainingStrategy,
    ChunkConfig,
    InferenceTaskMode,
    InferenceDispatchStrategy
)
from .models_meta import MODEL_LAYER_MAP, DEFAULT_PRECISION
from .signatures import AdapterSigKey, SIG_FILENAME