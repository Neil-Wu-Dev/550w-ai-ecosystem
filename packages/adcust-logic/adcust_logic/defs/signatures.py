"""
本文件定义适配器（Adapter）签名文件的持久化标准。
包含：JSON 签名文件中的所有标准字段 Key 以及统一的签名文件名。
用于确保模型资产在存储、读取和身份识别过程中的跨应用一致性。
"""

class AdapterSigKey:
    """签名文件标准字段 (用于识别 Adapter 身份、基座模型及训练背景)"""
    SIG_VERSION = "signature_version"
    BASE_MODEL_NAME = "base_model_name"
    BASE_MODEL_PATH = "base_model_path"
    ARCH = "architecture"
    LAYERS = "trainable_layers"
    STRATEGY = "training_strategy"
    MODE = "training_mode"
    CREATED_AT = "created_at"
    DATASET_SOURCE = "dataset_source"
    CHUNK_SIZE = "chunk_size"

# 统一的签名文件名标准
SIG_FILENAME = "adcust_signature.json"