from enum import Enum

class TrainingMode(str, Enum):
    """
    训练模式：决定任务的物理组织结构
    """
    SEQUENTIAL = "sequel"  # 路径 A: 叠罗汉 (串行) - [PoC 阶段首选]
    # BLENDED = "blended"      # 路径 B: 混合 (未来扩展)
    # MODULAR = "modular"      # 路径 C: 多插件 (未来扩展)

class TrainingStrategy(str, Enum):
    """
    训练策略：意图驱动，决定语料格式与算法
    """
    SMOKE_TEST = "smoke"              # 链路验证 (PoC 阶段唯一激活项)
    # TONE_ALIGNMENT = "tone"         # 语气校准
    KNOWLEDGE_INJECTION = "knowledge" # 知识注入
    # LOGIC_ALIGNMENT = "logic"       # 逻辑塑造
    # BEHAVIOR_ALIGNMENT = "behavior" # 行为刻画
    # VALUE_ALIGNMENT = "value"       # 价值观塑造




# --- 2. 签名文件标准字段 (用于 Adapter 身份识别) ---
# 这样你以后改字段名，只需要改这里，不用满地找字符串
class AdapterSigKey:
    SIG_VERSION = "signature_version"
    BASE_MODEL_NAME = "base_model_name"
    BASE_MODEL_PATH = "base_model_path"
    ARCH = "architecture"
    LAYERS = "trainable_layers"
    STRATEGY = "training_strategy"
    MODE = "training_mode"
    CREATED_AT = "created_at"

    # --- 核心修复：添加以下两个缺失的属性 ---
    DATASET_SOURCE = "dataset_source"
    CHUNK_SIZE = "chunk_size"

SIG_FILENAME = "adcust_signature.json"  # 签名文件名也存这里









# 模型可训练层级字典=======================================================

MODEL_LAYER_MAP = {
    # --- DeepSeek 系列 (包含 V2, V3, MoE) ---
    # q_a_proj, q_b_proj 是 DeepSeek MLA 架构特有的压缩层
    "deepseek_v2": ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj", "gate_proj", "up_proj",
                    "down_proj"],
    "deepseek": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 早期或兼容模式

    # --- 主流 Llama 系 (Llama 2/3, Mistral, Qwen, Yi) ---
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

    # --- Google 系 ---
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

    # --- 其他经典架构 ---
    "gpt2": ["c_attn", "c_proj"],
    "chatglm": ["query_key_value"],
    "bloom": ["query_key_value"],
    "baichuan": ["W_pack"],
    "phi": ["q_proj", "k_proj", "v_proj", "fc1", "fc2"],
    "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "bert": ["query", "key", "value"],
    "roberta": ["query", "key", "value"],
}

DEFAULT_PRECISION = "fp32"









# Chunk字段每段长度选择
class ChunkConfig:
    RECOMMENDED = [256, 512, 1024]
    HARD_LIMIT = 2048  # PoC 阶段建议上限，取决于显卡显存
    DEFAULT = 512

# 签名文件名
SIG_FILENAME = "adcust_signature.json"







# 调用本地大模型进行测试
class InferenceTaskMode(str, Enum):
    SINGLE = "single"   # 单窗口模式
    COMPARE = "compare" # 对比模式

class InferenceDispatchStrategy(str, Enum):
    SERIAL = "serial"           # 串行模式：先跑完 A，再跑 B
    INTERLEAVED = "interleaved" # 交替模式：每路出 N 个 token 切换一次