"""
本文件存放所有支持的 AI 模型架构元数据。
包含：模型层级名称映射表（用于 LoRA/Fine-tuning 识别可训练层）及默认计算精度。
该文件会随着支持模型的增加而频繁更新，但不影响业务逻辑。
"""

# 模型可训练层级字典：映射模型系列到其对应的注意力层/线性层名称
MODEL_LAYER_MAP = {
    "deepseek_v2": ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "deepseek": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gpt2": ["c_attn", "c_proj"],
    "chatglm": ["query_key_value"],
    "bloom": ["query_key_value"],
    "baichuan": ["W_pack"],
    "phi": ["q_proj", "k_proj", "v_proj", "fc1", "fc2"],
    "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "bert": ["query", "key", "value"],
    "roberta": ["query", "key", "value"],
}

# 默认推理/训练精度
DEFAULT_PRECISION = "fp32"