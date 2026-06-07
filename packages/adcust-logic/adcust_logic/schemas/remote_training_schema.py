# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RemoteTrainingStartRequest(BaseModel):
    """远程训练启动 DTO。

    注意：AdCust 只尝试连接用户已经手动开启的远程 SSH 服务器。
    它不负责启动、关闭、销毁云厂商 pod，也不承诺停止云厂商计费。
    """

    provider_id: str = Field(..., description="已登记的远程 SSH 节点 ID")
    dataset_path: str = Field(..., description="本地数据集文件路径，支持 json/jsonl/txt/md/pdf；PDF 会先抽取纯文本并转成临时 JSON")
    local_base_model_path: str = Field(..., description="本地推理使用的 HuggingFace 底座模型目录")
    base_model_name_or_path: str = Field(..., description="HuggingFace 模型 ID 或已有远程模型目录；远程会自动下载并校验 manifest")
    adapter_name: str = Field(..., min_length=2, max_length=64, description="生成的 adapter 名称")
    local_output_root: str = Field(..., description="本地 adapter 保存根目录，必须由用户选择")
    epochs: int = Field(..., ge=1)
    learning_rate: float = Field(..., gt=0)
    max_seq_length: int = Field(..., ge=1)
    lora_rank: int = Field(..., ge=1)
    lora_alpha: int = Field(..., ge=1)
    lora_dropout: float = Field(..., ge=0, le=1)
    target_modules: List[str] = Field(..., min_length=1)
    use_qlora: bool = Field(...)
    cleanup_remote: bool = Field(...)


class RemoteTrainingStatusResponse(BaseModel):
    job_id: str
    status: str
    message: str
    elapsed_seconds: float
    estimated_cost_usd: float
    local_output_dir: str
    training_params: Dict[str, Any]
    requires_manual_shutdown: bool = True
    percentage: float = 0
    stage: str = "created"
    resources: List[Dict[str, Any]] = Field(default_factory=list)
    loss: Optional[float] = None
    train_progress: Optional[float] = None
    eta_seconds: Optional[float] = None
    step: Optional[int] = None
    total_steps: Optional[int] = None
