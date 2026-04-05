from pydantic import BaseModel
from typing import List

class ModelSelectRequest(BaseModel):
    """手动选定模型时提交的请求"""
    local_path: str  # 用户在界面上输入或选定的路径

class ModelResponse(BaseModel):
    """返回给前端的模型详细信息"""
    name: str
    local_path: str
    architecture: str
    precision: str
    trainable_layers: List[str]