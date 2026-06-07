# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime

class ComputeProviderCreateRequest(BaseModel):
    """前端添加新算力节点时提交的请求"""
    id: str = Field(..., description="唯一的供应商ID")
    name: str = Field(..., min_length=2, max_length=32, description="显示名称")
    provider_type: str = Field(..., description="类型: SSH, KAGGLE, 等")
    connection_info: Dict[str, Any] = Field(..., description="核心连接字典(host, port, user等)")

class ComputeProviderResponse(BaseModel):
    """返回给前端的供应商详细信息(脱敏后)"""
    id: str
    name: str
    provider_type: str
    endpoint_summary: str
    connection_info: Dict[str, Any]
    remote_workspace_path: str
    hourly_rate_usd: float
    is_active: bool
    last_heartbeat: Optional[datetime] = None
    telemetry_data: Dict[str, Any]
    created_at: datetime

class ProviderStatusDTO:
    """静态数据传输对象：用于快速转换列表显示"""
    @staticmethod
    def from_entity(entity: Any) -> Dict[str, Any]:
        """将领域实体转换为前端可读的字典 (模仿 AdapterDTO 风格)"""
        if not entity: return {}
        return {
            "id": entity.id,
            "name": entity.name,
            "type": entity.provider_type,
            "endpoint": entity.endpoint_summary,
            "is_active": entity.is_active,
            "telemetry": entity.telemetry_data
        }
