# -*- coding: utf-8 -*-
from typing import Any, Dict, List
from adcust_logic.schemas.compute_provider_schema import ComputeProviderCreateRequest
from adcust_logic.models.infra.compute_provider_asset import ComputeProviderAsset

class ComputeProviderMapper:
    """
    量化职责：负责 ComputeProviderAsset 实体与 DTO 的物理字段双向转换。
    """

    @staticmethod
    def to_response_dict(entity: ComputeProviderAsset) -> Dict[str, Any]:
        """
        正向转换：ComputeProviderAsset (Entity) -> Dict (用于 API 响应)
        主要职责：提取摘要信息，并确保时间戳等格式正确。
        """
        if not entity:
            return {}

        return {
            "id": entity.id,
            "name": entity.name,
            "provider_type": entity.provider_type,
            "endpoint_summary": entity.endpoint_summary,
            "is_active": entity.is_active,
            "last_heartbeat": entity.last_heartbeat,
            "telemetry_data": entity.telemetry_data,
            "created_at": entity.created_at
        }

    @staticmethod
    def to_entity(request_dto: ComputeProviderCreateRequest) -> ComputeProviderAsset:
        """
        反向转换：ComputeProviderCreateRequest (DTO) -> ComputeProviderAsset (Entity)
        主要职责：将前端提交的扁平化 JSON 重新构建为逻辑层受保护的实体对象。
        """
        if not request_dto:
            raise ValueError("Provider request data is required")

        return ComputeProviderAsset(
            id=request_dto.id,
            name=request_dto.name,
            provider_type=request_dto.provider_type,
            connection_info=request_dto.connection_info,
            is_active=False, # 新创建的节点默认为非活跃，直到执行 verify_connection
            telemetry_data={}
        )

    @staticmethod
    def map_list(entities: List[ComputeProviderAsset]) -> List[Dict[str, Any]]:
        """批量转换：将实体列表转换为前端可展示的字典列表"""
        return [ComputeProviderMapper.to_response_dict(e) for e in entities]