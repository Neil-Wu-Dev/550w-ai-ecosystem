# -*- coding: utf-8 -*-
import logging
from typing import List
from fastapi import APIRouter, Depends, Query
from app.config.dependencies import get_infra_service
from adcust_logic.interfaces.services.infra.i_compute_provider_service import IComputeProviderService
from adcust_logic.schemas.compute_provider_schema import ComputeProviderCreateRequest, ComputeProviderResponse
from adcust_logic.mappers.compute_provider_mapper import ComputeProviderMapper

# 保持一致：清空 prefix，由 main 统一分发
router = APIRouter(prefix="", tags=["Compute"])
logger = logging.getLogger(__name__)

@router.get("/", response_model=List[ComputeProviderResponse])
def list_providers(infra_svc: IComputeProviderService = Depends(get_infra_service)):
    """获取所有已登记的算力资源列表"""
    entities = infra_svc.list_all_providers()
    return ComputeProviderMapper.map_list(entities)

@router.post("/", response_model=ComputeProviderResponse)
def add_provider(
    req: ComputeProviderCreateRequest,
    infra_svc: IComputeProviderService = Depends(get_infra_service)
):
    """登记新的算力供应商 (SSH/Kaggle等)"""
    entity = ComputeProviderMapper.to_entity(req)
    infra_svc.save_provider(entity)
    return ComputeProviderMapper.to_response_dict(entity)

@router.post("/{provider_id}/verify")
def verify_provider(
    provider_id: str,
    infra_svc: IComputeProviderService = Depends(get_infra_service)
):
    """执行静默握手测试，验证连接是否可用"""
    provider = infra_svc.get_provider(provider_id)
    # 如果找不到，infra_svc 内部抛出的异常会被拦截器翻译为前端可见的错误
    is_ok = infra_svc.verify_connection(provider)
    return {"status": "connected" if is_ok else "failed"}

@router.post("/{provider_id}/sync", response_model=ComputeProviderResponse)
def sync_status(
    provider_id: str,
    infra_svc: IComputeProviderService = Depends(get_infra_service)
):
    """远程嗅探服务器资源（显存、温度、磁盘），更新快照"""
    updated_entity = infra_svc.sync_resource_status(provider_id)
    return ComputeProviderMapper.to_response_dict(updated_entity)

@router.delete("/{provider_id}")
def remove_provider(
    provider_id: str,
    cleanup: bool = Query(False),
    infra_svc: IComputeProviderService = Depends(get_infra_service)
):
    """移除供应商登记，可选是否清理远端残留载荷"""
    success = infra_svc.remove_provider(provider_id, cleanup_remote=cleanup)
    return {"status": "success" if success else "failed"}