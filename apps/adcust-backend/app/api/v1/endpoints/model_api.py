import logging
from fastapi import APIRouter, Depends, HTTPException
from app.services.interfaces.i_model_manager import IModelManagerService
from app.core.dependencies import get_model_service
from app.schemas.model_schema import ModelSelectRequest, ModelResponse
from app.mappers.model_mapper import ModelMapper

# 修改：清空 prefix，由 main 统一分发
router = APIRouter(prefix="", tags=["Model"])
logger = logging.getLogger(__name__)

@router.post("/select", response_model=ModelResponse)
def select_model(
    req: ModelSelectRequest,
    model_svc: IModelManagerService = Depends(get_model_service)
):
    try:
        model_entity = ModelMapper.to_entity(req)
        result_entity = model_svc.select_model_by_path(model_entity.local_path)
        return ModelMapper.to_response_dict(result_entity)
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))