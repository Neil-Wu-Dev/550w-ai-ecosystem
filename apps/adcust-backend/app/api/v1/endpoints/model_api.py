import logging
from fastapi import APIRouter, Depends
from adcust_logic.interfaces.services.client.i_model_manager import IModelManagerService
from app.config.dependencies import get_model_service
from adcust_logic.schemas.model_schema import ModelSelectRequest, ModelResponse
from adcust_logic.mappers.model_mapper import ModelMapper

# 修改：清空 prefix，由 main 统一分发
router = APIRouter(prefix="", tags=["Model"])
logger = logging.getLogger(__name__)


@router.post("/select", response_model=ModelResponse)
def select_model(
        req: ModelSelectRequest,
        model_svc: IModelManagerService = Depends(get_model_service)
):
    # --- 核心改动：移除手动捕获逻辑 ---
    # 1. 如果 Mapper 转换失败，它会抛出异常。
    # 2. 如果 Service 找不到路径，它会抛出异常。
    # 3. 这里的代码不再拦路，让定制异常通过冒泡机制由全局拦截器统一处理。

    model_entity = ModelMapper.to_entity(req)
    result_entity = model_svc.select_model_by_path(model_entity.local_path)

    return ModelMapper.to_response_dict(result_entity)