import logging
import json
from typing import List, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from adcust_logic.interfaces.services.i_inference_service import IInferenceService
from app.config.dependencies import get_inference_service
from adcust_logic.defs import InferenceDispatchStrategy
from adcust_logic.mappers.adapter_mapper import AdapterMapper

# 修改：清空 prefix
router = APIRouter(prefix="", tags=["Inference"])
logger = logging.getLogger(__name__)

class MountRequest(BaseModel):
    adapter_path: str
    slot_id: str

class ChatRequest(BaseModel):
    prompt: str
    targets: List[Dict[str, Any]]
    strategy: InferenceDispatchStrategy = InferenceDispatchStrategy.INTERLEAVED
    max_tokens: int = 512

@router.post("/engine/boot")
def boot_engine(model_path: str, inf_svc: IInferenceService = Depends(get_inference_service)):
    # 移除 try-except：让底层 BusinessException 或 ValidationException 自动上抛
    success = inf_svc.run_engine(model_path)
    return {"status": "running" if success else "failed"}

@router.post("/adapter/mount")
def mount_adapter(req: MountRequest, inf_svc: IInferenceService = Depends(get_inference_service)):
    # 移除 try-except：如果路径非法或挂载失败，底层抛出的异常会触发全局翻译
    asset_entity = inf_svc.mount_adapter(req.adapter_path, req.slot_id)
    return {
        "status": "success",
        "slot": req.slot_id,
        "adapter": AdapterMapper.to_dto(asset_entity)
    }

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, inf_svc: IInferenceService = Depends(get_inference_service)):
    async def multi_generator():
        # 移除内部 try-except：
        # 大厂规范中，流式传输若发生异常应由框架层捕获，确保错误格式的一致性
        for t in req.targets:
            slot_id = t["id"]
            for token in inf_svc.generate_single_path(req.prompt, slot_id, req.max_tokens):
                yield json.dumps({"source": slot_id, "token": token}) + "\n"

    return StreamingResponse(multi_generator(), media_type="application/x-ndjson")