import logging
import json
from typing import List, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.services.interfaces.i_inference_service import IInferenceService
from app.core.dependencies import get_inference_service
from app.core.constants import InferenceDispatchStrategy
from app.mappers.adapter_mapper import AdapterMapper

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
    try:
        success = inf_svc.run_engine(model_path)
        return {"status": "running" if success else "failed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/adapter/mount")
def mount_adapter(req: MountRequest, inf_svc: IInferenceService = Depends(get_inference_service)):
    try:
        asset_entity = inf_svc.mount_adapter(req.adapter_path, req.slot_id)
        return {
            "status": "success",
            "slot": req.slot_id,
            "adapter": AdapterMapper.to_dto(asset_entity)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, inf_svc: IInferenceService = Depends(get_inference_service)):
    async def multi_generator():
        try:
            for t in req.targets:
                slot_id = t["id"]
                for token in inf_svc.generate_single_path(req.prompt, slot_id, req.max_tokens):
                    yield json.dumps({"source": slot_id, "token": token}) + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(multi_generator(), media_type="application/x-ndjson")