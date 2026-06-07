import logging
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from adcust_logic.interfaces.services.client.i_inference_service import IInferenceService
from app.config.dependencies import get_inference_service
from adcust_logic.defs import InferenceDispatchStrategy
from adcust_logic.mappers.adapter_mapper import AdapterMapper

# 修改：清空 prefix
router = APIRouter(prefix="", tags=["Inference"])
logger = logging.getLogger(__name__)

class MountRequest(BaseModel):
    adapter_path: str
    slot_id: str
    base_model_path: Optional[str] = None

class SlotRequest(BaseModel):
    slot_id: str

class StopGenerationRequest(BaseModel):
    slot_id: Optional[str] = None

class ChatMessageRequest(BaseModel):
    role: str
    content: str

class ChatTargetRequest(BaseModel):
    id: str
    history: List[ChatMessageRequest] = Field(default_factory=list)

class ChatRequest(BaseModel):
    prompt: str
    session_id: str = "default"
    system_prompt: Optional[str] = None
    targets: List[ChatTargetRequest]
    strategy: InferenceDispatchStrategy = InferenceDispatchStrategy.INTERLEAVED
    max_tokens: int = 512

class ClearHistoryRequest(BaseModel):
    session_id: str = "default"
    slot_id: Optional[str] = None

conversation_store: Dict[str, List[Dict[str, str]]] = {}

def _history_key(session_id: str, slot_id: str) -> str:
    return f"{session_id}:{slot_id.lower()}"

def _normalize_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    for item in history:
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant", "system"} and content:
            cleaned.append({"role": role, "content": content})
    return cleaned[-40:]

@router.post("/engine/boot")
def boot_engine(model_path: str, inf_svc: IInferenceService = Depends(get_inference_service)):
    # 移除 try-except：让底层 BusinessException 或 ValidationException 自动上抛
    success = inf_svc.run_engine(model_path)
    return {"status": "running" if success else "failed"}

@router.post("/engine/shutdown")
def shutdown_engine(inf_svc: IInferenceService = Depends(get_inference_service)):
    """关闭本地推理引擎，释放显存/内存并卸载所有 adapter。"""
    inf_svc.stop_engine()
    return {"status": "stopped"}

@router.post("/engine/stop-generation")
def stop_generation(req: StopGenerationRequest, inf_svc: IInferenceService = Depends(get_inference_service)):
    """中断当前生成请求，但不卸载模型。"""
    inf_svc.request_stop_generation()
    return {"status": "stopping", "slot": req.slot_id}

@router.get("/engine/status")
def engine_status(inf_svc: IInferenceService = Depends(get_inference_service)):
    """获取本地推理引擎状态。"""
    return inf_svc.get_engine_status()

@router.post("/adapter/mount")
def mount_adapter(req: MountRequest, inf_svc: IInferenceService = Depends(get_inference_service)):
    # 移除 try-except：如果路径非法或挂载失败，底层抛出的异常会触发全局翻译
    asset_entity = inf_svc.mount_adapter(req.adapter_path, req.slot_id, req.base_model_path)
    return {
        "status": "success",
        "slot": req.slot_id,
        "adapter": AdapterMapper.to_dto(asset_entity),
        "message": "Adapter mounted and loaded." if inf_svc.get_engine_status().get("status") == "running" else "Adapter validated and queued for the selected base model.",
    }

@router.post("/adapter/unmount")
def unmount_adapter(req: SlotRequest, inf_svc: IInferenceService = Depends(get_inference_service)):
    """卸载指定推理槽位的 adapter。"""
    inf_svc.unmount_adapter(req.slot_id)
    return {
        "status": "success",
        "slot": req.slot_id,
        "adapter": None
    }

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, inf_svc: IInferenceService = Depends(get_inference_service)):
    async def multi_generator():
        for t in req.targets:
            slot_id = t.id
            key = _history_key(req.session_id, slot_id)
            request_history = _normalize_history([
                item.model_dump() if hasattr(item, "model_dump") else item.dict()
                for item in t.history
            ])
            stored_history = conversation_store.get(key, [])
            effective_history = request_history if len(request_history) >= len(stored_history) else stored_history
            assistant_chunks: List[str] = []
            try:
                for token in inf_svc.generate_single_path(
                    req.prompt,
                    slot_id,
                    req.max_tokens,
                    system_prompt=req.system_prompt,
                    history=effective_history
                ):
                    assistant_chunks.append(token)
                    yield json.dumps({"source": slot_id, "token": token}) + "\n"
                assistant_text = "".join(assistant_chunks).strip()
                if assistant_text:
                    conversation_store[key] = _normalize_history([
                        *effective_history,
                        {"role": "user", "content": req.prompt},
                        {"role": "assistant", "content": assistant_text}
                    ])
                yield json.dumps({"source": slot_id, "is_final": True}) + "\n"
            except Exception as exc:
                yield json.dumps({"source": slot_id, "error": str(exc), "is_final": True}) + "\n"

    return StreamingResponse(multi_generator(), media_type="application/x-ndjson")

@router.post("/chat/history/clear")
def clear_chat_history(req: ClearHistoryRequest):
    """清理指定 session 的服务端会话历史。"""
    if req.slot_id:
        conversation_store.pop(_history_key(req.session_id, req.slot_id), None)
    else:
        prefix = f"{req.session_id}:"
        for key in list(conversation_store.keys()):
            if key.startswith(prefix):
                conversation_store.pop(key, None)
    return {"status": "cleared"}
