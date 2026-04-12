import logging
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Body, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.services.interfaces.i_orchestrator import IOrchestrator
from app.core.dependencies import get_orchestrator
from app.schemas.model_schema import ModelSelectRequest, ModelResponse
from app.schemas.adapter_schema import AdapterDTO
from app.core.constants import InferenceDispatchStrategy

router = APIRouter()
logger = logging.getLogger(__name__)


# --- 为了对齐前端数据格式定义的内部 Pydantic 模型 ---

class TrainRequest(BaseModel):
    target_dir: str = Field(..., description="适配器保存的根目录")
    layers: List[str] = Field(..., description="可训练层列表")
    mode: str = Field(..., description="训练模式，如 sequential")
    name: Optional[str] = Field(None, description="自定义适配器名称")
    # 【核心修改】新增 epochs 字段，默认值为 1
    epochs: int = Field(1, ge=1, description="训练迭代轮数，必须大于等于 1")


class StopRequest(BaseModel):
    mode: str = "sequential"


class MountRequest(BaseModel):
    adapter_path: str
    slot_id: str


class ChatRequest(BaseModel):
    prompt: str
    targets: List[Dict[str, Any]]
    strategy: InferenceDispatchStrategy = InferenceDispatchStrategy.INTERLEAVED
    max_tokens: int = 512


# --- 路由实现 ---

@router.get("/init-options")
def get_options(orch: IOrchestrator = Depends(get_orchestrator)):
    """获取系统静态配置选项"""
    return orch.get_static_options()


@router.post("/select-model", response_model=ModelResponse)
def select_model(req: ModelSelectRequest, orch: IOrchestrator = Depends(get_orchestrator)):
    """选择并加载底座模型"""
    try:
        return orch.set_base_model(req.local_path)
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/process-dataset")
def process_dataset(
        file_path: str = Query(..., description="数据集文件路径"),
        chunk_size: int = Query(..., description="切片大小"),
        strategy: str = Query(..., description="训练策略"),
        orch: IOrchestrator = Depends(get_orchestrator)
):
    """处理并准备训练数据集"""
    try:
        return orch.prepare_training_data(file_path, chunk_size, strategy)
    except Exception as e:
        logger.error(f"数据集处理失败: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/start-train")
async def start_train(
        req: TrainRequest,
        orch: IOrchestrator = Depends(get_orchestrator)
):
    """
    启动定制化训练流，支持 SSE/NDJSON 实时进度转发
    """

    def generate_progress():
        try:
            logger.info(f"--- [Router] 收到训练请求: 模式={req.mode}, 轮数={req.epochs} ---")

            # 【关键修改】调用 execute_customization 时传入 req.epochs
            progress_stream = orch.execute_customization(
                target_dir=req.target_dir,
                trainable_layers=req.layers,
                mode=req.mode,
                custom_name=req.name,
                epochs=req.epochs  # <--- 正确透传参数
            )

            for progress in progress_stream:
                # 确保 progress 字典包含由底层传来的 epoch/step 等字段
                yield json.dumps(progress, ensure_ascii=False) + "\n"

        except Exception as e:
            logger.error(f"训练流异常: {str(e)}")
            yield json.dumps({"status": "error", "message": f"Router Error: {str(e)}"}, ensure_ascii=False) + "\n"

    return StreamingResponse(
        generate_progress(),
        media_type="application/x-ndjson",
        headers={
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓存，确保流式输出
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.post("/stop-train")
def stop_train(
        req: StopRequest,
        orch: IOrchestrator = Depends(get_orchestrator)
):
    """强制停止训练任务"""
    try:
        result = orch.stop_customization(mode=req.mode)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"终止任务失败: {str(e)}")


@router.post("/engine/boot")
def boot_engine(orch: IOrchestrator = Depends(get_orchestrator)):
    """启动推理引擎"""
    try:
        message = orch.boot_inference_engine()
        return {"status": "running", "message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/engine/shutdown")
def shutdown_engine(orch: IOrchestrator = Depends(get_orchestrator)):
    """关闭推理引擎"""
    try:
        orch.shutdown_inference_engine()
        return {"status": "stopped", "message": "Engine shutdown successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/adapter/mount")
def mount_adapter(
        req: MountRequest,
        orch: IOrchestrator = Depends(get_orchestrator)
):
    """挂载适配器到指定槽位"""
    try:
        asset_entity = orch.load_adapter_asset(req.adapter_path, req.slot_id)
        return {
            "status": "success",
            "slot": req.slot_id,
            "adapter": AdapterDTO.from_entity(asset_entity)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(
        req: ChatRequest,
        orch: IOrchestrator = Depends(get_orchestrator)
):
    """多路径流式聊天推理"""
    try:
        generator = orch.stream_chat(req.targets, req.prompt, req.strategy, req.max_tokens)
        return StreamingResponse(
            generator,
            media_type="application/x-ndjson",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
    except RuntimeError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))