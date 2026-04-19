import logging
import json
from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from adcust_logic.interfaces.services.i_training_service import ITrainingService
from app.config.dependencies import get_training_service
from adcust_logic.defs import TrainingMode

# 修改：清空 prefix
router = APIRouter(prefix="", tags=["Training"])
logger = logging.getLogger(__name__)

class TrainRequest(BaseModel):
    target_dir: str
    layers: List[str]
    mode: TrainingMode = TrainingMode.SEQUENTIAL
    name: Optional[str] = None
    epochs: int = 1

@router.post("/start")
async def start_train(req: TrainRequest, train_svc: ITrainingService = Depends(get_training_service)):
    def generate_progress():
        # --- 核心改动：移除 generate_progress 内部的 try-except ---
        # 逻辑：
        # 1. 如果 customize_new_adapter 在参数校验阶段失败（如 layers 为空），
        #    ValidationException 会直接在 StreamingResponse 启动前抛出。
        # 2. 如果训练中途发生 BusinessException，流会自然中断，
        #    这种“直接崩溃”比发送一个伪造的 {"status": "error"} 更符合大厂的系统健康度监控。
        progress_stream = train_svc.customize_new_adapter(
            target_dir=req.target_dir,
            trainable_layers=req.layers,
            mode=req.mode,
            custom_name=req.name,
            epochs=req.epochs
        )
        for progress in progress_stream:
            yield json.dumps(progress, ensure_ascii=False) + "\n"

    return StreamingResponse(
        generate_progress(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no"}
    )

@router.post("/stop")
def stop_train(mode: TrainingMode = TrainingMode.SEQUENTIAL, train_svc: ITrainingService = Depends(get_training_service)):
    # --- 核心改动：移除 stop_train 的 try-except ---
    # 如果终止指令执行失败，底层抛出的异常会通过冒泡机制被 main.py 拦截器捕获。
    train_svc.abort_training(mode=mode)
    return {"status": "success", "message": "终止指令已发送"}