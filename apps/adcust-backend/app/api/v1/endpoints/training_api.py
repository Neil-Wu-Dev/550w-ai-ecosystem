import logging
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.services.interfaces.i_training_service import ITrainingService
from app.core.dependencies import get_training_service
from app.core.constants import TrainingMode

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
        try:
            progress_stream = train_svc.customize_new_adapter(
                target_dir=req.target_dir,
                trainable_layers=req.layers,
                mode=req.mode,
                custom_name=req.name,
                epochs=req.epochs
            )
            for progress in progress_stream:
                yield json.dumps(progress, ensure_ascii=False) + "\n"
        except Exception as e:
            logger.error(f"训练流异常: {str(e)}")
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"

    return StreamingResponse(
        generate_progress(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no"}
    )

@router.post("/stop")
def stop_train(mode: TrainingMode = TrainingMode.SEQUENTIAL, train_svc: ITrainingService = Depends(get_training_service)):
    try:
        train_svc.abort_training(mode=mode)
        return {"status": "success", "message": "终止指令已发送"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))