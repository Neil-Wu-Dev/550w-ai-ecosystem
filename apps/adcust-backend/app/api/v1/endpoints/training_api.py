import logging
import json
from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from adcust_logic.interfaces.services.remote.i_training_service import ITrainingService
from app.config.dependencies import get_training_service
from adcust_logic.defs import TrainingMode
from adcust_logic.schemas.remote_training_schema import RemoteTrainingStartRequest, RemoteTrainingStatusResponse

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
    """兼容旧入口。新远程训练应使用 /remote/start。"""

    def generate_progress():
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
    """尝试中止训练进程；这不等于关闭或销毁云服务器。"""
    train_svc.abort_training(mode=mode)
    return {
        "status": "success",
        "message": "Abort signal sent. AdCust cannot stop or terminate the cloud server."
    }


@router.post("/remote/start")
async def start_remote_train(req: RemoteTrainingStartRequest, train_svc: ITrainingService = Depends(get_training_service)):
    """通过 SSH 连接用户已经手动开启的远程机器，并启动远程训练。"""

    def generate_progress():
        request_data = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        try:
            for progress in train_svc.start_remote_adapter_job(request_data):
                yield json.dumps(progress, ensure_ascii=False) + "\n"
        except Exception as exc:
            logger.exception("Remote training stream failed before a terminal event was emitted")
            yield json.dumps(
                {
                    "status": "failed",
                    "stage": "preflight_failed",
                    "percentage": 0,
                    "message": str(exc),
                    "error": str(exc),
                },
                ensure_ascii=False,
            ) + "\n"

    return StreamingResponse(
        generate_progress(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no"}
    )


@router.get("/remote/{job_id}", response_model=RemoteTrainingStatusResponse)
def get_remote_train_status(job_id: str, train_svc: ITrainingService = Depends(get_training_service)):
    """获取远程训练任务快照。"""
    job = train_svc.get_job_status(job_id)
    return job.status_snapshot()


@router.post("/remote/{job_id}/abort", response_model=RemoteTrainingStatusResponse)
def abort_remote_train(job_id: str, train_svc: ITrainingService = Depends(get_training_service)):
    """尝试终止远程训练进程；这不等于关闭或销毁云服务器。"""
    job = train_svc.abort_remote_job(job_id)
    return job.status_snapshot()
