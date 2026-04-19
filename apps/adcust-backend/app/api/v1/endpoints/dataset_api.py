import logging
from fastapi import APIRouter, Depends, Query
from adcust_logic.interfaces.services.i_dataset_service import IDatasetService
from app.config.dependencies import get_dataset_service

# 修改：清空 prefix
router = APIRouter(prefix="", tags=["Dataset"])
logger = logging.getLogger(__name__)


@router.post("/process")
def process_dataset(
        file_path: str = Query(..., description="数据集文件路径"),
        chunk_size: int = Query(..., description="切片大小"),
        strategy: str = Query(..., description="训练策略"),
        dataset_svc: IDatasetService = Depends(get_dataset_service)
):
    # --- 核心改动：移除 try-except ---
    # 逻辑：
    # 1. dataset_svc 会创建 DatasetAsset。
    # 2. DatasetAsset 内部的 _validate 会检查数据。
    # 3. 如果非法，抛出 ValidationException。
    # 4. 异常自动冒泡，越过此 Endpoint，被 main.py 的全局拦截器捕捉并翻译。

    asset = dataset_svc.prepare_dataset_asset(file_path, chunk_size, strategy)

    return {
        "source_name": asset.source_name,
        "chunk_count": len(asset.chunks),
        "status": "ready"
    }