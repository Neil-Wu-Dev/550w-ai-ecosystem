import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from domain_logic.interfaces.services.i_dataset_service import IDatasetService
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
    try:
        asset = dataset_svc.prepare_dataset_asset(file_path, chunk_size, strategy)
        return {
            "source_name": asset.source_name,
            "chunk_count": len(asset.chunks),
            "status": "ready"
        }
    except Exception as e:
        logger.error(f"数据集处理失败: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))