import logging
from typing import List, Optional
from fastapi import FastAPI

# --- 1. 导入接口 ---
from app.services.interfaces.i_model_manager import IModelManagerService
from app.services.interfaces.i_dataset_service import IDatasetService
from app.services.interfaces.i_training_service import ITrainingService
from app.services.interfaces.i_inference_service import IInferenceService

# --- 2. 导入具体的实现类 ---
from app.services.model_manager_service import ModelManagerService
from app.services.dataset_service import DatasetService
from app.services.training_service import TrainingService
from app.services.inference_service import InferenceService

logger = logging.getLogger(__name__)

# --- 4. 实例化所有单例 Service ---
# 第一步：先创建不依赖别人的原子服务
_model_manager_instance = ModelManagerService()
_dataset_service_instance = DatasetService()

# 第二步：创建需要注入依赖的复合服务（关键：手动将单例塞入构造函数）
_training_service_instance = TrainingService(
    model_manager=_model_manager_instance,
    dataset_service=_dataset_service_instance
)
_inference_service_instance = InferenceService(
    model_manager=_model_manager_instance
)


# --- 6. 供外部 (API) 调用的 Getter 函数 ---
# 确保这里的函数名与 API 层 Depends() 里的调用完全一致

def get_model_service() -> IModelManagerService:  # 修改了函数名，对齐 API
    return _model_manager_instance


def get_dataset_service() -> IDatasetService:
    return _dataset_service_instance


def get_training_service() -> ITrainingService:
    return _training_service_instance


def get_inference_service() -> IInferenceService:
    return _inference_service_instance


# --- 7. 点火函数 (供 main.py 调用) ---
def setup_dependencies(app: FastAPI) -> None:
    """
    点火函数：确保所有 Service 单例已正确组装。
    """
    logger.info("--- [Dependencies] 正在启动全局依赖检查 (解耦模式)... ---")

    # 检查所有单例是否实例化成功
    services = [
        _model_manager_instance,
        _dataset_service_instance,
        _training_service_instance,
        _inference_service_instance
    ]

    if all(services):
        logger.info("--- [Dependencies] 所有服务单例组装完成 ---")
    else:
        logger.error("--- [Dependencies] 服务单例初始化存在缺失！ ---")