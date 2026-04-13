import logging  # 导入日志模块
from typing import List, Optional
from fastapi import FastAPI  # 导入 FastAPI 类型声明

# --- 1. 导入接口 ---
from app.services.interfaces.i_model_manager import IModelManagerService
from app.services.interfaces.i_dataset_service import IDatasetService
from app.services.interfaces.i_training_service import ITrainingService
from app.services.interfaces.i_orchestrator import IOrchestrator
from app.services.interfaces.i_inference_service import IInferenceService

# --- 2. 导入具体的实现类 ---
from app.services.model_manager_service import ModelManagerService
from app.services.dataset_service import DatasetService
from app.services.training_service import TrainingService
from app.services.orchestrator import Orchestrator
from app.services.inference_service import InferenceService

# --- 3. 初始化全局 Logger ---
# 修复 Unresolved reference 'logger' 报错
logger = logging.getLogger(__name__)

# --- 4. 实例化所有单例 Service (零件) ---
# 这些是底层的原子服务，在模块加载时即完成初始化
_model_manager_instance = ModelManagerService()
_dataset_service_instance = DatasetService()
_training_service_instance = TrainingService()
_inference_service_instance = InferenceService()

# --- 5. 实例化 Orchestrator (组装大脑) ---
# 通过构造函数注入上面的单例，确保全局唯一性
_orchestrator_instance = Orchestrator(
    model_manager=_model_manager_instance,
    dataset_service=_dataset_service_instance,
    training_service=_training_service_instance,
    inference_service=_inference_service_instance
)

# --- 6. 供外部调用的 Getter 函数 ---

def get_model_manager() -> IModelManagerService:
    return _model_manager_instance

def get_dataset_service() -> IDatasetService:
    return _dataset_service_instance

def get_training_service() -> ITrainingService:
    return _training_service_instance

def get_inference_service() -> IInferenceService:
    return _inference_service_instance

def get_orchestrator() -> IOrchestrator:
    """
    供 API Router 调用的最终挂载点。
    它是整个 AdCust 业务逻辑的唯一入口。
    """
    return _orchestrator_instance

# --- 7. 点火函数 (供 main.py 调用) ---
# 修复 Unresolved reference 'FastAPI' 报错
def setup_dependencies(app: FastAPI) -> None:
    """
    点火函数：在 main.py 中调用。
    作用：确保所有 Service 单例在应用启动时已正确加载并组装。
    """
    logger.info("--- [Dependencies] 正在启动全局依赖检查... ---")

    # 验证单例状态
    if _orchestrator_instance:
        logger.info("--- [Dependencies] Orchestrator 组装完成，单例就绪 ---")
    else:
        logger.error("--- [Dependencies] Orchestrator 组装失败！ ---")