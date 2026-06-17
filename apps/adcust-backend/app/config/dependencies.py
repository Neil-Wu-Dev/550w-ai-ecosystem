import logging
import os
from pathlib import Path
from fastapi import FastAPI

# --- 1. 导入接口 (Interfaces) ---
# 包含新增的基础设施接口
from adcust_logic.interfaces.services.infra.i_compute_provider_service import IComputeProviderService
from adcust_logic.interfaces.services.client.i_model_manager import IModelManagerService
from adcust_logic.interfaces.services.client.i_dataset_service import IDatasetService
from adcust_logic.interfaces.services.remote.i_training_service import ITrainingService
from adcust_logic.interfaces.services.client.i_inference_service import IInferenceService

# --- 2. 导入具体的实现类 (Implementations) ---
from adcust_logic.services.infra.compute_provider_service import ComputeProviderService
from adcust_logic.services.client.model_manager_service import ModelManagerService
from adcust_logic.services.client.dataset_service import DatasetService
from adcust_logic.services.client.training_service import TrainingService
from adcust_logic.services.client.inference_service import InferenceService

logger = logging.getLogger(__name__)

# --- 4. 实例化所有单例 Service ---

# 第一步：先创建不依赖他人的原子服务 (Atomic Services)
# 为基础设施服务指定一个存储路径，用于持久化供应商 Key/配置
# 持久化目录必须与当前工作目录无关，否则不同启动入口会读取不同的数据文件。
_default_data_dir = Path(__file__).resolve().parents[1] / "data"
_data_dir = Path(os.getenv("ADCUST_DATA_DIR", str(_default_data_dir))).expanduser().resolve()
_provider_storage_path = str(_data_dir / "providers.json")
_infra_service_instance = ComputeProviderService(storage_path=_provider_storage_path)

_model_manager_instance = ModelManagerService()
_dataset_service_instance = DatasetService()

# 第二步：创建需要注入依赖的复合服务 (Composite Services)
# 关键变更：TrainingService 必须注入 infra_service，否则无法执行远端部署
_training_service_instance = TrainingService(
    model_manager=_model_manager_instance,
    dataset_service=_dataset_service_instance,
    infra_service=_infra_service_instance
)

_inference_service_instance = InferenceService(
    model_manager=_model_manager_instance
)


# --- 6. 供外部 (API) 调用的 Getter 函数 ---
# 供 FastAPI 路由层通过 Depends() 注入使用

def get_infra_service() -> IComputeProviderService:
    """获取基础设施 (云端管理) 服务单例"""
    return _infra_service_instance


def get_model_service() -> IModelManagerService:
    """获取模型管理服务单例"""
    return _model_manager_instance


def get_dataset_service() -> IDatasetService:
    """获取数据集服务单例"""
    return _dataset_service_instance


def get_training_service() -> ITrainingService:
    """获取训练调度服务单例"""
    return _training_service_instance


def get_inference_service() -> IInferenceService:
    """获取本地推理服务单例"""
    return _inference_service_instance


# --- 7. 点火函数 (供 main.py 调用) ---
def setup_dependencies(app: FastAPI) -> None:
    """
    点火函数：确保所有 Service 单例已正确组装，并建立依赖链路。
    """
    logger.info("--- [Dependencies] 正在启动全局依赖检查 (跨云调度模式)... ---")
    logger.info("[Dependencies] Persistent data directory: %s", _data_dir)

    # 检查所有单例是否实例化成功
    services = [
        _infra_service_instance,
        _model_manager_instance,
        _dataset_service_instance,
        _training_service_instance,
        _inference_service_instance
    ]

    if all(services):
        logger.info("--- [Dependencies] 所有服务单例组装完成，基础设施已就绪 ---")
    else:
        # 统计缺失的服务项
        missing = [s for s in services if s is None]
        logger.error(f"--- [Dependencies] 服务单例初始化失败！缺失数量: {len(missing)} ---")
