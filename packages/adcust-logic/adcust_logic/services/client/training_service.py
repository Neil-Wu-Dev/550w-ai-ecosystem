import os
import json
import logging
from datetime import datetime
from typing import List, Optional, Generator, Dict, Any

# 接口导入
from adcust_logic.interfaces.services.remote.i_training_service import ITrainingService
from adcust_logic.interfaces.services.client.i_model_manager import IModelManagerService
from adcust_logic.interfaces.services.client.i_dataset_service import IDatasetService
from adcust_logic.interfaces.services.infra.i_compute_provider_service import IComputeProviderService

# 内部依赖
from adcust_logic.services.engines.remote.training_engines.sequential_engine import SequentialEngine
from adcust_logic.defs import TrainingMode, SIG_FILENAME

logger = logging.getLogger(__name__)


class TrainingService(ITrainingService):
    def __init__(
            self,
            model_manager: IModelManagerService,
            dataset_service: IDatasetService,
            infra_service: IComputeProviderService  # 注入基础设施服务，用于调度云端
    ):
        self._model_manager = model_manager
        self._dataset_service = dataset_service
        self._infra_service = infra_service

        # 本地训练引擎初始化 (仅保留接口兼容性，逻辑层已封死调用)
        self._engines = {
            TrainingMode.SEQUENTIAL: SequentialEngine(),
        }

    def customize_new_adapter(
            self,
            target_dir: str,
            trainable_layers: List[str],
            mode: TrainingMode = TrainingMode.SEQUENTIAL,
            custom_name: Optional[str] = None,
            epochs: int = 1
    ) -> Generator[Dict[str, Any], None, None]:
        """
        核心调度入口：强制执行云端调度，彻底禁用本地环境。
        """
        # 1. 获取当前所有就绪资产
        base_model = self._model_manager.get_active_model()
        dataset = self._dataset_service.get_active_dataset()

        # 从基础设施服务获取当前选中的“算力节点”
        current_provider = self._infra_service.get_active_provider()

        if not base_model or not dataset:
            yield {"status": "error", "message": "底座模型或数据集未加载", "percentage": 0}
            return

        try:
            # 2. 强制云端逻辑：只要不是合法的远端 Provider，直接报错，不再判断 LOCAL
            if current_provider and current_provider.provider_type != "LOCAL":
                logger.info(f"--- [Training] 启动远端训练模式: {current_provider.name} ---")
                yield from self._run_remote_training(
                    current_provider, base_model, dataset, trainable_layers, epochs
                )
            else:
                # 3. 彻底封禁本地训练
                logger.error("--- [Training] 拒绝执行：当前环境未配置有效的云端算力节点 ---")
                yield {
                    "status": "error",
                    "message": "本地训练功能已禁用。请在设置中选择并连接一个远程算力节点（如 SSH 或 Kaggle）后再启动。抽烟去吧你！",
                    "percentage": 0
                }
                return

        except Exception as e:
            logger.error(f"远端调度执行失败: {str(e)}")
            yield {"status": "error", "message": f"执行异常: {str(e)}", "percentage": 0}

        finally:
            # 无论成功失败，销毁本地语料缓存，保护内存
            self._dataset_service.release_dataset()
            logger.info("--- [TrainingService] 训练任务生命周期结束，资源已回收 ---")

    def _run_remote_training(self, provider, model, dataset, layers, epochs):
        """
        跨云调度核心逻辑 (实现六步走方案)
        """
        yield {"status": "init", "message": f"连接远端节点: {provider.endpoint_summary}", "percentage": 5}

        # 获取物理驱动 (如 SSHDriver 或 KaggleDriver)
        driver = self._infra_service.get_driver_for(provider)
        driver.connect(provider.connection_info)

        # --- [第三步: 部署远端载荷] ---
        if "/tmp/adcust/engine.tar.gz" not in provider.remote_manifest:
            yield {"status": "deploy", "message": "正在部署远端计算载荷...", "percentage": 10}
            # 推送本地 adcust_logic/remote 目录下的执行包
            payload_path = os.path.join(os.getcwd(), "adcust_logic", "remote", "engine.tar.gz")

            if not os.path.exists(payload_path):
                raise FileNotFoundError(f"致命错误：在 {payload_path} 找不到载荷包！请确保你已经打好包放进去了！")

            driver.push_file(payload_path, "/tmp/adcust/engine.tar.gz")
            driver.exec_command("tar -xzf /tmp/adcust/engine.tar.gz -C /tmp/adcust/")
            provider.mark_remote_file("/tmp/adcust/engine.tar.gz", "LATEST")

        # --- [第四步: 模型身份确认与同步] ---
        if model.name not in provider.remote_manifest:
            yield {"status": "model_sync", "message": f"远端同步底座模型: {model.name}", "percentage": 20}
            provider.mark_remote_file(f"models/{model.name}", "STABLE")

        # --- [第五步: 数据上传与生命周期管理] ---
        yield {"status": "upload", "message": "正在上传训练语料数据...", "percentage": 40}
        temp_data_name = f"data_{datetime.now().strftime('%H%M%S')}.json"
        temp_data_path = os.path.join(os.getcwd(), "data", temp_data_name)

        with open(temp_data_path, 'w', encoding='utf-8') as f:
            json.dump(dataset.chunks, f)

        driver.push_file(temp_data_path, "/tmp/adcust/train_data.json")
        os.remove(temp_data_path)  # 上传完立即清理本地临时数据

        # --- [第六步: 执行训练与结果回收] ---
        yield {"status": "training", "message": "远端引擎已启动，正在监听日志...", "percentage": 50}

        remote_cmd = f"python3 /tmp/adcust/train_entry.py --model {model.name} --epochs {epochs}"

        # 驱动层返回生成器，实时捕获远端 stdout 日志并透传给 UI
        for log_line in driver.exec_command(remote_cmd):
            yield {"status": "training", "message": log_line.strip(), "percentage": 70}

        # 训练结束，将生成的适配器文件拉回本地
        yield {"status": "collect", "message": "正在拉回训练产出 (Adapter)...", "percentage": 90}
        remote_adapter_path = "/tmp/adcust/output/adapter_model.bin"
        local_adapter_path = os.path.join(os.getcwd(), "storage", "adapters", f"remote_{model.name}.bin")
        driver.pull_file(remote_adapter_path, local_adapter_path)

        yield {"status": "completed", "message": "云端训练成功，适配器已入库", "percentage": 100}

    def abort_training(self, mode: TrainingMode = TrainingMode.SEQUENTIAL):
        """中止指令下发"""
        logger.warning("接收到停止信号，正在尝试切断远端连接或进程...")
        engine = self._engines.get(mode)
        if engine:
            engine.abort()

    def _validate_support(self, strategy: str, mode: TrainingMode):
        if mode != TrainingMode.SEQUENTIAL:
            raise NotImplementedError("本 Sprint 仅支持 SEQUENTIAL (顺序) 模式")

    def _save_signature(self, path, base_model, dataset, layers, mode, epochs):
        # 此逻辑现在主要保留用于记录云端训练的元数据
        sig = {
            "base_model": base_model.name,
            "strategy": dataset.strategy,
            "epochs": epochs,
            "timestamp": datetime.now().isoformat(),
            "execution": "REMOTE"
        }
        if os.path.exists(path):
            with open(os.path.join(path, SIG_FILENAME), "w") as f:
                json.dump(sig, f, indent=4)