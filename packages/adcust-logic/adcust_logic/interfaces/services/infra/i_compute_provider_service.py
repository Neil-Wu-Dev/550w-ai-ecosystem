from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from adcust_logic.models.infra.compute_provider_asset import ComputeProviderAsset
from adcust_logic.interfaces.services.infra.i_compute_driver import IComputeDriver


class IComputeProviderService(ABC):
    """计算供应商基础设施服务接口 (Infrastructure Provisioning Interface).

    本接口是 'adcust-logic' 连接各种异构算力的唯一契约。它不关心底层是 Kaggle、SSH 还是 GPU 云，
    通过标准化的动作（验证、同步、部署）实现“一个应用接管全球算力”的目标。

    遵循原则:
        A (Automation): 自动化环境检测与载荷部署。
        B (Brightness): 云端状态实时透明化，消除“黑盒”担忧。
    """

    @abstractmethod
    def save_provider(self, provider: ComputeProviderAsset) -> None:
        """持久化并建立供应商映射。

        在本地存储（如 SQLite 或 JSON 文件）中保存或更新供应商的连接规格 (connection_info)。
        这是实现“多云管理”的第一步。

        Args:
            provider: 待保存的供应商资产实体。
        """
        pass

    @abstractmethod
    def verify_connection(self, provider: ComputeProviderAsset) -> bool:
        """执行静默连通性测试 (Handshake).

        根据 provider_type 调用对应的底层驱动，尝试使用 connection_info 进行握手。
        对于 SSH 是拨通端口，对于 Kaggle 是调用身份验证 API。

        Args:
            provider: 需要验证的资产实体。

        Returns:
            bool: 连接成功返回 True，否则返回 False 或抛出 ValidationException。
        """
        pass

    @abstractmethod
    def sync_resource_status(self, provider_id: str) -> ComputeProviderAsset:
        """同步并获取云端实时资源状态 (Sync & Fetch).

        这是 fetch_telemetry 的高级版。它会远程执行指令（如 nvidia-smi），
        获取显存、温度、磁盘占用等，并更新 Asset 的 telemetry_data。

        Args:
            provider_id: 资产唯一标识。

        Returns:
            ComputeProviderAsset: 包含最新遥测数据的资产对象。
        """
        pass

    @abstractmethod
    def deploy_remote_payload(self, provider_id: str) -> bool:
        """部署远端执行载荷 (Payload Deployment).

        一键将 'adcust_logic/remote/' 目录下的所有训练逻辑、引擎和环境脚本推送到云端，
        并初始化远程运行环境。实现用户“无感上云”。

        Args:
            provider_id: 目标算力节点的 ID。

        Returns:
            bool: 部署并初始化成功返回 True。
        """
        pass

    @abstractmethod
    def get_provider(self, provider_id: str) -> Optional[ComputeProviderAsset]:
        """根据 ID 获取特定供应商资产。"""
        pass

    @abstractmethod
    def list_all_providers(self) -> List[ComputeProviderAsset]:
        """索引所有已登记的算力资源。

        用于前端展示资源列表，支持用户在不同供应商之间进行“热切换”。
        """
        pass

    @abstractmethod
    def remove_provider(self, provider_id: str, cleanup_remote: bool = False) -> bool:
        """移除供应商登记。

        Args:
            provider_id: 待移除的 ID。
            cleanup_remote: 是否尝试清理远端残留的训练逻辑及临时文件，确保隐私。
        """
        pass

    @abstractmethod
    def get_driver_for(self, provider: ComputeProviderAsset) -> IComputeDriver:
        """为指定远程节点创建物理执行驱动。"""
        pass

    @abstractmethod
    def mark_provider_disconnected(self, provider_id: str) -> ComputeProviderAsset:
        """释放 AdCust 对节点的会话状态标记。"""
        pass
