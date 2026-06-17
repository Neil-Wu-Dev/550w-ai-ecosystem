from datetime import datetime
from typing import Dict, Optional, Any
from adcust_logic.exceptions import ValidationException

class ComputeProviderAsset:
    """
    远程算力节点实体。
    名字沿用 ComputeProviderAsset 是为了兼容现有架构，但语义上它只代表一台可达的 GPU Linux 主机。
    """

    def __init__(
            self,
            id: str,
            name: str,
            provider_type: str,
            connection_info: Dict[str, Any],
            is_active: bool = False,
            last_heartbeat: Optional[datetime] = None,
            telemetry_data: Optional[Dict] = None,
            created_at: Optional[datetime] = None
    ):
        # 1. 触发元数据层级的业务校验 (保持原样)
        self._validate_base(id, name, provider_type)

        # 2. 基础元数据赋值 (保持原样)
        self.id = id
        self.name = name
        self.provider_type = provider_type.upper()

        # 3. 动态配置赋值 (保持原样)
        self.connection_info = connection_info
        self._validate_connection(connection_info)
        self.remote_workspace_path = connection_info.get("remote_workspace_path")
        self.hourly_rate_usd = float(connection_info.get("hourly_rate_usd"))

        # 4. 状态与遥测信息 (保持原样)
        self.is_active = is_active
        if isinstance(last_heartbeat, str):
            last_heartbeat = datetime.fromisoformat(last_heartbeat)
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        self.last_heartbeat = last_heartbeat
        self.telemetry_data = telemetry_data or {}
        self.created_at = created_at or datetime.now()

        # 远端文件指纹库：Key 为远端路径，Value 为哈希或版本标记。
        self.remote_manifest: Dict[str, str] = self.telemetry_data.get("manifest", {})

    def _validate_base(self, id: str, name: str, p_type: str):
        """执行基础元数据校验。"""
        if not id:
            raise ValidationException("ERR_PROVIDER_ID_MISSING")
        if not name or not (2 <= len(name) <= 32):
            raise ValidationException("ERR_PROVIDER_NAME_INVALID", name=name)
        supported_types = ["SSH", "CUSTOM"]
        if p_type.upper() not in supported_types:
            raise ValidationException("ERR_PROVIDER_TYPE_UNSUPPORTED", type=p_type)

    def _validate_connection(self, connection_info: Dict[str, Any]):
        """校验真实 SSH 连接信息，避免后端生成任何隐式默认值。"""
        required_fields = ["host", "port", "username", "remote_workspace_path", "hourly_rate_usd"]
        for field in required_fields:
            if connection_info.get(field) in (None, ""):
                raise ValidationException("ERR_PROVIDER_CONNECTION_FIELD_REQUIRED", field=field)
        if int(connection_info.get("port")) <= 0:
            raise ValidationException("ERR_PROVIDER_PORT_INVALID", port=connection_info.get("port"))
        if not str(connection_info.get("remote_workspace_path")).startswith("/"):
            raise ValidationException("ERR_REMOTE_WORKSPACE_INVALID", path=connection_info.get("remote_workspace_path"))
        if float(connection_info.get("hourly_rate_usd")) < 0:
            raise ValidationException("ERR_PROVIDER_COST_INVALID", cost=connection_info.get("hourly_rate_usd"))

    @property
    def endpoint_summary(self) -> str:
        """获取连接终结点摘要，避免在 UI 暴露敏感凭据。"""
        if self.provider_type == "SSH":
            host = self.connection_info.get('host', 'Unknown-Host')
            port = self.connection_info.get('port', 22)
            return f"SSH://{host}:{port}"
        return str(self.connection_info.get("endpoint_url", "Cloud-API-Endpoint"))

    def update_telemetry(self, data: Dict):
        """原子化更新遥测数据。"""
        self.telemetry_data.update(data)
        self.last_heartbeat = datetime.now()
        self.is_active = True

    def mark_remote_file(self, remote_path: str, file_hash: str):
        """记录已同步的文件，避免重复上传。"""
        self.remote_manifest[remote_path] = file_hash
        if "manifest" not in self.telemetry_data:
            self.telemetry_data["manifest"] = {}
        self.telemetry_data["manifest"][remote_path] = file_hash

    def get_env_stat(self, key: str, default: Any = None) -> Any:
        """从遥测数据中提取特定环境参数。"""
        return self.telemetry_data.get("env_probe", {}).get(key, default)

    def dict(self) -> Dict[str, Any]:
        """持久化专用字典，明确保留连接配置和非敏感运行状态。"""
        return {
            "id": self.id,
            "name": self.name,
            "provider_type": self.provider_type,
            "connection_info": self.connection_info,
            "is_active": self.is_active,
            "last_heartbeat": self.last_heartbeat.isoformat() if isinstance(self.last_heartbeat, datetime) else self.last_heartbeat,
            "telemetry_data": self.telemetry_data,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
        }

    def __repr__(self):
        return f"<ComputeProviderAsset {self.name} [{self.provider_type}]>"
