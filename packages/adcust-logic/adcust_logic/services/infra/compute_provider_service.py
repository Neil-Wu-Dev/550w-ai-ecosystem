import os
import json
import logging
from typing import List, Optional, Dict, Any
from adcust_logic.interfaces.services.infra.i_compute_provider_service import IComputeProviderService
from adcust_logic.models.infra.compute_provider_asset import ComputeProviderAsset
from adcust_logic.services.engines.remote.ssh_driver import SSHComputeDriver
from adcust_logic.interfaces.services.infra.i_compute_driver import IComputeDriver
from adcust_logic.exceptions import BusinessException

logger = logging.getLogger(__name__)

class ComputeProviderService(IComputeProviderService):
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self._providers: Dict[str, ComputeProviderAsset] = {}
        self._load_from_storage()

    def _load_from_storage(self):
        """从本地 JSON 加载节点配置。"""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    # 兼容处理：如果数据是列表则转为字典，如果是字典则直接用
                    if isinstance(data, list):
                        for item in data:
                            self._providers[item['id']] = ComputeProviderAsset(**item)
                    else:
                        for p_id, p_data in data.items():
                            self._providers[p_id] = ComputeProviderAsset(**p_data)
                except Exception as e:
                    logger.error(f"加载供应商配置失败: {e}")

    def save_provider(self, provider: ComputeProviderAsset) -> None:
        self._providers[provider.id] = provider
        self._sync_to_disk()

    def _sync_to_disk(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            output = {k: v.dict() for k, v in self._providers.items()}
            json.dump(output, f, default=str, indent=4)

    def remove_provider(self, provider_id: str, cleanup_remote: bool = False) -> bool:
        """删除节点登记，可选清理远端工作目录。"""
        provider = self._providers.get(provider_id)
        if provider and cleanup_remote:
            driver = self.get_driver_for(provider)
            try:
                driver.connect(provider.connection_info)
                workspace = provider.remote_workspace_path
                for _ in driver.exec_command(f"rm -rf '{workspace}'"):
                    pass
            finally:
                driver.disconnect()
        if provider_id in self._providers:
            del self._providers[provider_id]
            self._sync_to_disk()
            return True
        return False

    def verify_connection(self, provider: ComputeProviderAsset) -> bool:
        """执行真实 SSH 握手，不吞掉错误细节。"""
        try:
            driver = self.get_driver_for(provider)
            ok = driver.test_connection(provider.connection_info)
            provider.is_active = ok
            provider.update_telemetry({
                "session": {
                    "state": "connected" if ok else "disconnected",
                    "note": "SSH connect attempt succeeded." if ok else "SSH connect attempt failed."
                }
            })
            provider.is_active = ok
            self.save_provider(provider)
            return ok
        except Exception as exc:
            provider.update_telemetry({
                "session": {
                    "state": "disconnected",
                    "note": f"SSH connect attempt failed. The server may be stopped or unreachable: {exc}"
                }
            })
            provider.is_active = False
            self.save_provider(provider)
            raise BusinessException("ERR_PROVIDER_VERIFY_FAILED", reason=str(exc))

    def get_driver_for(self, provider: ComputeProviderAsset) -> IComputeDriver:
        """根据节点类型创建底层驱动。MVP 只实现 SSH，但入口保持可扩展。"""
        if provider.provider_type not in {"SSH", "CUSTOM"}:
            raise BusinessException("ERR_PROVIDER_TYPE_UNSUPPORTED", type=provider.provider_type)
        return SSHComputeDriver()

    def get_driver(self, provider: ComputeProviderAsset) -> IComputeDriver:
        return self.get_driver_for(provider)

    def sync_resource_status(self, provider_id: str) -> ComputeProviderAsset:
        provider = self.get_provider(provider_id)
        if not provider:
            raise BusinessException("ERR_PROVIDER_NOT_FOUND", provider_id=provider_id)

        driver = self.get_driver_for(provider)
        try:
            driver.connect(provider.connection_info)
            provider.is_active = True
            command = (
                "python3 - <<'PY'\n"
                "import json, shutil, subprocess\n"
                "data={'has_python': True}\n"
                "try:\n"
                " out=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'], text=True)\n"
                " g=[]\n"
                " for line in out.strip().splitlines():\n"
                "  name,total,free,temp,util=[x.strip() for x in line.split(',')]\n"
                "  g.append({'name':name,'memory_total_mb':int(total),'memory_free_mb':int(free),'temperature_c':int(temp),'utilization_percent':int(util)})\n"
                " data['gpus']=g; data['has_gpu']=len(g)>0\n"
                "except Exception as e:\n"
                " data['has_gpu']=False; data['gpu_error']=str(e)\n"
                "try:\n"
                " import torch\n"
                " data['has_pytorch']=True; data['torch_version']=torch.__version__; data['cuda_available']=torch.cuda.is_available()\n"
                "except Exception as e:\n"
                " data['has_pytorch']=False; data['torch_error']=str(e)\n"
                "du=shutil.disk_usage('.')\n"
                "data['disk_free_gb']=round(du.free/1024/1024/1024,2)\n"
                "print(json.dumps(data, ensure_ascii=False))\n"
                "PY"
            )
            output = ""
            for line in driver.exec_command(command):
                output += line
            env_data: Dict[str, Any] = json.loads(output.strip().splitlines()[-1])
            provider.update_telemetry({
                "env_probe": env_data,
                "session": {
                    "state": "connected",
                    "note": "SSH connection check succeeded. The server is reachable right now."
                }
            })
            provider.is_active = True
            self.save_provider(provider)
            return provider
        except Exception as exc:
            provider.update_telemetry({
                "session": {
                    "state": "disconnected",
                    "note": f"SSH connection check failed. The server may be stopped or unreachable: {exc}"
                }
            })
            provider.is_active = False
            self.save_provider(provider)
            return provider
        finally:
            driver.disconnect()

    def deploy_remote_payload(self, provider_id: str) -> bool:
        provider = self.get_provider(provider_id)
        if not provider:
            raise BusinessException("ERR_PROVIDER_NOT_FOUND", provider_id=provider_id)
        logger.info(f"--- [Infra] 节点 {provider.name} 将在训练任务启动时按需同步载荷 ---")
        return True

    def get_provider(self, provider_id: str) -> Optional[ComputeProviderAsset]:
        return self._providers.get(provider_id)

    def list_all_providers(self) -> List[ComputeProviderAsset]:
        return list(self._providers.values())

    # 接口可能还要求 get_all_providers 这个名字，我们做个别名映射
    def get_all_providers(self) -> List[ComputeProviderAsset]:
        return self.list_all_providers()

    def mark_provider_disconnected(self, provider_id: str) -> ComputeProviderAsset:
        """释放 AdCust 对该节点的会话占用标记。不会关闭云厂商实例。"""
        provider = self.get_provider(provider_id)
        if not provider:
            raise BusinessException("ERR_PROVIDER_NOT_FOUND", provider_id=provider_id)
        provider.is_active = False
        provider.update_telemetry({
            "session": {
                "state": "released",
                "note": "AdCust SSH session released. Stop or terminate the server on the provider website to stop billing."
            }
        })
        provider.is_active = False
        self.save_provider(provider)
        return provider
