"""测试用基础设施服务，向训练服务注入 fake driver。"""

from __future__ import annotations

from typing import Optional

try:
    from _bootstrap import ROOT  # noqa: F401
except ModuleNotFoundError:
    from tests._bootstrap import ROOT  # noqa: F401

from adcust_logic.models.infra.compute_provider_asset import ComputeProviderAsset

from .fake_compute_driver import FakeComputeDriver


class FakeInfraService:
    def __init__(self, provider: ComputeProviderAsset, driver: FakeComputeDriver):
        self.provider = provider
        self.driver = driver

    def save_provider(self, provider: ComputeProviderAsset) -> None:
        self.provider = provider

    def verify_connection(self, provider: ComputeProviderAsset) -> bool:
        return True

    def sync_resource_status(self, provider_id: str) -> ComputeProviderAsset:
        return self.provider

    def deploy_remote_payload(self, provider_id: str) -> bool:
        return True

    def get_provider(self, provider_id: str) -> Optional[ComputeProviderAsset]:
        return self.provider if provider_id == self.provider.id else None

    def list_all_providers(self):
        return [self.provider]

    def remove_provider(self, provider_id: str, cleanup_remote: bool = False) -> bool:
        return provider_id == self.provider.id

    def get_driver_for(self, provider: ComputeProviderAsset):
        return self.driver

    def mark_provider_disconnected(self, provider_id: str) -> ComputeProviderAsset:
        self.provider.is_active = False
        return self.provider
