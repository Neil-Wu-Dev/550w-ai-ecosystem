from __future__ import annotations

import unittest

try:
    from _bootstrap import ROOT  # noqa: F401
except ModuleNotFoundError:
    from tests._bootstrap import ROOT  # noqa: F401

from adcust_logic.exceptions import ValidationException
from adcust_logic.models.infra.compute_provider_asset import ComputeProviderAsset

from tests.fake_server import make_provider


class ComputeProviderAssetModelTest(unittest.TestCase):
    def test_provider_exposes_safe_endpoint_summary_and_dict(self):
        provider = make_provider()

        self.assertEqual(provider.endpoint_summary, "SSH://fake.local:22")
        payload = provider.dict()
        self.assertEqual(payload["id"], "fake-ssh-node")
        self.assertIn("connection_info", payload)

    def test_provider_rejects_missing_connection_fields(self):
        with self.assertRaises(ValidationException):
            ComputeProviderAsset(
                id="bad",
                name="Bad Provider",
                provider_type="SSH",
                connection_info={
                    "host": "fake.local",
                    "port": 22,
                    "username": "tester",
                    "hourly_rate_usd": 1,
                },
            )

    def test_provider_rejects_relative_workspace_and_negative_cost(self):
        with self.assertRaises(ValidationException):
            ComputeProviderAsset(
                id="bad-workspace",
                name="Bad Workspace",
                provider_type="SSH",
                connection_info={
                    "host": "fake.local",
                    "port": 22,
                    "username": "tester",
                    "remote_workspace_path": "relative",
                    "hourly_rate_usd": 1,
                },
            )

        with self.assertRaises(ValidationException):
            ComputeProviderAsset(
                id="bad-cost",
                name="Bad Cost",
                provider_type="SSH",
                connection_info={
                    "host": "fake.local",
                    "port": 22,
                    "username": "tester",
                    "remote_workspace_path": "/workspace",
                    "hourly_rate_usd": -1,
                },
            )
