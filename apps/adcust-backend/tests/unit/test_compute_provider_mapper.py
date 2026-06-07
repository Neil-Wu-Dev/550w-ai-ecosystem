from __future__ import annotations

import unittest

try:
    from _bootstrap import ROOT  # noqa: F401
except ModuleNotFoundError:
    from tests._bootstrap import ROOT  # noqa: F401

from adcust_logic.mappers.compute_provider_mapper import ComputeProviderMapper
from adcust_logic.schemas.compute_provider_schema import ComputeProviderCreateRequest

from tests.fake_server import make_provider


class ComputeProviderMapperTest(unittest.TestCase):
    def test_mapper_converts_create_request_to_entity_and_response(self):
        request = ComputeProviderCreateRequest(
            id="mapper-node",
            name="Mapper Node",
            provider_type="SSH",
            connection_info={
                "host": "mapper.local",
                "port": 22,
                "username": "tester",
                "remote_workspace_path": "/workspace",
                "hourly_rate_usd": 2.5,
            },
        )

        entity = ComputeProviderMapper.to_entity(request)
        response = ComputeProviderMapper.to_response_dict(entity)

        self.assertEqual(entity.id, "mapper-node")
        self.assertFalse(entity.is_active)
        self.assertEqual(response["endpoint_summary"], "SSH://mapper.local:22")
        self.assertEqual(response["hourly_rate_usd"], 2.5)

    def test_mapper_handles_none_and_lists(self):
        self.assertEqual(ComputeProviderMapper.to_response_dict(None), {})
        mapped = ComputeProviderMapper.map_list([make_provider()])
        self.assertEqual(len(mapped), 1)
        self.assertEqual(mapped[0]["id"], "fake-ssh-node")
