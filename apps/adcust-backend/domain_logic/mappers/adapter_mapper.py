from typing import Any, Dict
from domain_logic.models.adapter_asset import AdapterAsset


class AdapterMapper:
    """
    量化职责：负责 AdapterAsset 实体与前端 DTO 的双向转换。
    """

    @staticmethod
    def to_dto(entity: AdapterAsset) -> Dict[str, Any]:
        """
        正向转换：Entity -> DTO (Dict)
        注意：base_model 在实体中是 base_model_name
        """
        if not entity:
            return {}

        return {
            "name": entity.name,
            "strategy": entity.strategy,
            "base_model": entity.base_model_name,  # 映射对齐
            "config": entity.config_params,
            "is_active": entity.is_active
        }

    @staticmethod
    def to_entity(dto_data: Dict[str, Any], local_path: str = "") -> AdapterAsset:
        """
        反向转换：DTO (Dict) -> Entity
        注意：实体需要 local_path，如果 DTO 没传，需要从参数补齐。
        """
        if not dto_data:
            raise ValueError("Mapping data cannot be empty")

        return AdapterAsset(
            name=dto_data.get("name", "unknown"),
            local_path=local_path,  # 关键字段补齐
            base_model_name=dto_data.get("base_model", "unknown"),
            strategy=dto_data.get("strategy", "TONE"),
            config_params=dto_data.get("config", {}),
            is_active=dto_data.get("is_active", True)
        )