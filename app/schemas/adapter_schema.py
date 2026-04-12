from typing import Dict, Any

class AdapterDTO:
    @staticmethod
    def from_entity(entity: Any) -> Dict[str, Any]:
        """将领域实体转换为前端可读的字典"""
        if not entity: return {}
        return {
            "name": entity.name,
            "strategy": entity.strategy,
            "base_model": entity.base_model_name,
            "config": entity.config_params,
            "is_active": entity.is_active
        }