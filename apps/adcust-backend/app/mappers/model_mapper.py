from typing import Any, Dict
from app.schemas.model_schema import ModelResponse, ModelSelectRequest
from app.models.model_asset import ModelAsset

class ModelMapper:
    """
    量化职责：负责 ModelAsset 实体与 DTO 的物理字段双向转换。
    """

    @staticmethod
    def to_response_dict(entity: ModelAsset) -> Dict[str, Any]:
        """
        正向转换：ModelAsset (Entity) -> Dict (用于 ModelResponse)
        """
        if not entity:
            return {}

        return {
            "name": entity.name,
            "local_path": entity.local_path,
            "architecture": entity.architecture,
            "precision": entity.precision,
            "trainable_layers": entity.trainable_layers
        }

    @staticmethod
    def to_entity(request_dto: ModelSelectRequest) -> ModelAsset:
        """
        反向转换：ModelSelectRequest (DTO) -> ModelAsset (Entity)
        """
        if not request_dto:
            raise ValueError("Request data is required")

        return ModelAsset(
            name="pending_load",
            local_path=request_dto.local_path,
            architecture="unknown",
            precision="unknown",
            trainable_layers=[]
        )