# -*- coding: utf-8 -*-
from datetime import datetime
from typing import Any, Dict, Optional

from adcust_logic.exceptions import ValidationException, BusinessException


class RemoteTrainingJob:
    """远程训练任务 rich model。

    AdCust 只负责通过 SSH 使用一台“用户已经手动开启”的远程机器。
    这里不表达 pod 启动、pod 关闭或云厂商计费控制能力。
    """

    VALID_STATUSES = {"created", "preparing", "uploading", "running", "collecting", "completed", "failed", "aborted"}

    def __init__(
        self,
        job_id: str,
        provider_id: str,
        dataset_path: str,
        base_model_name_or_path: str,
        adapter_name: str,
        local_output_dir: str,
        remote_workspace_path: str,
        training_params: Dict[str, Any],
        base_model_manifest: Dict[str, Any],
        hourly_rate_usd: float = 0,
        status: str = "created",
        created_at: Optional[datetime] = None,
    ):
        self._validate(job_id, provider_id, dataset_path, base_model_name_or_path, adapter_name, local_output_dir, remote_workspace_path)
        self.job_id = job_id
        self.provider_id = provider_id
        self.dataset_path = dataset_path
        self.base_model_name_or_path = base_model_name_or_path
        self.adapter_name = adapter_name
        self.local_output_dir = local_output_dir
        self.remote_workspace_path = remote_workspace_path.rstrip("/")
        self.training_params = training_params
        self.base_model_manifest = base_model_manifest
        self.hourly_rate_usd = max(0, float(hourly_rate_usd or 0))
        self.status = status
        self.created_at = created_at or datetime.now()
        self.started_at: Optional[datetime] = None
        self.finished_at: Optional[datetime] = None
        self.last_message = "任务已创建"
        self.remote_output_dir = f"{self.remote_workspace_path}/jobs/{self.job_id}/output"
        self.remote_pid_file = f"{self.remote_workspace_path}/jobs/{self.job_id}/train.pid"
        self.remote_manifest_path = f"{self.remote_workspace_path}/jobs/{self.job_id}/expected_base_model_manifest.json"
        self.requires_manual_shutdown = True
        self.current_percentage = 0.0
        self.current_stage = "created"
        self.latest_details: Dict[str, Any] = {}
        self.resources: Dict[str, Dict[str, Any]] = {}

    def _validate(self, job_id: str, provider_id: str, dataset_path: str, base_model: str, adapter_name: str, output_dir: str, workspace: str):
        if not job_id:
            raise ValidationException("ERR_TRAINING_JOB_ID_MISSING")
        if not provider_id:
            raise ValidationException("ERR_PROVIDER_ID_MISSING")
        if not dataset_path:
            raise ValidationException("ERR_DATASET_PATH_REQUIRED")
        if not base_model:
            raise ValidationException("ERR_BASE_MODEL_REQUIRED")
        if not adapter_name or not (2 <= len(adapter_name) <= 64):
            raise ValidationException("ERR_ADAPTER_NAME_INVALID", name=adapter_name)
        if not output_dir:
            raise ValidationException("ERR_PATH_INVALID", path=output_dir)
        if not workspace.startswith("/"):
            raise ValidationException("ERR_REMOTE_WORKSPACE_INVALID", path=workspace)

    def mark(self, status: str, message: str):
        if status not in self.VALID_STATUSES:
            raise BusinessException("ERR_TRAINING_STATUS_INVALID", status=status)
        self.status = status
        self.last_message = message
        if status == "running" and not self.started_at:
            self.started_at = datetime.now()
        if status in {"completed", "failed", "aborted"}:
            self.finished_at = datetime.now()

    @property
    def elapsed_seconds(self) -> float:
        start = self.started_at or self.created_at
        end = self.finished_at or datetime.now()
        return max(0, (end - start).total_seconds())

    @property
    def estimated_cost_usd(self) -> float:
        return round((self.elapsed_seconds / 3600) * self.hourly_rate_usd, 4)

    def progress_event(self, percentage: float, message: Optional[str] = None, **extra) -> Dict[str, Any]:
        next_percentage = max(self.current_percentage, min(100.0, float(percentage)))
        self.current_percentage = next_percentage
        self.current_stage = str(extra.get("stage") or self.current_stage)
        if message:
            self.last_message = message
        resource = extra.get("resource")
        if isinstance(resource, dict) and resource.get("id"):
            self.resources[str(resource["id"])] = dict(resource)
        self.latest_details.update(
            {
                key: value
                for key, value in extra.items()
                if key not in {"resource", "resources", "stage"}
            }
        )
        return {
            "job_id": self.job_id,
            "status": self.status,
            "message": message or self.last_message,
            "percentage": round(self.current_percentage, 2),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "estimated_cost_usd": self.estimated_cost_usd,
            "local_output_dir": self.local_output_dir,
            "requires_manual_shutdown": self.requires_manual_shutdown,
            "resources": list(self.resources.values()),
            **extra,
        }

    def status_snapshot(self) -> Dict[str, Any]:
        """返回不会让前端进度倒退的完整任务快照。"""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "message": self.last_message,
            "percentage": round(self.current_percentage, 2),
            "stage": self.current_stage,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "estimated_cost_usd": self.estimated_cost_usd,
            "local_output_dir": self.local_output_dir,
            "training_params": self.training_params,
            "requires_manual_shutdown": self.requires_manual_shutdown,
            "resources": list(self.resources.values()),
            **self.latest_details,
        }
