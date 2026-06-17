# -*- coding: utf-8 -*-
import json
import logging
import os
import re
import shutil
import uuid
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

from adcust_logic.defs import SIG_FILENAME, TrainingMode
from adcust_logic.exceptions import BusinessException, ValidationException
from adcust_logic.interfaces.services.client.i_dataset_service import IDatasetService
from adcust_logic.interfaces.services.client.i_model_manager import IModelManagerService
from adcust_logic.interfaces.services.infra.i_compute_provider_service import IComputeProviderService
from adcust_logic.interfaces.services.remote.i_training_service import ITrainingService
from adcust_logic.models.remote_training_job import RemoteTrainingJob
from adcust_logic.services.client.model_manifest_service import ModelManifestService
from adcust_logic.services.engines.remote.training_engines.sequential_engine import SequentialEngine

logger = logging.getLogger(__name__)


class TrainingService(ITrainingService):
    def __init__(
        self,
        model_manager: IModelManagerService,
        dataset_service: IDatasetService,
        infra_service: IComputeProviderService,
    ):
        self._model_manager = model_manager
        self._dataset_service = dataset_service
        self._infra_service = infra_service
        self._engines = {TrainingMode.SEQUENTIAL: SequentialEngine()}
        self._jobs: Dict[str, RemoteTrainingJob] = {}
        self._active_driver = None
        self._manifest_service = ModelManifestService()

    def customize_new_adapter(
        self,
        target_dir: str,
        trainable_layers: List[str],
        mode: TrainingMode = TrainingMode.SEQUENTIAL,
        custom_name: Optional[str] = None,
        epochs: int = 1,
    ) -> Generator[Dict[str, Any], None, None]:
        """兼容旧 API 的入口；真实云训练必须走显式远程训练 workflow。"""
        base_model = self._model_manager.get_active_model()
        dataset = self._dataset_service.get_active_dataset()
        if not base_model or not dataset:
            yield {"status": "error", "message": "Base model or dataset is not loaded", "percentage": 0}
            return
        yield {
            "status": "error",
            "message": "Legacy implicit training entry is disabled. Use /api/v1/training/remote/start with an explicit provider_id.",
            "percentage": 0,
        }

    def start_remote_adapter_job(self, request_data: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """完整远程训练流水线：准备环境 -> 准备数据 -> 上传 -> 远端训练 -> 下载 adapter。"""
        provider_id = request_data.get("provider_id")
        provider = self._infra_service.get_provider(provider_id)
        if not provider:
            raise BusinessException("ERR_PROVIDER_NOT_FOUND", provider_id=provider_id)

        dataset_path = os.path.abspath(request_data.get("dataset_path"))
        if not os.path.exists(dataset_path):
            raise ValidationException("ERR_DATASET_PATH_REQUIRED", path=dataset_path)

        local_base_model_path = request_data.get("local_base_model_path")
        if not local_base_model_path:
            raise ValidationException("ERR_LOCAL_BASE_MODEL_REQUIRED")
        requested_model = str(request_data.get("base_model_name_or_path") or "").strip()
        base_model_manifest = self._manifest_service.build_local_manifest(local_base_model_path, requested_model)
        repository_id = base_model_manifest.get("repository_id")
        revision = base_model_manifest.get("revision")
        if not repository_id or not revision:
            raise ValidationException(
                "ERR_BASE_MODEL_PROVENANCE_REQUIRED",
                path=local_base_model_path,
            )
        if "/" in requested_model and requested_model.lower() != str(repository_id).lower():
            raise ValidationException(
                "ERR_BASE_MODEL_REPOSITORY_MISMATCH",
                requested=requested_model,
                detected=repository_id,
            )

        adapter_name = request_data.get("adapter_name")
        output_root = request_data.get("local_output_root")
        if not adapter_name:
            raise ValidationException("ERR_ADAPTER_NAME_INVALID", name=adapter_name)
        if not output_root:
            raise ValidationException("ERR_LOCAL_OUTPUT_ROOT_REQUIRED")

        job = RemoteTrainingJob(
            job_id=f"job_{uuid.uuid4().hex[:10]}",
            provider_id=provider.id,
            dataset_path=dataset_path,
            base_model_name_or_path=repository_id,
            adapter_name=adapter_name,
            local_output_dir=os.path.abspath(os.path.join(output_root, adapter_name)),
            remote_workspace_path=provider.remote_workspace_path,
            training_params=self._build_training_params(request_data),
            base_model_manifest=base_model_manifest,
            hourly_rate_usd=provider.hourly_rate_usd,
        )
        self._jobs[job.job_id] = job

        driver = self._infra_service.get_driver_for(provider)
        self._active_driver = driver
        temp_local_files: List[str] = []
        upload_dataset_path = dataset_path

        remote_job_dir = f"{job.remote_workspace_path}/jobs/{job.job_id}"
        remote_model_dir = f"{job.remote_workspace_path}/models/{self._safe_remote_name(job.base_model_manifest.get('combined_hash'))}"
        remote_venv_dir = f"{job.remote_workspace_path}/runtime/venv"
        remote_python_path = f"{remote_venv_dir}/bin/python"
        remote_config_path = f"{remote_job_dir}/train_config.json"
        remote_script_path = f"{remote_job_dir}/train_entry.py"
        remote_manifest_path = job.remote_manifest_path
        local_config_path = os.path.join(os.getcwd(), "data", f"{job.job_id}_train_config.json")
        local_manifest_path = os.path.join(os.getcwd(), "data", f"{job.job_id}_base_model_manifest.json")

        try:
            job.mark("uploading", "Connecting to user-managed SSH node. AdCust does not start or stop cloud pods.")
            yield job.progress_event(
                5,
                stage="ssh_connect",
                resource={"id": "ssh", "kind": "connection", "name": "SSH connection", "status": "checking"},
            )
            driver.connect(provider.connection_info)
            yield job.progress_event(
                7,
                message="SSH connection verified",
                stage="ssh_connected",
                resource={"id": "ssh", "kind": "connection", "name": "SSH connection", "status": "ready"},
            )

            job.mark("preparing", "Checking remote Python, CUDA, and training dependencies")
            yield job.progress_event(8, stage="dependency_check")
            for line in driver.exec_command(self._build_remote_dependency_command(remote_venv_dir)):
                clean_line = line.strip()
                if clean_line:
                    parsed = self._parse_remote_training_event(clean_line)
                    if parsed:
                        percentage = parsed.pop("percentage", 12)
                        message = parsed.pop("message", clean_line)
                        yield job.progress_event(percentage, message=message, log_line=clean_line, **parsed)
                    else:
                        yield job.progress_event(12, message=clean_line, log_line=clean_line, stage="dependency_check")
            job.mark("preparing", "Remote training dependencies are ready")
            yield job.progress_event(15, stage="dependency_ready")

            if dataset_path.lower().endswith(".pdf"):
                job.mark("preparing", "Extracting PDF text and creating a temporary JSON training dataset")
                yield job.progress_event(16, stage="dataset_prepare")
                upload_dataset_path = self._build_temporary_json_dataset_from_pdf(dataset_path, job.job_id)
                temp_local_files.append(upload_dataset_path)
                yield job.progress_event(18, message="PDF text extracted into temporary JSON dataset", stage="dataset_ready")

            job.mark("uploading", "Creating remote job workspace")
            yield job.progress_event(20, stage="remote_workspace")
            if hasattr(driver, "ensure_dir"):
                driver.ensure_dir(remote_job_dir)
                driver.ensure_dir(remote_model_dir)

            remote_dataset_path = f"{remote_job_dir}/dataset{os.path.splitext(upload_dataset_path)[1] or '.json'}"
            script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "remote", "train_entry.py"))
            config_payload = {
                **job.training_params,
                "dataset_path": remote_dataset_path,
                "output_dir": job.remote_output_dir,
                "base_model_name_or_path": job.base_model_name_or_path,
                "base_model_revision": revision,
                "remote_model_dir": remote_model_dir,
                "adapter_name": job.adapter_name,
                "expected_base_model_manifest": job.base_model_manifest,
                "resume_from_checkpoint": False,
            }

            os.makedirs(os.path.dirname(local_config_path), exist_ok=True)
            with open(local_config_path, "w", encoding="utf-8") as f:
                json.dump(config_payload, f, ensure_ascii=False, indent=2)
            with open(local_manifest_path, "w", encoding="utf-8") as f:
                json.dump(job.base_model_manifest, f, ensure_ascii=False, indent=2)

            job.mark("uploading", "Uploading dataset, training config, base model manifest, and remote training script")
            yield job.progress_event(25, stage="upload")
            driver.push_file(upload_dataset_path, remote_dataset_path)
            yield job.progress_event(
                27,
                message="Training dataset uploaded",
                stage="dataset_uploaded",
                resource={"id": "dataset", "kind": "training_asset", "name": os.path.basename(upload_dataset_path), "status": "ready"},
            )
            driver.push_file(local_config_path, remote_config_path)
            yield job.progress_event(
                29,
                message="Training configuration uploaded",
                stage="config_uploaded",
                resource={"id": "training_config", "kind": "training_asset", "name": "train_config.json", "status": "ready"},
            )
            driver.push_file(local_manifest_path, remote_manifest_path)
            yield job.progress_event(
                31,
                message=f"Base model identity bound to {repository_id}@{revision[:12]}",
                stage="manifest_uploaded",
                resource={"id": "model_manifest", "kind": "training_asset", "name": "Base model manifest", "status": "ready"},
            )
            driver.push_file(script_path, remote_script_path)
            yield job.progress_event(
                33,
                message="Remote training engine uploaded",
                stage="engine_uploaded",
                resource={"id": "training_engine", "kind": "training_asset", "name": "train_entry.py", "status": "ready"},
            )

            job.mark("running", "Remote training process started")
            yield job.progress_event(35, stage="training_start")
            command = self._build_remote_command(remote_job_dir, remote_script_path, remote_config_path, remote_python_path)
            for line in driver.exec_command(command):
                clean_line = line.strip()
                if not clean_line:
                    continue
                parsed = self._parse_remote_training_event(clean_line)
                if parsed:
                    percentage = parsed.pop("percentage", 70)
                    message = parsed.pop("message", clean_line)
                    yield job.progress_event(percentage, message=message, log_line=clean_line, **parsed)
                else:
                    yield job.progress_event(70, message=clean_line, log_line=clean_line, stage="training")

            job.mark("collecting", "Downloading adapter artifacts")
            yield job.progress_event(90, stage="download")
            if os.path.exists(job.local_output_dir):
                shutil.rmtree(job.local_output_dir)
            if hasattr(driver, "pull_dir"):
                driver.pull_dir(job.remote_output_dir, job.local_output_dir)
            else:
                raise BusinessException("ERR_DRIVER_CAPABILITY_MISSING", capability="pull_dir")

            self._save_remote_signature(job)
            if request_data.get("cleanup_remote"):
                job.mark("collecting", "Cleaning remote job workspace")
                yield job.progress_event(96, stage="remote_cleanup")
                for _ in driver.exec_command(f"rm -rf '{remote_job_dir}'"):
                    pass

            job.mark("completed", "Training completed. Adapter artifacts were saved locally. Manually stop or terminate the remote server from the provider website now.")
            yield job.progress_event(
                100,
                stage="completed",
                requires_manual_shutdown=True,
                shutdown_warning="Training finished and adapter was downloaded. AdCust only closes SSH; your remote server may still be running and billing. Stop or terminate it on the provider website now.",
            )
        except Exception as exc:
            job.mark("failed", f"Training failed: {str(exc)}")
            yield job.progress_event(0, stage="failed", error=str(exc))
        finally:
            for path in [local_config_path, local_manifest_path, *temp_local_files]:
                if path and os.path.exists(path):
                    os.remove(path)
            # 只释放 AdCust 的 SSH 会话；这不是关闭/销毁云厂商 pod。
            driver.disconnect()
            self._active_driver = None

    def _build_remote_dependency_command(self, remote_venv_dir: str) -> str:
        packages = "torch transformers peft bitsandbytes accelerate datasets huggingface_hub safetensors"
        return (
            "bash -lc 'set +e; "
            "echo \"[AdCust] Checking remote Python runtime\"; "
            "if ! command -v python3 >/dev/null; then "
            "echo \"[AdCust] python3 is missing; attempting automatic OS package installation\"; "
            "if command -v apt-get >/dev/null; then "
            "(command -v sudo >/dev/null && sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv) || "
            "(apt-get update && apt-get install -y python3 python3-pip python3-venv); "
            "elif command -v yum >/dev/null; then "
            "(command -v sudo >/dev/null && sudo yum install -y python3 python3-pip) || "
            "(yum install -y python3 python3-pip); "
            "elif command -v dnf >/dev/null; then "
            "(command -v sudo >/dev/null && sudo dnf install -y python3 python3-pip) || "
            "(dnf install -y python3 python3-pip); "
            "fi; "
            "fi; "
            "command -v python3 >/dev/null || { echo \"python3 is missing and could not be installed automatically\"; exit 127; }; "
            f"mkdir -p \"{remote_venv_dir}\"; "
            f"if [ ! -x \"{remote_venv_dir}/bin/python\" ]; then "
            f"echo \"[AdCust] Creating isolated Python virtual environment: {remote_venv_dir}\"; "
            f"python3 -m venv --system-site-packages \"{remote_venv_dir}\"; "
            "venv_status=$?; "
            "if [ $venv_status -ne 0 ]; then "
            "echo \"[AdCust] python3 venv module is missing; attempting to install python3-venv\"; "
            "if command -v apt-get >/dev/null; then "
            "(command -v sudo >/dev/null && sudo apt-get update && sudo apt-get install -y python3-venv) || "
            "(apt-get update && apt-get install -y python3-venv); "
            "fi; "
            f"python3 -m venv --system-site-packages \"{remote_venv_dir}\"; "
            "fi; "
            "fi; "
            f"if [ -f \"{remote_venv_dir}/pyvenv.cfg\" ]; then "
            f"sed -i \"s/include-system-site-packages = false/include-system-site-packages = true/\" \"{remote_venv_dir}/pyvenv.cfg\"; "
            "fi; "
            f"\"{remote_venv_dir}/bin/python\" -m ensurepip --upgrade >/dev/null 2>&1 || true; "
            f"\"{remote_venv_dir}/bin/python\" -m pip --version; "
            f"python3 - <<\"PY\"\n"
            f"import json\n"
            f"print(json.dumps({{\"adcust_event\":\"dependency_plan\",\"message\":\"Remote dependency plan created inside isolated venv\",\"packages\":\"{packages}\",\"venv\":\"{remote_venv_dir}\"}}))\n"
            f"PY\n"
            f"\"{remote_venv_dir}/bin/python\" - <<\"PY\"\n"
            "import importlib.util, json, pathlib\n"
            "mods=[\"torch\",\"transformers\",\"peft\",\"bitsandbytes\",\"accelerate\",\"datasets\",\"huggingface_hub\",\"safetensors\"]\n"
            "missing=[m for m in mods if importlib.util.find_spec(m) is None]\n"
            "for name in mods:\n"
            "    status=\"missing\" if name in missing else \"ready\"\n"
            "    print(json.dumps({\"adcust_event\":\"dependency_status\",\"message\":f\"{name}: {status}\",\"resource\":{\"id\":f\"dependency:{name}\",\"kind\":\"python_dependency\",\"name\":name,\"status\":status}}))\n"
            f"pathlib.Path(\"{remote_venv_dir}/missing_packages.txt\").write_text(\" \".join(missing), encoding=\"utf-8\")\n"
            "print(json.dumps({\"adcust_event\":\"dependency_probe\",\"missing\":missing,\"message\":\"Dependency probe completed\"}))\n"
            "PY\n"
            f"missing=$(cat \"{remote_venv_dir}/missing_packages.txt\" 2>/dev/null); "
            "if [ -n \"$missing\" ]; then "
            "echo \"[AdCust] Installing missing remote training dependencies into isolated venv\"; "
            f"\"{remote_venv_dir}/bin/python\" -m pip install --upgrade pip setuptools wheel; "
            f"\"{remote_venv_dir}/bin/python\" -m pip install $missing; "
            "fi; "
            f"\"{remote_venv_dir}/bin/python\" - <<\"PY\"\n"
            "import torch, transformers, peft, bitsandbytes, accelerate, datasets, huggingface_hub, safetensors, json\n"
            "mods=[\"torch\",\"transformers\",\"peft\",\"bitsandbytes\",\"accelerate\",\"datasets\",\"huggingface_hub\",\"safetensors\"]\n"
            "for name in mods:\n"
            "    print(json.dumps({\"adcust_event\":\"dependency_status\",\"message\":f\"{name}: ready\",\"resource\":{\"id\":f\"dependency:{name}\",\"kind\":\"python_dependency\",\"name\":name,\"status\":\"ready\"}}))\n"
            "print(json.dumps({\"adcust_event\":\"dependency_ready\",\"message\":\"Remote dependencies ready\",\"torch\":torch.__version__,\"cuda_available\":torch.cuda.is_available()}))\n"
            "PY'"
        )

    def _build_remote_command(
        self,
        remote_job_dir: str,
        remote_script_path: str,
        remote_config_path: str,
        remote_python_path: str = "python3",
    ) -> str:
        return (
            f"cd '{remote_job_dir}' && "
            f"bash -lc 'echo $$ > train.pid; exec \"{remote_python_path}\" \"{remote_script_path}\" --config \"{remote_config_path}\"'"
        )

    def _build_temporary_json_dataset_from_pdf(self, pdf_path: str, job_id: str) -> str:
        try:
            import fitz
        except Exception as exc:
            raise ValidationException("ERR_PDF_PARSER_MISSING", reason=str(exc))

        chunks: List[str] = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text = page.get_text("text").strip()
                if text:
                    chunks.append(re.sub(r"\s+", " ", text))
        if not chunks:
            raise ValidationException("ERR_PDF_TEXT_EMPTY", path=pdf_path)

        # PDF 只保留纯文本。远端引擎会按 token 滑窗执行持续预训练，
        # 用户不需要把故事人工改造成问答对。
        records = [{"text": "\n\n".join(chunks)}]
        os.makedirs(os.path.join(os.getcwd(), "data"), exist_ok=True)
        temp_path = os.path.abspath(os.path.join(os.getcwd(), "data", f"{job_id}_pdf_dataset.json"))
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return temp_path

    def _parse_remote_training_event(self, line: str) -> Optional[Dict[str, Any]]:
        marker = "ADCUST_EVENT "
        if marker in line:
            candidate = line.split(marker, 1)[1]
        else:
            json_start = line.find('{"adcust_event"')
            candidate = line[json_start:] if json_start >= 0 else line
        try:
            payload, _ = json.JSONDecoder().raw_decode(candidate.lstrip())
        except Exception:
            return None
        event_type = payload.get("adcust_event")
        if not event_type:
            return None
        result = {key: value for key, value in payload.items() if key != "adcust_event"}
        result["stage"] = event_type
        if event_type == "train_metric":
            train_progress = float(payload.get("train_progress") or 0)
            result["percentage"] = 60 + min(24, train_progress * 0.24)
        elif event_type == "model_download_progress":
            download_progress = float(payload.get("download_progress") or 0)
            result["percentage"] = 45 + min(6, download_progress * 0.06)
        else:
            result["percentage"] = {
                "dependency_probe": 10,
                "dependency_plan": 9,
                "dependency_status": 12,
                "dependency_ready": 15,
                "config_loaded": 36,
                "dataset_ready": 40,
                "knowledge_synthesis_started": 55,
                "knowledge_synthesis_progress": 57,
                "knowledge_synthesis_ready": 59,
                "knowledge_synthesis_fallback": 59,
                "adapter_trainability_verified": 59,
                "model_download": 45,
                "model_download_progress": 45,
                "model_snapshot_finalize": 51,
                "model_file_status": 50,
                "model_cache_invalid": 44,
                "model_ready": 52,
                "model_verified": 58,
                "training_started": 60,
                "training_summary": 85,
                "saving_adapter": 86,
                "training_completed": 88,
            }.get(event_type, 70)
        return result

    def _safe_remote_name(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "model")).strip("_")[:96] or "model"

    def get_job_status(self, job_id: str) -> RemoteTrainingJob:
        job = self._jobs.get(job_id)
        if not job:
            raise BusinessException("ERR_TRAINING_JOB_NOT_FOUND", job_id=job_id)
        return job

    def abort_remote_job(self, job_id: str) -> RemoteTrainingJob:
        """通过远端 pid 文件终止训练进程，并同步本地任务状态。"""
        job = self.get_job_status(job_id)
        provider = self._infra_service.get_provider(job.provider_id)
        if not provider:
            raise BusinessException("ERR_PROVIDER_NOT_FOUND", provider_id=job.provider_id)
        driver = self._infra_service.get_driver_for(provider)
        try:
            driver.connect(provider.connection_info)
            command = (
                f"if [ -f '{job.remote_pid_file}' ]; then "
                f"kill -TERM $(cat '{job.remote_pid_file}') 2>/dev/null || true; "
                f"echo 'AdCust remote termination signal sent'; "
                f"else echo 'AdCust remote pid file not found'; fi"
            )
            for _ in driver.exec_command(command):
                pass
            job.mark("aborted", "Remote training termination signal sent")
            return job
        finally:
            driver.disconnect()

    def _build_training_params(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        required = [
            "epochs",
            "learning_rate",
            "max_seq_length",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "target_modules",
            "use_qlora",
        ]
        for key in required:
            if request_data.get(key) in (None, "", []):
                raise ValidationException("ERR_TRAINING_PARAM_REQUIRED", field=key)
        return {
            "epochs": int(request_data.get("epochs")),
            "learning_rate": float(request_data.get("learning_rate")),
            "max_seq_length": int(request_data.get("max_seq_length")),
            "lora_rank": int(request_data.get("lora_rank")),
            "lora_alpha": int(request_data.get("lora_alpha")),
            "lora_dropout": float(request_data.get("lora_dropout")),
            "target_modules": request_data.get("target_modules"),
            "use_qlora": bool(request_data.get("use_qlora")),
            "checkpoint_save_steps": int(request_data.get("checkpoint_save_steps", 25)),
            "checkpoint_save_total_limit": int(request_data.get("checkpoint_save_total_limit", 3)),
        }

    def _run_remote_training(self, provider, model, dataset, layers, epochs):
        """旧私有入口已停用，避免隐式状态和硬编码远端路径。"""
        raise BusinessException("ERR_LEGACY_TRAINING_ENTRY_DISABLED")

    def abort_training(self, mode: TrainingMode = TrainingMode.SEQUENTIAL):
        """中止当前本地或远端命令。"""
        logger.warning("收到停止信号，正在尝试中止当前训练命令。")
        if self._active_driver and hasattr(self._active_driver, "abort_current_command"):
            self._active_driver.abort_current_command()
        engine = self._engines.get(mode)
        if engine:
            engine.abort()

    def _validate_support(self, strategy: str, mode: TrainingMode):
        if mode != TrainingMode.SEQUENTIAL:
            raise NotImplementedError("当前版本仅支持 SEQUENTIAL 模式")

    def _save_signature(self, path, base_model, dataset, layers, mode, epochs):
        sig = {
            "base_model": base_model.name,
            "strategy": dataset.strategy,
            "epochs": epochs,
            "timestamp": datetime.now().isoformat(),
            "execution": "REMOTE",
        }
        if os.path.exists(path):
            with open(os.path.join(path, SIG_FILENAME), "w", encoding="utf-8") as f:
                json.dump(sig, f, indent=4)

    def _save_remote_signature(self, job: RemoteTrainingJob):
        os.makedirs(job.local_output_dir, exist_ok=True)
        signature_path = os.path.join(job.local_output_dir, SIG_FILENAME)
        remote_signature: Dict[str, Any] = {}
        if os.path.isfile(signature_path):
            try:
                with open(signature_path, "r", encoding="utf-8") as f:
                    remote_signature = json.load(f)
            except (OSError, ValueError, json.JSONDecodeError):
                remote_signature = {}
        sig = {
            **remote_signature,
            "signature_version": "1.0",
            "base_model": job.base_model_name_or_path,
            "adapter_name": job.adapter_name,
            "provider_id": job.provider_id,
            "job_id": job.job_id,
            "execution": "REMOTE_SSH",
            "model_format": "huggingface_transformers",
            "adapter_format": "peft_lora",
            "quantization": "qlora_4bit" if job.training_params.get("use_qlora") else "none",
            "training_params": {
                **remote_signature.get("training_params", {}),
                **job.training_params,
            },
            "base_model_manifest": job.base_model_manifest,
            "base_model_manifest_hash": job.base_model_manifest.get("combined_hash"),
            "estimated_cost_usd": job.estimated_cost_usd,
            "requires_manual_shutdown": True,
            "shutdown_warning": "Training finished and adapter was downloaded. AdCust only closes SSH; your remote server may still be running and billing. Stop or terminate it on the provider website now.",
            "timestamp": datetime.now().isoformat(),
        }
        with open(signature_path, "w", encoding="utf-8") as f:
            json.dump(sig, f, ensure_ascii=False, indent=2)
