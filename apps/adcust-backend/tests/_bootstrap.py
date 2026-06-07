"""测试启动辅助。

这个文件只服务测试环境：它把后端与 logic package 加入 import path，
并为本轮测试不需要真正调用的重型训练依赖提供最小 stub，避免单元测试
为了导入 TrainingService 就要求本机安装 torch/transformers/peft。
"""

from __future__ import annotations

import sys
import types
import logging
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LOGIC_ROOT = ROOT / "packages" / "adcust-logic"
BACKEND_ROOT = ROOT / "apps" / "adcust-backend"

for path in (str(LOGIC_ROOT), str(BACKEND_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

# 测试终端只展示测试报告，不展示后端启动时的业务日志。
# 这样右键运行测试时，用户看到的是 PASS/FAIL/SKIP，而不是 app.main 的启动日志。
logging.disable(logging.CRITICAL)


def _install_optional_training_stubs() -> None:
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.float16 = "float16"
        torch_stub.float32 = "float32"
        torch_stub.LongTensor = object
        torch_stub.FloatTensor = object
        torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        sys.modules["torch"] = torch_stub

    if "transformers" not in sys.modules:
        transformers_stub = types.ModuleType("transformers")

        class _AutoModelForCausalLM:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise RuntimeError("测试 stub 不执行真实模型加载")

        class _AutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise RuntimeError("测试 stub 不执行真实 tokenizer 加载")

        class _GenerationConfig:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class _StoppingCriteria:
            pass

        class _StoppingCriteriaList(list):
            pass

        class _TextIteratorStreamer:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        transformers_stub.AutoModelForCausalLM = _AutoModelForCausalLM
        transformers_stub.AutoTokenizer = _AutoTokenizer
        transformers_stub.BitsAndBytesConfig = object
        transformers_stub.GenerationConfig = _GenerationConfig
        transformers_stub.StoppingCriteria = _StoppingCriteria
        transformers_stub.StoppingCriteriaList = _StoppingCriteriaList
        transformers_stub.TextIteratorStreamer = _TextIteratorStreamer
        sys.modules["transformers"] = transformers_stub

    if "peft" not in sys.modules:
        peft_stub = types.ModuleType("peft")

        class _LoraConfig:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        peft_stub.LoraConfig = _LoraConfig
        peft_stub.PeftModel = type("PeftModel", (), {})
        peft_stub.TaskType = types.SimpleNamespace(CAUSAL_LM="CAUSAL_LM")
        peft_stub.get_peft_model = lambda model, config: model
        peft_stub.prepare_model_for_kbit_training = lambda model: model
        sys.modules["peft"] = peft_stub


_install_optional_training_stubs()

if "fitz" not in sys.modules:
    fitz_stub = types.ModuleType("fitz")

    class _EmptyDocument:
        def __enter__(self):
            return []

        def __exit__(self, exc_type, exc, tb):
            return False

    fitz_stub.open = lambda *args, **kwargs: _EmptyDocument()
    sys.modules["fitz"] = fitz_stub

if "paramiko" not in sys.modules:
    paramiko_stub = types.ModuleType("paramiko")

    class _SSHClient:
        def set_missing_host_key_policy(self, *args, **kwargs):
            return None

        def connect(self, *args, **kwargs):
            return None

        def close(self):
            return None

    paramiko_stub.SSHClient = _SSHClient
    paramiko_stub.AutoAddPolicy = lambda: object()
    paramiko_stub.PKey = object

    class _RSAKey:
        @staticmethod
        def from_private_key_file(*args, **kwargs):
            return object()

        @staticmethod
        def from_private_key(*args, **kwargs):
            return object()

    paramiko_stub.RSAKey = _RSAKey
    paramiko_stub.SSHException = RuntimeError
    sys.modules["paramiko"] = paramiko_stub
