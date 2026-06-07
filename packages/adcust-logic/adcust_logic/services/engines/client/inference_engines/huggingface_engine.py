import torch
import gc
from threading import Thread
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList
)
from adcust_logic.interfaces.engines.i_inference_engine import IInferenceEngine
from adcust_logic.models.adapter_asset import AdapterAsset
from typing import Optional, List, Dict


# --- 新增：自定义停止准则类 ---
class TokenStopCriteria(StoppingCriteria):
    def __init__(self, stop_ids: List[int]):
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 检查最后一个生成的 token 是否在停止列表中
        last_token_id = input_ids[0, -1].item()
        return last_token_id in self.stop_ids


class AbortStopCriteria(StoppingCriteria):
    def __init__(self, engine):
        self.engine = engine

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return self.engine.abort_generation


class HuggingfaceEngine(IInferenceEngine):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_base_path = None
        self.active_adapter: Optional[AdapterAsset] = None
        self.abort_generation = False

    def load_model(self, model_path: str) -> bool:
        if self.model is not None and self.current_base_path == model_path:
            return True

        if self.model is not None:
            self.unload_model()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=False,
            )
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                model_bytes = sum(parameter.numel() * parameter.element_size() for parameter in self.model.parameters())
                if model_bytes * 1.2 <= total_memory:
                    self.model.to("cuda")
            self.current_base_path = model_path
            return True
        except Exception as e:
            raise RuntimeError(f"模型点火失败: {str(e)}")

    def unload_model(self):
        self.request_stop()
        if self.model is not None:
            try:
                if not any(getattr(parameter, "is_meta", False) for parameter in self.model.parameters()):
                    self.model.cpu()
            except (NotImplementedError, RuntimeError):
                pass
            self.model = None
            self.tokenizer = None
            self.current_base_path = None
            self.active_adapter = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def request_stop(self):
        """请求当前流式生成尽快停止。"""
        self.abort_generation = True

    def switch_adapter(self, adapter: Optional[AdapterAsset]):
        if not self.model:
            raise ValueError("推理引擎未启动")

        if adapter is None:
            if isinstance(self.model, PeftModel):
                self.model.base_model.disable_adapter_layers()
            self.active_adapter = None
        else:
            adapter_id = adapter.name
            if not isinstance(self.model, PeftModel):
                self.model = PeftModel.from_pretrained(self.model, adapter.local_path, adapter_name=adapter_id)
            else:
                if adapter_id not in self.model.peft_config:
                    self.model.load_adapter(adapter.local_path, adapter_name=adapter_id)
                self.model.set_adapter(adapter_id)
            self.model.base_model.enable_adapter_layers()
            self.active_adapter = adapter

    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        cleaned_system_prompt = (system_prompt or "").strip()
        if cleaned_system_prompt:
            messages.append({"role": "system", "content": cleaned_system_prompt})

        for item in history or []:
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if role in {"system", "user", "assistant"} and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": prompt})
        return messages

    def _messages_to_plain_prompt(self, messages: List[Dict[str, str]]) -> str:
        lines = []
        for item in messages:
            lines.append(f"{item['role'].capitalize()}: {item['content']}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ):
        if not self.model:
            raise RuntimeError("模型未加载")
        self.abort_generation = False

        messages = self._build_messages(prompt, system_prompt=system_prompt, history=history)
        if getattr(self.tokenizer, "chat_template", None):
            try:
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True
                ).to(self.model.device)
            except TypeError:
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(self.model.device)
                inputs = {"input_ids": input_ids}
        else:
            inputs = self.tokenizer(self._messages_to_plain_prompt(messages), return_tensors="pt").to(self.model.device)
        # 本地 1.5B/3B 模型在低显存机器上可能主要跑 CPU，首 token 经常超过 10 秒。
        # 这里不设置短超时，避免把慢推理误判为网络中断。
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=None)

        # 1. 准备停止 ID：只使用 EOS。不要把换行当硬停止，Qwen/Gemma 常会先输出换行。
        stop_ids = [self.tokenizer.eos_token_id]

        stop_criteria = StoppingCriteriaList([TokenStopCriteria(list(set(stop_ids))), AbortStopCriteria(self)])

        # 2. 生成配置
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

        kwargs = dict(
            **inputs,
            streamer=streamer,
            generation_config=gen_config,
            stopping_criteria=stop_criteria  # 注入停止准则
        )

        generation_error = {"error": None}

        def run_generate():
            try:
                self.model.generate(**kwargs)
            except Exception as exc:
                generation_error["error"] = exc
                # 唤醒 streamer，避免前端一直卡在 assistant 占位符。
                streamer.on_finalized_text("", stream_end=True)

        thread = Thread(target=run_generate, daemon=True)
        thread.start()

        for new_text in streamer:
            if generation_error["error"]:
                raise RuntimeError(f"Generation failed: {generation_error['error']}")
            # 过滤掉输出中可能残留的停止标记字符串
            if new_text:
                # 如果模型输出了 eos 标记对应的文本，则直接切断输出
                if self.tokenizer.eos_token in new_text:
                    yield new_text.replace(self.tokenizer.eos_token, "").strip()
                    break
                yield new_text
