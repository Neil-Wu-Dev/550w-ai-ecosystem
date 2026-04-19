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
from typing import Optional, List


# --- 新增：自定义停止准则类 ---
class TokenStopCriteria(StoppingCriteria):
    def __init__(self, stop_ids: List[int]):
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 检查最后一个生成的 token 是否在停止列表中
        last_token_id = input_ids[0, -1].item()
        return last_token_id in self.stop_ids


class HuggingfaceEngine(IInferenceEngine):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_base_path = None
        self.active_adapter: Optional[AdapterAsset] = None

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
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.current_base_path = model_path
            return True
        except Exception as e:
            raise RuntimeError(f"模型点火失败: {str(e)}")

    def unload_model(self):
        if self.model is not None:
            self.model.cpu()
            self.model = None
            self.tokenizer = None
            self.current_base_path = None
            self.active_adapter = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

    def generate_stream(self, prompt: str, max_new_tokens: int = 512):
        if not self.model:
            raise RuntimeError("模型未加载")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=10.0)

        # 1. 准备停止 ID：包含 EOS 以及常见的换行/对话分隔符
        stop_ids = [self.tokenizer.eos_token_id]

        # 将 "\n" 和 "User:" 也作为硬停止条件（针对 GPT-2 续写控制）
        # 注意：某些 tokenizer 会将 "\n" 编码为多个 token，这里取最常见的情况
        extra_stops = ["\n", "User:", "Assistant:"]
        for word in extra_stops:
            ids = self.tokenizer.encode(word, add_special_tokens=False)
            if ids:
                stop_ids.append(ids[-1])  # 取最后一个 token id

        stop_criteria = StoppingCriteriaList([TokenStopCriteria(list(set(stop_ids)))])

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

        thread = Thread(target=self.model.generate, kwargs=kwargs, daemon=True)
        thread.start()

        for new_text in streamer:
            # 过滤掉输出中可能残留的停止标记字符串
            if new_text:
                # 如果模型输出了 eos 标记对应的文本，则直接切断输出
                if self.tokenizer.eos_token in new_text:
                    yield new_text.replace(self.tokenizer.eos_token, "").strip()
                    break
                yield new_text