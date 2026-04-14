import os
import time
import torch
import logging
from typing import Generator, List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from app.models.model_asset import ModelAsset
from app.models.dataset_asset import DatasetAsset
from app.core.training_engines.interfaces.i_training_engine import ITrainingEngine

logger = logging.getLogger(__name__)


class SequentialEngine(ITrainingEngine):
    def __init__(self):
        self._is_aborting = False  # 用于控制外部终止

    def abort(self):
        """外部调用此方法可触发训练停止"""
        self._is_aborting = True

    def train(
            self,
            base_model: ModelAsset,
            dataset: DatasetAsset,
            save_path: str,
            target_modules: List[str],
            epochs: int = 1  # <--- 新增参数：由前端传入，默认为1
    ) -> Generator[Dict[str, Any], None, None]:

        self._is_aborting = False
        # 简单校验：确保至少训练一轮
        safe_epochs = max(1, int(epochs))

        yield {"status": "start", "message": f"正在初始化算力引擎 (预设轮数: {safe_epochs})...", "percentage": 0}

        try:
            # 1. 加载模型和分词器 (真实 PyTorch 加载)
            yield {"status": "running", "message": f"正在加载底座模型: {base_model.name}", "percentage": 10}
            tokenizer = AutoTokenizer.from_pretrained(base_model.local_path)
            tokenizer.pad_token = tokenizer.eos_token  # 防止 pad 报错

            model = AutoModelForCausalLM.from_pretrained(
                base_model.local_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )

            # 2. 配置 LoRA (适配器技术)
            yield {"status": "running", "message": "正在配置 LoRA 适配层...", "percentage": 20}
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=target_modules  # 严格执行前端传来的层
            )
            model = get_peft_model(model, peft_config)

            # 3. 准备数据并计算总步数
            chunks_count = len(dataset.chunks)
            total_global_steps = chunks_count * safe_epochs
            yield {"status": "running",
                   "message": f"准备就绪，共计 {chunks_count} 个数据块，总计训练 {total_global_steps} 步",
                   "percentage": 30}

            # 4. 核心训练循环
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

            current_global_step = 0

            # --- 开始 Epochs 大循环 ---
            for epoch_idx in range(safe_epochs):
                for i, chunk in enumerate(dataset.chunks):
                    if self._is_aborting:
                        # 计算中止时的进度
                        abort_percent = 30 + (current_global_step / total_global_steps * 60)
                        yield {"status": "aborted", "message": f"训练在第 {epoch_idx + 1} 轮被手动终止",
                               "percentage": round(abort_percent, 2)}
                        return

                    # 构建真实输入
                    full_text = f"Instruction: {chunk['instruction']}\nInput: {chunk['input']}\nResponse: {chunk['output']}"
                    inputs = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True,
                                       max_length=512).to(
                        model.device)

                    # 前向传播与反向传播
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    current_global_step += 1

                    # 动态进度计算 (30% -> 90%)
                    progress_step = (current_global_step / total_global_steps) * 60
                    current_percent = 30 + progress_step

                    yield {
                        "status": "running",
                        "epoch": epoch_idx + 1,
                        "step": i + 1,
                        "total_steps": total_global_steps,
                        "message": f"第 {epoch_idx + 1}/{safe_epochs} 轮 | 进度 [{i + 1}/{chunks_count}] | Loss: {loss.item():.4f}",
                        "percentage": round(current_percent, 2)
                    }
            # --- 循环结束 ---

            # 5. 保存结果
            yield {"status": "running", "message": f"正在保存适配器至: {save_path}", "percentage": 95}

            # 确保目录存在
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

            yield {"status": "completed", "message": f"训练成功！已完成 {safe_epochs} 轮训练并生成适配器。",
                   "percentage": 100}

        except Exception as e:
            logger.error(f"引擎内部异常: {str(e)}")
            yield {"status": "error", "message": f"算力引擎崩溃: {str(e)}", "percentage": 0}