# -*- coding: utf-8 -*-
"""远端训练入口脚本。

本文件会被 AdCust 上传到任意 Linux + NVIDIA GPU 主机执行，因此不能依赖本项目包路径。
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import time


EVENT_PREFIX = "ADCUST_EVENT "


def emit_event(event_type, **payload):
    print(
        EVENT_PREFIX + json.dumps({"adcust_event": event_type, **payload}, ensure_ascii=False),
        flush=True,
    )


def load_documents(dataset_path):
    """读取原始知识文档，不把故事伪装成问答或指令样本。"""
    records = []
    ext = os.path.splitext(dataset_path)[1].lower()
    with open(dataset_path, "r", encoding="utf-8") as f:
        if ext == ".jsonl":
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        elif ext == ".json":
            data = json.load(f)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = data.get("documents", data.get("records", data.get("data", [data])))
        else:
            records = [f.read()]

    documents = []
    for record in records:
        text = record_to_document(record)
        if text:
            documents.append(text)
    return documents


def record_to_document(record):
    """兼容纯文本和常见 JSON 字段，并优先保留真正的知识正文。"""
    if isinstance(record, str):
        return record.strip()
    if not isinstance(record, dict):
        return ""

    for key in ("text", "content", "document"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    instruction = str(record.get("instruction") or record.get("prompt") or "").strip()
    input_text = str(record.get("input") or record.get("context") or "").strip()
    output = str(record.get("output") or record.get("response") or record.get("completion") or "").strip()
    if instruction.startswith("Learn the following extracted PDF text") and output:
        return output
    return "\n\n".join(part for part in (instruction, input_text, output) if part)


def build_token_chunks(documents, tokenizer, max_seq_length):
    """把文档切成有重叠的 token 窗口，让原始故事直接参与 next-token 学习。"""
    chunk_size = min(int(max_seq_length), 256)
    overlap = min(64, max(16, chunk_size // 4))
    stride = max(1, chunk_size - overlap)
    eos_token_id = tokenizer.eos_token_id
    chunks = []
    total_tokens = 0

    for document in documents:
        token_ids = tokenizer.encode(document, add_special_tokens=False)
        if eos_token_id is not None:
            token_ids.append(eos_token_id)
        total_tokens += len(token_ids)
        for start in range(0, len(token_ids), stride):
            chunk = token_ids[start:start + chunk_size]
            if not chunk:
                continue
            if start > 0 and len(chunk) < min(32, chunk_size // 4):
                break
            chunks.append(chunk)
            if start + chunk_size >= len(token_ids):
                break

    return chunks, {
        "document_count": len(documents),
        "sample_count": len(chunks),
        "token_count": total_tokens,
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
    }


def extract_json_array(text):
    """从模型输出中提取 JSON 数组，兼容 Markdown 代码块和前后说明文字。"""
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        value = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def extract_qa_pairs(text):
    """兼容 JSON、Markdown 和常见 Q/A 文本格式。"""
    cleaned = str(text or "").strip()
    pairs = []

    json_items = extract_json_array(cleaned)
    if not json_items:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(cleaned[start:end + 1])
                if isinstance(value, dict):
                    json_items = value.get("qa_pairs") or value.get("items") or [value]
            except json.JSONDecodeError:
                json_items = []

    for item in json_items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or item.get("q") or item.get("问") or "").strip()
        answer = str(item.get("answer") or item.get("a") or item.get("答") or "").strip()
        if question and answer:
            pairs.append({"question": question, "answer": answer})

    qa_pattern = re.compile(
        r"(?:^|\n)\s*(?:Q(?:uestion)?|问(?:题)?)\s*[:：]\s*(.+?)"
        r"\s*(?:\n|\r\n?)\s*(?:A(?:nswer)?|答(?:案)?)\s*[:：]\s*(.+?)"
        r"(?=(?:\n\s*(?:Q(?:uestion)?|问(?:题)?)\s*[:：])|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    for match in qa_pattern.finditer(cleaned):
        question = match.group(1).strip()
        answer = match.group(2).strip()
        if question and answer:
            pairs.append({"question": question, "answer": answer})

    return deduplicate_qa_pairs(pairs)


def build_fallback_qa_pairs(source_text):
    """当底座模型未输出可解析问答时，从故事句子确定性构造事实监督。"""
    source = re.sub(r"\s+", " ", str(source_text or "")).strip()
    if not source:
        return []
    sentences = [
        sentence.strip(" \t\r\n-")
        for sentence in re.split(r"(?<=[。！？!?；;])|(?<=[.!?])\s+|\n+", source)
        if len(sentence.strip()) >= 6
    ]
    if not sentences:
        sentences = [source]

    pairs = []
    for sentence in sentences[:12]:
        has_cjk = bool(re.search(r"[\u3400-\u9fff]", sentence))
        if has_cjk:
            topic = re.split(r"[，。！？；：]", sentence, maxsplit=1)[0].strip()[:18]
            question = f"故事中关于“{topic}”发生了什么？"
        else:
            words = sentence.split()
            topic = " ".join(words[: min(8, len(words))]).strip(" ,.;:!?")
            question = f'What does the story say about "{topic}"?'
        if topic:
            pairs.append({"question": question, "answer": sentence})

    if bool(re.search(r"[\u3400-\u9fff]", source)):
        pairs.append({
            "question": "请准确概括这段故事中的人物、事件与关系。",
            "answer": source,
        })
    else:
        pairs.append({
            "question": "What are the important people, events, and relationships in this story?",
            "answer": source,
        })
    return deduplicate_qa_pairs(pairs)


def deduplicate_qa_pairs(pairs):
    unique = []
    seen = set()
    for pair in pairs:
        question = str(pair.get("question") or "").strip()
        answer = str(pair.get("answer") or "").strip()
        key = (question.casefold(), answer.casefold())
        if question and answer and key not in seen:
            seen.add(key)
            unique.append({"question": question, "answer": answer})
    return unique


def normalize_token_ids(value):
    """把 BatchEncoding、Tensor、嵌套列表统一转换为一维 token ID 列表。"""
    if isinstance(value, dict) or hasattr(value, "keys"):
        try:
            value = value["input_ids"]
        except (KeyError, TypeError):
            return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = list(value[0])
    if not isinstance(value, list):
        return []
    return [int(token_id) for token_id in value if isinstance(token_id, (int, float))]


def build_qa_training_example(tokenizer, question, answer, max_seq_length):
    """构造与本地聊天模板一致的监督样本，只对 assistant 答案计算 loss。"""
    system_message = (
        "You are a helpful assistant. Answer factual questions using the knowledge "
        "learned during adapter training."
    )
    prompt_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": question},
    ]
    full_messages = [
        *prompt_messages,
        {"role": "assistant", "content": answer},
    ]
    prompt_ids = []
    full_ids = []
    if getattr(tokenizer, "chat_template", None):
        try:
            prompt_ids = normalize_token_ids(
                tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
            full_ids = normalize_token_ids(
                tokenizer.apply_chat_template(
                    full_messages,
                    tokenize=True,
                    add_generation_prompt=False,
                )
            )
        except Exception:
            prompt_ids = []
            full_ids = []
    if not prompt_ids or not full_ids:
        prompt_text = f"System: {system_message}\nUser: {question}\nAssistant:"
        full_text = f"{prompt_text} {answer}"
        prompt_ids = normalize_token_ids(tokenizer.encode(prompt_text, add_special_tokens=False))
        full_ids = normalize_token_ids(tokenizer.encode(full_text, add_special_tokens=False))

    if not prompt_ids or not full_ids:
        return None
    removed_prefix = max(0, len(full_ids) - int(max_seq_length))
    full_ids = full_ids[removed_prefix:]
    supervised_start = max(0, min(len(full_ids), len(prompt_ids) - removed_prefix))
    labels = ([-100] * supervised_start) + full_ids[supervised_start:]
    if not any(label != -100 for label in labels):
        return None
    return {"input_ids": full_ids, "labels": labels}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_identity(path):
    """与本地清单使用相同规则：文本统一 LF，权重保持严格二进制哈希。"""
    if path.lower().endswith((".json", ".txt")):
        with open(path, "rb") as file:
            content = file.read().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        return {
            "sha256": hashlib.sha256(content).hexdigest(),
            "hash_mode": "text_lf_sha256",
            "canonical_size_bytes": len(content),
        }
    return {
        "sha256": sha256_file(path),
        "hash_mode": "binary_sha256",
        "canonical_size_bytes": os.path.getsize(path),
    }


def build_directory_manifest(model_path, expected_files=None):
    files = []
    if os.path.isdir(model_path):
        names = {item["path"] for item in (expected_files or [])}
        if not names:
            names = {
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
                "sentencepiece.bpe.model",
                "tokenizer.model",
            }
            for name in os.listdir(model_path):
                if name.endswith((".safetensors", ".bin", ".pt", ".pth")):
                    names.add(name)
        for name in sorted(names):
            full_path = os.path.join(model_path, name)
            if os.path.isfile(full_path):
                files.append({
                    "path": name.replace("\\", "/"),
                    "size_bytes": os.path.getsize(full_path),
                    **content_identity(full_path),
                })
    identity = [
        {
            "path": item["path"],
            "sha256": item["sha256"],
            "hash_mode": item.get("hash_mode", "binary_sha256"),
            "canonical_size_bytes": item.get("canonical_size_bytes", item.get("size_bytes")),
        }
        for item in files
    ]
    combined_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "manifest_version": "2.0",
        "source_type": "remote_huggingface_model",
        "model_name_or_path": model_path,
        "files": files,
        "combined_hash": combined_hash,
    }


def assert_manifest_matches(expected, actual):
    if expected.get("combined_hash") != actual.get("combined_hash"):
        expected_by_path = {item["path"]: item for item in expected.get("files", [])}
        actual_by_path = {item["path"]: item for item in actual.get("files", [])}
        mismatched = []
        for path, expected_file in expected_by_path.items():
            actual_file = actual_by_path.get(path)
            if not actual_file:
                mismatched.append({"path": path, "reason": "missing"})
            elif actual_file.get("sha256") != expected_file.get("sha256"):
                mismatched.append({"path": path, "reason": "sha256_mismatch"})
        raise RuntimeError(
            "Base model manifest mismatch after downloading the exact HuggingFace revision. "
            f"Mismatched files: {json.dumps(mismatched[:20], ensure_ascii=False)}"
        )


def dir_size_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            full_path = os.path.join(root, name)
            if os.path.isfile(full_path):
                total += os.path.getsize(full_path)
    return total


def emit_model_file_inventory(expected_manifest, status):
    for item in expected_manifest.get("files", []):
        emit_event(
            "model_file_status",
            message=f"{item['path']}: {status}",
            resource={
                "id": f"model:{item['path']}",
                "kind": "model_file",
                "name": item["path"],
                "status": status,
                "size_bytes": item.get("size_bytes"),
            },
        )


def verify_model_files(model_path, expected_manifest):
    actual = build_directory_manifest(model_path, expected_manifest.get("files", []))
    expected_by_path = {item["path"]: item for item in expected_manifest.get("files", [])}
    actual_by_path = {item["path"]: item for item in actual.get("files", [])}
    for path, expected in expected_by_path.items():
        current = actual_by_path.get(path)
        status = "ready" if current and current.get("sha256") == expected.get("sha256") else "mismatch"
        emit_event(
            "model_file_status",
            message=f"{path}: {status}",
            resource={
                "id": f"model:{path}",
                "kind": "model_file",
                "name": path,
                "status": status,
                "size_bytes": expected.get("size_bytes"),
            },
        )
    assert_manifest_matches(expected_manifest, actual)
    return actual


def prepare_model_path(cfg):
    source = cfg["base_model_name_or_path"]
    revision = cfg.get("base_model_revision")
    expected_manifest = cfg.get("expected_base_model_manifest") or {}
    if os.path.isdir(source):
        verify_model_files(source, expected_manifest)
        emit_event("model_ready", message=f"Remote base model already exists: {source}", model_path=source)
        return source

    remote_model_dir = cfg.get("remote_model_dir")
    if not remote_model_dir:
        raise RuntimeError("remote_model_dir is required when base_model_name_or_path is a HuggingFace model ID")
    if not revision:
        raise RuntimeError("base_model_revision is required for strict local/remote model binding")

    marker_path = os.path.join(remote_model_dir, ".adcust_download_complete")
    if os.path.exists(marker_path):
        try:
            with open(marker_path, "r", encoding="utf-8") as file:
                marker = json.load(file)
            if marker.get("repository_id") == source and marker.get("revision") == revision:
                verify_model_files(remote_model_dir, expected_manifest)
                emit_event(
                    "model_ready",
                    message=f"Verified remote model cache: {source}@{revision[:12]}",
                    model_path=remote_model_dir,
                    size_bytes=dir_size_bytes(remote_model_dir),
                    revision=revision,
                )
                return remote_model_dir
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            emit_event("model_cache_invalid", message=f"Cached model is invalid and will be replaced: {exc}")
        shutil.rmtree(remote_model_dir, ignore_errors=True)

    emit_model_file_inventory(expected_manifest, "pending_download")
    emit_event(
        "model_download",
        message=f"Downloading exact HuggingFace revision: {source}@{revision}",
        model_path=remote_model_dir,
        source=source,
        revision=revision,
    )
    from huggingface_hub import hf_hub_download, snapshot_download
    from tqdm.auto import tqdm

    os.makedirs(remote_model_dir, exist_ok=True)
    expected_files = expected_manifest.get("files", [])
    total_expected_bytes = sum(int(item.get("size_bytes") or 0) for item in expected_files)
    completed_bytes = 0
    for item in expected_files:
        file_name = item["path"]
        file_size = int(item.get("size_bytes") or 0)

        class AdcustFileProgress(tqdm):
            """把当前文件计数换算成全部关键模型文件的真实总进度。"""

            def display(self, msg=None, pos=None):
                super().display(msg=msg, pos=pos)
                now = time.time()
                last_emit = getattr(self, "_adcust_last_emit", 0.0)
                if now - last_emit < 0.5 and self.n < (self.total or 0):
                    return
                self._adcust_last_emit = now
                current_file_bytes = min(file_size, int(self.n or 0))
                aggregate_bytes = min(total_expected_bytes, completed_bytes + current_file_bytes)
                percentage = round((aggregate_bytes / total_expected_bytes) * 100, 2) if total_expected_bytes else 100.0
                emit_event(
                    "model_download_progress",
                    message=f"{file_name}: {aggregate_bytes}/{total_expected_bytes} bytes",
                    download_progress=percentage,
                    downloaded_bytes=aggregate_bytes,
                    total_bytes=total_expected_bytes,
                    resource={
                        "id": f"model:{file_name}",
                        "kind": "model_file",
                        "name": file_name,
                        "status": "downloading",
                        "size_bytes": file_size,
                        "download_progress": round((current_file_bytes / file_size) * 100, 2) if file_size else 100.0,
                        "downloaded_bytes": current_file_bytes,
                        "total_bytes": file_size,
                    },
                )

        hf_hub_download(
            repo_id=source,
            filename=file_name,
            revision=revision,
            local_dir=remote_model_dir,
            tqdm_class=AdcustFileProgress,
        )
        completed_bytes += file_size
        emit_event(
            "model_file_status",
            message=f"{file_name}: ready",
            resource={
                "id": f"model:{file_name}",
                "kind": "model_file",
                "name": file_name,
                "status": "ready",
                "size_bytes": file_size,
                "download_progress": 100.0,
                "downloaded_bytes": file_size,
                "total_bytes": file_size,
            },
        )

    emit_event("model_snapshot_finalize", message="Downloading remaining repository metadata and model support files")
    snapshot_download(
        repo_id=source,
        revision=revision,
        local_dir=remote_model_dir,
    )
    try:
        remote_manifest = verify_model_files(remote_model_dir, expected_manifest)
    except RuntimeError as exc:
        if cfg.get("_model_download_retry"):
            raise
        emit_event(
            "model_cache_invalid",
            message=f"Downloaded model verification failed. Clearing only this model cache and retrying once: {exc}",
        )
        shutil.rmtree(remote_model_dir, ignore_errors=True)
        retry_cfg = dict(cfg)
        retry_cfg["_model_download_retry"] = True
        return prepare_model_path(retry_cfg)
    with open(marker_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "repository_id": source,
                "revision": revision,
                "combined_hash": remote_manifest.get("combined_hash"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    emit_event(
        "model_ready",
        message=f"Remote model downloaded and verified: {source}@{revision[:12]}",
        model_path=remote_model_dir,
        size_bytes=dir_size_bytes(remote_model_dir),
        revision=revision,
    )
    return remote_model_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    emit_event("config_loaded", message="Remote training config loaded")

    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig, TrainerCallback
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except Exception as exc:
        raise RuntimeError(f"Remote training dependencies are missing. Install torch/transformers/peft/bitsandbytes: {exc}")

    class KnowledgeTrainingDataset(Dataset):
        """混合原文语言建模与自动知识问答监督，padding 不计入 loss。"""

        def __init__(self, examples, tokenizer, sequence_length):
            self.items = []
            pad_token_id = tokenizer.pad_token_id
            for example in examples:
                token_ids = list(example["input_ids"])
                raw_labels = list(example.get("labels", token_ids))
                padding_length = max(0, sequence_length - len(token_ids))
                input_ids = token_ids + ([pad_token_id] * padding_length)
                attention_mask = ([1] * len(token_ids)) + ([0] * padding_length)
                labels = raw_labels + ([-100] * padding_length)
                self.items.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                })

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    documents = load_documents(cfg["dataset_path"])
    if not documents:
        raise RuntimeError("Dataset is empty; training cannot start")

    model_path = prepare_model_path(cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_chunks, dataset_stats = build_token_chunks(documents, tokenizer, int(cfg["max_seq_length"]))
    if not token_chunks:
        raise RuntimeError("Dataset contains no trainable text tokens")
    emit_event(
        "dataset_ready",
        message=(
            f"Knowledge dataset ready: {dataset_stats['document_count']} documents, "
            f"{dataset_stats['sample_count']} token windows, {dataset_stats['token_count']} tokens"
        ),
        training_mode="continued_pretraining",
        **dataset_stats,
    )

    quant_config = None
    if cfg.get("use_qlora", True):
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quant_config,
    )

    remote_manifest = build_directory_manifest(model_path, cfg.get("expected_base_model_manifest", {}).get("files", []))
    expected_manifest = cfg.get("expected_base_model_manifest")
    if expected_manifest:
        assert_manifest_matches(expected_manifest, remote_manifest)
        emit_event("model_verified", message="Base model manifest verified", model_path=model_path)

    emit_event(
        "knowledge_synthesis_started",
        message="Generating factual supervision from the uploaded raw knowledge text",
        source_window_count=len(token_chunks),
    )
    qa_pairs = []
    generated_qa_count = 0
    fallback_qa_count = 0
    model.eval()
    for index, token_chunk in enumerate(token_chunks, start=1):
        source_text = tokenizer.decode(token_chunk, skip_special_tokens=True).strip()
        if not source_text:
            continue
        synthesis_messages = [
            {
                "role": "system",
                "content": (
                    "Create factual training supervision from source text. Return only a JSON array. "
                    "Each item must have question and answer strings. Questions must be self-contained, "
                    "must not mention 'the passage', and answers must be supported exactly by the source. "
                    "Preserve the source language. Cover names, events, relationships, causes, locations, "
                    "and chronology when present."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Generate several diverse factual question-answer items from this source:\n\n"
                    + source_text
                ),
            },
        ]
        generated_pairs = []
        generation_error = None
        try:
            if getattr(tokenizer, "chat_template", None):
                try:
                    synthesis_inputs = tokenizer.apply_chat_template(
                        synthesis_messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                    )
                except TypeError:
                    synthesis_input_ids = tokenizer.apply_chat_template(
                        synthesis_messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                    synthesis_inputs = {
                        "input_ids": synthesis_input_ids,
                        "attention_mask": torch.ones_like(synthesis_input_ids),
                    }
            else:
                rendered = (
                    synthesis_messages[0]["content"]
                    + "\n\n"
                    + synthesis_messages[1]["content"]
                    + "\n\nJSON:"
                )
                synthesis_inputs = tokenizer(rendered, return_tensors="pt")
            synthesis_inputs = {
                key: value.to(model.device)
                for key, value in synthesis_inputs.items()
            }
            with torch.no_grad():
                generated = model.generate(
                    **synthesis_inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated_only = generated[0, synthesis_inputs["input_ids"].shape[-1]:]
            generated_text = tokenizer.decode(generated_only, skip_special_tokens=True)
            generated_pairs = [
                pair
                for pair in extract_qa_pairs(generated_text)
                if re.sub(r"\s+", " ", pair["answer"]).casefold()
                in re.sub(r"\s+", " ", source_text).casefold()
            ]
        except Exception as exc:
            generation_error = str(exc)
        fallback_pairs = build_fallback_qa_pairs(source_text)
        generated_qa_count += len(generated_pairs)
        fallback_qa_count += len(fallback_pairs)
        qa_pairs.extend(generated_pairs)
        qa_pairs.extend(fallback_pairs)
        qa_pairs = deduplicate_qa_pairs(qa_pairs)
        emit_event(
            "knowledge_synthesis_progress",
            message=(
                f"Knowledge supervision ready for source window {index}/{len(token_chunks)}: "
                f"{len(generated_pairs)} model-generated and {len(fallback_pairs)} deterministic samples"
            ),
            completed_windows=index,
            total_windows=len(token_chunks),
            qa_pair_count=len(qa_pairs),
            generated_qa_count=generated_qa_count,
            fallback_qa_count=fallback_qa_count,
            generation_error=generation_error,
        )

    raw_examples = [
        {"input_ids": token_ids, "labels": list(token_ids)}
        for token_ids in token_chunks
    ]
    qa_examples = []
    for pair in qa_pairs:
        example = build_qa_training_example(
            tokenizer,
            pair["question"],
            pair["answer"],
            int(cfg["max_seq_length"]),
        )
        if example:
            qa_examples.append(example)
    if not qa_examples:
        emit_event(
            "knowledge_synthesis_fallback",
            message=(
                "Structured QA tokenization produced no samples; continuing with raw knowledge "
                "language-modeling samples instead of aborting the remote job"
            ),
            raw_sample_count=len(raw_examples),
        )
    training_examples = raw_examples + qa_examples
    dataset_stats["qa_pair_count"] = len(qa_examples)
    dataset_stats["generated_qa_count"] = generated_qa_count
    dataset_stats["fallback_qa_count"] = fallback_qa_count
    dataset_stats["training_sample_count"] = len(training_examples)
    emit_event(
        "knowledge_synthesis_ready",
        message=(
            f"Mixed knowledge training set ready: {len(raw_examples)} raw text windows and "
            f"{len(qa_examples)} automatically generated factual QA samples"
        ),
        training_mode="knowledge_internalization",
        **dataset_stats,
    )

    if cfg.get("use_qlora", True):
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=int(cfg["lora_rank"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=cfg.get("target_modules") or None,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.config.use_cache = False
    model.print_trainable_parameters()
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    initial_trainable_parameters = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if trainable_parameter_count <= 0:
        raise RuntimeError("LoRA configuration produced zero trainable parameters")
    emit_event(
        "adapter_trainability_verified",
        message=f"LoRA adapter exposes {trainable_parameter_count} trainable parameters",
        trainable_parameter_count=trainable_parameter_count,
    )

    train_dataset = KnowledgeTrainingDataset(
        training_examples,
        tokenizer,
        int(cfg["max_seq_length"]),
    )
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=int(cfg["epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        logging_nan_inf_filter=False,
        save_strategy="steps",
        save_steps=int(cfg.get("checkpoint_save_steps", 25)),
        save_total_limit=int(cfg.get("checkpoint_save_total_limit", 3)),
        report_to="none",
        disable_tqdm=True,
        fp16=torch.cuda.is_available(),
    )

    total_steps = max(1, len(train_dataset) * int(cfg["epochs"]))

    class AdcustProgressCallback(TrainerCallback):
        def __init__(self):
            self.started_at = time.time()
            self.emitted_steps = set()

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" not in logs:
                return
            current_step = max(1, int(state.global_step or 1))
            self.emitted_steps.add(current_step)
            max_steps = max(1, int(state.max_steps or total_steps))
            train_progress = min(100.0, (current_step / max_steps) * 100)
            elapsed = time.time() - self.started_at
            eta = max(0.0, (elapsed / (train_progress / 100.0)) - elapsed) if train_progress > 0 else None
            emit_event(
                "train_metric",
                message=f"step={current_step} loss={float(logs['loss']):.6f}",
                step=current_step,
                total_steps=max_steps,
                loss=float(logs["loss"]),
                train_progress=round(train_progress, 2),
                eta_seconds=round(eta, 2) if eta is not None else None,
            )

    progress_callback = AdcustProgressCallback()
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, callbacks=[progress_callback])
    emit_event(
        "training_started",
        message="LoRA/QLoRA mixed knowledge internalization training started",
        training_mode="knowledge_internalization",
        total_steps=total_steps,
        raw_sample_count=len(raw_examples),
        qa_sample_count=len(qa_examples),
    )
    train_result = trainer.train(resume_from_checkpoint=bool(cfg.get("resume_from_checkpoint", False)))
    final_loss = train_result.metrics.get("train_loss")
    if final_loss is None or not math.isfinite(float(final_loss)):
        raise RuntimeError(f"Training did not produce a finite train loss: {final_loss}")
    adapter_update_l1 = sum(
        float(
            (
                parameter.detach().float().cpu()
                - initial_trainable_parameters[name]
            ).abs().sum()
        )
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if adapter_update_l1 <= 0:
        raise RuntimeError("Training completed without changing any trainable adapter parameter")
    if final_loss is not None and int(trainer.state.global_step or 0) not in progress_callback.emitted_steps:
        emit_event(
            "train_metric",
            message=f"final loss={float(final_loss):.6f}",
            step=int(trainer.state.global_step or total_steps),
            total_steps=max(1, int(trainer.state.max_steps or total_steps)),
            loss=float(final_loss),
            train_progress=100.0,
            eta_seconds=0,
        )
    emit_event(
        "training_summary",
        message=(
            f"Training executed {int(trainer.state.global_step or 0)} optimizer steps; "
            f"final train loss={float(final_loss):.6f}"
            if final_loss is not None
            else f"Training executed {int(trainer.state.global_step or 0)} optimizer steps"
        ),
        actual_steps=int(trainer.state.global_step or 0),
        total_steps=max(1, int(trainer.state.max_steps or total_steps)),
        final_loss=float(final_loss) if final_loss is not None else None,
        raw_sample_count=len(raw_examples),
        qa_sample_count=len(qa_examples),
        trainable_parameter_count=trainable_parameter_count,
        adapter_update_l1=adapter_update_l1,
    )
    emit_event("saving_adapter", message="Saving adapter artifacts")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    signature = {
        "signature_version": "1.0",
        "adapter_name": cfg.get("adapter_name"),
        "base_model_name_or_path": cfg["base_model_name_or_path"],
        "remote_model_path": model_path,
        "model_format": "huggingface_transformers",
        "adapter_format": "peft_lora",
        "training_mode": "knowledge_internalization",
        "quantization": "qlora_4bit" if cfg.get("use_qlora") else "none",
        "expected_base_model_manifest": expected_manifest,
        "remote_base_model_manifest": remote_manifest,
        "base_model_manifest_hash": expected_manifest.get("combined_hash") if expected_manifest else remote_manifest.get("combined_hash"),
        "training_params": {
            "epochs": cfg.get("epochs"),
            "learning_rate": cfg.get("learning_rate"),
            "max_seq_length": cfg.get("max_seq_length"),
            "lora_rank": cfg.get("lora_rank"),
            "lora_alpha": cfg.get("lora_alpha"),
            "lora_dropout": cfg.get("lora_dropout"),
            "target_modules": cfg.get("target_modules"),
            "use_qlora": cfg.get("use_qlora"),
            "dataset_stats": dataset_stats,
            "actual_steps": int(trainer.state.global_step or 0),
            "final_train_loss": float(final_loss) if final_loss is not None else None,
            "trainable_parameter_count": trainable_parameter_count,
            "adapter_update_l1": adapter_update_l1,
        },
    }
    with open(os.path.join(cfg["output_dir"], "adcust_signature.json"), "w", encoding="utf-8") as f:
        json.dump(signature, f, ensure_ascii=False, indent=2)
    emit_event("training_completed", message="Training completed")


if __name__ == "__main__":
    main()
