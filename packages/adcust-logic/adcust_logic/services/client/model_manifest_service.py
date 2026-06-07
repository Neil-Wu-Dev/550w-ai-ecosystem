# -*- coding: utf-8 -*-
import hashlib
import json
import os
import re
import subprocess
import configparser
from typing import Any, Dict, List


class ModelManifestService:
    """生成可验证的本地 HuggingFace 模型清单。

    除逐文件哈希外，也读取模型目录的 Git 来源和 commit。远端必须下载
    同一个仓库的同一个 commit，避免同名模型因版本更新而出现字节差异。
    """

    IMPORTANT_FILES = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "sentencepiece.bpe.model",
        "tokenizer.model",
    ]

    WEIGHT_EXTENSIONS = (".safetensors", ".bin", ".pt", ".pth")
    TEXT_EXTENSIONS = (".json", ".txt")

    def build_local_manifest(self, model_path: str, repository_hint: str = "") -> Dict[str, Any]:
        if not os.path.isdir(model_path):
            raise ValueError(f"Model path is not a directory: {model_path}")
        files = self._collect_files(model_path)
        provenance = self._read_git_provenance(model_path)
        if not provenance.get("revision"):
            provenance = self._read_huggingface_local_metadata(model_path, repository_hint)
        return {
            "manifest_version": "2.0",
            "source_type": "local_huggingface_directory",
            "local_path": os.path.abspath(model_path),
            "directory_name": os.path.basename(os.path.normpath(model_path)),
            **provenance,
            "files": files,
            "combined_hash": self._combined_hash(files),
        }

    def _read_huggingface_local_metadata(self, model_path: str, repository_hint: str) -> Dict[str, Any]:
        """读取 snapshot_download(local_dir=...) 生成的本地 revision 元数据。"""
        metadata_root = os.path.join(model_path, ".cache", "huggingface", "download")
        if not os.path.isdir(metadata_root):
            return {"repository_id": None, "revision": None}
        revisions = set()
        for root, _, names in os.walk(metadata_root):
            for name in names:
                if not name.endswith(".metadata"):
                    continue
                try:
                    with open(os.path.join(root, name), "r", encoding="utf-8") as file:
                        values = file.read().split()
                    if values and re.fullmatch(r"[0-9a-fA-F]{40,64}", values[0]):
                        revisions.add(values[0])
                except OSError:
                    continue
        if len(revisions) != 1 or not self._looks_like_repository_id(repository_hint):
            return {"repository_id": None, "revision": None}
        return {
            "repository_id": repository_hint.strip(),
            "revision": revisions.pop(),
            "remote_url": f"https://huggingface.co/{repository_hint.strip()}",
        }

    def _looks_like_repository_id(self, value: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", str(value or "").strip()))

    def _collect_files(self, model_path: str) -> List[Dict[str, Any]]:
        selected = set(self.IMPORTANT_FILES)
        for name in os.listdir(model_path):
            if name.endswith(self.WEIGHT_EXTENSIONS):
                selected.add(name)

        files = []
        for relative_path in sorted(selected):
            full_path = os.path.join(model_path, relative_path)
            if os.path.isfile(full_path):
                files.append(
                    {
                        "path": relative_path.replace("\\", "/"),
                        "size_bytes": os.path.getsize(full_path),
                        **self._content_identity(full_path),
                    }
                )
        return files

    def _read_git_provenance(self, model_path: str) -> Dict[str, Any]:
        """读取 HuggingFace Git 来源；无法确认时交由训练预检阻止任务。"""
        if not os.path.exists(os.path.join(model_path, ".git")):
            return {"repository_id": None, "revision": None}
        try:
            revision = self._run_git(model_path, "rev-parse", "HEAD")
            remote_url = self._run_git(model_path, "remote", "get-url", "origin")
            return {
                "repository_id": self._repository_id_from_url(remote_url),
                "revision": revision,
                "remote_url": remote_url,
            }
        except (OSError, subprocess.SubprocessError, ValueError):
            return self._read_git_files(model_path)

    def _read_git_files(self, model_path: str) -> Dict[str, Any]:
        """直接读取 .git，避免 safe.directory 等用户级 Git 配置影响模型识别。"""
        git_dir = os.path.join(model_path, ".git")
        try:
            parser = configparser.ConfigParser()
            parser.read(os.path.join(git_dir, "config"), encoding="utf-8")
            remote_url = parser['remote "origin"']["url"]
            with open(os.path.join(git_dir, "HEAD"), "r", encoding="utf-8") as file:
                head = file.read().strip()
            if head.startswith("ref: "):
                ref_name = head[5:]
                ref_path = os.path.join(git_dir, *ref_name.split("/"))
                if os.path.isfile(ref_path):
                    with open(ref_path, "r", encoding="utf-8") as file:
                        revision = file.read().strip()
                else:
                    revision = self._read_packed_ref(git_dir, ref_name)
            else:
                revision = head
            if not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
                raise ValueError("Invalid Git revision")
            return {
                "repository_id": self._repository_id_from_url(remote_url),
                "revision": revision,
                "remote_url": remote_url,
            }
        except (OSError, KeyError, configparser.Error, ValueError):
            return {"repository_id": None, "revision": None}

    def _read_packed_ref(self, git_dir: str, ref_name: str) -> str:
        with open(os.path.join(git_dir, "packed-refs"), "r", encoding="utf-8") as file:
            for line in file:
                value = line.strip()
                if not value or value.startswith(("#", "^")):
                    continue
                revision, name = value.split(" ", 1)
                if name == ref_name:
                    return revision
        raise ValueError(f"Git ref not found: {ref_name}")

    def _run_git(self, model_path: str, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", model_path, *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        value = result.stdout.strip()
        if not value:
            raise ValueError("Git metadata is empty")
        return value

    def _repository_id_from_url(self, remote_url: str) -> str:
        normalized = remote_url.strip().replace("\\", "/")
        match = re.search(
            r"(?:huggingface\.co[:/])([^/]+/[^/]+?)(?:\.git)?$",
            normalized,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError(f"Unsupported HuggingFace remote URL: {remote_url}")
        return match.group(1)

    def _content_identity(self, path: str) -> Dict[str, Any]:
        """文本统一为 LF 后计算哈希；模型权重始终使用原始二进制哈希。"""
        if path.lower().endswith(self.TEXT_EXTENSIONS):
            with open(path, "rb") as file:
                content = file.read().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            return {
                "sha256": hashlib.sha256(content).hexdigest(),
                "hash_mode": "text_lf_sha256",
                "canonical_size_bytes": len(content),
            }

        digest = hashlib.sha256()
        size = 0
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
        return {
            "sha256": digest.hexdigest(),
            "hash_mode": "binary_sha256",
            "canonical_size_bytes": size,
        }

    def _combined_hash(self, files: List[Dict[str, Any]]) -> str:
        identity = [
            {
                "path": item["path"],
                "sha256": item["sha256"],
                "hash_mode": item.get("hash_mode", "binary_sha256"),
                "canonical_size_bytes": item.get("canonical_size_bytes", item.get("size_bytes")),
            }
            for item in files
        ]
        payload = json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
