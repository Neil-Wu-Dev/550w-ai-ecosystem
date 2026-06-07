#!/usr/bin/env python3
"""Git 大文件安全门。

规则：
1. 任意新增或修改后的单个文件不得超过 10 MiB。
2. 单次提交中所有新增或修改文件的完整内容总量不得超过 50 MiB。
3. pre-commit 检查当前暂存区，pre-push 复检即将推送的每个提交。

这里采用“拒绝提交/推送”而不是截断文件。自动截断会破坏模型、数据库、
压缩包等二进制文件，也可能让代码仓库出现无法察觉的数据损坏。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


MIB = 1024 * 1024
MAX_FILE_BYTES = 10 * MIB
MAX_COMMIT_BYTES = 50 * MIB
ZERO_SHA = "0" * 40


@dataclass(frozen=True)
class FileSize:
    path: str
    size: int


def run_git(arguments: Sequence[str], *, text: bool = False) -> bytes | str:
    """执行 Git 命令；失败时保留 Git 自己的错误信息。"""
    result = subprocess.run(
        ["git", *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or f"Git command failed: git {' '.join(arguments)}")
    if text:
        return result.stdout.decode("utf-8", errors="surrogateescape")
    return result.stdout


def split_null_paths(payload: bytes) -> list[str]:
    """解析 Git 的 NUL 分隔路径，避免空格和特殊字符造成误判。"""
    return [
        item.decode("utf-8", errors="surrogateescape")
        for item in payload.split(b"\0")
        if item
    ]


def staged_files() -> list[FileSize]:
    """读取暂存区中新增或修改后的 blob 大小。"""
    payload = run_git(
        ["diff", "--cached", "--name-only", "-z", "--diff-filter=ACMRTUXB"]
    )
    files: list[FileSize] = []
    for path in split_null_paths(payload):
        size_text = run_git(["cat-file", "-s", f":{path}"], text=True)
        files.append(FileSize(path=path, size=int(size_text.strip())))
    return files


def commit_files(commit_sha: str) -> list[FileSize]:
    """读取指定提交中新增或修改后的 blob 大小。"""
    payload = run_git(
        [
            "diff-tree",
            "--root",
            "--no-commit-id",
            "--name-only",
            "-r",
            "-z",
            "--diff-filter=ACMRTUXB",
            commit_sha,
        ]
    )
    files: list[FileSize] = []
    for path in split_null_paths(payload):
        size_text = run_git(["cat-file", "-s", f"{commit_sha}:{path}"], text=True)
        files.append(FileSize(path=path, size=int(size_text.strip())))
    return files


def format_size(size: int) -> str:
    return f"{size / MIB:.2f} MiB ({size} bytes)"


def validate_files(files: Iterable[FileSize], label: str) -> bool:
    """检查单文件和本次提交总量，并输出可直接处理的失败原因。"""
    file_list = list(files)
    oversized = [item for item in file_list if item.size > MAX_FILE_BYTES]
    total_size = sum(item.size for item in file_list)

    if not oversized and total_size <= MAX_COMMIT_BYTES:
        print(
            f"[Git Size Guard] PASS: {label}; "
            f"{len(file_list)} file(s), {format_size(total_size)}."
        )
        return True

    print(f"[Git Size Guard] BLOCKED: {label}.", file=sys.stderr)
    for item in sorted(oversized, key=lambda value: value.size, reverse=True):
        print(
            f"  - File exceeds 10 MiB: {item.path} ({format_size(item.size)})",
            file=sys.stderr,
        )
    if total_size > MAX_COMMIT_BYTES:
        print(
            "  - Commit content exceeds 50 MiB: "
            f"{format_size(total_size)} across {len(file_list)} file(s).",
            file=sys.stderr,
        )
    print(
        "Move generated data, models, adapters, databases, logs, or archives "
        "outside Git, then update .gitignore if necessary.",
        file=sys.stderr,
    )
    return False


def commits_for_push(
    local_sha: str,
    remote_sha: str,
    remote_name: str | None,
) -> list[str]:
    """计算本次推送真正引入远端的提交，删除远端分支时不检查。"""
    if local_sha == ZERO_SHA:
        return []
    if remote_sha != ZERO_SHA:
        revision = f"{remote_sha}..{local_sha}"
        output = run_git(["rev-list", "--reverse", revision], text=True)
    else:
        # 新分支只检查尚未存在于任意远端跟踪引用中的提交，
        # 避免把整个仓库历史重复扫描一遍。
        remote_selector = (
            f"--remotes={remote_name}"
            if remote_name and remote_name != "."
            else "--remotes"
        )
        output = run_git(
            ["rev-list", "--reverse", local_sha, "--not", remote_selector],
            text=True,
        )
    return [line.strip() for line in output.splitlines() if line.strip()]


def check_staged() -> int:
    return 0 if validate_files(staged_files(), "staged commit") else 1


def check_push(remote_name: str | None) -> int:
    """读取 pre-push 标准输入，并逐提交执行相同限制。"""
    checked: set[str] = set()
    all_valid = True

    for line in sys.stdin:
        fields = line.strip().split()
        if len(fields) != 4:
            continue
        _local_ref, local_sha, _remote_ref, remote_sha = fields
        for commit_sha in commits_for_push(local_sha, remote_sha, remote_name):
            if commit_sha in checked:
                continue
            checked.add(commit_sha)
            subject = run_git(
                ["show", "-s", "--format=%h %s", commit_sha],
                text=True,
            ).strip()
            if not validate_files(commit_files(commit_sha), f"commit {subject}"):
                all_valid = False

    if all_valid:
        print(f"[Git Size Guard] Push check passed for {len(checked)} commit(s).")
        return 0
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 Git 提交和推送中的大文件。")
    parser.add_argument("mode", choices=("staged", "push"))
    parser.add_argument(
        "remote_name",
        nargs="?",
        help="pre-push 传入的目标远端名称。",
    )
    arguments = parser.parse_args()
    try:
        return (
            check_staged()
            if arguments.mode == "staged"
            else check_push(arguments.remote_name)
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[Git Size Guard] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
