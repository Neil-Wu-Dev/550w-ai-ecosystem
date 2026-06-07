#!/usr/bin/env python3
"""验证 CI 构建产物包含后端运行必须随包发布的资源。"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path


REQUIRED_WHEEL_FILES = {
    "adcust_logic/__init__.py",
    "adcust_logic/locales/en_US.json",
    "adcust_logic/locales/zh_CN.json",
    "adcust_logic/remote/train_entry.py",
}


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: verify_backend_build.py <dist-directory>", file=sys.stderr)
        return 2

    dist_dir = Path(sys.argv[1]).resolve()
    wheels = sorted(dist_dir.glob("*.whl"))
    source_archives = sorted(dist_dir.glob("*.tar.gz"))

    if len(wheels) != 1:
        print(
            f"Expected exactly one wheel in {dist_dir}, found {len(wheels)}.",
            file=sys.stderr,
        )
        return 1
    if len(source_archives) != 1:
        print(
            f"Expected exactly one source archive in {dist_dir}, "
            f"found {len(source_archives)}.",
            file=sys.stderr,
        )
        return 1

    with zipfile.ZipFile(wheels[0]) as wheel:
        packaged_files = set(wheel.namelist())

    missing = sorted(REQUIRED_WHEEL_FILES - packaged_files)
    if missing:
        print("Build artifact is missing required files:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 1

    print(f"Wheel verified: {wheels[0].name}")
    print(f"Source distribution verified: {source_archives[0].name}")
    print("Backend build contents: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
