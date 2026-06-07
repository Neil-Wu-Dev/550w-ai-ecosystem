"""Right-click entry: run unit tests only."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from tests import _bootstrap  # noqa: F401


if __name__ == "__main__":
    tests_dir = Path(__file__).resolve().parent / "unit"
    print("\n=== AdCust Unit Tests ===")
    print("What this checks: rich models, mappers, services, manifest hashing, and inference message building.\n")
    suite = unittest.defaultTestLoader.discover(str(tests_dir), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2, buffer=False).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
