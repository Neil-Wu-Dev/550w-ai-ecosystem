"""Right-click entry: run every AdCust backend test package."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from tests import _bootstrap  # noqa: F401


if __name__ == "__main__":
    tests_dir = Path(__file__).resolve().parent
    print("\n=== AdCust Backend Test Suite ===")
    print("Mode: right-click runner")
    print("Packages: unit, integration")
    print("Note: integration tests need the same Python interpreter that can run the backend.\n")
    suite = unittest.defaultTestLoader.discover(str(tests_dir), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2, buffer=False).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
