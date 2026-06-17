"""Right-click entry: run backend integration tests only."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from tests import _bootstrap  # noqa: F401


if __name__ == "__main__":
    tests_dir = Path(__file__).resolve().parent / "integration"
    print("\n=== AdCust Backend Integration Tests ===")
    print("What this checks: FastAPI routes, DTO parsing, dependency overrides, and response contracts.\n")
    suite = unittest.defaultTestLoader.discover(str(tests_dir), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2, buffer=False).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
