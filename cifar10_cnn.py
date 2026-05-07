"""Compatibility entry point for the CIFAR-10 research lab.

The original project ran everything from this file. The implementation now
lives in ``src/cifar10_research_lab`` so the coursework code is easier to read,
test, and extend.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cifar10_research_lab.train import main


if __name__ == "__main__":
    raise SystemExit(main())
