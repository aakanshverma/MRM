#!/usr/bin/env python3
"""Root entry point so the pipeline runs with a bare ``python main.py``.

The implementation lives in :mod:`model_monitoring.main`; this shim only makes
sure the package is importable when the script is run directly from the
repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_monitoring.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
