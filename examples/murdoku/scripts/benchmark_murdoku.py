"""Delegate standalone puzzle benchmarking to the Murdoku repository."""

import os, runpy, sys
from pathlib import Path

root = Path(os.environ["LONG_HORIZON_MURDOKU_ROOT"]).resolve()
sys.path.insert(0, str(root))
runpy.run_path(str(root / "scripts/benchmark_tools.py"), run_name="__main__")
