"""Export queries and successful interactions as one unified dataset."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from murdoku_demo.teacher_dataset import export_dataset

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--queries", required=True)
    p.add_argument("--run-root", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(json.dumps(export_dataset(a.queries, a.run_root, a.output)), flush=True)
