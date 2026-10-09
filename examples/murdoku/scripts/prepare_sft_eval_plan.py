import argparse
import json
from pathlib import Path
from murdoku_demo.sft_eval import build_eval_plan

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--sft-manifest", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    rows = [json.loads(line) for line in Path(a.dataset).open() if line.strip()]
    manifest = json.loads(Path(a.sft_manifest).read_text())
    plan = build_eval_plan(rows, set(manifest["prompt_uids"]))
    Path(a.output).write_text(json.dumps(plan, indent=2) + "\n")
    print("EVALUATION_PLAN_READY", len(plan["tasks"]), flush=True)
