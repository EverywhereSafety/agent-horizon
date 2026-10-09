import argparse
import json
from pathlib import Path
from murdoku_demo.sft_eval import summarize_results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    a = p.parse_args()
    root = Path(a.root)
    report = {}
    for phase in ["base", "sft"]:
        rows = []
        for path in sorted((root / phase).glob("worker-*/evaluation-summary.json")):
            rows.extend(json.loads(path.read_text())["results"])
        if len(rows) != 64 or len({r["task_id"] for r in rows}) != 64:
            raise ValueError(f"{phase}: incomplete or duplicate evaluation tasks")
        report[phase] = summarize_results(rows)
    for split in ["train", "validation"]:
        base, sft = report["base"][split], report["sft"][split]
        report.setdefault("difference", {})[split] = {
            "strict_success_count": sft["strict_successes"] - base["strict_successes"],
            "questions_solved_at_least_once": sft["questions_solved_at_least_once"]
            - base["questions_solved_at_least_once"],
        }
    (root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
