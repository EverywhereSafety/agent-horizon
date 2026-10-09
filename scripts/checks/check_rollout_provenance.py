"""Check saved rollout inputs against cached training segments once, offline."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from long_horizon_rl.contracts import Segment
from long_horizon_rl.diagnostics.provenance_check import check_segments

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--audit-dir", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
reports = []
for cache in sorted(args.audit_dir.glob("*.replay.json")):
    audit = cache.with_name(cache.name.removesuffix(".replay.json") + ".jsonl")
    try:
        state = json.loads(cache.read_text())["result"]
        segments = [Segment(**s) for s in state["segments"]]
        with audit.open() as handle:
            result = check_segments(
                (json.loads(line) for line in handle if line.strip()), segments
            )
        reports.append(
            {"trajectory": state["trajectory_uid"], "status": "pass", **result}
        )
    except Exception as exc:
        reports.append({"cache": str(cache), "status": "fail", "error": str(exc)})
summary = {
    "checked": len(reports),
    "failed": sum(r["status"] == "fail" for r in reports),
    "policy_tokens": sum(r.get("policy_tokens", 0) for r in reports),
    "results": reports,
}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "results"}))
raise SystemExit(bool(summary["failed"]) or not summary["checked"])
