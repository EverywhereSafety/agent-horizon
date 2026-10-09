"""Create new protocol projections without changing frozen puzzle splits."""

import argparse, json
from pathlib import Path
from murdoku_demo.murdoku_protocol import simplify_record

p = argparse.ArgumentParser()
p.add_argument("--train", required=True)
p.add_argument("--validation", required=True)
p.add_argument("--out", required=True)
p.add_argument("--placement-reward-weight", type=float, default=0.2)
p.add_argument("--workspace", action="store_true")
p.add_argument("--python-timeout", type=float, default=120)
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
manifest = {
    "python_workspace": a.workspace,
    "python_timeout_seconds": a.python_timeout if a.workspace else 30,
    "source_files": {},
    "counts": {},
    "structural_split_unchanged": True,
}
ids = {}
for split, source in [("train", a.train), ("validation", a.validation)]:
    src = Path(source)
    records = [
        simplify_record(
            json.loads(line), workspace=a.workspace, python_timeout=a.python_timeout
        )
        for line in src.read_text().splitlines()
    ]
    if split == "train":
        for r in records:
            r["murdoku_reward"] = {
                "mode": "placement_shaped",
                "placement_weight": a.placement_reward_weight,
            }
    else:
        for r in records:
            r["murdoku_reward"] = {"mode": "strict"}
    ids[split] = {r["prompt_uid"] for r in records}
    assert len(ids[split]) == len(records), "duplicate structural prompt"
    destination = out / (split + ".jsonl")
    destination.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    )
    manifest["source_files"][split] = {"path": str(src.resolve())}
    manifest["counts"][split] = len(records)
assert not ids["train"] & ids["validation"], "training/validation overlap"
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest), flush=True)
