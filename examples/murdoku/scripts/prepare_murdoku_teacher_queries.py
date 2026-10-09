"""Project frozen RL queries onto the cleaned teacher protocol without changing splits."""

import argparse, collections, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from murdoku_demo.teacher_protocol import prepare_teacher_record

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--out", required=True)
p.add_argument("--shards", type=int, default=24)
p.add_argument("--turns", type=int, default=20)
p.add_argument("--python-timeout", type=float, default=120)
p.add_argument("--seed", type=int, default=20261005)
a = p.parse_args()
if a.shards < 1:
    raise ValueError("shards must be positive")
root = Path(a.out)
root.mkdir(parents=True, exist_ok=True)
rows = [
    prepare_teacher_record(json.loads(l), a.turns, a.python_timeout)
    for l in Path(a.input).open()
    if l.strip()
]
ids = [r["prompt_uid"] for r in rows]
if len(ids) != len(set(ids)):
    raise ValueError("duplicate query IDs")
if any(r.get("split") not in ("train", "validation") for r in rows):
    raise ValueError("unknown split")
train = [r for r in rows if r["split"] == "train"]
validation = [r for r in rows if r["split"] == "validation"]
assert not {r["prompt_uid"] for r in train} & {r["prompt_uid"] for r in validation}


def write(path, records):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


write(root / "rl_queries.jsonl", rows)
write(root / "train.jsonl", train)
write(root / "validation.jsonl", validation)
random.Random(a.seed).shuffle(train)
for i in range(a.shards):
    write(root / f"shard-{i:02d}.jsonl", train[i :: a.shards])
(root / "system-prompt.txt").write_text(rows[0]["messages"][0]["content"] + "\n")
manifest = {
    "python_workspace": True,
    "train_queries": len(train),
    "validation_queries": len(validation),
    "total_queries": len(rows),
    "shards": a.shards,
    "assistant_turn_limit": a.turns,
    "python_timeout_seconds": a.python_timeout,
    "seed": a.seed,
    "source": str(Path(a.input).resolve()),
    "split_unchanged": True,
    "collection_split": "train",
    "shard_counts": [len(train[i :: a.shards]) for i in range(a.shards)],
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest))
