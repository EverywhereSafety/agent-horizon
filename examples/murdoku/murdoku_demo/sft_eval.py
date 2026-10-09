"""Fixed matched evaluation tasks for base versus SFT checkpoints."""

import json
import random
from pathlib import Path

from long_horizon_rl.queries import query_from_row

TEACHER_EVAL_ARGS = [
    "--parallel",
    "1",
    "--turns",
    "20",
    "--tokens",
    "81920",
    "--generation-limit",
    "0",
    "--context",
    "131072",
    "--context-clear",
    "--clear-trigger",
    "49152",
    "--clear-target",
    "32768",
    "--context-feedback",
    "--thinking",
    "--preserve-thinking",
    "--temperature",
    "1",
    "--top-p",
    ".95",
    "--top-k",
    "20",
    "--min-p",
    "0",
    "--presence-penalty",
    "0",
    "--repetition-penalty",
    "1",
    "--workspace",
    "--python-timeout",
    "120",
    "--keep-input-protocol",
    "--purpose",
    "evaluation",
    "--request-timeout",
    "900",
]


def build_eval_plan(rows, sft_uids, per_split=8, replicates=4, seed=20261005):
    queries = [query_from_row(row) for row in rows]
    train = [
        q for q in queries if q.get("split") == "train" and q["prompt_uid"] in sft_uids
    ]
    validation = [q for q in queries if q.get("split") == "validation"]
    if {q["prompt_uid"] for q in train} & {q["prompt_uid"] for q in validation}:
        raise ValueError("train/validation overlap")
    rng = random.Random(seed)
    tasks = []
    for split, pool in [("train", train), ("validation", validation)]:
        selected = rng.sample(sorted(pool, key=lambda q: q["prompt_uid"]), per_split)
        for query in selected:
            for replicate in range(replicates):
                tasks.append(
                    {
                        "task_id": len(tasks),
                        "split": split,
                        "prompt_uid": query["prompt_uid"],
                        "replicate": replicate,
                        "seed": seed + replicate * 10000,
                        "query": query,
                    }
                )
    return {
        "tasks": tasks,
        "teacher_eval_args": TEACHER_EVAL_ARGS,
        "episode_timeout_seconds": 900,
        "questions_per_split": per_split,
        "replicates": replicates,
        "selection_seed": seed,
        "scope": "train diagnostic and disjoint validation reported separately",
    }


def summarize_results(results):
    report = {}
    for split in ["train", "validation"]:
        rows = [r for r in results if r["split"] == split]
        valid = [r for r in rows if r.get("status") != "infrastructure_error"]
        questions = {r["prompt_uid"] for r in rows}
        solved = {r["prompt_uid"] for r in valid if r.get("strict_success")}
        report[split] = {
            "attempts": len(rows),
            "valid_attempts": len(valid),
            "infrastructure_errors": len(rows) - len(valid),
            "strict_successes": sum(bool(r.get("strict_success")) for r in valid),
            "positive_rewards": sum((r.get("reward") or 0) > 0 for r in valid),
            "questions": len(questions),
            "questions_solved_at_least_once": len(solved),
            "total_generated_tokens": sum(r.get("generated_tokens", 0) for r in rows),
            "tool_errors": sum(r.get("execution_errors", 0) for r in rows),
        }
    return report
