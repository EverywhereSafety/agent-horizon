"""Summarize evaluation episodes without counting cleared segments twice."""

import argparse
import json
import time
from pathlib import Path


def summarize(path, expected_episodes=None):
    sessions = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        uid = row.get("uid")
        if not isinstance(uid, str):
            raise ValueError("Validation row lacks an episode UID")
        parts = uid.rsplit("_", 2)
        session = "_".join(parts[:2]) if len(parts) == 3 else uid
        prior = sessions.get(session)
        if prior and prior.get("strict_success") != row.get("strict_success"):
            raise ValueError("Segments disagree on episode success")
        sessions[session] = row
    rows = list(sessions.values())
    valid = [r for r in rows if isinstance(r.get("strict_success"), (bool, int, float))]
    result = {
        "step": path.stem,
        "completed_episodes": len(rows),
        "scored_episodes": len(valid),
        "strict_solved": sum(bool(r["strict_success"]) for r in valid),
        "strict_success_rate": (
            sum(bool(r["strict_success"]) for r in valid) / len(valid)
            if valid
            else None
        ),
    }
    complete = expected_episodes is None or (
        len(rows) == expected_episodes and len(valid) == expected_episodes
    )
    result["status"] = "complete" if complete else "partial"
    if not complete:
        result["strict_success_rate"] = None
    for field in (
        "submitted",
        "turns",
        "generated_tokens",
        "tool_errors",
        "context_clears",
        "placement_accuracy",
    ):
        values = [r[field] for r in rows if isinstance(r.get(field), (int, float))]
        if values:
            result[field + "_mean"] = sum(values) / len(values)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_dir", type=Path)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    last = None
    while True:
        files = sorted(
            args.validation_dir.glob("*.jsonl"),
            key=lambda p: (int(p.stem) if p.stem.isdigit() else -1, p.name),
        )
        results = []
        for path in files:
            try:
                results.append(summarize(path, args.expected_episodes))
            except (ValueError, OSError) as exc:
                results.append(
                    {
                        "step": path.stem,
                        "status": "pending or invalid",
                        "detail": str(exc),
                    }
                )
        if results != last:
            print(
                json.dumps(
                    {
                        "validation": results,
                        "note": "Fixed held-out monitoring; small samples are exploratory.",
                    }
                ),
                flush=True,
            )
            last = results
        if not args.follow:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
