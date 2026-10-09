"""Convert supported raw queries to verl parquet without changing task semantics."""

import argparse
import json
from pathlib import Path
from long_horizon_rl.queries import load_queries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    records, deferred = load_queries(a.source)
    import pandas as pd

    rows = [
        {
            "prompt": r["messages"],
            "record_json": json.dumps(r, ensure_ascii=False),
            "agent_name": "long_horizon",
            "data_source": r.get("data_source", "long_horizon"),
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {"index": i, "prompt_uid": r["prompt_uid"]},
        }
        for i, r in enumerate(records)
    ]
    if not rows:
        raise ValueError("no self-contained records")
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    out.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "supported": len(rows),
                "deferred": deferred,
                "semantics": "original messages and record turn limits preserved",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
