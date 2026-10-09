"""Ingest supplied query formats without executing their code."""

import ast
import hashlib
import json
from pathlib import Path

if __package__:
    from .environments import environment_factory
else:
    from long_horizon_rl.environments import environment_factory


def query_from_row(row):
    if not isinstance(row, dict):
        raise ValueError("dataset row must be an object")
    query = row.get("query", row)
    if not isinstance(query, dict):
        raise ValueError("query must be an object")
    if "query" in row and (
        row.get("prompt_uid", query.get("prompt_uid")) != query.get("prompt_uid")
        or row.get("split", query.get("split")) != query.get("split")
    ):
        raise ValueError("outer query identity/split mismatch")
    return query


def load_queries(path, split=None):
    supported, deferred = [], []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = query_from_row(json.loads(line))
        if split is not None and record.get("split") != split:
            continue
        if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
            raise ValueError(f"line {line_number}: messages required")
        family = record.get("environment_type", "dynamic")
        factory = environment_factory(family)
        validator = getattr(factory, "validate_record", None)
        if validator is not None:
            validator(record)
        if family == "dynamic" and "latent_dynamics" not in record:
            deferred.append(
                {
                    "line": line_number,
                    "id": record.get("id"),
                    "env_name": record.get("env_name"),
                    "reason": "external environment/release required",
                }
            )
            continue
        for source in (
            [record["latent_dynamics"], *record.get("tool_implementations", [])]
            if family == "dynamic"
            else []
        ):
            ast.parse(source)  # Syntax check only. Runtime code belongs in the sandbox.
        record = dict(record)
        record["prompt_uid"] = (
            record.get("prompt_uid")
            or record.get("id")
            or record.get("task_id")
            or hashlib.sha256(line.encode()).hexdigest()[:24]
        )
        record["agent_info"] = dict(record.get("agent_info", {}))
        if "assistant_turn_limit" in record:
            limit = record["assistant_turn_limit"]
            if type(limit) is not int or limit < 1:
                raise ValueError("invalid assistant_turn_limit")
            record["agent_info"]["max_turn"] = limit
        elif "max_turn" in record:
            record["agent_info"].setdefault("max_turn", record["max_turn"])
        if not record.get("tool_schemas"):
            raise ValueError("tool schemas required")
        supported.append(record)
    return supported, deferred


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=("train", "validation"))
    args = p.parse_args()
    records, deferred = load_queries(args.input, split=args.split)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    out.with_suffix(".report.json").write_text(
        json.dumps({"supported": len(records), "deferred": deferred}, indent=2)
    )
    print(json.dumps({"supported": len(records), "deferred": len(deferred)}))
