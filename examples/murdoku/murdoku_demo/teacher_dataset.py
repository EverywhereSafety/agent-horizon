"""One self-contained query/teacher-rollout file for RL, evaluation and SFT."""

import json
import random
from pathlib import Path


def select_sft_examples(examples, count=None, seed=20261005):
    ordered = sorted(examples, key=lambda row: row["prompt_uid"])
    if len({row["prompt_uid"] for row in ordered}) != len(ordered):
        raise ValueError("duplicate SFT question")
    if count is None:
        return ordered
    if count < 1 or count > len(ordered):
        raise ValueError(
            "requested SFT count exceeds accepted examples or is nonpositive"
        )
    selected = set(random.Random(seed).sample(range(len(ordered)), count))
    return [row for i, row in enumerate(ordered) if i in selected]


def read_jsonl(path, allow_incomplete_tail=False):
    lines = Path(path).read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            if (
                allow_incomplete_tail
                and index == len(lines) - 1
                and not line.endswith("\n")
            ):
                return
            raise


from long_horizon_rl.queries import query_from_row


def sft_example_from_row(row, split="train", asset_root=None, view=None):
    """Validate a unified row and return only the model-visible conversation."""
    query = query_from_row(row)
    if view is not None and query.get("murdoku_observation", "text") != view:
        return None
    if query.get("split") != split:
        return None
    rollout = row.get("rollout")
    if rollout is None:
        return None
    if rollout.get("replay_strict_success") is not True:
        raise ValueError("SFT requires a replay-qualified rollout")
    if rollout.get("prompt_uid") != query["prompt_uid"]:
        raise ValueError("rollout identity mismatch")
    prefix = rollout["messages"][: len(query["messages"])]
    if prefix != query["messages"]:
        # Collection inlines images for the model while queries retain portable paths.
        if query.get("murdoku_observation") != "vision" or asset_root is None:
            raise ValueError("rollout prompt differs from query")
        from murdoku_lab.environment.queries import model_inputs

        expected = model_inputs(query, asset_root=asset_root)["messages"]
        if prefix != expected:
            raise ValueError("rollout prompt differs from query")
    return {
        "prompt_uid": query["prompt_uid"],
        "messages": rollout["messages"],
        "tools": rollout["tools"],
    }


def load_sft_examples(path, split="train", asset_root=None, view="text", report=None):
    """Trusted grader metadata stays out; local visual assets are resolved explicitly."""
    examples = []
    counts = {"input": 0, "selected": 0, "excluded": 0}
    for row in read_jsonl(path):
        counts["input"] += 1
        example = sft_example_from_row(row, split, asset_root, view)
        if example is None:
            counts["excluded"] += 1
        else:
            examples.append(example)
            counts["selected"] += 1
    if report is not None:
        report.update(counts)
    return examples


def export_dataset(queries, run_root, output):
    root = Path(run_root)
    records = list(read_jsonl(queries))
    by_id = {r["prompt_uid"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate source queries")
    accepted, attempted = {}, set()
    for path in sorted(root.glob("worker-*/attempts.jsonl")):
        attempted.update(r["uid"] for r in read_jsonl(path, allow_incomplete_tail=True))
    for path in sorted(root.glob("worker-*/query_rollouts.jsonl")):
        for row in read_jsonl(path, allow_incomplete_tail=True):
            query_from_row(row)
            uid, trace = row["prompt_uid"], row["rollout"]
            if uid not in by_id or row["query"] != by_id[uid]:
                raise ValueError("accepted rollout does not match source query")
            if (
                row["split"] != "train"
                or trace.get("replay_strict_success") is not True
            ):
                raise ValueError(
                    "only replay-qualified training rollouts can be exported"
                )
            if (
                trace["prompt_uid"] != uid
                or trace["messages"][: len(by_id[uid]["messages"])]
                != by_id[uid]["messages"]
            ):
                raise ValueError("accepted rollout identity/prompt mismatch")
            metadata = {
                "assistant_turns": trace["turns"],
                "tool_errors": trace.get("recovered_tool_errors", 0),
            }
            candidate_path = trace.get("provenance", {}).get("candidate_file")
            if candidate_path and Path(candidate_path).exists():
                candidate = json.loads(Path(candidate_path).read_text())
                result = candidate["result"]
                metadata.update(
                    generated_tokens=result["generated_tokens"],
                    seconds=result["seconds"],
                    context_clears=result.get("context_clears", 0),
                )
                # Preserve original sampled events and context clears in the same artifact.
                events = []
                for original in candidate["events"]:
                    event = dict(original)
                    public = lambda observation: {
                        k: v
                        for k, v in observation.items()
                        if k not in ("score", "reward", "final_state_variable")
                    }
                    if "observations" in event:
                        event["observations"] = [
                            public(o) for o in event["observations"]
                        ]
                    if "observation" in event:
                        event["observation"] = public(event["observation"])
                    events.append(event)
                trace = dict(
                    trace,
                    events=events,
                    context_events=candidate.get("context_events", []),
                    memory=candidate.get("memory", {}),
                )
            rank = (
                metadata["tool_errors"],
                metadata.get("generated_tokens", float("inf")),
                metadata["assistant_turns"],
            )
            if uid not in accepted or rank < accepted[uid][0]:
                accepted[uid] = (rank, trace, metadata)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    counts = {
        "queries": len(records),
        "train": 0,
        "validation": 0,
        "accepted_rollouts": 0,
    }
    with tmp.open("w") as stream:
        for query in records:
            uid, split = query["prompt_uid"], query["split"]
            if split not in ("train", "validation"):
                raise ValueError("unknown query split")
            counts[split] += 1
            choice = accepted.get(uid)
            counts["accepted_rollouts"] += choice is not None
            status = (
                "accepted"
                if choice
                else (
                    "held_out"
                    if split == "validation"
                    else "not_accepted" if uid in attempted else "pending"
                )
            )
            row = {
                "schema_version": 1,
                "prompt_uid": uid,
                "split": split,
                "query": query,
                "rollout": choice[1] if choice else None,
                "rollout_status": status,
                "rollout_metadata": choice[2] if choice else None,
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(output)
    return counts
