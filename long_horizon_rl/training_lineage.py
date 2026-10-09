"""Identify the actual logical queries in a prepared actor batch."""

import json


def training_lineage(keys, counts, records):
    if len(keys) != len(counts) or len(records) != len(keys):
        raise ValueError("training lineage rows are unaligned")
    prompts = set()
    native_prompts = set()
    trajectories = set()
    tokens = 0
    for key, count, record in zip(keys, counts, records):
        if not count:
            continue
        uid = json.loads(record)["prompt_uid"]
        if not isinstance(uid, str) or not uid:
            raise ValueError("logical prompt UID required")
        parts = key.rsplit("_", 2)
        if len(parts) != 3:
            raise ValueError("native segment key required")
        prompts.add(uid)
        native_prompts.add(parts[0])
        trajectories.add(tuple(parts[:2]))
        tokens += count
    return {
        "prompt_uids": sorted(prompts),
        "native_prompt_uids": sorted(native_prompts),
        "policy_trajectories": len(trajectories),
        "policy_tokens": tokens,
        "policy_segments": {key: count for key, count in zip(keys, counts) if count},
    }
