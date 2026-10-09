"""Offline audit-to-segment checks; never invoked on the training hot path."""

import math


def check_segments(audit_rows, segments, probability_tolerance=1e-6):
    slots = []
    for segment in segments:
        segment.validate()
        ranges = segment.generation_version_ranges
        covered = sum(r["response_end"] - r["response_start"] for r in ranges)
        if covered != sum(segment.response_mask):
            raise ValueError("sampled-token provenance coverage mismatch")
        for r in ranges:
            start, end = r["response_start"], r["response_end"]
            slots.append((segment, start, end, r))
    calls = tokens = 0
    for row in audit_rows:
        if calls >= len(slots):
            raise ValueError("audit generation absent from training segments")
        segment, start, end, version = slots[calls]
        if segment.prompt_ids + segment.response_ids[:start] != row["input_ids"]:
            raise ValueError("learner conditioning differs from rollout input")
        if segment.response_ids[start:end] != row["output_ids"]:
            raise ValueError("learner targets differ from sampled tokens")
        probs = row["log_probs"]
        if len(probs) != end - start:
            raise ValueError("audit probabilities are unaligned")
        if any(
            not math.isfinite(p) or abs(p - q) > probability_tolerance
            for p, q in zip(probs, segment.rollout_log_probs[start:end])
        ):
            raise ValueError("behavior probabilities changed in packing")
        if (version["min_served_version"], version["max_served_version"]) != (
            row["served_version"],
            row.get("max_served_version", row["served_version"]),
        ):
            raise ValueError("behavior version range changed in packing")
        calls += 1
        tokens += end - start
    if calls != len(slots):
        raise ValueError("training generation absent from audit")
    return {"generations": calls, "policy_tokens": tokens, "segments": len(segments)}
