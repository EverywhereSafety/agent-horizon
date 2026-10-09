"""Trajectory-level training diagnostics; segments must not multiply observations."""

import math
from collections import defaultdict


def episode_statistics(extras):
    unique = {}
    for extra in extras:
        if isinstance(extra, dict) and extra.get("trajectory_uid"):
            unique[extra["trajectory_uid"]] = extra
    rows = list(unique.values())
    if not rows:
        return {}
    stats = {"episodes/count": len(rows)}
    for field in [
        "reward",
        "submitted",
        "strict_success",
        "placement_accuracy",
        "turns",
        "generated_tokens",
        "context_clears",
        "tool_errors",
    ]:
        values = [r.get("reward_extra_info", {}).get(field) for r in rows]
        values = [
            float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)
        ]
        if values:
            stats["episodes/" + field + "_mean"] = sum(values) / len(values)
    rewards = [r.get("reward_extra_info", {}).get("reward") for r in rows]
    rewards = [
        float(v) for v in rewards if isinstance(v, (int, float)) and math.isfinite(v)
    ]
    if rewards:
        stats["episodes/nonzero_reward_fraction"] = sum(v > 0 for v in rewards) / len(
            rewards
        )
        stats["episodes/reward_min"] = min(rewards)
        stats["episodes/reward_max"] = max(rewards)
    groups = defaultdict(list)
    for r in rows:
        reward = r.get("reward_extra_info", {}).get("reward")
        if r.get("native_prompt_uid") and isinstance(reward, (int, float)):
            groups[r["native_prompt_uid"]].append(float(reward))
    complete = [vs for vs in groups.values() if len(vs) > 1]
    if complete:
        stats["episodes/groups_with_reward_variation"] = sum(
            max(vs) > min(vs) for vs in complete
        ) / len(complete)
    return stats
