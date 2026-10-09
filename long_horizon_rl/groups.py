import asyncio
import math


async def select_group(factory, attempt_n=18, retain_n=16):
    """Return first terminal successes; acknowledge cancellation and fail closed on shortage."""
    if not 0 < retain_n <= attempt_n:
        raise ValueError("invalid group sizes")
    tasks = {asyncio.create_task(factory(i)): i for i in range(attempt_n)}
    pending = set(tasks)
    retained = []
    failed = 0
    cancelled = 0
    try:
        while pending and len(retained) < retain_n:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in sorted(done, key=lambda t: tasks[t]):
                try:
                    result = task.result()
                except Exception:
                    failed += 1
                    continue
                if result.termination != "terminal" or not math.isfinite(result.reward):
                    failed += 1
                    continue
                retained.append(result)
        if len(retained) < retain_n:
            raise RuntimeError("insufficient terminal trajectories")
        selected = retained[:retain_n]
        if (
            len({r.trajectory_uid for r in selected}) != retain_n
            or len({r.prompt_uid for r in selected}) != 1
        ):
            raise ValueError("duplicate trajectories or mixed prompt group")
        return selected, {
            "attempted": attempt_n,
            "retained": retain_n,
            "invalid": failed,
            "completed_redundant": len(retained) - retain_n,
            "cancelled": len(pending),
        }
    finally:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


def advantages(group):
    if not group or len({r.prompt_uid for r in group}) != 1:
        raise ValueError("mixed or empty group")
    if len({r.trajectory_uid for r in group}) != len(group):
        raise ValueError("duplicate trajectory")
    rewards = [r.reward for r in group]
    if not all(math.isfinite(r) for r in rewards):
        raise ValueError("nonfinite rewards")
    mean = sum(rewards) / len(rewards)
    return {r.trajectory_uid: r.reward - mean for r in group}


def pack_group(group):
    adv = advantages(group)
    rows = []
    for r in group:
        train_segments = [s for s in r.segments if any(s.response_mask)]
        total_tokens = sum(sum(s.response_mask) for s in train_segments)
        for index, s in enumerate(train_segments):
            s.validate()
            rows.append(
                {
                    "prompt_uid": r.prompt_uid,
                    "trajectory_uid": r.trajectory_uid,
                    "segment_uid": f"{r.trajectory_uid}/{index}",
                    "prompt_ids": s.prompt_ids,
                    "response_ids": s.response_ids,
                    "response_mask": s.response_mask,
                    "rollout_log_probs": s.rollout_log_probs,
                    "advantages": [adv[r.trajectory_uid] * m for m in s.response_mask],
                    "trajectory_policy_tokens": total_tokens,
                    "dispatch_version": r.dispatch_version,
                    "min_served_version": r.min_served_version,
                    "max_served_version": r.max_served_version,
                }
            )
    return rows
