"""Positive advantage weights for native token-sum trajectory mean loss."""

from collections import defaultdict


def trajectory_weights(keys, policy_token_counts, num_minibatches=1):
    if len(keys) != len(policy_token_counts) or num_minibatches < 1:
        raise ValueError("invalid trajectory weighting batch")
    totals = defaultdict(int)
    sessions = []
    for key, count in zip(keys, policy_token_counts):
        if type(count) is not int or count < 0:
            raise ValueError("invalid policy token count")
        if not count:
            sessions.append(None)
            continue
        fields = key.rsplit("_", 2)
        if len(fields) != 3 or not fields[-1].isdigit():
            raise ValueError("invalid upstream trajectory key")
        session = tuple(fields[:2])
        sessions.append(session)
        totals[session] += count
    if not totals:
        raise ValueError("no trainable trajectories")
    # Native token-sum compensates DP averaging. Scale each minibatch to an
    # unbiased estimate of the trajectory mean over retained policy tokens.
    return [
        num_minibatches / (len(totals) * totals[s]) if s is not None else 0.0
        for s in sessions
    ]
