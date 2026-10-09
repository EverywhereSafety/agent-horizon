"""Check that every selected policy segment survived into the actor batch."""

import argparse, json, re
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--episodes", type=Path, required=True)
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--log", type=Path, required=True)
a = p.parse_args()
receipt = json.loads((a.checkpoint / "training_lineage.json").read_text())
actual = {}
for batch in receipt["actor_batches"]:
    for key, count in batch["policy_segments"].items():
        if key in actual:
            raise AssertionError("segment repeated in actor batches")
        actual[key] = count
expected = {}
cleared = 0
reward_sets = []
actor_group_uids = {key.rsplit("_", 2)[0] for key in actual}
for path in a.episodes.glob("*.group.json"):
    group = json.loads(path.read_text())
    rewards = []
    if group["uid"] not in actor_group_uids:
        continue
    for session, uid in enumerate(group["selected_trajectories"]):
        state = json.loads(
            (a.episodes / (uid.replace("/", "_") + ".replay.json")).read_text()
        )["result"]
        assert state["generated_tokens"] == sum(
            sum(s["response_mask"]) for s in state["segments"]
        )
        cleared += state["context_clears"] > 0
        rewards.append(state["reward"])
        # Native AgentLoopWorkerTQ numbers policy-bearing outputs in order.
        native_uid = group["uid"]
        index = 0
        for segment in state["segments"]:
            count = sum(segment["response_mask"])
            if not count:
                continue
            expected[f"{native_uid}_{session}_{index}"] = count
            index += 1
    reward_sets.append(rewards)
assert cleared > 0, "no selected trajectory cleared context"
assert any(len(set(r)) > 1 for r in reward_sets), "no group-relative learning signal"
assert (
    expected == actual
), f"actor coverage mismatch: expected {len(expected)}, actual {len(actual)}"
text = a.log.read_text()
grads = [float(x) for x in re.findall(r"actor/grad_norm:([0-9.eE+-]+)", text)]
assert grads and all(x == x for x in grads) and any(x > 0 for x in grads)
assert "CONTEXT_CLEAR_UPDATE_COMPLETE" in text
print(
    json.dumps(
        {
            "status": "CONTEXT_CLEAR_ALL_SEGMENTS_UPDATE_PASS",
            "cleared_trajectories": cleared,
            "policy_segments": len(actual),
            "policy_tokens": sum(actual.values()),
            "grad_norms": grads,
        }
    )
)
