"""Scripted CPU walkthrough of episode state, context clearing and replay."""

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from long_horizon_rl.contracts import Config, Generation
from long_horizon_rl.episode import run_episode
from long_horizon_rl.groups import select_group, pack_group
from long_horizon_rl.replay import VersionedReplay


class CounterEnvironment:
    """Trusted synthetic dynamics; no user/generated Python is executed."""

    def __init__(self, target=8):
        if target <= 0:
            raise ValueError("target must be positive")
        self.target, self.value, self.closed = target, 0, False

    def step(self, action):
        if self.closed:
            raise RuntimeError("environment closed")
        if action.get("tool") != "advance":
            raise ValueError("unknown tool")
        amount = action.get("amount", 1)
        if type(amount) is not int or amount not in (1, 2):
            raise ValueError("amount must be 1 or 2")
        self.value += amount
        done = self.value >= self.target
        return {
            "value": self.value,
            "done": done,
            "reward": float(self.value == self.target) if done else 0.0,
        }

    def snapshot(self):
        return {"target": self.target, "value": self.value}

    def restore(self, state):
        self.target, self.value = state["target"], state["value"]

    def close(self):
        self.closed = True


class FixtureTokenizer:
    """Character IDs for CPU contracts only; never a model tokenizer."""

    def initial(self, messages):
        return [ord(c) for c in json.dumps(messages)]

    def observation(self, text):
        return [ord(c) for c in "\nTOOL:" + text + "\nASSISTANT:"]

    def memory(self, memory):
        return [ord(c) for c in json.dumps(memory, sort_keys=True)] if memory else []


class FixtureBackend:
    async def generate(self, ids, cap, version, uid, turn):
        await asyncio.sleep(0)
        if turn == 0:
            action = {"tool": "memory_write", "key": "goal", "value": "nine"}
        elif turn == 1:
            action = {"tool": "memory_read", "key": "goal"}
        else:
            action = {
                "tool": "advance",
                "amount": 1 if int(uid.split("-")[-1]) % 2 == 0 else 2,
            }
        text = json.dumps(action)
        tokens = [ord(c) for c in text]
        return Generation(tokens, [-0.5] * len(tokens), text, version)


async def smoke(output):
    config = Config(
        max_context_tokens=2048,
        clear_trigger_tokens=300,
        clear_target_tokens=180,
        max_new_tokens=128,
    )
    record = {
        "prompt_uid": "counter-smoke",
        "messages": [
            {"role": "user", "content": "Reach exactly nine; remember the goal."}
        ],
    }

    async def factory(i):
        return await run_episode(
            record,
            f"attempt-{i}",
            0,
            FixtureBackend(),
            FixtureTokenizer(),
            CounterEnvironment(9),
            config,
        )

    group, metrics = await select_group(factory, 18, 16)
    rows = pack_group(group)
    replay = VersionedReplay()
    replay.add(group)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    replay.save(output / "replay.json")
    restored = VersionedReplay.load(output / "replay.json")
    assert [r.trajectory_uid for r in restored.take()] == [
        r.trajectory_uid for r in group
    ]
    assert restored.take() is None
    assert all(r.memory["goal"] == "nine" and r.context_clears >= 2 for r in group)
    assert any(x for row in rows for x in row["advantages"])
    report = {
        "status": "CPU_CONTRACTS_PASS",
        "model_training_validated": False,
        "metrics": metrics,
        "segments": len(rows),
        "context_clears": sum(r.context_clears for r in group),
        "rewards": [r.reward for r in group],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "episodes.json").write_text(json.dumps([asdict(r) for r in group]))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/cpu-smoke")
    asyncio.run(smoke(parser.parse_args().output))
