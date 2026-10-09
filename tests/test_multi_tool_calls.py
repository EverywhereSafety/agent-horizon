import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from long_horizon_rl.parser import parse_actions
from long_horizon_rl.contracts import Config, Generation
from long_horizon_rl.episode import run_episode
from examples.smoke import FixtureTokenizer
from long_horizon_rl.diagnostics.audit_replay import rebuild_completed


def call(name, args):
    return (
        "<tool_call>" + json.dumps({"name": name, "arguments": args}) + "</tool_call>"
    )


class BatchBackend:
    async def generate(self, ids, cap, version, uid, turn):
        text = (
            call("memory", {"action": "write", "title": "progress", "content": "saved"})
            + call("advance", {})
            + call("submit", {})
            + call("advance", {})
        )
        tokens = list(text.encode())
        return Generation(tokens, [-0.5] * len(tokens), text, version)


class BatchEnvironment:
    def __init__(self):
        self.calls = []

    def step(self, action):
        self.calls.append(action["tool"])
        return (
            {"done": True, "reward": 1.0, "score": {"solved": True}}
            if action["tool"] == "submit"
            else {"done": False, "value": 1}
        )

    def close(self):
        pass


class MultiToolTests(unittest.IsolatedAsyncioTestCase):
    def test_all_calls_validated(self):
        schema = [{"name": "advance", "parameters": {"required": ["amount"]}}]
        with self.assertRaisesRegex(ValueError, "missing argument"):
            parse_actions(call("advance", {"amount": 1}) + call("advance", {}), schema)
        self.assertEqual(
            len(
                parse_actions(
                    "<think>ignore</think>" + call("advance", {}) + call("submit", {})
                )
            ),
            2,
        )

    async def test_order_terminal_mask_and_exact_audit_replay(self):
        config = Config(
            max_context_tokens=4096,
            clear_trigger_tokens=3000,
            clear_target_tokens=2000,
            max_new_tokens=1024,
        )
        record = {"prompt_uid": "batch", "messages": []}
        env = BatchEnvironment()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "audit.jsonl"
            original = await run_episode(
                record,
                "batch-1",
                0,
                BatchBackend(),
                FixtureTokenizer(),
                env,
                config,
                audit_path=path,
            )
            self.assertEqual(env.calls, ["advance", "submit"])
            self.assertEqual(original.reward, 1.0)
            self.assertEqual(original.turns, 1)
            self.assertEqual(original.tool_error_count, 1)
            self.assertEqual(original.memory["notes"]["progress"], "saved")
            row = json.loads(path.read_text())
            self.assertEqual(len(row["observation"]["tool_results"]), 4)
            segment = original.segments[0]
            self.assertEqual(sum(segment.response_mask), len(row["output_ids"]))
            summary = {
                k: getattr(original, k)
                for k in (
                    "trajectory_uid",
                    "reward",
                    "termination",
                    "turns",
                    "generated_tokens",
                    "context_clears",
                )
            }
            summary.update(
                min_served_version=original.min_served_version,
                retained_segments=len(original.segments),
            )
            replay = await rebuild_completed(
                record, config, FixtureTokenizer(), path, summary
            )
            self.assertEqual(
                [asdict(s) for s in original.segments],
                [asdict(s) for s in replay.segments],
            )
            self.assertEqual(original.memory, replay.memory)
