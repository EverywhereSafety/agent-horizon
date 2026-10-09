import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from long_horizon_rl.diagnostics.audit_replay import rebuild_completed
from long_horizon_rl.contracts import Config
from long_horizon_rl.episode import run_episode
from examples.smoke import CounterEnvironment
from examples.smoke import FixtureBackend, FixtureTokenizer


class AuditReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_segments_after_compaction_and_memory(self):
        config = Config(
            max_context_tokens=2048,
            clear_trigger_tokens=300,
            clear_target_tokens=180,
            max_new_tokens=128,
        )
        record = {"prompt_uid": "p", "messages": []}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "audit.jsonl"
            original = await run_episode(
                record,
                "p-1",
                2,
                FixtureBackend(),
                FixtureTokenizer(),
                CounterEnvironment(30),
                config,
                audit_path=path,
            )
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
            replayed = await rebuild_completed(
                record, config, FixtureTokenizer(), path, summary
            )
            self.assertEqual(
                [asdict(s) for s in original.segments],
                [asdict(s) for s in replayed.segments],
            )
            self.assertEqual(original.memory, replayed.memory)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[1]["input_ids"].append(99)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                await rebuild_completed(
                    record, config, FixtureTokenizer(), path, summary
                )
