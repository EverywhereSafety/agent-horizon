import json
from pathlib import Path
import tempfile
import unittest
from long_horizon_rl.contracts import Config
from examples.smoke import CounterEnvironment
from long_horizon_rl.episode import run_episode
from examples.smoke import FixtureBackend, FixtureTokenizer


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_segments_and_complete_disk_audit(self):
        cfg = Config(
            max_context_tokens=2048,
            clear_trigger_tokens=300,
            clear_target_tokens=180,
            max_new_tokens=128,
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "audit.jsonl"
            r = await run_episode(
                {"prompt_uid": "p", "messages": []},
                "a-0",
                0,
                FixtureBackend(),
                FixtureTokenizer(),
                CounterEnvironment(30),
                cfg,
                audit_path=path,
            )
            self.assertEqual(r.termination, "terminal")
            self.assertGreater(len(r.segments), 5)
            self.assertEqual(
                sum(sum(s.response_mask) for s in r.segments), r.generated_tokens
            )
            self.assertEqual(r.segments[0].turn_start, 0)
            self.assertEqual(r.audit, [])
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(records), r.turns)
            self.assertEqual(records[-1]["observation"]["reward"], 1.0)
