import json
import tempfile
from pathlib import Path
import unittest
from long_horizon_rl.continuation import ContinuationStore
from long_horizon_rl.contracts import Config
from examples.smoke import CounterEnvironment
from long_horizon_rl.episode import run_episode
from examples.smoke import FixtureBackend, FixtureTokenizer


class ContinuationTests(unittest.IsolatedAsyncioTestCase):
    async def test_resume_preserves_environment_memory_and_exact_audit(self):
        cfg = Config(
            max_context_tokens=2048,
            clear_trigger_tokens=300,
            clear_target_tokens=180,
            max_new_tokens=128,
        )
        record = {"prompt_uid": "p", "messages": []}

        class Interrupted(FixtureBackend):
            async def generate(self, ids, cap, version, uid, turn):
                if turn == 5:
                    raise RuntimeError("injected interruption")
                return await super().generate(ids, cap, version, uid, turn)

        with tempfile.TemporaryDirectory() as d:
            store = ContinuationStore(Path(d) / "continuation.json", record, cfg)
            audit = Path(d) / "audit.jsonl"
            with self.assertRaises(RuntimeError):
                await run_episode(
                    record,
                    "a-0",
                    0,
                    Interrupted(),
                    FixtureTokenizer(),
                    CounterEnvironment(9),
                    cfg,
                    audit_path=audit,
                    continuation=store,
                    checkpoint_interval=2,
                )
            self.assertIsNotNone(store.load())
            resumed = await run_episode(
                record,
                "a-0",
                0,
                FixtureBackend(),
                FixtureTokenizer(),
                CounterEnvironment(9),
                cfg,
                audit_path=audit,
                continuation=store,
                checkpoint_interval=2,
            )
            reference = await run_episode(
                record,
                "a-0",
                0,
                FixtureBackend(),
                FixtureTokenizer(),
                CounterEnvironment(9),
                cfg,
            )
            records = [json.loads(line) for line in audit.read_text().splitlines()]
            self.assertEqual(records, reference.audit)
            self.assertEqual(
                (resumed.reward, resumed.turns, resumed.memory),
                (reference.reward, reference.turns, reference.memory),
            )
            self.assertIsNone(store.load())

    async def test_changed_input_rejected(self):
        cfg = Config()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "r.json"
            ContinuationStore(path, {"x": 1}, cfg).save({"next_turn": 3})
            with self.assertRaises(ValueError):
                ContinuationStore(path, {"x": 2}, cfg).load()
