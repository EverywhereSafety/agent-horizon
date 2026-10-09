import asyncio
import tempfile
import unittest
from pathlib import Path
from long_horizon_rl.contracts import Config
from long_horizon_rl.continuation import ContinuationStore
from long_horizon_rl.episode import run_episode, EpisodeDeadlineExceeded
from examples.smoke import CounterEnvironment
from examples.smoke import FixtureTokenizer


class DeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def test_deadline_closes_environment_and_emits_typed_record(self):
        class Backend:
            cancelled = False

            async def generate(self, *args):
                try:
                    await asyncio.Event().wait()
                finally:
                    self.cancelled = True

        with tempfile.TemporaryDirectory() as d:
            backend = Backend()
            env = CounterEnvironment(3)
            audit = Path(d) / "audit.jsonl"
            with self.assertRaises(EpisodeDeadlineExceeded):
                await run_episode(
                    {"prompt_uid": "p", "messages": []},
                    "p/1",
                    0,
                    backend,
                    FixtureTokenizer(),
                    env,
                    Config(episode_timeout_seconds=0.05),
                    audit_path=audit,
                )
            self.assertTrue(backend.cancelled)
            self.assertTrue(env.closed)
            self.assertIn(
                "episode_timeout", audit.with_suffix(".termination.json").read_text()
            )

    async def test_resume_deadline_cannot_reset_budget(self):
        with tempfile.TemporaryDirectory() as d:
            config = Config(episode_timeout_seconds=1)
            record = {"prompt_uid": "p", "messages": []}
            store = ContinuationStore(Path(d) / "continue.json", record, config)
            store.save({"elapsed_seconds": 2})
            env = CounterEnvironment(3)
            with self.assertRaises(EpisodeDeadlineExceeded):
                await run_episode(
                    record,
                    "p/1",
                    0,
                    None,
                    FixtureTokenizer(),
                    env,
                    config,
                    continuation=store,
                )
            self.assertTrue(env.closed)

    def test_invalid_deadline(self):
        for value in [0, -1, float("nan"), float("inf"), True]:
            with self.assertRaises(ValueError):
                Config(episode_timeout_seconds=value)
