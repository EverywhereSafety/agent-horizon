import tempfile
import unittest
from pathlib import Path
from long_horizon_rl.contracts import Config, Generation
from long_horizon_rl.continuation import ContinuationStore
from long_horizon_rl.episode import run_episode
from examples.smoke import CounterEnvironment
from examples.smoke import FixtureTokenizer


class Backend:
    def __init__(self, interrupt=None):
        self.interrupt = interrupt
        self.caps = []

    async def generate(self, ids, cap, version, uid, turn):
        if turn == self.interrupt:
            raise RuntimeError("interrupted")
        self.caps.append(cap)
        return Generation([1], [-0.1], '{"tool":"advance","amount":1}', version)


class EpisodeLimitTests(unittest.IsolatedAsyncioTestCase):
    def config(self, **kwargs):
        return Config(
            max_context_tokens=2048,
            clear_trigger_tokens=300,
            clear_target_tokens=180,
            max_new_tokens=128,
            **kwargs,
        )

    async def test_optional_generation_limit(self):
        backend = Backend()
        result = await run_episode(
            {"prompt_uid": "p", "messages": []},
            "a",
            0,
            backend,
            FixtureTokenizer(),
            CounterEnvironment(100),
            self.config(max_generated_tokens=3),
        )
        self.assertEqual(
            (result.turns, result.generated_tokens, result.termination),
            (3, 3, "generation_limit"),
        )
        self.assertEqual(backend.caps, [3, 2, 1])

    async def test_config_caps_record_turn_limit(self):
        record = {"prompt_uid": "p", "messages": [], "agent_info": {"max_turn": 2000}}
        result = await run_episode(
            record,
            "a",
            0,
            Backend(),
            FixtureTokenizer(),
            CounterEnvironment(100),
            self.config(max_turns=4),
        )
        self.assertEqual(result.turns, 4)
        self.assertIsNone(Config().max_generated_tokens)

    async def test_generation_budget_survives_resume(self):
        record = {"prompt_uid": "p", "messages": []}
        config = self.config(max_generated_tokens=3)
        with tempfile.TemporaryDirectory() as root:
            store = ContinuationStore(Path(root) / "state.json", record, config)
            with self.assertRaises(RuntimeError):
                await run_episode(
                    record,
                    "a",
                    0,
                    Backend(interrupt=2),
                    FixtureTokenizer(),
                    CounterEnvironment(100),
                    config,
                    continuation=store,
                    checkpoint_interval=1,
                )
            self.assertEqual(store.load()["generated_tokens"], 2)
            result = await run_episode(
                record,
                "a",
                0,
                Backend(),
                FixtureTokenizer(),
                CounterEnvironment(100),
                config,
                continuation=store,
            )
            self.assertEqual((result.turns, result.generated_tokens), (3, 3))

    async def test_tool_observation_overflow_compacts_without_losing_actions(self):
        class LongObservationTokenizer(FixtureTokenizer):
            def observation(self, text):
                return [8] * 2200

        cfg = self.config(max_turns=3)
        result = await run_episode(
            {"prompt_uid": "p", "messages": []},
            "a",
            0,
            Backend(),
            LongObservationTokenizer(),
            CounterEnvironment(3),
            cfg,
        )
        self.assertEqual(result.termination, "terminal")
        self.assertEqual(result.context_clears, 2)
        self.assertEqual(
            sum(sum(s.response_mask) for s in result.segments), result.generated_tokens
        )
        self.assertTrue(
            all(
                len(s.prompt_ids) + len(s.response_ids) <= cfg.max_context_tokens
                for s in result.segments
            )
        )

    async def test_friction_called_once_per_clear_only_when_enabled(self):
        class LongObservationTokenizer(FixtureTokenizer):
            def observation(self, text):
                return [8] * 2200

        class Env(CounterEnvironment):
            def __init__(self):
                super().__init__(3)
                self.updates = 0

            async def context_update(self):
                self.updates += 1

        for enabled in (False, True):
            env = Env()
            result = await run_episode(
                {
                    "prompt_uid": "p",
                    "messages": [],
                    "agent_info": {"apply_context_update_friction": enabled},
                },
                "a",
                0,
                Backend(),
                LongObservationTokenizer(),
                env,
                self.config(max_turns=3),
            )
            self.assertEqual(result.context_clears, 2)
            self.assertEqual(env.updates, 2 if enabled else 0)
