import asyncio
import copy
import tempfile
import unittest
from pathlib import Path
from long_horizon_rl.contracts import Config, Generation, Segment, EpisodeResult
from examples.smoke import CounterEnvironment
from long_horizon_rl.episode import run_episode
from long_horizon_rl.groups import select_group, pack_group
from long_horizon_rl.replay import VersionedReplay
from examples.smoke import FixtureBackend, FixtureTokenizer


def result(uid="a", prompt="p", version=0, reward=1):
    s = Segment([1])
    s.append([2], 1, [-0.5], version)
    s.append([3], 0, [0.0], -1)
    return EpisodeResult(prompt, uid, version, [s], reward, "terminal", 1, 0, [], {})


class Contracts(unittest.TestCase):
    def test_configuration_headroom(self):
        with self.assertRaises(ValueError):
            Config(max_new_tokens=40000)

    def test_provenance_rejects_misaligned_nonfinite(self):
        for g in (Generation([1], [], "", 0), Generation([1], [float("nan")], "", 0)):
            with self.assertRaises(ValueError):
                g.validate(2)

    def test_trajectory_baseline_and_mask(self):
        a, b = result(reward=1), result("b", reward=0)
        a.segments.append(copy.deepcopy(a.segments[0]))
        rows = pack_group([a, b])
        self.assertEqual(
            [r["advantages"] for r in rows], [[0.5, 0.0], [0.5, 0.0], [-0.5, 0.0]]
        )
        self.assertEqual(rows[0]["trajectory_policy_tokens"], 2)
        self.assertEqual(len({r["segment_uid"] for r in rows}), 3)

    def test_mixed_and_duplicate_groups(self):
        for group in ([result(), result("b", "q")], [result(), result()]):
            with self.assertRaises(ValueError):
                pack_group(group)

    def test_staleness_boundary_and_resume(self):
        replay = VersionedReplay()
        replay.add([result()])
        for v in range(1, 4):
            replay.publish_version(v, ["r"], {"r": v})
        with tempfile.TemporaryDirectory() as d:
            replay.save(Path(d) / "r.json")
            replay = VersionedReplay.load(Path(d) / "r.json")
        self.assertIsNotNone(replay.take())
        with self.assertRaises(ValueError):
            replay.add([result()])
        replay = VersionedReplay()
        replay.add([result()])
        for v in range(1, 5):
            replay.publish_version(v, ["r"], {"r": v})
        self.assertIsNone(replay.take())

    def test_weight_publication_transaction(self):
        replay = VersionedReplay()
        with self.assertRaises(ValueError):
            replay.publish_version(1, ["a", "b"], {"a": 1})
        self.assertEqual(replay.version, 0)
        replay.publish_version(1, ["a", "b"], {"a": 1, "b": 1})
        self.assertEqual(replay.version, 1)

    def test_future_served_version_rejected(self):
        replay = VersionedReplay()
        with self.assertRaises(ValueError):
            replay.add([result(version=1)])


class AsyncContracts(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_acknowledgement(self):
        closed = []

        async def factory(i):
            try:
                if i == 2:
                    await asyncio.Event().wait()
                return result(str(i))
            finally:
                closed.append(i)

        group, metrics = await select_group(factory, 3, 2)
        self.assertEqual(len(group), 2)
        self.assertEqual(metrics["cancelled"], 1)
        self.assertEqual(set(closed), {0, 1, 2})

    async def test_failure_shortage(self):
        async def factory(i):
            raise RuntimeError("injected")

        with self.assertRaises(RuntimeError):
            await select_group(factory, 3, 2)

    async def test_backend_failure_cleans_environment(self):
        class Failed:
            async def generate(self, *args):
                raise RuntimeError("server down")

        env = CounterEnvironment()
        with self.assertRaises(RuntimeError):
            await run_episode(
                {"prompt_uid": "p", "messages": []},
                "a",
                0,
                Failed(),
                FixtureTokenizer(),
                env,
                Config(),
            )
        self.assertTrue(env.closed)

    async def test_context_clear_memory_and_audit(self):
        cfg = Config(
            max_context_tokens=2048,
            clear_trigger_tokens=300,
            clear_target_tokens=180,
            max_new_tokens=128,
        )
        env = CounterEnvironment(9)
        r = await run_episode(
            {"prompt_uid": "p", "messages": []},
            "a-0",
            0,
            FixtureBackend(),
            FixtureTokenizer(),
            env,
            cfg,
        )
        self.assertEqual(r.termination, "terminal")
        self.assertGreaterEqual(r.context_clears, 2)
        self.assertEqual(r.memory["goal"], "nine")
        self.assertEqual(len(r.audit), r.turns)
        self.assertTrue(env.closed)
        for s in r.segments:
            self.assertLessEqual(
                len(s.prompt_ids) + len(s.response_ids), cfg.max_context_tokens
            )

    async def test_malformed_action_is_a_masked_observation(self):
        class Malformed:
            async def generate(self, ids, cap, version, uid, turn):
                return Generation([1], [-0.2], "bad json", version)

        r = await run_episode(
            {"prompt_uid": "p", "messages": [], "agent_info": {"max_turn": 1}},
            "a",
            0,
            Malformed(),
            FixtureTokenizer(),
            CounterEnvironment(),
            Config(),
        )
        self.assertEqual(r.termination, "turn_limit")
        self.assertIn("error", r.audit[0]["observation"])
        self.assertEqual(sum(r.segments[0].response_mask), 1)

    async def test_record_turn_limit(self):
        r = await run_episode(
            {"prompt_uid": "p", "messages": [], "agent_info": {"max_turn": 1}},
            "a-0",
            0,
            FixtureBackend(),
            FixtureTokenizer(),
            CounterEnvironment(),
            Config(),
        )
        self.assertEqual((r.turns, r.termination, r.reward), (1, "turn_limit", 0.0))


if __name__ == "__main__":
    unittest.main()
