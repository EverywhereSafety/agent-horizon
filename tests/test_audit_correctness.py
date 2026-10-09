import json, tempfile, unittest
from dataclasses import asdict
from pathlib import Path
from long_horizon_rl.contracts import Config, Generation, Segment
from long_horizon_rl.continuation import ContinuationStore
from long_horizon_rl.episode import run_episode
from examples.smoke import FixtureTokenizer
from long_horizon_rl.group_config import resolve_group_counts
from long_horizon_rl.diagnostics.audit_replay import rebuild_completed


class FaultBackend:
    def __init__(self, interrupt=None):
        self.interrupt = interrupt

    async def generate(self, ids, cap, version, uid, turn):
        if turn == self.interrupt:
            raise RuntimeError("crash")
        # The earlier request crosses versions; later requests have a lower max.
        return Generation([10], [-0.3], "invalid action", 2, 9 if turn == 0 else 3)


class ErrorEnvironment:
    def snapshot(self):
        return {}

    def restore(self, state):
        pass

    def close(self):
        pass


class AuditCorrectness(unittest.IsolatedAsyncioTestCase):
    def config(self):
        return Config(
            max_context_tokens=2048,
            clear_trigger_tokens=1000,
            clear_target_tokens=600,
            max_new_tokens=64,
            max_turns=3,
        )

    async def test_streamed_error_count_and_version_range_survive_resume(self):
        record = {"prompt_uid": "p", "messages": []}
        cfg = self.config()
        with tempfile.TemporaryDirectory() as d:
            audit = Path(d) / "audit.jsonl"
            store = ContinuationStore(Path(d) / "state.json", record, cfg)
            with self.assertRaisesRegex(RuntimeError, "crash"):
                await run_episode(
                    record,
                    "p/0",
                    2,
                    FaultBackend(2),
                    FixtureTokenizer(),
                    ErrorEnvironment(),
                    cfg,
                    audit_path=audit,
                    continuation=store,
                    checkpoint_interval=1,
                )
            saved = store.load()
            self.assertEqual(saved["tool_error_count"], 2)
            self.assertEqual(
                saved["segments"][0]["generation_version_ranges"][0][
                    "max_served_version"
                ],
                9,
            )
            resumed = await run_episode(
                record,
                "p/0",
                2,
                FaultBackend(),
                FixtureTokenizer(),
                ErrorEnvironment(),
                cfg,
                audit_path=audit,
                continuation=store,
                checkpoint_interval=1,
            )
            reference = await run_episode(
                record,
                "p/0",
                2,
                FaultBackend(),
                FixtureTokenizer(),
                ErrorEnvironment(),
                cfg,
            )
            self.assertEqual(resumed.audit, [])
            self.assertEqual(resumed.tool_error_count, 3)
            self.assertEqual(reference.tool_error_count, 3)
            self.assertEqual(
                (resumed.min_served_version, resumed.max_served_version), (2, 9)
            )
            self.assertTrue(resumed.version_range_complete)
            self.assertEqual(
                [asdict(s) for s in resumed.segments],
                [asdict(s) for s in reference.segments],
            )
            summary = {
                k: getattr(resumed, k)
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
                min_served_version=2, retained_segments=len(resumed.segments)
            )
            replay = await rebuild_completed(
                record, cfg, FixtureTokenizer(), audit, summary
            )
            self.assertEqual(
                (replay.tool_error_count, replay.max_served_version), (3, 9)
            )

    async def test_legacy_count_recovers_only_committed_audit_rows(self):
        record = {"prompt_uid": "p", "messages": []}
        cfg = self.config()
        with tempfile.TemporaryDirectory() as d:
            audit = Path(d) / "audit.jsonl"
            store = ContinuationStore(Path(d) / "state.json", record, cfg)
            with self.assertRaises(RuntimeError):
                await run_episode(
                    record,
                    "p/0",
                    2,
                    FaultBackend(1),
                    FixtureTokenizer(),
                    ErrorEnvironment(),
                    cfg,
                    audit_path=audit,
                    continuation=store,
                    checkpoint_interval=1,
                )
            saved = store.load()
            saved.pop("tool_error_count")
            store.save(saved)
            with audit.open("a") as handle:
                handle.write(
                    json.dumps({"observation": {"error": "uncommitted"}}) + "\n"
                )
            result = await run_episode(
                record,
                "p/0",
                2,
                FaultBackend(),
                FixtureTokenizer(),
                ErrorEnvironment(),
                cfg,
                audit_path=audit,
                continuation=store,
                checkpoint_interval=1,
            )
            self.assertEqual(result.tool_error_count, 3)
            self.assertEqual(len(audit.read_text().splitlines()), 3)

    def test_generation_range_rejects_inversion(self):
        with self.assertRaises(ValueError):
            Generation([1], [-0.1], "x", 3, 2).validate(1)
        segment = Segment([1])
        segment.append([2], 1, [-0.1], 2, 9)
        segment.validate()
        segment.generation_version_ranges[0]["response_end"] = 2
        with self.assertRaises(ValueError):
            segment.validate()


class GroupConfiguration(unittest.TestCase):
    def test_budget_is_used_and_native_retain_is_authoritative(self):
        self.assertEqual(resolve_group_counts(16), (18, 16))
        self.assertEqual(resolve_group_counts(2, {"attempt_n": 7}), (7, 2))
        Config(attempt_n=7)
        for settings in ({"retain_n": 3}, {"attempt_n": 1}, {"attempt_n": True}):
            with self.assertRaises(ValueError):
                resolve_group_counts(2, settings)
        self.assertEqual(
            resolve_group_counts(1, {"attempt_n": 7, "retain_n": 16}, validate=True),
            (1, 1),
        )
        self.assertEqual(
            resolve_group_counts(2, {"attempt_n": 7}, distillation=True), (2, 2)
        )
