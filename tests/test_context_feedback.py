import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from long_horizon_rl.context_feedback import memory_title_preview
from long_horizon_rl.contracts import Config, Generation
from long_horizon_rl.continuation import ContinuationStore
from long_horizon_rl.episode import run_episode
from examples.smoke import CounterEnvironment
from examples.smoke import FixtureTokenizer


class NotesTokenizer(FixtureTokenizer):
    def memory(self, memory):
        return []  # Note bodies are read through tools, never injected.


class NotesBackend:
    def __init__(self, interrupt=None):
        self.interrupt = interrupt

    async def generate(self, ids, cap, version, uid, turn):
        if turn == self.interrupt:
            raise RuntimeError("injected interruption")
        if turn == 0:
            action = {
                "tool": "memory",
                "arguments": {
                    "action": "write",
                    "title": "solver",
                    "content": "private-note-body",
                },
            }
        elif turn == 2:
            action = {
                "tool": "memory",
                "arguments": {"action": "read", "title": "solver"},
            }
        else:
            action = {"tool": "advance", "amount": 1}
        text = json.dumps(action)
        return Generation(list(map(ord, text)), [-0.3] * len(text), text, version)


class FeedbackTests(unittest.IsolatedAsyncioTestCase):
    def config(self):
        return Config(
            max_context_tokens=4096,
            clear_trigger_tokens=1000,
            clear_target_tokens=700,
            max_new_tokens=256,
            context_feedback_enabled=True,
            context_warning_margin_tokens=400,
        )

    def record(self):
        return {
            "prompt_uid": "feedback",
            "messages": [
                {"role": "system", "content": "Rules."},
                {"role": "user", "content": "Reach eight."},
            ],
        }

    async def test_notices_preserve_exact_prefixes_and_all_policy_tokens(self):
        record = self.record()
        original = json.dumps(record)
        result = await run_episode(
            record,
            "a",
            0,
            NotesBackend(),
            NotesTokenizer(),
            CounterEnvironment(8),
            self.config(),
        )
        self.assertEqual(json.dumps(record), original)
        self.assertGreater(result.context_clears, 0)
        events = [e for row in result.audit for e in row["context_events"]]
        self.assertTrue(any(e["type"] == "warning" for e in events))
        self.assertTrue(any(e["type"] == "clear" for e in events))
        self.assertEqual(
            sum(sum(s.response_mask) for s in result.segments), result.generated_tokens
        )
        self.assertEqual(result.memory["notes"]["solver"], "private-note-body")
        for row in result.audit:
            segment = max(
                (s for s in result.segments if s.turn_start <= row["turn"]),
                key=lambda s: s.turn_start,
            )
            offset = len(row["input_ids"]) - len(segment.prompt_ids)
            self.assertEqual(
                segment.prompt_ids + segment.response_ids[:offset], row["input_ids"]
            )
            end = offset + len(row["output_ids"])
            self.assertEqual(segment.response_ids[offset:end], row["output_ids"])
            self.assertEqual(
                segment.response_mask[offset:end], [1] * len(row["output_ids"])
            )
            self.assertEqual(segment.rollout_log_probs[offset:end], row["log_probs"])
            self.assertLessEqual(len(row["input_ids"]) + len(row["output_ids"]), 4096)
        for segment in result.segments:
            self.assertTrue(
                all(
                    p == 0
                    for p, mask in zip(segment.rollout_log_probs, segment.response_mask)
                    if not mask
                )
            )
            # Generated notices live in conditioning, never in the sampled span.
            system_text = "".join(map(chr, segment.prompt_ids)).split('"role": "user"')[
                0
            ]
            self.assertNotIn("private-note-body", system_text)
            policy_text = "".join(
                chr(t) for t, m in zip(segment.response_ids, segment.response_mask) if m
            )
            self.assertNotIn("Context clear:", policy_text)
            self.assertNotIn("Context budget reminder:", policy_text)

    async def test_notice_and_memory_state_resume_matches_uninterrupted_run(self):
        record = self.record()
        with tempfile.TemporaryDirectory() as directory:
            store = ContinuationStore(
                Path(directory) / "state.json", record, self.config()
            )
            audit = Path(directory) / "audit.jsonl"
            with self.assertRaises(RuntimeError):
                await run_episode(
                    record,
                    "a",
                    0,
                    NotesBackend(interrupt=6),
                    NotesTokenizer(),
                    CounterEnvironment(8),
                    self.config(),
                    audit_path=audit,
                    continuation=store,
                    checkpoint_interval=2,
                )
            self.assertIn("context_feedback", store.load())
            resumed = await run_episode(
                record,
                "a",
                0,
                NotesBackend(),
                NotesTokenizer(),
                CounterEnvironment(8),
                self.config(),
                audit_path=audit,
                continuation=store,
                checkpoint_interval=2,
            )
            reference = await run_episode(
                record,
                "a",
                0,
                NotesBackend(),
                NotesTokenizer(),
                CounterEnvironment(8),
                self.config(),
            )
            self.assertEqual(
                [json.loads(line) for line in audit.read_text().splitlines()],
                reference.audit,
            )
            self.assertEqual(asdict(resumed)["segments"], asdict(reference)["segments"])
            self.assertEqual(resumed.memory, reference.memory)

    def test_disabled_feature_preserves_existing_checkpoint_fingerprint(self):
        config = Config()
        old = asdict(config)
        old.update(attempt_n=18, retain_n=16)
        for key in (
            "enable_thinking",
            "episode_timeout_seconds",
            "context_feedback_enabled",
            "context_warning_margin_tokens",
            "context_memory_title_limit",
            "memory_embedding_model",
            "memory_embedding_revision",
            "memory_embedding_max_tokens",
        ):
            old.pop(key)
        payload = {"record": self.record(), "config": old}
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        self.assertEqual(
            ContinuationStore("unused", self.record(), config).fingerprint, expected
        )

    def test_title_preview_does_not_disclose_bodies_and_has_a_bound(self):
        preview = memory_title_preview({"notes": {"b": "secret", "a": "secret2"}}, 1)
        self.assertEqual(preview, {"titles": ["a"], "omitted": 1})
