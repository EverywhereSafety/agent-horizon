"""Atomic, model-independent episode continuation artifacts."""

import hashlib
import json
import os
from pathlib import Path


class ContinuationStore:
    def __init__(self, path, record, config):
        self.path = Path(path)
        from dataclasses import asdict

        settings = asdict(config)
        # Group sizes were previously unused default episode fields. Preserve
        # fingerprints for unchanged production configurations; resolved group
        # contracts are checked separately before dispatch/replay.
        if settings.get("attempt_n") is None:
            settings["attempt_n"] = 18
        if settings.get("retain_n") is None:
            settings["retain_n"] = 16
        # Preserve compatibility with existing no-deadline continuations.
        if settings.get("episode_timeout_seconds") is None:
            settings.pop("episode_timeout_seconds", None)
        # A disabled additive feature must not invalidate existing checkpoints.
        if not settings.get("context_feedback_enabled", False):
            for key in (
                "context_feedback_enabled",
                "context_warning_margin_tokens",
                "context_memory_title_limit",
            ):
                settings.pop(key, None)
        if not settings.get("enable_thinking", False):
            settings.pop("enable_thinking", None)
        if settings.get("memory_embedding_model") is None:
            for key in (
                "memory_embedding_model",
                "memory_embedding_revision",
                "memory_embedding_max_tokens",
            ):
                settings.pop(key, None)
        payload = {"record": record, "config": settings}
        self.fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()

    def load(self):
        if not self.path.exists():
            return None
        payload = json.loads(self.path.read_text())
        if payload["fingerprint"] != self.fingerprint:
            raise ValueError("episode inputs/config changed across resume")
        return payload["state"]

    def save(self, state):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w") as handle:
            json.dump(
                {"fingerprint": self.fingerprint, "state": state},
                handle,
                allow_nan=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)

    def complete(self):
        if self.path.exists():
            self.path.unlink()
