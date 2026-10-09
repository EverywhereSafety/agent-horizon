"""Rebuild completed training segments from token-faithful episode audits."""

import json
from dataclasses import asdict
from pathlib import Path
from long_horizon_rl.contracts import Generation
from long_horizon_rl.episode import run_episode
from long_horizon_rl.parser import parse_actions


class AuditReplay:
    def __init__(self, path, tokenizer):
        self.file = Path(path).open()
        self.tokenizer = tokenizer
        self.row = None
        self.call_index = 0

    async def generate(self, ids, cap, version, uid, turn):
        line = self.file.readline()
        if not line:
            raise RuntimeError("incomplete episode audit")
        self.row = json.loads(line)
        self.call_index = 0
        if self.row["turn"] != turn or self.row["input_ids"] != ids:
            raise RuntimeError("audit prompt/tokenizer/config mismatch")
        self.tokenizer.last_generated = list(self.row["output_ids"])
        generation = Generation(
            self.row["output_ids"],
            self.row["log_probs"],
            self.row["action_text"],
            self.row["served_version"],
            self.row.get("max_served_version"),
        )
        generation.validate(cap)
        return generation

    def step(self, action):
        observation = self.row["observation"]
        if "tool_results" not in observation:
            return observation
        actions = parse_actions(self.row["action_text"])
        results = [
            reply
            for action, reply in zip(actions, observation["tool_results"])
            if action.get("tool")
            not in ("memory_write", "memory_read", "memory", "operational_memory")
        ]
        result = results[self.call_index]
        self.call_index += 1
        return result

    def close(self):
        pass


async def rebuild_completed(record, config, tokenizer, audit_path, summary):
    replay = AuditReplay(audit_path, tokenizer)
    try:
        result = await run_episode(
            record,
            summary["trajectory_uid"],
            summary["min_served_version"],
            replay,
            tokenizer,
            replay,
            config,
        )
        if replay.file.readline():
            raise RuntimeError("audit has unused rows")
        for field in (
            "reward",
            "termination",
            "turns",
            "generated_tokens",
            "context_clears",
        ):
            if getattr(result, field) != summary[field]:
                raise RuntimeError(f"audit replay differs in {field}")
        if len(result.segments) != summary["retained_segments"]:
            raise RuntimeError("audit replay differs in retained segments")
        result.audit_path = str(audit_path)
        # The audit is already durable; do not duplicate its full contents in caches.
        result.audit = []
        return result
    finally:
        replay.file.close()
