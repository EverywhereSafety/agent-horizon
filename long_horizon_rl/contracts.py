from dataclasses import dataclass, field
import math
from typing import List, Dict, Optional


@dataclass
class Config:
    max_context_tokens: int = 131072
    clear_trigger_tokens: int = 98304
    clear_target_tokens: int = 65536
    max_new_tokens: int = 16384
    max_turns: int = 1000
    # Production retain count is authoritative in native rollout.n. Optional
    # legacy retain_n is a consistency assertion; attempt_n sets the budget.
    attempt_n: Optional[int] = None
    retain_n: Optional[int] = None
    max_policy_lag: int = 3
    max_generated_tokens: Optional[int] = None
    episode_timeout_seconds: Optional[float] = None
    enable_thinking: bool = False
    context_feedback_enabled: bool = False
    context_warning_margin_tokens: Optional[int] = None
    context_memory_title_limit: int = 32
    memory_embedding_model: Optional[str] = None
    memory_embedding_revision: Optional[str] = None
    memory_embedding_max_tokens: int = 256

    def __post_init__(self):
        if bool(self.memory_embedding_model) != bool(self.memory_embedding_revision):
            raise ValueError(
                "embedding model and immutable revision must be configured together"
            )
        if self.memory_embedding_revision is not None and (
            len(self.memory_embedding_revision) != 40
            or any(c not in "0123456789abcdef" for c in self.memory_embedding_revision)
        ):
            raise ValueError("invalid embedding revision")
        if (
            type(self.memory_embedding_max_tokens) is not int
            or self.memory_embedding_max_tokens < 1
        ):
            raise ValueError("invalid embedding token limit")
        if type(self.enable_thinking) is not bool:
            raise ValueError("enable_thinking must be boolean")
        if type(self.context_feedback_enabled) is not bool:
            raise ValueError("context_feedback_enabled must be boolean")
        if self.context_warning_margin_tokens is not None and (
            type(self.context_warning_margin_tokens) is not int
            or not 0 < self.context_warning_margin_tokens < self.clear_trigger_tokens
        ):
            raise ValueError("invalid context warning margin")
        if (
            type(self.context_memory_title_limit) is not int
            or self.context_memory_title_limit < 0
        ):
            raise ValueError("invalid memory title limit")
        if self.episode_timeout_seconds is not None and (
            isinstance(self.episode_timeout_seconds, bool)
            or not isinstance(self.episode_timeout_seconds, (int, float))
            or not math.isfinite(self.episode_timeout_seconds)
            or self.episode_timeout_seconds <= 0
        ):
            raise ValueError("invalid episode deadline")
        if self.max_generated_tokens is not None and (
            type(self.max_generated_tokens) is not int or self.max_generated_tokens <= 0
        ):
            raise ValueError("invalid cumulative generation limit")
        if (
            not 0
            < self.clear_target_tokens
            < self.clear_trigger_tokens
            < self.max_context_tokens
        ):
            raise ValueError("require target < trigger < context limit")
        if (
            not 0
            < self.max_new_tokens
            <= self.max_context_tokens - self.clear_trigger_tokens
        ):
            raise ValueError("generation budget must fit above clear trigger")
        for value in (self.attempt_n, self.retain_n):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError("group sizes must be positive integers")
        if (
            self.attempt_n is not None
            and self.retain_n is not None
            and self.retain_n > self.attempt_n
        ):
            raise ValueError("retain count exceeds attempt budget")
        if self.max_turns <= 0 or self.max_policy_lag < 0:
            raise ValueError("invalid group, turn or lag limit")


@dataclass
class Generation:
    token_ids: List[int]
    log_probs: List[float]
    text: str
    served_version: int
    # Inclusive range for this generate call, not per-token exact provenance.
    max_served_version: Optional[int] = None

    def validate(self, cap):
        if len(self.token_ids) != len(self.log_probs) or len(self.token_ids) > cap:
            raise ValueError("unaligned or over-budget generation")
        if (
            self.max_served_version is not None
            and self.max_served_version < self.served_version
        ):
            raise ValueError("invalid generation version range")
        if (
            self.served_version < 0
            or not self.token_ids
            or not all(math.isfinite(x) for x in self.log_probs)
        ):
            raise ValueError("invalid generation provenance")


@dataclass
class Segment:
    prompt_ids: List[int]
    response_ids: List[int] = field(default_factory=list)
    response_mask: List[int] = field(default_factory=list)
    rollout_log_probs: List[float] = field(default_factory=list)
    served_versions: List[int] = field(default_factory=list)
    turn_start: int = 0
    generation_version_ranges: List[Dict] = field(default_factory=list)

    def append(self, ids, mask, probs, version, max_version=None):
        start = len(self.response_ids)
        if mask and ids:
            self.generation_version_ranges.append(
                {
                    "response_start": start,
                    "response_end": start + len(ids),
                    "min_served_version": version,
                    "max_served_version": (
                        version if max_version is None else max_version
                    ),
                }
            )
        self.response_ids.extend(ids)
        self.response_mask.extend([mask] * len(ids))
        self.rollout_log_probs.extend(probs)
        self.served_versions.extend([version] * len(ids))

    def validate(self):
        n = len(self.response_ids)
        if (
            not n
            == len(self.response_mask)
            == len(self.rollout_log_probs)
            == len(self.served_versions)
        ):
            raise ValueError("segment alignment failure")
        previous_end = 0
        for item in self.generation_version_ranges:
            start, end = item["response_start"], item["response_end"]
            minimum, maximum = item["min_served_version"], item["max_served_version"]
            if not previous_end <= start < end <= n or not 0 <= minimum <= maximum:
                raise ValueError("invalid generation version range")
            if any(
                not self.response_mask[i] or self.served_versions[i] != minimum
                for i in range(start, end)
            ):
                raise ValueError("version range must cover its sampled policy tokens")
            previous_end = end
        if any(x not in (0, 1) for x in self.response_mask):
            raise ValueError("invalid loss mask")
        if not all(math.isfinite(x) for x in self.rollout_log_probs):
            raise ValueError("nonfinite behavior log probability")


@dataclass
class EpisodeResult:
    prompt_uid: str
    trajectory_uid: str
    dispatch_version: int
    segments: List[Segment]
    reward: float
    termination: str
    turns: int
    context_clears: int
    audit: List[Dict]
    memory: Dict[str, str]
    audit_path: Optional[str] = None
    generated_tokens: int = 0
    terminal_metrics: Dict = field(default_factory=dict)
    tool_error_count: int = 0

    @property
    def min_served_version(self):
        return min(
            (
                v
                for s in self.segments
                for v, m in zip(s.served_versions, s.response_mask)
                if m
            ),
            default=self.dispatch_version,
        )

    @property
    def max_served_version(self):
        upper = [
            item["max_served_version"]
            for s in self.segments
            for item in s.generation_version_ranges
        ]
        lower = [
            v
            for s in self.segments
            for v, m in zip(s.served_versions, s.response_mask)
            if m
        ]
        return max(upper + lower, default=self.dispatch_version)

    @property
    def version_range_complete(self):
        return all(
            sum(
                r["response_end"] - r["response_start"]
                for r in s.generation_version_ranges
            )
            == sum(s.response_mask)
            for s in self.segments
        )
