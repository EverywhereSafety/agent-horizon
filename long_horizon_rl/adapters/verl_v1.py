from long_horizon_rl.group_selection import CancellationLedger, select_attempts
from long_horizon_rl.cancellation import cancel_aware_client

"""Thin adapter for the pinned verl V1 AgentLoop/TransferQueue interfaces."""
import asyncio
import json
import time
import logging
import uuid
import os
from dataclasses import asdict
from pathlib import Path
import ray
import torch
import transfer_queue as tq
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    AgentLoopMetrics,
)
from verl.trainer.ppo.v1.agent_loop_tq import (
    AgentLoopWorkerTQ,
    AgentLoopManagerTQ,
    apply_greedy_sampling_params,
)
from verl.utils.tokenizer.continuous_token import QwenContinuousTokenBuilder
from long_horizon_rl.contracts import Config, Generation, EpisodeResult, Segment
from long_horizon_rl.episode import run_episode
from long_horizon_rl.environments import make_environment
from long_horizon_rl.continuation import ContinuationStore

logger = logging.getLogger(__name__)


def recovery_uid(record, fallback):
    path = os.environ.get("LONG_HORIZON_RECOVERY_GROUPS")
    if not path:
        return fallback
    return json.loads(Path(path).read_text()).get(record["prompt_uid"], fallback)


class LongHorizonOutput(AgentLoopOutput):
    def as_dict(self):
        output = super().as_dict()
        if "rm_scores" in output:
            indices = output["response_mask"].nonzero().flatten()
            if not len(indices):
                raise ValueError("empty assistant loss mask")
            output["rm_scores"].zero_()
            output["rm_scores"][indices[-1]] = self.reward_score
        return output


class QwenTokenizer:
    def __init__(self, tokenizer, tool_schemas=None, enable_thinking=False):
        self.tokenizer = tokenizer
        self.assistant_prefix = "<|im_start|>assistant\n" + (
            "<think>\n" if enable_thinking else ""
        )
        template_kwargs = {"enable_thinking": enable_thinking}
        self.tools = (
            [{"type": "function", "function": schema} for schema in tool_schemas]
            if tool_schemas
            else None
        )
        self.builder = QwenContinuousTokenBuilder(
            tokenizer, chat_template_kwargs=template_kwargs
        )
        self.last_generated = []

    def initial(self, messages):
        return self.builder.build_initial_tokens(messages, tools=self.tools)

    def memory(self, memory):
        # A runtime context message; branch on writes so training conditioning matches.
        if not memory:
            return []
        memory = {
            k: v
            for k, v in memory.items()
            if k not in ("archival", "notes") and not k.startswith("_")
        }
        if not memory:
            return []
        return self.tokenizer.encode(
            "<|im_end|>\n<|im_start|>system\nOperational memory: "
            + json.dumps(memory, ensure_ascii=False)
            + "<|im_end|>\n"
            + self.assistant_prefix,
            add_special_tokens=False,
        )

    def observation(self, text):
        return self.observations([text])

    def observations(self, texts):
        prefix = (
            "\n"
            if self.last_generated
            and self.last_generated[-1]
            == self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            else "<|im_end|>\n"
        )
        # Match the model's native template: consecutive tool results share a user block.
        body = (
            "<|im_start|>user"
            + "".join(
                "\n<tool_response>\n" + text + "\n</tool_response>" for text in texts
            )
            + "<|im_end|>\n"
        )
        return self.tokenizer.encode(
            prefix + body + self.assistant_prefix, add_special_tokens=False
        )


class VerlBackend:
    def __init__(self, client, tokenizer, sampling_params):
        self.client = cancel_aware_client(client, on_abort=self._record_abort)
        self.tokenizer = tokenizer
        self.params = sampling_params
        self.min_versions = []
        self.max_versions = []

    def _record_abort(self, receipt):
        path = Path(os.environ.get("LONG_HORIZON_AUDIT_DIR", "outputs/episodes")) / (
            self._trajectory_uid.replace("/", "_") + ".abort.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"trajectory_uid": self._trajectory_uid, **receipt}) + "\n"
        )
        tmp.replace(path)

    async def generate(self, ids, cap, version, uid, turn):
        self._trajectory_uid = uid
        params = dict(self.params)
        params["max_new_tokens"] = cap
        params["logprobs"] = True
        output = await self.client.generate(
            request_id=uid, prompt_ids=ids, sampling_params=params
        )
        minimum = output.extra_fields.get("min_global_steps")
        maximum = output.extra_fields.get("max_global_steps")
        if minimum is None or maximum is None:
            raise ValueError("serving version lineage required")
        self.min_versions.append(int(minimum))
        self.max_versions.append(int(maximum))
        self.tokenizer.last_generated = list(output.token_ids)
        return Generation(
            list(output.token_ids),
            list(output.log_probs),
            self.tokenizer.tokenizer.decode(
                output.token_ids, skip_special_tokens=False
            ),
            int(minimum),
            int(maximum),
        )


class LongHorizonAgentLoop(AgentLoopBase):
    def __init__(self, *args, episode_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_config = Config(**(episode_config or {}))

    async def run(self, sampling_params, **kwargs):
        record = json.loads(kwargs["record_json"])
        uid = f"{recovery_uid(record,kwargs['uid'])}/{kwargs['session_id']}"
        ledger = CancellationLedger(
            os.environ.get("LONG_HORIZON_AUDIT_DIR", "outputs/episodes")
        )
        ledger.check(uid)
        tokenizer = QwenTokenizer(
            self.tokenizer,
            record.get("tool_schemas"),
            enable_thinking=self.episode_config.enable_thinking,
        )
        backend = VerlBackend(self.server_manager, tokenizer, sampling_params)
        start = time.monotonic()
        audit_path = Path(
            os.environ.get("LONG_HORIZON_AUDIT_DIR", "outputs/episodes")
        ) / (uid.replace("/", "_") + ".jsonl")
        store = ContinuationStore(
            audit_path.with_suffix(".continuation.json"), record, self.episode_config
        )
        replay_path = audit_path.with_suffix(".replay.json")
        cached = None
        if replay_path.exists():
            cached = json.loads(replay_path.read_text())
            if cached["fingerprint"] != store.fingerprint:
                raise ValueError("completed replay inputs/config changed")
            state = cached["result"]
            state["segments"] = [Segment(**s) for s in state["segments"]]
            if "tool_error_count" not in state and audit_path.exists():
                with audit_path.open() as handle:
                    state["tool_error_count"] = sum(
                        bool(json.loads(line).get("observation", {}).get("error"))
                        for line in handle
                    )
            result = EpisodeResult(**state)
            if result.trajectory_uid != uid:
                raise ValueError("completed replay identity changed")
            minimum = cached["summary"]["min_served_version"]
            maximum = cached["summary"]["max_served_version"]
            logger.info(
                "Reused completed trajectory %s with original behavior versions %s..%s",
                uid,
                minimum,
                maximum,
            )
        else:
            env = await make_environment(record).start()
            try:
                if store.load() is None and audit_path.exists():
                    audit_path.unlink()
                result = await run_episode(
                    record,
                    uid,
                    int(kwargs["global_steps"]),
                    backend,
                    tokenizer,
                    env,
                    self.episode_config,
                    audit_path=audit_path,
                    continuation=store,
                )
            finally:
                await env.close()
            minimum = min(backend.min_versions + [result.min_served_version])
            maximum = max(backend.max_versions + [result.max_served_version])
        ledger.check(uid)
        result.terminal_metrics.setdefault("tool_error_count", result.tool_error_count)
        summary_path = audit_path.with_suffix(".result.json")
        summary_tmp = summary_path.with_suffix(".tmp")
        summary_tmp.write_text(
            json.dumps(
                {
                    "prompt_uid": record["prompt_uid"],
                    "trajectory_uid": uid,
                    "termination": result.termination,
                    "reward": result.reward,
                    "terminal_metrics": result.terminal_metrics,
                    "tool_error_count": result.tool_error_count,
                    "turns": result.turns,
                    "generated_tokens": result.generated_tokens,
                    "context_clears": result.context_clears,
                    "retained_segments": len(result.segments),
                    "max_retained_sequence_tokens": max(
                        len(s.prompt_ids) + len(s.response_ids) for s in result.segments
                    ),
                    "policy_tokens_retained": sum(
                        sum(s.response_mask) for s in result.segments
                    ),
                    "min_served_version": minimum,
                    "max_served_version": maximum,
                    "version_range_complete": result.version_range_complete,
                    "elapsed_seconds": time.monotonic() - start,
                    "audit_path": str(audit_path),
                },
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        if cached:
            summary_tmp.unlink()
        else:
            summary_tmp.replace(summary_path)
            replay_tmp = replay_path.with_suffix(".tmp")
            state = asdict(result)
            state["audit"] = []
            replay_tmp.write_text(
                json.dumps(
                    {
                        "fingerprint": store.fingerprint,
                        "summary": json.loads(summary_path.read_text()),
                        "result": state,
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
            replay_tmp.replace(replay_path)
        outputs = []
        for i, segment in enumerate(result.segments):
            if not any(segment.response_mask):
                continue
            outputs.append(
                LongHorizonOutput(
                    prompt_ids=segment.prompt_ids,
                    response_ids=segment.response_ids,
                    response_mask=segment.response_mask,
                    response_logprobs=segment.rollout_log_probs,
                    reward_score=result.reward,
                    num_turns=result.turns,
                    metrics=AgentLoopMetrics(
                        generate_sequences=time.monotonic() - start
                    ),
                    extra_fields={
                        "trajectory_uid": uid,
                        "segment_uid": f"{uid}/turn-{segment.turn_start}",
                        "audit_path": result.audit_path,
                        "termination": result.termination,
                        "min_global_steps": minimum,
                        "max_global_steps": maximum,
                        "generation_version_ranges": segment.generation_version_ranges,
                        "version_range_complete": result.version_range_complete,
                        "reward_extra_info": {
                            "context_clears": result.context_clears,
                            "terminal": result.termination == "terminal",
                            "reward": result.reward,
                            "submitted": float(result.termination == "terminal"),
                            "strict_success": result.terminal_metrics.get("solved"),
                            "placement_accuracy": result.terminal_metrics.get(
                                "placement_accuracy"
                            ),
                            "turns": result.turns,
                            "generated_tokens": result.generated_tokens,
                            "tool_errors": result.tool_error_count,
                        },
                    },
                )
            )
        if not outputs:
            raise ValueError("no policy tokens produced")
        return outputs


# Pinned upstream exposes this Ray-decorated worker class. Keep this access
# isolated here and guard it with an import test on every upstream change.
_Worker = AgentLoopWorkerTQ.__ray_metadata__.modified_class


@ray.remote
class GroupWorker(_Worker):
    def _compute_multi_modal_inputs(self, output, input_ids):
        if not output.multi_modal_data:
            return {}
        return super()._compute_multi_modal_inputs(output, input_ids)

    def _compute_position_ids(self, input_ids, attention_mask, multi_modal_inputs):
        media_keys = (
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "input_features",
        )
        if not any(key in multi_modal_inputs for key in media_keys):
            # Text-only Qwen expands these positions into its rotary axes itself.
            # Avoid the upstream 3D ragged-axis workaround corrupting TQ offsets.
            return (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)
        return super()._compute_position_ids(
            input_ids, attention_mask, multi_modal_inputs
        )

    async def _agent_loop_postprocess(self, output, validate, **kwargs):
        # Defer all publication until the logical prompt group has been selected.
        return output

    async def _run_prompt(self, prompt, sampling_params, trajectory, trace=False):
        uid = prompt["uid"]
        partition = "val" if trajectory["validate"] else "train"
        await tq.async_kv_put(
            key=uid, partition_id=partition, tag={"status": "running"}
        )
        try:
            return await self._run_prompt_selected(
                prompt, sampling_params, trajectory, trace
            )
        except (Exception, asyncio.CancelledError) as exc:
            logger.exception("Prompt preparation failed for %s", uid)
            await tq.async_kv_put(
                key=uid, partition_id=partition, tag={"status": "failure"}
            )
            if isinstance(exc, asyncio.CancelledError):
                raise

    async def _run_prompt_selected(
        self, prompt, sampling_params, trajectory, trace=False
    ):
        prompt = dict(prompt)
        uid = prompt["uid"]
        validate = trajectory["validate"]
        partition = "val" if validate else "train"
        retain = prompt.pop(
            "__rollout_n__",
            (
                self.config.actor_rollout_ref.rollout.val_kwargs.n
                if validate
                else self.config.actor_rollout_ref.rollout.n
            ),
        )
        from verl.experimental.agent_loop.agent_loop import _agent_loop_registry
        from long_horizon_rl.group_config import resolve_group_counts

        agent_name = prompt.get(
            "agent_name",
            getattr(
                getattr(self.config.actor_rollout_ref.rollout, "agent", None),
                "default_agent_loop",
                None,
            ),
        )
        agent_settings = _agent_loop_registry.get(agent_name, {}).get(
            "episode_config", {}
        )
        attempts, retain = resolve_group_counts(
            retain,
            agent_settings,
            validate=validate,
            distillation=self.config.distillation.enabled,
        )
        do_sample = prompt.pop("__do_sample__", True)
        run_sampling_params = dict(sampling_params)
        if not validate and not do_sample:
            apply_greedy_sampling_params(run_sampling_params)
        original_uid = recovery_uid(json.loads(prompt["record_json"]), uid)
        previous_group = Path(
            os.environ.get("LONG_HORIZON_AUDIT_DIR", "outputs/episodes")
        ) / (str(original_uid) + ".group.json")
        sessions = list(range(attempts))
        replay_order = None
        if not validate and previous_group.exists():
            group = json.loads(previous_group.read_text())
            if (
                group["retained"] != retain
                or group["attempts"] != attempts
                or group["prompt_uid"]
                != json.loads(prompt["record_json"])["prompt_uid"]
            ):
                raise ValueError("completed group contract changed")
            replay_order = group["selected_trajectories"]
            if len(set(replay_order)) != retain or any(
                not u.startswith(str(original_uid) + "/") for u in replay_order
            ):
                raise ValueError("invalid completed group identities")
            sessions = [int(u.rsplit("/", 1)[1]) for u in replay_order]
            for session in sessions:
                cache = previous_group.parent / (
                    f"{original_uid}_{session}.replay.json"
                )
                if not cache.exists():
                    raise ValueError("selected completed trajectory cache missing")
        ledger = CancellationLedger(previous_group.parent)
        candidate_uids = [f"{original_uid}/{i}" for i in sessions]

        async def factory(candidate_uid):
            session = int(candidate_uid.rsplit("/", 1)[1])
            return await self._run_agent_loop(
                dict(run_sampling_params),
                trajectory=trajectory,
                trace=trace,
                session_id=session,
                **prompt,
            )

        def valid(outputs):
            return bool(outputs) and outputs[-1].extra_fields["termination"] in (
                "terminal",
                "turn_limit",
                "generation_limit",
            )

        try:
            selected = await select_attempts(
                factory, candidate_uids, retain, valid, ledger
            )
            if replay_order:
                selected.sort(
                    key=lambda outputs: replay_order.index(
                        outputs[-1].extra_fields["trajectory_uid"]
                    )
                )
            if self.config.distillation.enabled and not validate:
                from .opd_scoring import score_earlier_segments

                await score_earlier_segments(self, selected[:retain], prompt)
            for session, outputs in enumerate(selected[:retain]):
                ledger.check(outputs[-1].extra_fields["trajectory_uid"])
                await super()._agent_loop_postprocess(
                    outputs, validate, session_id=session, **prompt
                )
            group_path = Path(
                os.environ.get("LONG_HORIZON_AUDIT_DIR", "outputs/episodes")
            ) / (str(uid).replace("/", "_") + ".group.json")
            group_path.parent.mkdir(parents=True, exist_ok=True)
            group_tmp = group_path.with_suffix(".tmp")
            group_tmp.write_text(
                json.dumps(
                    {
                        "uid": uid,
                        "prompt_uid": json.loads(prompt["record_json"])["prompt_uid"],
                        "trajectory_group_uid": original_uid,
                        "reused_group": replay_order is not None,
                        "attempts": attempts,
                        "dispatched_attempts": len(sessions),
                        "retained": retain,
                        "partition": partition,
                        "selected_trajectories": [
                            o[-1].extra_fields["trajectory_uid"]
                            for o in selected[:retain]
                        ],
                        "cancelled_trajectories": [
                            candidate_uid
                            for candidate_uid in candidate_uids
                            if candidate_uid
                            not in {
                                o[-1].extra_fields["trajectory_uid"]
                                for o in selected[:retain]
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            )
            group_tmp.replace(group_path)
            await tq.async_kv_put(
                key=uid, partition_id=partition, tag={"status": "finished"}
            )
        except (Exception, asyncio.CancelledError) as exc:
            for candidate_uid in candidate_uids:
                ledger.mark(candidate_uid, "group_failed_or_cancelled")
            logger.exception("Prompt group failed for %s", uid)
            await tq.async_kv_put(
                key=uid, partition_id=partition, tag={"status": "failure"}
            )
            if isinstance(exc, asyncio.CancelledError):
                raise


class AgentLoopManager(AgentLoopManagerTQ):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = GroupWorker
        # AgentLoopManagerTQ overwrites the worker class, so call its parent directly.
        from verl.experimental.agent_loop.agent_loop import (
            AgentLoopManager as BaseManager,
        )

        BaseManager.__init__(self, *args, **kwargs)
