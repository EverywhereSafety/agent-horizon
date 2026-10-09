import asyncio
import time
import json
import math
import inspect
from .parser import parse_actions
from .memory import operate, operate_notes
from .context_feedback import warning_notice, clear_notice
from pathlib import Path
from dataclasses import asdict
from .contracts import Segment, EpisodeResult


async def _run_episode(
    record,
    trajectory_uid,
    dispatch_version,
    backend,
    tokenizer,
    env,
    config,
    audit_path=None,
    continuation=None,
    checkpoint_interval=10,
):
    """Backend generates exact IDs + probabilities; text is only a tool parser sidecar.

    Tokenizer must provide initial(messages), observation(text), memory(dict).
    Complete tool groups are removable; immutable prefix and operational memory survive.
    Every clear starts a new train segment; old tokens stay in the audit/segments.
    """
    started = time.monotonic()
    elapsed_before = 0.0
    prefix = tokenizer.initial(record["messages"])
    feedback = {"notice": "", "warning_sent": False}

    def set_notice(text):
        nonlocal prefix, segment
        messages = [dict(message) for message in record["messages"]]
        if not messages or messages[0].get("role") != "system":
            messages.insert(
                0, {"role": "system", "content": "Context management information."}
            )
        messages[0]["content"] += "\n" + text
        prefix = tokenizer.initial(messages)
        feedback["notice"] = text
        # A changed prefix is new conditioning; preserve the preceding row.
        segment = None

    memory = {}
    memory_encoder = None
    if config.memory_embedding_model:
        from .vector_memory import local_encoder

        memory_encoder = local_encoder(
            config.memory_embedding_model,
            config.memory_embedding_revision,
            config.memory_embedding_max_tokens,
        )
    groups = []
    audit = []
    segments = []
    clears = 0
    reward = 0.0
    reason = "turn_limit"
    turns = 0
    generated_tokens = 0
    tool_error_count = 0
    terminal_metrics = {}
    max_turns = record.get("agent_info", {}).get("max_turn", config.max_turns)
    if type(max_turns) is not int or max_turns <= 0:
        raise ValueError("invalid record max_turn")
    max_turns = min(max_turns, config.max_turns)

    def prompt():
        return prefix + tokenizer.memory(memory) + [t for g in groups for t in g]

    segment = None
    start_turn = 0
    restored = continuation.load() if continuation is not None else None
    if restored:
        elapsed_before = restored.get("elapsed_seconds", 0.0)
        memory = restored["memory"]
        groups = restored["groups"]
        segments = [Segment(**s) for s in restored["segments"]]
        segment = segments[-1] if restored["segment_active"] and segments else None
        audit = restored["audit"]
        clears = restored["clears"]
        turns = restored["turns"]
        generated_tokens = restored.get(
            "generated_tokens", sum(len(row["output_ids"]) for row in audit)
        )
        tool_error_count = restored.get("tool_error_count")
        if tool_error_count is None:
            rows = audit
            if audit_path is not None and Path(audit_path).exists():
                # Old continuations lack the independent count; recover only
                # committed rows, excluding any post-checkpoint file tail.
                rows = []
                with Path(audit_path).open() as handle:
                    offset = restored.get("audit_offset", 0)
                    while handle.tell() < offset:
                        line = handle.readline()
                        if not line:
                            raise ValueError("incomplete committed audit")
                        rows.append(json.loads(line))
            tool_error_count = sum(
                bool(reply.get("error"))
                for row in rows
                for reply in row.get("observation", {}).get(
                    "tool_results", [row.get("observation", {})]
                )
            )
        start_turn = restored["next_turn"]
        if config.context_feedback_enabled:
            feedback.update(restored.get("context_feedback", {}))
            if feedback["notice"]:
                active_segment = segment
                set_notice(feedback["notice"])
                segment = active_segment
        state_restore = env.restore(restored["environment"])
        if inspect.isawaitable(state_restore):
            await state_restore
    audit_file = None
    try:
        if audit_path is not None:
            path = Path(audit_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            audit_file = path.open("r+" if restored and path.exists() else "x")
            if restored:
                audit_file.seek(restored["audit_offset"])
                audit_file.truncate()
        for turn in range(start_turn, max_turns):
            if (
                config.max_generated_tokens is not None
                and generated_tokens >= config.max_generated_tokens
            ):
                reason = "generation_limit"
                break
            ids = prompt()
            context_events = []
            if len(ids) > config.clear_trigger_tokens:
                before_tokens = len(ids)
                removed_groups = 0
                removed_tokens = 0
                while True:
                    while groups and len(prompt()) > config.clear_target_tokens:
                        removed = groups.pop(0)
                        removed_groups += 1
                        removed_tokens += len(removed)
                    if config.context_feedback_enabled:
                        set_notice(
                            clear_notice(
                                config.max_context_tokens,
                                config.clear_trigger_tokens,
                                removed_groups,
                                removed_tokens,
                                memory,
                                config.context_memory_title_limit,
                            )
                        )
                    if len(prompt()) <= config.clear_target_tokens or not groups:
                        break
                ids = prompt()
                if len(ids) > config.clear_target_tokens:
                    reason = "immutable_context_overflow"
                    break
                clears += 1
                segment = None
                feedback["warning_sent"] = False
                if config.context_feedback_enabled:
                    context_events.append(
                        {
                            "type": "clear",
                            "before_tokens": before_tokens,
                            "after_tokens": len(ids),
                            "removed_groups": removed_groups,
                            "removed_tokens": removed_tokens,
                        }
                    )
                if record.get("agent_info", {}).get(
                    "apply_context_update_friction", False
                ):
                    hook = getattr(env, "context_update", None)
                    if callable(hook):
                        update = hook()
                        if inspect.isawaitable(update):
                            await update
            margin = config.context_warning_margin_tokens or min(
                config.max_new_tokens, config.clear_trigger_tokens - 1
            )
            if (
                config.context_feedback_enabled
                and not context_events
                and not feedback["warning_sent"]
                and len(ids) >= config.clear_trigger_tokens - margin
            ):
                set_notice(
                    warning_notice(
                        config.max_context_tokens, config.clear_trigger_tokens
                    )
                )
                feedback["warning_sent"] = True
                ids = prompt()
                context_events.append({"type": "warning", "input_tokens": len(ids)})
            cap = min(config.max_new_tokens, config.max_context_tokens - len(ids))
            if config.max_generated_tokens is not None:
                cap = min(cap, config.max_generated_tokens - generated_tokens)
            if cap <= 0:
                reason = "context_overflow"
                break
            if segment is None:
                segment = Segment(list(ids), turn_start=turn)
                segments.append(segment)
            generation = await backend.generate(
                ids, cap, dispatch_version, trajectory_uid, turn
            )
            generation.validate(cap)
            generated_tokens += len(generation.token_ids)
            segment.append(
                generation.token_ids,
                1,
                generation.log_probs,
                generation.served_version,
                generation.max_served_version,
            )
            turns += 1
            action = {}
            actions = []
            observations = []
            memory_changed = False
            try:
                actions = parse_actions(generation.text, record.get("tool_schemas"))
            except (ValueError, KeyError, TypeError) as exc:
                observations = [{"error": str(exc), "done": False}]
            terminal = None
            for action in actions:
                if terminal is not None:
                    observations.append(
                        {
                            "error": "Episode already ended; call not executed.",
                            "done": False,
                        }
                    )
                    continue
                try:
                    if action.get("tool") == "memory_write":
                        key, value = action["key"], action["value"]
                        if not isinstance(key, str) or not isinstance(value, str):
                            raise ValueError("memory strings required")
                        memory[key] = value
                        memory_changed = True
                        reply = {"stored": key, "done": False}
                    elif action.get("tool") == "memory_read":
                        reply = {"value": memory.get(action["key"]), "done": False}
                    elif action.get("tool") == "memory":
                        reply = operate_notes(memory, action.get("arguments", {}))
                    elif action.get("tool") == "operational_memory":
                        reply = operate(
                            memory, action.get("arguments", {}), encoder=memory_encoder
                        )
                        memory_changed = True
                    else:
                        reply = env.step(action)
                        if inspect.isawaitable(reply):
                            reply = await reply
                except (ValueError, KeyError, TypeError) as exc:
                    reply = {"error": str(exc), "done": False}
                observations.append(reply)
                if reply.get("done"):
                    terminal = reply
            tool_error_count += sum(bool(reply.get("error")) for reply in observations)
            observation = dict(terminal or observations[-1])
            if len(observations) > 1:
                observation["tool_results"] = observations
            max_chars = record.get("agent_info", {}).get(
                "max_tool_response_chars", 16384
            )

            def visible_result(reply):
                text = json.dumps(reply, sort_keys=True)
                if len(text) > max_chars:
                    visible = {k: reply[k] for k in ("done",) if k in reply}
                    visible.update(truncated=True, result_excerpt=text[:max_chars])
                    text = json.dumps(visible, sort_keys=True)
                return text

            texts = [visible_result(reply) for reply in observations]
            if len(texts) > 1 and callable(getattr(tokenizer, "observations", None)):
                observation_ids = tokenizer.observations(texts)
            else:
                observation_ids = tokenizer.observation(
                    texts[0]
                    if len(texts) == 1
                    else json.dumps({"tool_results": observations}, sort_keys=True)
                )
            audit_record = {
                "turn": turn,
                "input_ids": list(ids),
                "output_ids": list(generation.token_ids),
                "log_probs": list(generation.log_probs),
                "served_version": generation.served_version,
                "action_text": generation.text,
                "observation": observation,
                "cumulative_generated_tokens": generated_tokens,
            }
            if generation.max_served_version is not None:
                audit_record["max_served_version"] = generation.max_served_version
            if config.context_feedback_enabled:
                audit_record["context_events"] = context_events
            if audit_file is None:
                audit.append(audit_record)
            else:
                audit_file.write(json.dumps(audit_record, ensure_ascii=False) + "\n")
                audit_file.flush()
            groups.append(generation.token_ids + observation_ids)
            if (
                len(ids) + len(generation.token_ids) + len(observation_ids)
                > config.max_context_tokens
            ):
                # The observation is not sampled policy data. Keep the complete
                # tool group in history/audit, close the bounded training row,
                # and compact history before the next generation. Never discard
                # the already-sampled assistant tokens or terminate the episode.
                segment = None
            else:
                segment.append(observation_ids, 0, [0.0] * len(observation_ids), -1)
            if (
                continuation is not None
                and not observation.get("done")
                and turns % checkpoint_interval == 0
            ):
                environment = env.snapshot()
                if inspect.isawaitable(environment):
                    environment = await environment
                continuation.save(
                    {
                        "memory": memory,
                        "groups": groups,
                        "segments": [asdict(s) for s in segments],
                        "segment_active": segment is not None and not memory_changed,
                        "audit": audit,
                        "audit_offset": audit_file.tell() if audit_file else 0,
                        "elapsed_seconds": elapsed_before + time.monotonic() - started,
                        "clears": clears,
                        "turns": turns,
                        "generated_tokens": generated_tokens,
                        "tool_error_count": tool_error_count,
                        "next_turn": turn + 1,
                        "environment": environment,
                        "context_feedback": dict(feedback),
                    }
                )
            if observation.get("done"):
                reward = float(observation["reward"])
                if not math.isfinite(reward):
                    raise ValueError("nonfinite reward")
                terminal_metrics = observation.get("score", {})
                reason = "terminal"
                break
            # A refreshed memory prefix changes the train conditioning: branch explicitly.
            if memory_changed:
                segment = None
        finalize = getattr(env, "finalize", None)
        if reason in (
            "turn_limit",
            "generation_limit",
            "context_overflow",
            "immutable_context_overflow",
        ) and callable(finalize):
            outcome = finalize(reason)
            if inspect.isawaitable(outcome):
                outcome = await outcome
            reward = float(outcome["reward"])
            if not math.isfinite(reward):
                raise ValueError("nonfinite finalization reward")
            terminal_metrics = outcome.get("score", {})
    finally:
        if audit_file is not None:
            audit_file.close()
        cleanup = env.close()
        if inspect.isawaitable(cleanup):
            await cleanup
    if continuation is not None:
        continuation.complete()
    for s in segments:
        s.validate()
    return EpisodeResult(
        record["prompt_uid"],
        trajectory_uid,
        dispatch_version,
        segments,
        reward,
        reason,
        turns,
        clears,
        audit,
        memory,
        str(audit_path) if audit_path else None,
        generated_tokens,
        terminal_metrics,
        tool_error_count,
    )


class EpisodeDeadlineExceeded(RuntimeError):
    """Wall-clock budget exhausted; never a successful training trajectory."""


async def run_episode(
    record,
    trajectory_uid,
    dispatch_version,
    backend,
    tokenizer,
    env,
    config,
    audit_path=None,
    continuation=None,
    checkpoint_interval=10,
):
    limit = getattr(config, "episode_timeout_seconds", None)
    previous = continuation.load() if continuation is not None else None
    elapsed = previous.get("elapsed_seconds", 0.0) if previous else 0.0
    started = time.monotonic()
    args = (record, trajectory_uid, dispatch_version, backend, tokenizer, env, config)
    kwargs = dict(
        audit_path=audit_path,
        continuation=continuation,
        checkpoint_interval=checkpoint_interval,
    )
    try:
        if limit is None:
            return await _run_episode(*args, **kwargs)
        remaining = limit - elapsed
        if remaining <= 0:
            cleanup = env.close()
            if inspect.isawaitable(cleanup):
                await cleanup
            raise EpisodeDeadlineExceeded("episode deadline exhausted before resume")
        try:
            return await asyncio.wait_for(_run_episode(*args, **kwargs), remaining)
        except asyncio.TimeoutError as exc:
            if time.monotonic() - started < remaining:
                raise
            raise EpisodeDeadlineExceeded(
                "episode wall-clock deadline exceeded"
            ) from exc
    except (EpisodeDeadlineExceeded, asyncio.CancelledError) as exc:
        if audit_path is not None:
            path = Path(audit_path).with_suffix(".termination.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "trajectory_uid": trajectory_uid,
                        "termination": (
                            "episode_timeout"
                            if isinstance(exc, EpisodeDeadlineExceeded)
                            else "cancelled"
                        ),
                        "elapsed_seconds": elapsed + time.monotonic() - started,
                    }
                )
                + "\n"
            )
            tmp.replace(path)
        raise
