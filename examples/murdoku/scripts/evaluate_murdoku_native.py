"""Evaluate immutable models through the same native token and environment path as RL."""

import argparse, ast, asyncio, json, shutil, time
from dataclasses import asdict
from pathlib import Path
from transformers import AutoTokenizer
from verl.utils.tokenizer.continuous_token import QwenContinuousTokenBuilder
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from long_horizon_rl.contracts import Config, Generation
from long_horizon_rl.environments import make_environment
from long_horizon_rl.episode import run_episode, EpisodeDeadlineExceeded
from long_horizon_rl.parser import parse_actions


def entrypoint():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--case-index", type=int)
    p.add_argument("--workspace", action="store_true")
    p.add_argument("--python-timeout", type=float, default=120)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, a.out / "evaluation-runner.py")
    config = Config(
        max_context_tokens=131072,
        clear_trigger_tokens=49152,
        clear_target_tokens=32768,
        max_new_tokens=81920,
        max_turns=50,
        episode_timeout_seconds=7200,
        enable_thinking=True,
        context_feedback_enabled=True,
    )
    root = Path(__file__).resolve().parents[1]
    node = next(
        n
        for n in ast.parse(
            (
                Path(__import__("long_horizon_rl").__file__).parent
                / "adapters/verl_v1.py"
            ).read_text()
        ).body
        if isinstance(n, ast.ClassDef) and n.name == "QwenTokenizer"
    )
    namespace = {"json": json, "QwenContinuousTokenBuilder": QwenContinuousTokenBuilder}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "<native-tokenizer>", "exec"),
        namespace,
    )
    QwenTokenizer = namespace["QwenTokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    engine_args = AsyncEngineArgs(
        model=a.model,
        dtype="bfloat16",
        max_model_len=131072,
        max_num_seqs=a.concurrency,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
        compilation_config={
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": list(range(1, a.concurrency + 1)),
        },
        gdn_prefill_backend="triton",
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    engine = None
    records = [json.loads(line) for line in a.input.open() if line.strip()]
    if a.case_index is not None:
        if not 0 <= a.case_index < len(records):
            raise ValueError("case index out of range")
        records = [records[a.case_index]]
    if a.workspace:
        from murdoku_demo.murdoku_protocol import simplify_record

        records = [
            simplify_record(r, workspace=True, python_timeout=a.python_timeout)
            for r in records
        ]
    (a.out / "evaluation-config.json").write_text(
        json.dumps(
            {
                "model": a.model,
                "config": asdict(config),
                "concurrency": a.concurrency,
                "inference": {
                    "cuda_graph": "FULL_DECODE_ONLY",
                    "capture_sizes": list(range(1, a.concurrency + 1)),
                    "torch_compile": False,
                    "gdn_prefill_backend": "triton",
                },
                "sampling": {
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 20,
                    "min_p": 0.0,
                    "presence_penalty": 1.5,
                    "repetition_penalty": 1.0,
                    "base_seed": 0,
                    "seed_policy": "base_seed_plus_turn_index",
                },
                "sampling_source": "Qwen official model card: thinking general tasks",
                "prompt_uids": [r["prompt_uid"] for r in records],
                "python_workspace": a.workspace,
                "python_timeout_seconds": a.python_timeout if a.workspace else 30,
                "training": False,
            },
            indent=2,
        )
        + "\n"
    )

    class Backend:
        def __init__(self, adapter):
            self.adapter = adapter
            self.pending = None

        async def generate(self, ids, cap, version, uid, turn):
            request = f"{uid}-{turn}"
            self.pending = request
            try:
                final = None
                async for output in engine.generate(
                    {"prompt_token_ids": ids},
                    SamplingParams(
                        temperature=1.0,
                        top_p=0.95,
                        top_k=20,
                        min_p=0.0,
                        presence_penalty=1.5,
                        repetition_penalty=1.0,
                        seed=turn,
                        max_tokens=cap,
                        logprobs=1,
                    ),
                    request_id=request,
                ):
                    final = output
                if final is None:
                    raise RuntimeError("vLLM returned no generation")
                sample = final.outputs[0]
                tokens = list(sample.token_ids)
                probs = [
                    float(row[t].logprob) for t, row in zip(tokens, sample.logprobs)
                ]
                self.adapter.last_generated = tokens
                print(
                    json.dumps(
                        {
                            "event": "turn",
                            "uid": uid,
                            "turn": turn + 1,
                            "generated_tokens": len(tokens),
                        }
                    ),
                    flush=True,
                )
                return Generation(
                    tokens,
                    probs,
                    tokenizer.decode(tokens, skip_special_tokens=False),
                    0,
                    0,
                )
            finally:
                await engine.abort(request)
                self.pending = None

    async def main():
        nonlocal engine
        engine = AsyncLLM.from_engine_args(engine_args)
        semaphore = asyncio.Semaphore(a.concurrency)

        async def evaluate(record):
            async with semaphore:
                uid = record["prompt_uid"]
                started = time.monotonic()
                adapter = QwenTokenizer(
                    tokenizer, record.get("tool_schemas"), enable_thinking=True
                )
                backend = Backend(adapter)
                env = await make_environment(record).start()
                env.workspace_export_path = a.out / (uid + ".workspace.json")
                try:
                    result = await run_episode(
                        record,
                        uid,
                        0,
                        backend,
                        adapter,
                        env,
                        config,
                        audit_path=a.out / (uid + ".jsonl"),
                    )
                    calls = []
                    for line in (a.out / (uid + ".jsonl")).open():
                        row = json.loads(line)
                        try:
                            calls.append(
                                len(
                                    parse_actions(
                                        row["action_text"], record.get("tool_schemas")
                                    )
                                )
                            )
                        except (ValueError, TypeError):
                            calls.append(0)
                    summary = {
                        "prompt_uid": uid,
                        "status": "complete",
                        "strict_success": bool(result.terminal_metrics.get("solved")),
                        "reward": result.reward,
                        "termination": result.termination,
                        "turns": result.turns,
                        "generated_tokens": result.generated_tokens,
                        "context_clears": result.context_clears,
                        "tool_errors": result.tool_error_count,
                        "multi_call_turns": sum(n > 1 for n in calls),
                        "valid_tool_calls": sum(calls),
                        "terminal_metrics": result.terminal_metrics,
                    }
                except EpisodeDeadlineExceeded as exc:
                    summary = {
                        "prompt_uid": uid,
                        "status": "episode_timeout",
                        "strict_success": False,
                        "detail": str(exc),
                    }
                except Exception as exc:
                    summary = {
                        "prompt_uid": uid,
                        "status": "infrastructure_error",
                        "strict_success": None,
                        "detail": repr(exc),
                    }
                finally:
                    await env.close()
                summary["seconds"] = time.monotonic() - started
                (a.out / (uid + ".summary.json")).write_text(
                    json.dumps(summary, indent=2) + "\n"
                )
                print(json.dumps({"event": "episode_complete", **summary}), flush=True)
                return summary

        summaries = await asyncio.gather(*(evaluate(record) for record in records))
        (a.out / "results.json").write_text(json.dumps(summaries, indent=2) + "\n")
        print("NATIVE_EVALUATION_COMPLETE", flush=True)

    try:
        asyncio.run(main())
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    entrypoint()
