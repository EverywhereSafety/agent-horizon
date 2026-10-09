"""Bounded training-only collection with real grading and fresh replay acceptance."""

import argparse, asyncio, collections, importlib.util, json, os, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from murdoku_demo.teacher_replay import replay_candidate


async def main(a):
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    if a.not_before and time.time() < a.not_before:
        print("WAITING_FOR_EXISTING_QUERY_CLAIMS", a.not_before, flush=True)
        await asyncio.sleep(a.not_before - time.time())
    records = [json.loads(l) for l in Path(a.input).open() if l.strip()]
    if a.skip_existing_root:
        existing = {
            p.parent.name
            for p in Path(a.skip_existing_root).glob("worker-*/*/query.jsonl")
        }
        records = [r for r in records if r["prompt_uid"] not in existing]
    if any(
        r.get("split") != "train"
        or not r.get("python_workspace")
        or not {"murdoku", "memory", "workspace", "run_python"}
        <= {s["name"] for s in r.get("tool_schemas", [])}
        for r in records
    ):
        raise ValueError("require prepared training-only workspace queries")
    spec = importlib.util.spec_from_file_location(
        "teacher_benchmark", Path(__file__).with_name("benchmark_murdoku_tools.py")
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    pending = collections.deque(enumerate(records))
    stats = {
        "attempted": 0,
        "completed": 0,
        "solved": 0,
        "accepted": 0,
        "timed_out": 0,
        "replay_pending": 0,
    }
    claims = Path(a.claim_dir) if a.claim_dir else root.parent / "query-claims"
    claims.mkdir(parents=True, exist_ok=True)

    def append(name, row):
        with (root / name).open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def progress():
        (root / "progress.json").write_text(
            json.dumps(
                {**stats, "remaining": len(pending), "stop_at_epoch": a.stop_at},
                indent=2,
            )
            + "\n"
        )

    async def worker():
        while pending and time.time() < a.stop_at - 120:
            index, record = pending.popleft()
            uid = record["prompt_uid"]
            try:
                (claims / uid).mkdir()
            except FileExistsError:
                continue
            attempt = root / uid
            attempt.mkdir(exist_ok=True)
            input_path = attempt / "query.jsonl"
            input_path.write_text(json.dumps(record) + "\n")
            args = bench.p.parse_args(
                [
                    "--input",
                    str(input_path),
                    "--endpoint",
                    a.endpoint,
                    "--out",
                    str(attempt),
                    "--parallel",
                    "1",
                    "--turns",
                    "20",
                    "--seed",
                    str(a.seed + index),
                    "--tokens",
                    "81920",
                    "--generation-limit",
                    "0",
                    "--context",
                    "131072",
                    "--context-clear",
                    "--clear-trigger",
                    "49152",
                    "--clear-target",
                    "32768",
                    "--context-feedback",
                    "--thinking",
                    "--preserve-thinking",
                    "--temperature",
                    "1",
                    "--top-p",
                    ".95",
                    "--top-k",
                    "20",
                    "--min-p",
                    "0",
                    "--presence-penalty",
                    "0",
                    "--repetition-penalty",
                    "1",
                    "--workspace",
                    "--python-timeout",
                    "120",
                    "--keep-input-protocol",
                    "--purpose",
                    "teacher",
                    "--request-timeout",
                    str(
                        max(
                            1, int(min(a.episode_timeout, a.stop_at - time.time() - 30))
                        )
                    ),
                ]
            )
            stats["attempted"] += 1
            progress()
            try:
                await asyncio.wait_for(
                    bench.main(args),
                    timeout=min(a.episode_timeout, a.stop_at - time.time() - 60),
                )
            except asyncio.TimeoutError:
                stats["timed_out"] += 1
                append("attempts.jsonl", {"uid": uid, "status": "collection_timeout"})
                progress()
                continue
            except Exception as exc:
                append(
                    "attempts.jsonl",
                    {"uid": uid, "status": "collection_error", "error": str(exc)},
                )
                progress()
                continue
            candidate_path = attempt / (uid + ".json")
            if not candidate_path.exists():
                append("attempts.jsonl", {"uid": uid, "status": "missing_candidate"})
                progress()
                continue
            candidate = json.loads(candidate_path.read_text())
            stats["completed"] += 1
            solved = bool((candidate["result"].get("score") or {}).get("solved"))
            stats["solved"] += solved
            append(
                "attempts.jsonl",
                {
                    "uid": uid,
                    "status": candidate["result"]["termination"],
                    "solved": solved,
                    "turns": candidate["result"]["turns"],
                    "seed": a.seed + index,
                },
            )
            if solved:
                try:
                    if time.time() > a.stop_at - 60:
                        raise asyncio.TimeoutError()
                    trace = await asyncio.wait_for(
                        replay_candidate(
                            record, candidate, allow_recoverable_errors=True
                        ),
                        timeout=min(180, a.stop_at - time.time() - 30),
                    )
                    trace["provenance"] = {
                        "teacher_model": a.model,
                        "revision": a.revision,
                        "sampling": {
                            "temperature": 1.0,
                            "top_p": 0.95,
                            "top_k": 20,
                            "min_p": 0.0,
                            "presence_penalty": 0.0,
                            "repetition_penalty": 1.0,
                            "seed": a.seed + index,
                            "preserve_thinking": True,
                        },
                        "python_workspace": True,
                        "assistant_turn_limit": 20,
                        "episode_timeout_seconds": a.episode_timeout,
                        "candidate_file": str(attempt / (uid + ".json")),
                        "fresh_replay": True,
                    }
                    append(
                        "query_rollouts.jsonl",
                        {
                            "prompt_uid": uid,
                            "split": "train",
                            "query": record,
                            "rollout": trace,
                        },
                    )
                    stats["accepted"] += 1
                except asyncio.TimeoutError:
                    stats["replay_pending"] += 1
                    append(
                        "replay-pending.jsonl",
                        {"uid": uid, "candidate_file": str(attempt / (uid + ".json"))},
                    )
                except Exception as exc:
                    append("replay-rejected.jsonl", {"uid": uid, "error": str(exc)})
            progress()

    await asyncio.gather(*(worker() for _ in range(a.parallel)))
    progress()
    print("TEACHER_COLLECTION_COMPLETE", json.dumps(stats), flush=True)


p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--endpoint", required=True)
p.add_argument("--out", required=True)
p.add_argument("--stop-at", type=float, required=True)
p.add_argument("--parallel", type=int, default=2)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--model", required=True)
p.add_argument("--revision", required=True)
p.add_argument(
    "--skip-existing-root",
    help="Exclude queries already attempted by an earlier collection",
)
p.add_argument(
    "--not-before",
    type=float,
    default=0,
    help="Wait until previous workers stop claiming queries before filtering",
)
p.add_argument(
    "--episode-timeout",
    type=float,
    default=900,
    help="Per-episode wall time; adjust for inference hardware",
)
p.add_argument(
    "--claim-dir",
    help="Shared directory for atomic query claims; defaults beside worker output directories",
)
if __name__ == "__main__":
    asyncio.run(main(p.parse_args()))
