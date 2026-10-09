"""Run one shard of a frozen plan through the teacher collection harness."""

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path

from murdoku_demo.sft_eval import summarize_results


async def run(a):
    spec = importlib.util.spec_from_file_location(
        "teacher_benchmark", Path(__file__).with_name("benchmark_murdoku_tools.py")
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    plan = json.loads(Path(a.plan).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(2)

    async def evaluate(task):
        async with semaphore:
            root = out / str(task["task_id"])
            root.mkdir(exist_ok=True)
            saved = root / "evaluation-result.json"
            if saved.exists():
                previous = json.loads(saved.read_text())
                if previous.get("status") != "infrastructure_error":
                    if any(
                        previous.get(k) != task[k]
                        for k in ["task_id", "prompt_uid", "seed", "replicate"]
                    ):
                        raise ValueError("saved evaluation identity differs from plan")
                    return previous
            query_file = root / "query.jsonl"
            query_file.write_text(json.dumps(task["query"]) + "\n")
            args = bench.p.parse_args(
                [
                    "--input",
                    str(query_file),
                    "--endpoint",
                    a.endpoint,
                    "--out",
                    str(root),
                    "--seed",
                    str(task["seed"]),
                    *plan["teacher_eval_args"],
                ]
            )
            row = {k: v for k, v in task.items() if k != "query"}
            try:
                await asyncio.wait_for(
                    bench.main(args), plan["episode_timeout_seconds"]
                )
                result = json.loads((root / "summary.json").read_text())["results"][0]
                row.update(
                    result,
                    status="complete",
                    strict_success=bool((result.get("score") or {}).get("solved")),
                )
                if result["termination"] in (
                    "invalid_input",
                    "request_or_environment_failure",
                ):
                    row.update(status="infrastructure_error", strict_success=None)
            except asyncio.TimeoutError:
                row.update(status="episode_timeout", strict_success=False, reward=0)
            except Exception as exc:
                row.update(
                    status="infrastructure_error", strict_success=None, error=repr(exc)
                )
            (root / "evaluation-result.json").write_text(
                json.dumps(row, indent=2) + "\n"
            )
            return row

    tasks = [t for t in plan["tasks"] if t["task_id"] % a.shards == a.shard]
    results = await asyncio.gather(*(evaluate(t) for t in tasks))
    (out / "evaluation-summary.json").write_text(
        json.dumps(
            {
                "shard": a.shard,
                "results": results,
                "summary": summarize_results(results),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, default=8)
    asyncio.run(run(p.parse_args()))
