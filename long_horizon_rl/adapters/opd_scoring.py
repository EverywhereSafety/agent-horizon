"""Bounded concurrent OPD scoring; native publication scores final segments."""

import asyncio
import time


async def score_earlier_segments(worker, groups, kwargs, parallel=4):
    if not isinstance(parallel, int) or parallel < 1:
        raise ValueError("teacher scoring parallelism must be a positive integer")
    semaphore = asyncio.Semaphore(parallel)
    outputs = [output for group in groups for output in group[:-1]]
    start = time.monotonic()

    async def score(output):
        async with semaphore:
            await worker._compute_teacher_logprobs(
                output,
                prompt_ids=output.prompt_ids,
                response_ids=output.response_ids,
                validate=False,
                sample_kwargs=kwargs,
            )
            if (
                "teacher_logprobs" not in output.extra_fields
                or "teacher_ids" not in output.extra_fields
            ):
                raise ValueError("Missing teacher scores on earlier segment")

    tasks = [asyncio.create_task(score(output)) for output in outputs]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        # Do not leave RPC tasks mutating a failed group's segments in background.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    print(
        "OPD_TEACHER_SCORED",
        {
            "earlier_segments": len(outputs),
            "parallel": parallel,
            "scored_context_tokens": sum(
                len(o.prompt_ids) + len(o.response_ids) for o in outputs
            ),
            "seconds": time.monotonic() - start,
        },
        flush=True,
    )
