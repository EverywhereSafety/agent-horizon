"""GPU fault probe: cancel an observed active native vLLM request and reject completion."""

import asyncio, json, time, uuid
from pathlib import Path
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient
from long_horizon_rl.cancellation import cancel_aware_client

from long_horizon_rl.adapters.vllm_cancel_v1 import request_status as active_request


async def probe(trainer):
    state = {}
    receipts = []

    class ObservedClient(FullyAsyncLLMServerClient):
        async def _acquire_server(self, *args, **kwargs):
            result = await super()._acquire_server(*args, **kwargs)
            state["server"] = result[1]
            return result

        def _vllm_request_id(self, request_id):
            result = super()._vllm_request_id(request_id)
            state["id"] = result
            return result

    client = cancel_aware_client(
        trainer.standalone_server_manager.get_client(client_cls=ObservedClient),
        on_abort=receipts.append,
    )
    prompt = trainer.tokenizer.encode(
        "Continue listing integers from 1 upward, one per line.",
        add_special_tokens=False,
    )
    task = asyncio.create_task(
        client.generate(
            uuid.uuid4().hex,
            prompt_ids=prompt,
            sampling_params={
                "max_tokens": 8192,
                "min_tokens": 8192,
                "ignore_eos": True,
                "temperature": 0.7,
            },
        )
    )
    deadline = time.monotonic() + 120
    observed = None
    try:
        while time.monotonic() < deadline:
            if task.done():
                # Retrieve the actual backend error, rather than hiding invalid
                # sampling parameters behind a premature-completion message.
                await task
                raise RuntimeError(
                    "generation finished before active cancellation could be tested"
                )
            if "id" in state:
                observed = await state["server"].__ray_call__.remote(
                    active_request, state["id"]
                )
                if observed["active"]:
                    break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError("native active request not observed")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancelled generation returned publishable result")
        assert len(receipts) == 1 and receipts[0]["acknowledgement"].get(
            "aborted"
        ), receipts
        after = await state["server"].__ray_call__.remote(active_request, state["id"])
        assert not after["active"], after
        record = {
            "status": "LIVE_NATIVE_ABORT_PASS",
            "observed_before": observed,
            "after": after,
            "receipt": receipts[0],
            "completion_rejected": True,
        }
        root = Path(trainer.config.trainer.default_local_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "live_abort.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except BaseException:
                pass
