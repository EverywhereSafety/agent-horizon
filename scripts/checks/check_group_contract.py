"""Exercise production group selection and native publication with CPU doubles."""

import asyncio, json, os, tempfile, types
from pathlib import Path
import transfer_queue as tq
from long_horizon_rl.adapters.verl_v1 import GroupWorker, LongHorizonOutput
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics


async def main():
    worker_class = GroupWorker.__ray_metadata__.modified_class
    with tempfile.TemporaryDirectory() as directory:
        prior = os.environ.get("LONG_HORIZON_AUDIT_DIR")
        os.environ["LONG_HORIZON_AUDIT_DIR"] = directory
        publications = []
        statuses = []
        closed = set()
        received = []
        started = asyncio.Queue()
        ready = asyncio.Event()

        async def capture(**kwargs):
            publications.append(kwargs)

        async def status(**kwargs):
            statuses.append(kwargs)

        async def noop(*args, **kwargs):
            pass

        async def loop(params, *args, session_id, **kwargs):
            assert "__do_sample__" not in kwargs and "__rollout_n__" not in kwargs
            received.append(dict(params))
            try:
                await started.put(session_id)
                if session_id >= 2:
                    await asyncio.Event().wait()
                await ready.wait()
                return [
                    LongHorizonOutput(
                        prompt_ids=[10],
                        response_ids=[20, 30],
                        response_mask=[1, 0],
                        response_logprobs=[-0.2, 0],
                        reward_score=float(session_id),
                        metrics=AgentLoopMetrics(),
                        extra_fields={
                            "trajectory_uid": f"{kwargs['uid']}/{session_id}",
                            "termination": "terminal",
                            "min_global_steps": 0,
                            "max_global_steps": 0,
                            "reward_extra_info": {},
                        },
                    )
                ]
            finally:
                closed.add(session_id)

        fake = types.SimpleNamespace(
            config=types.SimpleNamespace(
                distillation=types.SimpleNamespace(enabled=False),
                actor_rollout_ref=types.SimpleNamespace(
                    rollout=types.SimpleNamespace(
                        n=2, val_kwargs=types.SimpleNamespace(n=2)
                    )
                ),
            ),
            _run_agent_loop=loop,
            _compute_score=noop,
            _compute_teacher_logprobs=noop,
        )
        # Method uses super(); fake must really be an instance of GroupWorker.
        worker = object.__new__(worker_class)
        worker.__dict__.update(fake.__dict__)
        original_put, original_batch = tq.async_kv_put, tq.async_kv_batch_put
        tq.async_kv_put = status
        tq.async_kv_batch_put = capture
        try:
            task = asyncio.create_task(
                worker._run_prompt(
                    {
                        "uid": "group",
                        "record_json": json.dumps({"prompt_uid": "p"}),
                        "global_steps": 0,
                        "__do_sample__": False,
                    },
                    {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
                    {"validate": False},
                )
            )
            for _ in range(4):
                await started.get()
            assert not publications
            ready.set()
            await task
            assert closed == {0, 1, 2, 3}
            assert len(publications) == 2
            assert all(
                p["temperature"] == 0 and p["top_p"] == 1 and p["top_k"] == -1
                for p in received
            )
            assert statuses[-1]["tag"]["status"] == "finished"
            assert len(list(Path(directory).glob("*.cancelled.json"))) == 2
            assert (
                json.loads((Path(directory) / "group.group.json").read_text())[
                    "retained"
                ]
                == 2
            )
            # Configure a budget other than the historical fixed retain+2.
            from verl.experimental.agent_loop.agent_loop import _agent_loop_registry

            _agent_loop_registry["_group_contract_test"] = {
                "episode_config": {"attempt_n": 5}
            }
            received.clear()
            publications.clear()
            closed.clear()
            await worker._run_prompt(
                {
                    "uid": "configured",
                    "agent_name": "_group_contract_test",
                    "record_json": json.dumps({"prompt_uid": "p"}),
                    "global_steps": 0,
                },
                {"temperature": 0.7},
                {"validate": False},
            )
            assert closed == {0, 1, 2, 3, 4} and len(received) == 5
            assert all(p["temperature"] == 0.7 for p in received)
            assert (
                json.loads((Path(directory) / "configured.group.json").read_text())[
                    "attempts"
                ]
                == 5
            )
            _agent_loop_registry["_group_contract_test"] = {
                "episode_config": {"attempt_n": 5, "retain_n": 3}
            }
            publications.clear()
            await worker._run_prompt(
                {
                    "uid": "conflict",
                    "agent_name": "_group_contract_test",
                    "record_json": json.dumps({"prompt_uid": "p"}),
                    "global_steps": 0,
                },
                {},
                {"validate": False},
            )
            assert not publications and statuses[-1]["tag"]["status"] == "failure"
            del _agent_loop_registry["_group_contract_test"]
            print("PRODUCTION_SELECTION_CANCEL_PUBLICATION_BUDGET_GREEDY_CONTRACT_PASS")
        finally:
            tq.async_kv_put = original_put
            tq.async_kv_batch_put = original_batch
            if prior is None:
                os.environ.pop("LONG_HORIZON_AUDIT_DIR", None)
            else:
                os.environ["LONG_HORIZON_AUDIT_DIR"] = prior


asyncio.run(asyncio.wait_for(main(), 30))
