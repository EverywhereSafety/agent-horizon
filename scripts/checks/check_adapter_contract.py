import json
import torch
from long_horizon_rl.adapters.verl_v1 import LongHorizonOutput
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics

output = LongHorizonOutput(
    prompt_ids=[10],
    response_ids=[20, 30, 40],
    response_mask=[1, 1, 0],
    response_logprobs=[-0.2, -0.3, 0.0],
    reward_score=0.7,
    metrics=AgentLoopMetrics(),
)
fields = output.as_dict()
assert fields["rm_scores"][-1] == 0
assert abs(fields["rm_scores"][1].item() - 0.7) < 1e-6
assert fields["rollout_log_probs"].shape == fields["response_mask"].shape
print(json.dumps({"status": "VERL_ADAPTER_REWARD_AND_ALIGNMENT_PASS"}))

# Exercise the actual TQ conversion and minibatch selection for text positions.
import pickle
from long_horizon_rl.adapters.verl_v1 import GroupWorker
from verl.utils.tensordict_utils import (
    list_of_dict_to_tensordict,
    maybe_fix_3d_position_ids,
    index_select_tensor_dict,
)

worker = GroupWorker.__ray_metadata__.modified_class
assert (
    worker._compute_multi_modal_inputs(None, output, torch.tensor(output.prompt_ids))
    == {}
)
rows = []
for length in (106, 108, 106):
    ids = torch.ones(1, length, dtype=torch.long)
    positions = worker._compute_position_ids(
        None, ids, ids, {"mm_token_type_ids": torch.zeros_like(ids)}
    )
    assert positions.shape == ids.shape and torch.equal(
        positions[0], torch.arange(length)
    )
    rows.append({"position_ids": positions.squeeze(0)})
data = pickle.loads(pickle.dumps(list_of_dict_to_tensordict(rows)))
maybe_fix_3d_position_ids(data)
selected = index_select_tensor_dict(data, [2, 1])
assert [x.numel() for x in selected["position_ids"].unbind()] == [106, 108]
print("TEXT_POSITION_TQ_ROUNDTRIP_PASS")

# Verify the real upstream publication method calls our text-only overrides.
import asyncio
import types
import transfer_queue as tq
from long_horizon_rl.adapters.verl_v1 import _Worker


async def publication_check():
    captured = []
    original = tq.async_kv_batch_put

    async def capture(**kwargs):
        captured.append(kwargs)

    async def reward(*args, **kwargs):
        return None

    fake = types.SimpleNamespace(
        _compute_score=reward, _compute_teacher_logprobs=reward
    )
    fake._compute_multi_modal_inputs = types.MethodType(
        worker._compute_multi_modal_inputs, fake
    )
    fake._compute_position_ids = types.MethodType(worker._compute_position_ids, fake)
    tq.async_kv_batch_put = capture
    try:
        await _Worker._agent_loop_postprocess(
            fake, output, False, uid="fixture", session_id=0, global_steps=0
        )
    finally:
        tq.async_kv_batch_put = original
    fields = captured[0]["fields"]
    assert fields["position_ids"].shape == (1, 4)
    assert torch.equal(fields["position_ids"][0], torch.arange(4))


asyncio.run(publication_check())
print("ACTUAL_TQ_PUBLICATION_TEXT_POSITION_PASS")
