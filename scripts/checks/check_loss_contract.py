"""CPU golden check against the pinned framework's actual registered PPO loss."""

import json
import torch
from omegaconf import OmegaConf
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

config = OmegaConf.create(
    {
        "clip_ratio": 0.2,
        "clip_ratio_low": None,
        "clip_ratio_high": None,
        "clip_ratio_c": 3.0,
        "global_batch_info": {},
    }
)
mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
old = torch.zeros(2, 2)


def gradient(sign, masked_value):
    current = torch.tensor(
        [[0.0, masked_value], [0.0, masked_value]], requires_grad=True
    )
    advantage = torch.tensor([[0.5 * sign, 0.0], [-0.5 * sign, 0.0]])
    loss, metrics = compute_policy_loss_vanilla(
        old, current, advantage, mask, config=config
    )
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(current.grad).all()
    return current.grad


base = gradient(1, 0)
assert torch.equal(base, torch.tensor([[-0.25, 0.0], [0.25, 0.0]]))
assert torch.equal(gradient(-1, 0), -base)
assert torch.equal(gradient(1, 10), base)
print(json.dumps({"status": "REGISTERED_PPO_GOLDEN_PASS", "gradient": base.tolist()}))

from long_horizon_rl.trajectory_weighting import trajectory_weights

mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
weights = torch.tensor(trajectory_weights(["p_0_0", "p_0_1", "p_1_0"], [1, 3, 2]))
current = torch.zeros(3, 3, requires_grad=True)
advantage = torch.tensor([[0.5] * 3, [0.5] * 3, [-0.5] * 3]) * weights[:, None]
loss, _ = compute_policy_loss_vanilla(
    torch.zeros_like(current),
    current,
    advantage,
    mask,
    loss_agg_mode="token-sum",
    config=config,
)
loss.backward()
assert torch.isfinite(current.grad).all()
assert current.grad[:2].sum().item() == -0.25 and current.grad[2].sum().item() == 0.25
assert torch.equal(current.grad * (1 - mask), torch.zeros_like(current))
print(
    json.dumps(
        {
            "status": "NATIVE_TRAJECTORY_NORMALIZED_GRADIENT_PASS",
            "gradient": current.grad.tolist(),
        }
    )
)
