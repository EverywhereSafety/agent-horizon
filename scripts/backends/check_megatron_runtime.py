"""GPU import/TE kernel probe; not a Megatron actor or MoE qualification."""

import argparse, importlib.metadata, json, math
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
assert torch.cuda.is_available(), "GPU required"
assert torch.__version__.startswith("2.11.0"), torch.__version__
import apex
import transformer_engine.pytorch as te
from megatron.core import parallel_state
from megatron.core.optimizer import OptimizerConfig
from verl.workers.engine.megatron.transformer_impl import MegatronEngine
from megatron.bridge import AutoBridge
from mbridge import AutoBridge as LegacyBridge

module = te.Linear(64, 64, params_dtype=torch.bfloat16).to("cuda")
x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
y = module(x)
loss = y.float().square().mean()
loss.backward()
assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())
report = {
    "status": "PASS",
    "scope": "GPU imports and Transformer Engine BF16 Linear backward; no full actor/MoE/update/sync/resume proof",
    "gpu": torch.cuda.get_device_name(),
    "cuda": torch.version.cuda,
    "versions": {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "megatron-core",
            "megatron-bridge",
            "transformer-engine",
            "apex",
            "flash-attn",
            "transferqueue",
        )
    },
    "loss": loss.item(),
    "input_grad_norm": x.grad.float().norm().item(),
    "muon_config_fields": [
        name for name in OptimizerConfig.__dataclass_fields__ if name.startswith("muon")
    ],
    "bridge_classes": [str(AutoBridge), str(LegacyBridge)],
}
path = Path(args.output)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
print("MEGATRON_IMPORT_AND_TE_KERNEL_PASS", flush=True)
