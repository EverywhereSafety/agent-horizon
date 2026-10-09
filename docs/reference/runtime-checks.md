# Runtime checks

Use the pinned versions in [runtime.json](../../configs/runtime.json). Validate the
model, serving backend and tool runtime for the context and batch configuration
used by your deployment.

## Runtime setup

Distributed training uses Linux, Python 3.12, NVIDIA GPUs and a compatible CUDA
runtime. `scripts/bootstrap.sh` prepares the pinned veRL, PyTorch, vLLM,
Transformers and TransferQueue environment plus an isolated tool runtime.
Isolated task execution requires Linux bubblewrap. See
[backend compatibility](architecture.md#backend-compatibility) for the current
model and input adapter.

GPU layout, precision, offload and batching are training configuration choices.
The default launcher uses four GPUs; choose capacity for your model and context.

## Qualify your configuration

| Check | Entry point |
|---|---|
| CPU episode, clear and recovery example | `python examples/smoke.py --output outputs/cpu-smoke` |
| Environment records | `scripts/prepare_queries.py` |
| Reward and backend field alignment | `scripts/checks/check_adapter_contract.py` |
| Rollout inputs versus training segments | `scripts/checks/check_rollout_provenance.py` |
| Context-clear policy coverage | `scripts/checks/verify_context_clear.py` |
| Generation/update overlap | `scripts/checks/summarize_infra_overlap.py` |

CPU contract tests exercise state, token masks and recovery logic. GPU checks
exercise the actual serving and training configuration. Keep model identity,
runtime configuration, resource layout and check results with the run artifacts.

[Context](../guides/context.md) · [Async execution](../guides/async.md) ·
[Recovery](../guides/recovery.md) · [Capacity](../guides/context.md)

## Local regression tests

```bash
python -m pip install -e '.[test]'
python -m pytest tests -q
```

The test extra supplies CPU tensor and configuration dependencies. Tests needing
Linux bubblewrap skip when it is unavailable; GPU/backend checks in the table
run separately in the pinned training runtime.

## Directory conventions

Training launchers stay in `scripts/`; qualification commands live in
`scripts/checks/`. Optional Megatron
environment preparation lives in `scripts/backends/`, with pins in
`configs/backends/`. Activate the installed Agent Horizon runtime before running
Python checks. These checks use compute resources supplied by your deployment.
