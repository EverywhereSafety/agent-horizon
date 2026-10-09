# Murdoku demo

This branch adds an optional task plugin under `examples/murdoku`.
Puzzle generation and grading stay in
[Murdoku Lab](https://github.com/EverywhereSafety/murdoku-lab); this package provides the environment
bridge, RL integration, SFT preparation and model evaluation.
Murdoku Lab owns the canonical prompt and fresh text/vision replay; the demo imports
them. Shared workspace, Python, memory and context tools live in Agent Horizon.
The native sampling/evaluation CLI also delegates to Murdoku Lab’s runner.

For query fields, puzzle dynamics, tool/reward interfaces and the first RL update,
follow the [Murdoku integration guide](https://github.com/EverywhereSafety/murdoku-lab/blob/main/docs/guides/rl.md).
For another task, use Agent Horizon’s [environment contract](../../docs/guides/environments.md).

Use two checkouts: Murdoku Lab for cases, grading and native sampling, and this
Agent Horizon demo branch for the training plugin and data preparation. Follow
Murdoku Lab's installation quickstart to create its `.venv`; run Agent Horizon's
[`scripts/bootstrap.sh`](../../scripts/bootstrap.sh) for the pinned GPU runtime.
Bootstrap also creates `.sandbox-env` for scientific Python tool execution.

From the Agent Horizon checkout root, with its runtime active:

```bash
export LONG_HORIZON_MURDOKU_ROOT=/absolute/path/to/murdoku-lab
source .venv/bin/activate
python -m pip install -e . -e "$LONG_HORIZON_MURDOKU_ROOT" -e examples/murdoku
python -m pip check
export LONG_HORIZON_TOOL_RUNTIME="$PWD/.sandbox-env"
python scripts/prepare_queries.py --help
python examples/murdoku/scripts/prepare_visual_sft.py --help
python examples/murdoku/scripts/prepare_murdoku_sft.py --help
```

Murdoku Lab's `.venv` runs the trusted grader. `LONG_HORIZON_TOOL_RUNTIME` points
to a separate environment directory containing `bin/python` for model-written
code. Isolated execution needs Linux with `bwrap`. The plugin
registers `environment_type=murdoku` through package entry points. Core query
loading and rollout dispatch discover it without task-specific imports.

## Two training routes

- **RL:** initialize the puzzle environment from query records, sample new
  interactions, and train with its executable gold rewards. Convert exported
  queries with `scripts/prepare_queries.py` and follow the integration guide's
  first update.
- **SFT:** use query + trained-solver rollout records as assistant supervision.
  `prepare_murdoku_sft.py` prepares text inputs;
  `prepare_visual_sft.py` prepares screenshot inputs using the model processor.
  See [visual SFT](docs/visual_sft.md) for the veRL launcher.

Model sampling and fresh replay use Murdoku Lab's canonical runner and environment.
The training backend, optimizer, distributed execution and checkpoint machinery
come from veRL.

The frozen dataset remains query+rollout in one JSONL. Only strict-replay-qualified
train rollouts become SFT examples; validation and hidden grader keys are not
model inputs. Qwen SFT conversion uses the official template, preserves thinking,
normalizes JSON tool arguments, masks non-assistant tokens, and rejects oversized
sequences instead of truncating them. train.pt is a derived tokenization cache.

Choose sampling limits and models in the collection command. Your deployment
provides GPU allocation and job scheduling. Summarize evaluation episodes with
`python examples/murdoku/scripts/summarize_validation.py validation-dir`.

Run demo tests separately after installation:

```bash
python -m pytest -q examples/murdoku/tests
```

The dependency-free dataset/mask tests do not execute the external grader. Full
sandbox and reward qualification needs Linux bubblewrap and the Murdoku Lab checkout.
