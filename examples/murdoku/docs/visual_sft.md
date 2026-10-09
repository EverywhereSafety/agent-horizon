# Screenshot SFT preparation

The input is a JSONL conversation, with local screenshot references in their
original turns. This interface does not depend on the puzzle setter layout.

```json
{"prompt_uid":"new-001","tools":[],"messages":[{"role":"user","content":[{"type":"image","image":"screenshots/new-001.png"},{"type":"text","text":"Solve this puzzle."}]},{"role":"assistant","content":"The answer is ..."}]}
```

Use only validated solver trajectories for training. Keep hidden answers and
verification metadata outside messages. Images may occur in user or tool results;
retain their original ordering. Actual processor support for tool-result images
must be qualified before collecting interactive screenshot trajectories.

```bash
python examples/murdoku/scripts/prepare_visual_sft.py --input query_rollouts.jsonl \
  --image-root /path/to/dataset --processor /path/to/full-qwen35-4b \
  --output /path/to/prepared/train.pt

VISUAL_SFT_MODEL=/path/to/full-qwen35-4b \
VISUAL_SFT_DATA=/path/to/prepared/train.pt \
VISUAL_SFT_OUTPUT=/path/to/checkpoints bash examples/murdoku/scripts/train_visual_sft.sh
```

Preparation uses the student's official AutoProcessor, preserves recorded
reasoning and tool calls, emits pixel tensors and image grids, and masks all
non-assistant tokens. Sequence limits include expanded vision tokens; overlong
examples fail rather than truncate. veRL dataset output includes its four-row
text-plus-vision position convention and multi_modal_inputs.

The launcher uses veRL's SFT trainer directly: BF16 FSDP2, no-padding inputs,
gradient checkpointing, no activation offload, and its fused output/loss backend.
`VISUAL_SFT_FUSED_BACKEND=liger` is the default; veRL uses its chunked Torch path
when the optional Liger implementation is unavailable. Full-vocabulary logits
are not materialized for the entire long trajectory. This path supports mixed
text and visual inputs without freezing the vision tower.

`VISUAL_SFT_GPUS` defaults to 4 and the global batch defaults to that GPU count.
Set `VISUAL_SFT_BATCH` to a multiple of it for gradient accumulation. The prepared
dataset must contain complete global batches: veRL's dataloader drops an incomplete
last batch. The current launcher uses veRL's distributed sampler; cross-rank
length grouping is not supplied by this launcher.

Checkpoints use veRL's model/optimizer/data-progress save and resume machinery.
`VISUAL_SFT_SAVE_FREQ=-1` saves only at completion; use a positive update interval
when intermediate recovery is wanted. Resume defaults to `auto`; override
`trainer.resume_mode=disable` or use a fresh output directory for a new run.
An external walltime kill before a final-only save loses unsaved progress.

The veRL `engine.use_torch_compile` switch compiles its entropy helper rather
than the whole transformer, so it remains off for this SFT loss path. Pre-tokenized
examples are held in memory, so worker processes default to zero. Adjust attention,
sequence parallelism, dynamic token batches, or worker count through the usual
trailing veRL overrides after qualifying the chosen model and hardware. Qualify
longest text and image examples and include startup and export in the time budget.
A text-only student export needs compatible vision weights and processor for
visual inference.

The preparation command also accepts unified query + rollout JSONL directly.
It selects only replay-qualified **vision** training rows and reports selected
and excluded counts. The text preparation CLI selects only **text** training rows
(legacy records without a view are text). Both accept the same mixed JSONL.
The visual CLI resolves the query images under
`--image-root` before checking the recorded initial prompt. Existing flattened
SFT examples remain supported.
