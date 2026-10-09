# Sampled-token on-policy distillation

`long_horizon_opd` reuses the pinned native `separate_async` trainer, teacher
server, sampled-token k1 loss and policy-gradient loss hook. Student trajectories provide the sampled tokens, including unsuccessful attempts.
Teacher inference scores the student's sampled response tokens under each
exact retained context. The server generates one unused token because its
prompt-logprob interface requires it; supervision covers the response tokens.

The group adapter scores every earlier segment with four bounded concurrent
requests. Native publication scores each final segment. A scoring failure
settles all outstanding local RPC tasks before group failure; it never publishes
partially scored earlier segments. Synthetic all-masked microbatches return a
zero differentiable loss instead of entering native empty min/max reductions.
Mixed real/padded microbatches retain the native masked loss.

## Four-GPU profile

Set `LONG_HORIZON_MODEL_PATH`, `LONG_HORIZON_TEACHER_PATH`,
`LONG_HORIZON_TRAIN_FILE` and `LONG_HORIZON_VAL_FILE`, then run:

```bash
bash scripts/train_opd.sh
```

The profile uses one trainer GPU, two independent student-serving GPUs and one
teacher GPU. Student CUDA graphs and vLLM async scheduling are enabled. The
teacher uses eager inference in this profile. Teacher-only k1 policy-gradient OPD,
learning rate 1e-6, one prompt with two student attempts, 50 turns, a 24k
per-turn generation cap and 64k student context are explicit choices. The
teacher capacity is 73,729: native distillation validates the configured 8k
prompt plus 64k response plus one token, even though the episode runner bounds
the student's full input to 64k. Context clears above 40,960 to 32,768, reserving
24k generation headroom. Thinking is retained in sampled history.

This profile uses teacher-only supervision. `episodes/reward_mean`,
`episodes/strict_success_mean`, `episodes/submitted_mean` and turn/token metrics
report actual task outcomes independently of native zero critic/reward tensors.
These observations are deduplicated across segments. Distillation loss is not
a success metric; use matched held-out tasks and seeds to measure improvement.

The strict-version profile uses warmup zero and version span one. For prefetch,
configure a larger inclusive version span and validate queue admission and policy
lag under the chosen episode budgets.

One outer step is not necessarily one optimizer update. With the native
minibatch setting `ppo_mini_batch_size * rollout.n = 2`, 32 emitted rows produce
16 optimizer minibatches per epoch. Microbatch size one bounds memory; it does
not change the optimizer boundary. The progress adapter records rows and
optimizer minibatches, and emits begin/end records for old-policy scoring,
actor updates and checkpoint saves. Trajectory-wide accumulation requires a profile that defines its numerical
weighting and optimizer boundary.

## Model and runtime configuration

Use student and teacher checkpoints with compatible token IDs. The teacher must
score each sampled student token under its retained context. Set teacher serving
capacity to cover the configured prompt/response bounds and prompt-logprob request.

`requirements/sandbox.txt` pins tool dependencies. `LONG_HORIZON_TOOL_RUNTIME`
selects the isolated scientific Python runtime. Configure checkpoint retention
and storage reserve for model, optimizer and queue state; see [recovery](recovery.md).

Evaluate model quality on held-out tasks under matching sampling and tool budgets.
Episode reward and strict success measure task outcomes independently of the
teacher-scoring loss.
