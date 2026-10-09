# Context and memory

An episode can span many context windows. Clearing bounds the current model input
while preserving task state, memory and the sampled actions used for training.

![Context clears preserve state and earlier training segments](../assets/context-continuity.svg)

## State

| State | Lifetime |
|---|---|
| Model input | Initial task plus retained complete interactions |
| Memory | Persists across clears; retrieved through the memory interface |
| Environment | Persists across tools and clears |
| Training segments | Retain sampled tokens and behavior probabilities throughout the episode |

`max_context_tokens` limits the full model input, including the initial prompt,
tool schemas and visible memory. `clear_trigger_tokens` initiates compaction;
`clear_target_tokens` sets the retained-input target. The runtime removes oldest
complete interaction groups while protecting the initial task. Generation is
bounded by remaining serving capacity and `max_new_tokens`.

A conditioning change closes the current training segment and starts another.
Earlier segments remain trainable. Tool observations and scaffold tokens have
zero policy-loss mask; sampled assistant tokens retain their exact IDs and
behavior log-probabilities. Parent trajectory identity keeps rewards and
advantages associated with the episode across its segments.

## Configure

See [agent_long_horizon.yaml](../../configs/agent_long_horizon.yaml). Context window,
assistant-turn budget, per-turn generation and cumulative generated-token budget
are independent settings. Align the serving backend's `max_model_len` with the
model input window.

## Memory and feedback

Explicit memory operations allow the model to save and retrieve notes. Environment
plugins can supply task-specific memory tools. The core also supports operational
memory and optional semantic archival retrieval.

`context_feedback_enabled` adds a budget reminder, clear notification and bounded
memory-title preview. `context_warning_margin_tokens` sets the reminder margin;
`context_memory_title_limit` defaults to 32. Notification tokens count toward the
model input and receive zero policy loss. Feedback state is checkpointed with the
episode. A large tool observation can cross the trigger before a reminder is sent.

Semantic retrieval is enabled with `memory_embedding_model` and an immutable
`memory_embedding_revision`; `memory_embedding_max_tokens` defaults to 256.
The local CPU encoder uses mean pooling and L2 normalization. Archive vectors
and text identities are retained in continuation state; updated entries are
re-embedded. Stage the embedding model before starting episodes.

## Continuation

An episode snapshot includes context groups, memory, environment state, training
segments and rollout progress. Resume restores a consistent saved boundary.
[Recovery](recovery.md) describes snapshot hooks, cancellation and external effects.

[Runtime checks](../reference/runtime-checks.md) · [Environment interfaces](environments.md)

## GPU capacity

The model context window bounds one forward pass. Cumulative episode tokens can
span many windows through context clearing and retained training segments.

GPU memory depends on model size, attention implementation, sequence length,
concurrency, actor sharding, optimizer state and offload. Rollout serving and actor
updates have different capacity requirements.

Use the training smoke launcher with your model and intended context to exercise
the full serving/update path. Record peak
allocated/reserved memory and timing for the selected precision, kernels and
resource layout. Measure initialization separately from steady-state updates.

Configure the serving context, episode context and generation headroom together.
Increase batch or context after checking both rollout KV-cache capacity and actor
training capacity. Retain completed-episode timing, generated tokens and tool
latency alongside GPU measurements.

[Configuration](../../configs/agent_long_horizon.yaml) ·
[Runtime checks](../reference/runtime-checks.md) · [Context management](context.md)
