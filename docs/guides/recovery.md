# Episode recovery, cancellation and tool execution

## Recovery contract

Native verl/TransferQueue restore prompt scheduling and trainer state. The project
continuation additionally restores the environment, model-visible context,
operational memory, complete training segments, audit offset and next turn from
an atomic snapshot every ten turns. A restarted episode rolls back to that saved
boundary; it does not attach saved conversation to a freshly initialized state.
Completed replay caches preserve original sampled tokens and log-probabilities.
This is checkpoint-boundary recovery, not exactly-once execution of all actions
since that boundary.

The generated-environment sandbox serializes the entire `env.__dict__`, including
nested dynamics objects, Python random state, the supported `_rng_module` RNG
state, database and declared `environment_state_globals`. The supplied
compound inventory example includes inventory, periods_done and pending next
inventory; restoring into a fresh sandbox reproduces subsequent observations.
This is not a universal snapshot adapter for arbitrary generated programs:
undeclared mutable globals, class attributes, additional RNGs and external
services need explicit contracts. Slot-only/custom environments can select
`environment_snapshot_mode: hooks` and implement `snapshot_state()` /
`restore_state(state)`. These hooks are responsible for complete state capture;
selecting hooks does not prove their completeness.

## Tools and side effects

Environment actions are sequential RPCs to the same isolated process. An RPC
timeout closes the process and fails the trajectory. It is not automatically
retried against potentially mutated state. Recovery rolls back the entire local
environment to its last snapshot. This policy applies to sandbox-local effects;
external services require action IDs, receipts/deduplication or an explicit
idempotence protocol before retry/recovery is supported.

Model `run_python` is stateless and has no environment handle. Execution timeout
kills and reaps its process and returns a nonterminal tool-error observation.
Cancellation still propagates and cleans up. Worker startup/transport failures
remain infrastructure errors. Tool observations that exceed the current context
close the bounded training segment; subsequent generation compacts complete
interaction groups. Sampled assistant tokens remain trainable and the full
observation remains audited.

`episode_timeout_seconds` is optional and disabled by default. When configured,
it bounds episode execution including model/tool waits, propagates cancellation
and emits a typed `.termination.json` receipt. Committed elapsed execution time
is carried in continuations rather than reset on resume. Work since the latest
checkpoint can be rolled back; elapsed accounting has that same granularity.
Token/turn limits alone do not bound wall-clock time.

## Permanent cancellation

The production group worker selects the first valid completions and marks all
losers with durable tombstones before stopping them. Generation cancellation
uses the native targeted vLLM abort API; local environment/Python workers are
closed/reaped. The group publishes only after cleanup acknowledgements. Abort
failure or cleanup timeout prevents successful publication. Cancelled IDs cannot
be resumed or admitted later. Completed-but-unselected attempts are tombstoned
without pretending they needed a live server abort. Failed groups require new
attempt IDs; do not erase tombstones to recycle cancelled attempts.

Native temporary interruption for weight synchronization uses the backend's
synchronization and checkpoint lifecycle.

## Staleness units

The pinned native replay buffer uses `current_step - dispatch_step + 1` for its
version span. A maximum span of four corresponds to a maximum version difference
of three for the admission comparison. The launcher sets this span to four. Record dispatch, minimum/maximum served and
current versions to understand admission and eviction.

## Validate recovery

Exercise interruption during environment mutation, restart from a saved boundary,
cancellation during model generation and tool execution, late result rejection,
and rollout-node failure during weight synchronization. Compare resumed results
and exact tokens/masks where the replay policy makes equality meaningful.

## Checkpoint storage peak

For project trainers using `PersistentRetentionCallback`, worker-level pruning
is disabled during serialization; complete step directories are removed only
by the callback after model, optimizer, dataloader, queue and latest pointer
commit. A save failure retains the prior complete checkpoint.

Keeping one checkpoint still requires space for two full training-state saves
while writing. Keeping two can require three. Estimate model, optimizer, queue and exported-weight sizes and check both
filesystem space and account quota. Failed saves can leave large
partial directories; inspect the committed pointer and retain the previous
complete checkpoint before deleting a partial save. SFT inference HF weights
can be retained without its duplicate training/optimizer shards after SFT ends.

The bounded OPD example sets `checkpoint_min_free_bytes` to 64 GiB; the generic
framework leaves the reserve configurable for each workload. The
progress adapter checks physical free space and, when `lfs quota` is available,
remaining user quota before serialization. The configured reserve gates checkpoint serialization.
Choose the reserve from actual checkpoint size and leave headroom for audits and
concurrent jobs. Storage preflight does not delete checkpoints to manufacture space.

## Final model checkpoint

The long-horizon RL launcher defaults to `trainer.save_freq=100`. The pinned PPO
trainer also saves on the final update when this value is positive, even if the
run ends before that interval. Trailing overrides take precedence;
`trainer.save_freq=-1` explicitly disables all PPO saves, including the final one.
SFT retains its separate trainer's final-save behavior.
