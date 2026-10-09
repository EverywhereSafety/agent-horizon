# Asynchronous execution

`separate_async` provides separate rollout resources, a versioned TransferQueue,
checkpointed in-flight prompts and weight synchronization. Prefetch supplies work to overlap generation with actor updates.

The native trainer submits one batch in `prepare_step`, waits for a completed
batch, then computes old log-probabilities and updates the actor. With
`num_warmup_batches=0`, it has no extra batch to generate during the actor update.
The ordinary long-horizon launcher now requests one warmup batch. Native warmup
counts restored queue groups before adding prompts, so resume does not blindly
duplicate an already populated prefetch window. The current batch is submitted
in addition to this window; while the trainer consumes one, another can run.

The inclusive version span stays four (`current - dispatch + 1`), permitting
at most three versions of dispatch lag. The project validates that span exceeds
the configured warmup batch count. Slow episodes can exceed the admission window and be evicted.
Native admission, eviction and behavior-probability correction remain active.
Changing version lag can change the learning distribution and requires a
controlled comparison; it is not solely a throughput setting.

`single_pass_unique_queries` rejects warmup above zero because the upstream
prefetch path can fetch beyond that experiment's fixed dataset boundary.
Drain-only recovery also retains its no-refill contract. Strict-version OPD
keeps warmup zero.

Student vLLM async scheduling and CUDA graphs are separate from this training
prefetch mechanism. The ordinary launcher enables both; the bounded smoke
launcher remains a conservative diagnostic baseline. Offload and model/kernel
backend decisions remain explicit overrides, not an automatic capacity policy.

## Measure overlap

Use stage logs and GPU activity on the same time axis. Compare rollout activity
with old-policy scoring, actor updates and checkpoint saves. The
`summarize_infra_overlap.py` utility aggregates recorded activity. Track dispatch,
served and current policy versions alongside queue depth and stale-group drops.

Enable vLLM async scheduling and CUDA graphs independently of training prefetch.
Choose actor precision, offload and kernels from the model and hardware profile.
[Runtime checks](../reference/runtime-checks.md) and [capacity planning](context.md)
provide the validation entry points.
