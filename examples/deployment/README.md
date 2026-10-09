# Ray deployment example

The scheduler or cloud platform allocates machines and GPUs. Activate the same
Agent Horizon runtime on every node and choose a reachable head hostname.

```bash
export LONG_HORIZON_RAY_HEAD=head-hostname
export LONG_HORIZON_RAY_PORT=6379
export LONG_HORIZON_NODES=4
export LONG_HORIZON_GPUS_PER_NODE=2
export LONG_HORIZON_NUM_CPUS=24

# On the head:
bash examples/deployment/ray_cluster.sh head
# On each worker (with the same variables):
bash examples/deployment/ray_cluster.sh worker
```

On the head, launch a training profile separately, passing
`+ray_kwargs.ray_init.address="$RAY_ADDRESS"` and
`ray_kwargs.ray_init.num_cpus=null`. Export `RAY_ADDRESS=hostname:6379` in the
launching shell. Set actor and rollout node counts to match the allocation;
for dedicated rollout, their sum must fit the available nodes. The helper starts
Ray only; model selection, training settings, checkpoints and scheduling belong
to the caller. After training, run `ray stop` on each node before releasing it.
