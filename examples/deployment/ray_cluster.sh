#!/usr/bin/env bash
# Start Ray on an allocation supplied by the deployment. Launch training separately.
set -euo pipefail
role="${1:?Usage: ray_cluster.sh head|worker}"
: "${LONG_HORIZON_RAY_HEAD:?Set the Ray head hostname}"
export LONG_HORIZON_RAY_PORT="${LONG_HORIZON_RAY_PORT:-6379}"
export RAY_ADDRESS="$LONG_HORIZON_RAY_HEAD:$LONG_HORIZON_RAY_PORT"
export LONG_HORIZON_NODES="${LONG_HORIZON_NODES:-1}"
export LONG_HORIZON_GPUS_PER_NODE="${LONG_HORIZON_GPUS_PER_NODE:-1}"
cpus="${LONG_HORIZON_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-8}}"
case "$role" in
    worker)
        exec ray start --address="$RAY_ADDRESS" --num-cpus="$cpus" \
            --num-gpus="$LONG_HORIZON_GPUS_PER_NODE" --block
        ;;
    head)
        ray start --head --port="$LONG_HORIZON_RAY_PORT" --num-cpus="$cpus" \
            --num-gpus="$LONG_HORIZON_GPUS_PER_NODE" --disable-usage-stats
        python - <<'CHECK'
import os, time, ray
ray.init(address=os.environ['RAY_ADDRESS'])
expected = int(os.environ['LONG_HORIZON_NODES'])
gpus = int(os.environ['LONG_HORIZON_GPUS_PER_NODE'])
try:
    for _ in range(120):
        nodes = [n for n in ray.nodes() if n['Alive'] and n['Resources'].get('GPU', 0) >= gpus]
        if len(nodes) >= expected:
            print(f'Ray ready: {len(nodes)} nodes, {gpus} GPUs per node', flush=True)
            break
        time.sleep(2)
    else:
        raise RuntimeError(f'Expected {expected} Ray nodes with at least {gpus} GPUs each')
finally:
    ray.shutdown()
CHECK
        ;;
    *) echo "Unknown role: $role" >&2; exit 2 ;;
esac
