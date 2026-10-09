#!/bin/bash -l
# Keep the qualified dense/FSDP environment untouched; resolve upstream's lock.
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export UV_CACHE_DIR="${UV_CACHE_DIR:-$project_root/../uv_cache}"
export UV_PROJECT_ENVIRONMENT="${LONG_HORIZON_MEGATRON_ENV:-$project_root/.venv-megatron}"
if test "$UV_PROJECT_ENVIRONMENT" = "$project_root/.venv"; then
 echo 'Megatron qualification requires a separate environment' >&2; exit 1
fi
cd "$project_root/vendor/verl"
# The original uv.lock has pre-migration wheelhouse URLs. Resolve all ordinary
# dependencies from that lock, then install identical versions from verified
# per-Python/per-Torch release assets. No vendor edits or Torch upgrade.
uv sync --frozen --extra megatron --extra vllm --python 3.12 \
 --no-install-package flash-attn --no-install-package apex \
 --no-install-package transformer-engine --no-install-package megatron-bridge
mapfile -t wheels < <("$project_root/.venv/bin/python" "$project_root/scripts/backends/fetch_megatron_wheels.py")
if test "${#wheels[@]}" -ne 4; then
 echo 'Failed to verify all four Megatron qualification wheels' >&2; exit 1
fi
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps "${wheels[@]}"
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps -e "$project_root"
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps \
 'TransferQueue @ git+https://github.com/Ascend/TransferQueue.git@7b31c0b6147413bf4b55d0a49e197c85250d814b'
# Hybrid GDN models require FLA even though the upstream Megatron extra does
# not include it. Match the qualified FSDP environment without upgrading Torch.
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps \
 flash-linear-attention==0.5.2 fla-core==0.5.2
# Import/kernel qualification must run on a compute GPU, with CUDA loaded.
