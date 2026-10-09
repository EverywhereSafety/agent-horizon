#!/bin/bash -l
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
upstream_commit=1a8a0f5ffd9d3f169ae4432b68526233ba028102
mkdir -p "$project_root/vendor"
if ! test -d "$project_root/vendor/verl/.git"; then
 git clone --filter=blob:none --no-checkout https://github.com/volcengine/verl.git "$project_root/vendor/verl"
 git -C "$project_root/vendor/verl" checkout "$upstream_commit"
fi
if test "$(git -C "$project_root/vendor/verl" rev-parse HEAD)" != "$upstream_commit"; then
 echo 'Unexpected upstream revision; refusing to overwrite checkout' >&2; exit 1
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$project_root/../uv_cache}"
export UV_PROJECT_ENVIRONMENT="$project_root/.venv"
cd "$project_root/vendor/verl"
uv sync --frozen --extra vllm --extra fsdp --no-install-package flash-attn --python 3.12
cd "$project_root"
uv pip install --python "$project_root/.venv/bin/python" --no-deps -e .
uv pip install --python "$project_root/.venv/bin/python" --no-deps \
 'TransferQueue @ git+https://github.com/Ascend/TransferQueue.git@7b31c0b6147413bf4b55d0a49e197c85250d814b'

uv venv "$project_root/.sandbox-env" --python "$project_root/.venv/bin/python"
uv pip sync --python "$project_root/.sandbox-env/bin/python" "$project_root/requirements/sandbox.txt"
