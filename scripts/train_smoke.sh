#!/bin/bash -l
# Short qualification profile using the shared launcher.
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$project_root/scripts/launch_train.sh" "$@"
