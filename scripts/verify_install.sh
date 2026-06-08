#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v python >/dev/null 2>&1; then
    echo "error: python not found in current environment"
    echo "activate your Conda environment first, then rerun this script"
    exit 1
fi

cd /tmp

python - <<'PY'
from mineral.scripts import run as mineral_run
from rewarped import environment as rewarped_environment

print("mineral import OK:", getattr(mineral_run, "__file__", "unknown"))
print("rewarped import OK:", getattr(rewarped_environment, "__file__", "unknown"))
PY
