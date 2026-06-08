#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [ ! -d "${VENV_DIR}" ]; then
    echo "error: virtual environment not found at ${VENV_DIR}"
    echo "run ./scripts/setup_env.sh first"
    exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python - <<'PY'
import mineral
import rewarped

print("mineral import OK:", getattr(mineral, "__file__", "unknown"))
print("rewarped import OK:", getattr(rewarped, "__file__", "unknown"))
PY
