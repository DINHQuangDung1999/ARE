#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v python >/dev/null 2>&1; then
    echo "error: python not found in current environment"
    exit 1
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install opacus
python -m pip install warp-lang==1.7.0
python -m pip install -e "${ROOT_DIR}/rewarped"
python -m pip install -e "${ROOT_DIR}/mineral"

cat <<EOF
Setup complete.

Next steps:
  "${ROOT_DIR}/scripts/verify_install.sh"
EOF
