#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/rewarped:${PYTHONPATH:-}"

# python -c "import rewarped; print(rewarped.__spec__)"
# python -c "import rewarped.envs.dflex.go2 as m; print(m.__file__)"
# python -c "from rewarped.warp import model_monkeypatch as m; print(m.__file__)"

python -m rewarped.envs.dflex.go2

