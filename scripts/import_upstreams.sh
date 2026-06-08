#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${1:-${ROOT_DIR}/.tmp/upstreams}"

MINERAL_URL="https://github.com/etaoxing/mineral.git"
REWARPED_URL="https://github.com/rewarped/rewarped.git"

mkdir -p "${WORK_DIR}"

clone_or_refresh() {
    local url="$1"
    local dir="$2"

    if [ -d "${dir}/.git" ]; then
        git -C "${dir}" fetch --all --tags
        git -C "${dir}" pull --ff-only
    else
        git clone "${url}" "${dir}"
    fi
}

replace_tree() {
    local src="$1"
    local dest="$2"

    rm -rf "${dest}"
    mkdir -p "${dest}"
    cp -a "${src}/." "${dest}/"
    rm -rf "${dest}/.git"
}

clone_or_refresh "${MINERAL_URL}" "${WORK_DIR}/mineral"
clone_or_refresh "${REWARPED_URL}" "${WORK_DIR}/rewarped"

replace_tree "${WORK_DIR}/mineral" "${ROOT_DIR}/mineral"
replace_tree "${WORK_DIR}/rewarped" "${ROOT_DIR}/rewarped"

cat <<EOF
Upstream import complete.

Vendored sources refreshed in:
  ${ROOT_DIR}/mineral
  ${ROOT_DIR}/rewarped

Recommended next steps:
  ${ROOT_DIR}/scripts/verify_install.sh
  git -C "${ROOT_DIR}" status
EOF
