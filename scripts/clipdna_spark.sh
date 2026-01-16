#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

MODE=dgx

cd "${ROOT_DIR}"

exec "${ROOT_DIR}/scripts/clipdna_wizard.py" "${MODE}"
