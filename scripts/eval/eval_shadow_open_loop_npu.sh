#!/usr/bin/env bash
# Ascend NPU wrapper for the offline Shadow open-loop evaluator.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ASCEND_DRIVER_ENV="${ASCEND_DRIVER_ENV:-/usr/local/Ascend/driver/bin/setenv.bash}"
ASCEND_CANN_ENV="${ASCEND_CANN_ENV:-/home/developer/Ascend/cann-9.2.0/set_env.sh}"
NPU_DEVICE="${NPU_DEVICE:-npu:0}"

[[ -f "${ASCEND_DRIVER_ENV}" ]] || {
  echo "[eval-shadow-open-loop-npu] Driver environment not found: ${ASCEND_DRIVER_ENV}" >&2
  exit 2
}
[[ -f "${ASCEND_CANN_ENV}" ]] || {
  echo "[eval-shadow-open-loop-npu] CANN environment not found: ${ASCEND_CANN_ENV}" >&2
  exit 2
}

# Ascend's vendor scripts read optional variables such as LD_LIBRARY_PATH.
# Temporarily disable nounset while sourcing them, then restore strict mode.
set +u
# shellcheck disable=SC1090
source "${ASCEND_DRIVER_ENV}"
# shellcheck disable=SC1090
source "${ASCEND_CANN_ENV}"
set -u

if [[ -z "${PYTHON_BIN:-}" ]]; then
  BEINGH_ENV="${BEINGH_ENV:-/opt/conda/envs/beingh05-npu}"
  PYTHON_BIN="${BEINGH_ENV}/bin/python"
fi
export PYTHON_BIN

[[ -x "${PYTHON_BIN}" ]] || {
  echo "[eval-shadow-open-loop-npu] Python executable not found: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN or BEINGH_ENV to an environment containing torch_npu." >&2
  exit 2
}

"${PYTHON_BIN}" -c 'import torch, torch_npu; assert torch.npu.is_available()' || {
  echo "[eval-shadow-open-loop-npu] torch_npu is unavailable in ${PYTHON_BIN}" >&2
  exit 3
}

exec "${SCRIPT_DIR}/eval_shadow_open_loop.sh" --device "${NPU_DEVICE}" "$@"
