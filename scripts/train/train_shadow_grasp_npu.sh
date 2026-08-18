#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

ASCEND_DRIVER_ENV="${ASCEND_DRIVER_ENV:-/usr/local/Ascend/driver/bin/setenv.bash}"
ASCEND_CANN_ENV="${ASCEND_CANN_ENV:-/home/developer/Ascend/cann-9.2.0/set_env.sh}"
BEINGH_ENV="${BEINGH_ENV:-/mnt/workspace/gitCode/cann/hsz/factory_dex/.venv-ascend}"

[[ -f "${ASCEND_DRIVER_ENV}" ]] || {
  printf 'Missing Ascend driver environment: %s\n' "${ASCEND_DRIVER_ENV}" >&2
  exit 2
}
[[ -f "${ASCEND_CANN_ENV}" ]] || {
  printf 'Missing CANN environment: %s\n' "${ASCEND_CANN_ENV}" >&2
  exit 2
}
[[ -x "${BEINGH_ENV}/bin/python" ]] || {
  printf 'Missing training Python: %s/bin/python\n' "${BEINGH_ENV}" >&2
  exit 2
}

# Ascend's vendor environment scripts append to variables such as
# LD_LIBRARY_PATH without first assigning them. Temporarily relax nounset so
# this wrapper remains safe when started from a clean non-interactive shell.
set +u
source "${ASCEND_DRIVER_ENV}"
source "${ASCEND_CANN_ENV}"
set -u

export ACCELERATOR=npu
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
export NUM_GPUS="${NUM_GPUS:-2}"
export BEINGH_ENV
export PYTHON_BIN="${PYTHON_BIN:-${BEINGH_ENV}/bin/python}"
export TORCHRUN_BIN="${TORCHRUN_BIN:-${BEINGH_ENV}/bin/torchrun}"
export FUSED_OPTIMIZER="${FUSED_OPTIMIZER:-False}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# Long NPU runs should retain optimizer/scheduler state so they can resume
# after interruption. SMOKE_TEST/PERF_TEST still force model-only saves in the
# shared launcher.
export SAVE_MODEL_ONLY="${SAVE_MODEL_ONLY:-False}"
export RESUME_MODEL_ONLY="${RESUME_MODEL_ONLY:-True}"
# This server exposes only 64 MiB of /dev/shm. Loading packed image batches in
# worker subprocesses can therefore raise SIGBUS; main-process loading avoids
# shared-memory transport and is the safe NPU default. Callers with a larger
# shm mount can override NUM_WORKERS explicitly.
export NUM_WORKERS="${NUM_WORKERS:-0}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
export SMOKE_NUM_WORKERS="${SMOKE_NUM_WORKERS:-0}"
export SMOKE_PREFETCH_FACTOR="${SMOKE_PREFETCH_FACTOR:-1}"

# The formal two-rank smoke also exercises sharded optimizer checkpointing.
# Single-rank smoke remains model-only unless explicitly overridden.
if [[ "${SMOKE_TEST:-False}" =~ ^([Tt]rue)$ ]] && (( NUM_GPUS > 1 )); then
  export SMOKE_SAVE_MODEL_ONLY="${SMOKE_SAVE_MODEL_ONLY:-False}"
else
  export SMOKE_SAVE_MODEL_ONLY="${SMOKE_SAVE_MODEL_ONLY:-True}"
fi

cd "${REPO_ROOT}"
exec bash "${SCRIPT_DIR}/train_shadow_grasp.sh" "$@"
