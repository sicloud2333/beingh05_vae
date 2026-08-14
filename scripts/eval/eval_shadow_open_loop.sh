#!/usr/bin/env bash
# Unified offline open-loop evaluator. The Python implementation and vae/
# sources are intentionally left untouched.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
BEINGH_ENV="${BEINGH_ENV:-/opt/conda/envs/beingh05}"
PYTHON_BIN="${PYTHON_BIN:-${BEINGH_ENV}/bin/python}"
EVAL_IMPL="${REPO_ROOT}/scripts/eval/eval_shadow_grasp_dataset_open_loop.py"

[[ -x "${PYTHON_BIN}" ]] || {
  echo "[eval-shadow-open-loop] Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
}
[[ -f "${EVAL_IMPL}" ]] || {
  echo "[eval-shadow-open-loop] Evaluator not found: ${EVAL_IMPL}" >&2
  exit 2
}

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

exec "${PYTHON_BIN}" -u "${EVAL_IMPL}" "$@"
