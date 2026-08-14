#!/usr/bin/env bash
# Unified MuJoCo evaluator entry point for Being-H05 Shadow-grasp experiments.
# The implementation remains in vae/; this wrapper only standardizes the
# environment and defaults while keeping every low-level argument available.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
BEINGH_ENV="${BEINGH_ENV:-/opt/conda/envs/beingh05}"
PYTHON_BIN="${PYTHON_BIN:-${BEINGH_ENV}/bin/python}"
EVAL_IMPL="${REPO_ROOT}/vae/examples/beingh_shadow_grasp_eval.py"

[[ -x "${PYTHON_BIN}" ]] || {
  echo "[eval-shadow] Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
}
[[ -f "${EVAL_IMPL}" ]] || {
  echo "[eval-shadow] Evaluator not found: ${EVAL_IMPL}" >&2
  exit 2
}

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# commanded is the project-level default. An explicit user argument always
# wins, so encoded-mode historical experiments remain reproducible.
latent_mode_seen=0
for arg in "$@"; do
  case "${arg}" in
    --latent-observation-mode|--latent-observation-mode=*)
      latent_mode_seen=1
      break
      ;;
  esac
done

extra_args=()
if (( latent_mode_seen == 0 )); then
  extra_args+=(--latent-observation-mode commanded)
fi

exec "${PYTHON_BIN}" -u "${EVAL_IMPL}" "${extra_args[@]}" "$@"
