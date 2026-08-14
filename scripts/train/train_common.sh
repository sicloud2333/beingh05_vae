#!/usr/bin/env bash
# Shared, side-effect-light training environment setup.
# Source this file after set -euo pipefail in a training entry point.

TRAIN_COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_REPO_ROOT="$(cd -- "${TRAIN_COMMON_DIR}/../.." && pwd)"

export PYTHONPATH="${TRAIN_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-8}"

# Python/torchrun paths remain owned by each legacy-compatible entry point.
