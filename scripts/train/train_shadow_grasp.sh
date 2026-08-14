#!/usr/bin/env bash
# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Shared environment defaults; dataset and optimization settings remain below.
source "${SCRIPT_DIR}/train_common.sh"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-8}"

log() {
  printf '[shadow-grasp] %s\n' "$*"
}

die() {
  printf '[shadow-grasp] ERROR: %s\n' "$*" >&2
  exit 2
}

on_exit() {
  status=$?
  if (( status != 0 )); then
    printf '[shadow-grasp] FAILED (exit=%d). Log: %s\n' \
      "${status}" "${LOG_FILE:-not-created}" >&2
  fi
}
trap on_exit EXIT

# =============================================================================
# Dataset selection and local paths
# Every value can be overridden as an environment variable.
# =============================================================================
EMBODIMENT="${EMBODIMENT:-shadow_grasp}"
EMBODIMENT_DATASET="${EMBODIMENT_DATASET:-shadow_grasp_0725_core_bottle_1071}"
DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/outputs/shadow_grasp_bottle_1071"
if [[ "${EMBODIMENT_DATASET}" != "shadow_grasp_0725_core_bottle_1071" ]]; then
  DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/outputs/${EMBODIMENT_DATASET}"
fi

PRETRAIN_MODEL="${PRETRAIN_MODEL:-${REPO_ROOT}/ckpts/InternVL3_5-2B}"
EXPERT_MODEL="${EXPERT_MODEL:-${REPO_ROOT}/ckpts/Qwen3-0.6B}"
RESUME_PATH="${RESUME_PATH:-${REPO_ROOT}/ckpts/Being-H05-2B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(command -v torchrun)}"
BEINGH_ENV="${BEINGH_ENV:-$(dirname "$(dirname "${PYTHON_BIN}")")}"

# =============================================================================
# Shadow Grasp dataset
# =============================================================================
NORMALIZATION="${NORMALIZATION:-q99}"

case "${NORMALIZATION}" in
  q99)
    DEFAULT_DATASET_CONFIG_FILE="${REPO_ROOT}/configs/posttrain/${EMBODIMENT}/${EMBODIMENT_DATASET}_q99.yaml"
    ;;
  minmax)
    DEFAULT_DATASET_CONFIG_FILE="${REPO_ROOT}/configs/posttrain/${EMBODIMENT}/${EMBODIMENT_DATASET}_minmax.yaml"
    ;;
  wrist_minmax_zraw)
    DEFAULT_DATASET_CONFIG_FILE="${REPO_ROOT}/configs/posttrain/${EMBODIMENT}/${EMBODIMENT_DATASET}_wrist_minmax_zraw.yaml"
    ;;
  wrist_euler_minmax_zraw)
    DEFAULT_DATASET_CONFIG_FILE="${REPO_ROOT}/configs/posttrain/${EMBODIMENT}/${EMBODIMENT_DATASET}_wrist_euler_minmax_zraw.yaml"
    ;;
  wrist_rot6d_minmax_zraw)
    DEFAULT_DATASET_CONFIG_FILE="${REPO_ROOT}/configs/posttrain/${EMBODIMENT}/${EMBODIMENT_DATASET}_wrist_rot6d_minmax_zraw.yaml"
    ;;
  wrist_rot6d_minmax_joints)
    DEFAULT_DATASET_CONFIG_FILE="${REPO_ROOT}/configs/posttrain/${EMBODIMENT}/${EMBODIMENT_DATASET}_wrist_rot6d_minmax_joints.yaml"
    ;;
  none)
    DEFAULT_DATASET_CONFIG_FILE="${REPO_ROOT}/configs/posttrain/${EMBODIMENT}/${EMBODIMENT_DATASET}.yaml"
    ;;
  *)
    die "NORMALIZATION must be 'q99', 'minmax', 'wrist_minmax_zraw', 'wrist_euler_minmax_zraw', 'wrist_rot6d_minmax_zraw', 'wrist_rot6d_minmax_joints', or 'none', got: ${NORMALIZATION}"
    ;;
esac

DATASET_CONFIG_FILE="${DATASET_CONFIG_FILE:-${DEFAULT_DATASET_CONFIG_FILE}}"
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/data/${EMBODIMENT_DATASET}}"
SAVE_MERGED_META="${SAVE_MERGED_META:-True}"
GENERATE_STATS="${GENERATE_STATS:-True}"

# =============================================================================
# Distributed training
#
# Current host CUDA ordering:
#   0-3 = A100-SXM4-80GB
#   4   = H100 PCIe
#
# Use A100 devices 1 and 2 by default. Verify both are free before launching.
# =============================================================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-29107}"

# =============================================================================
# Optimization
# The 52-episode dataset is small, so use a conservative default run.
# =============================================================================
MAX_STEPS="${MAX_STEPS:-40000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_STEPS_START="${SAVE_STEPS_START:-15000}"
SAVE_MODEL_ONLY="${SAVE_MODEL_ONLY:-True}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
FUSED_OPTIMIZER="${FUSED_OPTIMIZER:-True}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SHARDING_STRATEGY="${SHARDING_STRATEGY:-SHARD_GRAD_OP}"

# =============================================================================
# Data loading and packed-sequence configuration
# Four workers are per rank, so the default below creates 12 workers total.
# A larger packed batch was benchmarked and reduced samples/s on this workload.
# =============================================================================
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-8704}"
EXPECTED_NUM_TOKENS="${EXPECTED_NUM_TOKENS:-8192}"
PREFER_BUFFER_BEFORE="${PREFER_BUFFER_BEFORE:-4096}"
MAX_BUFFER_SIZE="${MAX_BUFFER_SIZE:-4}"
ATTN_MODE="${ATTN_MODE:-causal}"

# =============================================================================
# Image and action configuration
# =============================================================================
FORCE_IMAGE_SIZE="${FORCE_IMAGE_SIZE:-224}"
MAX_VIEW_NUM="${MAX_VIEW_NUM:--1}"
USE_FIXED_VIEW="${USE_FIXED_VIEW:-False}"
DOWN_SAMPLE_RATIO="${DOWN_SAMPLE_RATIO:-0.5}"
ACTION_CHUNK_LENGTH="${ACTION_CHUNK_LENGTH:-16}"
WRIST_ACTION_LOSS_WEIGHT="${WRIST_ACTION_LOSS_WEIGHT:-1.0}"
DEFAULT_WRIST_ACTION_DIM=6
if [[ "${NORMALIZATION}" == "wrist_rot6d_minmax_zraw" \
   || "${NORMALIZATION}" == "wrist_rot6d_minmax_joints" ]]; then
  DEFAULT_WRIST_ACTION_DIM=9
fi
WRIST_ACTION_DIM="${WRIST_ACTION_DIM:-${DEFAULT_WRIST_ACTION_DIM}}"
TEMPORAL_DELTA_LOSS_WEIGHT="${TEMPORAL_DELTA_LOSS_WEIGHT:-0.0}"
TEMPORAL_DELTA_HUBER_BETA="${TEMPORAL_DELTA_HUBER_BETA:-1.0}"

# =============================================================================
# Freezing
# Freeze the pretrained VLM and connector by default for this small dataset.
# The action expert, robot encoders/decoder and MPG remain trainable.
# =============================================================================
FREEZE_MLLM="${FREEZE_MLLM:-True}"
FREEZE_VIT_MLP="${FREEZE_VIT_MLP:-True}"

# =============================================================================
# MPG
# =============================================================================
USE_MPG="${USE_MPG:-True}"
MPG_LAMBDA="${MPG_LAMBDA:-0.1}"
MPG_NUM_PROJECTIONS="${MPG_NUM_PROJECTIONS:-32}"
MPG_REFINEMENT_ITERS="${MPG_REFINEMENT_ITERS:-1}"
MPG_GATE_TEMPERATURE="${MPG_GATE_TEMPERATURE:-1.0}"
MPG_USE_STOP_GRADIENT="${MPG_USE_STOP_GRADIENT:-True}"

# =============================================================================
# Training-Time RTC
# =============================================================================
USE_TRAINING_TIME_RTC="${USE_TRAINING_TIME_RTC:-False}"
SIMULATED_DELAY="${SIMULATED_DELAY:-0}"
RTC_DELAY_EXP_WEIGHT="${RTC_DELAY_EXP_WEIGHT:-True}"
USE_INFERENCE_PREFIX_OVERWRITE="${USE_INFERENCE_PREFIX_OVERWRITE:-True}"

# =============================================================================
# Execution modes
#
# SMOKE_TEST=True:
#   Run 20 steps and save one model-only checkpoint.
#
# PREFLIGHT_ONLY=True:
#   Validate paths/data and generate stats.json, but do not launch torchrun.
# =============================================================================
SMOKE_TEST="${SMOKE_TEST:-False}"
PERF_TEST="${PERF_TEST:-False}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-False}"

if [[ "${SMOKE_TEST}" == "True" || "${SMOKE_TEST}" == "true" ]]; then
  MAX_STEPS="${SMOKE_MAX_STEPS:-20}"
  SAVE_STEPS="${SMOKE_SAVE_STEPS:-20}"
  SAVE_STEPS_START="${SMOKE_SAVE_STEPS_START:-0}"
  SAVE_MODEL_ONLY=True
  NUM_WORKERS="${SMOKE_NUM_WORKERS:-2}"
  PREFETCH_FACTOR="${SMOKE_PREFETCH_FACTOR:-2}"
fi

if [[ "${PERF_TEST}" == "True" || "${PERF_TEST}" == "true" ]]; then
  MAX_STEPS="${PERF_MAX_STEPS:-30}"
  SAVE_STEPS="${PERF_SAVE_STEPS:-30}"
  SAVE_STEPS_START="${PERF_SAVE_STEPS_START:-30}"
  LOGGING_STEPS="${PERF_LOGGING_STEPS:-5}"
  SAVE_MODEL_ONLY=True
fi

# =============================================================================
# Preflight validation
# =============================================================================
command -v "${PYTHON_BIN}" >/dev/null 2>&1 \
  || die "Python executable not found: ${PYTHON_BIN}. Activate the beingh environment."
command -v "${TORCHRUN_BIN}" >/dev/null 2>&1 \
  || die "torchrun executable not found: ${TORCHRUN_BIN}. Activate the beingh environment."

for model_dir in "${PRETRAIN_MODEL}" "${EXPERT_MODEL}" "${RESUME_PATH}"; do
  [[ -d "${model_dir}" ]] || die "Model directory not found: ${model_dir}"
  [[ -f "${model_dir}/config.json" ]] || die "Missing ${model_dir}/config.json"

  model_weight_found=False
  for model_weight in "${model_dir}"/*.safetensors; do
    if [[ -f "${model_weight}" ]]; then
      model_weight_found=True
      break
    fi
  done
  [[ "${model_weight_found}" == "True" ]] \
    || die "No safetensors found in ${model_dir}"
done

[[ -f "${PRETRAIN_MODEL}/tokenizer_config.json" ]] \
  || die "Missing ${PRETRAIN_MODEL}/tokenizer_config.json"
[[ -f "${EXPERT_MODEL}/tokenizer_config.json" ]] \
  || die "Missing ${EXPERT_MODEL}/tokenizer_config.json"
[[ -f "${RESUME_PATH}/tokenizer_config.json" ]] \
  || die "Missing ${RESUME_PATH}/tokenizer_config.json"
[[ -f "${DATASET_CONFIG_FILE}" ]] \
  || die "Dataset config not found: ${DATASET_CONFIG_FILE}"

for meta_file in info.json episodes.jsonl episodes_stats.jsonl tasks.jsonl; do
  [[ -f "${DATASET_PATH}/meta/${meta_file}" ]] \
    || die "Dataset metadata not found: ${DATASET_PATH}/meta/${meta_file}"
done

PARQUET_COUNT="$(find "${DATASET_PATH}/data" -type f -name '*.parquet' | wc -l)"
VIDEO_COUNT="$(find "${DATASET_PATH}/videos" -type f -name '*.mp4' | wc -l)"
(( PARQUET_COUNT > 0 )) || die "No parquet files found under ${DATASET_PATH}/data"
(( VIDEO_COUNT > 0 )) || die "No MP4 files found under ${DATASET_PATH}/videos"

IFS=',' read -r -a SELECTED_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
(( NUM_GPUS > 0 )) || die "NUM_GPUS must be positive"
(( NUM_GPUS <= ${#SELECTED_GPUS[@]} )) \
  || die "NUM_GPUS=${NUM_GPUS}, but CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

for integer_value in \
  "${MAX_STEPS}" "${SAVE_STEPS}" "${SAVE_STEPS_START}" \
  "${NUM_WORKERS}" "${PREFETCH_FACTOR}" "${ACTION_CHUNK_LENGTH}" "${WRIST_ACTION_DIM}" \
  "${GRADIENT_ACCUMULATION_STEPS}" "${MAX_NUM_TOKENS}" \
  "${EXPECTED_NUM_TOKENS}"; do
  [[ "${integer_value}" =~ ^[0-9]+$ ]] \
    || die "Expected a non-negative integer, got: ${integer_value}"
done
(( MAX_STEPS > 0 )) || die "MAX_STEPS must be positive"
(( SAVE_STEPS > 0 )) || die "SAVE_STEPS must be positive"
(( ACTION_CHUNK_LENGTH > 0 )) || die "ACTION_CHUNK_LENGTH must be positive"
(( WRIST_ACTION_DIM > 0 )) || die "WRIST_ACTION_DIM must be positive"
(( GRADIENT_ACCUMULATION_STEPS > 0 )) \
  || die "GRADIENT_ACCUMULATION_STEPS must be positive"
(( EXPECTED_NUM_TOKENS <= MAX_NUM_TOKENS )) \
  || die "EXPECTED_NUM_TOKENS cannot exceed MAX_NUM_TOKENS"

# Calculate statistics exactly once before torchrun. Otherwise every rank can
# race while creating the same meta/stats.json.
STATS_FILE="${DATASET_PATH}/meta/stats.json"
if [[ ! -f "${STATS_FILE}" ]]; then
  if [[ "${GENERATE_STATS}" == "True" || "${GENERATE_STATS}" == "true" ]]; then
    log "Generating ${STATS_FILE} from ${PARQUET_COUNT} parquet files..."
    "${PYTHON_BIN}" - "${DATASET_PATH}" <<'PY'
import json
import os
import sys
from pathlib import Path

from BeingH.dataset.parquet_utils import calculate_dataset_statistics

dataset_path = Path(sys.argv[1])
parquet_files = sorted(dataset_path.glob("data/**/*.parquet"))
if not parquet_files:
    raise RuntimeError(f"No parquet files found in {dataset_path / 'data'}")

statistics = calculate_dataset_statistics(parquet_files)
output_path = dataset_path / "meta/stats.json"
temporary_path = output_path.with_suffix(".json.tmp")
temporary_path.write_text(json.dumps(statistics, indent=2), encoding="utf-8")
os.replace(temporary_path, output_path)
print(f"Saved statistics to {output_path}")
PY
  else
    die "Missing ${STATS_FILE}; set GENERATE_STATS=True or create it first."
  fi
fi

log "Preflight passed"
log "Dataset: ${DATASET_PATH} (${PARQUET_COUNT} parquet, ${VIDEO_COUNT} videos)"
log "Data config: ${DATASET_CONFIG_FILE}; normalization=${NORMALIZATION}"
log "InternVL: ${PRETRAIN_MODEL}"
log "Expert: ${EXPERT_MODEL}"
log "Resume: ${RESUME_PATH}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; ranks=${NUM_GPUS}"
log "Steps=${MAX_STEPS}; save_every=${SAVE_STEPS}; save_start=${SAVE_STEPS_START}"
log "Packed tokens: target=${EXPECTED_NUM_TOKENS}; hard_max=${MAX_NUM_TOKENS}"
log "Workers/rank=${NUM_WORKERS}; prefetch=${PREFETCH_FACTOR}; fused AdamW=${FUSED_OPTIMIZER}"
log "FSDP=${SHARDING_STRATEGY}; grad_accum=${GRADIENT_ACCUMULATION_STEPS}"
log "Freeze MLLM=${FREEZE_MLLM}; freeze connector=${FREEZE_VIT_MLP}"
log "Wrist action loss weight=${WRIST_ACTION_LOSS_WEIGHT} (unified slots 0:${WRIST_ACTION_DIM})"
log "Temporal delta loss weight=${TEMPORAL_DELTA_LOSS_WEIGHT}; Huber beta=${TEMPORAL_DELTA_HUBER_BETA}"

if [[ "${PREFLIGHT_ONLY}" == "True" || "${PREFLIGHT_ONLY}" == "true" ]]; then
  log "PREFLIGHT_ONLY=True; torchrun was not launched."
  trap - EXIT
  exit 0
fi

# =============================================================================
# Output and reproducibility snapshot
# =============================================================================
RESUME_MODEL="$(basename "${RESUME_PATH}")"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_KIND="train"
if [[ "${SMOKE_TEST}" == "True" || "${SMOKE_TEST}" == "true" ]]; then
  RUN_KIND="smoke"
elif [[ "${PERF_TEST}" == "True" || "${PERF_TEST}" == "true" ]]; then
  RUN_KIND="perf"
fi
DEFAULT_RUN_NAME="${RUN_KIND}-${EMBODIMENT_DATASET}_${RESUME_MODEL}_freeze-mllm-${FREEZE_MLLM}_chunk-${ACTION_CHUNK_LENGTH}_tok-${EXPECTED_NUM_TOKENS}_norm-${NORMALIZATION}_wristw-${WRIST_ACTION_LOSS_WEIGHT}_tdelta-${TEMPORAL_DELTA_LOSS_WEIGHT}_mpg-${USE_MPG}_${RUN_TIMESTAMP}"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/tensorboard}"
LOG_FILE="${OUTPUT_DIR}/training.log"
SNAPSHOT_DIR="${OUTPUT_DIR}/run_config"

mkdir -p "${SNAPSHOT_DIR}" "${LOG_DIR}"
cp "${SCRIPT_DIR}/train_shadow_grasp.sh" "${SNAPSHOT_DIR}/"
cp "${DATASET_CONFIG_FILE}" "${SNAPSHOT_DIR}/"
cp "${REPO_ROOT}/configs/data_config.py" "${SNAPSHOT_DIR}/"
cp "${REPO_ROOT}/configs/dataset_info.py" "${SNAPSHOT_DIR}/"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse HEAD > "${SNAPSHOT_DIR}/git_commit.txt"
  git status --short > "${SNAPSHOT_DIR}/git_status.txt"
  git diff -- BeingH configs scripts/train/train_shadow_grasp.sh \
    > "${SNAPSHOT_DIR}/local_changes.patch"
fi

log "Output: ${OUTPUT_DIR}"
log "Log: ${LOG_FILE}"

# =============================================================================
# Launch training
# =============================================================================
"${TORCHRUN_BIN}" \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  BeingH/train/train.py \
  --mllm_path "${PRETRAIN_MODEL}" \
  --expert_path "${EXPERT_MODEL}" \
  --resume_from "${RESUME_PATH}" \
  --resume_model_only True \
  --layer_module Qwen3MoTDecoderLayer \
  --use_expert True \
  --use_flow_matching True \
  --llm_qk_norm True \
  --freeze_mllm "${FREEZE_MLLM}" \
  --freeze_vit_mlp "${FREEZE_VIT_MLP}" \
  --action_chunk_length "${ACTION_CHUNK_LENGTH}" \
  --wrist_action_loss_weight "${WRIST_ACTION_LOSS_WEIGHT}"   --wrist_action_dim "${WRIST_ACTION_DIM}" \
  --temporal_delta_loss_weight "${TEMPORAL_DELTA_LOSS_WEIGHT}" \
  --temporal_delta_huber_beta "${TEMPORAL_DELTA_HUBER_BETA}" \
  --max_num_tokens "${MAX_NUM_TOKENS}" \
  --max_num_tokens_per_sample "${MAX_NUM_TOKENS}" \
  --expected_num_tokens "${EXPECTED_NUM_TOKENS}" \
  --prefer_buffer_before "${PREFER_BUFFER_BEFORE}" \
  --max_buffer_size "${MAX_BUFFER_SIZE}" \
  --attn_mode "${ATTN_MODE}" \
  --max_view_num "${MAX_VIEW_NUM}" \
  --use_fixed_view "${USE_FIXED_VIEW}" \
  --force_image_size "${FORCE_IMAGE_SIZE}" \
  --down_sample_ratio "${DOWN_SAMPLE_RATIO}" \
  --dataset_config_file "${DATASET_CONFIG_FILE}" \
  --save_merged_metadata "${SAVE_MERGED_META}" \
  --conv_style being_h0 \
  --vision_select_layer -1 \
  --prompt_template long \
  --output_dir "${OUTPUT_DIR}" \
  --logging_dir "${LOG_DIR}" \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  --max_steps "${MAX_STEPS}" \
  --save_model_only "${SAVE_MODEL_ONLY}" \
  --save_steps "${SAVE_STEPS}" \
  --save_steps_start "${SAVE_STEPS_START}" \
  --logging_steps "${LOGGING_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler cosine \
  --grad_checkpoint False \
  --fused_optimizer "${FUSED_OPTIMIZER}" \
  --sharding_strategy "${SHARDING_STRATEGY}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --use_mpg "${USE_MPG}" \
  --mpg_lambda "${MPG_LAMBDA}" \
  --mpg_num_projections "${MPG_NUM_PROJECTIONS}" \
  --mpg_refinement_iters "${MPG_REFINEMENT_ITERS}" \
  --mpg_gate_temperature "${MPG_GATE_TEMPERATURE}" \
  --mpg_use_stop_gradient "${MPG_USE_STOP_GRADIENT}" \
  --use_training_time_rtc "${USE_TRAINING_TIME_RTC}" \
  --simulated_delay "${SIMULATED_DELAY}" \
  --rtc_delay_exp_weight "${RTC_DELAY_EXP_WEIGHT}" \
  --use_inference_prefix_overwrite "${USE_INFERENCE_PREFIX_OVERWRITE}" \
  2>&1 | tee "${LOG_FILE}"

trap - EXIT
printf '%s\n' \
  "==========================================" \
  "Training complete" \
  "Output: ${OUTPUT_DIR}" \
  "Log: ${LOG_FILE}" \
  "=========================================="
