#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VAE_RUN="${VAE_RUN:-${REPO_ROOT}/vae/runs/native_n2_pair_vector_distance_20260817_143354}"
VAE_PID="${VAE_PID:-2426987}"
VAE_CHECKPOINT="${VAE_CHECKPOINT:-${VAE_RUN}/checkpoints/inference.pt}"
SOURCE_DATASET="${SOURCE_DATASET:-${REPO_ROOT}/data/shadow_grasp_bottle22249179_aug100_2cam}"
OUTPUT_DATASET="${OUTPUT_DATASET:-${REPO_ROOT}/data/shadow_grasp_bottle22249179_aug100_npuvae_2cam}"
DATASET_NAME="${DATASET_NAME:-shadow_grasp_bottle22249179_aug100_npuvae_2cam}"
PIPELINE_LOG_DIR="${PIPELINE_LOG_DIR:-${REPO_ROOT}/logs/npu_full_vae_pipeline}"
BEINGH_ENV="${BEINGH_ENV:-/mnt/workspace/gitCode/cann/hsz/factory_dex/.venv-ascend}"
ASCEND_DRIVER_ENV="${ASCEND_DRIVER_ENV:-/usr/local/Ascend/driver/bin/setenv.bash}"
ASCEND_CANN_ENV="${ASCEND_CANN_ENV:-/home/developer/Ascend/cann-9.2.0/set_env.sh}"
REUSE_OUTPUT_DATASET="${REUSE_OUTPUT_DATASET:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"

mkdir -p "${PIPELINE_LOG_DIR}"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] [full-vae-npu] %s\n' -1 "$*"
}

log "Waiting for NativeVAE pid=${VAE_PID}; checkpoint=${VAE_CHECKPOINT}"
wait_cycles=0
while kill -0 "${VAE_PID}" 2>/dev/null; do
  if (( wait_cycles % 5 == 0 )); then
    latest_metric="$(tail -n 1 "${VAE_RUN}/logs/metrics.jsonl" 2>/dev/null || true)"
    log "NativeVAE is running; latest_metric=${latest_metric:-not-yet-available}"
  fi
  wait_cycles=$((wait_cycles + 1))
  sleep 60
done

[[ -f "${VAE_CHECKPOINT}" ]] || {
  log "ERROR: NativeVAE process exited without ${VAE_CHECKPOINT}"
  exit 3
}
log "NativeVAE complete: $(stat -c '%s bytes, modified=%y' "${VAE_CHECKPOINT}")"

# The latent builder imports torch_npu before it parses CLI arguments. Load
# both vendor environments here so libhccl.so and the custom-op libraries are
# resolvable even when this orchestrator was started from a clean shell.
[[ -f "${ASCEND_DRIVER_ENV}" ]] || {
  log "ERROR: missing Ascend driver environment: ${ASCEND_DRIVER_ENV}"
  exit 5
}
[[ -f "${ASCEND_CANN_ENV}" ]] || {
  log "ERROR: missing CANN environment: ${ASCEND_CANN_ENV}"
  exit 5
}
set +u
source "${ASCEND_DRIVER_ENV}"
source "${ASCEND_CANN_ENV}"
set -u

if [[ -e "${OUTPUT_DATASET}" ]]; then
  if [[ "${REUSE_OUTPUT_DATASET}" == "1" ]]; then
    log "Reusing existing formal dataset after prior successful build: ${OUTPUT_DATASET}"
  else
    log "ERROR: refusing to overwrite existing formal dataset: ${OUTPUT_DATASET}"
    exit 4
  fi
else
  log "Building formal z_gesture dataset on NPU0"
  "${BEINGH_ENV}/bin/python" -u "${REPO_ROOT}/vae/scripts/build_lerobot_z_dataset.py" \
    --source-dataset "${SOURCE_DATASET}" \
    --output-dataset "${OUTPUT_DATASET}" \
    --vae-checkpoint "${VAE_CHECKPOINT}" \
    --device npu:0 \
    --batch-size 4096 \
    2>&1 | tee "${PIPELINE_LOG_DIR}/build_z_dataset.log"
fi

if [[ "${SKIP_SMOKE}" == "1" ]]; then
  log "Skipping two-rank smoke test after prior successful smoke checkpoint"
else
  log "Running two-rank HCCL smoke test with the formal z_gesture dataset"
  ASCEND_RT_VISIBLE_DEVICES=0,1 \
  NUM_GPUS=2 \
  MASTER_PORT=29131 \
  SMOKE_TEST=True \
  SMOKE_MAX_STEPS=1 \
  SMOKE_SAVE_STEPS=1 \
  SMOKE_SAVE_STEPS_START=0 \
  SMOKE_NUM_WORKERS=0 \
  SMOKE_PREFETCH_FACTOR=1 \
  LOGGING_STEPS=1 \
  MAX_NUM_TOKENS=4096 \
  EXPECTED_NUM_TOKENS=4096 \
  PREFER_BUFFER_BEFORE=2048 \
  MAX_BUFFER_SIZE=2 \
  EMBODIMENT_DATASET="${DATASET_NAME}" \
  NORMALIZATION=wrist_rot6d_minmax_zraw \
  OUTPUT_ROOT="${REPO_ROOT}/outputs/npu2_formal_z_smoke" \
  bash "${SCRIPT_DIR}/train_shadow_grasp_npu.sh" \
    2>&1 | tee "${PIPELINE_LOG_DIR}/two_rank_smoke.log"
fi

run_timestamp="$(date +%Y%m%d_%H%M%S)"
log "Starting formal two-NPU 40k-step post-training"
ASCEND_RT_VISIBLE_DEVICES=0,1 \
NUM_GPUS=2 \
MASTER_PORT=29132 \
NUM_WORKERS=0 \
PREFETCH_FACTOR=1 \
MAX_STEPS=40000 \
SAVE_STEPS=5000 \
SAVE_STEPS_START=15000 \
LOGGING_STEPS=10 \
MAX_NUM_TOKENS=4096 \
EXPECTED_NUM_TOKENS=4096 \
PREFER_BUFFER_BEFORE=2048 \
MAX_BUFFER_SIZE=2 \
GRADIENT_ACCUMULATION_STEPS=2 \
EMBODIMENT_DATASET="${DATASET_NAME}" \
NORMALIZATION=wrist_rot6d_minmax_zraw \
RUN_NAME="full-npu2-vae-tok4096-acc2-${run_timestamp}" \
OUTPUT_ROOT="${REPO_ROOT}/outputs/npu_full_vae_posttrain" \
bash "${SCRIPT_DIR}/train_shadow_grasp_npu.sh" \
  2>&1 | tee "${PIPELINE_LOG_DIR}/full_posttrain.log"

log "Formal VAE NPU training pipeline completed"
