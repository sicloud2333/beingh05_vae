# End-to-end VAE training on Ascend NPU

This runbook reproduces the repository's complete VAE path on a two-device
Ascend host:

1. generate deterministic NativeVAE train/validation tensors;
2. train the three-hand NativeVAE for 800 epochs;
3. encode the Shadow dataset into deterministic 24D posterior means;
4. run a two-rank HCCL/FSDP checkpoint smoke test;
5. post-train Being-H05 for the repository-standard 40,000 optimizer steps;
6. run 100-episode open-loop, matched repository-baseline, and MuJoCo tests.

## Runtime

The commands below use the server paths exercised by this reproduction. They
can be overridden through the environment variables accepted by the wrappers.

```bash
cd /mnt/workspace/gitCode/cann/hsz/beingh05_vae

set +u
source /usr/local/Ascend/driver/bin/setenv.bash
source /home/developer/Ascend/cann-9.2.0/set_env.sh
set -u

export BEINGH_ENV=/mnt/workspace/gitCode/cann/hsz/factory_dex/.venv-ascend
export PYTHON_BIN="$BEINGH_ENV/bin/python"
```

The host used here has only 64 MiB of `/dev/shm`. The NPU launcher therefore
defaults to `NUM_WORKERS=0` and `PREFETCH_FACTOR=1`; worker subprocesses caused
SIGBUS with packed image batches on this machine.

## 1. NativeVAE tensors

`vae/configs/train_native_n2.yaml` defines 50,000 train and 5,000 validation
poses per hand for Shadow, Gaia, and Sharpa. The resulting datasets contain
150,000 and 15,000 samples respectively.

```bash
cd vae
"$PYTHON_BIN" scripts/generate_random_data.py \
  --config configs/train_native_n2.yaml \
  --device npu:0
```

Expected outputs:

- `vae/data/train_50k_per_hand_seed42.pt`
- `vae/data/val_5k_per_hand_seed10042.pt`

## 2. NativeVAE training

```bash
cd /mnt/workspace/gitCode/cann/hsz/beingh05_vae/vae
"$PYTHON_BIN" -u scripts/train.py \
  --config configs/train_native_n2.yaml \
  --device npu:0 \
  --epochs 800 \
  --no_wandb
```

The configuration enables same-hand reconstruction immediately and enables
cross-hand absolute-tip, pair-vector, and pair-distance losses at epoch 300.
It writes a full optimizer checkpoint every 50 epochs plus `best.pt`, `last.pt`,
and `inference.pt`. `vae/scripts/run_native_vae_milestone_eval.sh` evaluates
epochs 300 and 500; `run_native_vae_eval_when_ready.sh` evaluates the final
checkpoint on all 15,000 validation poses.

## 3. Automatic conversion and post-training

Start the orchestrator while NativeVAE training is running. Supply the actual
run directory and process ID rather than relying on the defaults from the
recorded server run.

```bash
cd /mnt/workspace/gitCode/cann/hsz/beingh05_vae
VAE_RUN=/absolute/path/to/vae/runs/<native-run> \
VAE_PID=<native-training-pid> \
BEINGH_ENV="$BEINGH_ENV" \
bash scripts/train/run_full_vae_npu_pipeline.sh
```

After `inference.pt` appears, the orchestrator:

- builds `data/shadow_grasp_bottle22249179_aug100_npuvae_2cam`;
- preserves dimensions `0:28` and replaces dimensions `28:52` with the final
  NativeVAE posterior mean `z_mu` for both state and action;
- verifies every parquet write by exact NPU round trip;
- records the NativeVAE path and SHA-256 in collection metadata and the
  completion marker;
- runs a two-rank HCCL/FSDP one-step smoke test with optimizer shards;
- starts the formal two-NPU 40,000-step Being-H05 post-training run.

The independent formal-data verifier reloads every one of the 100 parquets,
checks all 17,700 rows, proves all non-target columns are unchanged, recomputes
the latent from the recorded checkpoint, and checks the checkpoint SHA-256.
CPU-vs-NPU encoding uses the calibrated absolute tolerance `0.002` and zero
relative tolerance; the exact same-device write is checked separately.

## 4. Two-NPU optimization contract

The repository default is 8,192 expected tokens per rank on two ranks with one
microbatch per optimizer update. The Ascend run uses:

```text
4,096 tokens/rank × 2 ranks × 2 accumulated microbatches
  = 16,384 global tokens/optimizer update
```

This preserves the repository's effective global token batch while staying
below 64 GiB HBM. `max_steps=40000` counts optimizer updates, not microbatches.
The NPU run saves recoverable model, optimizer-shard, scheduler, tokenizer, and
metadata checkpoints every 5,000 steps from step 15,000 through step 40,000.

## 5. Final evaluation gates

`scripts/eval/run_final_vae_eval_when_ready.sh` waits for a complete step-40000
checkpoint and clean training-pipeline exit. It then runs:

- all 100 offline open-loop episodes and 17,700 aligned actions;
- independent NPZ/JSON/plot verification and aggregate-metric recomputation;
- end-to-end policy-query latency (cold query, warm mean/P50/P95, query and
  action throughput);
- three matched NPU trials for VAE, Geometry retargeting, and the Gaia
  hand-specific joint checkpoint;
- all 100 MuJoCo episodes with ego-opposite and wrist videos;
- independent MuJoCo summary, step/query count, rollout, metadata, timing, and
  video verification.

The final `outputs/npu_final_vae_evaluation/COMPLETE` marker is written only
after every gate succeeds. Intermediate smoke or repository-provided
checkpoints do not satisfy this marker.

For lower-level NPU inference setup and a one-query smoke test, see
`docs/npu_inference.md`.
