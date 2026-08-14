# Hugging Face artifact guide

GitHub only contains source code. Upload datasets and checkpoints to Hugging
Face repositories and keep the revision IDs in the experiment record.

Replace `zju` with the actual organization and login once:

```bash
huggingface-cli login
# or: hf auth login
```

## Recommended delivery set

The following is the minimum artifact set for the current project results. Do
not upload every historical `outputs/` run.

### Required for the main Native-VAE result

| Artifact | Local source | Purpose |
|---|---|---|
| Shadow 2-camera training data | `data/shadow_grasp_bottle22249179_aug100_2cam` | Reproduce the primary VAE policy training/evaluation |
| Primary VAE policy checkpoint | `outputs/shadow_grasp_bottle22249179_aug100_2cam/train-shadow_grasp_bottle22249179_aug100_2cam_Being-H05-2B_freeze-mllm-True_chunk-16_tok-8192_norm-wrist_rot6d_minmax_zraw_wristw-1.0_tdelta-0.0_mpg-True_20260804_154819/0060000` | Primary wrist Rot6D + raw z_gesture policy |
| Native VAE checkpoint | `vae/checkpoints/native_n2_epoch800_inference.pt` | Decode z_gesture into target-hand joints |

### Required for the physical-joint baseline

| Artifact | Local source | Purpose |
|---|---|---|
| Shadow 2-camera data | `data/shadow_grasp_bottle22249179_aug100_2cam` | Train/evaluate the Shadow physical-joint baseline |
| Joint baseline checkpoint | `outputs/shadow_grasp_bottle22249179_aug100_2cam/train-shadow_grasp_bottle22249179_aug100_2cam_Being-H05-2B_freeze-mllm-True_chunk-16_tok-8192_norm-wrist_rot6d_minmax_joints_wristw-1.0_tdelta-0.0_mpg-True_20260805_141947/0030000` | Wrist Rot6D + 22D Shadow joint policy |

### Optional baseline datasets/checkpoints

Upload these only if the customer must reproduce the separately trained Gaia
and Sharpa baseline policies:

```text
data/sharpa_grasp_bottle22249179_geo_visual100_2cam
data/gaia_grasp_bottle22249179_geo_visual100_2cam
outputs/sharpa_grasp_bottle22249179_geo_visual100_2cam_joint/.../0040000
outputs/gaia_grasp_bottle22249179_geo_visual100_2cam_joint/.../0040000
```

The older `data/shadow_grasp_0725_core_bottle_1071` (52 episodes) and its
`...20260802_100014/0040000` VAE checkpoint may be uploaded as an optional
legacy experiment. The original `data/shadow_grasp_0725` (1207 episodes), smoke
runs, perf runs, old Euler/q99/min-max experiments and other intermediate
checkpoints are not needed for the main delivery. Keep them as an internal
archive.

### Lightweight evaluation manifests

The `vae/evaluation/object_episodes/*.jsonl` files are small manifests rather
than training data. Store the manifests in the corresponding dataset repository
under `evaluation/object_episodes/`, or upload them as a separate lightweight
artifact so the MuJoCo commands in the root README can be run unchanged.

## Dataset repositories

Create one dataset repository per dataset:

```bash
huggingface-cli repo create zju/shadow_grasp_0725 \
  --repo-type dataset
huggingface-cli repo create zju/shadow_grasp_0725_core_bottle_1071 \
  --repo-type dataset
huggingface-cli repo create zju/shadow_grasp_bottle22249179_aug100_2cam \
  --repo-type dataset
huggingface-cli repo create zju/sharpa_grasp_bottle22249179_geo_visual100_2cam \
  --repo-type dataset
huggingface-cli repo create zju/gaia_grasp_bottle22249179_geo_visual100_2cam \
  --repo-type dataset
```

Upload each local LeRobot directory. The command uploads parquet data, videos,
metadata and collection documentation; it skips Python caches and local logs.
Hugging Face handles large files through its storage backend.

```bash
DATA_ROOT=/path/to/data

huggingface-cli upload \
  zju/shadow_grasp_0725 \
  "$DATA_ROOT/shadow_grasp_0725" . \
  --repo-type dataset \
  --exclude='**/__pycache__/**' \
  --exclude='**/*.pyc' \
  --exclude='**/*.log'

huggingface-cli upload \
  zju/shadow_grasp_0725_core_bottle_1071 \
  "$DATA_ROOT/shadow_grasp_0725_core_bottle_1071" . \
  --repo-type dataset \
  --exclude='**/__pycache__/**' \
  --exclude='**/*.pyc' \
  --exclude='**/*.log'

huggingface-cli upload \
  zju/shadow_grasp_bottle22249179_aug100_2cam \
  "$DATA_ROOT/shadow_grasp_bottle22249179_aug100_2cam" . \
  --repo-type dataset \
  --exclude='**/__pycache__/**' \
  --exclude='**/*.pyc' \
  --exclude='**/*.log'

huggingface-cli upload \
  zju/sharpa_grasp_bottle22249179_geo_visual100_2cam \
  "$DATA_ROOT/sharpa_grasp_bottle22249179_geo_visual100_2cam" . \
  --repo-type dataset \
  --exclude='**/__pycache__/**' \
  --exclude='**/*.pyc' \
  --exclude='**/*.log'

huggingface-cli upload \
  zju/gaia_grasp_bottle22249179_geo_visual100_2cam \
  "$DATA_ROOT/gaia_grasp_bottle22249179_geo_visual100_2cam" . \
  --repo-type dataset \
  --exclude='**/__pycache__/**' \
  --exclude='**/*.pyc' \
  --exclude='**/*.log'
```

Download a dataset on the client machine:

```bash
huggingface-cli download \
  zju/shadow_grasp_bottle22249179_aug100_2cam \
  --repo-type dataset \
  --local-dir data/shadow_grasp_bottle22249179_aug100_2cam
```

## Model repositories

Do not upload the full `outputs/` run directory. Upload a self-contained numeric
checkpoint directory containing at least:

```text
config.json
model.safetensors
tokenizer_config.json
tokenizer.json (if present)
merges.txt/vocab.json (if present)
shadow_grasp_posttrain_metadata.json
```

Create a model repository for the base checkpoint and, preferably, one repo per
trained experiment:

```bash
huggingface-cli repo create zju/Being-H05-2B --repo-type model
huggingface-cli repo create zju/Being-H05-shadow-grasp-rot6d --repo-type model
huggingface-cli repo create zju/Being-H05-shadow-grasp-joints --repo-type model
```

Upload the base checkpoint:

```bash
huggingface-cli upload \
  zju/Being-H05-2B \
  /path/to/ckpts/Being-H05-2B . \
  --repo-type model \
  --exclude='**/__pycache__/**' \
  --exclude='**/*.log'
```

Upload a trained checkpoint as a numbered directory so multiple checkpoints can
share one model repository:

```bash
RUN=/path/to/outputs/<training-run>
CKPT=0040000

huggingface-cli upload \
  zju/Being-H05-shadow-grasp-rot6d \
  "$RUN/$CKPT" "$CKPT" \
  --repo-type model \
  --exclude='**/__pycache__/**' \
  --exclude='**/*.log' \
  --exclude='run_config/**'
```

Download a checkpoint into the layout expected by the evaluation scripts:

```bash
huggingface-cli download \
  zju/Being-H05-shadow-grasp-rot6d \
  0040000/config.json \
  0040000/model.safetensors \
  --local-dir ckpts/Being-H05-shadow-grasp-rot6d
```

If the repository contains tokenizer and metadata files under the checkpoint
folder, download the whole folder instead:

```bash
huggingface-cli download \
  zju/Being-H05-shadow-grasp-rot6d \
  --include='0040000/**' \
  --local-dir ckpts/Being-H05-shadow-grasp-rot6d
```

## Official dependency models

InternVL and Qwen base models can be downloaded directly from their official
repositories instead of being duplicated in the project organization:

```bash
huggingface-cli download OpenGVLab/InternVL3_5-2B \
  --local-dir ckpts/InternVL3_5-2B

huggingface-cli download Qwen/Qwen3-0.6B \
  --local-dir ckpts/Qwen3-0.6B
```

## Revision record

For every release, record:

```text
GitHub code tag
Hugging Face model repo + commit/revision
Hugging Face dataset repo + commit/revision
VAE source commit
```
