# Hugging Face artifact guide

GitHub only contains source code. Upload datasets and checkpoints to Hugging
Face repositories and keep the revision IDs in the experiment record.

Replace `<HF_ORG>` with the actual organization and login once:

```bash
huggingface-cli login
# or: hf auth login
```

## Dataset repositories

Create one dataset repository per dataset:

```bash
huggingface-cli repo create <HF_ORG>/shadow_grasp_0725 \
  --repo-type dataset
huggingface-cli repo create <HF_ORG>/shadow_grasp_0725_core_bottle_1071 \
  --repo-type dataset
huggingface-cli repo create <HF_ORG>/shadow_grasp_bottle22249179_aug100_2cam \
  --repo-type dataset
huggingface-cli repo create <HF_ORG>/sharpa_grasp_bottle22249179_geo_visual100_2cam \
  --repo-type dataset
huggingface-cli repo create <HF_ORG>/gaia_grasp_bottle22249179_geo_visual100_2cam \
  --repo-type dataset
```

Upload each local LeRobot directory. The command uploads parquet data, videos,
metadata and collection documentation; it skips Python caches and local logs.
Hugging Face handles large files through its storage backend.

```bash
DATA_ROOT=/path/to/data

huggingface-cli upload \
  <HF_ORG>/shadow_grasp_0725 \
  "$DATA_ROOT/shadow_grasp_0725" . \
  --repo-type dataset \
  --exclude='**/__pycache__/**' \
  --exclude='**/*.pyc' \
  --exclude='**/*.log'

huggingface-cli upload \
  <HF_ORG>/shadow_grasp_0725_core_bottle_1071 \
  "$DATA_ROOT/shadow_grasp_0725_core_bottle_1071" . \
  --repo-type dataset \
  --exclude='**/__pycache__/**' \
  --exclude='**/*.pyc' \
  --exclude='**/*.log'

huggingface-cli upload \
  <HF_ORG>/shadow_grasp_bottle22249179_aug100_2cam \
  "$DATA_ROOT/shadow_grasp_bottle22249179_aug100_2cam" . \
  --repo-type dataset \
  --exclude='**/__pycache__/**' \
  --exclude='**/*.pyc' \
  --exclude='**/*.log'

huggingface-cli upload \
  <HF_ORG>/sharpa_grasp_bottle22249179_geo_visual100_2cam \
  "$DATA_ROOT/sharpa_grasp_bottle22249179_geo_visual100_2cam" . \
  --repo-type dataset \
  --exclude='**/__pycache__/**' \
  --exclude='**/*.pyc' \
  --exclude='**/*.log'

huggingface-cli upload \
  <HF_ORG>/gaia_grasp_bottle22249179_geo_visual100_2cam \
  "$DATA_ROOT/gaia_grasp_bottle22249179_geo_visual100_2cam" . \
  --repo-type dataset \
  --exclude='**/__pycache__/**' \
  --exclude='**/*.pyc' \
  --exclude='**/*.log'
```

Download a dataset on the client machine:

```bash
huggingface-cli download \
  <HF_ORG>/shadow_grasp_bottle22249179_aug100_2cam \
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
huggingface-cli repo create <HF_ORG>/Being-H05-2B --repo-type model
huggingface-cli repo create <HF_ORG>/Being-H05-shadow-grasp-rot6d --repo-type model
huggingface-cli repo create <HF_ORG>/Being-H05-shadow-grasp-joints --repo-type model
```

Upload the base checkpoint:

```bash
huggingface-cli upload \
  <HF_ORG>/Being-H05-2B \
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
  <HF_ORG>/Being-H05-shadow-grasp-rot6d \
  "$RUN/$CKPT" "$CKPT" \
  --repo-type model \
  --exclude='**/__pycache__/**' \
  --exclude='**/*.log' \
  --exclude='run_config/**'
```

Download a checkpoint into the layout expected by the evaluation scripts:

```bash
huggingface-cli download \
  <HF_ORG>/Being-H05-shadow-grasp-rot6d \
  0040000/config.json \
  0040000/model.safetensors \
  --local-dir ckpts/Being-H05-shadow-grasp-rot6d
```

If the repository contains tokenizer and metadata files under the checkpoint
folder, download the whole folder instead:

```bash
huggingface-cli download \
  <HF_ORG>/Being-H05-shadow-grasp-rot6d \
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
