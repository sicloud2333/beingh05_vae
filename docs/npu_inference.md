# Ascend NPU open-loop inference

The NPU path uses dense PyTorch SDPA instead of CUDA FlexAttention and does not
require `flash-attn`. It currently supports checkpoints configured with causal
attention.

## Environment

The selected Python environment must provide compatible `torch` and `torch_npu`
packages. The wrapper loads the Ascend driver and CANN environment before
starting Python.

```bash
export PYTHON_BIN=/path/to/npu-env/bin/python
export ASCEND_DRIVER_ENV=/usr/local/Ascend/driver/bin/setenv.bash
export ASCEND_CANN_ENV=/home/developer/Ascend/cann-9.2.0/set_env.sh
export NPU_DEVICE=npu:0
```

Verify the runtime:

```bash
source "$ASCEND_DRIVER_ENV"
source "$ASCEND_CANN_ENV"
"$PYTHON_BIN" -c 'import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())'
```

## Required artifacts

Download the Shadow policy checkpoint and dataset described in the repository
README. Keep them outside Git, for example under `ckpts/` and `data/`.

## One-query reproduction

Start with one episode and one model query to validate model loading, image
preprocessing, dense SDPA, flow matching, and action decoding:

```bash
PYTHON_BIN=/path/to/npu-env/bin/python \
bash scripts/eval/eval_shadow_open_loop_npu.sh \
  --model-path ckpts/Being-H05-shadow-grasp-2cam-rot6d-zraw \
  --dataset-path data/shadow_grasp_bottle22249179_aug100_2cam \
  --episode-index 0 \
  --max-queries 1
```

After the one-query run succeeds, remove `--max-queries 1` or pass multiple
episode indices:

```bash
PYTHON_BIN=/path/to/npu-env/bin/python \
bash scripts/eval/eval_shadow_open_loop_npu.sh \
  --model-path ckpts/Being-H05-shadow-grasp-2cam-rot6d-zraw \
  --dataset-path data/shadow_grasp_bottle22249179_aug100_2cam \
  --episode-indices 0 1 2 3
```
