# Being-H05 release layout

The GitHub repository contains source code, configs, evaluation scripts and
VAE source. Datasets and model checkpoints are intentionally excluded.

## Download artifacts

```bash
huggingface-cli download BeingBeyond/Being-H05-2B \
  --local-dir ckpts/Being-H05-2B

huggingface-cli download BeingBeyond/shadow_grasp_bottle22249179_aug100_2cam \
  --repo-type dataset \
  --local-dir data/shadow_grasp_bottle22249179_aug100_2cam
```

Set paths in `.env` or export `BEINGH_DATA_ROOT`, `BEINGH_CKPT_ROOT`,
`LIBERO_DATA_ROOT`, `ROBOCASA_DATA_ROOT` and `REAL_DATA_ROOT` before training
or evaluation. New MuJoCo evaluation uses `scripts/eval/eval_shadow_grasp.sh`,
whose default latent observation mode is `commanded`.

## Reproducibility

Record the Being-H05 Git commit, the VAE commit, the Hugging Face model revision
and the dataset revision together for each experiment.
