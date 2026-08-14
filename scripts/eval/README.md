# Being-H05 evaluation entry points

Use the wrappers in this directory for new runs:

```bash
bash scripts/eval/eval_shadow_grasp.sh \
  --model-path /path/to/checkpoint \
  --dataset vae/evaluation/object_episodes/shadow_grasp_0725.jsonl \
  --episode 20 --hand shadow_hand_right
```

`eval_shadow_grasp.sh` defaults to `--latent-observation-mode commanded`.
Pass `--latent-observation-mode encoded` explicitly to reproduce an encoded
observation experiment. The underlying implementation remains in
`vae/examples/beingh_shadow_grasp_eval.py` and is not modified by this
organization layer.

For recorded-trajectory inference use:

```bash
bash scripts/eval/eval_shadow_open_loop.sh \
  --model-path /path/to/checkpoint \
  --episode-index 20
```
