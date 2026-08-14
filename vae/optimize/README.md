# Geometry optimization retargeting baseline

This package provides a non-learned cross-hand baseline for the physical
Shadow-joint Being-H policy. It does not load or call NativeVAE weights.

The optimizer reuses the same hand specifications and geometric gesture used
by NativeVAE:

- the same URDFs, active and fixed joints, and joint limits;
- the same fixed semantic palm frame;
- the same five finger roots and three chain vectors per finger (60D total);
- per-hand palm-radius normalization.

At evaluation time the canonical policy space remains the Shadow hand:

1. actual Gaia/Sharpa joints are optimized into native Shadow joints;
2. native Shadow joints are converted to the training-dataset sign convention;
3. Being-H predicts absolute Shadow dataset-coordinate joints;
4. predictions are converted back to native Shadow joints;
5. a second optimization produces Gaia/Sharpa native joint commands.

Observation feedback is selectable with `--latent-observation-mode`:

- `encoded` retargets the actual target-hand joints back into Shadow joints;
- `commanded` initializes from the matched Shadow episode's first 22D joint
  state, then feeds the previous selected canonical Shadow-joint command back
  to Being-H. Actual target joints are still reverse-retargeted for auditing.

For synchronous chunk execution the action direction is genuinely batched:
the full `[16, 22]` Shadow chunk is optimized into `[16, target_dim]` with one
batched FK/optimizer run, cached, and then executed without per-step action
retargeting. The target-to-Shadow observation solve runs only on replan steps.

`raw` minimizes geometry only. `stable` also regularizes target joint velocity
and acceleration. Deployment rate limiting and other policy smoothing remain
separate evaluation controls.

The evaluation script selects this baseline automatically when a physical
Shadow-joint checkpoint is combined with `--hand gaia_hand_right` or
`--hand sharpa_hand_right`:

```bash
/opt/conda/envs/beingh05/bin/python -u \
  vae/examples/beingh_shadow_grasp_eval.py \
  --model-path "$CKPT" \
  --episode-range 0 99 \
  --hand gaia_hand_right \
  --joint-retargeting geometry \
  --geometry-retargeting-profile raw \
  --geometry-action-chunk-mode batch \
  --latent-observation-mode commanded \
  --device cuda:0
```

The default of 12 LBFGS iterations is a throughput/accuracy compromise. Use
`--geometry-max-iterations 20` for a stricter but slower geometry solve. The
rollout stores canonical Shadow observations/actions, retargeted native target
actions, and per-direction geometry RMSE and latency.
