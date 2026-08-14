# Validation

Validation date: 2026-07-25

## Model

- Source run: `Native-N2-pair-vector-distance-refine`
- Source epoch: 800
- Inference checkpoint: `checkpoints/native_n2_epoch800_inference.pt`
- Original checkpoint size: 13 MB
- Inference checkpoint size: 4.4 MB

The inference checkpoint contains only the model state and the minimal model metadata. Optimizer and scheduler states were removed.

## Numerical regression

A fixed set of four Shadow poses was evaluated with both the original OrthoHand implementation and this package.

Compared outputs:

- Shadow `z_gesture`
- Shadow reconstruction q and finger-pad positions
- Shadow-to-Gaia q and finger-pad positions
- Shadow-to-Sharpa q and finger-pad positions

Tolerance:

```text
rtol = 1e-6
atol = 1e-6
```

All comparisons passed.

## Runtime checks

| Hand | Input q | z_gesture | Decoded q | Finger pads |
| --- | ---: | ---: | ---: | ---: |
| Shadow | `[2048, 22]` | `[2048, 24]` | `[2048, 22]` | `[2048, 5, 3]` |
| Gaia | `[2048, 15]` | `[2048, 24]` | `[2048, 15]` | `[2048, 5, 3]` |
| Sharpa | `[2048, 22]` | `[2048, 24]` | `[2048, 22]` | `[2048, 5, 3]` |

The test suite was also copied to `/tmp/native_vae_standalone_test` and executed there. It passed without importing files from the parent OrthoHand project.

## MuJoCo grasp environment

The extended project was copied to
`/tmp/native_vae_sim_standalone_validation_20260725` and tested without the
original repository on `PYTHONPATH`.

Validated for Shadow, Gaia, and Sharpa:

- local MJCF and visual/collision meshes load;
- `ctrl` and `qpos` modes step without NaN/Inf;
- state dimensions are 28, 21, and 28;
- `ego_opposite` returns non-blank RGB `uint8 [240, 320, 3]`;
- native q can pass through `q -> z -> target q -> MuJoCo step`;
- replay video output is 320 x 240 at 30 Hz.

Gaia uses explicit hand self-collision filtering plus minimum finger armature
and damping. This removes the wrist acceleration instability observed with the
raw Gaia MJCF while preserving hand-object and hand-table contacts.
