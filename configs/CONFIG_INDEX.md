# Being-H05 configuration index

This index is for the Being-H05 codebase only. The `vae/` directory has its
own configuration and is intentionally not changed here.

## Training configuration families

| Family | Location | Purpose |
|---|---|---|
| Shadow grasp | `configs/posttrain/shadow_grasp/` | Wrist/z_gesture, wrist/Rot6D, and physical-joint variants |
| LIBERO | `configs/posttrain/libero/` | LIBERO post-training |
| RoboCasa | `configs/posttrain/robocasa/` | Human RoboCasa post-training |
| Cross-embodiment | `configs/posttrain/cross-embodiment/` | Mixed embodiment training |

## Shadow normalization naming

The suffix after the dataset name describes the action transform:

- no suffix: legacy/raw transform
- `_q99`: percentile normalization
- `_minmax`: full state/action min-max normalization
- `_wrist_minmax_zraw`: wrist min-max, raw `z_gesture`
- `_wrist_euler_minmax_zraw`: Euler wrist min-max, raw `z_gesture`
- `_wrist_rot6d_minmax_zraw`: Rot6D wrist min-max, raw `z_gesture`
- `_wrist_rot6d_minmax_joints`: Rot6D wrist min-max, physical joints

## Canonical entry points

- Training: `scripts/train/train_shadow_grasp.sh`
- MuJoCo evaluation: `scripts/eval/eval_shadow_grasp.sh`
- Offline open-loop evaluation: `scripts/eval/eval_shadow_open_loop.sh`

New scripts should use these entry points rather than duplicating the long
Python command lines used by historical experiments.
