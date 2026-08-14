# Grasp Policy Evaluation

`sim` provides a MuJoCo evaluation layer for `shadow_hand_right`,
`gaia_hand_right`, and `sharpa_hand_right`. Its local environment scene is a
standalone snapshot of `assets/grasp_scene/base_scene.xml`, with the same
table, physics, lighting, fixed `ego`/`ego_opposite` cameras, and the moving
`wrist` camera attached to each hand's `wrist_rz_link`.

## Interface

All three MuJoCo environments use one physical convention:

```text
environment action/state = [world wrist xyz, absolute wrist Euler XYZ, native joint q]
```

Policy data configs may declare a nonzero `WRIST_WORLD_ORIGIN`. The current
two-camera Shadow dataset uses `[0, 0, 0.4]` metres, so the evaluation adapter
applies exactly one translation in each direction:

```text
policy_xyz = world_xyz - wrist_world_origin
world_xyz  = policy_xyz + wrist_world_origin
```

Euler RPY stays absolute and is never translated. Legacy data configs declare
the zero origin, preserving their previous evaluation behavior.

| Hand | Wrist | Native joints | Dimension |
| --- | ---: | ---: | ---: |
| Shadow | 6 | 22 | 28 |
| Gaia | 6 | 15 | 21 |
| Sharpa | 6 | 22 | 28 |

`step()` returns the actual state in the same ordering, the primary RGB
`image`, an `images` dictionary keyed by requested MuJoCo camera name, object
`pose`/`velocity`, and simulation time. `observation_cameras=()` preserves the
legacy single-camera behavior; pass `("ego_opposite", "wrist")` for a
two-view policy. The default control rate is 30 Hz;
MuJoCo runs enough 2 ms substeps for each policy step. Gaia stabilization
matches the self-collision and finger-dynamics treatment used by the existing
evaluation pipeline.

```python
from sim import GraspEnv, GraspEnvConfig

with GraspEnv(
    GraspEnvConfig(
        hand="shadow_hand_right",
        observation_cameras=("ego_opposite", "wrist"),
    )
) as env:
    obs, info = env.reset()
    action = policy.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
```

Pass the experiment's first action to `reset(initial_action=...)` when it
defines a specific initial wrist pose.

## MuJoCo GUI

`GraspEnv` directly manages a passive viewer:

```python
import time
from sim import GraspEnv, GraspEnvConfig

with GraspEnv(
    GraspEnvConfig(hand="shadow_hand_right", render_images=False)
) as env:
    obs, _ = env.reset()
    action = obs["state"].copy()
    env.launch_gui()
    while env.gui_is_running:
        obs, _, _, _, _ = env.step(action)
        env.sync_gui()
        time.sleep(env.dt)
```

For a quick visual inspection without policy actions:

```bash
python scripts/sim_view_environment.py --hand shadow_hand_right
python scripts/sim_view_environment.py --hand gaia_hand_right
python scripts/sim_view_environment.py --hand sharpa_hand_right
```

Add `--camera ego_opposite` to inspect the fixed policy camera or
`--camera wrist` to inspect the moving under-wrist view instead of the
interactive free camera.

## Objects and evaluation scenes

List the object assets included with the package:

```bash
python scripts/sim_view_environment.py --list_objects
```

Create a Shadow scene with a packaged mug:

```bash
python scripts/sim_view_environment.py \
  --hand shadow_hand_right \
  --object_id core_mug_1038e4eac0e18dcce02ae6d2a21d494a \
  --object_scale 0.06 \
  --object_position 0 0 0.041
```

The same object can be paired with Gaia or Sharpa by changing `--hand`.
`object_scale`, `object_position`, and `object_quaternion` are explicit because
they are episode-level properties in the source dataset.

The Python API uses the same fields:

```python
config = GraspEnvConfig(
    hand="gaia_hand_right",
    object_id="core_bottle_1071fa4cddb2da2fc8724d5673a063a6",
    object_scale=0.08,
    object_position=(0.0, 0.0, 0.035),
)
env = GraspEnv(config)
```

### Portable evaluation scene sets

`evaluation/object_episodes/` contains object configurations that can be used
without the original LeRobot dataset, source NPZ, or generated scene XML:

| Manifest | Contents |
| --- | --- |
| `shadow_grasp_0725.jsonl` | All 1207 evaluation object scenes |
| `shadow_grasp_0725_core_bottle_1071.jsonl` | 52 scenes for one bottle |

Each JSONL record stores:

- evaluation, source, and original episode indices
- packaged object ID, category, and task text
- mesh scale
- final world position
- MuJoCo quaternion in `wxyz` order
- episode frame count

Create the same object scene with any supported hand:

```bash
python scripts/sim_view_dataset_episode.py \
  --dataset evaluation/object_episodes/shadow_grasp_0725.jsonl \
  --episode 20 \
  --hand shadow_hand_right

python scripts/sim_view_dataset_episode.py \
  --dataset evaluation/object_episodes/shadow_grasp_0725.jsonl \
  --episode 20 \
  --hand gaia_hand_right

python scripts/sim_view_dataset_episode.py \
  --dataset evaluation/object_episodes/shadow_grasp_0725.jsonl \
  --episode 20 \
  --hand sharpa_hand_right
```

The simulator combines the selected record with the packaged collision and
visual meshes, the requested hand, and `sim/assets/environment/base_scene.xml`.
The manifest intentionally excludes images, actions, and policy checkpoints.

### Reproduce an external dataset episode

The loader can also reconstruct a scene directly from the original LeRobot
dataset:

```bash
python scripts/sim_view_dataset_episode.py \
  --dataset ../datasets/lerobot_v21/shadow_grasp_0725 \
  --episode 20 \
  --hand shadow_hand_right
```

This maps the LeRobot episode to its source NPZ episode, reads `object_id`,
`object_scale`, `object_rotmat`, and `object_world_xy`, and computes the
tabletop height from the rotated collision mesh. Change `--hand` to create the
same object configuration with Gaia or Sharpa. If the source dataset recorded
in `collection_metadata.json` moved, pass its new path with
`--source_dataset`.

## Gesture-z policy

`GesturePolicyAdapter` exposes this policy contract:

```text
policy input state = [wrist 6D, z_gesture 24D]
policy output      = [wrist 6D, z_gesture 24D]
environment action = [wrist 6D, decoded target native q]
```

```python
from native_vae import NativeVAE
from sim import CallablePolicy, GesturePolicyAdapter

vae = NativeVAE.from_pretrained(device="cuda")
policy = GesturePolicyAdapter(
    CallablePolicy(model_predict),
    vae=vae,
    target_hand="sharpa_hand_right",
)
```

The six wrist outputs must already use the target MuJoCo wrist frame. Transform
them in `model_predict` when a policy emits a shared/source-hand palm frame.

## Evaluation client

```python
from sim import PolicyEvaluationClient

result = PolicyEvaluationClient(env, policy).run(
    initial_action=initial_action,
    max_steps=300,
)
print(result.success, result.max_lift_m)
```

The default success rule is lift >= 0.20 m for 10 consecutive policy frames.
Both values are configurable.

## Commands

```bash
# Physics and state API
python scripts/sim_smoke_test.py --no_render

# RGB API on a headless machine
MUJOCO_GL=osmesa python scripts/sim_smoke_test.py

# Replay [T,D] native actions and save a video
python scripts/sim_replay_actions.py \
  --hand shadow_hand_right \
  --actions trajectory.npz \
  --episode 0 \
  --output_video outputs/shadow_episode0.mp4
```

Use `--scene_xml` for a concrete hand/object scene. The shared base scene is
still applied so table, camera, lighting, and physics remain consistent.
Saving MP4 additionally requires the system `ffmpeg` executable.
