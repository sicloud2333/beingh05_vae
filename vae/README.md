# Native-URDF Gesture VAE

这是 Shadow、Gaia、Sharpa 三种右手的独立 Native-URDF VAE 项目，包含：

- 随机关节角数据生成与 batch FK
- Native-N2 模型训练及 WandB 日志
- 24D `z_gesture` 编码、同手重建、跨手重定向
- 三手共同语义坐标系下的骨架与指腹可视化
- 最终选择的 epoch-800 checkpoint
- 独立的 MuJoCo 抓取策略评估环境

项目不依赖 OrthoHand 主仓库中的 `src/`、`config/` 或 `outputs/`。

## 项目结构

```text
assets/                         # 三个 Native URDF
configs/
  model.yaml                    # 网络结构
  right_hands.yaml              # 关节顺序、语义槽位、指腹点
  train_native_n2.yaml          # 数据与训练配置
native_vae/
  dataset.py                    # tensor bundle 生成与读取
  losses.py                     # same-hand / cross-hand losses
  trainer.py                    # 最小训练循环
  api.py                        # Python 推理 API
  hand_runtime.py               # Native URDF、joint limit、batch FK
  model_components/             # Gesture/Morphology Encoder 和 Decoder
scripts/
  generate_random_data.py       # 生成随机 q 数据
  train.py                      # 训练
  infer.py                      # 编码、重建、重定向
  visualize.py                  # 骨架和指腹可视化
checkpoints/
  native_n2_epoch800_inference.pt
data/                           # 本地生成，不提交 Git
tests/                          # checkpoint 与接口回归测试
sim/                            # 可选的 MuJoCo 策略评估层
```

## 安装

仅推理：

```bash
cd vae
/home/wsy/anaconda3/envs/pch/bin/python -m pip install -e .
```

包含训练和 WandB：

```bash
/home/wsy/anaconda3/envs/pch/bin/python -m pip install -e '.[train]'
```

## 三只手的 q 顺序

输入 q 必须遵循 `configs/right_hands.yaml` 中的 `active_joint_names`：

| Hand | q dimension |
| --- | ---: |
| `shadow_hand_right` | 22 |
| `gaia_hand_right` | 15 |
| `sharpa_hand_right` | 22 |

Shadow 的 `WRJ1/WRJ2` 固定为零，只参与完整 URDF FK，不进入 VAE 输入输出。

## 1. 生成训练数据

默认配置生成：

- train：每只手 50,000 个姿态，共 150,000；
- val：每只手 5,000 个姿态，共 15,000；
- 采样范围：joint limit 中心 95%；
- FK batch size：2048；
- train/val seed：42/10042。

```bash
python scripts/generate_random_data.py \
  --config configs/train_native_n2.yaml
```

快速 smoke 数据：

```bash
python scripts/generate_random_data.py \
  --config configs/train_native_n2.yaml \
  --train_samples_per_hand 32 \
  --val_samples_per_hand 16
```

## 2. 训练

```bash
python scripts/train.py \
  --config configs/train_native_n2.yaml \
  --device cuda
```

训练计划：

- epoch 1-299：`q + same-hand finger-pad + KL`；
- epoch 300-800：加入 cross-hand absolute tip、pair-vector、pair-distance。

主要 loss 权重：

```text
1.0 * loss_q
+ 5.0 * loss_tip
+ 5.0 * loss_cross_tip_abs
+ 1.0 * loss_cross_pair_vector
+ 0.3 * loss_cross_pair_distance
+ 1e-4 * KL_action
+ 1e-5 * KL_morphology
```

训练输出位于 `runs/<run_name>_<timestamp>/`。包含：

```text
checkpoints/best.pt
checkpoints/last.pt
checkpoints/inference.pt
logs/metrics.jsonl
config_resolved.yaml
```

训练代码 smoke test：

```bash
python scripts/train.py \
  --config configs/train_native_n2.yaml \
  --device cpu \
  --epochs 1 \
  --max_train_batches 1 \
  --max_val_batches 1 \
  --no_wandb
```

## 3. 推理

统一输入格式为 `.npy`，或含 `q`、`action`、`joint_q` 的 `.npz`。

编码：

```bash
python scripts/infer.py \
  --mode encode \
  --source_hand shadow_hand_right \
  --input examples/shadow_q.npy \
  --output outputs/shadow_z.npz
```

同手重建：

```bash
python scripts/infer.py \
  --mode reconstruct \
  --source_hand shadow_hand_right \
  --input examples/shadow_q.npy \
  --output outputs/shadow_reconstruction.npz
```

跨手重定向：

```bash
python scripts/infer.py \
  --mode retarget \
  --source_hand shadow_hand_right \
  --target_hand sharpa_hand_right \
  --input examples/shadow_q.npy \
  --output outputs/shadow_to_sharpa.npz
```

重定向输出包含 `z_gesture`、目标 q、两只手的 finger-pad 位置和误差。

## 4. 可视化

三手零位姿、手指链和指腹点：

```bash
python scripts/visualize.py \
  --mode fingerpads \
  --output outputs/three_hand_fingerpads.html
```

跨手重定向 overlap：

```bash
python scripts/visualize.py \
  --mode retarget \
  --source_hand shadow_hand_right \
  --target_hand gaia_hand_right \
  --input examples/shadow_q.npy \
  --frame 0 \
  --output outputs/shadow_to_gaia.html
```

可视化采用三个 Native URDF 的共同语义坐标系。实线为 source，虚线为 target，
每根手指末端的大点是训练使用的 finger-pad。

## 5. 最终 checkpoint

默认推理权重：

```text
checkpoints/native_n2_epoch800_inference.pt
```

来源：

```text
Native-N2-pair-vector-distance-refine
epoch_0800.pt
```

该文件只保存模型推理权重，约 4.4 MB，不包含 optimizer 和 scheduler。

## Python API

```python
from native_vae import NativeVAE

vae = NativeVAE.from_pretrained(device="cuda")
z = vae.encode(shadow_q, hand="shadow_hand_right")
shadow_q_hat = vae.decode(z, hand="shadow_hand_right")
sharpa_q = vae.decode(z, hand="sharpa_hand_right")
```

## 测试

```bash
PYTHONPATH=. /home/wsy/anaconda3/envs/pch/bin/python -m unittest discover -s tests -v
```

测试覆盖最终 checkpoint 的 `z_gesture`、目标 q、finger-pad 位置以及 MuJoCo 接口。

## MuJoCo 抓取策略评估

`sim/` 是独立可选模块，接收 `wrist 6D + native q`，负责物理步进并返回 state、
物体位姿和图像。详细接口见 [SIMULATION.md](SIMULATION.md)。它不参与 Native-VAE
随机数据生成和表征训练。
