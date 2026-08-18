# Being-H05 Shadow Grasp

这是 Being-H05 Shadow grasp 项目的交付仓库，包含：

- Being-H05 模型、训练和推理代码；
- Shadow grasp 数据配置和归一化配置；
- Native VAE 跨手 retargeting 实现；
- Shadow physical-joint baseline；
- geometry retargeting baseline；
- Shadow、Sharpa、Gaia 的 MuJoCo evaluation 入口。

数据集、checkpoint需要从Hugging Face 单独下载。

## 目录结构

```text
BeingH/                 Being-H05 模型、数据、训练和推理代码
configs/                数据配置和 post-training 配置
scripts/train/          训练入口
scripts/eval/           open-loop 和 MuJoCo evaluation 入口
docs/                   训练、推理、评估和发布说明
vae/                    Native VAE、仿真资产和 geometry baseline
assets/                 仓库级资源
```

## 安装

```bash
git clone <BEING_H05_GITHUB_URL> Being-H05
cd Being-H05

conda create -n beingh05 python=3.10 -y
conda activate beingh05
pip install -r requirements.txt
pip install -r vae/requirements.txt
```

GPU 环境需要安装与 CUDA 驱动匹配的 PyTorch，并根据环境安装兼容版本的
`flash-attn`。

配置本地路径：

```bash
cp .env.example .env
```

至少需要设置：

```bash
BEINGH05_ROOT=/path/to/Being-H05
BEINGH_DATA_ROOT=/path/to/data
BEINGH_CKPT_ROOT=/path/to/ckpts
BEINGH_ENV=/path/to/conda/env
```


## 下载模型和数据

Being-H05 的基础 checkpoint 使用官方 Hugging Face collection：

<https://huggingface.co/collections/BeingBeyond/being-h05>

下载官方基础模型：

```bash
hf download BeingBeyond/Being-H05-2B \
  --local-dir ckpts/Being-H05-2B
```

下载主训练数据集：

```bash
hf download zju/shadow_grasp_bottle22249179_aug100_2cam \
  --repo-type dataset \
  --local-dir data/shadow_grasp_bottle22249179_aug100_2cam
```

主策略模型：

```bash
hf download zju/Being-H05-shadow-grasp-2cam-rot6d-zraw \
  --local-dir ckpts/Being-H05-shadow-grasp-2cam-rot6d-zraw
```

Native VAE：

```bash
mkdir -p vae/checkpoints
hf download zju/Being-H05-native-vae \
  native_n2_epoch800_inference.pt \
  --local-dir vae/checkpoints
```

physical-joint baseline：

```bash
hf download zju/Being-H05-shadow-grasp-2cam-joints \
  --local-dir ckpts/Being-H05-shadow-grasp-2cam-joints
```

Sharpa/Gaia 数据集和 baseline：

```bash
hf download zju/sharpa_grasp_bottle22249179_geo_visual100_2cam \
  --repo-type dataset \
  --local-dir data/sharpa_grasp_bottle22249179_geo_visual100_2cam

hf download zju/gaia_grasp_bottle22249179_geo_visual100_2cam \
  --repo-type dataset \
  --local-dir data/gaia_grasp_bottle22249179_geo_visual100_2cam

hf download zju/Being-H05-sharpa-joint-baseline \
  --local-dir ckpts/Being-H05-sharpa-joint-baseline

hf download zju/Being-H05-gaia-joint-baseline \
  --local-dir ckpts/Being-H05-gaia-joint-baseline
```

仓库链接：

- [Sharpa dataset](https://huggingface.co/datasets/zju/sharpa_grasp_bottle22249179_geo_visual100_2cam)
- [Gaia dataset](https://huggingface.co/datasets/zju/gaia_grasp_bottle22249179_geo_visual100_2cam)
- [Sharpa baseline](https://huggingface.co/zju/Being-H05-sharpa-joint-baseline)
- [Gaia baseline](https://huggingface.co/zju/Being-H05-gaia-joint-baseline)

官方依赖模型：

```bash
hf download OpenGVLab/InternVL3_5-2B --local-dir ckpts/InternVL3_5-2B
hf download Qwen/Qwen3-0.6B --local-dir ckpts/Qwen3-0.6B
```

## 训练

Shadow grasp 的标准训练入口：

```bash
CUDA_VISIBLE_DEVICES=1,2 \
NUM_GPUS=2 \
BEINGH_ENV=/path/to/conda/env \
EMBODIMENT_DATASET=shadow_grasp_bottle22249179_aug100_2cam \
NORMALIZATION=wrist_rot6d_minmax_zraw \
bash scripts/train/train_shadow_grasp.sh
```

短 smoke test：

```bash
SMOKE_TEST=True \
BEINGH_ENV=/path/to/conda/env \
bash scripts/train/train_shadow_grasp.sh
```

## Offline open-loop inference

```bash
bash scripts/eval/eval_shadow_open_loop.sh \
  --model-path outputs/<training-run>/0020000 \
  --dataset-path data/shadow_grasp_bottle22249179_aug100_2cam \
  --episode-index 0
```

多个 episode 会复用同一个已加载模型：

```bash
bash scripts/eval/eval_shadow_open_loop.sh \
  --model-path outputs/<training-run>/0040000 \
  --dataset-path data/shadow_grasp_bottle22249179_aug100_2cam \
  --episode-indices 0 1 2 3
```

Ascend NPU setup and the one-query reproduction path are documented in [docs/npu_inference.md](docs/npu_inference.md).
The full NativeVAE-to-Being-H05 training and verification runbook is in
[docs/npu_vae_training.md](docs/npu_vae_training.md).

## MuJoCo evaluation

```bash
bash scripts/eval/eval_shadow_grasp.sh \
  --model-path outputs/<training-run>/0040000 \
  --dataset vae/evaluation/object_episodes/<manifest>.jsonl \
  --episode-range 0 99 \
  --hand shadow_hand_right \
  --device cuda:0
```

Sharpa/Gaia zero-shot：

```bash
bash scripts/eval/eval_shadow_grasp.sh \
  --model-path outputs/<training-run>/0040000 \
  --dataset vae/evaluation/object_episodes/<manifest>.jsonl \
  --episode-range 0 99 \
  --hand sharpa_hand_right \
  --device cuda:0
```

## Geometry baseline

Shadow physical-joint checkpoint 在 Sharpa/Gaia 上使用 geometry retargeting：

```bash
bash scripts/eval/eval_shadow_grasp.sh \
  --model-path outputs/<joint-baseline-run>/0040000 \
  --dataset vae/evaluation/object_episodes/<manifest>.jsonl \
  --episode-range 0 99 \
  --hand sharpa_hand_right \
  --joint-retargeting geometry \
  --geometry-action-chunk-mode batch \
  --latent-observation-mode commanded \
  --device cuda:0
```

推理速度 benchmark：

```bash
python vae/scripts/benchmark_beingh_retargeting.py --help
```

## 验证

源码和入口检查（不需要下载数据）：

```bash
PYTHONPATH="$PWD" python scripts/smoke_test_beingh05.py
```

安装依赖并下载数据后：

```bash
PYTHONPATH="$PWD" python scripts/smoke_test_beingh05.py \
  --import-config \
  --require-local-data
```


## 文档

- [Training](docs/training.md)
- [Inference](docs/inference.md)
- [Evaluation](docs/evaluation.md)
- [Data configuration](docs/data_configuration.md)
- [Unified action space](docs/unified_action_space.md)
- [Release layout](docs/release.md)
- [Native VAE 总览、训练与推理](vae/README.md)
- [Native VAE 验证说明](vae/VALIDATION.md)
- [VAE 仿真与评估](vae/evaluation/README.md)
