#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from native_vae.dataset import generate_tensor_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic random-q Native-URDF datasets.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train_native_n2.yaml")
    parser.add_argument("--device", default=None, help="Override generation device, e.g. cpu or cuda.")
    parser.add_argument("--train_samples_per_hand", type=int, default=None)
    parser.add_argument("--val_samples_per_hand", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    generation = config["data_generation"]
    device = args.device or generation.get("device", "cpu")
    common = {
        "hand_config": ROOT / config["hand_config"],
        "fk_batch_size": int(generation["fk_batch_size"]),
        "limit_shrink_ratio": float(generation["limit_shrink_ratio"]),
        "device": device,
    }
    generate_tensor_bundle(
        output=ROOT / config["train_data"],
        samples_per_hand=int(args.train_samples_per_hand or generation["train_samples_per_hand"]),
        seed=int(generation["train_seed"]),
        **common,
    )
    generate_tensor_bundle(
        output=ROOT / config["val_data"],
        samples_per_hand=int(args.val_samples_per_hand or generation["val_samples_per_hand"]),
        seed=int(generation["val_seed"]),
        **common,
    )


if __name__ == "__main__":
    main()
