#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from native_vae.trainer import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the three-hand Native-URDF VAE.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train_native_n2.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    train(
        args.config,
        device_override=args.device,
        epochs_override=args.epochs,
        max_train_batches_override=args.max_train_batches,
        max_val_batches_override=args.max_val_batches,
        wandb_enabled_override=False if args.no_wandb else None,
    )


if __name__ == "__main__":
    main()
