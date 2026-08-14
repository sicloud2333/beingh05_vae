#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from native_vae import NativeVAE  # noqa: E402


def load_q(path: Path) -> np.ndarray:
    payload = np.load(path)
    if isinstance(payload, np.ndarray):
        return np.asarray(payload, dtype=np.float32)
    for key in ("q", "action", "joint_q"):
        if key in payload:
            return np.asarray(payload[key], dtype=np.float32)
    raise KeyError(f"{path} must be .npy or contain one of: q, action, joint_q")


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode, reconstruct, or retarget Native-URDF hand q.")
    parser.add_argument("--mode", choices=("encode", "reconstruct", "retarget"), required=True)
    parser.add_argument("--source_hand", required=True)
    parser.add_argument("--target_hand", default=None)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/native_n2_epoch800_inference.pt")
    parser.add_argument("--model_config", type=Path, default=ROOT / "configs/model.yaml")
    parser.add_argument("--hand_config", type=Path, default=ROOT / "configs/right_hands.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    q = load_q(args.input)
    model = NativeVAE.from_pretrained(
        checkpoint=args.checkpoint,
        config=args.model_config,
        hand_config=args.hand_config,
        device=args.device,
    )
    source_q = model._q_tensor(q, args.source_hand)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "encode":
        z = model.encode(source_q, args.source_hand)
        np.savez_compressed(
            args.output,
            source_q=source_q.cpu().numpy(),
            z_gesture=z.cpu().numpy(),
            source_hand=np.asarray(args.source_hand),
        )
        print(f"[done] {args.output}: z_gesture={tuple(z.shape)}")
        return

    target_hand = args.source_hand if args.mode == "reconstruct" else args.target_hand
    if target_hand is None:
        parser.error("--target_hand is required for --mode retarget")
    result = model.retarget(source_q, args.source_hand, target_hand)
    np.savez_compressed(
        args.output,
        source_q=source_q.cpu().numpy(),
        z_gesture=result.z_gesture.cpu().numpy(),
        target_q=result.target_q.cpu().numpy(),
        source_fingerpads=result.source_fingerpads.cpu().numpy(),
        target_fingerpads=result.target_fingerpads.cpu().numpy(),
        fingerpad_error=result.fingerpad_error.cpu().numpy(),
        source_hand=np.asarray(args.source_hand),
        target_hand=np.asarray(target_hand),
    )
    mean_mm = float(result.fingerpad_error.mean() * 1000.0)
    print(f"[done] {args.output}: target_q={tuple(result.target_q.shape)} tip_error={mean_mm:.3f} mm")


if __name__ == "__main__":
    main()
