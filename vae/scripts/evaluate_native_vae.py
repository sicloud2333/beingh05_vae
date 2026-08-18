#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch


VAE_ROOT = Path(__file__).resolve().parents[1]
if str(VAE_ROOT) not in sys.path:
    sys.path.insert(0, str(VAE_ROOT))

from native_vae import NativeVAE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate same-hand reconstruction and all ordered cross-hand "
            "retargeting pairs on a NativeVAE tensor bundle."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summary(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all():
        raise RuntimeError("Metric input is empty or non-finite")
    return {
        "mean": float(flat.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(flat)))),
        "median": float(np.median(flat)),
        "p95": float(np.quantile(flat, 0.95)),
        "max": float(flat.max()),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_path = args.dataset.expanduser().resolve()
    if not checkpoint.is_file() or not dataset_path.is_file():
        raise FileNotFoundError(
            f"checkpoint={checkpoint.is_file()} dataset={dataset_path.is_file()}"
        )

    payload = torch.load(dataset_path, map_location="cpu")
    hand_names = tuple(payload["metadata"]["hand_names"])
    hand_ids = torch.as_tensor(payload["hand_ids"], dtype=torch.int64)
    padded_q = torch.as_tensor(payload["tensors"]["q"], dtype=torch.float32)
    indices = {
        hand: torch.nonzero(hand_ids == hand_id, as_tuple=False).flatten()
        for hand_id, hand in enumerate(hand_names)
    }
    vae = NativeVAE.from_pretrained(
        checkpoint=checkpoint,
        device=args.device,
    )
    if set(hand_names) != set(vae.hand_names):
        raise RuntimeError(
            f"Dataset/checkpoint hands differ: {hand_names} vs {vae.hand_names}"
        )

    result: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_path),
        "device": str(vae.device),
        "num_samples": int(len(hand_ids)),
        "samples_per_hand": {
            hand: int(len(indices[hand])) for hand in hand_names
        },
        "same_hand": {},
        "cross_hand": {},
    }
    started = perf_counter()

    for source_hand in hand_names:
        joint_dim = len(vae.joint_names(source_hand))
        source_q = padded_q[indices[source_hand], :joint_dim]
        q_errors: list[np.ndarray] = []
        tip_errors_m: list[np.ndarray] = []
        latent_values: list[np.ndarray] = []
        for start in range(0, len(source_q), args.batch_size):
            batch = source_q[start : start + args.batch_size]
            reconstructed = vae.reconstruct(batch, source_hand)
            q_errors.append(
                (reconstructed.target_q.cpu() - batch).numpy()
            )
            tip_errors_m.append(reconstructed.fingerpad_error.cpu().numpy())
            latent_values.append(reconstructed.z_gesture.cpu().numpy())
        q_error = np.concatenate(q_errors, axis=0)
        tip_error_m = np.concatenate(tip_errors_m, axis=0)
        z = np.concatenate(latent_values, axis=0)
        result["same_hand"][source_hand] = {
            "joint_error_rad": summary(q_error),
            "fingerpad_l2_m": summary(tip_error_m),
            "latent_abs": summary(np.abs(z)),
        }
        print(
            f"[same] {source_hand}: q_rmse={np.sqrt(np.mean(q_error**2)):.6f} "
            f"tip_mean_mm={tip_error_m.mean() * 1000:.3f}",
            flush=True,
        )

        for target_hand in hand_names:
            if target_hand == source_hand:
                continue
            raw_errors: list[np.ndarray] = []
            normalized_errors: list[np.ndarray] = []
            source_radius = float(vae.runtimes[source_hand].palm_radius)
            target_radius = float(vae.runtimes[target_hand].palm_radius)
            for start in range(0, len(source_q), args.batch_size):
                batch = source_q[start : start + args.batch_size]
                retargeted = vae.retarget(batch, source_hand, target_hand)
                source_tips = retargeted.source_fingerpads.cpu().numpy()
                target_tips = retargeted.target_fingerpads.cpu().numpy()
                raw_errors.append(
                    np.linalg.norm(target_tips - source_tips, axis=-1)
                )
                normalized_errors.append(
                    np.linalg.norm(
                        target_tips / target_radius - source_tips / source_radius,
                        axis=-1,
                    )
                )
            raw_error = np.concatenate(raw_errors, axis=0)
            normalized_error = np.concatenate(normalized_errors, axis=0)
            pair = f"{source_hand}->{target_hand}"
            result["cross_hand"][pair] = {
                "fingerpad_l2_m": summary(raw_error),
                "fingerpad_l2_palm_normalized": summary(normalized_error),
            }
            print(
                f"[cross] {pair}: tip_mean_mm={raw_error.mean() * 1000:.3f} "
                f"normalized_mean={normalized_error.mean():.6f}",
                flush=True,
            )

    result["elapsed_seconds"] = float(perf_counter() - started)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[done] {output}", flush=True)


if __name__ == "__main__":
    main()
