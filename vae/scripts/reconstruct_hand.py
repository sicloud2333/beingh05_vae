#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import as_numpy, load_q
from native_vae import NativeVAE


def main() -> None:
    parser = argparse.ArgumentParser(description="Run same-hand Native-VAE reconstruction.")
    parser.add_argument("--hand", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input_key", default="q")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    vae = NativeVAE.from_pretrained(device=args.device)
    q = load_q(args.input, args.input_key)
    result = vae.reconstruct(q, args.hand)
    error = as_numpy(result.fingerpad_error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        source_q=q,
        reconstructed_q=as_numpy(result.target_q),
        z_gesture=as_numpy(result.z_gesture),
        source_fingerpads=as_numpy(result.source_fingerpads),
        reconstructed_fingerpads=as_numpy(result.target_fingerpads),
        fingerpad_error=error,
        hand=np.asarray(args.hand),
        joint_names=np.asarray(vae.joint_names(args.hand)),
    )
    print(
        f"[done] {len(q)} poses, mean finger-pad error="
        f"{1000.0 * float(error.mean()):.3f} mm -> {args.output}"
    )


if __name__ == "__main__":
    main()
