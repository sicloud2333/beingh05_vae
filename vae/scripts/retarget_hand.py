#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import as_numpy, load_q
from native_vae import NativeVAE


def main() -> None:
    parser = argparse.ArgumentParser(description="Retarget native joint poses between hand embodiments.")
    parser.add_argument("--source_hand", required=True)
    parser.add_argument("--target_hand", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input_key", default="q")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    vae = NativeVAE.from_pretrained(device=args.device)
    source_q = load_q(args.input, args.input_key)
    result = vae.retarget(source_q, args.source_hand, args.target_hand)
    error = as_numpy(result.fingerpad_error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        source_q=source_q,
        target_q=as_numpy(result.target_q),
        z_gesture=as_numpy(result.z_gesture),
        source_fingerpads=as_numpy(result.source_fingerpads),
        target_fingerpads=as_numpy(result.target_fingerpads),
        fingerpad_error=error,
        source_hand=np.asarray(args.source_hand),
        target_hand=np.asarray(args.target_hand),
        source_joint_names=np.asarray(vae.joint_names(args.source_hand)),
        target_joint_names=np.asarray(vae.joint_names(args.target_hand)),
    )
    print(
        f"[done] {args.source_hand} -> {args.target_hand}, {len(source_q)} poses, "
        f"mean finger-pad error={1000.0 * float(error.mean()):.3f} mm -> {args.output}"
    )


if __name__ == "__main__":
    main()
