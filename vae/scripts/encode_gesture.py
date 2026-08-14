#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import as_numpy, load_q
from native_vae import NativeVAE


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode native hand joint angles into 24D z_gesture.")
    parser.add_argument("--hand", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input_key", default="q")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    vae = NativeVAE.from_pretrained(device=args.device)
    q = load_q(args.input, args.input_key)
    z = as_numpy(vae.encode(q, args.hand))
    tips = as_numpy(vae.fingerpads(q, args.hand))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        q=q,
        z_gesture=z,
        fingerpads=tips,
        hand=np.asarray(args.hand),
        joint_names=np.asarray(vae.joint_names(args.hand)),
    )
    print(f"[done] encoded {len(q)} poses -> {args.output}")


if __name__ == "__main__":
    main()
