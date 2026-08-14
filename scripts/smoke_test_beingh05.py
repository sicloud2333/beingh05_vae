#!/usr/bin/env python3
"""Lightweight repository smoke test for the Being-H05 integration layer.

This intentionally does not load a model or modify vae/. It validates the
canonical entry points, local Shadow datasets, and registered data configs.
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--import-config", action="store_true", help="Import configs.data_config")
    parser.add_argument("--require-local-data", action="store_true", help="Require downloaded dataset metadata")
    args = parser.parse_args()

    required = [
        ROOT / "BeingH/train/train.py",
        ROOT / "BeingH/inference/beingh_policy.py",
        ROOT / "scripts/train/train_shadow_grasp.sh",
        ROOT / "scripts/eval/eval_shadow_grasp.sh",
        ROOT / "scripts/eval/eval_shadow_open_loop.sh",
        ROOT / "configs/data_config.py",
        ROOT / "configs/dataset_info.py",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing required files:\n  " + "\n  ".join(missing))

    dataset = ROOT / "data/shadow_grasp_bottle22249179_aug100_2cam"
    if args.require_local_data:
        for relative in (
            "meta/info.json",
            "meta/episodes.jsonl",
            "meta/tasks.jsonl",
            "meta/stats.json",
        ):
            if not (dataset / relative).is_file():
                raise SystemExit(f"Missing local dataset metadata: {dataset / relative}")
    elif not dataset.exists():
        print("Dataset not present; source-only smoke test continues.")

    for script in (
        ROOT / "scripts/train/train_shadow_grasp.sh",
        ROOT / "scripts/train/train_shadow_grasp_h100.sh",
        ROOT / "scripts/train/train_common.sh",
        ROOT / "scripts/eval/eval_shadow_grasp.sh",
        ROOT / "scripts/eval/eval_shadow_open_loop.sh",
    ):
        result = subprocess.run(["bash", "-n", str(script)], check=False)
        if result.returncode:
            raise SystemExit(f"bash syntax check failed: {script}")

    if args.import_config:
        module = importlib.import_module("configs.data_config")
        names = set(getattr(module, "DATA_CONFIG_MAP", {}).keys())
        expected = {
            "shadow_grasp_wrist_gesture",
            "shadow_grasp_wrist_gesture_q99",
            "shadow_grasp_2cam_wrist_rot6d_minmax_gesture_raw",
        }
        missing_configs = sorted(expected - names)
        if missing_configs:
            raise SystemExit(f"Missing expected data configs: {missing_configs}")

    print("Being-H05 smoke test passed (repository/config/entrypoint checks).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
