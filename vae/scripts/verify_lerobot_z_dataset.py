#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq


VAE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VAE_ROOT.parent
for root in (VAE_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from native_vae import NativeVAE  # noqa: E402


STATE_KEYS = ("observation.state", "action")
WRIST_DIM = 6
SHADOW_JOINT_DIM = 22
PRESERVED_DIM = 28
EXPECTED_DIM = 52
Z_DIM = EXPECTED_DIM - PRESERVED_DIM


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that a z_gesture LeRobot conversion preserves wrist and "
            "Shadow joints exactly while replacing finite 24D latent values."
        )
    )
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--converted-dataset", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--atol",
        type=float,
        default=2e-3,
        help="Absolute CPU-vs-NPU encoding tolerance (calibrated on all 17,700 rows).",
    )
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def stats(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all():
        raise RuntimeError("Metric input is empty or non-finite")
    return {
        "mae": float(np.mean(np.abs(flat))),
        "rmse": float(np.sqrt(np.mean(np.square(flat)))),
        "max_abs": float(np.max(np.abs(flat))),
    }


def encode_in_batches(
    vae: NativeVAE,
    joints: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    chunks = []
    for start in range(0, len(joints), batch_size):
        z = vae.encode(
            joints[start : start + batch_size], "shadow_hand_right"
        )
        chunks.append(z.detach().cpu().numpy().astype(np.float32))
    result = np.concatenate(chunks, axis=0)
    if result.shape != (len(joints), Z_DIM) or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid re-encoded latent array: {result.shape}")
    return result


def main() -> None:
    args = parse_args()
    source = args.source_dataset.expanduser().resolve()
    converted = args.converted_dataset.expanduser().resolve()
    checkpoint = args.vae_checkpoint.expanduser().resolve()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.atol < 0.0 or args.rtol < 0.0:
        raise ValueError("--atol and --rtol must be non-negative")
    if not source.is_dir() or not converted.is_dir():
        raise FileNotFoundError(f"source={source} converted={converted}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint}")

    source_parquets = sorted(source.glob("data/**/*.parquet"))
    converted_parquets = sorted(converted.glob("data/**/*.parquet"))
    source_relatives = [path.relative_to(source) for path in source_parquets]
    converted_relatives = [path.relative_to(converted) for path in converted_parquets]
    if source_relatives != converted_relatives or not source_relatives:
        raise RuntimeError("Source/converted parquet manifests differ")

    vae = NativeVAE.from_pretrained(
        checkpoint=checkpoint,
        device=args.device,
    )
    checkpoint_sha256 = sha256_file(checkpoint)
    latent_differences: dict[str, list[np.ndarray]] = {key: [] for key in STATE_KEYS}
    latent_reencoding_errors: dict[str, list[np.ndarray]] = {
        key: [] for key in STATE_KEYS
    }
    total_rows = 0
    for index, relative in enumerate(source_relatives, start=1):
        source_table = pq.read_table(source / relative)
        converted_table = pq.read_table(converted / relative)
        if source_table.num_rows != converted_table.num_rows:
            raise RuntimeError(f"Row-count mismatch: {relative}")
        if source_table.schema.names != converted_table.schema.names:
            raise RuntimeError(f"Parquet schema column names changed: {relative}")
        for name in source_table.schema.names:
            if name not in STATE_KEYS and not source_table[name].equals(
                converted_table[name]
            ):
                raise RuntimeError(
                    f"Non-target parquet column changed at {relative}:{name}"
                )
        for key in STATE_KEYS:
            source_values = np.asarray(
                source_table[key].combine_chunks().to_pylist(), dtype=np.float32
            )
            converted_values = np.asarray(
                converted_table[key].combine_chunks().to_pylist(), dtype=np.float32
            )
            if source_values.shape != converted_values.shape:
                raise RuntimeError(f"Shape mismatch {relative}:{key}")
            if source_values.ndim != 2 or source_values.shape[1] != EXPECTED_DIM:
                raise RuntimeError(
                    f"Expected [N,{EXPECTED_DIM}] at {relative}:{key}, "
                    f"got {source_values.shape}"
                )
            if not np.array_equal(
                source_values[:, :PRESERVED_DIM],
                converted_values[:, :PRESERVED_DIM],
            ):
                maximum = float(
                    np.max(
                        np.abs(
                            source_values[:, :PRESERVED_DIM]
                            - converted_values[:, :PRESERVED_DIM]
                        )
                    )
                )
                raise RuntimeError(
                    f"Preserved prefix changed at {relative}:{key}; max_abs={maximum}"
                )
            latent = converted_values[:, PRESERVED_DIM:]
            if latent.shape[1] != EXPECTED_DIM - PRESERVED_DIM:
                raise RuntimeError(f"Invalid latent width at {relative}:{key}")
            if not np.isfinite(latent).all():
                raise RuntimeError(f"Non-finite latent at {relative}:{key}")
            expected_latent = encode_in_batches(
                vae,
                source_values[
                    :, WRIST_DIM : WRIST_DIM + SHADOW_JOINT_DIM
                ],
                args.batch_size,
            )
            reencoding_error = latent - expected_latent
            if not np.allclose(
                latent,
                expected_latent,
                atol=args.atol,
                rtol=args.rtol,
            ):
                maximum = float(np.max(np.abs(reencoding_error)))
                raise RuntimeError(
                    f"Latent does not match checkpoint encoding at "
                    f"{relative}:{key}; max_abs={maximum}, "
                    f"atol={args.atol}, rtol={args.rtol}"
                )
            latent_differences[key].append(
                latent - source_values[:, PRESERVED_DIM:]
            )
            latent_reencoding_errors[key].append(reencoding_error)
        total_rows += source_table.num_rows
        print(
            f"[verify] {index}/{len(source_relatives)} {relative} "
            f"rows={source_table.num_rows}",
            flush=True,
        )

    source_videos = sorted(
        path.relative_to(source) for path in source.glob("videos/**/*.mp4")
    )
    converted_videos = sorted(
        path.relative_to(converted) for path in converted.glob("videos/**/*.mp4")
    )
    if source_videos != converted_videos:
        raise RuntimeError("Source/converted video manifests differ")
    changed_video_sizes = [
        str(relative)
        for relative in source_videos
        if (source / relative).stat().st_size != (converted / relative).stat().st_size
    ]
    if changed_video_sizes:
        raise RuntimeError(
            f"Copied video sizes differ: {changed_video_sizes[:5]}"
        )

    required_metadata = (
        "meta/info.json",
        "meta/episodes.jsonl",
        "meta/episodes_stats.jsonl",
        "meta/tasks.jsonl",
        "meta/stats.json",
        "collection_metadata.json",
    )
    missing_metadata = [
        relative
        for relative in required_metadata
        if not (converted / relative).is_file()
    ]
    if missing_metadata:
        raise RuntimeError(f"Missing converted metadata: {missing_metadata}")

    marker_path = converted / ".z_gesture_conversion_complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError(f"Conversion marker is missing: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected_marker = {
        "source_dataset": str(source),
        "vae_checkpoint": str(checkpoint),
        "vae_checkpoint_sha256": checkpoint_sha256,
        "parquet_files": len(converted_parquets),
        "frames": int(total_rows),
        "latent_dimension": Z_DIM,
        "encoding": "posterior_mean_z_mu",
        "roundtrip_verified": True,
        "preserved_prefix_dimensions": PRESERVED_DIM,
    }
    for key, value in expected_marker.items():
        if marker.get(key) != value:
            raise RuntimeError(
                f"Conversion marker mismatch for {key}: "
                f"expected={value!r}, actual={marker.get(key)!r}"
            )

    collection_metadata = json.loads(
        (converted / "collection_metadata.json").read_text(encoding="utf-8")
    )
    z_metadata = collection_metadata.get("z_gesture")
    if not isinstance(z_metadata, dict):
        raise RuntimeError("collection_metadata.json lacks z_gesture metadata")
    if z_metadata.get("checkpoint") != str(checkpoint):
        raise RuntimeError("collection_metadata checkpoint does not match verifier")
    if z_metadata.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError("collection_metadata checkpoint SHA-256 is invalid")
    if int(z_metadata.get("dimension", -1)) != Z_DIM:
        raise RuntimeError("collection_metadata latent dimension is invalid")
    if z_metadata.get("value") != "posterior mean z_mu (deterministic)":
        raise RuntimeError("collection_metadata latent semantics are invalid")

    report = {
        "source_dataset": str(source),
        "converted_dataset": str(converted),
        "parquet_files": len(source_relatives),
        "video_files": len(source_videos),
        "rows": int(total_rows),
        "preserved_prefix_dimensions": PRESERVED_DIM,
        "latent_dimensions": EXPECTED_DIM - PRESERVED_DIM,
        "latent_change_vs_source": {
            key: stats(np.concatenate(chunks, axis=0))
            for key, chunks in latent_differences.items()
        },
        "vae_checkpoint": str(checkpoint),
        "vae_checkpoint_sha256": checkpoint_sha256,
        "verification_device": args.device,
        "content_tolerance": {"atol": args.atol, "rtol": args.rtol},
        "latent_reencoding_error": {
            key: stats(np.concatenate(chunks, axis=0))
            for key, chunks in latent_reencoding_errors.items()
        },
        "non_target_parquet_columns_exact": True,
        "marker_verified": True,
        "collection_metadata_verified": True,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[done] {output}: rows={total_rows}", flush=True)


if __name__ == "__main__":
    main()
