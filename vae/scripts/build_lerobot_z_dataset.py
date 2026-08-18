#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


VAE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VAE_ROOT.parent
for root in (VAE_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from BeingH.dataset.parquet_utils import calculate_dataset_statistics  # noqa: E402
from native_vae import NativeVAE  # noqa: E402


STATE_KEYS = ("observation.state", "action")
WRIST_DIM = 6
SHADOW_JOINT_DIM = 22
Z_DIM = 24
EXPECTED_DIM = WRIST_DIM + SHADOW_JOINT_DIM + Z_DIM


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-encode Shadow joint state/action with a NativeVAE checkpoint "
            "and build a self-contained 52D LeRobot v2.1 dataset."
        )
    )
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def numeric_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def encode_in_batches(
    vae: NativeVAE,
    joints: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    encoded = []
    for start in range(0, len(joints), batch_size):
        batch = joints[start : start + batch_size]
        z = vae.encode(batch, "shadow_hand_right")
        encoded.append(z.detach().cpu().numpy().astype(np.float32))
    result = np.concatenate(encoded, axis=0)
    if result.shape != (len(joints), Z_DIM) or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid encoded z_gesture shape/values: {result.shape}")
    return result


def replace_column(
    table: pa.Table,
    key: str,
    values: np.ndarray,
) -> pa.Table:
    index = table.schema.get_field_index(key)
    if index < 0:
        raise KeyError(f"Missing parquet column {key!r}")
    field = table.schema.field(index)
    encoded = pa.array(values.tolist(), type=field.type)
    return table.set_column(index, field, encoded)


def update_collection_metadata(
    output_dataset: Path,
    source_dataset: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
) -> None:
    path = output_dataset / "collection_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    metadata["source_dataset"] = str(source_dataset.resolve())
    metadata["repo_id"] = f"local/{output_dataset.name}"
    metadata["state"] = (
        "52D: measured wrist 6D + measured Shadow finger 22D + "
        "state z_gesture 24D"
    )
    metadata["action"] = (
        "52D: target wrist 6D + target Shadow finger 22D + "
        "action z_gesture 24D"
    )
    metadata["z_gesture"] = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "dimension": Z_DIM,
        "value": "posterior mean z_mu (deterministic)",
        "state_source": "observation.state[6:28]",
        "action_source": "action[6:28]",
        "builder": str(Path(__file__).resolve()),
    }
    path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    source = args.source_dataset.expanduser().resolve()
    output = args.output_dataset.expanduser().resolve()
    checkpoint = args.vae_checkpoint.expanduser().resolve()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {source}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint}")
    if output.exists():
        raise FileExistsError(
            f"Output dataset already exists: {output}. Choose a new directory."
        )
    checkpoint_sha256 = sha256_file(checkpoint)

    parquet_paths = sorted(source.glob("data/**/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files below {source / 'data'}")

    print(f"[copy] {source} -> {output}", flush=True)
    shutil.copytree(source, output, ignore=shutil.ignore_patterns(".cache"))
    vae = NativeVAE.from_pretrained(checkpoint=checkpoint, device=args.device)

    episode_stats: dict[int, dict[str, Any]] = {}
    for item_index, source_parquet in enumerate(parquet_paths, start=1):
        relative = source_parquet.relative_to(source)
        output_parquet = output / relative
        table = pq.read_table(source_parquet)
        updated_values: dict[str, np.ndarray] = {}
        for key in STATE_KEYS:
            raw = np.asarray(table[key].combine_chunks().to_pylist(), dtype=np.float32)
            if raw.ndim != 2 or raw.shape[1] != EXPECTED_DIM:
                raise ValueError(
                    f"{source_parquet}:{key} expected [N,{EXPECTED_DIM}], got {raw.shape}"
                )
            z = encode_in_batches(
                vae,
                raw[:, WRIST_DIM : WRIST_DIM + SHADOW_JOINT_DIM],
                args.batch_size,
            )
            values = raw.copy()
            values[:, WRIST_DIM + SHADOW_JOINT_DIM :] = z
            updated_values[key] = values
            table = replace_column(table, key, values)

        pq.write_table(table, output_parquet, compression=args.compression)
        roundtrip_table = pq.read_table(output_parquet)
        for name in table.schema.names:
            if name not in STATE_KEYS and not table[name].equals(
                roundtrip_table[name]
            ):
                raise RuntimeError(
                    f"Non-target column changed on write at {relative}:{name}"
                )
        for key, expected_values in updated_values.items():
            roundtrip_values = np.asarray(
                roundtrip_table[key].combine_chunks().to_pylist(),
                dtype=np.float32,
            )
            if not np.array_equal(roundtrip_values, expected_values):
                maximum = float(
                    np.max(np.abs(roundtrip_values - expected_values))
                )
                raise RuntimeError(
                    f"Encoded values changed on parquet round-trip at "
                    f"{relative}:{key}; max_abs={maximum}"
                )
        episode_column = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        episode_index = int(episode_column[0])
        if not np.all(episode_column == episode_index):
            raise ValueError(f"Mixed episode indices in {source_parquet}")
        episode_stats[episode_index] = {
            key: numeric_stats(values) for key, values in updated_values.items()
        }
        print(
            f"[encode] {item_index}/{len(parquet_paths)} "
            f"episode={episode_index} frames={table.num_rows}",
            flush=True,
        )

    episode_stats_path = output / "meta/episodes_stats.jsonl"
    existing_records = []
    if episode_stats_path.exists():
        existing_records = [
            json.loads(line)
            for line in episode_stats_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    records_by_episode = {
        int(record["episode_index"]): record for record in existing_records
    }
    for episode_index, stats in episode_stats.items():
        record = records_by_episode.setdefault(
            episode_index,
            {"episode_index": episode_index, "stats": {}},
        )
        record.setdefault("stats", {}).update(stats)
    episode_stats_path.write_text(
        "".join(
            json.dumps(records_by_episode[index], ensure_ascii=False) + "\n"
            for index in sorted(records_by_episode)
        ),
        encoding="utf-8",
    )

    output_parquets = sorted(output.glob("data/**/*.parquet"))
    statistics = calculate_dataset_statistics(output_parquets)
    (output / "meta/stats.json").write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    update_collection_metadata(
        output,
        source,
        checkpoint,
        checkpoint_sha256,
    )
    frame_count = sum(
        pq.read_metadata(path).num_rows for path in output_parquets
    )
    # Written last: downstream automation can use this as an unambiguous
    # signal that parquet rewriting and all metadata updates are complete.
    completion_marker = output / ".z_gesture_conversion_complete.json"
    completion_marker.write_text(
        json.dumps(
            {
                "source_dataset": str(source),
                "vae_checkpoint": str(checkpoint),
                "vae_checkpoint_sha256": checkpoint_sha256,
                "parquet_files": len(output_parquets),
                "frames": frame_count,
                "latent_dimension": Z_DIM,
                "encoding": "posterior_mean_z_mu",
                "roundtrip_verified": True,
                "preserved_prefix_dimensions": WRIST_DIM + SHADOW_JOINT_DIM,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[done] {output}: episodes={len(output_parquets)} "
        f"frames={frame_count} marker={completion_marker}",
        flush=True,
    )


if __name__ == "__main__":
    main()
