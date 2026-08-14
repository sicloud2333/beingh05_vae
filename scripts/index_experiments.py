#!/usr/bin/env python3
"""Build compact indexes for existing Being-H05 outputs and results.

The script only reads artifacts and writes a Markdown index. It never moves,
renames, or deletes checkpoints/results.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def checkpoint_rows(outputs: Path) -> list[tuple[str, str, int, str]]:
    rows = []
    if not outputs.is_dir():
        return rows
    for run in sorted(outputs.iterdir()):
        if not run.is_dir():
            continue
        checkpoints = sorted(
            (child for child in run.iterdir() if child.is_dir() and child.name.isdigit()),
            key=lambda item: int(item.name),
        )
        if not checkpoints:
            continue
        latest = checkpoints[-1]
        rows.append((run.name, str(latest.relative_to(ROOT)), len(checkpoints), "yes" if (run / "training.log").is_file() else "no"))
    return rows


def result_rows(results: Path) -> list[tuple[str, int, int]]:
    rows = []
    if not results.is_dir():
        return rows
    for group in sorted(results.iterdir()):
        if not group.is_dir():
            continue
        summaries = list(group.rglob("summary*.json"))
        rollouts = list(group.rglob("rollout.npz"))
        if summaries or rollouts:
            rows.append((group.name, len(summaries), len(rollouts)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiment_index.md")
    args = parser.parse_args()
    outputs = ROOT / "outputs"
    results = ROOT / "results"
    lines = [
        "# Being-H05 experiment index",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "This is an index only; no checkpoint or result directory is moved or deleted.",
        "",
        "## Training runs and checkpoints",
        "",
        "| Run | Latest checkpoint | Number of checkpoints | training.log |",
        "|---|---|---:|:---:|",
    ]
    rows = checkpoint_rows(outputs)
    if rows:
        lines.extend(f"| `{run}` | `{latest}` | {count} | {log} |" for run, latest, count, log in rows)
    else:
        lines.append("| *(none found)* | | | |")
    lines += [
        "",
        "## Result groups",
        "",
        "| Group | Summary JSON files | Rollout NPZ files |",
        "|---|---:|---:|",
    ]
    rows = result_rows(results)
    if rows:
        lines.extend(f"| `{group}` | {summaries} | {rollouts} |" for group, summaries, rollouts in rows)
    else:
        lines.append("| *(none found)* | 0 | 0 |")
    lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
