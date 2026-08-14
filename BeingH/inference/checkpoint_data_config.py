"""Resolve the data transform associated with a Being-H checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Collection


def _collect_named_values(payload: Any, key_names: set[str]) -> list[str]:
    values: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in key_names:
                if isinstance(value, str):
                    values.append(value)
                elif isinstance(value, list):
                    values.extend(item for item in value if isinstance(item, str))
            values.extend(_collect_named_values(value, key_names))
    elif isinstance(payload, list):
        for value in payload:
            values.extend(_collect_named_values(value, key_names))
    return values


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def resolve_checkpoint_data_config(
    checkpoint: Path,
    requested: str | None,
    supported: Collection[str],
) -> tuple[str, str]:
    """Return ``(data_config_name, source)`` for an evaluation checkpoint.

    Explicit CLI selection takes precedence. Automatic detection first uses
    config.json fields written by current training code, then falls back to the
    run_config YAML snapshot kept next to older numeric checkpoints.
    """

    checkpoint = checkpoint.expanduser().resolve()
    supported_set = set(supported)
    if requested is not None:
        if requested not in supported_set:
            raise ValueError(
                f"Unsupported data config {requested!r}; "
                f"supported={sorted(supported_set)}"
            )
        return requested, "command line"

    config_path = checkpoint / "config.json"
    if config_path.is_file():
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        config_names = _unique(
            _collect_named_values(
                config_payload,
                {"data_config_name", "data_config_names"},
            )
        )
        if config_names:
            unsupported = [name for name in config_names if name not in supported_set]
            if unsupported:
                raise ValueError(
                    f"{config_path} records unsupported data config(s) "
                    f"{unsupported}; supported={sorted(supported_set)}"
                )
            if len(config_names) != 1:
                raise ValueError(
                    f"{config_path} records multiple data configs {config_names}. "
                    "Pass --data-config-name explicitly for this evaluation."
                )
            return config_names[0], str(config_path)

    # Checkpoints produced before data_config_name was persisted still retain
    # an exact copy of the dataset YAML under <training-run>/run_config.
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to recover data_config_name from legacy run_config"
        ) from exc

    yaml_paths: list[Path] = []
    for base in (checkpoint, checkpoint.parent):
        run_config_dir = base / "run_config"
        if run_config_dir.is_dir():
            yaml_paths.extend(sorted(run_config_dir.glob("*.yaml")))
            yaml_paths.extend(sorted(run_config_dir.glob("*.yml")))

    yaml_names: list[str] = []
    sources: list[Path] = []
    for yaml_path in _unique([str(path) for path in yaml_paths]):
        path = Path(yaml_path)
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        names = _collect_named_values(payload, {"data_config_names"})
        if names:
            yaml_names.extend(names)
            sources.append(path)
    yaml_names = _unique(yaml_names)

    if yaml_names:
        unsupported = [name for name in yaml_names if name not in supported_set]
        if unsupported:
            raise ValueError(
                f"Legacy run_config records unsupported data config(s) "
                f"{unsupported}; supported={sorted(supported_set)}"
            )
        if len(yaml_names) != 1:
            raise ValueError(
                f"Legacy run_config records multiple data configs {yaml_names}. "
                "Pass --data-config-name explicitly for this evaluation."
            )
        return yaml_names[0], ", ".join(str(path) for path in sources)

    raise ValueError(
        f"Cannot determine the data config used by checkpoint {checkpoint}. "
        "Its config.json has no data_config_name and no run_config YAML was found. "
        "Pass --data-config-name explicitly for this legacy checkpoint."
    )
