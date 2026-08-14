from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_q(path: str | Path, key: str = "q") -> np.ndarray:
    path = Path(path)
    payload = np.load(path)
    if isinstance(payload, np.lib.npyio.NpzFile):
        if key not in payload:
            raise KeyError(f"{path} has keys {payload.files}, not {key!r}")
        value = payload[key]
    else:
        value = payload
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim != 2:
        raise ValueError(f"Expected q [B,D], got {value.shape}")
    return value


def as_numpy(value):
    return value.detach().cpu().numpy()
