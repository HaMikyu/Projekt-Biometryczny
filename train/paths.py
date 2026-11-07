from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = Path(__file__).resolve().parent
BEST_DIR = TRAIN_DIR / "best"
RUNS_DIR = TRAIN_DIR / "runs"


def ensure_sys_path() -> Path:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return ROOT


def config_name(prefix: str, **parts: object) -> str:
    bits = [prefix]
    for k, v in parts.items():
        bits.append(f"{k}-{v}")
    return "_".join(str(b) for b in bits)
