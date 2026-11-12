"""Average saved test predictions of several runs.

    python sota/ensemble.py sota_vitb sota_vitb_hr
"""
import sys
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parents[1] / "runs"
NAMES = ["kcal", "mass", "fat", "carb", "protein"]

runs = [np.load(RUNS / r / "test_predictions.npz") for r in sys.argv[1:]]
assert runs and all((r["dish_ids"] == runs[0]["dish_ids"]).all() for r in runs)
y = runs[0]["target"]
pred = np.mean([r["pred"] for r in runs], 0)
pct = 100 * np.abs(pred - y).mean(0) / y.mean(0)
print("  ".join(f"{n} {p:.1f}%" for n, p in zip(NAMES, pct)), f"| mean {pct.mean():.2f}%")
