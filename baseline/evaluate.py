"""Evaluate every exported baseline head (baseline/best/macros_*.pt) on the full test split.

    python baseline/evaluate.py

Prints % MAE (MAE / mean target) for fat, carb, protein, plus kcal derived as
9*fat + 4*carb + 4*protein, so it is directly comparable with sota/train.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from baseline.macros_model import MacrosLitModule  # noqa: E402
from baseline.paths import BEST_DIR  # noqa: E402
from nutrition5k.rgbd_macros_datamodule import Nutrition5kRgbDepthMacrosDataModule  # noqa: E402


@torch.no_grad()
def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = Nutrition5kRgbDepthMacrosDataModule()
    dm.setup("test")
    batches = list(DataLoader(dm.test_ds, batch_size=64))
    y = torch.cat([b[2] for b in batches])[:, :3].numpy()
    kcal = np.array([s["calories"] for s in dm.test_ds.samples])
    print(f"test dishes: {len(y)}\n")
    print(f"{'config':<42}{'kcal*':>8}{'fat':>8}{'carb':>8}{'prot':>8}{'macro':>8}")

    for f in sorted(BEST_DIR.glob("macros_*.pt")):
        blob = torch.load(f, map_location="cpu", weights_only=False)
        m = MacrosLitModule(**blob["hyper_parameters"]).to(dev).eval()
        m.load_state_dict(blob["state_dict"], strict=False)
        preds = []
        for img, depth, _, _ in batches:
            out = m(img.to(dev), depth.to(dev))
            preds.append((out if m.task == "regression" else m._logits_to_values(out)).cpu())
        p = torch.cat(preds).numpy()
        pct = 100 * np.abs(p - y).mean(0) / y.mean(0)
        kcal_pct = 100 * np.abs((p * [9, 4, 4]).sum(1) - kcal).mean() / kcal.mean()
        name = f.stem.replace("macros_", "")
        print(f"{name:<42}{kcal_pct:>7.1f}%" + "".join(f"{v:>7.1f}%" for v in pct) + f"{pct.mean():>7.1f}%")
    print("\n* kcal derived as 9*fat + 4*carb + 4*protein (heads do not predict kcal/mass)")


if __name__ == "__main__":
    main()
