"""Decode the whole overhead RGB-D set once into compact numpy arrays.

Output (data/nutrition5k/cache/):
    rgb_{H}x{W}.npy     uint8   [N, H, W, 3]
    height_{H}x{W}.npy  float16 [N, H, W]   height above table in cm (NaN = no depth reading)
    meta.npz            dish_ids, targets [N, 5] (kcal, mass, fat, carb, protein), split [N] (0 train, 1 val, 2 test)

Training then runs entirely from RAM/GPU with no PNG decoding per epoch.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from helpers import load_metadata, load_split_ids, resolve_split_map  # noqa: E402

DATA_DIR = ROOT / "data" / "nutrition5k"
TARGETS = ("calories", "total_mass", "total_fat", "total_carb", "total_protein")
DEPTH_UNIT_CM = 1e-2  # depth_raw.png is in 1e-4 m -> 0.01 cm
TABLE_FALLBACK = 3600.0


def table_level(depth: np.ndarray) -> float:
    """Robust per-image table distance: high percentile of plausible valid readings."""
    v = depth[(depth > 2500) & (depth < 5000)]
    if v.size < 1000:
        return TABLE_FALLBACK
    return float(np.percentile(v, 80))


def load_one(args):
    dish_dir, size = args
    h, w = size
    rgb = Image.open(dish_dir / "rgb.png").convert("RGB").resize((w, h), Image.BICUBIC)
    depth = np.asarray(Image.open(dish_dir / "depth_raw.png"), dtype=np.float32)
    table = table_level(depth)
    height = (table - depth) * DEPTH_UNIT_CM
    height[(depth <= 0) | (depth > 6000)] = np.nan
    # nearest-ish resize that keeps NaNs: resize value + validity separately
    valid = np.isfinite(height).astype(np.float32)
    hv = np.nan_to_num(height, nan=0.0)
    hv = np.asarray(Image.fromarray(hv).resize((w, h), Image.BILINEAR))
    vv = np.asarray(Image.fromarray(valid).resize((w, h), Image.BILINEAR))
    out = np.where(vv > 0.5, hv / np.maximum(vv, 1e-6), np.nan).astype(np.float16)
    return np.asarray(rgb, dtype=np.uint8), out


def load_ingredients(meta_dir: Path, ids: list[str]):
    """Per-dish ingredient grams [N, K] and per-gram nutrient density [K, 5] (kcal, g, fat, carb, prot).

    CSV row: dish_id, kcal, mass, fat, carb, protein, then repeated
    (ingr_id, name, grams, kcal, fat, carb, protein).
    """
    per_dish: dict[str, dict[str, float]] = {}
    names: dict[str, str] = {}
    totals: dict[str, np.ndarray] = {}
    for f in ("dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv"):
        with open(meta_dir / f, newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if not row:
                    continue
                parts = per_dish.setdefault(row[0], {})
                for k in range(6, len(row) - 6, 7):
                    iid, name = row[k], row[k + 1]
                    g, kcal, fat, carb, prot = map(float, row[k + 2:k + 7])
                    names[iid] = name
                    parts[iid] = parts.get(iid, 0.0) + g
                    totals.setdefault(iid, np.zeros(5))[:] += [kcal, g, fat, carb, prot]
    ingr_ids = sorted(names)
    col = {iid: j for j, iid in enumerate(ingr_ids)}
    mass = np.zeros((len(ids), len(ingr_ids)), np.float32)
    for i, d in enumerate(ids):
        for iid, g in per_dish.get(d, {}).items():
            mass[i, col[iid]] = g
    density = np.stack([totals[i] / max(totals[i][1], 1e-6) for i in ingr_ids]).astype(np.float32)
    return np.array(ingr_ids), np.array([names[i] for i in ingr_ids]), mass, density


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    args = p.parse_args()

    images_root = DATA_DIR / "imagery" / "realsense_overhead"
    labels = load_metadata(str(DATA_DIR / "metadata"))
    split_map = resolve_split_map(str(DATA_DIR / "dish_ids"))

    ids, splits = [], []
    for s_idx, key in enumerate(("train", "val", "test")):
        for dish in load_split_ids(split_map[key]):
            d = images_root / dish
            if dish in labels and (d / "rgb.png").is_file() and (d / "depth_raw.png").is_file() \
                    and (d / "depth_raw.png").stat().st_size > 0:
                ids.append(dish)
                splits.append(s_idx)
    print(f"dishes: {len(ids)}  (train {splits.count(0)}, val {splits.count(1)}, test {splits.count(2)})")

    size = (args.height, args.width)
    rgb = np.empty((len(ids), *size, 3), np.uint8)
    hgt = np.empty((len(ids), *size), np.float16)
    with ThreadPoolExecutor(args.workers) as pool:
        for i, (r, h) in enumerate(pool.map(load_one, [(images_root / d, size) for d in ids])):
            rgb[i], hgt[i] = r, h
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(ids)}")

    y = np.array([[labels[d][k] for k in TARGETS] for d in ids], np.float32)
    ingr_ids, ingr_names, ingr_mass, density = load_ingredients(DATA_DIR / "metadata", ids)
    out = DATA_DIR / "cache"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.height}x{args.width}"
    np.save(out / f"rgb_{tag}.npy", rgb)
    np.save(out / f"height_{tag}.npy", hgt)
    np.savez(out / "meta.npz", dish_ids=np.array(ids), targets=y, split=np.array(splits, np.int8),
             ingr_ids=ingr_ids, ingr_names=ingr_names, ingr_mass=ingr_mass, ingr_density=density)
    print(f"ingredients: {len(ingr_ids)}")
    print(f"saved -> {out}  rgb {rgb.nbytes / 2**30:.2f} GB, height {hgt.nbytes / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
