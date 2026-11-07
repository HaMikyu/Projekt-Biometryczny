import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def parse_metadata_row(row):
    dish_id = row[0]
    try:
        total_calories = float(row[1])
        total_mass = float(row[2])
        total_fat = float(row[3])
        total_carb = float(row[4])
        total_protein = float(row[5])
    except (IndexError, ValueError):
        return None
    return {
        "dish_id": dish_id,
        "calories": total_calories,
        "total_mass": total_mass,
        "total_fat": total_fat,
        "total_carb": total_carb,
        "total_protein": total_protein,
    }


def load_metadata(meta_dir):
    out = {}
    for name in ("dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv"):
        path = os.path.join(meta_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                item = parse_metadata_row(row)
                if item is not None:
                    out[item["dish_id"]] = item
    return out


def load_split_ids(path):
    ids = []
    if not os.path.isfile(path):
        return ids
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s)
    return ids


def find_overhead_image(images_root, dish_id):
    base = Path(images_root) / dish_id
    candidates = [
        base / "rgb.png",
        base / "rgb.jpg",
        base / "overhead_rgb.png",
        base / "overhead_rgb.jpg",
        Path(images_root) / f"{dish_id}.png",
        Path(images_root) / f"{dish_id}.jpg",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def resolve_split_map(dish_ids_dir):
    split_map = {
        "train": os.path.join(dish_ids_dir, "train_ids.txt"),
        "val": os.path.join(dish_ids_dir, "val_ids.txt"),
        "test": os.path.join(dish_ids_dir, "test_ids.txt"),
    }
    if os.path.isfile(split_map["train"]):
        return split_map
    alt = {
        "train": os.path.join(dish_ids_dir, "overhead_train_ids.txt"),
        "val": os.path.join(dish_ids_dir, "overhead_val_ids.txt"),
        "test": os.path.join(dish_ids_dir, "overhead_test_ids.txt"),
    }
    if os.path.isfile(alt["train"]):
        return alt
    return split_map


def denormalize_imagenet(img):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1)
    out = img * std + mean
    return out.clamp(0, 1)


def show_batch_images(images, targets, dish_ids, n=4):
    n = min(n, images.shape[0])
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.8))
    if n == 1:
        axes = [axes]
    for i in range(n):
        img = denormalize_imagenet(images[i].detach().cpu())
        img = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        t = targets[i]
        title = f"{dish_ids[i]}\nfat={t[0]:.1f} carb={t[1]:.1f} prot={t[2]:.1f}"
        if t.numel() > 3:
            title += f" mass={t[3]:.1f}"
        axes[i].set_title(title, fontsize=9, pad=18)
        axes[i].axis("off")
    plt.tight_layout(pad=2.0, rect=[0, 0, 1, 0.92])
    fig.subplots_adjust(top=0.78)
    plt.show()
