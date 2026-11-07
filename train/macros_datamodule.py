from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import lightning as L
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helpers import (
    find_overhead_depth,
    find_overhead_image,
    is_readable_image,
    load_metadata,
    load_split_ids,
    resolve_split_map,
)

DATA_DIR = ROOT / "data" / "nutrition5k"
BATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_WORKERS = 0
DEPTH_MAX = 4000.0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Nutrition5kRgbDepthMacrosDataset(Dataset):
    def __init__(self, samples, image_size=IMAGE_SIZE, train=False, predict_mass=False):
        self.samples = samples
        self.image_size = image_size
        self.train = train
        self.predict_mass = predict_mass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        depth = np.array(Image.open(item["depth_path"]), dtype=np.float32) / DEPTH_MAX
        depth = torch.from_numpy(depth).unsqueeze(0)

        img = TF.resize(img, [self.image_size, self.image_size])
        depth = TF.resize(depth, [self.image_size, self.image_size], antialias=True)

        if self.train and random.random() < 0.5:
            img = TF.hflip(img)
            depth = TF.hflip(depth)

        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        depth = depth.clamp(0, 1)

        y = [item["total_fat"], item["total_carb"], item["total_protein"]]
        if self.predict_mass:
            y.append(item["total_mass"])
        target = torch.tensor(y, dtype=torch.float32)
        return img, depth, target, item["dish_id"]


class Nutrition5kRgbDepthMacrosDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir=None,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE,
        predict_mass=False,
    ):
        super().__init__()
        self.data_dir = str(DATA_DIR if data_dir is None else data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.predict_mass = predict_mass
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def _build_split_samples(self, labels, split_ids, images_root):
        samples = []
        for dish_id in split_ids:
            if dish_id not in labels:
                continue
            img = find_overhead_image(images_root, dish_id)
            depth = find_overhead_depth(images_root, dish_id)
            if img is None or depth is None:
                continue
            if not is_readable_image(img) or not is_readable_image(depth):
                continue
            lab = labels[dish_id]
            samples.append(
                {
                    "dish_id": dish_id,
                    "image_path": img,
                    "depth_path": depth,
                    "total_fat": lab["total_fat"],
                    "total_carb": lab["total_carb"],
                    "total_protein": lab["total_protein"],
                    "total_mass": lab["total_mass"],
                    "calories": lab["calories"],
                }
            )
        return samples

    def setup(self, stage=None):
        meta_dir = os.path.join(self.data_dir, "metadata")
        dish_ids_dir = os.path.join(self.data_dir, "dish_ids")
        images_root = os.path.join(self.data_dir, "imagery", "realsense_overhead")
        labels = load_metadata(meta_dir)
        split_map = resolve_split_map(dish_ids_dir)

        if stage in (None, "fit"):
            self.train_ds = Nutrition5kRgbDepthMacrosDataset(
                self._build_split_samples(
                    labels, load_split_ids(split_map["train"]), images_root
                ),
                image_size=self.image_size,
                train=True,
                predict_mass=self.predict_mass,
            )
            self.val_ds = Nutrition5kRgbDepthMacrosDataset(
                self._build_split_samples(
                    labels, load_split_ids(split_map["val"]), images_root
                ),
                image_size=self.image_size,
                train=False,
                predict_mass=self.predict_mass,
            )
        if stage in (None, "test", "fit"):
            self.test_ds = Nutrition5kRgbDepthMacrosDataset(
                self._build_split_samples(
                    labels, load_split_ids(split_map["test"]), images_root
                ),
                image_size=self.image_size,
                train=False,
                predict_mass=self.predict_mass,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def macro_max(self) -> float:
        if self.train_ds is None:
            self.setup("fit")
        vals = []
        for s in self.train_ds.samples:
            vals.extend([s["total_fat"], s["total_carb"], s["total_protein"]])
            if self.predict_mass:
                vals.append(s["total_mass"])
        return float(max(vals)) if vals else 100.0
