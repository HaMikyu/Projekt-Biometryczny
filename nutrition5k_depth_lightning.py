import os
import random
from pathlib import Path

import lightning as L
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from helpers import find_overhead_depth, find_overhead_image, load_split_ids, resolve_split_map

DATA_DIR = Path(__file__).resolve().parent / "data" / "nutrition5k"
BATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_WORKERS = 0
DEPTH_MAX = 4000.0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Nutrition5kDepthDataset(Dataset):
    def __init__(self, samples, image_size=IMAGE_SIZE, train=False):
        self.samples = samples
        self.image_size = image_size
        self.train = train

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
        return img, depth, item["dish_id"]


class Nutrition5kDepthDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir=None,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE,
    ):
        super().__init__()
        self.data_dir = str(DATA_DIR if data_dir is None else data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def _build_split_samples(self, split_ids, images_root):
        samples = []
        for dish_id in split_ids:
            img = find_overhead_image(images_root, dish_id)
            depth = find_overhead_depth(images_root, dish_id)
            if img is None or depth is None:
                continue
            samples.append(
                {
                    "dish_id": dish_id,
                    "image_path": img,
                    "depth_path": depth,
                }
            )
        return samples

    def setup(self, stage=None):
        dish_ids_dir = os.path.join(self.data_dir, "dish_ids")
        images_root = os.path.join(self.data_dir, "imagery", "realsense_overhead")
        split_map = resolve_split_map(dish_ids_dir)

        if stage in (None, "fit"):
            self.train_ds = Nutrition5kDepthDataset(
                self._build_split_samples(load_split_ids(split_map["train"]), images_root),
                image_size=self.image_size,
                train=True,
            )
            self.val_ds = Nutrition5kDepthDataset(
                self._build_split_samples(load_split_ids(split_map["val"]), images_root),
                image_size=self.image_size,
                train=False,
            )
        if stage in (None, "test", "fit"):
            self.test_ds = Nutrition5kDepthDataset(
                self._build_split_samples(load_split_ids(split_map["test"]), images_root),
                image_size=self.image_size,
                train=False,
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


if __name__ == "__main__":
    dm = Nutrition5kDepthDataModule(batch_size=4)
    dm.setup("fit")

    n_train = 0 if dm.train_ds is None else len(dm.train_ds)
    n_val = 0 if dm.val_ds is None else len(dm.val_ds)
    print(f"data_dir: {dm.data_dir}")
    print(f"train samples: {n_train}, val samples: {n_val}")

    if n_train == 0 and n_val == 0:
        print("Brak probek RGB+depth. Sprawdz depth_raw.png w data/nutrition5k/imagery/realsense_overhead.")
    else:
        loader = dm.train_dataloader() if n_train > 0 else dm.val_dataloader()
        images, depths, dish_ids = next(iter(loader))
        print(f"images shape: {tuple(images.shape)}")
        print(f"depths shape: {tuple(depths.shape)}")
        print(f"dish_ids peek: {list(dish_ids[: min(3, len(dish_ids))])}")
        print(f"depth min/max: {depths.min().item():.4f} / {depths.max().item():.4f}")
