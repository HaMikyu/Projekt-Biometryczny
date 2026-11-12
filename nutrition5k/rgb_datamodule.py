import os
from pathlib import Path

import lightning as L
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from nutrition5k.helpers import (
    find_overhead_image,
    load_metadata,
    load_split_ids,
    resolve_split_map,
    show_batch_images,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "nutrition5k"
BATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_WORKERS = 0


class Nutrition5kDataset(Dataset):
    def __init__(self, samples, transform=None, predict_mass=False):
        self.samples = samples
        self.transform = transform
        self.predict_mass = predict_mass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        y = [item["total_fat"], item["total_carb"], item["total_protein"]]
        if self.predict_mass:
            y.append(item["total_mass"])
        target = torch.tensor(y, dtype=torch.float32)
        return img, target, item["dish_id"]


class Nutrition5kDataModule(L.LightningDataModule):
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

        self.train_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.eval_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _build_split_samples(self, labels, split_ids, images_root):
        samples = []
        for dish_id in split_ids:
            if dish_id not in labels:
                continue
            img = find_overhead_image(images_root, dish_id)
            if img is None:
                continue
            lab = labels[dish_id]
            samples.append(
                {
                    "dish_id": dish_id,
                    "image_path": img,
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
            self.train_ds = Nutrition5kDataset(
                self._build_split_samples(
                    labels, load_split_ids(split_map["train"]), images_root
                ),
                transform=self.train_tf,
                predict_mass=self.predict_mass,
            )
            self.val_ds = Nutrition5kDataset(
                self._build_split_samples(
                    labels, load_split_ids(split_map["val"]), images_root
                ),
                transform=self.eval_tf,
                predict_mass=self.predict_mass,
            )
        if stage in (None, "test", "fit"):
            self.test_ds = Nutrition5kDataset(
                self._build_split_samples(
                    labels, load_split_ids(split_map["test"]), images_root
                ),
                transform=self.eval_tf,
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


if __name__ == "__main__":
    dm = Nutrition5kDataModule(batch_size=4)
    dm.setup("fit")

    n_train = 0 if dm.train_ds is None else len(dm.train_ds)
    n_val = 0 if dm.val_ds is None else len(dm.val_ds)
    print(f"data_dir: {dm.data_dir}")
    print(f"train samples: {n_train}, val samples: {n_val}")

    if n_train == 0 and n_val == 0:
        print("Brak probek Nutrition5k (puste train i val). Sprawdz data/nutrition5k.")
    else:
        if n_train > 0:
            loader = dm.train_dataloader()
            split_name = "train"
        else:
            loader = dm.val_dataloader()
            split_name = "val"

        images, targets, dish_ids = next(iter(loader))
        print(f"batch from: {split_name}")
        print(f"images shape: {tuple(images.shape)}, dtype: {images.dtype}")
        print(f"targets shape: {tuple(targets.shape)}, dtype: {targets.dtype}")
        print(f"dish_ids: {len(dish_ids)} items (type={type(dish_ids[0]).__name__})")
        print(f"images min/max: {images.min().item():.4f} / {images.max().item():.4f}")
        print(f"dish_ids peek: {list(dish_ids[: min(3, len(dish_ids))])}")
        print(f"targets peek (fat, carb, protein[, mass]):")
        print(targets[: min(3, targets.shape[0])])
        show_batch_images(images, targets, dish_ids, n=min(4, images.shape[0]))
