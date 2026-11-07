from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from train.dinov3_backbone import DEFAULT_MODEL_NAME, resolve_local_dino_dir
from train.export_best import export_best_weights
from train.macros_datamodule import Nutrition5kRgbDepthMacrosDataModule
from train.macros_model import MacrosLitModule, infer_num_classes
from train.paths import BEST_DIR, RUNS_DIR, config_name


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=str, default="auto")
    p.add_argument("--precision", type=str, default="32-true")
    p.add_argument("--head", type=str, choices=["mlp", "transformer", "cnn"], default="mlp")
    p.add_argument(
        "--task",
        type=str,
        choices=["regression", "classification"],
        default="regression",
    )
    p.add_argument("--bin-size", type=float, default=10.0)
    p.add_argument("--dino-model-id", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--dino-local-dir", type=str, default=None)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    L.seed_everything(args.seed, workers=True)

    name = config_name(
        "macros",
        head=args.head,
        task=args.task,
        bin=int(args.bin_size) if args.task == "classification" else "na",
    )
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    BEST_DIR.mkdir(parents=True, exist_ok=True)

    dm = Nutrition5kRgbDepthMacrosDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        predict_mass=False,
    )
    dm.setup("fit")
    num_classes = infer_num_classes(dm.macro_max(), args.bin_size)

    model = MacrosLitModule(
        head=args.head,
        task=args.task,
        bin_size=args.bin_size,
        num_classes=num_classes,
        num_outputs=3,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dino_model_id=args.dino_model_id,
        dino_local_dir=resolve_local_dino_dir(args.dino_local_dir),
        local_files_only=args.local_files_only,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="best",
        monitor="val_mae",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
    )
    early = EarlyStopping(monitor="val_mae", mode="min", patience=7)

    trainer = L.Trainer(
        default_root_dir=str(run_dir),
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=[ckpt_cb, early],
        log_every_n_steps=10,
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")

    if ckpt_cb.best_model_path:
        out = BEST_DIR / f"{name}.pt"
        export_best_weights(ckpt_cb.best_model_path, out)
        print(f"config={name}")
        print(f"num_classes={num_classes}")
        print(f"best_ckpt={ckpt_cb.best_model_path}")
        print(f"exported={out}")
        print(f"best_val_mae={ckpt_cb.best_model_score}")


if __name__ == "__main__":
    main()
