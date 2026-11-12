from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-depth", action="store_true")
    p.add_argument("--max-epochs-depth", type=int, default=20)
    p.add_argument("--max-epochs-macros", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dino-local-dir", type=str, default=None)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=str, default="auto")
    p.add_argument("--precision", type=str, default="32-true")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main():
    args = parse_args()
    py = sys.executable
    common = [
        "--batch-size",
        str(args.batch_size),
        "--accelerator",
        args.accelerator,
        "--devices",
        str(args.devices),
        "--precision",
        args.precision,
    ]
    if args.dino_local_dir:
        common += ["--dino-local-dir", args.dino_local_dir]
    if args.local_files_only:
        common += ["--local-files-only"]

    if not args.skip_depth:
        run(
            [
                py,
                str(TRAIN_DIR / "train_depth.py"),
                "--max-epochs",
                str(args.max_epochs_depth),
                *common,
            ]
        )

    macros_grid = [
        ("mlp", "regression", 10),
        ("transformer", "regression", 10),
        ("cnn", "regression", 10),
        ("mlp", "classification", 5),
        ("mlp", "classification", 10),
        ("transformer", "classification", 5),
        ("transformer", "classification", 10),
        ("cnn", "classification", 5),
        ("cnn", "classification", 10),
    ]

    for head, task, bin_size in macros_grid:
        cmd = [
            py,
            str(TRAIN_DIR / "train_macros.py"),
            "--head",
            head,
            "--task",
            task,
            "--bin-size",
            str(bin_size),
            "--max-epochs",
            str(args.max_epochs_macros),
            *common,
        ]
        run(cmd)


if __name__ == "__main__":
    main()
