"""Fast single-GPU fine-tuning of the RGB-D DINOv3 nutrition regressor.

    python train/sota/build_cache.py            # once (~15 s)
    python train/sota/train.py                  # train + test, ViT-B, 40 epochs
    python train/sota/train.py --eval-only --ckpt train/runs/sota_vitb/best.pt

Everything after the cache lives in RAM; augmentation runs on the GPU, so the
loop is bounded by the GPU, not by PNG decoding or Windows dataloader workers.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from train.sota.model import NutritionNet, param_groups  # noqa: E402

CACHE = ROOT / "data" / "nutrition5k" / "cache"
RUNS = ROOT / "train" / "runs"
NAMES = ["kcal", "mass", "fat", "carb", "protein"]
# Nutrition5k paper (Thames et al., CVPR 2021), overhead RGB-D direct prediction, % MAE
PAPER_RGBD_PCT = {"kcal": 18.8, "mass": 18.9, "fat": 18.1, "carb": 23.8, "protein": 20.9}
# Ingredient-Guided RGB-D Fusion Network (2024), reported PMAE - recent published RGB-D SOTA
SOTA_RGBD_PCT = {"kcal": 12.2, "mass": 9.4, "fat": 19.1, "carb": 18.3, "protein": 16.0}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="sota_vitb")
    p.add_argument("--backbone", default="vit_base_patch16_dinov3.lvd1689m")
    p.add_argument("--cache-size", default="384x512")
    p.add_argument("--input-size", default="336x448", help="HxW fed to the ViT (multiple of 16)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16, help="effective batch (per optimizer step)")
    p.add_argument("--accum", type=int, default=1, help="split each batch into N micro-batches (VRAM)")
    p.add_argument("--lr", type=float, default=6e-5, help="LR of the last ViT block")
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--layer-decay", type=float, default=0.8)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=2.0)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--consistency", type=float, default=0.1, help="weight of kcal ~ 9f+4c+4p loss")
    p.add_argument("--max-kcal", type=float, default=3000.0, help="drop label outliers from TRAIN only")
    p.add_argument("--no-depth", action="store_true")
    p.add_argument("--ingr", action="store_true",
                   help="add the ingredient-grams head (aux; on the test split it did not help)")
    p.add_argument("--ingr-w", type=float, default=0.05, help="weight of per-ingredient grams loss")
    p.add_argument("--ingr-total-w", type=float, default=1.0, help="weight of totals derived from grams")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--train-on-val", action="store_true", help="final run: merge val into train")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--limit-steps", type=int, default=0, help="debug: stop epoch after N steps")
    return p.parse_args()


def hw(s: str):
    h, w = s.lower().split("x")
    return int(h), int(w)


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self, cache_size: str):
        meta = np.load(CACHE / "meta.npz")
        self.ids = meta["dish_ids"]
        self.y = meta["targets"].astype(np.float32)
        self.split = meta["split"]
        self.ingr = meta["ingr_mass"].astype(np.float32)
        self.density = meta["ingr_density"].astype(np.float32)
        self.rgb = np.load(CACHE / f"rgb_{cache_size}.npy", mmap_mode="r")
        self.hgt = np.load(CACHE / f"height_{cache_size}.npy", mmap_mode="r")
        # pull into RAM once (3 GB) - way faster than mmap random access
        t = time.time()
        self.rgb = np.ascontiguousarray(self.rgb)
        self.hgt = np.ascontiguousarray(self.hgt)
        print(f"cache loaded in {time.time() - t:.1f}s: {len(self.ids)} dishes")

    def batch(self, idx: np.ndarray):
        idx = np.sort(idx)
        rgb = torch.from_numpy(self.rgb[idx]).pin_memory()
        hgt = torch.from_numpy(self.hgt[idx]).pin_memory()
        y = torch.from_numpy(self.y[idx]).pin_memory()
        g = torch.from_numpy(self.ingr[idx]).pin_memory()
        return rgb, hgt, y, g, idx


def prefetch(data: Data, batches, depth: int = 3):
    """Background thread that slices + pins batches while the GPU works."""
    q: queue.Queue = queue.Queue(depth)

    def worker():
        for b in batches:
            q.put(data.batch(b))
        q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


def to_gpu(rgb_u8, hgt_f16, dev):
    rgb = rgb_u8.to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    h = hgt_f16.to(dev, non_blocking=True).float().unsqueeze(1)
    valid = torch.isfinite(h).float()
    h = torch.nan_to_num(h, nan=0.0)
    return rgb, h, valid


def gpu_augment(rgb, h, valid, out_hw, train: bool, flip: tuple[bool, bool] = (False, False)):
    """Crop to out_hw with random translation/rotation/flips (train) or center crop (eval).

    No scale jitter on purpose: the camera is fixed, apparent size == portion size.
    """
    b, _, H, W = rgb.shape
    oh, ow = out_hw
    dev = rgb.device
    sx, sy = ow / W, oh / H
    if train:
        ang = (torch.rand(b, device=dev) * 2 - 1) * math.radians(12)
        tx = (torch.rand(b, device=dev) * 2 - 1) * (1 - sx)
        ty = (torch.rand(b, device=dev) * 2 - 1) * (1 - sy)
        fx = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
        fy = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
    else:
        ang = torch.zeros(b, device=dev)
        tx = ty = torch.zeros(b, device=dev)
        fx = torch.full((b,), -1.0 if flip[0] else 1.0, device=dev)
        fy = torch.full((b,), -1.0 if flip[1] else 1.0, device=dev)
    cos, sin = torch.cos(ang), torch.sin(ang)
    # rotation in pixel space (aspect-correct), expressed in normalised coords
    ar = W / H
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = cos * sx * fx
    theta[:, 0, 1] = -sin * sy / ar * fy
    theta[:, 1, 0] = sin * sx * ar * fx
    theta[:, 1, 1] = cos * sy * fy
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, (b, 1, oh, ow), align_corners=False)
    rgb = F.grid_sample(rgb, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    hv = F.grid_sample(torch.cat([h * valid, valid], 1), grid, mode="bilinear",
                       padding_mode="zeros", align_corners=False)
    valid = (hv[:, 1:] > 0.5).float()
    h = hv[:, :1] / hv[:, 1:].clamp(min=1e-6) * valid

    if train:  # photometric jitter
        br = 1 + (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * 0.2
        co = 1 + (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * 0.2
        sa = 1 + (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * 0.2
        rgb = rgb * br
        m = rgb.mean((1, 2, 3), keepdim=True)
        rgb = (rgb - m) * co + m
        g = rgb.mean(1, keepdim=True)
        rgb = ((rgb - g) * sa + g).clamp(0, 1)
        h = h + torch.randn(b, 1, 1, 1, device=dev) * 0.2 * valid  # table-level jitter (cm)
    rgb = (rgb - MEAN.to(dev)) / STD.to(dev)
    return rgb, h, valid


# ----------------------------------------------------------------------------- utils
class Ema:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.step = 0

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        ema_p = [p for p in self.model.state_dict().values() if p.dtype.is_floating_point]
        src_p = [p for p in model.state_dict().values() if p.dtype.is_floating_point]
        torch._foreach_lerp_(ema_p, src_p, 1 - d)


@torch.no_grad()
def predict(model, data: Data, idx, args, dev, tta=True):
    model.eval()
    out_hw = hw(args.input_size)
    flips = [(False, False), (True, False), (False, True), (True, True)] if tta else [(False, False)]
    preds, order = [], []
    batches = [idx[i:i + 32] for i in range(0, len(idx), 32)]
    for rgb_u8, h16, _, _, bidx in prefetch(data, batches):
        rgb0, h0, v0 = to_gpu(rgb_u8, h16, dev)
        acc = 0
        for f in flips:
            rgb, h, v = gpu_augment(rgb0, h0, v0, out_hw, train=False, flip=f)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                acc = acc + model(rgb, h, v)
        preds.append(acc / len(flips))
        order.append(bidx)
    return torch.cat(preds).cpu().numpy(), np.concatenate(order)


def blend(pred, alpha):
    """pred [N, 2, 5] (direct, from-ingredients) -> [N, 5]; alpha = weight of direct per target."""
    alpha = np.asarray(alpha, np.float32)
    return alpha * pred[:, 0] + (1 - alpha) * pred[:, 1]


def fit_alpha(pred, y):
    grid = np.linspace(0, 1, 11)
    errs = np.stack([np.abs(blend(pred, a) - y).mean(0) for a in grid])  # [11, 5]
    return grid[errs.argmin(0)]


def metrics(pred, y):
    mae = np.abs(pred - y).mean(0)
    pct = 100 * mae / y.mean(0)
    return {n: {"mae": float(m), "pct": float(p)} for n, m, p in zip(NAMES, mae, pct)}


def fmt(m):
    return "  ".join(f"{n} {m[n]['mae']:.1f} ({m[n]['pct']:.1f}%)" for n in NAMES)


def mean_pct(m):
    return float(np.mean([m[n]["pct"] for n in NAMES]))


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")

    run_dir = RUNS / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    data = Data(args.cache_size)

    tr = np.where(data.split == 0)[0]
    va = np.where(data.split == 1)[0]
    te = np.where(data.split == 2)[0]
    if args.train_on_val:
        tr = np.concatenate([tr, va])
    keep = (data.y[tr, 0] <= args.max_kcal) & (data.y[tr, 1] <= args.max_kcal)
    print(f"train {len(tr)} -> {keep.sum()} after dropping label outliers; val {len(va)}; test {len(te)}")
    tr = tr[keep]

    y_mean = torch.tensor(data.y[tr].mean(0), device=dev)
    y_std = torch.tensor(data.y[tr].std(0), device=dev)

    num_ingr = data.ingr.shape[1] if args.ingr else 0
    model = NutritionNet(args.backbone, drop_path=args.drop_path, use_depth=not args.no_depth,
                         num_ingr=num_ingr).to(dev)
    density = torch.tensor(data.density, device=dev)
    if args.grad_ckpt:
        model.set_grad_checkpointing(True)

    def denorm(o):
        return o * y_std + y_mean

    class Wrapped(torch.nn.Module):  # -> [B, 2, 5] in kcal / grams: (direct, from ingredients)
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, *a):
            direct, grams = self.m(*a)
            direct = denorm(direct.float())
            derived = grams @ density if grams is not None else direct
            return torch.stack([direct, derived], 1)

    if args.eval_only:
        ck = torch.load(args.ckpt or run_dir / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        y_mean.copy_(torch.tensor(ck["y_mean"]))
        y_std.copy_(torch.tensor(ck["y_std"]))
        evaluate_and_report(Wrapped(model), data, va, te, args, dev, run_dir, ck.get("epoch"))
        return

    opt = torch.optim.AdamW(
        param_groups(model, args.lr, args.head_lr, args.weight_decay, args.layer_decay),
        betas=(0.9, 0.999), fused=True,
    )
    ema = Ema(model, args.ema)
    steps_per_epoch = len(tr) // args.batch_size
    if args.limit_steps:
        steps_per_epoch = min(steps_per_epoch, args.limit_steps)
    total = steps_per_epoch * args.epochs
    warm = int(steps_per_epoch * args.warmup_epochs)
    kcal_w = torch.tensor([9.0, 4.0, 4.0], device=dev)
    out_hw = hw(args.input_size)
    assert args.batch_size % args.accum == 0
    micro = args.batch_size // args.accum

    best, best_epoch, step = float("inf"), -1, 0
    log = open(run_dir / "log.csv", "w", encoding="utf-8")
    log.write("epoch,train_loss,val_mean_pct," + ",".join(f"val_{n}_pct" for n in NAMES) + ",sec\n")
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, {steps_per_epoch} steps/epoch")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        perm = np.random.permutation(tr)[: steps_per_epoch * args.batch_size]
        batches = perm.reshape(steps_per_epoch, args.batch_size)
        run_loss = torch.zeros((), device=dev)
        for rgb_u8, h16, y, g_true, _ in prefetch(data, batches):
            lr_scale = step / max(warm, 1) if step < warm else \
                0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_scale

            rgb_a, h_a, v_a = to_gpu(rgb_u8, h16, dev)
            rgb_a, h_a, v_a = gpu_augment(rgb_a, h_a, v_a, out_hw, train=True)
            y_a = y.to(dev, non_blocking=True)
            g_a = g_true.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            for c in range(args.accum):  # gradient accumulation over micro-batches
                sl = slice(c * micro, (c + 1) * micro)
                rgb, h, v, y, g_true = rgb_a[sl], h_a[sl], v_a[sl], y_a[sl], g_a[sl]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out, grams = model(rgb, h, v)
                out = out.float()
                yn = (y - y_mean) / y_std
                loss = F.smooth_l1_loss(out, yn, beta=0.05)
                if args.consistency > 0:
                    p = denorm(out)
                    kcal_from_macros = (p[:, 2:5] * kcal_w).sum(1)
                    loss = loss + args.consistency * F.smooth_l1_loss(
                        p[:, 0] / y_std[0], kcal_from_macros / y_std[0], beta=0.05)
                if grams is not None:
                    loss = loss + args.ingr_w * F.smooth_l1_loss(
                        grams / 20.0, g_true / 20.0, beta=0.05, reduction="none").sum(1).mean()
                    derived = grams @ density
                    loss = loss + args.ingr_total_w * F.smooth_l1_loss(derived / y_std, y / y_std, beta=0.05)
                (loss / args.accum).backward()
                run_loss += loss.detach() / args.accum
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            step += 1

        train_loss = (run_loss / steps_per_epoch).item()
        vp, vidx = predict(Wrapped(ema.model), data, va, args, dev, tta=False)
        vm = metrics(blend(vp, 0.5), data.y[vidx])
        score = mean_pct(vm)
        dt = time.time() - t0
        flag = ""
        if score < best:
            best, best_epoch, flag = score, epoch, " *"
            torch.save({"model": ema.model.state_dict(), "y_mean": y_mean.cpu().numpy(),
                        "y_std": y_std.cpu().numpy(), "args": vars(args), "epoch": epoch},
                       run_dir / "best.pt")
        print(f"ep {epoch:3d}  loss {train_loss:.4f}  val {score:.2f}%  [{fmt(vm)}]  {dt:.0f}s{flag}", flush=True)
        log.write(f"{epoch},{train_loss:.5f},{score:.3f}," + ",".join(f"{vm[n]['pct']:.3f}" for n in NAMES)
                  + f",{dt:.1f}\n")
        log.flush()

    if args.train_on_val:  # no clean val -> keep the final EMA weights
        torch.save({"model": ema.model.state_dict(), "y_mean": y_mean.cpu().numpy(),
                    "y_std": y_std.cpu().numpy(), "args": vars(args), "epoch": args.epochs - 1},
                   run_dir / "best.pt")
        best_epoch = args.epochs - 1
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    print(f"best epoch {best_epoch} (val mean {best:.2f}%)")
    evaluate_and_report(Wrapped(model), data, va, te, args, dev, run_dir, best_epoch)


def evaluate_and_report(model, data, va, te, args, dev, run_dir, epoch):
    vp, vidx = predict(model, data, va, args, dev, tta=True)
    tp, tidx = predict(model, data, te, args, dev, tta=True)
    yv, yt = data.y[vidx], data.y[tidx]
    if not args.ingr:  # both slots hold the direct head
        alpha = np.ones(len(NAMES), np.float32)
    else:
        alpha = fit_alpha(vp, yv)  # chosen on VAL only
        print("blend alpha (weight of direct head, fit on val):", dict(zip(NAMES, alpha.round(1).tolist())))
    for nm, a in ((("direct", 1.0), ("from ingredients", 0.0), ("50/50", 0.5)) if args.ingr else ()):
        print(f"[{nm:>16}] val {mean_pct(metrics(blend(vp, a), yv)):.2f}%  "
              f"test {mean_pct(metrics(blend(tp, a), yt)):.2f}%")
    vp, tp = blend(vp, alpha), blend(tp, alpha)
    vm, tm = metrics(vp, yv), metrics(tp, yt)
    print(f"\nVAL  (TTA) mean {mean_pct(vm):.2f}%  {fmt(vm)}")
    print(f"TEST (TTA) mean {mean_pct(tm):.2f}%  {fmt(tm)}")
    print("\n           " + "".join(f"{n:>10}" for n in NAMES))
    print("paper RGBD " + "".join(f"{PAPER_RGBD_PCT[n]:>9.1f}%" for n in NAMES))
    print("IG-RGBD    " + "".join(f"{SOTA_RGBD_PCT[n]:>9.1f}%" for n in NAMES))
    print("ours       " + "".join(f"{tm[n]['pct']:>9.1f}%" for n in NAMES))
    np.savez(run_dir / "test_predictions.npz", dish_ids=data.ids[tidx], pred=tp, target=data.y[tidx])
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"epoch": epoch, "alpha": dict(zip(NAMES, alpha.tolist())), "val": vm, "test": tm, "val_mean_pct": mean_pct(vm),
                   "test_mean_pct": mean_pct(tm), "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()
