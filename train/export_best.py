from __future__ import annotations

from pathlib import Path

import torch


def export_best_weights(ckpt_path: str | Path, out_path: str | Path) -> Path:
    ckpt_path = Path(ckpt_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    keep = {
        k: v
        for k, v in state.items()
        if not k.startswith("backbone.")
    }
    payload = {
        "state_dict": keep,
        "hyper_parameters": blob.get("hyper_parameters") if isinstance(blob, dict) else None,
        "source_ckpt": str(ckpt_path),
    }
    torch.save(payload, out_path)
    return out_path
