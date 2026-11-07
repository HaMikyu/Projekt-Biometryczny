from __future__ import annotations

import csv
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

TRAIN_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = TRAIN_DIR / "runs"
BEST_DIR = TRAIN_DIR / "best"
STATIC_DIR = Path(__file__).resolve().parent
PORT = 8765

JOBS = [
    {
        "id": 1,
        "key": "depth_head-cnn_decoder",
        "title": "Depth",
        "subtitle": "RGB → depth · CNN decoder",
        "kind": "depth",
        "max_epochs": 15,
    },
    {
        "id": 2,
        "key": "macros_head-mlp_task-regression_bin-na",
        "title": "Makro · MLP · regresja",
        "subtitle": "RGB+depth → fat/carb/protein",
        "kind": "macros",
        "max_epochs": 20,
    },
    {
        "id": 3,
        "key": "macros_head-transformer_task-regression_bin-na",
        "title": "Makro · Transformer · regresja",
        "subtitle": "RGB+depth → fat/carb/protein",
        "kind": "macros",
        "max_epochs": 20,
    },
    {
        "id": 4,
        "key": "macros_head-cnn_task-regression_bin-na",
        "title": "Makro · CNN · regresja",
        "subtitle": "RGB+depth → fat/carb/protein",
        "kind": "macros",
        "max_epochs": 20,
    },
    {
        "id": 5,
        "key": "macros_head-mlp_task-classification_bin-5",
        "title": "Makro · MLP · klasyfikacja · bin 5",
        "subtitle": "CE loss · bin 5 g",
        "kind": "macros",
        "max_epochs": 20,
    },
    {
        "id": 6,
        "key": "macros_head-mlp_task-classification_bin-10",
        "title": "Makro · MLP · klasyfikacja · bin 10",
        "subtitle": "CE loss · bin 10 g",
        "kind": "macros",
        "max_epochs": 20,
    },
    {
        "id": 7,
        "key": "macros_head-transformer_task-classification_bin-5",
        "title": "Makro · Transformer · klasyfikacja · bin 5",
        "subtitle": "CE loss · bin 5 g",
        "kind": "macros",
        "max_epochs": 20,
    },
    {
        "id": 8,
        "key": "macros_head-transformer_task-classification_bin-10",
        "title": "Makro · Transformer · klasyfikacja · bin 10",
        "subtitle": "CE loss · bin 10 g",
        "kind": "macros",
        "max_epochs": 20,
    },
    {
        "id": 9,
        "key": "macros_head-cnn_task-classification_bin-5",
        "title": "Makro · CNN · klasyfikacja · bin 5",
        "subtitle": "CE loss · bin 5 g",
        "kind": "macros",
        "max_epochs": 20,
    },
    {
        "id": 10,
        "key": "macros_head-cnn_task-classification_bin-10",
        "title": "Makro · CNN · klasyfikacja · bin 10",
        "subtitle": "CE loss · bin 10 g",
        "kind": "macros",
        "max_epochs": 20,
    },
]


def latest_metrics_path(run_key: str) -> Path | None:
    base = RUNS_DIR / run_key / "lightning_logs"
    if not base.is_dir():
        return None
    versions = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.startswith("version_")],
        key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1,
    )
    for v in reversed(versions):
        m = v / "metrics.csv"
        if m.is_file() and m.stat().st_size > 0:
            return m
    return None


def parse_metrics(path: Path) -> dict:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    def fnum(key):
        out = []
        for r in rows:
            raw = (r.get(key) or "").strip()
            if raw == "":
                out.append(None)
            else:
                try:
                    out.append(float(raw))
                except ValueError:
                    out.append(None)
        return out

    epochs = []
    for r in rows:
        raw = (r.get("epoch") or "").strip()
        try:
            epochs.append(int(float(raw)) if raw else None)
        except ValueError:
            epochs.append(None)

    series = {
        "epoch": epochs,
        "train_loss": fnum("train_loss"),
        "train_mae": fnum("train_mae"),
        "val_loss": fnum("val_loss"),
        "val_mae": fnum("val_mae"),
        "val_mae_fat": fnum("val_mae_fat"),
        "val_mae_carb": fnum("val_mae_carb"),
        "val_mae_protein": fnum("val_mae_protein"),
    }

    last_val_mae = None
    last_train_mae = None
    last_epoch = None
    best_val_mae = None
    for i, e in enumerate(epochs):
        if e is not None:
            last_epoch = e
        v = series["val_mae"][i]
        t = series["train_mae"][i]
        if v is not None:
            last_val_mae = v
            best_val_mae = v if best_val_mae is None else min(best_val_mae, v)
        if t is not None:
            last_train_mae = t

    return {
        "path": str(path),
        "mtime": path.stat().st_mtime,
        "n_rows": len(rows),
        "last_epoch": last_epoch,
        "last_val_mae": last_val_mae,
        "last_train_mae": last_train_mae,
        "best_val_mae": best_val_mae,
        "series": series,
    }


def build_status() -> dict:
    now = time.time()
    jobs = []
    for job in JOBS:
        key = job["key"]
        best_pt = BEST_DIR / f"{key}.pt"
        metrics_path = latest_metrics_path(key)
        metrics = parse_metrics(metrics_path) if metrics_path else None
        age = (now - metrics["mtime"]) if metrics else None

        if best_pt.is_file() and best_pt.stat().st_size > 0:
            status = "done"
        elif metrics and age is not None and age < 120:
            status = "training"
        elif metrics:
            status = "stalled"
        else:
            status = "queued"

        jobs.append(
            {
                **job,
                "status": status,
                "best_pt": str(best_pt) if best_pt.is_file() else None,
                "best_size_mb": round(best_pt.stat().st_size / (1024 * 1024), 2)
                if best_pt.is_file()
                else None,
                "metrics": metrics,
                "progress": None
                if not metrics or metrics["last_epoch"] is None
                else round(
                    min(1.0, (metrics["last_epoch"] + 1) / max(job["max_epochs"], 1)),
                    3,
                ),
            }
        )

    active = [j for j in jobs if j["status"] == "training"]
    if not active:
        for j in jobs:
            if j["status"] == "queued":
                prev_done = all(
                    x["status"] in ("done", "stalled") for x in jobs if x["id"] < j["id"]
                )
                if prev_done or j["id"] == 1:
                    break

    done_n = sum(1 for j in jobs if j["status"] == "done")
    return {
        "server_time": now,
        "done": done_n,
        "total": len(jobs),
        "training": sum(1 for j in jobs if j["status"] == "training"),
        "queued": sum(1 for j in jobs if j["status"] == "queued"),
        "jobs": jobs,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _send(self, code: int, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            html = (STATIC_DIR / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
            return
        if path == "/api/status":
            payload = json.dumps(build_status()).encode("utf-8")
            self._send(200, payload, "application/json; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"dashboard: http://127.0.0.1:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
