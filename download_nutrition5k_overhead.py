import argparse
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "nutrition5k"
API = "https://storage.googleapis.com/storage/v1/b/nutrition5k_dataset/o"
BASE = "https://storage.googleapis.com/nutrition5k_dataset/"
PREFIX = "nutrition5k_dataset/imagery/realsense_overhead/"
KEEP = {"rgb.png", "depth_raw.png"}

EXTRA = [
    "nutrition5k_dataset/dish_ids/splits/rgb_train_ids.txt",
    "nutrition5k_dataset/dish_ids/splits/rgb_test_ids.txt",
    "nutrition5k_dataset/metadata/dish_metadata_cafe1.csv",
    "nutrition5k_dataset/metadata/dish_metadata_cafe2.csv",
]


def list_files(prefix):
    token = None
    while True:
        q = {"prefix": prefix, "maxResults": "1000"}
        if token:
            q["pageToken"] = token
        url = API + "?" + urllib.parse.urlencode(q)
        with urllib.request.urlopen(url, timeout=120) as r:
            data = json.loads(r.read().decode())
        for item in data.get("items", []):
            yield item["name"], int(item["size"])
        token = data.get("nextPageToken")
        if not token:
            break


def download(obj, dest, retries=5):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return False
    tmp = dest.with_suffix(dest.suffix + f".{os.getpid()}.{threading.get_ident()}.part")
    err = None
    for i in range(retries):
        try:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            urllib.request.urlretrieve(BASE + obj, tmp)
            if dest.is_file() and dest.stat().st_size > 0:
                try:
                    tmp.unlink()
                except OSError:
                    pass
                return False
            if (not tmp.is_file()) or tmp.stat().st_size <= 0:
                raise IOError(f"empty download: {obj}")
            if dest.suffix.lower() == ".png":
                from PIL import Image

                with Image.open(tmp) as im:
                    im.verify()
                with Image.open(tmp) as im:
                    im.load()
            os.replace(tmp, dest)
            return True
        except Exception as e:
            err = e
            time.sleep(1 + i)
    raise err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    print("Pobieram metadata i splity (tylko brakujace)...")
    for obj in EXTRA:
        name = Path(obj).name
        if "metadata" in obj:
            dest = DATA_DIR / "metadata" / name
        else:
            dest = DATA_DIR / "dish_ids" / name
        nowy = download(obj, dest)
        print(f"  {name}: {'pobrano' if nowy else 'juz jest'}")

    images = DATA_DIR / "imagery" / "realsense_overhead"
    print("Listuje rgb.png + depth_raw.png...")
    files = []
    for name, size in list_files(PREFIX):
        if Path(name).name not in KEEP:
            continue
        rel = name[len(PREFIX) :]
        files.append((name, images / rel, size))

    total = len(files)
    gb = sum(s for _, _, s in files) / (1024 ** 3)
    todo = [(o, d, s) for o, d, s in files if not (d.is_file() and d.stat().st_size > 0)]
    skip = total - len(todo)
    print(f"Plikow w bazie: {total} (~{gb:.2f} GB)")
    print(f"Do pobrania: {len(todo)}, juz jest: {skip}")
    print(f"Pobieram rownolegle (workers={args.workers})...")

    nowe = 0
    done = 0
    lock = threading.Lock()
    errors = []

    def job(item):
        obj, dest, _ = item
        return download(obj, dest)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(job, item): item for item in todo}
        for fut in as_completed(futures):
            try:
                created = fut.result()
            except Exception as e:
                created = False
                errors.append((futures[fut][0], str(e)))
            with lock:
                done += 1
                if created:
                    nowe += 1
                if done % 100 == 0 or done == len(todo):
                    print(
                        f"  {done}/{len(todo)} (nowe: {nowe}, skip_start: {skip}, bledy: {len(errors)})"
                    )

    if errors:
        print(f"Bledy ({len(errors)}), przyklad:")
        for obj, msg in errors[:5]:
            print(f"  {obj}: {msg}")
        print("Odpal ponownie — dociagnie brakujace.")

    dishes = [d for d in images.iterdir() if d.is_dir()] if images.is_dir() else []
    both = sum(
        1
        for d in dishes
        if (d / "rgb.png").is_file() and (d / "depth_raw.png").is_file()
    )
    print(f"Gotowe. dish z rgb+depth: {both}/{len(dishes)}")


if __name__ == "__main__":
    main()
