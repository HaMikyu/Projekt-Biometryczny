import json
import os
import time
import urllib.parse
import urllib.request
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
    tmp = dest.with_suffix(dest.suffix + ".part")
    err = None
    for i in range(retries):
        try:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            urllib.request.urlretrieve(BASE + obj, tmp)
            os.replace(tmp, dest)
            return True
        except Exception as e:
            err = e
            time.sleep(1 + i)
    raise err


def main():
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
    print(f"Plikow w bazie: {total} (~{gb:.2f} GB)")
    print("Pobieram brakujace...")

    nowe = 0
    skip = 0
    for i, (obj, dest, _) in enumerate(files, 1):
        if dest.is_file() and dest.stat().st_size > 0:
            skip += 1
        else:
            download(obj, dest)
            nowe += 1
        if i % 100 == 0 or i == total:
            print(f"  {i}/{total} (nowe: {nowe}, skip: {skip})")

    dishes = [d for d in images.iterdir() if d.is_dir()] if images.is_dir() else []
    both = sum(
        1
        for d in dishes
        if (d / "rgb.png").is_file() and (d / "depth_raw.png").is_file()
    )
    print(f"Gotowe. dish z rgb+depth: {both}/{len(dishes)}")


if __name__ == "__main__":
    main()
