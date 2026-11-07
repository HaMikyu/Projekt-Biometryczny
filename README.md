# Projekt biometryczny – Nutrition5k

Michał Dąbrowski, Dominik Zabielski, Jakub Jaromski

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python download_nutrition5k_overhead.py
```

Skrypt dociaga brakujace `rgb.png` + `depth_raw.png` (bez filmow). Folder `data/` nie idzie na gita.

## Moduly

- `nutrition5k_lightning.py` – RGB -> makroskladniki
- `nutrition5k_depth_lightning.py` – RGB -> depth
- `train/` – DINOv3 (lokalnie) + trening depth / makro

```python
from nutrition5k_lightning import Nutrition5kDataModule
from nutrition5k_depth_lightning import Nutrition5kDepthDataModule
```

```bash
python train/train_depth.py
python train/train_macros.py --head mlp --task regression
python train/train_macros.py --head transformer --task classification --bin-size 10
python train/run_all.py
```

Najlepsze wagi (bez backbone) laduja w `train/best/`.
