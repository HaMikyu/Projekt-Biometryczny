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

```python
from nutrition5k_lightning import Nutrition5kDataModule
from nutrition5k_depth_lightning import Nutrition5kDepthDataModule
```
