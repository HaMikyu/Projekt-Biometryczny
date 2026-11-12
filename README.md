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

## SOTA: RGB-D DINOv3 fine-tuning (`train/sota/`)

Nowy pipeline (czysty PyTorch, NVIDIA/CUDA, bf16):

- DINOv3 ViT-B/16 douczany w calosci (layer-wise LR decay, drop-path, EMA wag),
- glebia jako druga, zero-init patch-embedding (mapa wysokosci nad stolem w cm + maska) dodana do tokenow RGB,
- jawne cechy geometryczne (objetosc / pole powierzchni jedzenia) w glowicy,
- 5 wyjsc naraz: kcal, masa, tluszcz, weglowodany, bialko + strata spojnosci kcal ~ 9F + 4C + 4P,
- caly zbior w RAM, augmentacje (obrot, przesuniecie, odbicia, kolory) na GPU, TTA z 4 odbic,
- bez skalowania przy augmentacji (kamera stala => rozmiar na zdjeciu = porcja).

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python train/sota/build_cache.py      # raz, ~15 s
python train/sota/train.py            # ~50 s/epoka na RTX 3070 Ti
python train/sota/train.py --eval-only --ckpt train/runs/sota_vitb/best.pt
```

Wyniki (`results.json`, `test_predictions.npz`) laduja w `train/runs/<name>/`. Metryka: % MAE = MAE / srednia wartosc na zbiorze testowym (jak w pracy Nutrition5k).

### Wyniki (test, 507 dan, % MAE)

| model | kcal | masa | tluszcz | wegl. | bialko | srednia |
|---|---|---|---|---|---|---|
| Nutrition5k paper, RGB-D | 18.8 | 18.9 | 18.1 | 23.8 | 20.9 | 20.1 |
| Ingredient-Guided RGB-D (2024) | 12.2 | 9.4 | 19.1 | 18.3 | 16.0 | 15.0 |
| stare glowice (zamrozony ViT-S, 310 dan treningowych) | - | - | ~23-30 | ~30 | ~34 | - |
| **`train/sota` ViT-B RGB-D** (`sota_vitb`) | **14.5** | **10.0** | 22.3 | 21.1 | 22.0 | 18.0 |
| + glowica skladnikow (`--ingr`) | 15.2 | 10.7 | 24.2 | 21.7 | 21.9 | 18.8 |
| high-res 448x576 (`sota_vitb_hr`, `--cache-size 480x640 --input-size 448x576 --accum 2`) | 14.2 | 10.2 | 22.6 | 20.8 | 21.6 | 17.9 |
| **ensemble `sota_vitb` + `sota_vitb_hr`** (`python train/sota/ensemble.py sota_vitb sota_vitb_hr`) | **14.1** | **10.0** | 22.1 | 20.7 | 21.5 | **17.7** |

Uwaga: walidacja (259 dan) wychodzi ~4 pp lepiej niz test, wiec do strojenia jest optymistyczna.
