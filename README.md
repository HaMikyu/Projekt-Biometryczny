# Projekt biometryczny – Nutrition5k

Michał Dąbrowski, Dominik Zabielski, Jakub Jaromski

Szacowanie kalorii, masy i makroskladnikow (tluszcz, weglowodany, bialko) dania
ze zdjecia RGB-D z gory (Nutrition5k, kamera `realsense_overhead`).

## Struktura

```
nutrition5k/          wspolny kod danych
  helpers.py            metadane, splity, wczytywanie obrazow
  rgb_datamodule.py     RGB -> makro (Lightning)
  depth_datamodule.py   RGB -> depth (Lightning)
  rgbd_macros_datamodule.py  RGB + depth -> makro (Lightning, uzywany przez baseline)
baseline/             zamrozony DINOv3 ViT-S + glowice MLP / CNN / Transformer
  best/                 wyeksportowane wagi glowic
  dashboard/            podglad treningu na zywo
  evaluate.py           ocena wszystkich glowic na pelnym tescie
sota/                 fine-tuning DINOv3 ViT-B z fuzja RGB-D (najlepszy model)
scripts/              pobieranie danych, instalacja torch pod AMD
docs/                 sklad grupy, tematy
data/                 dane Nutrition5k (poza gitem)
runs/                 checkpointy, logi, wyniki treningow (poza gitem)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_nutrition5k_overhead.py
```

Skrypt dociaga brakujace `rgb.png` + `depth_raw.png` (bez filmow), ~3.5k dan.

GPU:
- NVIDIA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- AMD (ROCm, Windows): `powershell scripts/install_amd_torch.ps1`

## SOTA: RGB-D DINOv3 ViT-B (`sota/`)

- DINOv3 ViT-B/16 douczany w calosci (layer-wise LR decay, drop-path, EMA wag),
- glebia jako druga, zero-init patch-embedding (mapa wysokosci nad stolem w cm + maska) dodana do tokenow RGB,
- jawne cechy geometryczne (objetosc / pole powierzchni jedzenia) w glowicy,
- 5 wyjsc naraz: kcal, masa, tluszcz, weglowodany, bialko + strata spojnosci kcal ~ 9F + 4C + 4P,
- caly zbior w RAM, augmentacje (obrot, przesuniecie, odbicia, kolory) na GPU, bf16, TTA z 4 odbic,
- bez skalowania przy augmentacji (kamera stala => rozmiar na zdjeciu = porcja),
- opcjonalna glowica gramow 256 skladnikow (`--ingr`).

```bash
python sota/build_cache.py                       # raz, ~15 s
python sota/train.py                             # ~45 s/epoka, 40 epok ~30 min (RTX 3070 Ti)
                                                 # (sota_vitb z tabeli: --name sota_vitb --head-hidden 512)
python sota/train.py --eval-only --ckpt runs/sota_vitb/best.pt

# high-res (natywne 480x640, ~2 h)
python sota/build_cache.py --height 480 --width 640
python sota/train.py --name sota_vitb_hr --cache-size 480x640 --input-size 448x576 --accum 2

python sota/ensemble.py sota_vitb sota_vitb_hr  # usrednienie predykcji testowych
```

Wyniki (`best.pt`, `log.csv`, `results.json`, `test_predictions.npz`) laduja w `runs/<name>/`.

## Baseline (`baseline/`)

Zamrozony DINOv3 ViT-S/16 (224x224) + male glowice na CLS / tokenach + enkoder glebi.

```bash
python baseline/train_depth.py
python baseline/train_macros.py --head mlp --task regression
python baseline/train_macros.py --head transformer --task classification --bin-size 10
python baseline/run_all.py
python baseline/evaluate.py
```

## Wyniki (test, 507 dan, % MAE)

% MAE = MAE / srednia wartosc na zbiorze testowym (jak w pracy Nutrition5k).

| model | kcal | masa | tluszcz | wegl. | bialko | sr. makro |
|---|---|---|---|---|---|---|
| SOTA: Ingredient-Guided RGB-D (2024) | 12.2 | 9.4 | 19.1 | 18.3 | 16.0 | 17.8 |
| Nutrition5k paper, RGB-D (2021) | 18.8 | 18.9 | 18.1 | 23.8 | 20.9 | 20.9 |
| **`sota` ViT-B RGB-D** (`sota_vitb`, ~30 min) | **14.5** | **10.0** | 22.3 | 21.1 | 22.0 | 21.8 |
| `sota` + glowica skladnikow (`--ingr`) | 15.2 | 10.7 | 24.2 | 21.7 | 21.9 | 22.6 |
| `sota` high-res 448x576 (`sota_vitb_hr`) | 14.2 | 10.2 | 22.6 | 20.8 | 21.6 | 21.7 |
| **ensemble `sota_vitb` + `sota_vitb_hr`** | **14.1** | **10.0** | 22.1 | 20.7 | 21.5 | **21.4** |
| baseline: CNN, regresja (najlepszy) | 21.3* | - | 30.5 | 29.5 | 29.9 | 30.0 |
| baseline: MLP, regresja | 24.9* | - | 34.1 | 30.7 | 32.7 | 32.5 |
| baseline: Transformer, regresja | 25.4* | - | 35.1 | 32.8 | 32.7 | 33.5 |
| baseline: MLP, klasyfikacja bin 5 | 25.7* | - | 34.4 | 30.0 | 34.8 | 33.1 |

\* baseline nie przewiduje kcal ani masy; kcal wyliczone jako 9F + 4C + 4P.
Baseline byl trenowany na ok. 310 daniach (tyle bylo lokalnie), `sota` na pelnym zbiorze (2494 dan).
Liczby SOTA z literatury sa przyblizone (preprinty / zrodla wtorne, mozliwe inne splity).
Walidacja (259 dan) wychodzi ~4 pp lepiej niz test, wiec do strojenia jest optymistyczna.
