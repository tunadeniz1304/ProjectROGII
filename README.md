<div align="center">

# 🛢️ ROGII — Wellbore Geology Prediction

**Automating geosteering: predicting a horizontal well's stratigraphic position (TVT) with leakage-safe machine learning.**

[![Kaggle](https://img.shields.io/badge/Kaggle-Code%20Competition-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-GBM-FFCC00)](https://catboost.ai/)
[![LightGBM](https://img.shields.io/badge/LightGBM-GBM-9ACD32)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-see%20LICENSE-lightgrey)](LICENSE)

*Metric: **RMSE** on TVT · Task: **regression** · Compute: **CPU-bound** (GPU optional)*

</div>

---

## 📖 Table of Contents

- [What is this?](#-what-is-this)
- [The domain in one picture](#-the-domain-in-one-picture)
- [The one constraint that decides everything: leakage](#-the-one-constraint-that-decides-everything-leakage)
- [Key insight](#-key-insight)
- [Repository layout](#-repository-layout)
- [The notebooks](#-the-notebooks)
- [The `src/` library](#-the-src-library)
- [Quickstart](#-quickstart)
- [Approach & roadmap](#-approach--roadmap)
- [Data](#-data)
- [Tech stack](#-tech-stack)

---

## 🎯 What is this?

A solution for the **ROGII Wellbore Geology Prediction** Kaggle *code competition*. The task: build an ML model that predicts **TVT (True Vertical Thickness)** — the well's interpreted vertical position within the geological layer stack — at each measured point along a **horizontal wellbore**.

In plain terms: we're **automating geosteering**, the manual process a geologist performs to figure out where the drill bit currently sits inside the rock layers.

| Orientation | RMSE (lower = better) |
|---|---|
| Naive baseline | ≈ 20 |
| Simple baselines | 12 – 15 |
| Strong public leaderboard | ≈ 9.2 – 9.5 |
| Leaderboard leaders | ≈ 5.7 – 7.3 |

---

## 🧭 The domain in one picture

A horizontal well is drilled **blind** through dipping rock layers. To locate the bit:

- A **typewell** is a reference (near-vertical) well with known logs (gamma ray *vs* depth).
- As the horizontal well is drilled, **LWD logs** (GR and others) are recorded along **measured depth (MD)**.
- By aligning the horizontal well's signature against the typewell's known curves, you recover where the bit sits stratigraphically → **TVT**.

At its core this is a **stratigraphic-alignment / curve-matching** problem with ML on top — *not* a generic tabular regression.

---

## 🚨 The one constraint that decides everything: leakage

> **This is the make-or-break of the whole competition.**

Each well has a **Prediction Start (PS)** point — a leakage boundary. `TVT_input` is **known for the heel** (the prefix) and **NaN for the toe** (the *prediction zone*, which is what we score).

In real geosteering **you cannot know what lies ahead / downhole of the bit.** So:

- ✅ **Every feature is leakage-safe by construction** — it uses only inputs legal in the prediction zone: `MD, X, Y, Z, GR` (known everywhere) and **pre-PS** `TVT_input` (the known heel).
- ❌ It **never** touches downhole `TVT` or train-only surface / geology columns.
- ✅ **CV respects the boundary**: `GroupKFold` by well (no well in both train and validation) **and** prediction-zone-aware splits (a downhole row never informs an earlier point's prediction).
- 📏 **CV must track the LB** before optimizing anything — otherwise every downstream decision is blind.

Anything that crosses the boundary **inflates CV but tanks the leaderboard.**

---

## 💡 Key insight

Documented and empirically validated on the 773-well training set (see [`ROGII_BATTLE_PLAN.md`](ROGII_BATTLE_PLAN.md)):

> **The dominant signal is the well's own Z (vertical) trajectory — not GR curve matching.**

| Method (773 wells, drift target) | RMSE |
|---|---|
| Carry-forward (drift = 0) | 15.91 |
| Global `c·dz` | 15.79 *(useless)* |
| **Per-well oracle line `dtvt = a·dz + b`** | **6.80** |

Globally `corr(drift, −dz) ≈ 0.14` (weak) because each well's **dip slope `a` differs**. But *within* a well the relationship is near-linear. So the whole game is: **estimate each well's local dip `a` and offset `b`, leakage-safely.** The spatial geology surface `formation(X, Y)` gives a principled estimate of both.

**Modeling trick used throughout:** predict the **drift from carry-forward** rather than TVT directly —
`y = TVT − TVT_ps` → `pred = TVT_ps + model`.

---

## 🗂️ Repository layout

```
ProjectROGII/
├── README.md                                   ← you are here
├── CLAUDE.md                                   project brief / agent context
├── ROGII_BATTLE_PLAN.md                        staged plan to climb the leaderboard (TR)
├── GECE_RAPORU.md                              running status / handoff log (TR)
├── LICENSE
│
├── submission_v1.ipynb                         CatBoost drift baseline (self-contained)
├── stage1_zprior.ipynb                         leakage-safe Z-prior + geology-plane model
├── 9-251-...-dwt-based.ipynb                   best public 9.251 DWT baseline (reference)
├── 9-251-plus-tabpfn3.ipynb                    9.251 + TabPFN-3 as an extra blend member
│
└── src/                                        notebook-ready Python library
    ├── pipeline.py                             v1 CatBoost drift pipeline
    ├── stage1_zprior.py                        Stage-1 Z-prior + FormationPlaneKNN (LightGBM)
    ├── make_notebook.py                        builds submission_v1.ipynb
    └── make_tabpfn_notebook.py                 builds 9-251-plus-tabpfn3.ipynb
```

> **Note:** competition data, `submission.csv`, `artifacts/`, `*.parquet`, and `catboost_info/` are git-ignored. Data lives **outside** the repo (never committed).

---

## 📓 The notebooks

Each notebook is **self-contained and Kaggle-ready** (path-aware, internet-off, fixed seeds) and writes `submission.csv`.

| Notebook | What it does | Approx. LB |
|---|---|---|
| **`submission_v1.ipynb`** | CatBoost predicting `ΔTVT` drift with leakage-safe GR / geometry features + GroupKFold CV. | ~14 |
| **`stage1_zprior.ipynb`** | Leakage-safe **Z-prior** + spatial **geology-plane** implied drift (`FormationPlaneKNN`), LightGBM + per-well Savitzky-Golay smoothing. | — |
| **`9-251-...-dwt-based.ipynb`** | The strongest **public** baseline (DWT / alignment-based, 6-GBM blend + hill-climbing). Kept as a reference / retrain target. | ~9.25 |
| **`9-251-plus-tabpfn3.ipynb`** | The 9.251 notebook with **TabPFN-3** grafted in as an *extra* hill-climbing member — same target, same features, same split, fully guarded so a missing package never regresses the 9.25 blend. | ~9.25+ |

---

## 🐍 The `src/` library

Modular, notebook-ready functions — each maps cleanly onto a notebook cell (`data_loader → features → cv → train → inference`).

### `pipeline.py` — v1 CatBoost drift pipeline
- `list_wells` / `load_well` / `load_typewell` / `find_ps` — data loading + PS-boundary detection.
- `well_features` — leakage-safe GR rolling stats, geometry (`dz`, displacement, inclination), boundary dip slope. Predicts the **drift** target.
- `build_matrix` — assembles the full per-row matrix (cached to parquet).
- `group_kfold_oof` / `report` — `GroupKFold`-by-well OOF training with CatBoost + a CV report vs the carry-forward baseline.
- `make_submission` — fold-averaged test predictions aligned to `sample_submission`.

### `stage1_zprior.py` — Stage-1 Z-prior + geology surface
- `FormationPlaneKNN` — **local-plane KNN** estimate of each formation surface at a query `(X, Y)`, built from other training wells (self-excluded → no leakage). Adapted from the proven public component.
- `well_features` — geometry + GR features **plus** geology-plane implied drift `TVT = formation(X, Y) − Z + b_well`.
- `main` — LightGBM `GroupKFold` CV + per-well smoothing + submission.
- Robust `find_competition_dir()` auto-detects the data path on Kaggle or locally (`ROGII_DATA` env override supported).

### Notebook builders
- `make_notebook.py` → emits `submission_v1.ipynb`.
- `make_tabpfn_notebook.py` → **surgically** inserts guarded TabPFN-3 cells into the 9.251 notebook (not a rewrite) → emits `9-251-plus-tabpfn3.ipynb`.

---

## ⚡ Quickstart

**Prerequisites:** Python 3.10+, and the competition data placed where the scripts expect it (default `C:\KaggleData\rogii\...`, or set `ROGII_DATA`).

```bash
# install core deps
pip install numpy pandas scipy scikit-learn lightgbm catboost

# point the loader at your data (optional; defaults are auto-detected)
export ROGII_DATA="/path/to/rogii-wellbore-geology-prediction"

# run the v1 CatBoost pipeline end-to-end (build matrix → CV → submission.csv)
python src/pipeline.py

# run the Stage-1 Z-prior + geology-plane model
python src/stage1_zprior.py

# regenerate the Kaggle notebooks from source
python src/make_notebook.py          # -> submission_v1.ipynb
python src/make_tabpfn_notebook.py   # -> 9-251-plus-tabpfn3.ipynb
```

Each run prints a **CV report** comparing the model's pooled RMSE against the carry-forward baseline, then writes `submission.csv`.

---

## 🧱 Approach & roadmap

The staged plan (full detail in [`ROGII_BATTLE_PLAN.md`](ROGII_BATTLE_PLAN.md)) — **exploit the dominant signal first, add deep learning for the residual:**

| Stage | Focus | Target RMSE |
|---|---|---|
| **0** | Working pipeline + honest, leakage-safe CV that tracks the LB | ~9.25 |
| **1** ⚡ | Geometric **Z-prior** (per-well dip `a` + offset `b`) — highest ROI | ~8 |
| **2** | Spatial **geology surface** (`FormationPlaneKNN`) + blend | ~7.3 |
| **3** | Learned **offset selector** classifier | ~6.8 |
| **4** 🎯 | Deep **Multi-Trajectory Prediction** (CNN/U-Net + mixture density) | ~6.2 |
| **5** | **Ensemble** (hill-climbing) + post-process + robust submit selection | ~5.9 |

Every stage: measure CV **and** LB, keep only what improves.

---

## 📦 Data

- Per well: a **horizontal CSV** + a **typewell CSV**, paired into one consistent table before modeling.
- Horizontal logs: gamma ray + other curves along MD; typewell: reference logs *vs* depth; plus stratigraphic surfaces.
- `TVT_input`: partial/known TVT supplied as input; **target** is the full TVT (prediction zone = where `TVT_input` is missing).
- ⚠️ **Train and test do *not* share an identical column set** — keep only columns present in **both** (`train ∩ test`). Any model dependence on a test-absent column breaks inference.

---

## 🛠️ Tech stack

`python` · `pandas` · `numpy` · `scipy` — core
`lightgbm` · `xgboost` · `catboost` — models
`scikit-learn` — CV / metrics
`scipy` (DTW / cross-correlation, Savitzky-Golay) — signal & alignment
`tabpfn` (optional) — extra blend member

---

<div align="center">

*Built for a Kaggle code competition. Internet is **OFF** at submission — every dependency must be a self-contained notebook or an attached Kaggle Dataset.*

</div>
