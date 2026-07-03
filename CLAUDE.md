# ROGII – Wellbore Geology Prediction (Kaggle)

> Project context for the AI coding agent. Read this before writing any code.
> The deliverable is a Kaggle submission notebook; code is developed in VS Code, then assembled into the notebook.

---

## 1. What this project is

Kaggle **code competition**. Build an ML model that predicts **TVT (True Vertical Thickness)** along a horizontal wellbore — i.e. automate *geosteering*.

- **TVT** = the well's interpreted vertical position within the geological layer stack, at each measured point along the well.
- **Task type:** regression.
- **Metric:** RMSE on TVT (lower = better).
- **LB orientation:** naive ≈ 20, baselines 12–15, strong public ≈ 9.2–9.5.
- **Prize:** $50K. Deadline ~mid-2026 (check the live countdown on the comp page).

---

## 2. Domain intuition (the "why")

A horizontal well is drilled "blind" through dipping rock layers. A **typewell** is a reference (near-vertical) well with known logs (gamma ray, etc.) vs depth. As the horizontal well is drilled, LWD logs (GR and others) are recorded along **measured depth (MD)**.

By aligning the horizontal well's GR signature against the typewell's known GR–depth curve, you recover where the bit currently sits stratigraphically → **TVT**. This is exactly what a geologist does by hand; we are automating it.

**Implication:** at its core this is a **curve-matching / stratigraphic-alignment** problem with ML layered on top — not a generic tabular regression.

---

## 3. Data

- `train/` and `test/` folders.
- Each well arrives as a **horizontal CSV + a typewell CSV**. These must be **paired** into one consistent per-well table before modeling.
- Horizontal logs: gamma ray + other curves along MD.
- Typewell: reference logs vs depth.
- Stratigraphic surfaces.
- `TVT_input`: a partial / known TVT supplied as input.
- **Target:** full TVT. The *prediction zone* is where `TVT_input` is not provided.
- Nearby-well location info (spatial signal is available).

**Before writing features: inspect the actual CSVs to lock exact column names, units, and dtypes.** Column names below are descriptive, not literal.

> **WARNING — train/test schema mismatch:** train and test do NOT share an identical column set. Keep **only columns present in both** (train ∩ test). Any model dependence on a test-absent column breaks inference.

---

## 4. CRITICAL CONSTRAINT — leakage boundary

This is the make-or-break of the whole competition.

- There is a **leakage boundary** between `TVT_input` and the target. In real geosteering you cannot know what lies **ahead / downhole** of the bit.
- In the prediction zone, features must **not** use future / downhole information. Anything that leaks across the boundary **inflates CV but tanks the LB**.
- **CV must respect this:**
  - `GroupKFold` by well (no well appears in both train and validation folds), **and**
  - within a well, respect the prediction zone — never let downhole rows inform an earlier point's prediction.
- **Validate that CV tracks LB before optimizing anything.** If it doesn't, everything downstream is blind.
- Reference: pilkwang's *"EDA + Leakage Risk Discussion"* notebook documents the boundary and column roles in detail.

---

## 5. Pipeline / plan

1. **Load & pair** — per well, merge horizontal + typewell into one table; keep train ∩ test columns only; clean missing values.
2. **Leakage-safe CV** — `GroupKFold` by well + prediction-zone-aware splits. Build this *first* and confirm CV ≈ LB.
3. **Feature engineering** (main effort, CPU-bound):
   - **Row-level:** GR + other logs → rolling mean/std/slope over MD windows, gradients, depth & distance signals.
   - **Alignment features (the domain edge):** DTW / cross-correlation between horizontal GR and typewell GR → a stratigraphic-position prior. High expected value — this is the geologist's manual step encoded as a feature.
   - **Spatial:** nearby-well signals.
4. **Models** — XGBoost / LightGBM / CatBoost, multi-seed → **hill-climbing blend** of OOF predictions.
5. **Postprocess** — per-well TVT **smoothing** + physical / continuity constraints. The target is smooth, so cleaning row-wise predictions lowers RMSE.
6. **Submit** — select on CV, submit sparingly.

---

## 6. Environment & submission

- **Code competition:** the submission is a **Kaggle Notebook** that writes `submission.csv` in the required format.
- **CPU-bound.** No GPU required — top approaches are GBM + signal/alignment. GPU only matters if a DL branch (LSTM / Temporal CNN on log sequences) is added, which is optional and not where the leaderboard currently is.
- **Internet is OFF at submission.** Any external dependency (extra package, pretrained weights) must be attached as a Kaggle Dataset.
- Reproducibility: fixed seeds, deterministic where feasible.

---

## 7. Tech stack

- Core: `python`, `pandas`, `numpy`, `scipy`
- Models: `lightgbm`, `xgboost`, `catboost`
- Alignment / signal: `scipy`, DTW via `dtaidistance` or `fastdtw`, optional `pywavelets` for wavelet features
- CV / metrics: `scikit-learn`

---

## 8. What I want from the agent

- Write **modular, notebook-ready** code: clear functions / cells — `data_loader`, `features`, `cv`, `train`, `postprocess`, `inference`.
- **Every feature must be leakage-safe by construction.** If a feature *could* cross the prediction-zone boundary, flag it explicitly rather than silently including it.
- Comment the non-obvious geosteering / alignment logic (assume the reader knows ML but not petroleum geology).
- Don't optimize the model before the leakage-safe CV is in place and shown to track the LB.
- Prefer clarity over cleverness.
