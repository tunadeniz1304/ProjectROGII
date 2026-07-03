"""Generate the self-contained Kaggle submission notebook (submission_v1.ipynb).

The notebook is path-aware (Kaggle /kaggle/input/... vs local), trains a CatBoost
model that predicts ΔTVT (drift from carry-forward) with leakage-safe features,
prints GroupKFold CV, and writes submission.csv. No internet / external deps.
"""
import json, os

CELLS = []

def md(src):
    CELLS.append({"cell_type": "markdown", "id": f"c{len(CELLS)}", "metadata": {},
                  "source": src.splitlines(keepends=True)})

def code(src):
    CELLS.append({"cell_type": "code", "id": f"c{len(CELLS)}", "metadata": {},
                  "execution_count": None, "outputs": [],
                  "source": src.strip("\n").splitlines(keepends=True)})

md("""# ROGII Wellbore Geology Prediction — CatBoost v1

Predicts **TVT** beyond the Prediction Start (PS) point for each horizontal well.

**Approach (leakage-safe):**
- The PS point splits each well: `TVT_input` is known for the heel, NaN for the toe (= what we score).
- Carry-forward (hold last known TVT) is the strong naive baseline (~16 pooled RMSE).
- We predict the **drift** `ΔTVT = TVT − TVT_ps` with CatBoost, then `pred = TVT_ps + ΔTVT̂`.
- Inputs are only the columns present in **both** train & test: `MD, X, Y, Z, GR, TVT_input(pre-PS)` + typewell `TVT, GR`.
- Validation = **GroupKFold by well** (no well in both train/val) — must track the LB.
""")

code('''
import os, glob, time
import numpy as np
import pandas as pd
from pathlib import Path

# ---- config + robust path detection (Kaggle competitions/, datasets/, local) ----
class CFG:
    seed = 42
    n_splits = 5

def find_competition_dir():
    """Locate the competition data across every layout it can mount in."""
    cands = [
        "/kaggle/input/competitions/rogii-wellbore-geology-prediction",  # reference layout
        "/kaggle/input/rogii-wellbore-geology-prediction",               # standard mount
        r"C:\\KaggleData\\rogii\\rogii-wellbore-geology-prediction",       # local
    ]
    for c in cands:
        if os.path.isdir(os.path.join(c, "train")):
            return c
    # recursive fallback: find any horizontal-well csv and back out the root
    for base in ("/kaggle/input", r"C:\\KaggleData"):
        hits = glob.glob(os.path.join(base, "**", "*__horizontal_well.csv"), recursive=True)
        if hits:
            d = os.path.dirname(hits[0])               # .../train or .../test
            return os.path.dirname(d) if os.path.basename(d) in ("train", "test") else d
    print("!! data not found. Contents of /kaggle/input:")
    for p in sorted(glob.glob("/kaggle/input/**", recursive=True))[:80]:
        print("   ", p)
    raise FileNotFoundError(
        "competition data not found - in the Kaggle editor click '+ Add Input' "
        "and add the competition 'rogii-wellbore-geology-prediction'.")

CFG.dataset_path = Path(find_competition_dir())
DATA = str(CFG.dataset_path)
print("DATA =", DATA)
HZ_TRAIN = ["MD","X","Y","Z","GR","TVT_input","TVT"]
HZ_TEST  = ["MD","X","Y","Z","GR","TVT_input"]
''')

code('''
# ---- data_loader ----------------------------------------------------------
def list_wells(split):
    return sorted(os.path.basename(p)[:8]
                  for p in glob.glob(os.path.join(DATA, split, "*__horizontal_well.csv")))

def load_well(wid, split):
    cols = HZ_TRAIN if split == "train" else HZ_TEST
    return pd.read_csv(os.path.join(DATA, split, f"{wid}__horizontal_well.csv"),
                       usecols=lambda c: c in cols)

def load_typewell(wid, split):
    return pd.read_csv(os.path.join(DATA, split, f"{wid}__typewell.csv"),
                       usecols=lambda c: c in ("TVT","GR"))

def find_ps(df):
    return int(np.where(df["TVT_input"].notna().values)[0][-1])
''')

code('''
# ---- features (leakage-safe; identical for train & test) ------------------
def _roll(a, w, fn):
    return getattr(pd.Series(a).rolling(w, min_periods=1), fn)().values

def well_features(wid, split):
    df = load_well(wid, split); tw = load_typewell(wid, split)
    n = len(df); ps = find_ps(df); n_known = ps + 1
    md = df["MD"].values.astype(float); x = df["X"].values.astype(float)
    y = df["Y"].values.astype(float);  z = df["Z"].values.astype(float)
    gr = df["GR"].values.astype(float); tin = df["TVT_input"].values.astype(float)
    md_ps, x_ps, y_ps, z_ps, tvt_ps = md[ps], x[ps], y[ps], z[ps], tin[ps]

    K = min(300, n_known)
    sl  = np.polyfit(md[ps-K+1:ps+1], tin[ps-K+1:ps+1], 1)[0]
    zsl = np.polyfit(md[ps-K+1:ps+1], z[ps-K+1:ps+1], 1)[0]
    gr_m = _roll(gr,25,"mean"); gr_sd = _roll(gr,25,"std"); gr_m5 = _roll(gr,5,"mean")
    gr_known_mean = np.nanmean(gr[:n_known]); gr_known_std = np.nanstd(gr[:n_known])
    tw_gr_mean, tw_gr_std = tw["GR"].mean(), tw["GR"].std()
    tw_span = tw["TVT"].max() - tw["TVT"].min()

    pz = np.arange(ps+1, n)
    d_md = md[pz]-md_ps; dz = z[pz]-z_ps; dx = x[pz]-x_ps; dy = y[pz]-y_ps
    disp = np.sqrt(dx*dx+dy*dy)
    feat = pd.DataFrame({
        "d_md": d_md, "dz": dz, "disp": disp, "dx": dx, "dy": dy,
        "inc_ratio": dz/np.where(d_md==0, np.nan, d_md),
        "gr": gr[pz], "gr_m25": gr_m[pz], "gr_sd25": gr_sd[pz], "gr_m5": gr_m5[pz],
        "gr_dev_known": gr[pz]-gr_known_mean,
        "gr_z_known": (gr[pz]-gr_known_mean)/(gr_known_std+1e-6),
        "gr_dev_tw": gr[pz]-tw_gr_mean,
        "bnd_slope": sl, "z_slope": zsl, "lin_delta": sl*d_md, "z_pred_delta": dz,
        "z_ps": z_ps, "md_ps": md_ps, "n_known": float(n_known),
        "tw_gr_mean": tw_gr_mean, "tw_gr_std": tw_gr_std, "tw_span": tw_span,
        "frac_into_pz": np.arange(len(pz))/max(1, len(pz)-1),
    }).astype(np.float32)
    ids = [f"{wid}_{i}" for i in pz]
    target = (df["TVT"].values[pz]-tvt_ps).astype(np.float32) if split=="train" else None
    feat["_tvt_ps"] = np.float32(tvt_ps)
    return feat, target, ids

def build_matrix(split):
    feats = []; t0 = time.time()
    wells = list_wells(split)
    for i, wid in enumerate(wells):
        f, t, ids = well_features(wid, split)
        f["_well"] = wid; f["_id"] = ids
        if t is not None: f["_y"] = t
        feats.append(f)
        if (i+1) % 200 == 0: print(f"  [{split}] {i+1}/{len(wells)} ({time.time()-t0:.0f}s)")
    m = pd.concat(feats, ignore_index=True)
    print(f"  [{split}] {len(m):,} rows / {len(wells)} wells in {time.time()-t0:.0f}s")
    fc = [c for c in m.columns if not c.startswith("_")]
    return m[fc], (m["_y"].values if "_y" in m else None), m["_well"].values, m["_id"].tolist(), m["_tvt_ps"].values
''')

code('''
# ---- train: GroupKFold CV + ensemble -------------------------------------
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold

PARAMS = dict(loss_function="RMSE", eval_metric="RMSE", iterations=1500,
              learning_rate=0.05, depth=8, l2_leaf_reg=5.0,
              random_seed=42, thread_count=-1, verbose=0, early_stopping_rounds=100)

Xtr, ytr, gtr, _, tps_tr = build_matrix("train")
feat_cols = [c for c in Xtr.columns if c != "_tvt_ps"]

oof = np.zeros(len(Xtr)); models = []
for fold,(tr,va) in enumerate(GroupKFold(5).split(Xtr, ytr, gtr)):
    m = CatBoostRegressor(**PARAMS)
    m.fit(Xtr.iloc[tr][feat_cols], ytr[tr],
          eval_set=(Xtr.iloc[va][feat_cols], ytr[va]), use_best_model=True)
    oof[va] = m.predict(Xtr.iloc[va][feat_cols]); models.append(m)
    r = np.sqrt(np.mean((tps_tr[va]+oof[va] - (tps_tr[va]+ytr[va]))**2))
    print(f"fold {fold}: best_iter={m.get_best_iteration()} pooled RMSE={r:.4f}")

true = tps_tr + ytr
carry  = np.sqrt(np.mean((tps_tr - true)**2))
pooled = np.sqrt(np.mean((tps_tr+oof - true)**2))
pw = pd.DataFrame({"w":gtr,"e2":(tps_tr+oof-true)**2}).groupby("w")["e2"].mean().pow(0.5)
print(f"\\nCARRY pooled RMSE = {carry:.4f}")
print(f"MODEL pooled RMSE = {pooled:.4f}  (gain {carry-pooled:+.4f})")
print(f"per-well RMSE mean={pw.mean():.3f} median={pw.median():.3f} p90={pw.quantile(.9):.3f} max={pw.max():.3f}")
''')

code('''
# ---- inference: write submission.csv -------------------------------------
Xte, _, gte, ids_te, tps_te = build_matrix("test")
pred = np.mean([m.predict(Xte[feat_cols]) for m in models], axis=0)
sub = pd.DataFrame({"id": ids_te, "tvt": tps_te + pred})

ss = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
sub = ss[["id"]].merge(sub, on="id", how="left")
assert sub["tvt"].notna().all(), "missing ids vs sample_submission!"
assert len(sub) == len(ss)
sub.to_csv("submission.csv", index=False)
print("submission.csv", sub.shape, "tvt[min/mean/max]=",
      round(sub.tvt.min(),1), round(sub.tvt.mean(),1), round(sub.tvt.max(),1))
sub.head()
''')

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

out = os.path.join(os.path.dirname(__file__), "..", "submission_v1.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", os.path.abspath(out), "cells:", len(CELLS))
