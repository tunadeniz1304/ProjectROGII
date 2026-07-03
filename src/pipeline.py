"""
ROGII Wellbore Geology Prediction — v1 baseline pipeline.

Notebook-ready modular functions (each maps to a notebook cell):
    data_loader  -> list_wells / load_well / find_ps
    features     -> well_features
    cv           -> build_matrix / group_kfold_oof
    train        -> train_lgbm
    inference    -> make_submission

Core idea (see CLAUDE.md):
- The leakage boundary is the Prediction Start (PS) point: TVT_input is known for
  the heel (prefix) and NaN for the toe (= prediction zone, what we score).
- Carry-forward (hold last known TVT) is the strong naive baseline (~16 pooled RMSE).
- So we predict the DRIFT from carry-forward:  y = TVT - tvt_ps ;  pred = tvt_ps + model.
- Every feature is leakage-safe by construction: it uses only MD/X/Y/Z/GR (legal
  inputs available in the prediction zone) and pre-PS TVT_input. NEVER downhole TVT
  or the train-only surface/geology columns.
"""
from __future__ import annotations
import os, glob, time
import numpy as np
import pandas as pd

DATA = r"C:\KaggleData\rogii\rogii-wellbore-geology-prediction"
ART  = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(ART, exist_ok=True)

HZ_COLS_TRAIN = ["MD", "X", "Y", "Z", "GR", "TVT_input", "TVT"]
HZ_COLS_TEST  = ["MD", "X", "Y", "Z", "GR", "TVT_input"]


# ---------------------------------------------------------------- data_loader
def list_wells(split: str) -> list[str]:
    paths = glob.glob(os.path.join(DATA, split, "*__horizontal_well.csv"))
    return sorted(os.path.basename(p)[:8] for p in paths)


def load_well(wid: str, split: str) -> pd.DataFrame:
    cols = HZ_COLS_TRAIN if split == "train" else HZ_COLS_TEST
    p = os.path.join(DATA, split, f"{wid}__horizontal_well.csv")
    return pd.read_csv(p, usecols=lambda c: c in cols)


def load_typewell(wid: str, split: str) -> pd.DataFrame:
    p = os.path.join(DATA, split, f"{wid}__typewell.csv")
    return pd.read_csv(p, usecols=lambda c: c in ("TVT", "GR"))


def find_ps(df: pd.DataFrame) -> int:
    """Index of the last known TVT_input row (the PS point). PZ = rows after it."""
    known = df["TVT_input"].notna().values
    return int(np.where(known)[0][-1])


# ------------------------------------------------------------------- features
def _causal_roll(a: np.ndarray, w: int, fn: str) -> np.ndarray:
    s = pd.Series(a)
    r = s.rolling(w, min_periods=1)
    out = getattr(r, fn)().values
    return out


def well_features(wid: str, split: str) -> tuple[pd.DataFrame, np.ndarray | None, list[str]]:
    """Build leakage-safe features for the prediction-zone rows of one well."""
    df = load_well(wid, split)
    tw = load_typewell(wid, split)
    n = len(df)
    ps = find_ps(df)
    n_known = ps + 1

    md = df["MD"].values.astype(np.float64)
    x  = df["X"].values.astype(np.float64)
    y  = df["Y"].values.astype(np.float64)
    z  = df["Z"].values.astype(np.float64)
    gr = df["GR"].values.astype(np.float64)
    tin = df["TVT_input"].values.astype(np.float64)

    md_ps, x_ps, y_ps, z_ps = md[ps], x[ps], y[ps], z[ps]
    tvt_ps = tin[ps]

    # local dip at the boundary: slope of TVT_input vs MD over last K known rows
    K = min(300, n_known)
    sl = np.polyfit(md[ps - K + 1:ps + 1], tin[ps - K + 1:ps + 1], 1)[0]
    # recent Z trend at boundary (inclination proxy)
    zsl = np.polyfit(md[ps - K + 1:ps + 1], z[ps - K + 1:ps + 1], 1)[0]

    # causal GR rolling stats over the whole well (GR is a legal input everywhere)
    gr_m  = _causal_roll(gr, 25, "mean")
    gr_sd = _causal_roll(gr, 25, "std")
    gr_m5 = _causal_roll(gr, 5, "mean")
    gr_known_mean = np.nanmean(gr[:n_known])
    gr_known_std  = np.nanstd(gr[:n_known])
    tw_gr_mean, tw_gr_std = tw["GR"].mean(), tw["GR"].std()
    tw_span = tw["TVT"].max() - tw["TVT"].min()

    pz = np.arange(ps + 1, n)
    d_md = md[pz] - md_ps
    dz = z[pz] - z_ps
    dx = x[pz] - x_ps
    dy = y[pz] - y_ps
    disp = np.sqrt(dx * dx + dy * dy)

    feat = pd.DataFrame({
        "d_md": d_md,
        "dz": dz,
        "disp": disp,
        "dx": dx, "dy": dy,
        "inc_ratio": dz / np.where(d_md == 0, np.nan, d_md),
        "gr": gr[pz],
        "gr_m25": gr_m[pz],
        "gr_sd25": gr_sd[pz],
        "gr_m5": gr_m5[pz],
        "gr_dev_known": gr[pz] - gr_known_mean,
        "gr_z_known": (gr[pz] - gr_known_mean) / (gr_known_std + 1e-6),
        "gr_dev_tw": gr[pz] - tw_gr_mean,
        "bnd_slope": np.float64(sl),
        "z_slope": np.float64(zsl),
        "lin_delta": sl * d_md,                 # linear-extrapolation drift estimate
        "z_pred_delta": dz,                     # geometry-only drift estimate
        "z_ps": np.float64(z_ps),
        "md_ps": np.float64(md_ps),
        "n_known": np.float64(n_known),
        "tw_gr_mean": np.float64(tw_gr_mean),
        "tw_gr_std": np.float64(tw_gr_std),
        "tw_span": np.float64(tw_span),
        "frac_into_pz": np.arange(len(pz)) / max(1, len(pz) - 1),
    }).astype(np.float32)

    ids = [f"{wid}_{i}" for i in pz]
    target = None
    if split == "train":
        target = (df["TVT"].values[pz] - tvt_ps).astype(np.float32)
    # carry-forward reconstruction anchor travels with the rows
    feat["_tvt_ps"] = np.float32(tvt_ps)
    return feat, target, ids


# -------------------------------------------------------------------- cv/build
def build_matrix(split: str, cache: bool = True) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    cache_p = os.path.join(ART, f"matrix_{split}.parquet")
    if cache and os.path.exists(cache_p):
        m = pd.read_parquet(cache_p)
    else:
        feats, targs, groups, ids = [], [], [], []
        wells = list_wells(split)
        t0 = time.time()
        for i, wid in enumerate(wells):
            f, t, wid_ids = well_features(wid, split)
            f["_well"] = wid
            f["_id"] = wid_ids
            if t is not None:
                f["_y"] = t
            feats.append(f)
            if (i + 1) % 100 == 0:
                print(f"  [{split}] {i+1}/{len(wells)} wells  ({time.time()-t0:.1f}s)")
        m = pd.concat(feats, ignore_index=True)
        if cache:
            m.to_parquet(cache_p)
        print(f"  [{split}] built {len(m):,} rows from {len(wells)} wells in {time.time()-t0:.1f}s")
    feat_cols = [c for c in m.columns if not c.startswith("_")]
    X = m[feat_cols]
    y = m["_y"].values if "_y" in m else None
    groups = m["_well"].values
    tvt_ps = m["_tvt_ps"].values
    ids = m["_id"].tolist()
    return X, y, groups, ids, tvt_ps


# ---------------------------------------------------------------------- train
def default_params(seed=42):
    return dict(
        loss_function="RMSE", eval_metric="RMSE",
        iterations=1500, learning_rate=0.05, depth=8,
        l2_leaf_reg=5.0, random_seed=seed,
        thread_count=-1, verbose=0, early_stopping_rounds=100,
    )


def group_kfold_oof(X, y, groups, tvt_ps, n_splits=5, params=None, seed=42):
    """OOF training with GroupKFold by well (CatBoost). Returns OOF drift preds + models."""
    from catboost import CatBoostRegressor
    from sklearn.model_selection import GroupKFold

    feat_cols = [c for c in X.columns if c != "_tvt_ps"]
    params = params or default_params(seed)
    oof = np.zeros(len(X), dtype=np.float64)
    gkf = GroupKFold(n_splits=n_splits)
    models = []
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        m = CatBoostRegressor(**params)
        m.fit(X.iloc[tr][feat_cols], y[tr],
              eval_set=(X.iloc[va][feat_cols], y[va]), use_best_model=True)
        oof[va] = m.predict(X.iloc[va][feat_cols])
        models.append(m)
        pred_tvt = tvt_ps[va] + oof[va]
        true_tvt = tvt_ps[va] + y[va]
        rmse = np.sqrt(np.mean((pred_tvt - true_tvt) ** 2))
        print(f"  fold {fold}: best_iter={m.get_best_iteration()}  pooled RMSE={rmse:.4f}")
    return oof, models, feat_cols


def report(oof, y, groups, tvt_ps):
    pred_tvt = tvt_ps + oof
    true_tvt = tvt_ps + y
    err = pred_tvt - true_tvt
    pooled = np.sqrt(np.mean(err ** 2))
    carry  = np.sqrt(np.mean((tvt_ps - true_tvt) ** 2))  # ΔTVT=0
    # per-well
    dfp = pd.DataFrame({"w": groups, "e2": err ** 2})
    pw = dfp.groupby("w")["e2"].mean().pow(0.5)
    print("\n================ CV REPORT ================")
    print(f"carry-forward pooled RMSE : {carry:.4f}")
    print(f"MODEL        pooled RMSE  : {pooled:.4f}   (improvement {carry-pooled:+.4f})")
    print(f"per-well RMSE  mean={pw.mean():.3f}  median={pw.median():.3f}  p90={pw.quantile(.9):.3f}  max={pw.max():.3f}")
    return pooled, carry


# ------------------------------------------------------------------ inference
def make_submission(models, feat_cols, out="submission.csv"):
    Xte, _, gte, ids, tvt_ps_te = build_matrix("test", cache=False)
    pred = np.mean([m.predict(Xte[feat_cols]) for m in models], axis=0)
    tvt = tvt_ps_te + pred
    sub = pd.DataFrame({"id": ids, "tvt": tvt})
    # align to sample_submission order
    ss = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    sub = ss[["id"]].merge(sub, on="id", how="left")
    assert sub["tvt"].notna().all(), "missing ids vs sample_submission!"
    assert len(sub) == len(ss), f"row count {len(sub)} != {len(ss)}"
    out_p = os.path.join(os.path.dirname(__file__), "..", out)
    sub.to_csv(out_p, index=False)
    print(f"\nwrote {out_p}  shape={sub.shape}  tvt[min/mean/max]={sub.tvt.min():.1f}/{sub.tvt.mean():.1f}/{sub.tvt.max():.1f}")
    return sub


if __name__ == "__main__":
    t0 = time.time()
    print(">>> building train matrix")
    X, y, groups, ids, tvt_ps = build_matrix("train")
    print(f">>> matrix {X.shape}  features={[c for c in X.columns if c!='_tvt_ps']}")
    print(">>> GroupKFold OOF")
    oof, models, feat_cols = group_kfold_oof(X, y, groups, tvt_ps)
    report(oof, y, groups, tvt_ps)
    print(">>> submission")
    make_submission(models, feat_cols)
    print(f">>> done in {time.time()-t0:.1f}s")
