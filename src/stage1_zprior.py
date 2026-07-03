"""
ROGII — Stage 1: leakage-safe Z-prior + geology-plane drift model.

WHY (see ROGII_BATTLE_PLAN.md):
- The dominant signal is the well's own Z trajectory, NOT GR matching.
- On the 773-well train matrix, an oracle per-well line  dtvt = a*dz + b  gives
  RMSE 6.80 (vs carry-forward 15.91, vs the public GBM notebook 9.25).
- So the whole game is estimating, leakage-safely, each well's local dip slope `a`
  and offset `b`. The geology surface ANCC(X,Y) gives a principled, spatial estimate:
        TVT = formation_surface(X,Y) - Z + b_well        (b_well from the known heel)
  which encodes BOTH the per-well dip (via how the surface & Z vary along the lateral)
  and the offset. We feed several such formation-implied drifts to a GBM on the drift
  target, with GroupKFold(well) CV, then smooth per well.

EVERY feature is legal in the prediction zone: it uses only MD/X/Y/Z/GR (known
everywhere) and pre-PS TVT_input (the known heel). The geology surface is built from
OTHER training wells (self excluded), so it never sees the target well's hidden TVT.

This is a Kaggle-ready, self-contained script (no external deps beyond
numpy/pandas/scipy/scikit-learn/lightgbm, all preinstalled on Kaggle).
"""
from __future__ import annotations
import os, glob, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupKFold

SEED = 42
np.random.seed(SEED)
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
N_SPLITS = 5
PLANE_K = 10


# ----------------------------------------------------------- data location
def find_competition_dir():
    env = os.environ.get("ROGII_DATA")
    if env and os.path.isdir(os.path.join(env, "train")):
        return env
    cands = [
        "/kaggle/input/competitions/rogii-wellbore-geology-prediction",
        "/kaggle/input/rogii-wellbore-geology-prediction",
        r"C:\KaggleData\rogii\rogii-wellbore-geology-prediction",
    ]
    for c in cands:
        if os.path.isdir(os.path.join(c, "train")):
            return c
    for base in ("/kaggle/input", r"C:\KaggleData"):
        hits = glob.glob(os.path.join(base, "**", "*__horizontal_well.csv"), recursive=True)
        if hits:
            d = os.path.dirname(hits[0])
            return os.path.dirname(d) if os.path.basename(d) in ("train", "test") else d
    raise FileNotFoundError("competition data not found; add the competition as Input.")


DATA = find_competition_dir()
print("DATA =", DATA)
HZ_TRAIN = ["MD", "X", "Y", "Z", "GR", "TVT_input", "TVT"] + FORMATIONS
HZ_TEST = ["MD", "X", "Y", "Z", "GR", "TVT_input"]


def list_wells(split):
    return sorted(os.path.basename(p)[:8]
                  for p in glob.glob(os.path.join(DATA, split, "*__horizontal_well.csv")))


def load_well(wid, split):
    cols = HZ_TRAIN if split == "train" else HZ_TEST
    return pd.read_csv(os.path.join(DATA, split, f"{wid}__horizontal_well.csv"),
                       usecols=lambda c: c in cols)


def find_ps(df):
    known = df["TVT_input"].notna().values
    return int(np.where(known)[0][-1])


# ------------------------------------------------ geology-surface spatial model
class FormationPlaneKNN:
    """Local-plane KNN estimate of each formation surface at a query (X,Y).
    Built from training wells' per-well median formation depths. Self-excluded.
    (Adapted from the public 9.251 notebook's FormationPlaneKNN — proven component.)"""
    def __init__(self, well_ids):
        rows = []
        for wid in well_ids:
            try:
                df = load_well(wid, "train")[["X", "Y"] + FORMATIONS].dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            row = {"wid": wid, "x": float(df["X"].median()), "y": float(df["Y"].median())}
            for c in FORMATIONS:
                row[f"{c}_m"] = float(df[c].median())
            rows.append(row)
        self.df = pd.DataFrame(rows)
        self.wmap = {w: i for i, w in enumerate(self.df["wid"])}
        xy = self.df[["x", "y"]].to_numpy()
        self.scale = np.where(xy.std(0) < 1e-3, 1.0, xy.std(0))
        self.tree = cKDTree(xy / self.scale)
        self.xa = self.df["x"].to_numpy(); self.ya = self.df["y"].to_numpy()
        self.fa = self.df[[f"{c}_m" for c in FORMATIONS]].to_numpy(np.float64)

    def impute(self, xy_q, self_wid=None, k=PLANE_K):
        xy_q = np.atleast_2d(xy_q).astype(np.float64)
        q = xy_q / self.scale
        nf = min(k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        dist = np.atleast_2d(dist); idx = np.atleast_2d(idx)
        if self_wid in self.wmap:
            dist = np.where(idx == self.wmap[self_wid], np.inf, dist)
        ordr = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ordr, 1); ik = np.take_along_axis(idx, ordr, 1)
        vk = np.isfinite(dk); w = np.where(vk, 1.0 / (dk + 1e-3), 0.0).astype(np.float64)
        xn = self.xa[ik]; yn = self.ya[ik]; fn = self.fa[ik]; wx = w * xn; wy = w * yn
        A = np.zeros((len(q), 3, 3))
        A[:, 0, 0] = (wx * xn).sum(1); A[:, 0, 1] = (wx * yn).sum(1); A[:, 0, 2] = wx.sum(1)
        A[:, 1, 0] = A[:, 0, 1]; A[:, 1, 1] = (wy * yn).sum(1); A[:, 1, 2] = wy.sum(1)
        A[:, 2, 0] = A[:, 0, 2]; A[:, 2, 1] = A[:, 1, 2]; A[:, 2, 2] = w.sum(1)
        A[:, 0, 0] += 1e-9; A[:, 1, 1] += 1e-9; A[:, 2, 2] += 1e-9
        rhs = np.stack([(wx[:, :, None] * fn).sum(1), (wy[:, :, None] * fn).sum(1),
                        (w[:, :, None] * fn).sum(1)], 1)
        try:
            coef = np.linalg.solve(A, rhs)
        except Exception:
            coef = np.zeros((len(q), 3, len(FORMATIONS)))
            for r in range(len(q)):
                try:
                    coef[r] = np.linalg.pinv(A[r]) @ rhs[r]
                except Exception:
                    pass
        Xq = xy_q[:, 0]; Yq = xy_q[:, 1]
        pred = (Xq[:, None] * coef[:, 0, :] + Yq[:, None] * coef[:, 1, :] + coef[:, 2, :])
        bad = ~vk.any(1)
        if bad.any():
            pred[bad] = self.fa.mean(0)
        return pred.astype(np.float32)   # (n_query, n_formations)


# ------------------------------------------------------------------- features
def _causal_roll(a, w, fn):
    return getattr(pd.Series(a).rolling(w, min_periods=1), fn)().values


def well_features(wid, split, geo: FormationPlaneKNN):
    df = load_well(wid, split)
    n = len(df); ps = find_ps(df); n_known = ps + 1
    md = df["MD"].values.astype(np.float64); x = df["X"].values.astype(np.float64)
    y = df["Y"].values.astype(np.float64);  z = df["Z"].values.astype(np.float64)
    gr = df["GR"].values.astype(np.float64); tin = df["TVT_input"].values.astype(np.float64)
    md_ps, x_ps, y_ps, z_ps, tvt_ps = md[ps], x[ps], y[ps], z[ps], tin[ps]

    # heel dip: slope of TVT_input and Z vs MD over last K known rows
    K = min(300, n_known)
    sl = np.polyfit(md[ps - K + 1:ps + 1], tin[ps - K + 1:ps + 1], 1)[0]
    zsl = np.polyfit(md[ps - K + 1:ps + 1], z[ps - K + 1:ps + 1], 1)[0]

    gr_m = _causal_roll(gr, 25, "mean"); gr_sd = _causal_roll(gr, 25, "std")
    gr_known_mean = np.nanmean(gr[:n_known]); gr_known_std = np.nanstd(gr[:n_known])

    pz = np.arange(ps + 1, n)
    d_md = md[pz] - md_ps; dz = z[pz] - z_ps
    dx = x[pz] - x_ps; dy = y[pz] - y_ps; disp = np.sqrt(dx * dx + dy * dy)

    # ---- geology-plane implied drift (the key leakage-safe dip+offset signal) ----
    self_wid = wid if split == "train" else None
    # surfaces at the known heel rows -> per-formation offset b_F = median(TVT_input + Z - F_hat)
    kxy = np.column_stack([x[:n_known], y[:n_known]])
    Fk = geo.impute(kxy, self_wid=self_wid)                 # (n_known, nF)
    bF = np.nanmedian((tin[:n_known] + z[:n_known])[:, None] - Fk, axis=0)  # (nF,)
    # surfaces along the prediction zone
    pxy = np.column_stack([x[pz], y[pz]])
    Fp = geo.impute(pxy, self_wid=self_wid)                 # (n_pz, nF)
    tvt_geo = Fp - z[pz][:, None] + bF[None, :]             # implied TVT per formation
    drift_geo = tvt_geo - tvt_ps                            # implied drift per formation
    drift_geo_mean = np.nanmean(drift_geo, axis=1)

    feat = {
        "d_md": d_md, "dz": dz, "disp": disp, "dx": dx, "dy": dy,
        "inc_ratio": dz / np.where(d_md == 0, np.nan, d_md),
        "gr": gr[pz], "gr_m25": gr_m[pz], "gr_sd25": gr_sd[pz],
        "gr_dev_known": gr[pz] - gr_known_mean,
        "gr_z_known": (gr[pz] - gr_known_mean) / (gr_known_std + 1e-6),
        "bnd_slope": np.float64(sl), "z_slope": np.float64(zsl),
        "lin_delta": sl * d_md, "z_pred_delta": dz,
        "z_ps": np.float64(z_ps), "md_ps": np.float64(md_ps),
        "n_known": np.float64(n_known),
        "frac_into_pz": np.arange(len(pz)) / max(1, len(pz) - 1),
        "drift_geo_mean": drift_geo_mean,
    }
    for j, f in enumerate(FORMATIONS):
        feat[f"drift_geo_{f}"] = drift_geo[:, j]

    feat = pd.DataFrame(feat).astype(np.float32)
    ids = [f"{wid}_{i}" for i in pz]
    target = (df["TVT"].values[pz] - tvt_ps).astype(np.float32) if split == "train" else None
    feat["_tvt_ps"] = np.float32(tvt_ps)
    feat["_well"] = wid
    feat["_id"] = ids
    feat["_row"] = pz
    if target is not None:
        feat["_y"] = target
    return feat


def build_matrix(split, geo):
    wells = list_wells(split); t0 = time.time(); out = []
    for i, wid in enumerate(wells):
        try:
            out.append(well_features(wid, split, geo))
        except Exception as e:
            print(f"  !! {split} {wid}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  [{split}] {i+1}/{len(wells)}  ({time.time()-t0:.0f}s)")
    m = pd.concat(out, ignore_index=True)
    print(f"  [{split}] {len(m):,} rows, {len(wells)} wells, {time.time()-t0:.0f}s")
    return m


def pooled_rmse(pred_drift, y, tvt_ps):
    return float(np.sqrt(np.mean(((tvt_ps + pred_drift) - (tvt_ps + y)) ** 2)))


def smooth_per_well(df, col):
    out = df[col].values.astype(float).copy()
    for _, g in df.groupby("_well", sort=False):
        idx = g.sort_values("_row").index.values
        v = df.loc[idx, col].values.astype(float); nn = len(v)
        wl = min(17, nn);  wl -= (wl % 2 == 0)
        if wl >= 5:
            out[df.index.get_indexer(idx)] = savgol_filter(v, wl, 2)
    df[col] = out
    return df


def main():
    import lightgbm as lgb
    t0 = time.time()
    print(">>> building geology-surface model")
    geo = FormationPlaneKNN(list_wells("train"))
    print(">>> train matrix"); tr = build_matrix("train", geo)
    feat_cols = [c for c in tr.columns if not c.startswith("_")]
    X = tr[feat_cols].values; y = tr["_y"].values.astype(np.float64)
    g = tr["_well"].values; tvt_ps = tr["_tvt_ps"].values

    print(">>> GroupKFold CV")
    oof = np.zeros(len(tr)); gkf = GroupKFold(N_SPLITS)
    params = dict(objective="regression", num_leaves=128, learning_rate=0.03,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  min_child_samples=40, n_jobs=-1, verbose=-1)
    models = []
    for f, (tri, vai) in enumerate(gkf.split(X, y, g)):
        d = lgb.Dataset(X[tri], y[tri]); dv = lgb.Dataset(X[vai], y[vai])
        mdl = lgb.train(params, d, num_boost_round=2000, valid_sets=[dv],
                        callbacks=[lgb.early_stopping(80, verbose=False)])
        oof[vai] = mdl.predict(X[vai]); models.append(mdl)
        print(f"  fold{f}: pooled RMSE={pooled_rmse(oof[vai], y[vai], tvt_ps[vai]):.4f}"
              f"  best_iter={mdl.best_iteration}")
    carry = pooled_rmse(np.zeros_like(y), y, tvt_ps)
    raw = pooled_rmse(oof, y, tvt_ps)
    tr["_pred"] = tvt_ps + oof
    tr = smooth_per_well(tr, "_pred")
    sm = float(np.sqrt(np.mean((tr["_pred"].values - (tvt_ps + y)) ** 2)))
    print("\n================ STAGE-1 CV ================")
    print(f"carry-forward pooled RMSE : {carry:.4f}")
    print(f"model      pooled RMSE    : {raw:.4f}")
    print(f"+per-well smoothing       : {sm:.4f}   <-- compare to public 9.25")

    print("\n>>> test inference")
    te = build_matrix("test", geo)
    Xte = te[feat_cols].values
    pred = np.mean([m.predict(Xte) for m in models], axis=0)
    te["_pred"] = te["_tvt_ps"].values + pred
    te = smooth_per_well(te, "_pred")

    ss = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    sub = ss[["id"]].merge(te[["_id", "_pred"]].rename(columns={"_id": "id", "_pred": "tvt"}),
                           on="id", how="left")
    # fallback fill for any unmatched id (carry-forward mean)
    sub["tvt"] = sub["tvt"].fillna(float(te["_tvt_ps"].mean()))
    sub.to_csv("submission.csv", index=False)
    print("wrote submission.csv  shape=", sub.shape)
    print("tvt min/mean/max:", round(float(sub.tvt.min()),1),
          round(float(sub.tvt.mean()),1), round(float(sub.tvt.max()),1))
    print(">>> done in %.0fs" % (time.time()-t0))


if __name__ == "__main__":
    main()
