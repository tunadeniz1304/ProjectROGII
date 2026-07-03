"""Build `9-251-plus-tabpfn3.ipynb` = the 9.251 reference notebook + TabPFN-3 as an
EXTRA hill-climbing member.

Surgery (not a rewrite): we load the original 9-251 notebook and insert two new,
fully-guarded cells:
  1) a SETUP cell right after `class CFG`   -> offline install + weights wiring + config
  2) a TRAIN cell right before the cell      -> GroupKFold OOF + test preds for TabPFN-3,
     `oof_preds = pd.DataFrame(...)`             registered into oof_preds/test_preds dicts

The TabPFN member predicts the SAME target (ΔTVT) on the SAME features X as the 6 GBMs,
using the SAME GroupKFold split, so its OOF aligns 1:1 for the Climber. Everything is
wrapped in try/except + a USE flag: if the package/weights are missing or anything errors,
the notebook prints a warning and runs the ORIGINAL 9.25 blend unchanged.
"""
import json, os

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "..", "9-251-rogii-wellbore-geology-prediction-dwt-based.ipynb")
OUT = os.path.join(HERE, "..", "9-251-plus-tabpfn3.ipynb")

# --- the two inserted code cells -------------------------------------------------------

SETUP_MD = "# 1b. TabPFN-3 setup (offline, fully guarded)"

SETUP_CODE = r'''
# === TabPFN-3 setup (offline, fully guarded) =================================
# Adds TabPFN-3 as an EXTRA hill-climbing member. If the package/weights are
# missing or anything fails, we print a warning and SKIP TabPFN -> the original
# 9.25 pipeline runs unchanged (never regresses the baseline).
#
# REQUIRED Kaggle datasets to attach (internet is OFF at submission):
#   1) tabpfn wheels    -> /kaggle/input/tabpfn-wheels    (built via `pip download tabpfn -d .`)
#   2) tabpfn-3 weights -> /kaggle/input/tabpfn3-weights  (gated HF repo Prior-Labs/tabpfn_3)
import os, sys, subprocess, time
import numpy as np

class TPF:
    use            = True
    wheels_dir     = "/kaggle/input/tabpfn-wheels"
    weights_dir    = "/kaggle/input/tabpfn3-weights"
    regressor_ckpt = "tabpfn-v3-regressor-v3_default.ckpt"  # 1M-capable default regressor
    device         = "cuda"     # notebook already runs on GPU; CPU is impractical for TabPFN-3
    context_max    = 50_000     # in-context rows per fold (<=100k -> 2000-feat regime, ~3GB GPU)
    pred_chunk     = 20_000     # rows per predict() call (predict recomputes the context each call)
    n_estimators   = 4          # ensemble members (8 = default/slower, 2 = faster)
    seed           = 42
    member_name    = "tabpfn3"
    model_path     = "auto"

# Offline env: never touch the network at submission time.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if os.path.isdir(TPF.weights_dir):
    os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", TPF.weights_dir)

def _ensure_tabpfn():
    try:
        import tabpfn  # already present?
        return True
    except Exception:
        pass
    if not os.path.isdir(TPF.wheels_dir):
        print(f"!! TabPFN wheels not found at {TPF.wheels_dir} -> skipping TabPFN member.")
        return False
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                        f"--find-links={TPF.wheels_dir}", "tabpfn"],
                       check=True, capture_output=True, text=True)
        import tabpfn  # noqa: F401
        print("TabPFN installed offline:", getattr(tabpfn, "__version__", "?"))
        return True
    except Exception as e:
        print("!! TabPFN install failed -> skipping TabPFN member (9.25 pipeline unaffected).")
        print(str(e)[:800])
        return False

TPF.use = _ensure_tabpfn()

# Resolve a local regressor checkpoint so loading stays fully offline.
if TPF.use and os.path.isdir(TPF.weights_dir):
    cand = os.path.join(TPF.weights_dir, TPF.regressor_ckpt)
    if os.path.isfile(cand):
        TPF.model_path = cand
    else:
        cks = sorted(f for f in os.listdir(TPF.weights_dir)
                     if f.endswith(".ckpt") and "regress" in f.lower())
        if cks:
            TPF.model_path = os.path.join(TPF.weights_dir, cks[0])
    print("TabPFN-3 ready | weights:", TPF.model_path, "| device:", TPF.device)
elif TPF.use:
    print(f"!! TabPFN weights dir not found at {TPF.weights_dir}; will try model_path='auto' "
          "(needs internet -> will be skipped offline).")
'''

TRAIN_MD = "## 3.3 TabPFN-3 (extra blend member)"

TRAIN_CODE = r'''
# === TabPFN-3 as an extra hill-climbing member ===============================
# Mirrors train_lightgbm / train_catboost: GroupKFold OOF + fold-averaged test
# preds, predicting the SAME target (ΔTVT) on the SAME features X, with the SAME
# CFG.cv split -> OOF aligns 1:1 with the GBM members. Registered into oof_preds/
# test_preds BEFORE they become DataFrames, so the Climber learns its weight.
# Climber has allow_negative_weights=False -> a useless member gets weight ~0 and
# CANNOT hurt the 9.25 blend.
if TPF.use:
    from tabpfn import TabPFNRegressor

    def _predict_chunked(reg, M, chunk):
        out = np.empty(len(M), np.float32)
        for s in range(0, len(M), chunk):
            out[s:s + chunk] = np.asarray(reg.predict(M[s:s + chunk]), dtype=np.float32)
        return out

    def train_tabpfn(name):
        t0 = time.time()
        rng = np.random.RandomState(TPF.seed)
        num_cols = X.select_dtypes(include=[np.number]).columns   # robust to a stray non-numeric col
        Xv  = X[num_cols].to_numpy(dtype=np.float32)  # NaNs kept -> TabPFN-3 native missing handling
        yv  = y.to_numpy(dtype=np.float32)
        Xtv = X_test[num_cols].to_numpy(dtype=np.float32)
        oof = np.zeros(len(train_df), np.float32)
        tst = np.zeros(len(test_df),  np.float32)
        fold_scores = []
        print(f"Training {name} | n_feat={Xv.shape[1]} context_max={TPF.context_max} "
              f"n_est={TPF.n_estimators}\n")
        for fold_idx, (tr, va) in enumerate(CFG.cv.split(X, y, groups=g)):
            sub = tr if len(tr) <= TPF.context_max else rng.choice(tr, TPF.context_max, replace=False)
            reg = TabPFNRegressor(
                model_path=TPF.model_path, device=TPF.device,
                n_estimators=TPF.n_estimators, random_state=TPF.seed,
                ignore_pretraining_limits=True,     # we subsample the context ourselves
                fit_mode="fit_with_cache",          # cache context so chunked predict is fast
                memory_saving_mode=True,
            )
            reg.fit(Xv[sub], yv[sub])
            oof[va]  = _predict_chunked(reg, Xv[va], TPF.pred_chunk)
            tst     += _predict_chunked(reg, Xtv,    TPF.pred_chunk) / CFG.n_splits
            sc = root_mean_squared_error(y.iloc[va], oof[va])
            fold_scores.append(sc)
            print(f"--- Fold {fold_idx} RMSE: {sc:.3f}  ({time.time() - t0:.0f}s, ctx={len(sub)})")
            del reg
        overall = root_mean_squared_error(y, oof)
        print(f"\nOverall RMSE: {overall:.4f}  ({time.time() - t0:.0f}s)")
        return oof, tst, overall, fold_scores

    try:
        _oof, _tst, _ov, _fs = train_tabpfn(TPF.member_name)
        oof_preds[TPF.member_name]      = _oof
        test_preds[TPF.member_name]     = _tst
        overall_scores[TPF.member_name] = _ov
        fold_scores[TPF.member_name]    = _fs
        print(f"\n>>> Added '{TPF.member_name}' to the blend (overall RMSE {_ov:.3f}).")
    except Exception:
        print("!! TabPFN training failed -> member skipped; 9.25 blend unaffected.")
        import traceback; traceback.print_exc()
else:
    print(">>> TabPFN-3 disabled -> running the original 9.25 blend unchanged.")
'''


def _lines(s):
    return s.strip("\n").splitlines(keepends=True)


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(src)}


def code_cell(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _lines(src)}


def main():
    nb = json.load(open(SRC, encoding="utf-8"))
    cells = nb["cells"]

    # --- Patch GBM load-guards: check for the actual models.pkl FILE, not just the
    # directory. The ravaghi artifacts dataset ships the hill_climbing module + features
    # but NOT the trained model pkls, so the original `.../name).exists()` (a directory)
    # is True yet `joblib.load(.../models.pkl)` then FileNotFoundErrors. Checking the file
    # makes the notebook auto-retrain from scratch when the pkls are absent (and still
    # load them fast when a full artifacts dataset is present).
    OLD_GUARD = 'if (CFG.artifacts_path / "models" / name).exists():'
    NEW_GUARD = 'if (CFG.artifacts_path / "models" / name / "models.pkl").exists():'
    n_patch = 0
    for c in cells:
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            if OLD_GUARD in src:
                c["source"] = src.replace(OLD_GUARD, NEW_GUARD).splitlines(keepends=True)
                n_patch += 1
    assert n_patch == 2, f"expected 2 GBM load-guards (lgb+cb), patched {n_patch}"

    def find(pred):
        for i, c in enumerate(cells):
            if pred("".join(c["source"])):
                return i
        raise RuntimeError("anchor not found")

    cfg_i = find(lambda s: "class CFG" in s)
    # Run TabPFN after the GBM dicts are populated but before the "# 4. Hill climbing"
    # section (and the dict->DataFrame conversion) -> keeps section headers in order.
    hc_md_i = find(lambda s: s.strip().startswith("# 4. Hill climbing"))
    dictdf_i = find(lambda s: "oof_preds = pd.DataFrame(oof_preds)" in s)
    assert hc_md_i < dictdf_i, "unexpected layout: hill-climbing md must precede dict->DataFrame"

    # Insert from the higher index first so earlier inserts don't shift later anchors.
    cells.insert(hc_md_i, code_cell(TRAIN_CODE))
    cells.insert(hc_md_i, md_cell(TRAIN_MD))
    cells.insert(cfg_i + 1, code_cell(SETUP_CODE))
    cells.insert(cfg_i + 1, md_cell(SETUP_MD))

    # strip outputs from all code cells -> clean inference notebook
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None

    json.dump(nb, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", os.path.abspath(OUT), "| cells:", len(cells))
    print(f"inserted SETUP after CFG (idx {cfg_i + 1}), TRAIN before '# 4. Hill climbing' (idx {hc_md_i})")


if __name__ == "__main__":
    main()
