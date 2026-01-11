#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

from scripts.label_utils import normalize_labels

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

FEAT_PATH = os.getenv("FEAT_PATH", "data/raw/features.csv")
PAT_PATH  = os.getenv("PAT_PATH",  "data/raw/patient_info.csv")
EMB_PATH  = os.getenv("EMB_PATH",  "data/processed/expertkg_node2vec_patient_embeddings.csv")

FEATURE_LIST_PATH = os.getenv("FEATURE_LIST", "")
RAW_BASELINE_MODE = os.getenv("RAW_BASELINE_MODE", "mean_only")  # "feature_list" or "mean_only"

TARGET_COL_IN_PATIENT = os.getenv("TARGET_COL", "ADHD")
TARGET_NAME_NORMALIZED = "adhd"
POS_LABEL = os.getenv("POS_LABEL", "0")

SEED = int(os.getenv("SEED", "42"))
N_SPLITS = int(os.getenv("N_SPLITS", "5"))
N_REPEATS = int(os.getenv("N_REPEATS", "50"))  # permutation repeats per fold

OUTDIR = os.getenv("OUTDIR", "data/processed")
os.makedirs(OUTDIR, exist_ok=True)

def _read_csv_auto(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python", sep=None)
    if df.shape[1] == 1 and ";" in df.columns[0]:
        df = pd.read_csv(path, sep=";")
    return df

def _load_feature_list(path: str) -> list[str]:
    if not path:
        raise ValueError("FEATURE_LIST is empty but RAW_BASELINE_MODE=feature_list was requested.")
    with open(path, "r", encoding="utf-8") as f:
        cols = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    return cols

def _make_clf():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=5000, solver="liblinear", random_state=SEED)),
    ])

def _align(pat, feat, emb):
    for df in (pat, feat, emb):
        df.columns = [c.strip() for c in df.columns]

    if TARGET_COL_IN_PATIENT not in pat.columns:
        raise KeyError(f"patient_info.csv missing '{TARGET_COL_IN_PATIENT}'")

    if "ID" not in pat.columns or "ID" not in feat.columns or "ID" not in emb.columns:
        raise KeyError("All of patient_info.csv, features.csv, embeddings.csv must contain 'ID'")

    pat = pat.copy()
    pat[TARGET_NAME_NORMALIZED] = normalize_labels(pat[TARGET_COL_IN_PATIENT], POS_LABEL)
    pat = pat.dropna(subset=[TARGET_NAME_NORMALIZED]).copy()
    pat[TARGET_NAME_NORMALIZED] = pat[TARGET_NAME_NORMALIZED].astype(int)

    common_ids = sorted(list(set(pat["ID"]).intersection(set(feat["ID"])).intersection(set(emb["ID"]))))

    pat_c  = pat[pat["ID"].isin(common_ids)].sort_values("ID").reset_index(drop=True)
    feat_c = feat[feat["ID"].isin(common_ids)].sort_values("ID").reset_index(drop=True)
    emb_c  = emb[emb["ID"].isin(common_ids)].sort_values("ID").reset_index(drop=True)

    y = pat_c[TARGET_NAME_NORMALIZED].to_numpy().astype(int)
    return common_ids, y, feat_c, emb_c

def _select_raw_cols(feat_c: pd.DataFrame) -> list[str]:
    if RAW_BASELINE_MODE == "feature_list":
        cols = _load_feature_list(FEATURE_LIST_PATH)
        missing = [c for c in cols if c not in feat_c.columns]
        if missing:
            raise KeyError(f"Missing columns in features.csv: {missing[:10]} ... (total {len(missing)})")
        return cols

    # default: mean-only accel
    raw_cols = []
    for c in feat_c.columns:
        if c == "ID":
            continue
        cc = c.lower()
        if ("acc__" in cc) and ("mean" in cc):
            raw_cols.append(c)

    if len(raw_cols) == 0:
        raw_cols = [c for c in feat_c.columns if c != "ID" and pd.api.types.is_numeric_dtype(feat_c[c])]

    return raw_cols

def _select_emb_cols(emb_c: pd.DataFrame) -> list[str]:
    emb_cols = [c for c in emb_c.columns if c.startswith("emb_")]
    if not emb_cols:
        raise KeyError("No emb_* columns found in embeddings CSV")
    return emb_cols

def _perm_importance_cv(X: np.ndarray, y: np.ndarray, feature_names: list[str], tag: str):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    clf = _make_clf()

    all_imps = []

    aucs = []
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])
        pos_index = list(clf.named_steps["lr"].classes_).index(1)
        p = proba[:, pos_index]
        auc = roc_auc_score(y[te], p)
        aucs.append(auc)

        r = permutation_importance(
            clf,
            X[te],
            y[te],
            scoring="roc_auc",
            n_repeats=N_REPEATS,
            random_state=SEED,
            n_jobs=1,
        )
        all_imps.append(r.importances_mean)

    aucs = np.array(aucs, dtype=float)
    all_imps = np.vstack(all_imps)  # folds x features

    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": all_imps.mean(axis=0),
        "importance_std": all_imps.std(axis=0),
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    out_csv = os.path.join(OUTDIR, f"perm_importance_{tag}.csv")
    df.to_csv(out_csv, index=False)

    print(f"\n[{tag}] AUC folds={np.round(aucs,3)} mean={aucs.mean():.3f} std={aucs.std():.3f}")
    print(f"[{tag}] saved: {out_csv}")
    print(df.head(20).to_string(index=False))

def main():
    feat = _read_csv_auto(FEAT_PATH)
    pat  = _read_csv_auto(PAT_PATH)
    emb  = _read_csv_auto(EMB_PATH)

    common_ids, y, feat_c, emb_c = _align(pat, feat, emb)

    raw_cols = _select_raw_cols(feat_c)
    emb_cols = _select_emb_cols(emb_c)

    X_raw = feat_c[raw_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    X_emb = emb_c[emb_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    X_comb = np.concatenate([X_raw, X_emb], axis=1)

    print(f"Subjects: {len(common_ids)} | class balance={np.bincount(y, minlength=2)}")
    print(f"RAW_BASELINE_MODE={RAW_BASELINE_MODE} | raw dims={X_raw.shape[1]} | emb dims={X_emb.shape[1]} | comb dims={X_comb.shape[1]}")
    if RAW_BASELINE_MODE == "feature_list":
        print(f"FEATURE_LIST={FEATURE_LIST_PATH}")

    _perm_importance_cv(X_raw, y, raw_cols, tag="raw")
    _perm_importance_cv(X_emb, y, emb_cols, tag="emb")
    _perm_importance_cv(X_comb, y, raw_cols + emb_cols, tag="concat")

if __name__ == "__main__":
    main()
