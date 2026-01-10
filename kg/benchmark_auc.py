#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# -----------------------------
# paths
# -----------------------------
FEAT_PATH = os.getenv("FEAT_PATH", "data/raw/features.csv")
PAT_PATH  = os.getenv("PAT_PATH",  "data/raw/patient_info.csv")
EMB_PATH  = os.getenv("EMB_PATH",  "data/processed/expertkg_node2vec_patient_embeddings.csv")

# IMPORTANT: for FAIR baseline
FEATURE_LIST_PATH = os.getenv("FEATURE_LIST", "kg/features_entropy_variability.txt")
RAW_BASELINE_MODE = os.getenv("RAW_BASELINE_MODE", "feature_list").lower()
# allowed: "feature_list" (FAIR, default) or "mean_only_accel" (legacy)

# target handling
TARGET_COL_IN_PATIENT = os.getenv("TARGET_COL", "ADHD")  # source column in patient_info.csv
TARGET_NAME_NORMALIZED = "adhd"

# reproducibility
SEED = int(os.getenv("SEED", "42"))
N_SPLITS = int(os.getenv("N_SPLITS", "5"))


def _read_csv_auto(path: str) -> pd.DataFrame:
    # autodetect separator
    return pd.read_csv(path, engine="python", sep=None)


def _read_feature_list(path: str) -> list[str]:
    feats = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            feats.append(s)
    return feats


def eval_auc(X: np.ndarray, y: np.ndarray, tag: str) -> None:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=5000, solver="liblinear", random_state=SEED)),
    ])

    aucs = []
    for tr, te in skf.split(X, y):
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])
        pos_index = list(clf.named_steps["lr"].classes_).index(1)
        p = proba[:, pos_index]
        aucs.append(roc_auc_score(y[te], p))

    aucs = np.array(aucs, dtype=float)
    print(f"{tag}: AUC mean={aucs.mean():.3f}  std={aucs.std():.3f}  folds={np.round(aucs,3)}")


def main():
    # -----------------------------
    # load
    # -----------------------------
    feat = _read_csv_auto(FEAT_PATH)
    pat  = _read_csv_auto(PAT_PATH)
    emb  = _read_csv_auto(EMB_PATH)

    feat.columns = [c.strip() for c in feat.columns]
    pat.columns  = [c.strip() for c in pat.columns]
    emb.columns  = [c.strip() for c in emb.columns]

    # -----------------------------
    # normalize target in patient_info
    # -----------------------------
    if TARGET_COL_IN_PATIENT not in pat.columns:
        raise KeyError(
            f"patient_info.csv missing target column '{TARGET_COL_IN_PATIENT}'. "
            f"Available: {pat.columns.tolist()}"
        )

    if "ID" not in pat.columns:
        raise KeyError("patient_info.csv must contain an 'ID' column")
    if "ID" not in feat.columns:
        raise KeyError("features.csv must contain an 'ID' column")
    if "ID" not in emb.columns:
        raise KeyError("embeddings csv must contain an 'ID' column")

    pat = pat.copy()
    pat[TARGET_NAME_NORMALIZED] = pd.to_numeric(pat[TARGET_COL_IN_PATIENT], errors="coerce").astype("Int64")
    pat = pat.dropna(subset=[TARGET_NAME_NORMALIZED]).copy()
    pat[TARGET_NAME_NORMALIZED] = pat[TARGET_NAME_NORMALIZED].astype(int)

    # -----------------------------
    # restrict to common IDs
    # -----------------------------
    common_ids = set(pat["ID"]).intersection(set(feat["ID"])).intersection(set(emb["ID"]))
    common_ids = sorted(list(common_ids))

    pat_c  = pat[pat["ID"].isin(common_ids)].sort_values("ID").reset_index(drop=True)
    feat_c = feat[feat["ID"].isin(common_ids)].sort_values("ID").reset_index(drop=True)
    emb_c  = emb[emb["ID"].isin(common_ids)].sort_values("ID").reset_index(drop=True)

    y = pat_c[TARGET_NAME_NORMALIZED].to_numpy().astype(int)

    # -----------------------------
    # RAW baseline (FAIR): use FEATURE_LIST
    # -----------------------------
    # Protect against label leakage in raw table
    leakage_cols = {TARGET_NAME_NORMALIZED.lower(), TARGET_COL_IN_PATIENT.lower(), "label", "class", "target"}

    if RAW_BASELINE_MODE == "mean_only_accel":
        # legacy heuristic (kept only if you explicitly want it)
        raw_cols = []
        for c in feat_c.columns:
            if c == "ID":
                continue
            cc = c.lower()
            if ("acc__" in cc) and ("mean" in cc):
                raw_cols.append(c)

        if len(raw_cols) == 0:
            raw_cols = [
                c for c in feat_c.columns
                if c != "ID"
                and pd.api.types.is_numeric_dtype(feat_c[c])
                and c.lower() not in leakage_cols
            ]

        raw_mode_info = f"RAW_BASELINE_MODE=mean_only_accel (legacy) using {len(raw_cols)} cols"

    else:
        # FAIR mode: strict feature list
        wanted = _read_feature_list(FEATURE_LIST_PATH)

        # keep only those that exist in features.csv
        raw_cols = [c for c in wanted if c in feat_c.columns]

        # remove any label-like columns if present
        raw_cols = [c for c in raw_cols if c.lower() not in leakage_cols]

        if len(raw_cols) == 0:
            raise RuntimeError(
                f"No columns from FEATURE_LIST were found in features.csv.\n"
                f"FEATURE_LIST={FEATURE_LIST_PATH}\n"
                f"Example wanted: {wanted[:10]}"
            )

        missing = [c for c in wanted if c not in feat_c.columns]
        raw_mode_info = (
            f"RAW_BASELINE_MODE=feature_list (FAIR) using {len(raw_cols)} cols | "
            f"missing_from_features.csv={len(missing)}"
        )

    X_raw = feat_c[raw_cols].apply(pd.to_numeric, errors="coerce").to_numpy()

    # -----------------------------
    # ExpertKG embeddings: ONLY emb_* columns
    # -----------------------------
    emb_cols = [c for c in emb_c.columns if c.startswith("emb_")]
    if len(emb_cols) == 0:
        raise KeyError("No embedding columns found. Expected emb_0, emb_1, ...")

    X_emb = emb_c[emb_cols].apply(pd.to_numeric, errors="coerce").to_numpy()

    # concat
    X_comb = np.concatenate([X_raw, X_emb], axis=1)

    # -----------------------------
    # prints
    # -----------------------------
    print(f"Subjects: {len(common_ids)}")
    bc = np.bincount(y, minlength=2)
    print(f"Class balance [0,1]: {bc}")
    print(raw_mode_info)
    print(f"Raw dims: {X_raw.shape[1]}")
    print(f"ExpertKG dims: {X_emb.shape[1]}")

    # -----------------------------
    # eval
    # -----------------------------
    eval_auc(X_raw, y, "RAW")
    eval_auc(X_emb, y, "ExpertKG(Node2Vec)")
    eval_auc(X_comb, y, "RAW + ExpertKG (concat)")


if __name__ == "__main__":
    main()
