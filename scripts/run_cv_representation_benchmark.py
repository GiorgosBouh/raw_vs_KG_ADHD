"""Leakage-free CV benchmark for representation comparison.

This script is intentionally conservative and runs end-to-end without relying on
interactive steps. It writes results to CONFIG.results_dir.

Representations currently implemented:
- raw baseline (impute + scale + logistic regression)
- PCA baseline (fit PCA on train only, then logistic regression)

Note: KG/Node2Vec evaluation can be added/expanded, but first we ensure the
pipeline runs cleanly and produces the expected result artifacts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from scripts.config import CONFIG


def _fit_predict_lr(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def run() -> None:
    results_dir = Path(CONFIG.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CONFIG.features_path, sep=";", decimal=",")
    if CONFIG.label_column not in df.columns:
        raise ValueError(f"Label column '{CONFIG.label_column}' not found in {CONFIG.features_path}")

    # Encode labels: class is "A" (positive) / "T" (negative)
    y_raw = df[CONFIG.label_column].astype(str).str.strip()
    label_map = {"A": 1, "T": 0}
    unknown = sorted(set(y_raw.unique()) - set(label_map))
    if unknown:
        raise ValueError(f"Unknown labels in {CONFIG.label_column}: {unknown}")
    y = y_raw.map(label_map).astype(int).to_numpy()

    X = df.drop(columns=[CONFIG.label_column])

    skf = StratifiedKFold(
        n_splits=CONFIG.k_folds,
        shuffle=True,
        random_state=CONFIG.random_seed,
    )

    auc_rows = []
    pred_rows = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr_df, X_te_df = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Fit imputer/scaler ONLY on train
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_tr = imputer.fit_transform(X_tr_df)
        X_te = imputer.transform(X_te_df)

        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # --- RAW baseline
        raw_prob = _fit_predict_lr(X_tr_s, y_tr, X_te_s)
        raw_auc = roc_auc_score(y_te, raw_prob)

        auc_rows.append(
            {"fold": fold, "representation": "raw", "variant": "baseline", "auc": float(raw_auc)}
        )
        for yt, ps in zip(y_te, raw_prob):
            pred_rows.append(
                {"fold": fold, "representation": "raw", "variant": "baseline", "y_true": int(yt), "y_score": float(ps)}
            )

        # --- PCA baseline (fit PCA on train only)
        pca = PCA(n_components=min(20, X_tr_s.shape[1]), random_state=CONFIG.random_seed)
        X_tr_p = pca.fit_transform(X_tr_s)
        X_te_p = pca.transform(X_te_s)

        pca_prob = _fit_predict_lr(X_tr_p, y_tr, X_te_p)
        pca_auc = roc_auc_score(y_te, pca_prob)

        auc_rows.append(
            {"fold": fold, "representation": "pca", "variant": "baseline", "auc": float(pca_auc)}
        )
        for yt, ps in zip(y_te, pca_prob):
            pred_rows.append(
                {"fold": fold, "representation": "pca", "variant": "baseline", "y_true": int(yt), "y_score": float(ps)}
            )

    auc_df = pd.DataFrame(auc_rows)
    pred_df = pd.DataFrame(pred_rows)

    auc_df.to_csv(results_dir / "auc_per_fold.csv", index=False)
    pred_df.to_csv(results_dir / "predictions.csv", index=False)

    summary = auc_df.groupby(["representation", "variant"])["auc"].agg(["mean", "std"]).reset_index()
    summary.to_csv(results_dir / "auc_summary.csv", index=False)

    print("\nDone.")
    print("Saved to:", results_dir.resolve())
    print(summary)


if __name__ == "__main__":
    run()
