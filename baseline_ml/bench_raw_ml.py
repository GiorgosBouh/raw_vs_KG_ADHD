#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def load_patient_info(csv_path: str) -> pd.DataFrame:
    # HYPERAKTIV patient_info.csv uses semicolon delimiter
    return pd.read_csv(csv_path, sep=";")


def build_feature_sets(df: pd.DataFrame):
    """
    WEAK-BY-DESIGN feature sets.
    IMPORTANT: We avoid near-label ADHD screening questionnaires (e.g., ASRS, WURS).
    """
    demo = [c for c in ["SEX", "AGE"] if c in df.columns]

    comorb = [c for c in ["BIPOLAR", "UNIPOLAR", "ANXIETY", "SUBSTANCE", "OTHER"] if c in df.columns]
    recording = [c for c in ["ACC", "ACC_DAYS", "HRV", "HRV_HOURS"] if c in df.columns]
    cpt = [c for c in ["CPT_II"] if c in df.columns]

    meds = [c for c in [
        "MED", "MED_Antidepr", "MED_Moodstab", "MED_Antipsych",
        "MED_Anxiety_Benzo", "MED_Sleep", "MED_Analgesics_Opioids", "MED_Stimulants"
    ] if c in df.columns]

    return {
        "demographics_only": demo,
        "demo_plus_comorb": demo + comorb,
        "demo_plus_recording_flags": demo + recording,
        "demo_plus_cpt": demo + cpt,
        "demo_plus_meds": demo + meds,
    }


def make_model(name: str, seed: int):
    if name == "lr":
        return LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
            min_samples_leaf=2,
        )
    raise ValueError("Model must be 'lr' or 'rf'.")


def evaluate_cv(X: pd.DataFrame, y: np.ndarray, model, n_splits: int, seed: int):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    y_true_all, y_prob_all, y_pred_all = [], [], []

    for tr, te in skf.split(X, y):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]

        pre = ColumnTransformer(
            [("num",
              Pipeline([
                  ("imp", SimpleImputer(strategy="median")),
                  ("sc", StandardScaler()),
              ]),
              list(X.columns))],
            remainder="drop",
        )

        pipe = Pipeline([
            ("pre", pre),
            ("clf", model),
        ])

        pipe.fit(Xtr, ytr)

        prob = pipe.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)

        y_true_all.append(yte)
        y_prob_all.append(prob)
        y_pred_all.append(pred)

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)
    y_pred = np.concatenate(y_pred_all)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "n": int(len(y_true)),
        "pos_rate": float(y_true.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/patient_info.csv")
    ap.add_argument("--label", default="ADHD")
    ap.add_argument("--model", choices=["lr", "rf"], default="lr")
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()

    df = load_patient_info(args.csv)

    # Keep only filtered rows if present
    if "filter_$" in df.columns:
        df = df[df["filter_$"] == 1].copy()

    if args.label not in df.columns:
        raise ValueError(f"Label '{args.label}' not in columns: {list(df.columns)}")

    y = df[args.label].astype(int).to_numpy()
    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        y = rng.permutation(y)

    feature_sets = build_feature_sets(df)
    Path("results").mkdir(exist_ok=True)

    print(f"\nCSV={args.csv}")
    print(f"Model={args.model}  splits={args.splits}  seed={args.seed}  shuffle={args.shuffle}")
    print(f"N={len(y)}  Positive rate={y.mean():.3f}\n")

    for fs_name, cols in feature_sets.items():
        if len(cols) == 0:
            print(f"[{fs_name}] SKIP (no columns found)")
            continue

        X = df[cols].copy()

        # Drop columns that are entirely missing (all-NaN)
        X = X.dropna(axis=1, how="all")

        if X.shape[1] == 0:
            print(f"[{fs_name}] SKIP (all selected columns are all-missing)")
            continue

        model = make_model(args.model, args.seed)
        m = evaluate_cv(X, y, model, args.splits, args.seed)

        print(
            f"[{fs_name}] features={X.shape[1]}  "
            f"ROC-AUC={m['roc_auc']:.3f}  PR-AUC={m['pr_auc']:.3f}  "
            f"ACC={m['accuracy']:.3f}  F1={m['f1']:.3f}  MCC={m['mcc']:.3f}"
        )


if __name__ == "__main__":
    main()
