"""Audit leakage, split integrity, and overfitting signals for rich-KG runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit leakage and split integrity.")
    parser.add_argument("--results-dir", default="results", help="Results directory to audit")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _audit_splits(results_dir: Path) -> list[str]:
    warnings: list[str] = []
    splits_dir = results_dir / "splits"
    if not splits_dir.exists():
        warnings.append("Missing splits/ directory; cannot verify split integrity.")
        return warnings

    for split_path in sorted(splits_dir.glob("split_*.csv")):
        df = pd.read_csv(split_path)
        train_ids = df["train_ids"].dropna().astype(str).tolist()
        test_ids = df["test_ids"].dropna().astype(str).tolist()
        if set(train_ids).intersection(test_ids):
            warnings.append(f"Overlap detected in {split_path.name}.")
        if len(train_ids) != len(set(train_ids)):
            warnings.append(f"Duplicate train IDs in {split_path.name}.")
        if len(test_ids) != len(set(test_ids)):
            warnings.append(f"Duplicate test IDs in {split_path.name}.")
    return warnings


def _audit_auc(results_dir: Path) -> list[str]:
    warnings: list[str] = []
    auc_path = results_dir / "auc_per_fold.csv"
    if not auc_path.exists():
        warnings.append("Missing auc_per_fold.csv; cannot inspect AUC distribution.")
        return warnings

    auc_df = pd.read_csv(auc_path)
    for col in ("auc_raw", "auc_kg", "auc_concat"):
        if col in auc_df.columns:
            extreme = auc_df[auc_df[col] > 0.9]
            if not extreme.empty:
                warnings.append(f"Extreme AUCs (>0.9) detected for {col}: {len(extreme)} folds.")
        else:
            warnings.append(f"Missing {col} in auc_per_fold.csv; cannot audit extremes.")
    return warnings


def _audit_split_sizes(results_dir: Path) -> list[str]:
    warnings: list[str] = []
    split_summary_path = results_dir / "splits_summary.csv"
    if not split_summary_path.exists():
        warnings.append("Missing splits_summary.csv; cannot audit class counts.")
        return warnings

    split_df = pd.read_csv(split_summary_path)
    for _, row in split_df.iterrows():
        test_total = int(row.get("test_pos", 0) + row.get("test_neg", 0))
        if test_total < 20:
            warnings.append(
                f"Small test set detected (n={test_total}) for repeat={row.get('repeat')} fold={row.get('fold')}."
            )
    return warnings


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    config_path = results_dir / "config_used.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")

    config = _load_json(config_path)
    run_args = config.get("args", {})
    feature_list_path = config.get("feature_list_path")
    evaluation_mode = run_args.get("evaluation_mode")

    print("Audit summary:")
    print(f"- results_dir: {results_dir}")
    print(f"- evaluation_mode: {evaluation_mode}")
    print(f"- feature_list_path: {feature_list_path}")
    print(f"- n_features_used: {config.get('n_features_used')}")
    print(f"- n_features_total_before_filter: {config.get('n_features_total_before_filter')}")
    print(f"- n_features_after_featurelist_before_drop: {config.get('n_features_after_featurelist_before_drop')}")

    warnings = []
    if evaluation_mode == "transductive":
        warnings.append("Evaluation mode is transductive; test nodes may be attached to the graph.")
    warnings.extend(_audit_splits(results_dir))
    warnings.extend(_audit_auc(results_dir))
    warnings.extend(_audit_split_sizes(results_dir))

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("\nNo warnings detected.")


if __name__ == "__main__":
    main()
