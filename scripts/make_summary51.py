"""Generate summary51.csv from a rich-KG results directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create summary51.csv from rich-KG results.")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--out", default="summary51.csv", help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    auc_path = results_dir / "auc_summary.csv"
    config_path = results_dir / "config_used.json"

    if not auc_path.exists():
        raise FileNotFoundError(f"Missing {auc_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")

    auc_df = pd.read_csv(auc_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    feature_list_path = config.get("feature_list_path")
    n_features_used = config.get("n_features_used")
    required_cols = {"model", "mean_auc", "std_auc", "ci95"}
    if not required_cols.issubset(set(auc_df.columns)):
        raise ValueError(
            f"auc_summary.csv missing columns: {sorted(required_cols - set(auc_df.columns))}"
        )

    out_df = auc_df[["model", "mean_auc", "std_auc", "ci95"]].copy()
    out_df["feature_list_path"] = feature_list_path
    out_df["n_features_used"] = n_features_used

    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
