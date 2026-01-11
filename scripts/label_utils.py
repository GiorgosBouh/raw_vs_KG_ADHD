from __future__ import annotations

import pandas as pd


def normalize_labels(series: pd.Series, positive_label_value: str) -> pd.Series:
    pos_value = str(positive_label_value).strip().upper()
    label_raw = series.copy()

    if label_raw.dtype == object:
        label_norm = label_raw.astype(str).str.strip().str.upper()
        if set(label_norm.dropna().unique()).issubset({"A", "T"}):
            mapped = label_norm.map({"A": 1, "T": 0})
            if pos_value in {"A", "T"}:
                pos_numeric = 1 if pos_value == "A" else 0
                return (mapped == pos_numeric).astype(int)
            return mapped.astype("Int64")
        label_numeric = pd.to_numeric(label_raw, errors="coerce")
    else:
        label_numeric = pd.to_numeric(label_raw, errors="coerce")

    try:
        pos_numeric = float(pos_value)
    except ValueError:
        pos_numeric = None

    if pos_numeric is not None:
        return (label_numeric.astype(float) == pos_numeric).astype(int)

    return label_numeric.astype("Int64")
