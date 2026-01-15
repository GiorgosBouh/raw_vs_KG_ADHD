"""Utilities for building a richer knowledge graph without leakage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import networkx as nx


@dataclass
class TrainStats:
    feature_columns: list[str]
    medians: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]
    sorted_values: dict[str, np.ndarray]
    bin_edges: dict[str, np.ndarray | None]


def load_csv_auto(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path_obj}")
    df = pd.read_csv(path_obj, engine="python", sep=None)
    if df.shape[1] == 1 and ";" in df.columns[0]:
        df = pd.read_csv(path_obj, sep=";")
    return df


def detect_id_column(df: pd.DataFrame) -> str:
    if "ID" in df.columns:
        return "ID"
    if "id" in df.columns:
        return "id"
    return df.columns[0]


def align_features_patient(
    features_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    id_col_features: str,
    id_col_patient: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = features_df.copy()
    patient = patient_df.copy()

    features["ID"] = features[id_col_features].astype(str)
    patient["ID"] = patient[id_col_patient].astype(str)

    common_ids = sorted(set(features["ID"]).intersection(set(patient["ID"])))
    features = features[features["ID"].isin(common_ids)].copy()
    patient = patient[patient["ID"].isin(common_ids)].copy()

    features = features.sort_values("ID").reset_index(drop=True)
    patient = patient.sort_values("ID").reset_index(drop=True)
    return features, patient


def derive_feature_group(feature_name: str) -> str:
    return feature_name.split("_")[0]


def fit_train_stats(train_df_features: pd.DataFrame, n_bins: int = 5) -> TrainStats:
    feature_columns = [c for c in train_df_features.columns if c != "ID"]
    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    sorted_values: dict[str, np.ndarray] = {}
    bin_edges: dict[str, np.ndarray | None] = {}

    for feat in feature_columns:
        values = pd.to_numeric(train_df_features[feat], errors="coerce").to_numpy()
        non_nan = values[~np.isnan(values)]
        if non_nan.size == 0:
            medians[feat] = float("nan")
            means[feat] = float("nan")
            stds[feat] = float("nan")
            sorted_values[feat] = np.array([])
            bin_edges[feat] = None
            continue
        medians[feat] = float(np.nanmedian(values))
        means[feat] = float(np.nanmean(values))
        std = float(np.nanstd(values))
        stds[feat] = std if std > 0 else 1.0
        sorted_values[feat] = np.sort(non_nan)
        if n_bins > 1 and non_nan.size >= n_bins:
            edges = np.unique(np.nanquantile(non_nan, np.linspace(0, 1, n_bins + 1)))
            bin_edges[feat] = edges if edges.size > 2 else None
        else:
            bin_edges[feat] = None

    return TrainStats(
        feature_columns=feature_columns,
        medians=medians,
        means=means,
        stds=stds,
        sorted_values=sorted_values,
        bin_edges=bin_edges,
    )


def compute_feature_correlations(
    train_matrix: pd.DataFrame,
    top_k: int,
) -> list[tuple[str, str, float]]:
    if top_k <= 0:
        return []
    corr = train_matrix.corr(method="spearman")
    edges: list[tuple[str, str, float]] = []
    for feat in corr.columns:
        series = corr[feat].drop(labels=[feat])
        if series.empty:
            continue
        top = series.reindex(series.abs().sort_values(ascending=False).head(top_k).index)
        for other, rho in top.items():
            if not np.isfinite(rho):
                continue
            edges.append((feat, other, float(rho)))
    return edges


def _rank_percentile(sorted_vals: np.ndarray, value: float) -> float:
    if sorted_vals.size == 0:
        return 0.0
    return float(np.searchsorted(sorted_vals, value, side="right") / sorted_vals.size)


def _bin_index(bin_edges: np.ndarray | None, value: float) -> int | None:
    if bin_edges is None:
        return None
    idx = int(np.searchsorted(bin_edges, value, side="right") - 1)
    idx = max(0, min(idx, len(bin_edges) - 2))
    return idx


def _is_clinical(column: str) -> bool:
    name = column.upper()
    tokens = ["DX", "DISORDER", "MED", "ANXIETY", "BIPOLAR", "DEPRESS", "CLINICAL"]
    return any(token in name for token in tokens)


def subject_context_nodes(
    subject_row: pd.Series,
    train_stats: TrainStats,
    options: dict[str, Any],
    weight_mode: str = "uniform",
) -> list[tuple[str, float]]:
    nodes: list[tuple[str, float]] = []
    for feat in train_stats.feature_columns:
        val = pd.to_numeric(subject_row.get(feat, np.nan), errors="coerce")
        if np.isnan(val):
            continue
        feature_node = f"F|{feat}"
        if weight_mode == "abs_value":
            weight = float(abs(val))
        elif weight_mode == "abs_z":
            mean = train_stats.means.get(feat, 0.0)
            std = train_stats.stds.get(feat, 1.0)
            weight = float(abs((val - mean) / std)) if std != 0 else 1.0
        else:
            weight = 1.0
        nodes.append((feature_node, weight))

        if options.get("include_bins"):
            bin_idx = _bin_index(train_stats.bin_edges.get(feat), float(val))
            if bin_idx is not None:
                bin_node = f"BIN|{feat}|{bin_idx}"
                nodes.append((bin_node, 1.0))

    if options.get("include_demo"):
        demo_columns = options.get("demo_columns", [])
        demo_bins: dict[str, np.ndarray | None] = options.get("demo_bin_edges", {})
        for col in demo_columns:
            val = subject_row.get(col, np.nan)
            if pd.isna(val):
                continue
            if col in demo_bins and demo_bins[col] is not None:
                bin_idx = _bin_index(demo_bins[col], float(val))
                nodes.append((f"DEMO|{col}|BIN{bin_idx}", 1.0))
                continue
            node_value = str(val)
            if _is_clinical(col):
                nodes.append((f"CLIN|{col}|{node_value}", 1.0))
            else:
                nodes.append((f"DEMO|{col}|{node_value}", 1.0))

    return nodes


def attach_subject_node(
    graph: nx.Graph,
    subject_row: pd.Series,
    train_stats: TrainStats,
    options: dict[str, Any],
    is_test: bool = True,
) -> None:
    subject_id = str(subject_row["ID"])
    subject_node = f"S{subject_id}"
    graph.add_node(subject_node, kind="Subject", sid=subject_id, split="test" if is_test else "train")

    for feat in train_stats.feature_columns:
        val = pd.to_numeric(subject_row.get(feat, np.nan), errors="coerce")
        if np.isnan(val):
            continue
        mean = train_stats.means.get(feat, 0.0)
        std = train_stats.stds.get(feat, 1.0)
        z = float((val - mean) / std) if std != 0 else 0.0
        rank = _rank_percentile(train_stats.sorted_values.get(feat, np.array([])), float(val))
        feature_node = f"F|{feat}"
        graph.add_edge(
            subject_node,
            feature_node,
            rel="HAS_VALUE",
            raw=float(val),
            z=z,
            abs=float(abs(val)),
            rank=rank,
        )

        if options.get("include_bins"):
            bin_idx = _bin_index(train_stats.bin_edges.get(feat), float(val))
            if bin_idx is not None:
                bin_node = f"BIN|{feat}|{bin_idx}"
                graph.add_node(bin_node, kind="Bin", feature=feat, bin_id=bin_idx)
                graph.add_edge(subject_node, bin_node, rel="IN_BIN")

    if options.get("include_demo"):
        demo_columns = options.get("demo_columns", [])
        demo_bins: dict[str, np.ndarray | None] = options.get("demo_bin_edges", {})
        for col in demo_columns:
            val = subject_row.get(col, np.nan)
            if pd.isna(val):
                continue
            if col in demo_bins and demo_bins[col] is not None:
                bin_idx = _bin_index(demo_bins[col], float(val))
                node_id = f"DEMO|{col}|BIN{bin_idx}"
                graph.add_node(node_id, kind="Demographic", key=col, value=f"BIN{bin_idx}")
                graph.add_edge(subject_node, node_id, rel="HAS_DEMO")
                continue

            node_value = str(val)
            if _is_clinical(col):
                node_id = f"CLIN|{col}|{node_value}"
                graph.add_node(node_id, kind="Clinical", key=col, value=node_value)
                graph.add_edge(subject_node, node_id, rel="HAS_CLINICAL")
            else:
                node_id = f"DEMO|{col}|{node_value}"
                graph.add_node(node_id, kind="Demographic", key=col, value=node_value)
                graph.add_edge(subject_node, node_id, rel="HAS_DEMO")


def build_train_graph(
    train_rows: pd.DataFrame,
    train_stats: TrainStats,
    options: dict[str, Any],
) -> nx.Graph:
    graph = nx.Graph()

    feature_groups: dict[str, str] = {}
    for feat in train_stats.feature_columns:
        feature_node = f"F|{feat}"
        graph.add_node(feature_node, kind="Feature", name=feat)
        group = derive_feature_group(feat)
        group_node = f"G|{group}"
        if group not in feature_groups:
            graph.add_node(group_node, kind="FeatureGroup", name=group)
            feature_groups[group] = group_node
        graph.add_edge(feature_node, group_node, rel="IN_GROUP")

    for _, row in train_rows.iterrows():
        attach_subject_node(graph, row, train_stats, options, is_test=False)

    if options.get("include_corr"):
        for feat_a, feat_b, rho in options.get("feature_corr_edges", []):
            node_a = f"F|{feat_a}"
            node_b = f"F|{feat_b}"
            if node_a == node_b:
                continue
            if graph.has_edge(node_a, node_b):
                continue
            weight = abs(rho)
            if not np.isfinite(weight):
                continue
            graph.add_edge(node_a, node_b, rel="CORRELATED_WITH", rho=rho, weight=weight)

    if options.get("include_similarity"):
        for sid_a, sid_b, sim in options.get("similarity_edges", []):
            node_a = f"S{sid_a}"
            node_b = f"S{sid_b}"
            if node_a == node_b:
                continue
            graph.add_edge(node_a, node_b, rel="SIMILAR_TO", weight=float(sim))

    return graph

