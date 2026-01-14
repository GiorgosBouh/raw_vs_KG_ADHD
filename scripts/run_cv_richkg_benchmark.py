"""Repeated CV benchmark for raw, rich KG, and concatenated representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from scripts.kg_rich_builder import (
    TrainStats,
    align_features_patient,
    compute_feature_correlations,
    detect_id_column,
    fit_train_stats,
    load_csv_auto,
    build_train_graph,
    attach_subject_node,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-free rich KG benchmark.")
    parser.add_argument("--features", required=True, help="Path to features.csv")
    parser.add_argument("--patient", required=True, help="Path to patient_info.csv")
    parser.add_argument("--label-column", required=True, help="Label column in patient file")
    parser.add_argument("--positive-label", default="1", help="Positive label value")
    parser.add_argument("--repeats", type=int, default=10, help="Number of repeats")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--kg-variant", default="rich", choices=[
        "bipartite_only",
        "bins",
        "bins_corr",
        "bins_demo",
        "rich",
        "rich_with_similarity",
    ])
    parser.add_argument("--bins", type=int, default=5, help="Number of quantile bins")
    parser.add_argument("--corr-topk", type=int, default=5, help="Top-K feature correlations")
    parser.add_argument("--knn-k", type=int, default=5, help="K for subject similarity")
    parser.add_argument("--node2vec-dim", type=int, default=64, help="Node2Vec dimensions")
    parser.add_argument("--walk-length", type=int, default=20, help="Node2Vec walk length")
    parser.add_argument("--num-walks", type=int, default=50, help="Node2Vec num walks")
    parser.add_argument("--window", type=int, default=10, help="Node2Vec window")
    parser.add_argument("--p", type=float, default=1.0, help="Node2Vec p")
    parser.add_argument("--q", type=float, default=1.0, help="Node2Vec q")
    parser.add_argument("--C", type=float, default=1.0, help="LogisticRegression C")
    parser.add_argument("--permutations", type=int, default=200, help="Permutation test runs")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    return parser.parse_args()


def _normalize_label(values: pd.Series, positive_label: str) -> pd.Series:
    pos_value = str(positive_label).strip().upper()
    if values.dtype == object:
        normalized = values.astype(str).str.strip().str.upper()
        if set(normalized.dropna().unique()).issubset({"A", "T"}):
            mapped = normalized.map({"A": 1, "T": 0})
            if pos_value in {"A", "T"}:
                target = 1 if pos_value == "A" else 0
                return (mapped == target).astype(int)
            return mapped.astype(int)
        numeric = pd.to_numeric(values, errors="coerce")
        if pos_value.isdigit():
            return (numeric == float(pos_value)).astype(int)
        return numeric.astype(float)
    numeric = pd.to_numeric(values, errors="coerce")
    if pos_value.isdigit():
        return (numeric == float(pos_value)).astype(int)
    return numeric.astype(float)


def _print_sanity(df: pd.DataFrame, name: str) -> None:
    if df.shape[1] <= 5:
        raise ValueError(f"{name} has too few columns after parsing: {df.shape[1]}")
    print(f"{name} first 5 columns: {list(df.columns[:5])}")
    print(f"{name} shape: {df.shape}")
    print(df.head(2))


def _prepare_features(
    features_df: pd.DataFrame,
    id_column: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    features = features_df.copy()
    features["ID"] = features[id_column].astype(str)
    feature_cols = [c for c in features.columns if c != id_column and c != "ID"]
    features[feature_cols] = features[feature_cols].apply(pd.to_numeric, errors="coerce")
    dropped = [c for c in feature_cols if features[c].isna().all()]
    if dropped:
        features = features.drop(columns=dropped)
    final_cols = [c for c in features.columns if c != "ID"]
    return features, final_cols, dropped


def _impute_dataframe(df: pd.DataFrame, stats: TrainStats | None = None) -> pd.DataFrame:
    if stats is None:
        imputer = SimpleImputer(strategy="median")
        imputed = imputer.fit_transform(df)
        return pd.DataFrame(imputed, columns=df.columns, index=df.index)
    values = df.copy()
    for col in df.columns:
        median = stats.medians.get(col, np.nan)
        values[col] = values[col].fillna(median)
    return values


def _prepare_demo_columns(patient_df: pd.DataFrame, label_column: str) -> list[str]:
    return [c for c in patient_df.columns if c not in {"ID", label_column}]


def _demo_bin_edges(train_patient: pd.DataFrame, demo_columns: list[str], n_bins: int) -> dict[str, np.ndarray | None]:
    edges: dict[str, np.ndarray | None] = {}
    for col in demo_columns:
        numeric = pd.to_numeric(train_patient[col], errors="coerce")
        non_nan = numeric.dropna()
        if non_nan.empty:
            edges[col] = None
            continue
        if non_nan.nunique() > n_bins:
            values = np.unique(np.nanquantile(non_nan, np.linspace(0, 1, n_bins + 1)))
            edges[col] = values if values.size > 2 else None
        else:
            edges[col] = None
    return edges


def _build_similarity_edges(train_ids: np.ndarray, train_matrix: np.ndarray, k: int) -> list[tuple[str, str, float]]:
    if k <= 0:
        return []
    norms = np.linalg.norm(train_matrix, axis=1) + 1e-9
    normalized = train_matrix / norms[:, None]
    sim = normalized @ normalized.T
    edges: list[tuple[str, str, float]] = []
    for i in range(sim.shape[0]):
        sims = sim[i].copy()
        sims[i] = -np.inf
        k_eff = min(k, sim.shape[0] - 1)
        if k_eff <= 0:
            continue
        idx = np.argpartition(-sims, k_eff - 1)[:k_eff]
        idx = idx[np.argsort(-sims[idx])]
        for j in idx:
            edges.append((str(train_ids[i]), str(train_ids[j]), float(sims[j])))
    return edges


def _fit_node2vec(graph: nx.Graph, seed: int, args: argparse.Namespace) -> Node2Vec:
    node2vec = Node2Vec(
        graph,
        dimensions=args.node2vec_dim,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        p=args.p,
        q=args.q,
        workers=1,
        seed=seed,
        quiet=True,
    )
    model = node2vec.fit(window=args.window, min_count=1, batch_words=128, seed=seed)
    return model


def _embedding_dataframe(model: Node2Vec, subject_ids: list[str]) -> pd.DataFrame:
    embeddings = []
    for sid in subject_ids:
        node = f"S{sid}"
        if node not in model.wv:
            embeddings.append([np.nan] * model.wv.vector_size)
        else:
            embeddings.append(model.wv[node].tolist())
    emb_df = pd.DataFrame(embeddings)
    emb_df.insert(0, "ID", subject_ids)
    return emb_df


def _summary_auc(auc_values: np.ndarray, label: str) -> dict[str, Any]:
    mean = float(np.mean(auc_values))
    std = float(np.std(auc_values, ddof=1)) if auc_values.size > 1 else 0.0
    ci95 = 1.96 * std / np.sqrt(max(1, auc_values.size))
    return {"model": label, "mean_auc": mean, "std_auc": std, "ci95": ci95}


def _graph_kind_counts(graph: nx.Graph) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        kind = data.get("kind", "Unknown")
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _sanitize_graph_weights(graph: nx.Graph, default_weight: float = 1.0) -> None:
    for _, _, data in graph.edges(data=True):
        weight = data.get("weight", default_weight)
        if not np.isfinite(weight):
            weight = default_weight
        data["weight"] = float(weight)


def _bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if values.size == 0:
        return 0.0, 0.0
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        means.append(float(np.mean(sample)))
    lower = float(np.percentile(means, 2.5))
    upper = float(np.percentile(means, 97.5))
    return lower, upper


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    features_df = load_csv_auto(args.features)
    patient_df = load_csv_auto(args.patient)

    _print_sanity(features_df, "features")
    _print_sanity(patient_df, "patient")

    id_col_features = detect_id_column(features_df)
    id_col_patient = detect_id_column(patient_df)

    features_df, patient_df = align_features_patient(features_df, patient_df, id_col_features, id_col_patient)

    n_features_rows = features_df.shape[0]
    n_patient_rows = patient_df.shape[0]
    intersection = len(set(features_df["ID"]).intersection(set(patient_df["ID"])))

    if intersection < 20:
        raise ValueError(f"Intersection too small: {intersection} (<20)")

    print(
        "Counts:",
        {
            "n_features_rows": n_features_rows,
            "n_patient_rows": n_patient_rows,
            "n_intersection": intersection,
            "n_final_aligned": features_df.shape[0],
        },
    )

    patient_df["label"] = _normalize_label(patient_df[args.label_column], args.positive_label)
    patient_df = patient_df.dropna(subset=["label"]).copy()

    features_df, patient_df = align_features_patient(features_df, patient_df, "ID", "ID")
    labels = patient_df.set_index("ID")["label"].astype(int)

    features_df, feature_columns, dropped_columns = _prepare_features(features_df, "ID")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    (results_dir / "splits").mkdir(exist_ok=True)

    with open(results_dir / "dropped_non_numeric_columns.json", "w", encoding="utf-8") as handle:
        json.dump({"dropped_all_nan": dropped_columns}, handle, indent=2)

    feature_columns_used: dict[str, Any] = {
        "all_columns": feature_columns,
        "dropped_all_nan": dropped_columns,
        "folds": {},
    }

    rskf = RepeatedStratifiedKFold(
        n_splits=args.folds,
        n_repeats=args.repeats,
        random_state=args.seed,
    )

    auc_records = []
    pred_records = []
    delta_records = []
    split_records = []

    perm_delta_kg_sums = np.zeros(args.permutations, dtype=float)
    perm_delta_concat_sums = np.zeros(args.permutations, dtype=float)
    perm_auc_raw_sums = np.zeros(args.permutations, dtype=float)
    perm_auc_kg_sums = np.zeros(args.permutations, dtype=float)
    perm_auc_concat_sums = np.zeros(args.permutations, dtype=float)

    raw_matrix = features_df.set_index("ID")[feature_columns]
    id_list = raw_matrix.index.astype(str).tolist()
    y_all = labels.loc[id_list].to_numpy()

    for split_idx, (train_idx, test_idx) in enumerate(rskf.split(raw_matrix, y_all)):
        repeat = split_idx // args.folds
        fold = split_idx % args.folds
        fold_id = f"repeat{repeat}_fold{fold}"

        train_ids = [id_list[i] for i in train_idx]
        test_ids = [id_list[i] for i in test_idx]

        train_labels = y_all[train_idx]
        test_labels = y_all[test_idx]

        split_df = pd.DataFrame(
            {
                "train_ids": pd.Series(train_ids),
                "test_ids": pd.Series(test_ids),
            }
        )
        split_df.to_csv(results_dir / "splits" / f"split_{fold_id}.csv", index=False)

        split_counts = {
            "repeat": repeat,
            "fold": fold,
            "train_pos": int(train_labels.sum()),
            "train_neg": int(len(train_labels) - train_labels.sum()),
            "test_pos": int(test_labels.sum()),
            "test_neg": int(len(test_labels) - test_labels.sum()),
        }
        split_records.append(split_counts)
        feature_columns_used["folds"][fold_id] = feature_columns

        train_features = raw_matrix.loc[train_ids]
        test_features = raw_matrix.loc[test_ids]

        train_patient = patient_df[patient_df["ID"].isin(train_ids)].copy()
        test_patient = patient_df[patient_df["ID"].isin(test_ids)].copy()

        train_stats = fit_train_stats(train_features.reset_index(), n_bins=args.bins)

        imputed_train = _impute_dataframe(train_features, train_stats)
        imputed_test = _impute_dataframe(test_features, train_stats)
        imputed_train_df = imputed_train.reset_index(drop=True)
        imputed_train_df["ID"] = train_ids
        imputed_test_df = imputed_test.reset_index(drop=True)
        imputed_test_df["ID"] = test_ids

        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(imputed_train)
        scaled_test = scaler.transform(imputed_test)

        raw_model = LogisticRegression(
            solver="liblinear",
            C=args.C,
            max_iter=5000,
            random_state=args.seed,
        )
        raw_model.fit(scaled_train, train_labels)
        prob_raw = raw_model.predict_proba(scaled_test)[:, 1]
        auc_raw = roc_auc_score(test_labels, prob_raw)

        demo_columns = _prepare_demo_columns(patient_df, "label")
        demo_bin_edges = _demo_bin_edges(train_patient, demo_columns, args.bins)

        train_corr_edges = compute_feature_correlations(imputed_train, args.corr_topk)

        similarity_edges = []
        if args.kg_variant in {"rich_with_similarity"}:
            similarity_edges = _build_similarity_edges(train_ids, scaled_train, args.knn_k)

        options = {
            "include_bins": args.kg_variant in {"bins", "bins_corr", "bins_demo", "rich", "rich_with_similarity"},
            "include_corr": args.kg_variant in {"bins_corr", "rich", "rich_with_similarity"},
            "include_demo": args.kg_variant in {"bins_demo", "rich", "rich_with_similarity"},
            "include_similarity": args.kg_variant in {"rich_with_similarity"},
            "feature_corr_edges": train_corr_edges,
            "similarity_edges": similarity_edges,
            "demo_columns": demo_columns,
            "demo_bin_edges": demo_bin_edges,
        }

        train_graph_rows = imputed_train_df.merge(
            train_patient[["ID"] + demo_columns],
            on="ID",
            how="left",
        )
        train_graph = build_train_graph(train_graph_rows, train_stats, options)

        test_graph_rows = imputed_test_df.merge(
            test_patient[["ID"] + demo_columns],
            on="ID",
            how="left",
        )
        for _, row in test_graph_rows.iterrows():
            attach_subject_node(train_graph, row, train_stats, options, is_test=True)

        _sanitize_graph_weights(train_graph)
        kind_counts = _graph_kind_counts(train_graph)
        print(
            f"[Repeat {repeat} Fold {fold}] Graph nodes={train_graph.number_of_nodes()} "
            f"edges={train_graph.number_of_edges()} kinds={kind_counts}"
        )

        n2v_model = _fit_node2vec(train_graph, args.seed + split_idx, args)
        emb_train_df = _embedding_dataframe(n2v_model, train_ids)
        emb_test_df = _embedding_dataframe(n2v_model, test_ids)

        emb_train_path = results_dir / f"embeddings_richkg_{args.kg_variant}_fold{fold_id}_train.csv"
        emb_test_path = results_dir / f"embeddings_richkg_{args.kg_variant}_fold{fold_id}_test.csv"
        emb_train_df.to_csv(emb_train_path, index=False)
        emb_test_df.to_csv(emb_test_path, index=False)

        emb_train = emb_train_df.drop(columns=["ID"]).to_numpy()
        emb_test = emb_test_df.drop(columns=["ID"]).to_numpy()

        emb_scaler = StandardScaler()
        emb_train_scaled = emb_scaler.fit_transform(emb_train)
        emb_test_scaled = emb_scaler.transform(emb_test)

        kg_model = LogisticRegression(
            solver="liblinear",
            C=args.C,
            max_iter=5000,
            random_state=args.seed,
        )
        kg_model.fit(emb_train_scaled, train_labels)
        prob_kg = kg_model.predict_proba(emb_test_scaled)[:, 1]
        auc_kg = roc_auc_score(test_labels, prob_kg)

        concat_train = np.hstack([scaled_train, emb_train_scaled])
        concat_test = np.hstack([scaled_test, emb_test_scaled])

        concat_model = LogisticRegression(
            solver="liblinear",
            C=args.C,
            max_iter=5000,
            random_state=args.seed,
        )
        concat_model.fit(concat_train, train_labels)
        prob_concat = concat_model.predict_proba(concat_test)[:, 1]
        auc_concat = roc_auc_score(test_labels, prob_concat)

        auc_records.append(
            {
                "repeat": repeat,
                "fold": fold,
                "auc_raw": auc_raw,
                "auc_kg": auc_kg,
                "auc_concat": auc_concat,
            }
        )

        pred_records.append(
            pd.DataFrame(
                {
                    "id": test_ids,
                    "y_true": test_labels,
                    "prob_raw": prob_raw,
                    "prob_kg": prob_kg,
                    "prob_concat": prob_concat,
                    "fold": fold,
                    "repeat": repeat,
                }
            )
        )

        delta_records.append(
            {
                "repeat": repeat,
                "fold": fold,
                "delta_kg": auc_kg - auc_raw,
                "delta_concat": auc_concat - auc_raw,
            }
        )

        for perm_idx in range(args.permutations):
            perm_train = rng.permutation(train_labels)
            perm_test = rng.permutation(test_labels)

            raw_model.fit(scaled_train, perm_train)
            perm_prob_raw = raw_model.predict_proba(scaled_test)[:, 1]
            perm_auc_raw = roc_auc_score(perm_test, perm_prob_raw)

            kg_model.fit(emb_train_scaled, perm_train)
            perm_prob_kg = kg_model.predict_proba(emb_test_scaled)[:, 1]
            perm_auc_kg = roc_auc_score(perm_test, perm_prob_kg)

            concat_model.fit(concat_train, perm_train)
            perm_prob_concat = concat_model.predict_proba(concat_test)[:, 1]
            perm_auc_concat = roc_auc_score(perm_test, perm_prob_concat)

            perm_delta_kg_sums[perm_idx] += perm_auc_kg - perm_auc_raw
            perm_delta_concat_sums[perm_idx] += perm_auc_concat - perm_auc_raw
            perm_auc_raw_sums[perm_idx] += perm_auc_raw
            perm_auc_kg_sums[perm_idx] += perm_auc_kg
            perm_auc_concat_sums[perm_idx] += perm_auc_concat

        print(f"[Repeat {repeat} Fold {fold}] AUC raw={auc_raw:.3f} kg={auc_kg:.3f} concat={auc_concat:.3f}")
        print(f"[Repeat {repeat} Fold {fold}] class counts {split_counts}")

    auc_df = pd.DataFrame(auc_records)
    auc_df.to_csv(results_dir / "auc_per_fold.csv", index=False)

    preds_df = pd.concat(pred_records, ignore_index=True)
    preds_df.to_csv(results_dir / "predictions_all.csv", index=False)

    delta_df = pd.DataFrame(delta_records)
    delta_df.to_csv(results_dir / "delta_per_fold.csv", index=False)

    split_df = pd.DataFrame(split_records)
    split_df.to_csv(results_dir / "splits_summary.csv", index=False)

    auc_summary = pd.DataFrame(
        [
            _summary_auc(auc_df["auc_raw"].to_numpy(), "raw"),
            _summary_auc(auc_df["auc_kg"].to_numpy(), "kg"),
            _summary_auc(auc_df["auc_concat"].to_numpy(), "concat"),
        ]
    )
    auc_summary.to_csv(results_dir / "auc_summary.csv", index=False)

    delta_kg = delta_df["delta_kg"].to_numpy()
    delta_concat = delta_df["delta_concat"].to_numpy()
    delta_kg_ci = _bootstrap_ci(delta_kg, args.seed)
    delta_concat_ci = _bootstrap_ci(delta_concat, args.seed + 1)
    delta_summary = pd.DataFrame(
        [
            {
                "metric": "delta_kg",
                "mean": float(np.mean(delta_kg)),
                "std": float(np.std(delta_kg, ddof=1)) if delta_kg.size > 1 else 0.0,
                "ci_low": delta_kg_ci[0],
                "ci_high": delta_kg_ci[1],
            },
            {
                "metric": "delta_concat",
                "mean": float(np.mean(delta_concat)),
                "std": float(np.std(delta_concat, ddof=1)) if delta_concat.size > 1 else 0.0,
                "ci_low": delta_concat_ci[0],
                "ci_high": delta_concat_ci[1],
            },
        ]
    )
    delta_summary.to_csv(results_dir / "delta_summary.csv", index=False)

    observed_delta_kg = float(np.mean(delta_kg))
    observed_delta_concat = float(np.mean(delta_concat))
    split_count = len(delta_records)
    perm_mean_delta_kg = perm_delta_kg_sums / max(1, split_count)
    perm_mean_delta_concat = perm_delta_concat_sums / max(1, split_count)
    perm_mean_auc_raw = perm_auc_raw_sums / max(1, split_count)
    perm_mean_auc_kg = perm_auc_kg_sums / max(1, split_count)
    perm_mean_auc_concat = perm_auc_concat_sums / max(1, split_count)
    pval_kg = float(np.mean(perm_mean_delta_kg >= observed_delta_kg))
    pval_concat = float(np.mean(perm_mean_delta_concat >= observed_delta_concat))

    permutation_summary = {
        "observed_delta_kg": observed_delta_kg,
        "observed_delta_concat": observed_delta_concat,
        "p_value_kg": pval_kg,
        "p_value_concat": pval_concat,
        "permutations": args.permutations,
        "perm_mean_auc_raw_mean": float(np.mean(perm_mean_auc_raw)),
        "perm_mean_auc_kg_mean": float(np.mean(perm_mean_auc_kg)),
        "perm_mean_auc_concat_mean": float(np.mean(perm_mean_auc_concat)),
    }
    with open(results_dir / "permutation_test_delta.json", "w", encoding="utf-8") as handle:
        json.dump(permutation_summary, handle, indent=2)

    if abs(float(np.mean(perm_mean_delta_kg))) > 0.05 or abs(float(np.mean(perm_mean_delta_concat))) > 0.05:
        print("WARNING: permutation deltas are not centered near 0. Check leakage or setup.")
    if (
        abs(float(np.mean(perm_mean_auc_raw)) - 0.5) > 0.05
        or abs(float(np.mean(perm_mean_auc_kg)) - 0.5) > 0.05
        or abs(float(np.mean(perm_mean_auc_concat)) - 0.5) > 0.05
    ):
        print("WARNING: permutation AUCs are far from 0.5. Check leakage or setup.")

    config_payload = {
        "args": vars(args),
        "feature_columns_used": feature_columns,
        "dropped_all_nan": dropped_columns,
    }
    with open(results_dir / "config_used.json", "w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)

    with open(results_dir / "feature_columns_used.json", "w", encoding="utf-8") as handle:
        json.dump(feature_columns_used, handle, indent=2)


if __name__ == "__main__":
    main()
