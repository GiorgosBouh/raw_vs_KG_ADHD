"""Leakage-free CV benchmark for raw and KG representations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

from dataclasses import asdict

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from scripts.config import CONFIG, Config


def read_csv_auto(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(
            f"File not found: {path_obj}. "
            "Set --features/--patient flags or update scripts/config.py with the correct paths."
        )
    df = pd.read_csv(path_obj, engine="python", sep=None)
    if df.shape[1] == 1 and ";" in df.columns[0]:
        df = pd.read_csv(path_obj, sep=";")
    return df


def read_feature_list(path: str) -> list[str]:
    items: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            items.append(entry)
    seen = set()
    ordered = []
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def expert_mapping(feature_name: str) -> tuple[str, str, str]:
    name = feature_name.lower()
    if "entropy" in name or "lempel_ziv" in name:
        prop = "Entropy/Complexity"
        beh = "Irregularity"
    elif "cv" in name or "__std" in name or "std" in name:
        prop = "Variability"
        beh = "Instability"
    else:
        prop = "Temporal"
        beh = "Rhythmicity"

    if ("__mean" in feature_name) or ("\"mean\"" in feature_name) or (
        "feature_\"mean\"" in feature_name
    ) or ("f_agg_\"mean\"" in feature_name):
        tmp = "Mean"
    elif "__std" in name or "std" in name:
        tmp = "Dispersion"
    else:
        tmp = "Temporal"

    return prop, beh, tmp


def align_inputs(
    features: pd.DataFrame,
    patient: pd.DataFrame,
    label_column: str,
    positive_label_value: str,
) -> tuple[pd.DataFrame, pd.Series]:
    for df in (features, patient):
        df.columns = [c.strip() for c in df.columns]

    if "ID" not in features.columns or "ID" not in patient.columns:
        raise KeyError("Both features.csv and patient_info.csv must include an 'ID' column.")
    if label_column not in patient.columns:
        raise KeyError(f"patient_info.csv missing label column '{label_column}'.")

    features = features.copy()
    patient = patient.copy()

    if "filter_$" in patient.columns:
        patient = patient[patient["filter_$"] == 1].copy()

    label_raw = patient[label_column]
    pos_value = str(positive_label_value).strip().upper()

    if label_raw.dtype == object:
        label_norm = label_raw.astype(str).str.strip().str.upper()
        if set(label_norm.dropna().unique()).issubset({"A", "T"}):
            mapped = label_norm.map({"A": 1, "T": 0})
            if pos_value in {"A", "T"}:
                pos_numeric = 1 if pos_value == "A" else 0
                patient[label_column] = (mapped == pos_numeric).astype(int)
            else:
                patient[label_column] = mapped
        else:
            patient[label_column] = pd.to_numeric(label_raw, errors="coerce")
    else:
        patient[label_column] = pd.to_numeric(label_raw, errors="coerce")

    patient = patient.dropna(subset=[label_column])
    try:
        pos_numeric = float(pos_value)
    except ValueError:
        pos_numeric = None

    if pos_numeric is not None:
        patient[label_column] = (patient[label_column].astype(float) == pos_numeric).astype(int)

    features["ID"] = pd.to_numeric(features["ID"], errors="coerce").astype("Int64")
    patient["ID"] = pd.to_numeric(patient["ID"], errors="coerce").astype("Int64")

    common_ids = sorted(set(features["ID"].dropna().astype(int)).intersection(
        set(patient["ID"].dropna().astype(int))
    ))
    if not common_ids:
        raise ValueError("No overlapping IDs between features and patient_info.")

    features = features[features["ID"].astype(int).isin(common_ids)].copy()
    patient = patient[patient["ID"].astype(int).isin(common_ids)].copy()

    features = features.sort_values("ID").reset_index(drop=True)
    patient = patient.sort_values("ID").reset_index(drop=True)

    labels = patient.set_index("ID")[label_column].astype(int)
    return features, labels


def build_similarity_edges(train_ids: np.ndarray, train_matrix: np.ndarray, k: int) -> list[tuple[int, int, float]]:
    norms = np.linalg.norm(train_matrix, axis=1) + 1e-9
    train_normed = train_matrix / norms[:, None]
    sim = train_normed @ train_normed.T

    edges: list[tuple[int, int, float]] = []
    n = sim.shape[0]
    for i in range(n):
        sims = sim[i].copy()
        sims[i] = -np.inf
        k_eff = min(k, n - 1)
        if k_eff <= 0:
            continue
        idx = np.argpartition(-sims, k_eff - 1)[:k_eff]
        idx = idx[np.argsort(-sims[idx])]
        for j in idx:
            edges.append((int(train_ids[i]), int(train_ids[j]), float(sims[j])))
    return edges


def build_similarity_graph(train_ids: np.ndarray, similarity_edges: list[tuple[int, int, float]]) -> nx.Graph:
    graph = nx.Graph()
    for sid in train_ids:
        graph.add_node(f"S{sid}", kind="Subject", sid=int(sid))
    for a, b, cos in similarity_edges:
        graph.add_edge(f"S{a}", f"S{b}", rel="SIMILAR_TO", weight=float(cos))
    return graph


def build_train_graph(
    train_df: pd.DataFrame,
    feature_names: list[str],
    include_similarity: bool,
    similarity_edges: list[tuple[int, int, float]],
) -> nx.Graph:
    graph = nx.Graph()

    for sid in train_df["ID"].astype(int).tolist():
        graph.add_node(f"S{sid}", kind="Subject", sid=int(sid))

    feature_nodes = {}
    property_nodes = {}
    temporal_nodes = {}
    behavior_nodes = {}

    for feat in feature_names:
        prop, beh, tmp = expert_mapping(feat)
        fn = f"F|{feat}"
        pn = f"P|{prop}"
        tn = f"T|{tmp}"
        bn = f"B|{beh}"

        feature_nodes[feat] = fn
        property_nodes[prop] = pn
        temporal_nodes[tmp] = tn
        behavior_nodes[beh] = bn

        graph.add_node(fn, kind="Feature", name=feat)
        graph.add_node(pn, kind="Property", name=prop)
        graph.add_node(tn, kind="Temporal", name=tmp)
        graph.add_node(bn, kind="Behavior", name=beh)

        graph.add_edge(fn, pn, rel="HAS_PROPERTY", weight=1.0)
        graph.add_edge(fn, tn, rel="HAS_TEMPORAL_PATTERN", weight=1.0)
        graph.add_edge(pn, bn, rel="INDICATES_BEHAVIOR", weight=1.0)

    for record in train_df[["ID"] + feature_names].to_dict(orient="records"):
        sid = int(record["ID"])
        sn = f"S{sid}"
        for feat in feature_names:
            val = record.get(feat)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            graph.add_edge(sn, feature_nodes[feat], rel="HAS_FEATURE", weight=1.0)

    if include_similarity:
        for a, b, cos in similarity_edges:
            graph.add_edge(f"S{a}", f"S{b}", rel="SIMILAR_TO", weight=float(cos))

    return graph


def fit_node2vec(graph: nx.Graph) -> Node2Vec:
    node2vec = Node2Vec(
        graph,
        dimensions=CONFIG.node2vec_dimensions,
        walk_length=CONFIG.node2vec_walk_length,
        num_walks=CONFIG.node2vec_num_walks,
        p=CONFIG.node2vec_p,
        q=CONFIG.node2vec_q,
        workers=CONFIG.node2vec_workers,
        seed=CONFIG.random_seed,
        weight_key="weight",
    )
    return node2vec.fit(window=CONFIG.node2vec_window, min_count=1, batch_words=64, seed=CONFIG.random_seed)


def project_test_embeddings(
    test_matrix: np.ndarray,
    train_matrix: np.ndarray,
    train_embeddings: np.ndarray,
    k: int,
) -> np.ndarray:
    # Inductive projection: map test subjects onto the training embedding space
    # via cosine-weighted kNN over training subjects only.
    train_norm = train_matrix / (np.linalg.norm(train_matrix, axis=1) + 1e-9)[:, None]
    test_norm = test_matrix / (np.linalg.norm(test_matrix, axis=1) + 1e-9)[:, None]
    sims = test_norm @ train_norm.T

    projected = np.zeros((test_matrix.shape[0], train_embeddings.shape[1]), dtype=float)
    for i in range(sims.shape[0]):
        row = sims[i]
        k_eff = min(k, row.shape[0])
        idx = np.argpartition(-row, k_eff - 1)[:k_eff]
        idx = idx[np.argsort(-row[idx])]
        weights = np.maximum(row[idx], 0.0)
        if weights.sum() == 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / weights.sum()
        projected[i] = np.dot(weights, train_embeddings[idx])
    return projected


def bootstrap_auc(y_true: np.ndarray, y_score: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(roc_auc_score(y_true[idx], y_score[idx]))
    if not values:
        return float("nan"), float("nan")
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def proba_positive(
    clf: LogisticRegression,
    X: np.ndarray,
    y_test: np.ndarray | None = None,
    debug: bool = False,
    tag: str = "",
) -> np.ndarray:
    proba = clf.predict_proba(X)
    pos_col = list(clf.classes_).index(1)
    if debug:
        y_unique = set(y_test) if y_test is not None else None
        print("classes:", clf.classes_, "pos_col:", pos_col, "y_test unique:", y_unique, "tag:", tag)
    return proba[:, pos_col]


def run(config: Config):
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "config_used.json").write_text(
        json.dumps(asdict(config), indent=2, default=list),
        encoding="utf-8",
    )

    features = read_csv_auto(config.features_path)
    patient = read_csv_auto(config.patient_path)

    features, labels = align_inputs(features, patient, config.label_column, config.positive_label_value)

    leakage_cols = {
        config.label_column.lower(),
        "adhd",
        "label",
        "class",
        "target",
    }
    raw_features = read_feature_list(config.raw_feature_list)
    raw_features = [
        c for c in raw_features if c in features.columns and c.lower() not in leakage_cols
    ]
    if not raw_features:
        raise ValueError("No raw features found in features.csv that match the feature list.")

    similarity_features = read_feature_list(config.similarity_feature_list)
    similarity_features = [
        c for c in similarity_features if c in features.columns and c.lower() not in leakage_cols
    ]
    if not similarity_features:
        raise ValueError("No similarity features found in features.csv that match the similarity feature list.")

    features_full = features.copy()

    features = features_full[["ID"] + raw_features].copy()
    for col in raw_features:
        features[col] = pd.to_numeric(features[col], errors="coerce")

    similarity_df = features_full[["ID"] + similarity_features].copy()
    for col in similarity_features:
        similarity_df[col] = pd.to_numeric(similarity_df[col], errors="coerce")

    ids = features["ID"].astype(int).to_numpy()
    y = labels.loc[ids].to_numpy()

    k_values = list(config.k_nn_values) if config.k_nn_values else [config.k_nn]
    skf = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=config.random_seed)
    class_balance = np.bincount(y, minlength=2)
    print(
        "Hyperparameters:",
        {
            "node2vec_dim": config.node2vec_dimensions,
            "node2vec_walk_length": config.node2vec_walk_length,
            "node2vec_num_walks": config.node2vec_num_walks,
            "node2vec_window": config.node2vec_window,
            "node2vec_p": config.node2vec_p,
            "node2vec_q": config.node2vec_q,
            "k_nn_values": k_values,
            "similarity_metric": config.similarity_metric,
            "lr_c": config.lr_c,
            "lr_solver": config.lr_solver,
            "lr_max_iter": config.lr_max_iter,
            "positive_label_value": config.positive_label_value,
            "class_balance": class_balance.tolist(),
        },
    )

    per_fold_rows = []
    pred_rows = []

    raw_importances = []
    emb_importances = []
    concat_importances = []

    random_label_aucs = []
    debug_done = False
    for fold, (train_idx, test_idx) in enumerate(skf.split(ids, y), start=1):
        train_ids = ids[train_idx]
        test_ids = ids[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_df = features.iloc[train_idx].reset_index(drop=True)
        test_df = features.iloc[test_idx].reset_index(drop=True)

        sim_train_df = similarity_df.iloc[train_idx].reset_index(drop=True)
        sim_test_df = similarity_df.iloc[test_idx].reset_index(drop=True)

        raw_imputer = SimpleImputer(strategy="median")
        raw_scaler = StandardScaler()
        X_raw_train = raw_imputer.fit_transform(train_df[raw_features])
        X_raw_train = raw_scaler.fit_transform(X_raw_train)
        X_raw_test = raw_scaler.transform(raw_imputer.transform(test_df[raw_features]))

        base_pca_dim = config.node2vec_dimensions
        if config.pca_components is not None:
            base_pca_dim = config.pca_components
        pca_components = min(base_pca_dim, X_raw_train.shape[1], len(train_idx) - 1)
        pca_components = max(pca_components, 1)
        pca = PCA(n_components=pca_components, random_state=config.random_seed)
        X_pca_train = pca.fit_transform(X_raw_train)
        X_pca_test = pca.transform(X_raw_test)

        sim_imputer = SimpleImputer(strategy="median")
        sim_scaler = StandardScaler()
        X_sim_train = sim_imputer.fit_transform(sim_train_df[similarity_features])
        X_sim_train = sim_scaler.fit_transform(X_sim_train)
        X_sim_test = sim_scaler.transform(sim_imputer.transform(sim_test_df[similarity_features]))

        outputs = {}
        auc_bipartite = None
        auc_full = None

        for k_value in k_values:
            similarity_edges = build_similarity_edges(train_ids, X_sim_train, k_value)
            for variant in ("bipartite", "subject_only", "full"):
                if variant == "bipartite":
                    graph = build_train_graph(train_df, raw_features, False, [])
                elif variant == "subject_only":
                    graph = build_similarity_graph(train_ids, similarity_edges)
                else:
                    graph = build_train_graph(train_df, raw_features, True, similarity_edges)

                model = fit_node2vec(graph)

                train_embeddings = np.vstack([model.wv[f"S{sid}"] for sid in train_ids])
                test_embeddings = project_test_embeddings(X_sim_test, X_sim_train, train_embeddings, k_value)

                emb_cols = [f"emb_{i}" for i in range(train_embeddings.shape[1])]

                fold_train = pd.DataFrame(train_embeddings, columns=emb_cols)
                fold_train.insert(0, "ID", train_ids)
                fold_test = pd.DataFrame(test_embeddings, columns=emb_cols)
                fold_test.insert(0, "ID", test_ids)

                fold_train.to_csv(
                    results_dir / f"embeddings_{variant}_k{k_value}_fold{fold}_train.csv", index=False
                )
                fold_test.to_csv(
                    results_dir / f"embeddings_{variant}_k{k_value}_fold{fold}_test.csv", index=False
                )

                emb_imputer = SimpleImputer(strategy="median")
                emb_scaler = StandardScaler()
                X_emb_train = emb_scaler.fit_transform(emb_imputer.fit_transform(train_embeddings))
                X_emb_test = emb_scaler.transform(emb_imputer.transform(test_embeddings))

                outputs[(variant, k_value)] = (X_emb_train, X_emb_test, emb_cols)

                lr = LogisticRegression(
                    C=config.lr_c,
                    solver=config.lr_solver,
                    max_iter=config.lr_max_iter,
                    random_state=config.random_seed,
                )
                lr.fit(X_emb_train, y_train)
                prob = proba_positive(lr, X_emb_test, y_test, debug=not debug_done, tag="kg")
                debug_done = True
                auc = roc_auc_score(y_test, prob)
                if variant == "bipartite" and k_value == config.k_nn_values[0]:
                    auc_bipartite = auc
                if variant == "full" and k_value == config.k_nn_values[0]:
                    auc_full = auc

                per_fold_rows.append(
                    {
                        "representation": "kg",
                        "variant": f"{variant}_k{k_value}",
                        "fold": fold,
                        "auc": auc,
                    }
                )
                pred_rows.extend(
                    {
                        "ID": int(sid),
                        "fold": fold,
                        "y_true": int(y_true),
                        "y_score": float(score),
                        "representation": "kg",
                        "variant": f"{variant}_k{k_value}",
                    }
                    for sid, y_true, score in zip(test_ids, y_test, prob)
                )

                if variant == "full" and k_value == config.k_nn_values[0]:
                    perm = permutation_importance(
                        lr,
                        X_emb_test,
                        y_test,
                        scoring="roc_auc",
                        n_repeats=config.permutation_repeats,
                        random_state=config.random_seed,
                        n_jobs=1,
                    )
                    emb_importances.append(perm.importances_mean)

        lr_raw = LogisticRegression(
            C=config.lr_c,
            solver=config.lr_solver,
            max_iter=config.lr_max_iter,
            random_state=config.random_seed,
        )
        lr_raw.fit(X_raw_train, y_train)
        prob_raw = proba_positive(lr_raw, X_raw_test, y_test, debug=not debug_done, tag="raw")
        debug_done = True
        auc_raw = roc_auc_score(y_test, prob_raw)
        y_train_rand = y_train.copy()
        rng = np.random.default_rng(config.random_seed + fold)
        rng.shuffle(y_train_rand)
        lr_rand = LogisticRegression(
            C=config.lr_c,
            solver=config.lr_solver,
            max_iter=config.lr_max_iter,
            random_state=config.random_seed,
        )
        lr_rand.fit(X_raw_train, y_train_rand)
        prob_rand = proba_positive(lr_rand, X_raw_test, y_test, debug=not debug_done, tag="random")
        debug_done = True
        random_label_aucs.append(roc_auc_score(y_test, prob_rand))
        per_fold_rows.append(
            {
                "representation": "raw",
                "variant": "baseline",
                "fold": fold,
                "auc": auc_raw,
            }
        )
        pred_rows.extend(
            {
                "ID": int(sid),
                "fold": fold,
                "y_true": int(y_true),
                "y_score": float(score),
                "representation": "raw",
                "variant": "baseline",
            }
            for sid, y_true, score in zip(test_ids, y_test, prob_raw)
        )

        perm_raw = permutation_importance(
            lr_raw,
            X_raw_test,
            y_test,
            scoring="roc_auc",
            n_repeats=config.permutation_repeats,
            random_state=config.random_seed,
            n_jobs=1,
        )
        raw_importances.append(perm_raw.importances_mean)

        lr_pca = LogisticRegression(
            C=config.lr_c,
            solver=config.lr_solver,
            max_iter=config.lr_max_iter,
            random_state=config.random_seed,
        )
        lr_pca.fit(X_pca_train, y_train)
        prob_pca = proba_positive(lr_pca, X_pca_test, y_test, debug=not debug_done, tag="pca")
        debug_done = True
        auc_pca = roc_auc_score(y_test, prob_pca)
        per_fold_rows.append(
            {
                "representation": "pca",
                "variant": "baseline",
                "fold": fold,
                "auc": auc_pca,
            }
        )
        pred_rows.extend(
            {
                "ID": int(sid),
                "fold": fold,
                "y_true": int(y_true),
                "y_score": float(score),
                "representation": "pca",
                "variant": "baseline",
            }
            for sid, y_true, score in zip(test_ids, y_test, prob_pca)
        )

        concat_key = ("full", k_values[0])
        X_emb_train_full, X_emb_test_full, emb_cols = outputs[concat_key]
        X_concat_train = np.concatenate([X_raw_train, X_emb_train_full], axis=1)
        X_concat_test = np.concatenate([X_raw_test, X_emb_test_full], axis=1)

        lr_concat = LogisticRegression(
            C=config.lr_c,
            solver=config.lr_solver,
            max_iter=config.lr_max_iter,
            random_state=config.random_seed,
        )
        lr_concat.fit(X_concat_train, y_train)
        prob_concat = proba_positive(lr_concat, X_concat_test, y_test, debug=not debug_done, tag="concat")
        debug_done = True
        auc_concat = roc_auc_score(y_test, prob_concat)
        per_fold_rows.append(
            {
                "representation": "concat",
                "variant": f"full_k{k_values[0]}",
                "fold": fold,
                "auc": auc_concat,
            }
        )
        pred_rows.extend(
            {
                "ID": int(sid),
                "fold": fold,
                "y_true": int(y_true),
                "y_score": float(score),
                "representation": "concat",
                "variant": f"full_k{k_values[0]}",
            }
            for sid, y_true, score in zip(test_ids, y_test, prob_concat)
        )

        perm_concat = permutation_importance(
            lr_concat,
            X_concat_test,
            y_test,
            scoring="roc_auc",
            n_repeats=config.permutation_repeats,
            random_state=config.random_seed,
            n_jobs=1,
        )
        concat_importances.append(perm_concat.importances_mean)

        print(
            f"Fold {fold}: raw={auc_raw:.3f} pca={auc_pca:.3f} "
            f"kg_bipartite={auc_bipartite:.3f} kg_full={auc_full:.3f} "
            f"concat={auc_concat:.3f}"
        )

    per_fold = pd.DataFrame(per_fold_rows)
    per_fold.to_csv(results_dir / "auc_per_fold.csv", index=False)

    predictions = pd.DataFrame(pred_rows)
    predictions.to_csv(results_dir / "predictions.csv", index=False)

    summary_rows = []
    for (rep, variant), grp in per_fold.groupby(["representation", "variant"]):
        mean_auc = grp["auc"].mean()
        std_auc = grp["auc"].std()

        mask = (predictions["representation"] == rep) & (predictions["variant"] == variant)
        y_true = predictions.loc[mask, "y_true"].to_numpy()
        y_score = predictions.loc[mask, "y_score"].to_numpy()
        ci_low, ci_high = bootstrap_auc(y_true, y_score, config.bootstrap_samples, config.random_seed)

        summary_rows.append(
            {
                "representation": rep,
                "variant": variant,
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(results_dir / "auc_summary.csv", index=False)

    raw_imp = np.vstack(raw_importances)
    raw_imp_df = pd.DataFrame(
        {
            "feature": raw_features,
            "importance_mean": raw_imp.mean(axis=0),
            "importance_std": raw_imp.std(axis=0),
        }
    ).sort_values("importance_mean", ascending=False)
    raw_imp_df.to_csv(results_dir / "perm_importance_raw.csv", index=False)

    emb_imp = np.vstack(emb_importances)
    emb_cols = [f"emb_{i}" for i in range(emb_imp.shape[1])]
    emb_imp_df = pd.DataFrame(
        {
            "feature": emb_cols,
            "importance_mean": emb_imp.mean(axis=0),
            "importance_std": emb_imp.std(axis=0),
        }
    ).sort_values("importance_mean", ascending=False)
    emb_imp_df.to_csv(results_dir / "perm_importance_emb.csv", index=False)

    concat_imp = np.vstack(concat_importances)
    concat_cols = raw_features + emb_cols
    concat_imp_df = pd.DataFrame(
        {
            "feature": concat_cols,
            "importance_mean": concat_imp.mean(axis=0),
            "importance_std": concat_imp.std(axis=0),
        }
    ).sort_values("importance_mean", ascending=False)
    concat_imp_df.to_csv(results_dir / "perm_importance_concat.csv", index=False)

    raw_pred = predictions[(predictions["representation"] == "raw") & (predictions["variant"] == "baseline")]
    raw_auc = roc_auc_score(raw_pred["y_true"], raw_pred["y_score"])
    rng = np.random.default_rng(config.random_seed)
    shuffled = raw_pred["y_true"].to_numpy().copy()
    rng.shuffle(shuffled)
    shuffled_auc = roc_auc_score(shuffled, raw_pred["y_score"])
    scores_inverted_auc = roc_auc_score(raw_pred["y_true"], 1 - raw_pred["y_score"].to_numpy())
    labels_flipped_auc = roc_auc_score(1 - raw_pred["y_true"].to_numpy(), raw_pred["y_score"])


    sanity = {
        "raw_auc": float(raw_auc),
        "shuffled_label_auc": float(shuffled_auc),
        "scores_inverted_auc": float(scores_inverted_auc),
        "labels_flipped_auc": float(labels_flipped_auc),
        "random_label_auc_mean": float(np.mean(random_label_aucs)),
        "random_label_auc_std": float(np.std(random_label_aucs)),
    }
    with (results_dir / "sanity_checks.json").open("w", encoding="utf-8") as handle:
        json.dump(sanity, handle, indent=2)

    print("\nSanity checks (raw baseline):")
    print(json.dumps(sanity, indent=2))
    if raw_auc < 0.5:
        print(
            "Warning: raw AUC < 0.5. Verify label encoding/positive label orientation "
            f"(positive_label_value={config.positive_label_value})."
        )
    if labels_flipped_auc > raw_auc or scores_inverted_auc > raw_auc:
        print("Note: inverted AUC > raw AUC suggests positive-label orientation mismatch.")
    print("\nSaved outputs to:", results_dir.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-free CV benchmark for raw/KG representations.")
    parser.add_argument("--features", default=CONFIG.features_path, help="Path to features.csv")
    parser.add_argument("--patient", default=CONFIG.patient_path, help="Path to patient_info.csv")
    parser.add_argument("--raw-feature-list", default=CONFIG.raw_feature_list, help="Path to raw feature list")
    parser.add_argument(
        "--similarity-feature-list",
        default=CONFIG.similarity_feature_list,
        help="Path to similarity feature list",
    )
    parser.add_argument("--results-dir", default=CONFIG.results_dir, help="Output directory for results")
    parser.add_argument(
        "--k-values",
        default=",".join(str(k) for k in CONFIG.k_nn_values),
        help="Comma-separated list of k values for kNN sensitivity (e.g., 3,5,10).",
    )
    parser.add_argument("--label-column", default=CONFIG.label_column, help="Label column in patient_info.csv")
    parser.add_argument(
        "--positive-label",
        default=CONFIG.positive_label_value,
        help="Value treated as positive class (mapped to 1). Accepts numbers or strings like 'A'/'T'.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    k_values = tuple(int(x) for x in args.k_values.split(",") if x.strip())
    run(
        Config(
            features_path=args.features,
            patient_path=args.patient,
            raw_feature_list=args.raw_feature_list,
            similarity_feature_list=args.similarity_feature_list,
            label_column=args.label_column,
            positive_label_value=args.positive_label,
            results_dir=args.results_dir,
            random_seed=CONFIG.random_seed,
            k_folds=CONFIG.k_folds,
            k_nn=CONFIG.k_nn,
            k_nn_values=k_values,
            node2vec_dimensions=CONFIG.node2vec_dimensions,
            node2vec_walk_length=CONFIG.node2vec_walk_length,
            node2vec_num_walks=CONFIG.node2vec_num_walks,
            node2vec_window=CONFIG.node2vec_window,
            node2vec_p=CONFIG.node2vec_p,
            node2vec_q=CONFIG.node2vec_q,
            node2vec_workers=CONFIG.node2vec_workers,
            lr_c=CONFIG.lr_c,
            lr_solver=CONFIG.lr_solver,
            lr_max_iter=CONFIG.lr_max_iter,
            pca_components=CONFIG.pca_components,
            permutation_repeats=CONFIG.permutation_repeats,
            bootstrap_samples=CONFIG.bootstrap_samples,
            similarity_metric=CONFIG.similarity_metric,
        )
    )
