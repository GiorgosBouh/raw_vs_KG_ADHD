from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from scripts.config import CONFIG


def read_csv_auto(path: str) -> pd.DataFrame:
    return pd.read_csv(path, engine="python", sep=None)


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


def align_inputs(features: pd.DataFrame, patient: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    for df in (features, patient):
        df.columns = [c.strip() for c in df.columns]

    if "ID" not in features.columns or "ID" not in patient.columns:
        raise KeyError("Both features.csv and patient_info.csv must include an 'ID' column.")

    features = features.copy()
    patient = patient.copy()

    if "filter_$" in patient.columns:
        patient = patient[patient["filter_$"] == 1].copy()

    patient[CONFIG.label_column] = pd.to_numeric(patient[CONFIG.label_column], errors="coerce")
    patient = patient.dropna(subset=[CONFIG.label_column])
    patient[CONFIG.label_column] = patient[CONFIG.label_column].astype(int)

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

    labels = patient.set_index("ID")[CONFIG.label_column].astype(int)
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


def run():
    results_dir = Path(CONFIG.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    features = read_csv_auto(CONFIG.features_path)
    patient = read_csv_auto(CONFIG.patient_path)

    features, labels = align_inputs(features, patient)

    raw_features = read_feature_list(CONFIG.raw_feature_list)
    raw_features = [c for c in raw_features if c in features.columns]
    if not raw_features:
        raise ValueError("No raw features found in features.csv that match the feature list.")

    similarity_features = read_feature_list(CONFIG.similarity_feature_list)
    similarity_features = [c for c in similarity_features if c in features.columns]
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

    skf = StratifiedKFold(n_splits=CONFIG.k_folds, shuffle=True, random_state=CONFIG.random_seed)

    per_fold_rows = []
    pred_rows = []

    raw_importances = []
    emb_importances = []
    concat_importances = []

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

        pca_components = min(CONFIG.pca_components, X_raw_train.shape[1], len(train_idx) - 1)
        pca_components = max(pca_components, 1)
        pca = PCA(n_components=pca_components, random_state=CONFIG.random_seed)
        X_pca_train = pca.fit_transform(X_raw_train)
        X_pca_test = pca.transform(X_raw_test)

        sim_imputer = SimpleImputer(strategy="median")
        sim_scaler = StandardScaler()
        X_sim_train = sim_imputer.fit_transform(sim_train_df[similarity_features])
        X_sim_train = sim_scaler.fit_transform(X_sim_train)
        X_sim_test = sim_scaler.transform(sim_imputer.transform(sim_test_df[similarity_features]))

        similarity_edges = build_similarity_edges(train_ids, X_sim_train, CONFIG.k_nn)

        outputs = {}

        auc_bipartite = None
        auc_full = None
        for variant, include_sim in ("bipartite", False), ("full", True):
            graph = build_train_graph(train_df, raw_features, include_sim, similarity_edges if include_sim else [])
            model = fit_node2vec(graph)

            train_embeddings = np.vstack([model.wv[f"S{sid}"] for sid in train_ids])
            test_embeddings = project_test_embeddings(X_sim_test, X_sim_train, train_embeddings, CONFIG.k_nn)

            emb_cols = [f"emb_{i}" for i in range(train_embeddings.shape[1])]

            fold_train = pd.DataFrame(train_embeddings, columns=emb_cols)
            fold_train.insert(0, "ID", train_ids)
            fold_test = pd.DataFrame(test_embeddings, columns=emb_cols)
            fold_test.insert(0, "ID", test_ids)

            fold_train.to_csv(results_dir / f"embeddings_{variant}_fold{fold}_train.csv", index=False)
            fold_test.to_csv(results_dir / f"embeddings_{variant}_fold{fold}_test.csv", index=False)

            emb_imputer = SimpleImputer(strategy="median")
            emb_scaler = StandardScaler()
            X_emb_train = emb_scaler.fit_transform(emb_imputer.fit_transform(train_embeddings))
            X_emb_test = emb_scaler.transform(emb_imputer.transform(test_embeddings))

            outputs[variant] = (X_emb_train, X_emb_test, emb_cols)

            lr = LogisticRegression(
                C=CONFIG.lr_c,
                solver=CONFIG.lr_solver,
                max_iter=CONFIG.lr_max_iter,
                random_state=CONFIG.random_seed,
            )
            lr.fit(X_emb_train, y_train)
            prob = lr.predict_proba(X_emb_test)[:, 1]
            auc = roc_auc_score(y_test, prob)
            if variant == "bipartite":
                auc_bipartite = auc
            else:
                auc_full = auc

            per_fold_rows.append(
                {
                    "representation": "kg",
                    "variant": variant,
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
                    "variant": variant,
                }
                for sid, y_true, score in zip(test_ids, y_test, prob)
            )

            if variant == "full":
                perm = permutation_importance(
                    lr,
                    X_emb_test,
                    y_test,
                    scoring="roc_auc",
                    n_repeats=CONFIG.permutation_repeats,
                    random_state=CONFIG.random_seed,
                    n_jobs=1,
                )
                emb_importances.append(perm.importances_mean)

        lr_raw = LogisticRegression(
            C=CONFIG.lr_c,
            solver=CONFIG.lr_solver,
            max_iter=CONFIG.lr_max_iter,
            random_state=CONFIG.random_seed,
        )
        lr_raw.fit(X_raw_train, y_train)
        prob_raw = lr_raw.predict_proba(X_raw_test)[:, 1]
        auc_raw = roc_auc_score(y_test, prob_raw)
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
            n_repeats=CONFIG.permutation_repeats,
            random_state=CONFIG.random_seed,
            n_jobs=1,
        )
        raw_importances.append(perm_raw.importances_mean)

        lr_pca = LogisticRegression(
            C=CONFIG.lr_c,
            solver=CONFIG.lr_solver,
            max_iter=CONFIG.lr_max_iter,
            random_state=CONFIG.random_seed,
        )
        lr_pca.fit(X_pca_train, y_train)
        prob_pca = lr_pca.predict_proba(X_pca_test)[:, 1]
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

        X_emb_train_full, X_emb_test_full, emb_cols = outputs["full"]
        X_concat_train = np.concatenate([X_raw_train, X_emb_train_full], axis=1)
        X_concat_test = np.concatenate([X_raw_test, X_emb_test_full], axis=1)

        lr_concat = LogisticRegression(
            C=CONFIG.lr_c,
            solver=CONFIG.lr_solver,
            max_iter=CONFIG.lr_max_iter,
            random_state=CONFIG.random_seed,
        )
        lr_concat.fit(X_concat_train, y_train)
        prob_concat = lr_concat.predict_proba(X_concat_test)[:, 1]
        auc_concat = roc_auc_score(y_test, prob_concat)
        per_fold_rows.append(
            {
                "representation": "concat",
                "variant": "full",
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
                "variant": "full",
            }
            for sid, y_true, score in zip(test_ids, y_test, prob_concat)
        )

        perm_concat = permutation_importance(
            lr_concat,
            X_concat_test,
            y_test,
            scoring="roc_auc",
            n_repeats=CONFIG.permutation_repeats,
            random_state=CONFIG.random_seed,
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
        ci_low, ci_high = bootstrap_auc(y_true, y_score, CONFIG.bootstrap_samples, CONFIG.random_seed)

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
    rng = np.random.default_rng(CONFIG.random_seed)
    shuffled = raw_pred["y_true"].to_numpy().copy()
    rng.shuffle(shuffled)
    shuffled_auc = roc_auc_score(shuffled, raw_pred["y_score"])
    inverted_auc = roc_auc_score(1 - raw_pred["y_true"].to_numpy(), raw_pred["y_score"])

    sanity = {
        "raw_auc": float(raw_auc),
        "shuffled_label_auc": float(shuffled_auc),
        "inverted_label_auc": float(inverted_auc),
    }
    with (results_dir / "sanity_checks.json").open("w", encoding="utf-8") as handle:
        json.dump(sanity, handle, indent=2)

    print("\nSanity checks (raw baseline):")
    print(json.dumps(sanity, indent=2))
    if raw_auc < 0.5:
        print("Warning: raw AUC < 0.5. Verify label encoding and probability direction.")
    print("\nSaved outputs to:", results_dir.resolve())


if __name__ == "__main__":
    run()
