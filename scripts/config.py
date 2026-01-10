from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Input paths
    features_path: str = "data/raw/features.csv"
    patient_path: str = "data/raw/patient_info.csv"

    # Optional feature lists (for KG construction later)
    raw_feature_list: str = "kg/features_entropy_variability.txt"
    similarity_feature_list: str = "kg/features_entropy_variability.txt"

    # Column name for target label
    label_column: str = "class"

    # Output directory
    results_dir: str = "results"

    # CV / randomness
    random_seed: int = 42
    k_folds: int = 5
    k_nn: int = 5

    # Node2Vec (if/when used)
    node2vec_dimensions: int = 64
    node2vec_walk_length: int = 20
    node2vec_num_walks: int = 20

    # Permutation importance repeats
    perm_importance_repeats: int = 1000


CONFIG = Config()
