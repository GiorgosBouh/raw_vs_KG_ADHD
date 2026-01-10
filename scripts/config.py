from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    features_path: str = "data/raw/features.csv"
    patient_path: str = "data/raw/patient_info.csv"
    raw_feature_list: str = "kg/features_entropy_variability.txt"
    similarity_feature_list: str = "kg/features_entropy_variability.txt"
    label_column: str = "ADHD"
    results_dir: str = "results"

    random_seed: int = 42
    k_folds: int = 5
    k_nn: int = 5

    node2vec_dimensions: int = 64
    node2vec_walk_length: int = 20
    node2vec_num_walks: int = 20

    perm_importance_repeats: int = 1000


CONFIG = Config()
