from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class Config:
    features_path: str = "data/raw/features.csv"
    patient_path: str = "data/raw/patient_info.csv"
    raw_feature_list: str = "kg/features_acc_selected.txt"
    similarity_feature_list: str = "kg/features_acc_selected.txt"
    label_column: str = "ADHD"
    positive_label_value: str = "0"
    invert_scores: bool = False
    results_dir: str = "results"

    random_seed: int = 42
    k_folds: int = 5
    k_nn: int = 5
    k_nn_values: Sequence[int] = (5,)
    similarity_metric: str = "cosine"

    node2vec_dimensions: int = 64
    node2vec_walk_length: int = 20
    node2vec_num_walks: int = 20
    node2vec_window: int = 10
    node2vec_p: float = 1.0
    node2vec_q: float = 1.0
    node2vec_workers: int = 1

    lr_c: float = 1.0
    lr_solver: str = "liblinear"
    lr_max_iter: int = 5000

    pca_components: Optional[int] = None
    permutation_repeats: int = 50
    bootstrap_samples: int = 1000


CONFIG = Config()
