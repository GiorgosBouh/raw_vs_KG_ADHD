Graph-Based Representations of Motor Behavior Improve Machine Learning Prediction of Attention-Deficit/Hyperactivity Disorder

This repository accompanies the study:

“Graph-Based Representations of Motor Behavior Improve Machine Learning Prediction of Attention-Deficit/Hyperactivity Disorder”

It provides a fully reproducible pipeline for comparing raw feature–based machine learning with knowledge graph–based representations derived from the same movement data, under a controlled experimental design.

The central goal is to isolate the effect of data representation on classification performance, while keeping:
	•	the feature set,
	•	the machine learning model,
	•	and the evaluation protocol
strictly constant.

⸻

Overview of the Approach

The project implements two alternative representations of the same motor behavior data:
	1.	Raw feature representation
Movement-derived descriptors (entropy, variability, complexity) are treated as independent numerical inputs to a machine learning classifier.
	2.	Graph-based representation (Expert Knowledge Graph)
The same features are reorganized into a subject–feature knowledge graph.
Subject embeddings are learned using Node2Vec, encoding relational and semantic structure among participants.

A single machine learning model (logistic regression) is trained and evaluated on:
	•	raw features,
	•	graph embeddings,
	•	and their concatenation.

Performance differences therefore reflect representation alone, not algorithmic changes.

Graph Based Representations of Motor Behavior Improve Machine Learning Prediction of Attention/
├── baseline_ml/        # Raw feature-based ML pipeline
├── kg/                 # Knowledge graph construction and embedding
├── scripts/            # Shared utilities and analysis scripts
├── results/            # Saved outputs (AUCs, embeddings, permutation importance)
├── data/               # Input data (if permitted to share)
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── manuscript.tex      # LaTeX manuscript (optional)

Requirements
	•	Python ≥ 3.9
	•	Recommended: virtual environment

Install dependencies:
pip install -r requirements.txt

Data

The pipeline assumes movement-derived features already computed from actigraphy or IMU-like acceleration signals.

Feature types include:
	•	entropy-based measures,
	•	complexity measures (e.g., Lempel–Ziv),
	•	variability descriptors.

The file features_complete.txt documents the full feature set used in the study.

Important:
No additional variables are introduced at any stage of the graph-based analysis.

⸻

Step-by-Step Reproducibility Guide

1. Raw Feature-Based Machine Learning (Baseline)

Located in: baseline_ml/

This stage evaluates conventional ML using flat feature representations.

Main script:
python bench_raw_ml.py

What it does:
	•	Loads raw movement features
	•	Trains a logistic regression classifier
	•	Uses stratified 5-fold cross-validation
	•	Reports ROC–AUC performance

Additional evaluation:
python benchmark_auc.py
python permutation_importance_lr.py

Outputs:
	•	Mean and fold-wise AUC scores
	•	Permutation-based feature importance (perm_importance_raw.csv)
2. Knowledge Graph Construction

Located in: kg/

This stage reorganizes the same features into a relational structure.

Key steps:
python build_expert_kg.py
python add_similarity_layer.py

What happens:
	•	Subjects and features are represented as nodes
	•	Subject–feature relationships are encoded
	•	Subject–subject similarity edges are added using entropy and complexity descriptors
	•	No new measurements are created

The graph structure is saved for downstream embedding.
python run_node2vec.py

This step:
	•	Applies Node2Vec to the subject graph
	•	Produces low-dimensional subject embeddings
	•	Preserves neighborhood and relational structure

Output:
expertkg_node2vec_patient_embeddings.csv

4. Machine Learning on Graph Embeddings
   python export_expert_kg_embeddings.py
   python benchmark_auc.py

This evaluates the same ML model using graph embeddings as input.

Outputs:
	•	ROC–AUC scores for the graph-based representation
	•	Permutation importance for embedding dimensions (perm_importance_emb.csv)

5. Combined Representation (Optional)

Raw features and embeddings are concatenated:
python permutation_importance_lr.py
This tests whether naïve feature fusion improves performance.

Results show:
	•	intermediate performance,
	•	partial redundancy between representations,
	•	no systematic additive gain.
Results Directory

The results/ folder contains:
	•	Cross-validation AUC results
	•	Learned graph embeddings
	•	Permutation importance analyses
	•	Intermediate graph objects

These files are directly referenced in the manuscript.

⸻

Key Design Principles
	•	Same ML model everywhere
Logistic regression is used consistently.
	•	Same features everywhere
Graphs are built from the raw features, not in addition to them.
	•	Same evaluation protocol
Stratified 5-fold cross-validation for all experiments.
	•	No hidden tuning advantages
Differences arise solely from representation.

⸻

Intended Use

This repository is intended for:
	•	reproducibility of the published results,
	•	methodological comparison of representation strategies,
	•	extension to other movement or clinical datasets.

It is not intended as a diagnostic tool.

Citation

If you use this code, please cite the accompanying paper.
