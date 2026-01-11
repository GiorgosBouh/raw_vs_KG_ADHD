# Leakage-Free Benchmark Audit Summary

## What was wrong
- The evaluation could be biased if subject nodes from the test fold leaked into the training graph or Node2Vec walks (transductive leakage risk).
- Raw-feature AUC values were implausibly high, indicating potential leakage or label contamination.
- Some baselines and graph ablations required by reviewers (subject-only graph, kNN sensitivity, PCA matched to embedding dimension) were missing or undocumented.

## What was fixed
- Graph construction and Node2Vec fitting are performed **within each CV fold** using only training subjects.
- Test subjects are embedded inductively via kNN projection onto training embeddings (no test nodes in the graph or walks).
- All preprocessing (imputation, standardization, PCA) is fit only on training folds.
- Added raw-label sanity checks, random-label baselines, and label-flip checks.
- Added ablations for bipartite, subject-only, and full graphs, plus kNN sensitivity.
- Added configuration logging to `results/config_used.json` and removed potential label leakage columns.

## How results changed
- AUCs for KG and raw baselines are expected to **drop to more realistic values** once leakage is removed.
- The relative ranking across representations may change; rerun the benchmark to quantify.
- The new sanity checks should show shuffled/random label AUCs near 0.5.

> Re-run the pipeline with: `python -m scripts.run_cv_representation_benchmark`
