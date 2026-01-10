<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Graph-Based Representations of Motor Behavior Improve Machine Learning Prediction of ADHD</title>
</head>

<body>

<h1>Graph-Based Representations of Motor Behavior Improve Machine Learning Prediction of ADHD</h1>

<p>
This repository accompanies the study investigating whether graph-based representations
of actigraphy-derived motor behavior improve ADHD classification compared to standard
flat feature vectors.
</p>

<p>
Using a small clinical dataset (HYPERAKTIV; N=85), we compare:
</p>

<ul>
    <li>Raw actigraphy-derived motor features</li>
    <li>PCA-reduced raw features</li>
    <li>Knowledge-graph embeddings (Node2Vec)</li>
    <li>Concatenation of raw features and graph embeddings</li>
</ul>

<p>
All methods use the same classifier (logistic regression) and the same evaluation protocol,
so that any performance differences arise solely from representation choice.
</p>

<hr>

<h2>Data</h2>

<p>
The pipeline assumes that movement-derived features have already been computed from
actigraphy or IMU-like acceleration signals.
</p>

<p>Feature types include:</p>

<ul>
    <li>entropy-based measures,</li>
    <li>complexity measures (e.g., Lempel–Ziv),</li>
    <li>variability descriptors.</li>
</ul>

<p>
The file <code>features_complete.txt</code> documents the full feature set used in the study.
</p>

<div class="note">
<strong>Important:</strong> No additional variables are introduced at any stage of the graph-based analysis.
</div>

<hr>

<h2>Step-by-Step Reproducibility Guide</h2>

<h3>Recommended Pipeline (Leakage-Free, Fold-Wise)</h3>

<p>
The primary pipeline is implemented in
<code>scripts/run_cv_representation_benchmark.py</code> and executes a leakage-free evaluation
across representations using stratified 5-fold cross-validation.
</p>

<p>
All preprocessing steps (median imputation, standardization) are fit exclusively on the
training folds and then applied to the corresponding test folds.
</p>

<pre>python -m scripts.run_cv_representation_benchmark</pre>

<p>This pipeline evaluates:</p>

<ul>
    <li>Raw motor behavior features (logistic regression)</li>
    <li>PCA baseline on raw features (logistic regression)</li>
    <li>Knowledge-graph embeddings (Node2Vec) with fold-wise graph construction</li>
    <li>Concatenation of raw features and projected embeddings</li>
    <li>Ablation: graph embeddings with and without subject–subject similarity edges</li>
</ul>

<h4>Leakage-Free Inductive Projection</h4>

<p>
Node2Vec is a transductive method. To ensure leakage-free evaluation, embeddings are learned
only on the training graph in each fold.
</p>

<p>
Test subjects are projected into the training embedding space via k-nearest-neighbor
similarity to training subjects (cosine similarity, weights normalized to sum to one).
All similarity normalization parameters are estimated from training data only.
</p>

<h4>Outputs</h4>

<p>All outputs are written to the <code>results/</code> directory:</p>

<ul>
    <li><code>auc_per_fold.csv</code> – per-fold ROC–AUC values</li>
    <li><code>auc_summary.csv</code> – mean AUC, standard deviation, and 95% confidence interval</li>
    <li><code>predictions.csv</code> – out-of-fold predictions</li>
    <li><code>perm_importance_raw.csv</code></li>
    <li><code>perm_importance_emb.csv</code></li>
    <li><code>perm_importance_concat.csv</code></li>
    <li><code>sanity_checks.json</code> – label-shuffle and label-inversion AUC checks</li>
    <li>
        <code>embeddings_*_fold*_train.csv</code> /
        <code>embeddings_*_fold*_test.csv</code>
    </li>
</ul>

<h4>Configuration</h4>

<p>
All reproducibility-related parameters (random seeds, number of folds, kNN parameters,
Node2Vec hyperparameters, classifier settings, PCA configuration, and feature list paths)
are centralized in <code>scripts/config.py</code>.
</p>

<p>
An RBF-SVM baseline is intentionally omitted, as it would introduce additional tuning
and dependency complexity. PCA provides a compact representation baseline aligned with
the study’s design philosophy.
</p>

<h3>Legacy Scripts (Optional)</h3>

<p>
Original scripts in <code>baseline_ml/</code> and <code>kg/</code> are preserved for reference.
However, the pipeline described above should be used for all reproducible,
leakage-free experiments.
</p>

<hr>

<h2>Key Design Principles</h2>

<ul>
    <li><strong>Same classifier everywhere:</strong> logistic regression</li>
    <li><strong>Same raw features everywhere:</strong> graphs are constructed from identical inputs</li>
    <li><strong>Same evaluation protocol:</strong> stratified 5-fold cross-validation</li>
    <li><strong>No hidden tuning advantages:</strong> representation is the only manipulated factor</li>
</ul>

<hr>

<h2>Example Command Sequence</h2>

<pre>
pip install -r requirements.txt
python -m scripts.run_cv_representation_benchmark
</pre>

<hr>

<h2>Intended Use</h2>

<p>
This repository is intended for methodological comparison, reproducibility,
and extension to other movement or clinical datasets.
</p>

<p><strong>It is not intended as a diagnostic tool.</strong></p>

<hr>

<h2>Citation</h2>

<p>
If you use this code or adapt the pipeline, please cite the accompanying paper.
</p>

<footer>
<hr>
<p>Maintained for research and reproducibility purposes.</p>
</footer>

</body>
</html>