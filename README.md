

<h1>Graph-Based Representations of Motor Behavior Improve Machine Learning Prediction of Attention-Deficit/Hyperactivity Disorder</h1>

<p>
This repository accompanies the study:
</p>

<p><strong>
“Graph-Based Representations of Motor Behavior Improve Machine Learning Prediction of Attention-Deficit/Hyperactivity Disorder”
</strong></p>

<p>
It provides a fully reproducible pipeline for comparing <strong>raw feature–based machine learning</strong>
with <strong>knowledge graph–based representations</strong> derived from the <em>same motor behavior data</em>,
under a strictly controlled experimental design.
</p>
<h2>Step-by-Step Reproducibility Guide</h2>

<h3>Recommended Pipeline (Leakage-Free, Fold-Wise)</h3>

<p>
The primary pipeline is implemented in <code>scripts/run_cv_representation_benchmark.py</code> and
executes a leakage-free evaluation across representations using stratified 5-fold CV. All
preprocessing (median imputation, standardization) is fit on training folds only.
</p>

<pre>python -m scripts.run_cv_representation_benchmark</pre>

<p>This pipeline performs:</p>
<ul>
    <li>Raw actigraphy-derived motor behavior features (logistic regression)</li>
    <li>PCA baseline on raw features (logistic regression)</li>
    <li>Knowledge-graph embeddings (Node2Vec) with fold-wise graph construction</li>
    <li>Concatenation of raw + projected embeddings</li>
    <li>Ablation: KG with subject–subject similarity edges vs. without</li>
</ul>

<h4>Leakage-Free Inductive Projection</h4>
<p>
Node2Vec is transductive, so each fold fits embeddings only on the training graph.
Test subjects are mapped to the training embedding space via kNN similarity to the
training subjects (cosine weights normalized to sum to 1). Similarity normalization
is learned from training data only.
</p>

<h4>Outputs</h4>
<p>All outputs are written to <code>results/</code>:</p>
<ul>
    <li><code>auc_per_fold.csv</code> (per-fold AUCs)</li>
    <li><code>auc_summary.csv</code> (mean, std, and 95% CI)</li>
    <li><code>predictions.csv</code> (out-of-fold predictions)</li>
    <li><code>perm_importance_raw.csv</code></li>
    <li><code>perm_importance_emb.csv</code></li>
    <li><code>perm_importance_concat.csv</code></li>
    <li><code>sanity_checks.json</code> (label-shuffle and label-inversion AUC checks)</li>
    <li><code>embeddings_*_fold*_train.csv</code> / <code>embeddings_*_fold*_test.csv</code></li>
</ul>

<h4>Configuration</h4>
<p>
All reproducibility parameters (seeds, folds, kNN, Node2Vec, logistic regression, PCA,
feature list paths) are centralized in <code>scripts/config.py</code>.
</p>

<p>
An RBF-SVM baseline is not included because it would require additional dependency
management beyond the lightweight baseline set; PCA provides a compact representation
baseline aligned with the study design.
</p>

<h3>Legacy Scripts (Optional)</h3>
<p>
The original scripts in <code>baseline_ml/</code> and <code>kg/</code> are kept for reference, but the
recommended pipeline above should be used for leakage-free evaluation.
</p>

<h2>Example Command Sequence</h2>

<pre>
pip install -r requirements.txt
python -m scripts.run_cv_representation_benchmark
</pre>