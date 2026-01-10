<<<<<<< ours
<h1>Graph-Based Representations of Motor Behavior Improve Machine Learning Prediction of Attention-Deficit/Hyperactivity Disorder</h1> <p> This repository accompanies the study: </p> <p><strong> “Graph-Based Representations of Motor Behavior Improve Machine Learning Prediction of Attention-Deficit/Hyperactivity Disorder” </strong></p> <p> It provides a fully reproducible pipeline for comparing <strong>raw feature–based machine learning</strong> with <strong>knowledge graph–based representations</strong> derived from the <em>same motor behavior data</em>, under a strictly controlled experimental design. </p> <p> The central goal of this work is to <strong>isolate the effect of data representation on classification performance</strong>, while keeping the following components constant across all experiments: </p> <ul> <li>the feature set,</li> <li>the machine learning model,</li> <li>and the evaluation protocol.</li> </ul> <h2>Overview of the Approach</h2> <p>The project implements two alternative representations of the same motor behavior data:</p> <h3>1. Raw Feature Representation</h3> <p> Minute-level actigraphy-derived motor activity descriptors (e.g., entropy, variability, complexity) are treated as <strong>independent numerical inputs</strong> to a machine learning classifier. </p> <h3>2. Graph-Based Representation (Expert Knowledge Graph)</h3> <p> The same features are reorganized into a <strong>subject–feature knowledge graph</strong>. Subject embeddings are learned using <strong>Node2Vec</strong>, encoding relational and semantic structure among participants. </p> <p> A single machine learning model (<strong>logistic regression</strong>) is trained and evaluated on: </p> <ul> <li>raw features,</li> <li>graph embeddings,</li> <li>and their direct concatenation.</li> </ul> <p> Performance differences therefore reflect <strong>representation alone</strong>, not changes in algorithms, features, or tuning. </p> <h2>Repository Structure</h2> <div class="tree"> Graph Based Representations of Motor Behavior Improve Machine Learning Prediction of Attention/<br> ├── baseline_ml/ &nbsp;&nbsp;&nbsp;# Raw feature-based ML pipeline (legacy)<br> ├── kg/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Knowledge graph construction and embeddings (legacy)<br> ├── scripts/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Leakage-free CV benchmark pipeline (recommended)<br> ├── results/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Outputs (AUCs, embeddings, permutation importance)<br> ├── data/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Input data (if permitted to share)<br> ├── requirements.txt # Python dependencies<br> ├── README.md &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# This file<br> └── manuscript.tex &nbsp;# LaTeX manuscript (optional) </div> <h2>Requirements</h2> <ul> <li>Python ≥ 3.9</li> <li>Virtual environment recommended</li> </ul> <pre>pip install -r requirements.txt</pre> <h2>Data</h2> <p> The pipeline assumes that minute-level actigraphy-derived motor activity patterns have already been computed from wearable acceleration signals. </p> <p>Feature types include:</p> <ul> <li>entropy-based measures,</li> <li>complexity measures (e.g., Lempel–Ziv),</li> <li>variability descriptors.</li> </ul> <p> The file <code>features_complete.txt</code> documents the full feature set used in the study. </p> <div class="note"> <strong>Important:</strong> No additional variables are introduced at any stage of the graph-based analysis. </div> <h2>Leakage-Free Evaluation (Recommended)</h2> <p> The primary pipeline is implemented in <code>scripts/run_cv_representation_benchmark.py</code> and executes a leakage-free evaluation across representations using stratified 5-fold CV. All preprocessing (median imputation, standardization, PCA) is fit on training folds only. </p> <pre>python -m scripts.run_cv_representation_benchmark</pre> <p> If your data lives outside the default paths (<code>data/raw/features.csv</code> and <code>data/raw/patient_info.csv</code>), pass explicit paths: </p> <pre> python -m scripts.run_cv_representation_benchmark \ --features path/to/features.csv \ --patient path/to/patient_info.csv \ --k-values 3,5,10 \ --label-column ADHD </pre> <h3>What This Pipeline Runs</h3> <ul> <li>Raw minute-level actigraphy-derived motor activity features → Logistic Regression</li> <li>PCA baseline on raw features (dimension matched to Node2Vec) → Logistic Regression</li> <li>Knowledge-graph embeddings (Node2Vec) with fold-wise graph construction</li> <li>Concatenation of raw + projected embeddings</li> <li>Graph ablations: bipartite vs subject–subject only vs full graph</li> <li>kNN sensitivity analysis for the subject–subject similarity layer</li> </ul> <h3>Leakage-Free Inductive Projection</h3> <p> Node2Vec is transductive, so each fold fits embeddings only on the training graph. Test subjects are mapped to the training embedding space via kNN similarity to the training subjects (cosine weights normalized to sum to 1). Similarity normalization is learned from training data only. </p> <h3>Outputs (Written to results/)</h3> <ul> <li><code>auc_per_fold.csv</code> (per-fold AUCs)</li> <li><code>auc_summary.csv</code> (mean, std, 95% CI)</li> <li><code>predictions.csv</code> (out-of-fold predictions)</li> <li><code>config_used.json</code> (full hyperparameter log)</li> <li><code>perm_importance_raw.csv</code></li> <li><code>perm_importance_emb.csv</code></li> <li><code>perm_importance_concat.csv</code></li> <li><code>sanity_checks.json</code> (label shuffle + label inversion checks)</li> <li><code>embeddings_*_fold*_train.csv</code> / <code>embeddings_*_fold*_test.csv</code></li> </ul> <h3>Configuration</h3> <p> All reproducibility parameters (seeds, folds, kNN, Node2Vec, logistic regression, PCA, feature list paths) are centralized in <code>scripts/config.py</code>. </p> <h2>Key Design Principles</h2> <ul> <li><strong>Same ML model everywhere:</strong> logistic regression</li> <li><strong>Same features everywhere:</strong> graphs built from raw features</li> <li><strong>Same evaluation protocol:</strong> stratified 5-fold cross-validation</li> <li><strong>No hidden tuning advantages:</strong> representation is the only difference</li> </ul> <h2>Example Command Sequence</h2> <pre> pip install -r requirements.txt python -m scripts.run_cv_representation_benchmark </pre> <h2>Legacy Scripts (Optional)</h2> <p> The original scripts in <code>baseline_ml/</code> and <code>kg/</code> are kept for reference, but the recommended pipeline above should be used for leakage-free evaluation. </p> <h2>Troubleshooting</h2> <p> If you see a <code>SyntaxError</code> that mentions conflict markers (e.g., <code>&lt;&lt;&lt;&lt;&lt;&lt;&lt;</code>) in <code>scripts/run_cv_representation_benchmark.py</code>, your local copy contains merge artifacts. Reset the file(s) to the committed version with: </p> <pre> git checkout -- scripts/run_cv_representation_benchmark.py scripts/config.py </pre> <h2>Intended Use</h2> <p> This repository is intended for reproducibility, methodological comparison of representation strategies, and extension to other movement or clinical datasets. </p> <p><strong>It is not intended as a diagnostic tool.</strong></p> <h2>Citation</h2> <p> If you use this code, please cite the accompanying paper. </p> <footer> <hr> <p>Maintained for research and reproducibility purposes.</p> </footer> </body> </html>
=======


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

<p>
The central goal of this work is to <strong>isolate the effect of data representation on classification performance</strong>,
while keeping the following components constant across all experiments:
</p>

<ul>
    <li>the feature set,</li>
    <li>the machine learning model,</li>
    <li>and the evaluation protocol.</li>
</ul>

<h2>Overview of the Approach</h2>

<p>The project implements two alternative representations of the same motor behavior data:</p>

<h3>1. Raw Feature Representation</h3>
<p>
Movement-derived descriptors (e.g., entropy, variability, complexity) are treated as
<strong>independent numerical inputs</strong> to a machine learning classifier.
</p>

<h3>2. Graph-Based Representation (Expert Knowledge Graph)</h3>
<p>
The same features are reorganized into a <strong>subject–feature knowledge graph</strong>.
Subject embeddings are learned using <strong>Node2Vec</strong>, encoding relational and semantic
structure among participants.
</p>

<p>
A single machine learning model (<strong>logistic regression</strong>) is trained and evaluated on:
</p>

<ul>
    <li>raw features,</li>
    <li>graph embeddings,</li>
    <li>and their direct concatenation.</li>
</ul>

<p>
Performance differences therefore reflect <strong>representation alone</strong>, not changes in
algorithms, features, or tuning.
</p>

<h2>Repository Structure</h2>

<div class="tree">
Graph Based Representations of Motor Behavior Improve Machine Learning Prediction of Attention/<br>
├── baseline_ml/ &nbsp;&nbsp;&nbsp;# Raw feature-based ML pipeline<br>
├── kg/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Knowledge graph construction and embeddings<br>
├── scripts/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Shared utilities and analysis scripts<br>
├── results/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Outputs (AUCs, embeddings, permutation importance)<br>
├── data/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Input data (if permitted to share)<br>
├── requirements.txt # Python dependencies<br>
├── README.md &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# This file<br>
└── manuscript.tex &nbsp;# LaTeX manuscript (optional)
</div>

<h2>Requirements</h2>

<ul>
    <li>Python ≥ 3.9</li>
    <li>Virtual environment recommended</li>
</ul>

<pre>pip install -r requirements.txt</pre>

<h2>Data</h2>

<p>
The pipeline assumes that minute-level actigraphy-derived motor activity patterns have
already been computed from wearable acceleration signals.
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

<h2>Step-by-Step Reproducibility Guide</h2>

<h3>Recommended Pipeline (Leakage-Free, Fold-Wise)</h3>

<p>
The primary pipeline is implemented in <code>scripts/run_cv_representation_benchmark.py</code> and
executes a leakage-free evaluation across representations using stratified 5-fold CV. All
preprocessing (median imputation, standardization) is fit on training folds only.
</p>

<pre>python -m scripts.run_cv_representation_benchmark</pre>

<p>
If your data lives outside the default paths (<code>data/raw/features.csv</code> and
<code>data/raw/patient_info.csv</code>), pass explicit paths:
</p>

<pre>
python -m scripts.run_cv_representation_benchmark \
  --features path/to/features.csv \
  --patient path/to/patient_info.csv \
  --k-values 3,5,10 \
  --label-column ADHD
</pre>

<p>This pipeline performs:</p>
<ul>
    <li>Raw minute-level actigraphy-derived motor activity features (logistic regression)</li>
    <li>PCA baseline on raw features (logistic regression)</li>
    <li>Knowledge-graph embeddings (Node2Vec) with fold-wise graph construction</li>
    <li>Concatenation of raw + projected embeddings</li>
    <li>Ablations: bipartite vs subject–subject only vs full graph</li>
    <li>Sensitivity to k in kNN similarity (configurable)</li>
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
    <li><code>config_used.json</code> (full hyperparameter log)</li>
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

<h2>Key Design Principles</h2>

<ul>
    <li><strong>Same ML model everywhere:</strong> logistic regression</li>
    <li><strong>Same features everywhere:</strong> graphs built from raw features</li>
    <li><strong>Same evaluation protocol:</strong> stratified 5-fold cross-validation</li>
    <li><strong>No hidden tuning advantages:</strong> representation is the only difference</li>
</ul>

<h2>Example Command Sequence</h2>

<pre>
pip install -r requirements.txt
python -m scripts.run_cv_representation_benchmark
</pre>

<h2>Troubleshooting</h2>
<p>
If you see a <code>SyntaxError</code> that mentions conflict markers (e.g., <code>&lt;&lt;&lt;&lt;&lt;&lt;&lt;</code>)
in <code>scripts/run_cv_representation_benchmark.py</code>, your local copy contains merge artifacts.
Reset the file(s) to the committed version with:
</p>
<pre>
git checkout -- scripts/run_cv_representation_benchmark.py scripts/config.py
</pre>

<h2>Intended Use</h2>

<p>
This repository is intended for reproducibility, methodological comparison of
representation strategies, and extension to other movement or clinical datasets.
</p>

<p><strong>It is not intended as a diagnostic tool.</strong></p>

<h2>Citation</h2>

<p>
If you use this code, please cite the accompanying paper.
</p>

<footer>
<hr>
<p>Maintained for research and reproducibility purposes.</p>
</footer>

</body>
</html>
>>>>>>> theirs
