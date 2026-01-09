

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

<h2>Step-by-Step Reproducibility Guide</h2>

<h3>1. Raw Feature-Based Machine Learning (Baseline)</h3>

<p><strong>Location:</strong> <code>baseline_ml/</code></p>

<pre>python bench_raw_ml.py</pre>

<p>
This script:
</p>

<ul>
    <li>loads raw movement features,</li>
    <li>trains a logistic regression classifier,</li>
    <li>uses stratified 5-fold cross-validation,</li>
    <li>reports ROC–AUC performance.</li>
</ul>

<p>Additional evaluation:</p>

<pre>
python benchmark_auc.py
python permutation_importance_lr.py
</pre>

<h3>2. Knowledge Graph Construction</h3>

<p><strong>Location:</strong> <code>kg/</code></p>

<pre>
python build_expert_kg.py
python add_similarity_layer.py
</pre>

<p>
Subjects and features are represented as nodes. Subject–subject similarity edges are
constructed using entropy- and complexity-based descriptors. No new measurements are created.
</p>

<h3>3. Graph Embedding (Node2Vec)</h3>

<pre>python run_node2vec.py</pre>

<p>
This produces low-dimensional subject embeddings that preserve relational structure.
</p>

<h3>4. Machine Learning on Graph Embeddings</h3>

<pre>
python export_expert_kg_embeddings.py
python benchmark_auc.py
</pre>

<h3>5. Combined Representation (Optional)</h3>

<pre>python permutation_importance_lr.py</pre>

<p>
This evaluates whether concatenating raw features and embeddings improves performance.
</p>

<h2>Key Design Principles</h2>

<ul>
    <li><strong>Same ML model everywhere:</strong> logistic regression</li>
    <li><strong>Same features everywhere:</strong> graphs built from raw features</li>
    <li><strong>Same evaluation protocol:</strong> stratified 5-fold cross-validation</li>
    <li><strong>No hidden tuning advantages:</strong> representation is the only difference</li>
</ul>

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
