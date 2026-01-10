# bADHD

This study aims to evaluate the effectiveness of graph-based approaches
for ADHD classification using weak-by-design behavioral and clinical data.

The core objective is to examine whether relational structure and knowledge
integration can overcome the predictive ceiling observed in raw machine
learning approaches.

## Models Compared

Four progressively complex modeling strategies are evaluated:

1. Raw clinical and behavioral features
   - Questionnaire scores
   - Activity / temporal summaries
   - Standard ML classifiers

2. Simple Knowledge Graphs (KGs)
   - Hand-crafted relationships
   - Basic symptom co-occurrence
   - Temporal adjacency

3. Enhanced Knowledge Graphs
   - Domain-specific expert knowledge
   - Behavioral patterns
   - Structured temporal and symptom hierarchies

4. Simulated GNN-style processing
   - Polynomial feature expansion
   - Graph-informed higher-order interactions
   - No true message passing (controlled comparison)

## Experimental Rules

- Participant-level splits only (GroupKFold / LOSO)
- No window-level random splits (strict leakage prevention)
- Raw baselines use weak aggregation by design
- Label-shuffle sanity checks are mandatory
- All experiments are runnable from project root: ~/badhd

## Scientific Rationale

The dataset is intentionally weak at the feature level.
Any performance gains must emerge from structure, relations,
and knowledge representation rather than feature engineering.

