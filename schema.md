# Expert Knowledge Graph for bADHD (v2)

## Nodes
- (:Subject {id:int, adhd:int})
- (:Diagnosis {name})
- (:MotorFeature {name})
- (:MotorProperty {name})
- (:MotorBehavior {name})
- (:TemporalPattern {name})

## Relationships
- (:Subject)-[:HAS_FEATURE {value:float}]->(:MotorFeature)
- (:MotorFeature)-[:HAS_PROPERTY]->(:MotorProperty)
- (:MotorFeature)-[:EXPRESSES]->(:MotorBehavior)
- (:MotorFeature)-[:HAS_TEMPORAL]->(:TemporalPattern)
- (:Subject)-[:HAS_DIAGNOSIS]->(:Diagnosis)

## Similarity Layer (derived)
- (:Subject)-[:SIMILAR_TO {cosine:float, rank:int}]->(:Subject)

## Notes
- SIMILAR_TO is built **after** embeddings, using **cosine similarity**
- k = 5 (fixed)
- No target leakage: similarity computed only from expert KG embeddings
