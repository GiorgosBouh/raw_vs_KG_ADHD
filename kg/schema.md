# Expert KG Schema (badhd) — FeatureState version

## Nodes
### (:Subject)
- id: int (UNIQUE)
- adhd: int (0/1)

### (:Diagnosis)
- name: string (UNIQUE)  e.g., "ADHD"

### (:MotorFeature)
- name: string (UNIQUE)  e.g., "ACC__sample_entropy"

### (:MotorProperty)
- name: string (UNIQUE)  e.g., "Entropy", "Variability", "Temporal"

### (:MotorBehavior)
- name: string (UNIQUE)  e.g., "Complexity", "Stability", "Rhythm"

### (:TemporalPattern)
- name: string (UNIQUE)  e.g., "Global", "ShortTerm", "LongTerm"

### (:FeatureState)
- name: string (UNIQUE)  e.g., "ACC__sample_entropy__qbin__3"
- feature: string        (MotorFeature.name)
- method: string         ("quantile" | "zscore" | "equal")
- bin: int               (0..N_BINS-1)
- label: string          ("very_low"|"low"|"mid"|"high"|"very_high"|...)
- lo: float              (lower bound)
- hi: float              (upper bound)

## Relationships
### (s:Subject)-[:HAS_DIAGNOSIS]->(d:Diagnosis)

### (s:Subject)-[:HAS_FEATURE {value: float}]->(f:MotorFeature)

### (f:MotorFeature)-[:HAS_PROPERTY]->(p:MotorProperty)

### (f:MotorFeature)-[:INDICATES_BEHAVIOR]->(b:MotorBehavior)

### (f:MotorFeature)-[:HAS_TEMPORAL_PATTERN]->(t:TemporalPattern)

### (f:MotorFeature)-[:HAS_STATE]->(fs:FeatureState)

### (s:Subject)-[:IN_STATE {value: float}]->(fs:FeatureState)

## Intuition
- HAS_FEATURE κρατάει το “continuous” value.
- FeatureState κάνει discretization (bins) ώστε πολλοί Subjects να μοιράζονται κοινά state-nodes.
- Αυτό ενισχύει το topology για Node2Vec χωρίς να “μπερδεύουμε” με dense SIMILAR_TO.
