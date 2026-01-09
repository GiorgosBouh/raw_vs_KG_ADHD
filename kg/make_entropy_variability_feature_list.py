# kg/make_entropy_variability_feature_list.py
import pandas as pd

FEAT_PATH = "data/raw/features.csv"
OUT_PATH  = "kg/features_entropy_variability.txt"

df = pd.read_csv(FEAT_PATH, engine="python", sep=None)
cols = [c.strip() for c in df.columns]

def pick(patterns):
    out = []
    for c in cols:
        if c == "ID":
            continue
        for p in patterns:
            if p in c:
                out.append(c)
                break
    return out

# 1) Entropy / complexity (ALL variants that exist)
entropy_patterns = [
    "__sample_entropy",
    "__approximate_entropy",
    "__permutation_entropy",
    "__lempel_ziv_complexity",
    "__fourier_entropy",
]

entropy_cols = pick(entropy_patterns)

# 2) Variability (STD / VAR / CV)
variability_patterns = [
    "__standard_deviation",
    "__variance",
    "__variation_coefficient",  # CV
]
variability_cols = pick(variability_patterns)

# 3) Temporal / dynamics (lags, autocorr)
temporal_patterns = [
    "__autocorrelation__lag_",
    "__partial_autocorrelation__lag_",
    '__agg_autocorrelation__f_agg_"mean"__maxlag_',
    '__agg_autocorrelation__f_agg_"median"__maxlag_',
    '__agg_autocorrelation__f_agg_"var"__maxlag_',
    "__time_reversal_asymmetry_statistic__lag_",
]
temporal_cols = pick(temporal_patterns)

# de-dup while preserving order
seen = set()
final = []
for c in (entropy_cols + variability_cols + temporal_cols):
    if c not in seen:
        seen.add(c)
        final.append(c)

final = sorted(final)  # σταθερή σειρά για reproducibility

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for c in final:
        f.write(c + "\n")

print("✅ Wrote feature list:", OUT_PATH)
print("Total features:", len(final))
print("Breakdown:",
      "entropy=", len(entropy_cols),
      "variability=", len(variability_cols),
      "temporal=", len(temporal_cols))
