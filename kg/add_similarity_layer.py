import os
import numpy as np
import pandas as pd

from kg.neo4j_conn import get_driver  # υποθέτω έχεις αυτό

FEAT_PATH = os.getenv("FEAT_PATH", "data/raw/features.csv")
PAT_PATH  = os.getenv("PAT_PATH",  "data/raw/patient_info.csv")
FEATURE_LIST = os.getenv("FEATURE_LIST", "kg/features_entropy_variability.txt")

SIM_K = int(os.getenv("SIM_K", "5"))
SIM_MODE = os.getenv("SIM_MODE", "entropy").strip().lower()  # entropy | all
CLEAR_OLD = os.getenv("CLEAR_OLD", "1") == "1"

ENTROPY_KEYS = [
    "sample_entropy",
    "approximate_entropy",
    "permutation_entropy",
    "lempel_ziv_complexity",
    "fourier_entropy",
]

def read_list(path: str):
    with open(path, "r", encoding="utf-8") as f:
        out = []
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out

def is_entropy_feature(col: str) -> bool:
    return any(k in col for k in ENTROPY_KEYS)

def main():
    feat = pd.read_csv(FEAT_PATH, engine="python", sep=None)
    pat  = pd.read_csv(PAT_PATH,  engine="python", sep=None)

    feat.columns = [c.strip() for c in feat.columns]
    pat.columns  = [c.strip() for c in pat.columns]

    wanted = read_list(FEATURE_LIST)

    if SIM_MODE == "entropy":
        sim_feats = [c for c in wanted if is_entropy_feature(c)]
        mode_used = "entropy"
    elif SIM_MODE == "all":
        sim_feats = list(wanted)
        mode_used = "all"
    else:
        # fallback
        sim_feats = [c for c in wanted if is_entropy_feature(c)]
        mode_used = "entropy-only (fallback)"

    # κρατάμε μόνο ό,τι υπάρχει
    sim_feats = [c for c in sim_feats if c in feat.columns]

    # κοινά subjects
    common_ids = sorted(set(feat["ID"]).intersection(set(pat["ID"])))
    df = feat[feat["ID"].isin(common_ids)][["ID"] + sim_feats].copy()

    # numeric
    for c in sim_feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # fill NaNs per-feature (median) για cosine
    X = df[sim_feats].to_numpy(dtype=float)
    # median per col
    med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])

    # standardize features (z-score) για να μην κυριαρχεί scale
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    Xz = (X - mu) / sd

    # cosine similarity matrix
    # normalize rows
    norms = np.linalg.norm(Xz, axis=1) + 1e-9
    Xn = Xz / norms[:, None]
    S = Xn @ Xn.T  # (n,n)

    ids = df["ID"].astype(int).tolist()
    n = len(ids)

    # topk per row (exclude self)
    edges = []
    for i in range(n):
        sims = S[i].copy()
        sims[i] = -1.0
        # argpartition for topk
        k = min(SIM_K, n - 1)
        idx = np.argpartition(-sims, k)[:k]
        # sort by similarity desc
        idx = idx[np.argsort(-sims[idx])]
        rank = 1
        for j in idx:
            edges.append((ids[i], ids[j], float(sims[j]), rank))
            rank += 1

    driver = get_driver()
    with driver.session() as session:
        if CLEAR_OLD:
            session.run("MATCH ()-[r:SIMILAR_TO]->() DELETE r")

        # write edges
        q = """
        MATCH (a:Subject {id:$a}), (b:Subject {id:$b})
        MERGE (a)-[r:SIMILAR_TO]->(b)
        SET r.cosine=$cos, r.rank=$rank
        """
        for a, b, cos, rank in edges:
            session.run(q, a=int(a), b=int(b), cos=float(cos), rank=int(rank))

    print(f"✅ SIMILAR_TO layer built with k={SIM_K} (edges written: {len(edges)})")
    print(f"Similarity mode: {mode_used}")
    print(f"Similarity features used: {len(sim_feats)}")
    print(f"Expected directed edges ~ {len(common_ids)*min(SIM_K, len(common_ids)-1)}")
    print(f"Subjects used: {len(common_ids)}")

if __name__ == "__main__":
    main()
