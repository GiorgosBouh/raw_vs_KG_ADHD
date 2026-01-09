import os
from pathlib import Path

import numpy as np
import pandas as pd
from neo4j import GraphDatabase

from kg.neo4j_conn import get_driver

FEATURES_PATH = os.getenv("FEATURES_PATH", "data/raw/features.csv")
PATIENT_PATH  = os.getenv("PATIENT_PATH",  "data/raw/patient_info.csv")

# feature list file (one feature name per line)
FEATURE_LIST = os.getenv("FEATURE_LIST", "kg/features_entropy_variability.txt")

# target handling
TARGET_COL_IN_PATIENT = os.getenv("TARGET_COL_IN_PATIENT", "ADHD")  # patient_info.csv has ADHD
TARGET_NAME_NORMALIZED = "adhd"

# optional binning -> FeatureState
BIN_METHOD = os.getenv("BIN_METHOD", "none").lower()  # none|quantile|uniform
N_BINS = int(os.getenv("N_BINS", "5"))
CREATE_HAS_STATE = os.getenv("CREATE_HAS_STATE", "0") == "1"  # default OFF (we don't want Feature->State bridges)

BATCH = int(os.getenv("BATCH", "5000"))


def read_list(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Feature list file not found: {path}")
    items = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            items.append(s)
    # unique, preserve order
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def expert_mapping(feature_name: str) -> tuple[str, str, str]:
    """
    Returns: (MotorProperty, MotorBehavior, TemporalPattern)
    Keep it small/consistent so the KG doesn't explode in taxonomy nodes.
    """
    name = feature_name.lower()

    # MotorProperty
    if "entropy" in name or "lempel_ziv" in name:
        prop = "Entropy/Complexity"
        beh  = "Irregularity"
    elif "cv" in name or "__std" in name or "std" in name:
        prop = "Variability"
        beh  = "Instability"
    else:
        prop = "Temporal"
        beh  = "Rhythmicity"

    # TemporalPattern
    if ("__mean" in feature_name) or ('"mean"' in feature_name) or ("feature_\"mean\"" in feature_name) or ("f_agg_\"mean\"" in feature_name):
        tmp = "Mean"
    elif ("__std" in name) or ("std" in name):
        tmp = "Dispersion"
    else:
        tmp = "Temporal"

    return prop, beh, tmp


def chunked(iterable, n):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def main():
    print(f"Reading features from: {FEATURES_PATH}")
    print(f"Reading patient info from: {PATIENT_PATH}")
    print("---------------")

    feat = pd.read_csv(FEATURES_PATH, engine="python", sep=None)
    pat  = pd.read_csv(PATIENT_PATH,  engine="python", sep=None)

    feat.columns = [c.strip() for c in feat.columns]
    pat.columns  = [c.strip() for c in pat.columns]

    # ID must exist
    if "ID" not in feat.columns:
        raise KeyError(f"'ID' column not found in {FEATURES_PATH}. Found: {list(feat.columns)}")
    if "ID" not in pat.columns:
        raise KeyError(f"'ID' column not found in {PATIENT_PATH}. Found: {list(pat.columns)}")

    # normalize ID
    feat["ID"] = pd.to_numeric(feat["ID"], errors="coerce").astype("Int64")
    pat["ID"]  = pd.to_numeric(pat["ID"],  errors="coerce").astype("Int64")

    feat = feat.dropna(subset=["ID"]).copy()
    pat  = pat.dropna(subset=["ID"]).copy()

    # target normalize
    if TARGET_COL_IN_PATIENT not in pat.columns:
        raise KeyError(f"'{TARGET_COL_IN_PATIENT}' target column not found in {PATIENT_PATH}. Found: {list(pat.columns)}")

    pat[TARGET_NAME_NORMALIZED] = pd.to_numeric(pat[TARGET_COL_IN_PATIENT], errors="coerce").astype("Int64")

    # keep only rows with ID + target
    pat = pat.dropna(subset=["ID", TARGET_NAME_NORMALIZED]).copy()

    # IMPORTANT: deduplicate patient rows by ID
    pat = pat.drop_duplicates(subset=["ID"], keep="first").copy()

    # wanted features
    wanted = read_list(FEATURE_LIST)

    missing = [c for c in wanted if c not in feat.columns]
    if missing:
        raise KeyError(f"Some listed features are missing in features.csv:\n{missing}")

    # restrict to common IDs (this is crucial for correct label counts)
    common_ids = sorted(set(feat["ID"].dropna().astype(int)).intersection(set(pat["ID"].dropna().astype(int))))
    print(f"Subjects (common): {len(common_ids)}")
    print(f"Expert feature list used: {len(wanted)}")
    print(f"Target column: {TARGET_COL_IN_PATIENT} → {TARGET_NAME_NORMALIZED}")

    pat = pat[pat["ID"].astype(int).isin(common_ids)].copy()
    feat = feat[feat["ID"].astype(int).isin(common_ids)].copy()

    print("Label counts (final, after common_ids + dedup):")
    print(pat[TARGET_NAME_NORMALIZED].value_counts(dropna=False))
    print("---------------")

    # subset features for those IDs
    feat_sub = feat[["ID"] + wanted].copy()

    # coerce wanted -> numeric
    for col in wanted:
        feat_sub[col] = pd.to_numeric(feat_sub[col], errors="coerce")

    # create subject rows
    subject_rows = [{"id": int(r.ID), "adhd": int(r.adhd)} for r in pat[["ID", TARGET_NAME_NORMALIZED]].itertuples(index=False)]

    # build HAS_FEATURE rows (avoid itertuples getattr because columns may contain quotes)
    feature_rows = []
    for rec in feat_sub.to_dict(orient="records"):
        sid = int(rec["ID"])
        for col in wanted:
            val = rec.get(col, None)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue

            prop, beh, tmp = expert_mapping(col)
            feature_rows.append({
                "id": sid,
                "feature": col,
                "value": float(val),
                "property": prop,
                "behavior": beh,
                "temporal": tmp
            })

    # OPTIONAL: FeatureState (binning)
    state_rows = []
    if BIN_METHOD != "none":
        # compute bins per feature across common IDs
        for col in wanted:
            series = feat_sub[["ID", col]].dropna().copy()
            if series.empty:
                continue

            x = series[col].astype(float).values
            ids = series["ID"].astype(int).values

            try:
                if BIN_METHOD == "quantile":
                    bins = pd.qcut(x, q=N_BINS, labels=[f"Q{i+1}" for i in range(N_BINS)], duplicates="drop")
                elif BIN_METHOD == "uniform":
                    bins = pd.cut(x, bins=N_BINS, labels=[f"B{i+1}" for i in range(N_BINS)], include_lowest=True)
                else:
                    raise ValueError("BIN_METHOD must be none|quantile|uniform")
            except Exception:
                # fallback: skip binning for this feature if qcut fails
                continue

            for sid, b in zip(ids, bins):
                if pd.isna(b):
                    continue
                state_name = f"{col}::{str(b)}"
                state_rows.append({
                    "id": int(sid),
                    "feature": col,
                    "state": state_name
                })

    driver = get_driver()
    with driver.session() as session:
        # wipe? (we leave wiping to the user)
        # Constraints assumed created already.

        # 1) Subjects + Diagnosis
        session.run("""
        UNWIND $rows AS row
        MERGE (s:Subject {id: row.id})
        SET s.adhd = row.adhd
        WITH s
        MERGE (d:Diagnosis {name: 'ADHD'})
        MERGE (s)-[:HAS_DIAGNOSIS]->(d)
        """, rows=subject_rows)

        # 2) Features taxonomy + HAS_FEATURE
        for batch in chunked(feature_rows, BATCH):
            session.run("""
            UNWIND $rows AS row
            MERGE (s:Subject {id: row.id})

            MERGE (f:MotorFeature {name: row.feature})

            MERGE (p:MotorProperty {name: row.property})
            MERGE (b:MotorBehavior {name: row.behavior})
            MERGE (t:TemporalPattern {name: row.temporal})

            MERGE (f)-[:HAS_PROPERTY]->(p)
            MERGE (f)-[:INDICATES_BEHAVIOR]->(b)
            MERGE (f)-[:HAS_TEMPORAL_PATTERN]->(t)

            MERGE (s)-[r:HAS_FEATURE]->(f)
            SET r.value = row.value
            """, rows=batch)

        # 3) FeatureState layer (Subject -> State), optional Feature -> State (OFF by default)
        if state_rows:
            # create FeatureState nodes and IN_STATE edges
            for batch in chunked(state_rows, BATCH):
                session.run("""
                UNWIND $rows AS row
                MERGE (s:Subject {id: row.id})
                MERGE (fs:FeatureState {name: row.state})
                MERGE (s)-[:IN_STATE]->(fs)
                """, rows=batch)

            if CREATE_HAS_STATE:
                # link Feature -> State (not recommended for your case unless you really want it)
                # note: needs the FeatureState nodes created already
                for batch in chunked(state_rows, BATCH):
                    session.run("""
                    UNWIND $rows AS row
                    MERGE (f:MotorFeature {name: row.feature})
                    MERGE (fs:FeatureState {name: row.state})
                    MERGE (f)-[:HAS_STATE]->(fs)
                    """, rows=batch)

    driver.close()

    if BIN_METHOD == "none":
        print("✅ Expert Knowledge Graph successfully built.")
    else:
        print("✅ Expert Knowledge Graph successfully built (FeatureState enabled).")


if __name__ == "__main__":
    main()
