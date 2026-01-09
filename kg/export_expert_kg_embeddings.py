import os
import random
import numpy as np
import pandas as pd
import networkx as nx

from node2vec import Node2Vec
from kg.neo4j_conn import get_driver

OUT_PATH = os.getenv("OUT_PATH", "data/processed/expertkg_node2vec_patient_embeddings.csv")

INCLUDE_SIM = os.getenv("INCLUDE_SIM", "1") == "1"
SUBJECT_ONLY = os.getenv("SUBJECT_ONLY", "0") == "1"

# ΚΡΙΣΙΜΟ: default = 0 (UNWEIGHTED), γιατί εκεί είδες το boost ~0.65
SIM_WEIGHT = os.getenv("SIM_WEIGHT", "0") == "1"  # 1 -> weight=cosine, 0 -> weight=1

UNDIRECT_SIM = os.getenv("UNDIRECT_SIM", "1") == "1"  # συνήθως βοηθάει

WORKERS = int(os.getenv("WORKERS", "1"))
SEED = int(os.getenv("SEED", "42"))
NUM_WALKS = int(os.getenv("NUM_WALKS", "20"))
WALK_LENGTH = int(os.getenv("WALK_LENGTH", "20"))
DIM = int(os.getenv("DIM", "64"))
WINDOW = int(os.getenv("WINDOW", "10"))

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

def main():
    driver = get_driver()

    # Χτίζουμε NetworkX graph
    # Αν θες directed, άλλαξε σε DiGraph. Για node2vec συνήθως Graph είναι πιο “σταθερό”.
    G = nx.Graph()

    with driver.session() as session:
        # subjects
        res = session.run("MATCH (s:Subject) RETURN s.id AS id, s.adhd AS adhd")
        subjects = [(int(r["id"]), int(r["adhd"])) for r in res]
        subj_ids = [sid for sid, _ in subjects]

        # add nodes
        for sid, _ in subjects:
            G.add_node(f"S{sid}", kind="Subject", sid=sid)

        if not SUBJECT_ONLY:
            # features + has_feature edges
            q = """
            MATCH (s:Subject)-[r:HAS_FEATURE]->(f:MotorFeature)
            RETURN s.id AS sid, f.name AS fname
            """
            res = session.run(q)
            for r in res:
                sid = int(r["sid"])
                fname = str(r["fname"])
                fn = f"F|{fname}"
                if not G.has_node(fn):
                    G.add_node(fn, kind="Feature", name=fname)
                G.add_edge(f"S{sid}", fn, rel="HAS_FEATURE", weight=1.0)

            # feature->property, feature->temporal, property->behavior
            q2 = """
            MATCH (f:MotorFeature)-[:HAS_PROPERTY]->(p:MotorProperty)
            RETURN f.name AS fname, p.name AS pname
            """
            for r in session.run(q2):
                fn = f"F|{r['fname']}"
                pn = f"P|{r['pname']}"
                if not G.has_node(pn):
                    G.add_node(pn, kind="Property", name=str(r["pname"]))
                if G.has_node(fn):
                    G.add_edge(fn, pn, rel="HAS_PROPERTY", weight=1.0)

            q3 = """
            MATCH (f:MotorFeature)-[:HAS_TEMPORAL_PATTERN]->(t:TemporalPattern)
            RETURN f.name AS fname, t.name AS tname
            """
            for r in session.run(q3):
                fn = f"F|{r['fname']}"
                tn = f"T|{r['tname']}"
                if not G.has_node(tn):
                    G.add_node(tn, kind="Temporal", name=str(r["tname"]))
                if G.has_node(fn):
                    G.add_edge(fn, tn, rel="HAS_TEMPORAL_PATTERN", weight=1.0)

            q4 = """
            MATCH (p:MotorProperty)-[:INDICATES_BEHAVIOR]->(b:MotorBehavior)
            RETURN p.name AS pname, b.name AS bname
            """
            for r in session.run(q4):
                pn = f"P|{r['pname']}"
                bn = f"B|{r['bname']}"
                if not G.has_node(bn):
                    G.add_node(bn, kind="Behavior", name=str(r["bname"]))
                if G.has_node(pn):
                    G.add_edge(pn, bn, rel="INDICATES_BEHAVIOR", weight=1.0)

        # similarity edges
        sim_count = 0
        if INCLUDE_SIM:
            qsim = """
            MATCH (a:Subject)-[r:SIMILAR_TO]->(b:Subject)
            RETURN a.id AS a, b.id AS b, r.cosine AS cosine, r.rank AS rank
            """
            for r in session.run(qsim):
                a = int(r["a"])
                b = int(r["b"])
                cos = float(r["cosine"]) if r["cosine"] is not None else 1.0
                w = cos if SIM_WEIGHT else 1.0

                na = f"S{a}"
                nb = f"S{b}"
                if G.has_node(na) and G.has_node(nb):
                    G.add_edge(na, nb, rel="SIMILAR_TO", weight=w)
                    sim_count += 1
                    if UNDIRECT_SIM:
                        # nx.Graph ήδη το κάνει undirected, αλλά το κρατάω explicit για καθαρότητα
                        pass

        if INCLUDE_SIM:
            if SIM_WEIGHT:
                print(f"✅ SIMILAR_TO included: {sim_count} edges (weight=cosine)")
            else:
                print(f"✅ SIMILAR_TO included: {sim_count} edges (UNWEIGHTED)")
        if SUBJECT_ONLY:
            print("✅ SUBJECT_ONLY=1 (graph built from SIMILAR_TO only)")

    # node2vec
    node2vec = Node2Vec(
        G,
        dimensions=DIM,
        walk_length=WALK_LENGTH,
        num_walks=NUM_WALKS,
        workers=WORKERS,
        seed=SEED,
        weight_key="weight",
    )
    model = node2vec.fit(window=WINDOW, min_count=1, batch_words=64, seed=SEED)

    # export patient embeddings (subjects μόνο)
    rows = []
    for sid, adhd in subjects:
        key = f"S{sid}"
        vec = model.wv[key]
        rows.append([sid, adhd] + vec.tolist())

    cols = ["ID", "adhd"] + [f"emb_{i}" for i in range(DIM)]
    out = pd.DataFrame(rows, columns=cols)
    out.to_csv(OUT_PATH, index=False)

    print(f"Graph nodes: {G.number_of_nodes()}")
    print(f"Graph edges: {G.number_of_edges()}")
    print(f"✅ Saved patient embeddings to: {OUT_PATH}")
    print(f"Shape: {out.shape}")

if __name__ == "__main__":
    main()
