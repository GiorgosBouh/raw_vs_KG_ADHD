import pickle
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

# paths
GRAPH_PATH = "data/processed/badhd_cpt_graph.pkl"
OUT_PATH = "data/processed/badhd_node2vec_subject_embeddings.csv"

# load graph
with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)

print("Warning: running Node2Vec on a full graph is transductive (includes all subjects).")
print("Graph loaded:",
      G.number_of_nodes(), "nodes,",
      G.number_of_edges(), "edges")

# Node2Vec
node2vec = Node2Vec(
    G,
    dimensions=64,
    walk_length=30,
    num_walks=200,
    p=1.0,
    q=1.0,
    workers=8,
    seed=42
)

model = node2vec.fit(
    window=10,
    min_count=1,
    batch_words=4
)

# --- keep ONLY Subject nodes ---
subject_nodes = [n for n in G.nodes if n.startswith("Subject_")]

rows = []
for n in subject_nodes:
    emb = model.wv[n]
    sid = int(n.split("_")[1])
    rows.append(
        {"ID": sid, **{f"emb_{i}": emb[i] for i in range(len(emb))}}
    )

df = pd.DataFrame(rows).sort_values("ID")
df.to_csv(OUT_PATH, index=False)

print("✅ Subject embeddings saved:", df.shape)
print("Saved to:", OUT_PATH)
