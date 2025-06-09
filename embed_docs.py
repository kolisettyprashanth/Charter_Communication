"""
Build a FAISS vector store from txt / jsonl docs.
Run once:  python embed_docs.py
"""
import os, json, glob, pickle, re
from pathlib import Path
from tqdm import tqdm
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(r"data")
OUT_DIR  = Path(r"vector_store")
OUT_DIR.mkdir(exist_ok=True)

def load_docs():
    """Return list[str] of small text chunks."""
    chunks = []

    # --- KB & incident docs (line-level chunks)
    for txt_path in DATA_DIR.glob("*.txt"):
        with open(txt_path, encoding="utf-8") as f:
            for line in f:
                ln = line.strip()
                if len(ln) >= 20:          # ignore headers / blanks
                    chunks.append(ln)

    # --- Syslog JSONL: keep whole line
    logs_path = DATA_DIR / "sample_logs.jsonl"
    with open(logs_path) as f:
        for ln in f:
            chunks.append(ln.strip())

    return chunks

print("Collecting docs …")
docs = load_docs()
print(f"Total text chunks: {len(docs)}")

print("Embedding …")
model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(docs,
                   batch_size=64,
                   normalize_embeddings=True,
                   show_progress_bar=True).astype("float32")

print("Building FAISS index …")
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)

faiss.write_index(index, str(OUT_DIR / "faiss.index"))
with open(OUT_DIR / "docs.pkl", "wb") as f:
    pickle.dump(docs, f)

print("✅  Vector store ready in", OUT_DIR)
