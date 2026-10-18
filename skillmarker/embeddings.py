"""Text embeddings for matching CVs to postings and for "Ask the market" retrieval.

Default: a small multilingual model (German and English) run locally on the CPU through
fastembed, no GPU and no API cost; it downloads once (about 220 MB). If it isn't
available (offline, tests), a hashing embedder keeps everything working with lower quality.
"""
import hashlib
import re

import numpy as np
import pandas as pd

from . import config, db
from .privacy import scrub


class HashEmbedder:
    """Offline fallback: hashed words and character trigrams. Fast, free, rough."""
    name = "hash-384"

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            words = re.findall(r"[a-zäöüß0-9+#/]+", (t or "").lower())
            grams = words + [w[j:j + 3] for w in words for j in range(max(1, len(w) - 2))]
            for g in grams:
                h = int(hashlib.md5(g.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 9) & 1 else -1.0
        return normalise(out)


class FastEmbedder:
    def __init__(self, model: str):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model)
        self.name = model

    def embed(self, texts: list[str]) -> np.ndarray:
        return normalise(np.array(list(self.model.embed(texts)), dtype=np.float32))


def normalise(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(n == 0, 1, n)


_cached = None


def get_embedder():
    """fastembed if possible, otherwise the hashing fallback (logged in `.fallback_reason`)."""
    global _cached
    if _cached is not None:
        return _cached
    if config.EMBED_BACKEND in ("auto", "fastembed"):
        try:
            _cached = FastEmbedder(config.EMBED_MODEL)
            return _cached
        except Exception as e:  # noqa: BLE001 - no internet for the first download, etc.
            if config.EMBED_BACKEND == "fastembed":
                raise
            _cached = HashEmbedder()
            _cached.fallback_reason = f"{type(e).__name__}: {str(e)[:120]}"
            return _cached
    _cached = HashEmbedder()
    return _cached


def posting_text(title: str, description: str) -> str:
    return f"{title}\n{scrub(description)[:1500]}"


def ensure_embeddings(path, embedder=None, batch: int = 64) -> int:
    """Embed postings that don't have a vector for this model yet. Returns how many were added."""
    embedder = embedder or get_embedder()
    con = db.connect(path)
    todo = con.execute("""SELECT refnr, title, description FROM postings
                          WHERE refnr NOT IN (SELECT refnr FROM embeddings WHERE model = ?)""", [embedder.name]).df()
    for i in range(0, len(todo), batch):
        chunk = todo.iloc[i:i + batch]
        vecs = embedder.embed([posting_text(t, d) for t, d in zip(chunk["title"], chunk["description"])])
        db.upsert(con, "embeddings", pd.DataFrame({"refnr": chunk["refnr"].values, "model": embedder.name,
                                                   "vector": [v.tolist() for v in vecs]}), ["refnr"])
    con.close()
    return len(todo)


def main():
    """python -m skillmarker.embeddings [--source live|sample]: precompute vectors (the daily collector
    runs this, so the hosted app doesn't have to embed thousands of postings on a small free server)."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["sample", "live"], default="live")
    args = ap.parse_args()
    emb = get_embedder()
    n = ensure_embeddings(config.SAMPLE_DB if args.source == "sample" else config.LIVE_DB, emb)
    print(f"Embedded {n} postings with {emb.name}" + (f" (fallback: {emb.fallback_reason})"
                                                     if getattr(emb, "fallback_reason", None) else ""))


def load_vectors(path, model: str) -> tuple[list[str], np.ndarray]:
    con = db.connect(path, read_only=True)
    df = con.execute("SELECT refnr, vector FROM embeddings WHERE model = ?", [model]).df()
    con.close()
    if df.empty:
        return [], np.zeros((0, 1), dtype=np.float32)
    return df["refnr"].tolist(), normalise(np.vstack(df["vector"].map(np.asarray).values).astype(np.float32))


if __name__ == "__main__":
    main()
