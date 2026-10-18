"""Ask the market: questions answered from retrieved postings, with citations (RAG).

1. Retrieve: embed the question and find the most similar postings (optionally one family).
2. Answer: the LLM sees only those postings (contact details removed) and must cite them
   as [1], [2], ... so every claim can be checked against a real ad.
"""
import pandas as pd

from . import config, db
from .embeddings import get_embedder, load_vectors
from .privacy import scrub
from .skills import find_skills


def retrieve(path, question: str, k: int = 8, family: str | None = None, embedder=None,
             allowed: set | None = None) -> pd.DataFrame:
    embedder = embedder or get_embedder()
    refnrs, vecs = load_vectors(path, embedder.name)
    if not refnrs:
        return pd.DataFrame()
    sims = pd.Series(vecs @ embedder.embed([question])[0], index=refnrs)
    con = db.connect(path, read_only=True)
    p = con.execute("""SELECT refnr, title, employer, city, family, published, description, source, external_url
                       FROM postings""").df()
    con.close()
    p = p.assign(similarity=p["refnr"].map(sims)).dropna(subset=["similarity"])
    if family:
        p = p[p["family"] == family]
    if allowed is not None:
        p = p[p["refnr"].isin(allowed)]
    return p.sort_values("similarity", ascending=False).head(k).reset_index(drop=True)


def answer(question: str, hits: pd.DataFrame, path=None) -> str:
    if hits.empty:
        return "No postings to search yet. Generate sample data or collect postings first."
    if config.MOCK_LLM:
        skills = pd.Series([s for d in hits["description"] for s in find_skills(d)]).value_counts().head(6)
        listing = ", ".join(f"{s} ({n} of {len(hits)})" for s, n in skills.items())
        return (f"Offline mode (no LLM key): here are the {len(hits)} most relevant postings. "
                f"The skills they mention most: {listing}. Add an OPENROUTER_API_KEY for written answers.")
    from .llm import chat

    context = "\n\n".join(f"[{i + 1}] {r.title}, {r.employer}, {r.city}\n{scrub(r.description)[:1200]}"
                          for i, r in enumerate(hits.itertuples()))
    return chat([{"role": "system", "content": "Answer questions about the German job market using ONLY the job postings "
                  "provided. Cite postings as [1], [2]. If the postings don't answer the question, say so. "
                  "Be concise: under 150 words."},
                 {"role": "user", "content": f"Postings:\n{context}\n\nQuestion: {question}"}], purpose="ask", path=path)
