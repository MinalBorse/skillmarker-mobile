"""Match a CV against postings: best-fitting jobs and the skills worth learning next.

Score = half meaning, half skills:
- meaning: cosine similarity between the CV and the posting embeddings (catches related
  experience even when words differ, e.g. "Berichtswesen" vs "reporting")
- skills: the share of the posting's required skills the CV covers (exact and explainable)
The CV is scrubbed of contact details, and only sent to the LLM if you allow it.
"""
import io
import json

import numpy as np
import pandas as pd

from . import config, db
from .embeddings import get_embedder, load_vectors
from .privacy import scrub
from .skills import CANONICAL, find_skills, normalise


def read_cv(data: bytes, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    return data.decode("utf-8", errors="ignore")


def cv_skills(text: str, use_llm: bool = False, path=None) -> list[str]:
    """Skills in the CV: taxonomy matching, optionally refined by the LLM (catches skills
    described in words, e.g. "built dashboards for sales" -> Data visualisation)."""
    found = find_skills(text)
    if not use_llm or config.MOCK_LLM:
        return found
    from .llm import chat

    raw = chat([{"role": "system", "content": "List the professional skills this CV demonstrates. Use these names where "
                 f"they fit: {', '.join(CANONICAL)}. Return JSON: {{\"skills\": [...]}}"},
                {"role": "user", "content": scrub(text)[:8000]}], purpose="cv", path=path, json_mode=True)
    try:
        llm = [normalise(s) for s in json.loads(raw).get("skills", [])]
    except (ValueError, AttributeError):
        llm = []
    return sorted(set(found) | {s for s in llm if s})


def posting_skill_sets(path) -> dict[str, set]:
    con = db.connect(path, read_only=True)
    df = con.execute("SELECT refnr, skill FROM posting_skills").df()
    con.close()
    return df.groupby("refnr")["skill"].apply(set).to_dict()


def match(path, cv_text: str, have: list[str], families: list[str] | None = None, top: int | None = 20,
          embedder=None, allowed: set | None = None) -> pd.DataFrame:
    embedder = embedder or get_embedder()
    refnrs, vecs = load_vectors(path, embedder.name)
    con = db.connect(path, read_only=True)
    postings = con.execute("""SELECT refnr, title, employer, city, family, published, description, source, external_url
                              FROM postings""").df()
    con.close()
    postings = postings.set_index("refnr")
    sem = pd.Series(vecs @ embedder.embed([scrub(cv_text)[:4000]])[0], index=refnrs) if refnrs else pd.Series(dtype=float)
    skills = posting_skill_sets(path)
    have_set = set(have)
    rows = []
    for refnr in postings.index:
        if families and postings.at[refnr, "family"] not in families:
            continue
        if allowed is not None and refnr not in allowed:
            continue
        req = skills.get(refnr, set())
        rows.append({"refnr": refnr, "semantic": float(sem.get(refnr, 0.0)),
                     "coverage": len(req & have_set) / len(req) if req else 0.0,
                     "have": sorted(req & have_set), "missing": sorted(req - have_set)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    s = df["semantic"]
    df["semantic_scaled"] = (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.5
    df["score"] = 0.5 * df["semantic_scaled"] + 0.5 * df["coverage"]
    df = df.sort_values("score", ascending=False)
    if top:
        df = df.head(top)
    return df.join(postings, on="refnr").reset_index(drop=True)


def gaps(path, have: list[str], families: list[str] | None = None, top: int = 12, matches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Skills you don't have, ranked by how many target postings ask for them, plus how many of
    your best matches would become a strong fit (80%+ of skills) if you learned only that one."""
    con = db.connect(path, read_only=True)
    df = con.execute("SELECT s.refnr, s.skill, p.family FROM posting_skills s JOIN postings p USING (refnr)").df()
    n_by_fam = con.execute("SELECT family, count(*) n FROM postings GROUP BY 1").df().set_index("family")["n"]
    con.close()
    if families:
        df = df[df["family"].isin(families)]
    total = int(n_by_fam.reindex(families).sum()) if families else int(n_by_fam.sum())
    have_set = set(have)
    demand = df.groupby("skill")["refnr"].nunique().rename("postings").reset_index()
    demand = demand[~demand["skill"].isin(have_set)]
    demand["share"] = demand["postings"] / max(total, 1)
    unlock = {}
    if matches is not None and not matches.empty:
        for r in matches.itertuples():
            req = set(r.have) | set(r.missing)
            if not req or len(r.have) / len(req) >= .8:
                continue
            for sk in r.missing:
                if (len(r.have) + 1) / len(req) >= .8:
                    unlock[sk] = unlock.get(sk, 0) + 1
    demand["unlocks"] = demand["skill"].map(unlock).fillna(0).astype(int)
    return demand.sort_values(["share"], ascending=False).head(top).reset_index(drop=True)


def learning_plan(have: list[str], gap: pd.DataFrame, target: str, path=None) -> str:
    """A short, grounded plan: the LLM sees only the gap table, not the CV."""
    top = gap.head(5)
    if config.MOCK_LLM:
        lines = [f"- **{r.skill}**: asked for in {r.share:.0%} of {target} postings"
                 + (f"; would make {r.unlocks} of your best matches a strong fit" if r.unlocks else "") for r in top.itertuples()]
        return "Focus on these, in order of demand:\n" + "\n".join(lines)
    from .llm import chat

    table = "\n".join(f"{r.skill}: in {r.share:.0%} of postings, unlocks {r.unlocks} matches" for r in top.itertuples())
    return chat([{"role": "system", "content": "You are a pragmatic career coach for data and AI roles in Germany. "
                  "Using ONLY the data given, write a 4-bullet learning plan, most valuable first. Each bullet: the "
                  "skill, why (cite the numbers), and one concrete portfolio project idea. Under 150 words."},
                 {"role": "user", "content": f"Target roles: {target}\nSkills I have: {', '.join(have)}\n"
                  f"Missing skills by demand:\n{table}"}], purpose="plan", path=path)


STRONG_FIT = 0.7


def fit_overview(all_matches: pd.DataFrame) -> dict:
    """Across ALL target postings (not just the best ones): how many could this CV apply to?"""
    if all_matches.empty:
        return {"postings": 0, "strong": 0, "share": 0.0}
    strong = int((all_matches["coverage"] >= STRONG_FIT).sum())
    return {"postings": len(all_matches), "strong": strong, "share": strong / len(all_matches)}
