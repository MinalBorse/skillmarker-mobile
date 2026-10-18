"""Market analysis: which skills are wanted, how that changes, and what roles look like."""
import json

import numpy as np
import pandas as pd

from . import db


def load(path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Postings joined with their extraction, plus a long table of (posting, skill)."""
    con = db.connect(path, read_only=True)
    try:
        # In scope: an analytics/AI role family, not a duplicate of a posting from another source,
        # and full-time, freelance or not stated (the source's own label wins over the AI's reading).
        p = con.execute("""
            SELECT p.* REPLACE (coalesce(p.published, CAST(p.first_seen AS DATE)) AS published,
                                CASE WHEN p.employment IN ('full_time', 'freelance', 'part_time', 'internship') THEN p.employment
                                     ELSE coalesce(e.employment, 'unknown') END AS employment,
                                coalesce(nullif(p.work_model, 'unknown'), e.remote, 'unknown') AS work_model),
                   e.method, e.seniority, e.salary_min, e.salary_max, e.german_level, e.english_required,
                   e.remote, e.years_experience, e.other_skills
            FROM postings p LEFT JOIN extractions e USING (refnr)
            WHERE p.family IS NOT NULL AND p.dup_of IS NULL""").df()
        # Postings without a publication date (some API versions omit it) use the day we first saw them.
        s = con.execute("""SELECT s.refnr, s.skill, s.category, p.family,
                                  coalesce(p.published, CAST(p.first_seen AS DATE)) AS published
                           FROM posting_skills s JOIN postings p USING (refnr)""").df()
    finally:
        con.close()
    p = p[~p["employment"].isin(["part_time", "internship"]) & (p["seniority"].fillna("") != "working_student")]
    s = s[s["refnr"].isin(p["refnr"])]
    p["published"] = pd.to_datetime(p["published"])
    s["published"] = pd.to_datetime(s["published"])
    return p.reset_index(drop=True), s.reset_index(drop=True)


def skill_demand(p: pd.DataFrame, s: pd.DataFrame, family: str | None = None, top: int = 20) -> pd.DataFrame:
    """Share of postings that ask for each skill."""
    if family:
        p, s = p[p["family"] == family], s[s["family"] == family]
    n = p["refnr"].nunique()
    if n == 0:
        return pd.DataFrame(columns=["skill", "category", "postings", "share"])
    d = s.groupby(["skill", "category"])["refnr"].nunique().rename("postings").reset_index()
    d["share"] = d["postings"] / n
    return d.sort_values("postings", ascending=False).head(top).reset_index(drop=True)


def rising(p: pd.DataFrame, s: pd.DataFrame, recent_days: int = 28, min_postings: int = 15) -> pd.DataFrame:
    """Skills whose share of postings grew most: last `recent_days` vs everything before."""
    cut = p["published"].max() - pd.Timedelta(days=recent_days)
    recent, before = p[p["published"] > cut], p[p["published"] <= cut]
    if len(recent) < 20 or len(before) < 20:
        return pd.DataFrame(columns=["skill", "share_before", "share_recent", "change_pts", "postings"])
    def share(ps):
        return s[s["refnr"].isin(ps["refnr"])].groupby("skill")["refnr"].nunique() / len(ps)
    d = pd.DataFrame({"share_before": share(before), "share_recent": share(recent)}).fillna(0)
    d["postings"] = s.groupby("skill")["refnr"].nunique()
    d = d[d["postings"] >= min_postings]
    d["change_pts"] = (d["share_recent"] - d["share_before"]) * 100
    return d.sort_values("change_pts", ascending=False).reset_index().rename(columns={"index": "skill"})


def weekly(p: pd.DataFrame) -> pd.DataFrame:
    w = p.assign(week=p["published"].dt.to_period("W").dt.start_time)
    return w.groupby(["week", "family"]).size().rename("postings").reset_index()


def skill_trend(p: pd.DataFrame, s: pd.DataFrame, skills: list[str]) -> pd.DataFrame:
    """Monthly share of postings mentioning each skill."""
    m = p.assign(month=p["published"].dt.to_period("M").dt.start_time)
    totals = m.groupby("month")["refnr"].nunique()
    sm = s[s["skill"].isin(skills)].merge(m[["refnr", "month"]], on="refnr")
    d = sm.groupby(["month", "skill"])["refnr"].nunique().rename("postings").reset_index()
    d["share"] = d["postings"] / d["month"].map(totals)
    return d


def seniority_mix(p: pd.DataFrame) -> pd.DataFrame:
    order = ["working_student", "junior", "mid", "senior", "lead"]
    d = p[p["seniority"].isin(order)].groupby(["family", "seniority"]).size().rename("postings").reset_index()
    d["share"] = d["postings"] / d.groupby("family")["postings"].transform("sum")
    d["seniority"] = pd.Categorical(d["seniority"], order, ordered=True)
    return d.sort_values(["family", "seniority"])


def salaries(p: pd.DataFrame) -> pd.DataFrame:
    """Median stated salary (midpoint of the range) per family and level, with sample size."""
    d = p.dropna(subset=["salary_min"]).copy()
    d["salary_max"] = d["salary_max"].fillna(d["salary_min"])
    d["mid"] = (d["salary_min"] + d["salary_max"]) / 2
    out = d.groupby(["family", "seniority"]).agg(median=("mid", "median"), low=("mid", lambda x: x.quantile(.25)),
                                                  high=("mid", lambda x: x.quantile(.75)), n=("mid", "size")).reset_index()
    return out


def salary_disclosure(p: pd.DataFrame) -> float:
    return float(p["salary_min"].notna().mean()) if len(p) else float("nan")


def language(p: pd.DataFrame) -> pd.DataFrame:
    d = p.groupby("family")["german_level"].value_counts(normalize=True).rename("share").reset_index()
    return d.pivot(index="family", columns="german_level", values="share").fillna(0).reset_index()


def remote_mix(p: pd.DataFrame) -> pd.DataFrame:
    return p["remote"].value_counts(normalize=True).rename("share").reset_index()


def cities(p: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    d = p.dropna(subset=["lat", "lon"]).groupby("city").agg(postings=("refnr", "size"), lat=("lat", "median"),
                                                            lon=("lon", "median")).reset_index()
    return d.sort_values("postings", ascending=False).head(top)


def cooccurring(s: pd.DataFrame, skill: str, top: int = 8) -> pd.DataFrame:
    """Skills that appear together with `skill` more often than chance (lift > 1)."""
    with_skill = set(s.loc[s["skill"] == skill, "refnr"])
    n_all = s["refnr"].nunique()
    if not with_skill:
        return pd.DataFrame(columns=["skill", "together", "lift"])
    base = s.groupby("skill")["refnr"].nunique() / n_all
    together = s[s["refnr"].isin(with_skill) & (s["skill"] != skill)].groupby("skill")["refnr"].nunique()
    d = pd.DataFrame({"together": together / len(with_skill)})
    d["lift"] = d["together"] / base.reindex(d.index)
    return d[d["together"] >= .1].sort_values("lift", ascending=False).head(top).reset_index()


def unknown_skills(p: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Skills the LLM found that aren't in the taxonomy yet: candidates to add."""
    vals = []
    for raw in p["other_skills"].dropna():
        try:
            vals += [str(x).strip() for x in json.loads(raw)]
        except (ValueError, TypeError):
            pass
    if not vals:
        return pd.DataFrame(columns=["skill", "postings"])
    return pd.Series(vals).str.strip().value_counts().head(top).rename_axis("skill").reset_index(name="postings")


def headline(p: pd.DataFrame, s: pd.DataFrame) -> dict:
    d = skill_demand(p, s, top=1)
    return {"postings": int(len(p)), "employers": int(p["employer"].nunique()),
            "first": p["published"].min(), "last": p["published"].max(),
            "top_skill": d["skill"].iloc[0] if len(d) else None, "top_share": float(d["share"].iloc[0]) if len(d) else np.nan,
            "families": int(p["family"].nunique())}
