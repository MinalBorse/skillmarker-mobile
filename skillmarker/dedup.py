"""Find the same job posted in several places (company page, Bundesagentur, aggregators).

Without this, a job found three times would count three times in the skill statistics.
Two postings are the same job when the employer matches (ignoring legal forms such as GmbH)
and the titles are near-identical (ignoring "(m/w/d)" and similar), in the same city or
where either is remote. The posting with the longest description is kept; the others get
`dup_of` set and are left out of all views (they stay in the database).
"""
import difflib
import re

from . import db

LEGAL = re.compile(r"\b(?:gmbh|ag|se|kg|kgaa|ug|mbh|co|inc|ltd|llc|group|holding|deutschland|germany)\b\.?|&", re.IGNORECASE)
GENDER = re.compile(r"\((?:[mwdfx]\s*[/|,]\s*){1,3}[mwdfx]\)|\(all genders?\)|\(gn\)|\*in\b", re.IGNORECASE)


def norm_employer(name: str) -> str:
    return re.sub(r"[^a-z0-9äöü]+", " ", LEGAL.sub(" ", (name or "").lower())).strip()


def norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9äöü+#]+", " ", GENDER.sub(" ", (title or "").lower())).strip()


def same_job(a, b) -> bool:
    if difflib.SequenceMatcher(None, a.t, b.t).ratio() < 0.92:
        return False
    if a.work_model == "remote" or b.work_model == "remote" or not a.city or not b.city:
        return True
    return str(a.city).lower()[:6] == str(b.city).lower()[:6]


def run(path) -> int:
    con = db.connect(path)
    df = con.execute("""SELECT refnr, employer, title, city, work_model, length(coalesce(description, '')) AS n
                        FROM postings WHERE family IS NOT NULL AND (source <> 'sample' OR source IS NULL)""").df()
    if df.empty:
        con.close()
        return 0
    df["e"], df["t"] = df["employer"].map(norm_employer), df["title"].map(norm_title)
    dup_of = {}
    for _, g in df[df["e"] != ""].groupby("e"):
        rows = sorted(g.itertuples(), key=lambda r: -r.n)       # longest description first = the one we keep
        kept = []
        for r in rows:
            match = next((k for k in kept if same_job(k, r)), None)
            if match is None:
                kept.append(r)
            else:
                dup_of[r.refnr] = match.refnr
    con.execute("UPDATE postings SET dup_of = NULL")
    if dup_of:
        import pandas as pd
        con.register("_d", pd.DataFrame({"refnr": list(dup_of), "dup": list(dup_of.values())}))
        con.execute("UPDATE postings p SET dup_of = d.dup FROM _d d WHERE p.refnr = d.refnr")
    con.close()
    return len(dup_of)
