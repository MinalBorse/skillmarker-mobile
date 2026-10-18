"""Extract skills, seniority, salary, language and remote policy from job postings.

    python -m skillmarker.extract                         # LLM if a key is set, else rules
    python -m skillmarker.extract --method llm --limit 200 --max-cost 0.50
    python -m skillmarker.extract --method rules          # free, instant, less precise
    python -m skillmarker.extract --source live

Two extractors with the same output:
- Rules: regular expressions over the skill taxonomy. Free and instant, but it can't tell a
  requirement from a mention ("we are an AWS partner", "no SAP knowledge needed").
- LLM: reads the posting like a person would and returns structured JSON, validated with
  Pydantic and normalised to the taxonomy. Skills outside the taxonomy are kept, so new
  tools show up in the data instead of being silently dropped.

Running an LLM over thousands of postings costs money, so every call is logged with its
tokens and cost, results are cached by a hash of the posting text (re-running costs nothing
for unchanged postings), and --max-cost stops the run at a budget.
"""
import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

from . import config, db
from .privacy import scrub
from .skills import CANONICAL, CATEGORY, find_skills, normalise

PROMPT_VERSION = "v1"
Seniority = Literal["working_student", "junior", "mid", "senior", "lead", "unknown"]


class Extraction(BaseModel):
    skills: list[str] = Field(default_factory=list)          # canonical names
    other_skills: list[str] = Field(default_factory=list)    # outside the taxonomy
    seniority: Seniority = "unknown"
    salary_min: float | None = None                          # EUR per year
    salary_max: float | None = None
    german_level: Literal["none", "basic", "fluent"] = "none"
    english_required: bool = False
    remote: Literal["onsite", "hybrid", "remote", "unknown"] = "unknown"
    years_experience: int | None = None
    employment: Literal["full_time", "freelance", "part_time", "internship", "unknown"] = "unknown"

    @field_validator("salary_min", "salary_max")
    @classmethod
    def plausible_salary(cls, v):
        return v if v is None or 15_000 <= v <= 400_000 else None


# ---- Rules ----------------------------------------------------------------------------------
NUM = r"\d{2,3}(?:[.,]\d{3})"
SALARY_PATTERNS = [
    (re.compile(rf"€?\s?({NUM})\s*(?:€|EUR|Euro)?\s*(?:–|-|bis|and|und)\s*€?\s?({NUM})", re.I), 1),
    (re.compile(r"€?\s?(\d{2,3})\s?k\s*(?:–|-)\s*€?\s?(\d{2,3})\s?k", re.I), 1000),
    (re.compile(r"(\d{2,3})\s*(?:–|-)\s*(\d{2,3})\s*T€", re.I), 1000),
]


def _num(s: str) -> float:
    return float(re.sub(r"[.,]", "", s))


def rule_extract(title: str, text: str) -> Extraction:
    full = f"{title}\n{text}"
    t = (title or "").lower()
    if re.search(r"werkstudent|working student|praktik|intern\b", t):
        seniority = "working_student"
    elif re.search(r"\b(lead|head|principal|leiter|teamlead)", t):
        seniority = "lead"
    elif re.search(r"\bsenior\b", t):
        seniority = "senior"
    elif re.search(r"\b(junior|einsteiger|graduate)\b", t):
        seniority = "junior"
    else:
        seniority = "mid"
    lo = hi = None
    for pat, mult in SALARY_PATTERNS:
        m = pat.search(full)
        if m:
            lo, hi = _num(m.group(1)) * mult, _num(m.group(2)) * mult
            break
    low = full.lower()
    if re.search(r"verhandlungssicher|fließend|\bc[12]\b|sehr gute deutschkenntnisse|fluent german", low):
        german = "fluent"
    elif re.search(r"grundkenntnisse in deutsch|basic german|deutsch \(b[12]\)|\bb1\b", low):
        german = "basic"
    else:
        german = "none"
    if re.search(r"100\s?%\s?remote|fully remote|remote innerhalb", low):
        remote = "remote"
    elif re.search(r"hybrid", low):
        remote = "hybrid"
    elif re.search(r"vor ort|on site|on-site", low):
        remote = "onsite"
    else:
        remote = "unknown"
    years = re.search(r"(\d{1,2})\+?\s*(?:years|jahre)", low)
    from .roles import employment_from_text
    employment = employment_from_text(title) if employment_from_text(title) != "unknown" else employment_from_text(text[:3000])
    return Extraction(employment=employment, skills=find_skills(full), seniority=seniority, salary_min=lo, salary_max=hi,
                      german_level=german, english_required=bool(re.search(r"english|englisch", low)),
                      remote=remote, years_experience=int(years.group(1)) if years else None)


# ---- LLM ------------------------------------------------------------------------------------
SYSTEM_PROMPT = f"""You extract structured facts from German or English job postings for a labour-market analysis.
Return ONLY a JSON object with these keys:
- "skills": skills the candidate is REQUIRED or EXPECTED to have (must-have and nice-to-have).
  Use these exact names where they fit: {", ".join(CANONICAL)}.
  Do NOT include tools only mentioned as company partners, products, perks or training offers,
  or skills explicitly not required (e.g. "keine SAP-Kenntnisse erforderlich").
- "other_skills": required skills that are not in the list above (short names).
- "seniority": one of working_student, junior, mid, senior, lead, unknown.
- "salary_min", "salary_max": annual gross salary in EUR as numbers, or null. Convert monthly to annual (x12);
  leave hourly wages null.
- "german_level": "fluent" (C1/C2, fließend, verhandlungssicher, sehr gut), "basic" (A2/B1, Grundkenntnisse), or "none".
- "english_required": true or false.
- "remote": onsite, hybrid, remote, or unknown.
- "years_experience": minimum years required as an integer, or null.
- "employment": full_time, freelance (also contract/interim), part_time, internship (also working student, trainee), or unknown."""


def _client():
    from openai import OpenAI

    return OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)


def _cost(usage) -> float:
    reported = (getattr(usage, "model_extra", None) or {}).get("cost")
    if isinstance(reported, (int, float)):
        return float(reported)
    return (usage.prompt_tokens * config.PRICE_INPUT_PER_M + usage.completion_tokens * config.PRICE_OUTPUT_PER_M) / 1e6


def llm_extract(title: str, text: str, client=None) -> tuple[Extraction, dict]:
    """Returns the extraction and a call record (tokens, cost, latency)."""
    client = client or _client()
    body = scrub(text)[:6000]
    start = time.perf_counter()
    extra = {"usage": {"include": True}} if "openrouter" in config.LLM_BASE_URL else None
    resp = client.chat.completions.create(
        model=config.LLM_MODEL, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": f"Title: {title}\n\n{body}"}],
        extra_body=extra)
    call = {"model": config.LLM_MODEL, "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens, "cost_usd": _cost(resp.usage),
            "latency_ms": int((time.perf_counter() - start) * 1000)}
    raw = json.loads(resp.choices[0].message.content)
    canonical, other = [], list(raw.get("other_skills") or [])
    for s in raw.get("skills") or []:
        n = normalise(str(s))
        if n and n not in canonical:
            canonical.append(n)
        elif not n:
            other.append(str(s))
    raw.update(skills=canonical, other_skills=sorted(set(other)))
    return Extraction.model_validate({k: v for k, v in raw.items() if k in Extraction.model_fields}), call


# ---- Running over a database ------------------------------------------------------------------
def content_hash(title: str, text: str, method: str) -> str:
    model = config.LLM_MODEL if method == "llm" else "rules"
    return hashlib.sha1(f"{PROMPT_VERSION}|{model}|{title}|{text}".encode()).hexdigest()


def save_many(con, items: list[tuple[str, Extraction, str, str]]):
    """Write many extractions at once: (refnr, extraction, method, content hash).
    One write per batch, not per posting: thousands of single writes take minutes."""
    if not items:
        return
    now = datetime.now()
    rows = pd.DataFrame([{**ex.model_dump(exclude={"skills", "other_skills"}), "refnr": refnr, "method": method,
                          "model": config.LLM_MODEL if method == "llm" else "rules", "content_hash": h,
                          "other_skills": json.dumps(ex.other_skills), "extracted_at": now}
                         for refnr, ex, method, h in items])
    db.upsert(con, "extractions", rows, ["refnr"])
    con.register("_done", pd.DataFrame({"refnr": [i[0] for i in items]}))
    con.execute("DELETE FROM posting_skills WHERE refnr IN (SELECT refnr FROM _done)")
    con.unregister("_done")
    skills = pd.DataFrame([{"refnr": refnr, "skill": sk, "category": CATEGORY[sk]}
                           for refnr, ex, _, _ in items for sk in ex.skills])
    if not skills.empty:
        db.upsert(con, "posting_skills", skills, ["refnr", "skill"])


def save(con, refnr: str, ex: Extraction, method: str, h: str):
    save_many(con, [(refnr, ex, method, h)])


def run(path, method: str = "auto", limit: int | None = None, max_cost: float = 1.0, workers: int = 4,
        verbose: bool = True) -> dict:
    if method == "auto":
        method = "rules" if config.MOCK_LLM else "llm"
    if method == "llm" and config.MOCK_LLM:
        raise SystemExit("No LLM key found. Add OPENROUTER_API_KEY to .env, or use --method rules.")
    con = db.connect(path)
    todo = con.execute("""
        SELECT p.refnr, p.title, p.description, e.content_hash
        FROM postings p LEFT JOIN extractions e USING (refnr)
        ORDER BY p.published DESC""").df()
    todo["new_hash"] = [content_hash(t, d, method) for t, d in zip(todo["title"], todo["description"])]
    todo = todo[todo["content_hash"] != todo["new_hash"]]          # cached: skip unchanged postings
    if limit:
        todo = todo.head(limit)
    stats = {"method": method, "todo": len(todo), "done": 0, "failed": 0, "cost_usd": 0.0, "stopped_at_budget": False}
    if method == "rules":
        save_many(con, [(r.refnr, rule_extract(r.title, r.description), "rules", r.new_hash) for r in todo.itertuples()])
        stats["done"] = len(todo)
        con.close()
        return stats

    client = _client()
    rows = list(todo.itertuples())

    def work(r):
        try:
            ex, call = llm_extract(r.title, r.description, client)
            return r, ex, call, None
        except (ValidationError, json.JSONDecodeError, KeyError) as e:
            return r, None, None, e
        except Exception as e:  # noqa: BLE001 - API errors: record and continue
            return r, None, None, e

    with ThreadPoolExecutor(max_workers=workers) as pool:
        i = 0
        while i < len(rows) and not stats["stopped_at_budget"]:
            batch = rows[i:i + workers * 4]
            i += len(batch)
            done, calls = [], []
            for fut in as_completed([pool.submit(work, r) for r in batch]):
                r, ex, call, err = fut.result()
                ok = err is None
                if ok:
                    done.append((r.refnr, ex, "llm", r.new_hash))
                    stats["done"] += 1
                    stats["cost_usd"] += call["cost_usd"]
                else:
                    stats["failed"] += 1
                calls.append([datetime.now(), r.refnr, config.LLM_MODEL, (call or {}).get("input_tokens", 0),
                              (call or {}).get("output_tokens", 0), (call or {}).get("cost_usd", 0.0),
                              (call or {}).get("latency_ms", 0), ok])
            save_many(con, done)        # results are saved after every batch, so a stopped run keeps its work
            con.executemany("INSERT INTO llm_calls VALUES (?, ?, 'extract', ?, ?, ?, ?, ?, ?)", calls)
            if verbose:
                print(f"  {stats['done']:,}/{len(rows):,} postings, ${stats['cost_usd']:.4f} so far")
            if stats["cost_usd"] >= max_cost:
                stats["stopped_at_budget"] = True
    con.close()
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["sample", "live"], default="sample")
    ap.add_argument("--method", choices=["auto", "llm", "rules"], default="auto")
    ap.add_argument("--limit", type=int, help="only the newest N postings that need extraction")
    ap.add_argument("--max-cost", type=float, default=1.0, help="stop when this many USD have been spent")
    ap.add_argument("--workers", type=int, default=4, help="parallel LLM requests")
    args = ap.parse_args()
    path = config.SAMPLE_DB if args.source == "sample" else config.LIVE_DB
    s = run(path, args.method, args.limit, args.max_cost, args.workers)
    print(f"\n{s['method']}: extracted {s['done']:,} of {s['todo']:,} postings"
          + (f", {s['failed']} failed" if s["failed"] else "")
          + (f", cost ${s['cost_usd']:.4f}" if s["method"] == "llm" else "")
          + (" (stopped at --max-cost)" if s["stopped_at_budget"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
