"""Collect jobs from aggregator APIs.

    python -m skillmarker.aggregators                  # Arbeitnow (no key) and Adzuna (if a key is set)
    python -m skillmarker.aggregators --source arbeitnow

- Arbeitnow: a free public API of jobs in Germany, many from startups' recruiting systems. No key.
- Adzuna: a job search engine with an official API for Germany. Free developer key
  (ADZUNA_APP_ID and ADZUNA_APP_KEY in .env). Its descriptions are short snippets, so skill
  extraction sees less text than for other sources.
Same rules as everywhere: analytics and AI roles only, full-time or freelance, Germany or remote.
Check each provider's current terms before relying on it.
"""
import argparse
import sys
import time

import requests

from . import config
from .careers import _date, _job, html_to_text, store


def arbeitnow(session=None, max_pages: int = 10) -> list[dict]:
    s = session or requests.Session()
    rows = []
    for page in range(1, max_pages + 1):
        r = s.get("https://www.arbeitnow.com/api/job-board-api", params={"page": page}, timeout=30)
        if r.status_code != 200:
            break
        data = r.json().get("data") or []
        for j in data:
            row = _job(j.get("company_name") or "", "arbeitnow", "board", j.get("slug"), j.get("title"),
                       j.get("location", ""), html_to_text(j.get("description")), j.get("url"), _date(j.get("created_at")),
                       " ".join(j.get("job_types") or []), "remote" if j.get("remote") else "", bool(j.get("remote")))
            if row:
                row.update(refnr=f"arbeitnow:{j.get('slug')}", source="arbeitnow")
                rows.append(row)
        if not data or not (r.json().get("links") or {}).get("next"):
            break
        time.sleep(0.5)
    return rows


def adzuna(session=None, days: int = 30, max_pages: int = 3, families: list[str] | None = None) -> list[dict]:
    if not (config.ADZUNA_APP_ID and config.ADZUNA_APP_KEY):
        return []
    s = session or requests.Session()
    rows = []
    for family in families or list(config.ROLE_FAMILIES):
        for term in config.ROLE_FAMILIES[family]:
            for page in range(1, max_pages + 1):
                r = s.get(f"https://api.adzuna.com/v1/api/jobs/de/search/{page}", timeout=30, params={
                    "app_id": config.ADZUNA_APP_ID, "app_key": config.ADZUNA_APP_KEY, "what": term,
                    "results_per_page": 50, "max_days_old": days, "content-type": "application/json"})
                if r.status_code != 200:
                    break
                results = r.json().get("results") or []
                for j in results:
                    title = html_to_text(j.get("title"))
                    contract = f"{j.get('contract_type') or ''} {j.get('contract_time') or ''}"
                    row = _job((j.get("company") or {}).get("display_name", ""), "adzuna", "de", j.get("id"), title,
                               (j.get("location") or {}).get("display_name", "") + ", Deutschland",
                               html_to_text(j.get("description")), j.get("redirect_url"), _date(j.get("created")), contract)
                    if row:
                        row.update(refnr=f"adzuna:{j.get('id')}", source="adzuna",
                                   lat=j.get("latitude") or row["lat"], lon=j.get("longitude") or row["lon"])
                        rows.append(row)
                if len(results) < 50:
                    break
                time.sleep(0.4)
    return rows


def run(sources=("arbeitnow", "adzuna"), path=None, verbose: bool = True) -> dict:
    path = path or config.LIVE_DB
    out = {}
    for name in sources:
        try:
            rows = arbeitnow() if name == "arbeitnow" else adzuna()
            res = store(path, rows)
            out[name] = {"relevant": len(rows), **res}
            if verbose:
                note = " (no ADZUNA key set)" if name == "adzuna" and not config.ADZUNA_APP_ID else ""
                print(f"  {name:10s} {len(rows):5d} relevant  {res['new']:5d} new{note}")
        except (requests.RequestException, ValueError) as e:
            out[name] = {"error": type(e).__name__}
            if verbose:
                print(f"  {name:10s} error: {type(e).__name__}")
    from .dedup import run as dedup
    out["duplicates"] = dedup(path)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["arbeitnow", "adzuna"], action="append")
    args = ap.parse_args()
    r = run(tuple(args.source or ("arbeitnow", "adzuna")))
    print(f"\n{r['duplicates']} duplicates of postings already found elsewhere. Next: python -m skillmarker.extract --source live")
    return 0


if __name__ == "__main__":
    sys.exit(main())
