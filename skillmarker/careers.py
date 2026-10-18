"""Collect jobs from company career pages, through their recruiting systems' public job feeds.

    python -m skillmarker.careers --discover              # find which system each company in companies.json uses
    python -m skillmarker.careers --add "Example GmbH"    # add a company by name (discovered automatically)
    python -m skillmarker.careers                         # collect jobs from all discovered companies
    python -m skillmarker.careers --list                  # show companies and their systems

Most companies run their careers page on a recruiting system (ATS) that publishes the job
list through a public feed, made so jobs can be shown and shared elsewhere. Supported:
Personio (very common in Germany), Greenhouse, Lever, Ashby, Recruitee and SmartRecruiters.

These feeds return ALL of a company's jobs, so every title goes through roles.classify():
only analytics and AI roles are kept, only full-time and freelance (or not stated), and only
in Germany or remote and open to Germany.
"""
import argparse
import hashlib
import html
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pandas as pd
import requests

from . import config, db
from .geo import locate
from .roles import KEEP_EMPLOYMENT, classify, employment_from_text, in_scope_title, location_ok, work_model_from_text

TIMEOUT = 20


def html_to_text(raw: str | None) -> str:
    """Readable text from HTML job descriptions (keeps line breaks and bullet points)."""
    if not raw:
        return ""
    t = html.unescape(raw)
    t = re.sub(r"(?i)<\s*li[^>]*>", "\n- ", t)
    t = re.sub(r"(?i)<\s*(?:br|/p|/div|/h\d|/ul|/ol)[^>]*>", "\n", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[ \t\u00a0]+", " ", t)
    return re.sub(r"\n\s*\n+", "\n\n", t).strip()


def _date(value) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):  # epoch seconds or milliseconds
        return datetime.fromtimestamp(value / 1000 if value > 1e11 else value, tz=timezone.utc).date().isoformat()
    m = re.match(r"\d{4}-\d{2}-\d{2}", str(value))
    return m.group(0) if m else None


def _job(company: str, system: str, slug: str, jid, title, location, description, url, published,
         employment_hint="", remote_hint="", remote_flag: bool = False) -> dict | None:
    """One normalised posting, or None if it's out of scope (role, contract type or place)."""
    title = (title or "").strip()
    family = classify(title)
    if not family or not in_scope_title(title):
        return None
    employment = employment_from_text(employment_hint) if employment_hint else employment_from_text(title)
    if employment not in KEEP_EMPLOYMENT:
        return None
    work_model = work_model_from_text(remote_hint, location, title)
    if remote_flag and work_model == "unknown":
        work_model = "remote"
    if not location_ok(location, remote=work_model in ("remote", "hybrid") or remote_flag):
        return None
    city, lat, lon = locate(location)
    now = datetime.now()
    return {"refnr": f"{system}:{slug}:{jid}", "family": family, "query": None, "title": title, "occupation": None,
            "employer": company, "city": city or (location or "")[:80] or None, "postcode": None, "region": None,
            "lat": lat, "lon": lon, "published": published, "external_url": url, "description": description,
            "source": f"careers:{system}", "first_seen": now, "last_seen": now, "employment": employment,
            "work_model": work_model, "dup_of": None}


# ---- One adapter per recruiting system --------------------------------------------------------
# Each returns a list of normalised postings, [] for a valid board without relevant jobs,
# or None if the board doesn't exist (used by discovery).

def greenhouse(s, slug, company):
    r = s.get(f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs", params={"content": "true"}, timeout=TIMEOUT)
    if r.status_code != 200:
        return None
    out = []
    for j in r.json().get("jobs", []):
        meta = " ".join(str(m.get("value")) for m in j.get("metadata") or [] if m.get("value"))
        loc = (j.get("location") or {}).get("name", "")
        out.append(_job(company, "greenhouse", slug, j.get("id"), j.get("title"), loc, html_to_text(j.get("content")),
                        j.get("absolute_url"), _date(j.get("first_published") or j.get("updated_at")), meta, loc))
    return [o for o in out if o]


def lever(s, slug, company):
    for host in ("api.lever.co", "api.eu.lever.co"):   # EU companies are often on the EU instance
        r = s.get(f"https://{host}/v0/postings/{slug}", params={"mode": "json"}, timeout=TIMEOUT)
        if r.status_code == 200 and isinstance(r.json(), list):
            break
    else:
        return None
    out = []
    for j in r.json():
        cat = j.get("categories") or {}
        body = "\n\n".join([j.get("descriptionPlain") or ""] +
                           [f"{x.get('text', '')}\n{html_to_text(x.get('content'))}" for x in j.get("lists") or []] +
                           [j.get("additionalPlain") or ""])
        out.append(_job(company, "lever", slug, j.get("id"), j.get("text"), cat.get("location", ""), body.strip(),
                        j.get("hostedUrl"), _date(j.get("createdAt")), cat.get("commitment", ""), j.get("workplaceType", "")))
    return [o for o in out if o]


def ashby(s, slug, company):
    r = s.get(f"https://api.ashbyhq.com/posting-api/job-board/{slug}", timeout=TIMEOUT)
    if r.status_code != 200 or "jobs" not in r.json():
        return None
    out = []
    for j in r.json()["jobs"]:
        if j.get("isListed") is False:
            continue
        locs = [j.get("location") or ""] + [x.get("location", "") for x in j.get("secondaryLocations") or []]
        out.append(_job(company, "ashby", slug, j.get("id"), j.get("title"), " / ".join(l for l in locs if l),
                        j.get("descriptionPlain") or html_to_text(j.get("descriptionHtml")), j.get("jobUrl"),
                        _date(j.get("publishedAt")), j.get("employmentType", ""), j.get("workplaceType", ""),
                        bool(j.get("isRemote"))))
    return [o for o in out if o]


def personio(s, slug, company):
    for domain in ("jobs.personio.de", "jobs.personio.com"):
        r = s.get(f"https://{slug}.{domain}/xml", timeout=TIMEOUT)
        if r.status_code == 200 and "<workzag-jobs" in r.text[:500]:
            break
    else:
        return None
    out = []
    for pos in ET.fromstring(r.content).findall("position"):
        get = lambda tag: (pos.findtext(tag) or "").strip()  # noqa: E731
        offices = [get("office")] + [o.text or "" for o in pos.findall("additionalOffices/office")]
        body = "\n\n".join(f"{d.findtext('name') or ''}\n{html_to_text(d.findtext('value'))}"
                           for d in pos.findall("jobDescriptions/jobDescription"))
        out.append(_job(company, "personio", slug, get("id"), get("name"), " / ".join(o for o in offices if o), body,
                        f"https://{slug}.{domain}/job/{get('id')}", _date(get("createdAt")),
                        f"{get('employmentType')} {get('schedule')}", get("office")))
    return [o for o in out if o]


def recruitee(s, slug, company):
    r = s.get(f"https://{slug}.recruitee.com/api/offers/", timeout=TIMEOUT)
    if r.status_code != 200 or "offers" not in r.json():
        return None
    out = []
    for j in r.json()["offers"]:
        if j.get("status") not in (None, "published"):
            continue
        loc = ", ".join(x for x in [j.get("city"), j.get("country")] if x) or j.get("location", "")
        body = f"{html_to_text(j.get('description'))}\n\n{html_to_text(j.get('requirements'))}"
        out.append(_job(company, "recruitee", slug, j.get("id"), j.get("title"), loc, body.strip(), j.get("careers_url"),
                        _date(j.get("published_at")), j.get("employment_type_code", ""),
                        "remote" if j.get("remote") else ("hybrid" if j.get("hybrid") else ""), bool(j.get("remote"))))
    return [o for o in out if o]


def smartrecruiters(s, slug, company):
    base = f"https://api.smartrecruiters.com/v1/companies/{slug}/postings"
    r = s.get(base, params={"limit": 100}, timeout=TIMEOUT)
    if r.status_code != 200 or not r.json().get("totalFound"):
        return None                                   # unknown companies return an empty list, not a 404
    items, offset = list(r.json().get("content", [])), 100
    while offset < min(r.json().get("totalFound", 0), 1000):
        items += s.get(base, params={"limit": 100, "offset": offset}, timeout=TIMEOUT).json().get("content", [])
        offset += 100
    out = []
    for j in items:
        loc = j.get("location") or {}
        where = ", ".join(x for x in [loc.get("city"), loc.get("country")] if x)
        if not classify(j.get("name", "")):
            continue                                  # details only for relevant titles (saves requests)
        d = s.get(f"{base}/{j.get('id')}", timeout=TIMEOUT)
        sections = (d.json().get("jobAd") or {}).get("sections", {}) if d.status_code == 200 else {}
        body = "\n\n".join(html_to_text((sections.get(k) or {}).get("text"))
                           for k in ("jobDescription", "qualifications", "additionalInformation"))
        out.append(_job(company, "smartrecruiters", slug, j.get("id"), j.get("name"), where, body,
                        (d.json().get("postingUrl") if d.status_code == 200 else None),
                        _date(j.get("releasedDate")), (j.get("typeOfEmployment") or {}).get("label", ""),
                        "remote" if loc.get("remote") else ("hybrid" if loc.get("hybrid") else ""), bool(loc.get("remote"))))
    return [o for o in out if o]


SYSTEMS = {"personio": personio, "greenhouse": greenhouse, "lever": lever, "ashby": ashby,
           "recruitee": recruitee, "smartrecruiters": smartrecruiters}


# ---- Company list and discovery -------------------------------------------------------------
LEGAL = re.compile(r"\b(?:gmbh|ag|se|kg|kgaa|ug|mbh|co|inc|ltd|llc|group|holding|deutschland|germany|technologies|tech)\b\.?",
                   re.IGNORECASE)


def slug_candidates(name: str) -> list[str]:
    base = LEGAL.sub(" ", name.lower().replace("&", " and ").replace("°", ""))
    words = re.findall(r"[a-z0-9äöüß]+", base)
    words = [w.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss") for w in words]
    if not words:
        return []
    cands = ["".join(words), "-".join(words), words[0]]
    return list(dict.fromkeys(c for c in cands if len(c) >= 2))


def load_companies(path=None) -> list[dict]:
    path = path or config.COMPANIES_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save_companies(companies: list[dict], path=None):
    path = path or config.COMPANIES_FILE
    path.write_text(json.dumps(companies, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def discover(name: str, session=None, systems=None) -> tuple[str, str] | None:
    """Try each recruiting system with likely slugs; return (system, slug) of the first that answers."""
    s = session or requests.Session()
    for slug in slug_candidates(name):
        for system, fn in (systems or SYSTEMS).items():
            try:
                if fn(s, slug, name) is not None:
                    return system, slug
            except (requests.RequestException, ValueError, ET.ParseError):
                continue
    return None


def discover_all(path=None, session=None, rediscover: bool = False, verbose: bool = True) -> list[dict]:
    companies = load_companies(path)
    for c in companies:
        if c.get("ats") and not rediscover:
            continue
        found = discover(c["name"], session)
        c["ats"], c["slug"] = (found if found else ("none", None))
        if verbose:
            print(f"  {c['name']:32s} {c['ats']:16s} {c['slug'] or ''}")
    save_companies(companies, path)
    return companies


# ---- Collecting -------------------------------------------------------------------------------
def store(path, rows: list[dict]) -> dict:
    """Insert new postings, refresh last_seen for known ones (shared by all non-BA sources)."""
    con = db.connect(path)
    df = pd.DataFrame(rows).reindex(columns=db.POSTING_COLUMNS) if rows else pd.DataFrame(columns=db.POSTING_COLUMNS)
    df = df.drop_duplicates("refnr")
    known = set(con.execute("SELECT refnr FROM postings").df()["refnr"])
    new = df[~df["refnr"].isin(known)]
    if not new.empty:
        db.upsert(con, "postings", new, ["refnr"])
    seen = df[df["refnr"].isin(known)]
    if not seen.empty:
        con.register("_seen", seen[["refnr"]])
        con.execute("UPDATE postings SET last_seen = ? WHERE refnr IN (SELECT refnr FROM _seen)", [datetime.now()])
    con.close()
    return {"new": len(new), "seen": len(seen)}


def run(path=None, session=None, verbose: bool = True) -> dict:
    path = path or config.LIVE_DB
    s = session or requests.Session()
    s.headers.update({"User-Agent": "skillmarker (labour-market research project)"})
    total = {"companies": 0, "postings": 0, "new": 0}
    for c in load_companies():
        if c.get("ats") not in SYSTEMS:
            continue
        try:
            rows = SYSTEMS[c["ats"]](s, c["slug"], c["name"]) or []
            res = store(path, rows)
            status = f"{len(rows):4d} relevant  {res['new']:4d} new"
            total["postings"] += len(rows)
            total["new"] += res["new"]
        except (requests.RequestException, ValueError, ET.ParseError) as e:
            status = f"error: {type(e).__name__}"
        total["companies"] += 1
        if verbose:
            print(f"  {c['name']:32s} {c['ats']:16s} {status}")
        time.sleep(0.3)
    from .dedup import run as dedup
    total["duplicates"] = dedup(path)
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--discover", action="store_true", help="find the recruiting system of companies not yet resolved")
    ap.add_argument("--rediscover", action="store_true", help="re-check every company")
    ap.add_argument("--add", metavar="NAME", help="add a company by name and discover its system")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for c in load_companies():
            print(f"  {c['name']:32s} {c.get('ats') or 'not checked':16s} {c.get('slug') or ''}")
        return 0
    if args.add:
        companies = load_companies()
        found = discover(args.add)
        companies.append({"name": args.add, "ats": found[0] if found else "none", "slug": found[1] if found else None})
        save_companies(companies)
        print(f"{args.add}: " + (f"{found[0]} ({found[1]})" if found else
              "no supported recruiting system found. You can set \"ats\" and \"slug\" in companies.json by hand."))
        return 0
    if args.discover or args.rediscover:
        discover_all(rediscover=args.rediscover)
        return 0
    t = run()
    print(f"\n{t['companies']} companies, {t['postings']} relevant postings, {t['new']} new, "
          f"{t['duplicates']} duplicates of postings from other sources -> {config.LIVE_DB}")
    print("Next: python -m skillmarker.extract --source live")
    return 0


if __name__ == "__main__":
    sys.exit(main())
