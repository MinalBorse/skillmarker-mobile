"""Collect real job postings from the Bundesagentur für Arbeit job search (Jobbörse).

    python -m skillmarker.collect                    # all role families, postings from the last 30 days
    python -m skillmarker.collect --days 7 --max-pages 3
    python -m skillmarker.collect --family "AI Engineer" --family "Agentic AI"

The API is free and public: the client id "jobboerse-jobsuche" goes in the X-API-Key header,
no registration needed. Two steps:
1. /pc/v4/jobs?was=<search term> lists postings (title, employer, place, date, reference number).
2. /pc/v4/jobdetails/<base64(refnr)> returns the full description, which we need for skills.

Running it regularly (e.g. daily) builds history: `first_seen` and `last_seen` show how
long postings stay online, and new postings drive the trend charts.
"""
import argparse
import base64
import re
import sys
import time
from datetime import datetime

import pandas as pd
import requests

from . import config, db
from .roles import (FREELANCE, FULL_TIME, KEEP_EMPLOYMENT, PART_TIME, UNKNOWN, classify, employment_from_text,
                    in_scope_title)

# Search terms are fuzzy, so results include unrelated jobs. Keep only titles that look
# like data or AI roles.
RELEVANT = re.compile(r"data|daten|analy|business intelligence|\bbi\b|power ?bi|scien|machine learning|\bml\b|"
                      r"\bai\b|\bki\b|k\.i\.|llm|genai|generative|agent|künstliche intelligenz|artificial intelligence|"
                      r"analytics|insight|reporting", re.IGNORECASE)


# The API is undocumented and changes over time. These are the request variants known to
# work at some point; probe() picks the first that works on your machine.
SEARCH_VARIANTS = [("/pc/v4/jobs", 100), ("/pc/v4/jobs", 50), ("/pc/v6/jobs", 50), ("/pc/v4/app/jobs", 50),
                   ("/pc/v4/app/jobs", 25)]
USER_AGENTS = {
    "default": "skillmarker (labour-market research project)",
    "app": "Jobsuche/2.9.2 (de.arbeitsagentur.jobboerse; build:1077; iOS 15.1.0) Alamofire/5.4.4",
    "browser": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
}


class BAUnavailable(RuntimeError):
    """No request variant worked; the message lists each attempt's status and reply."""


class BAClient:
    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()
        self.session.headers.update({"X-API-Key": config.BA_API_KEY, "Accept": "application/json",
                                     "User-Agent": USER_AGENTS["default"]})
        self._interval = 1 / config.BA_REQUESTS_PER_SECOND
        self._last = 0.0
        self.requests = 0
        self.search_path, self.page_size, self.user_agent = None, 100, "default"

    def probe(self, term: str = "Data Analyst") -> list[dict]:
        """Find a working combination of endpoint, page size and User-Agent.
        Returns all attempts (for --probe); raises BAUnavailable if none works."""
        attempts = []
        for ua_name, ua in USER_AGENTS.items():
            self.session.headers["User-Agent"] = ua
            for path, size in SEARCH_VARIANTS:
                self.requests += 1
                try:
                    resp = self.session.get(f"{config.BA_BASE_URL}{path}", timeout=30, params={
                        "was": term, "angebotsart": 1, "page": 1, "size": size})
                    status, body = resp.status_code, (resp.text or "")[:160].replace("\n", " ")
                    ok = status == 200 and bool(parse_search(resp.json()))
                except (requests.RequestException, ValueError) as e:
                    status, body, ok = type(e).__name__, str(e)[:160], False
                attempts.append({"path": path, "size": size, "user_agent": ua_name, "status": status,
                                 "ok": ok, "reply": body})
                if ok:
                    self.search_path, self.page_size, self.user_agent = path, size, ua_name
                    return attempts
                time.sleep(self._interval)
        lines = "\n".join(f"  {a['path']:18s} size {a['size']:3d}  UA {a['user_agent']:8s} -> {a['status']}  {a['reply']}"
                          for a in attempts)
        raise BAUnavailable("No request variant worked. Attempts:\n" + lines)

    def _get(self, path: str, params: dict | None = None) -> dict | None:
        for attempt in range(4):
            wait = self._interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.requests += 1
            resp = self.session.get(f"{config.BA_BASE_URL}{path}", params=params, timeout=30)
            if resp.status_code == 404:
                return None
            if resp.status_code in (429, 500, 502, 503, 504):   # busy: back off and retry
                time.sleep(2 ** attempt * 2)
                continue
            if resp.status_code >= 400:
                raise requests.HTTPError(f"HTTP {resp.status_code}: {(resp.text or '')[:120]}", response=resp)
            return resp.json()
        raise RuntimeError(f"Bundesagentur API kept failing for {path}")

    def search(self, term: str, page: int = 1, days: int = 30, angebotsart: int = 1) -> dict | None:
        """angebotsart 1 = employment, 2 = self-employment (freelance)."""
        if self.search_path is None:
            self.probe(term)
        return self._get(self.search_path, {"was": term, "angebotsart": angebotsart, "page": page,
                                            "size": self.page_size, "veroeffentlichtseit": days})

    def details(self, refnr: str) -> dict | None:
        code = base64.b64encode(refnr.encode()).decode()
        return self._get(f"/pc/v4/jobdetails/{code}")


# ---- Tolerant field finding --------------------------------------------------------------
# API versions name and nest fields differently (arbeitsort vs arbeitsorte[0].adresse,
# lat/lon vs breite/laenge, ...). Instead of hard-coding one layout, search the JSON.
DATE_KEYS = ("aktuelleVeroeffentlichungsdatum", "veroeffentlichungsdatum", "datumErsteVeroeffentlichung",
             "ersteVeroeffentlichungsdatum", "veroeffentlichtAm", "modifikationsTimestamp", "aenderungsdatum")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
META_FIELDS = ["city", "postcode", "region", "lat", "lon", "published"]


def _walk(obj):
    """Yield every (key, value) pair in nested dicts and lists."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def _first(obj, keys):
    lowered = {k.lower() for k in keys}
    return next((v for k, v in _walk(obj) if k.lower() in lowered and v not in (None, "", [], {})), None)


def _number(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def find_meta(obj) -> dict:
    """City, postcode, region, coordinates and publication date, wherever they are in the JSON."""
    date = _first(obj, DATE_KEYS)
    if not (isinstance(date, str) and DATE_RE.match(date)):
        date = next((v for k, v in _walk(obj) if "veroeffentlich" in k.lower() and isinstance(v, str)
                     and DATE_RE.match(v)), None)
    city, plz, region = _first(obj, ("ort", "stadt", "city")), _first(obj, ("plz", "postleitzahl")), \
        _first(obj, ("region", "bundesland"))
    return {"city": city if isinstance(city, str) else None,
            "postcode": str(plz) if plz is not None and not isinstance(plz, (dict, list)) else None,
            "region": region if isinstance(region, str) else None,
            "lat": _number(_first(obj, ("lat", "breite", "latitude"))),
            "lon": _number(_first(obj, ("lon", "lng", "laenge", "longitude"))),
            "published": date[:10] if isinstance(date, str) and DATE_RE.match(date) else None}


def parse_search(payload: dict | None) -> list[dict]:
    """One dict per posting from a search response (v4, with v6 field names as fallback)."""
    payload = payload or {}
    jobs = payload.get("stellenangebote")
    if jobs is None:  # newer versions may use another key: take the first list of postings
        jobs = next((v for v in payload.values() if isinstance(v, list) and v and isinstance(v[0], dict)), [])
    out = []
    for job in jobs or []:
        refnr = job.get("refnr") or job.get("referenznummer")
        if not refnr:
            continue
        employer = job.get("arbeitgeber") or job.get("arbeitgeberName") or job.get("firma")   # "firma" in v6
        if isinstance(employer, dict):
            employer = employer.get("name") or employer.get("bezeichnung")
        out.append({
            "refnr": refnr, "title": job.get("titel") or job.get("stellenangebotsTitel") or job.get("beruf") or "",
            "occupation": job.get("beruf") or job.get("hauptberuf"), "employer": employer, **find_meta(job),
            "external_url": job.get("externeUrl"), "employment": employment_of(job), "work_model": work_model_of(job),
        })
    return out


def parse_details(payload: dict | None) -> dict:
    payload = payload or {}
    text = payload.get("stellenangebotsBeschreibung") or payload.get("stellenbeschreibung") or ""
    return {"title": payload.get("stellenangebotsTitel"), "description": text.strip(), "meta": find_meta(payload),
            # postings from partner job boards link to the original ad
            "partner_url": payload.get("allianzpartnerUrl") or payload.get("externeUrl")}


def ad_url(refnr: str) -> str:
    """Public page of a posting on arbeitsagentur.de, where you can read the full ad and apply."""
    return f"https://www.arbeitsagentur.de/jobsuche/jobdetail/{refnr}"


def is_relevant(title: str) -> bool:
    """An analytics or AI role (see roles.py), and not a student job, internship or part-time role."""
    return classify(title) is not None and in_scope_title(title)


def employment_of(job: dict, angebotsart: int = 1) -> str:
    """Full-time / freelance / part-time from the Bundesagentur's structured fields."""
    if angebotsart == 2 or str(job.get("stellenangebotsart", "")).upper().startswith("SELBST"):
        return FREELANCE
    if job.get("arbeitszeitVollzeit"):
        return FULL_TIME
    if any(job.get(k) for k in ("arbeitszeitTeilzeitAbend", "arbeitszeitTeilzeitNachmittag", "arbeitszeitTeilzeitVormittag",
                                "arbeitszeitTeilzeitFlexibel", "istGeringfuegigeBeschaeftigung")):
        return PART_TIME
    # No structured answer: the title often says it ("Freelance ...", "... (Teilzeit)").
    return employment_from_text(job.get("stellenangebotsTitel") or job.get("titel"))


def work_model_of(job: dict) -> str:
    typ = str(job.get("homeofficetyp") or "").upper()
    if "AUSSCHLIESS" in typ or "VOLLSTAENDIG" in typ or "100" in typ:
        return "remote"
    if job.get("homeofficemoeglich") or typ:
        return "hybrid"
    return "unknown"


def reclassify(con) -> int:
    """Apply the current role rules to stored real postings: role family by title, and family
    NULL (excluded from all views) for roles that are out of scope. Cheap: no API calls."""
    df = con.execute("SELECT refnr, title, family FROM postings WHERE source <> 'sample' OR source IS NULL").df()
    if df.empty:
        return 0
    df["new_family"] = [classify(t) if in_scope_title(t) else None for t in df["title"]]
    changed = df[df["new_family"].fillna("") != df["family"].fillna("")]
    if not changed.empty:
        con.register("_fam", changed[["refnr", "new_family"]])
        con.execute("UPDATE postings p SET family = f.new_family FROM _fam f WHERE p.refnr = f.refnr")
        con.unregister("_fam")
    return len(changed)


def run(families: list[str] | None = None, days: int = 30, max_pages: int = 5, path=None,
        client: BAClient | None = None, verbose: bool = True) -> dict:
    path = path or config.LIVE_DB
    client = client or BAClient()
    if isinstance(client, BAClient) and client.search_path is None:
        client.probe()            # fails fast with a clear report instead of 20 identical errors
        if verbose:
            print(f"  using {client.search_path} (page size {client.page_size}, {client.user_agent} user agent)")
    con = db.connect(path)
    reclassify(con)
    known = set(con.execute("SELECT refnr FROM postings").df()["refnr"])
    now = datetime.now()
    seen_now: set[str] = set()
    repairs: list[dict] = []    # known postings: fill in fields an older parser missed
    stats = {"found": 0, "new": 0, "irrelevant": 0, "details": 0}
    for family in families or list(config.ROLE_FAMILIES):
        for term, art in [(t, a) for t in config.ROLE_FAMILIES[family] for a in (1, 2)]:
            found = new = details = 0
            status = "ok"
            try:
                for page in range(1, max_pages + 1):
                    jobs = parse_search(client.search(term, page=page, days=days, angebotsart=art))
                    if art == 2:
                        for j in jobs:
                            j["employment"] = FREELANCE
                    found += len(jobs)
                    rows = []
                    for j in jobs:
                        if j["refnr"] in seen_now:
                            continue
                        seen_now.add(j["refnr"])
                        if not is_relevant(j["title"]) or j.get("employment") not in KEEP_EMPLOYMENT:
                            stats["irrelevant"] += 1
                            continue
                        if j["refnr"] in known:
                            repairs.append({"refnr": j["refnr"], "employer": j.get("employer"),
                                            "employment": j.get("employment"), "work_model": j.get("work_model"),
                                            "occupation": j.get("occupation"), **{k: j.get(k) for k in META_FIELDS}})
                            continue
                        d = parse_details(client.details(j["refnr"]))
                        details += 1
                        j.update({k: j[k] if j.get(k) is not None else d["meta"].get(k) for k in META_FIELDS})
                        j["external_url"] = j.get("external_url") or d.get("partner_url")
                        rows.append({**j, "title": d["title"] or j["title"], "description": d["description"],
                                     "family": classify(j["title"]) or family, "query": term, "source": "ba_api",
                                     "first_seen": now, "last_seen": now})
                        known.add(j["refnr"])
                    if rows:
                        db.upsert(con, "postings", pd.DataFrame(rows).reindex(columns=db.POSTING_COLUMNS), ["refnr"])
                        new += len(rows)
                    if len(jobs) < getattr(client, "page_size", 100):
                        break
            except (requests.RequestException, RuntimeError) as e:
                status = f"error: {str(e)[:80] or type(e).__name__}"
            label = term if art == 1 else f"{term} (freelance)"
            con.execute("INSERT INTO collection_log VALUES (?, ?, ?, ?, ?, ?, ?)",
                        [now, family, label, found, new, details, status])
            stats["found"] += found
            stats["new"] += new
            stats["details"] += details
            if verbose:
                print(f"  {family:30s} {label:36s} {found:5d} found  {new:4d} new  {status}")
    if repairs:
        rep = pd.DataFrame(repairs).drop_duplicates("refnr")
        rep["published"] = pd.to_datetime(rep["published"], errors="coerce").dt.date
        con.register("_rep", rep)
        con.execute("""UPDATE postings p SET employer = coalesce(p.employer, r.employer),
                         employment = coalesce(p.employment, r.employment), work_model = coalesce(p.work_model, r.work_model),
                         occupation = coalesce(p.occupation, r.occupation),
                         city = coalesce(p.city, r.city), postcode = coalesce(p.postcode, r.postcode),
                         region = coalesce(p.region, r.region), lat = coalesce(p.lat, r.lat), lon = coalesce(p.lon, r.lon),
                         published = coalesce(p.published, r.published)
                       FROM _rep r WHERE p.refnr = r.refnr""")
    # Postings still online today: extend their last_seen, so we learn how long ads stay up.
    if seen_now:
        con.register("_seen", pd.DataFrame({"refnr": list(seen_now)}))
        con.execute("UPDATE postings SET last_seen = ? WHERE refnr IN (SELECT refnr FROM _seen)", [now])
    con.close()
    stats["requests"] = client.requests
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", action="append", choices=list(config.ROLE_FAMILIES), help="limit to role families")
    ap.add_argument("--days", type=int, default=30, help="postings published in the last N days (max 100)")
    ap.add_argument("--max-pages", type=int, default=5, help="pages of results per search term")
    ap.add_argument("--probe", action="store_true", help="only test which API request variant works, then exit")
    ap.add_argument("--show-raw", action="store_true", help="print one raw posting and what is parsed from it")
    args = ap.parse_args()
    if args.show_raw:
        import json
        client = BAClient()
        client.probe()
        payload = client.search("Data Analyst", days=args.days) or {}
        jobs = payload.get("stellenangebote") or next((v for v in payload.values() if isinstance(v, list)), [])
        print(f"Endpoint: {client.search_path}\nTop-level keys: {list(payload.keys())}")
        print("First posting from the search:\n" + json.dumps(jobs[0] if jobs else payload, ensure_ascii=False, indent=2)[:2500])
        rows = parse_search(payload)
        if rows:
            det = client.details(rows[0]["refnr"]) or {}
            print("\nDetail keys: " + ", ".join(det.keys()))
            print("Parsed: " + json.dumps({k: rows[0][k] for k in ["refnr", "title", "employer"] + META_FIELDS},
                                          ensure_ascii=False, default=str))
        return 0
    if args.probe:
        client = BAClient()
        try:
            attempts = client.probe()
            for a in attempts:
                print(f"  {a['path']:18s} size {a['size']:3d}  UA {a['user_agent']:8s} -> {a['status']}"
                      + ("  WORKS" if a["ok"] else f"  {a['reply'][:90]}"))
        except BAUnavailable as e:
            print(e)
            return 1
        return 0
    try:
        s = run(args.family, args.days, args.max_pages)
    except BAUnavailable as e:
        print(f"Could not reach the Bundesagentur job search.\n{e}\n"
              "Check your internet connection (and VPN/proxy). If it persists, paste this output to get help.")
        return 1
    print(f"\n{s['new']:,} new postings saved ({s['found']:,} found, {s['irrelevant']:,} unrelated titles skipped, "
          f"{s['requests']:,} requests) -> {config.LIVE_DB}")
    print("Next: python -m skillmarker.extract --source live")
    return 0


if __name__ == "__main__":
    sys.exit(main())
