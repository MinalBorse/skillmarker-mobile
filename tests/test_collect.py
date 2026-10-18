import base64

import duckdb

from skillmarker import collect as C

SEARCH = {"stellenangebote": [
    {"refnr": "10001-1", "titel": "Data Analyst (m/w/d)", "beruf": "Data Analyst", "arbeitgeber": "Firma A",
     "arbeitsort": {"ort": "Berlin", "plz": "10115", "region": "Berlin", "koordinaten": {"lat": 52.5, "lon": 13.4}},
     "aktuelleVeroeffentlichungsdatum": "2026-10-01", "externeUrl": "https://example.org/job"},
    {"refnr": "10001-2", "titel": "Lagerist (m/w/d)", "arbeitgeber": "Firma B", "arbeitsort": {"ort": "Hamburg"}},
    {"referenznummer": "10001-3", "titel": "KI-Engineer LLM", "arbeitgeber": "Firma C", "arbeitsort": {}},
]}


def test_parse_search_handles_v4_and_v6_fields():
    rows = C.parse_search(SEARCH)
    assert [r["refnr"] for r in rows] == ["10001-1", "10001-2", "10001-3"]
    assert rows[0]["lat"] == 52.5 and rows[0]["published"] == "2026-10-01" and rows[0]["employer"] == "Firma A"
    assert C.parse_search(None) == []


def test_parse_details():
    d = C.parse_details({"stellenangebotsTitel": "Data Analyst", "stellenangebotsBeschreibung": " SQL und Python "})
    assert (d["title"], d["description"]) == ("Data Analyst", "SQL und Python")
    assert set(d["meta"]) == {"city", "postcode", "region", "lat", "lon", "published"}


def test_relevance_filter():
    assert C.is_relevant("Data Analyst (m/w/d)") and C.is_relevant("KI-Engineer LLM")
    assert not C.is_relevant("Lagerist (m/w/d)")


class FakeClient:
    def __init__(self):
        self.requests, self.detail_calls = 0, []

    def search(self, term, page=1, days=30, angebotsart=1):
        self.requests += 1
        return SEARCH if page == 1 else {"stellenangebote": []}

    def details(self, refnr):
        self.requests += 1
        self.detail_calls.append(refnr)
        return {"stellenangebotsBeschreibung": f"Beschreibung {refnr}: SQL"}


def test_run_dedupes_skips_irrelevant_and_tracks_last_seen(tmp_path):
    path = tmp_path / "live.duckdb"
    client = FakeClient()
    s = C.run(["Data Analyst", "AI Engineer"], path=path, client=client, verbose=False)
    assert s["new"] == 2 and sorted(client.detail_calls) == ["10001-1", "10001-3"]   # each fetched once, though
    # every term is searched twice (employment and freelance)
    con = duckdb.connect(str(path))
    rows = con.execute("SELECT refnr, family, description, source FROM postings ORDER BY refnr").fetchall()
    assert [r[0] for r in rows] == ["10001-1", "10001-3"] and rows[0][1] == "Data Analyst" and rows[0][3] == "ba_api"
    con.close()
    client2 = FakeClient()
    s2 = C.run(["Data Analyst"], path=path, client=client2, verbose=False)
    assert s2["new"] == 0 and client2.detail_calls == []          # known postings: no detail calls


def test_details_url_is_base64_refnr():
    assert base64.b64encode(b"10001-1002716922-S").decode() == "MTAwMDEtMTAwMjcxNjkyMi1T"


class FakeResp:
    def __init__(self, status, payload=None):
        self.status_code, self._payload = status, payload
        self.text = "Forbidden" if status == 403 else str(payload)[:50]

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class PickySession:
    """Rejects everything except /pc/v6/jobs with size 50 and the browser user agent."""
    def __init__(self):
        self.headers, self.calls = {}, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {}), self.headers.get("User-Agent")))
        if url.endswith("/pc/v6/jobs") and (params or {}).get("size") == 50 and "Mozilla" in self.headers["User-Agent"]:
            return FakeResp(200, {"stellenangebote": [{"referenznummer": "1-A", "titel": "Data Analyst"}]})
        return FakeResp(403)


def test_probe_finds_the_working_variant():
    c = C.BAClient(session=PickySession())
    c._interval = 0
    attempts = c.probe()
    assert (c.search_path, c.page_size, c.user_agent) == ("/pc/v6/jobs", 50, "browser") and attempts[-1]["ok"]
    url, params, ua = c.session.calls[-1]
    c.search("Data Analyst")       # later searches use the found variant
    url, params, ua = c.session.calls[-1]
    assert url.endswith("/pc/v6/jobs") and params["size"] == 50 and "Mozilla" in ua


def test_probe_reports_every_attempt_when_nothing_works():
    class Nope(PickySession):
        def get(self, url, params=None, timeout=None):
            return FakeResp(403)
    c = C.BAClient(session=Nope())
    c._interval = 0
    import pytest
    with pytest.raises(C.BAUnavailable, match="403"):
        c.probe()


def test_parse_search_without_stellenangebote_key():
    rows = C.parse_search({"ergebnisse": [{"referenznummer": "9", "stellenangebotsTitel": "AI Engineer",
                                           "arbeitsorte": [{"ort": "Köln"}]}], "maxErgebnisse": 1})
    assert rows[0]["refnr"] == "9" and rows[0]["title"] == "AI Engineer" and rows[0]["city"] == "Köln"


def test_find_meta_handles_other_layouts():
    job = {"referenznummer": "7", "titel": "Data Engineer", "arbeitgeber": {"name": "Firma D"},
           "arbeitsorte": [{"adresse": {"ort": "Leipzig", "plz": "04109", "region": "Sachsen"},
                            "koordinaten": {"breite": 51.34, "laenge": 12.37}}],
           "veroeffentlichungsdatum": "2026-10-09T08:00:00"}
    r = C.parse_search({"stellenangebote": [job]})[0]
    assert (r["city"], r["postcode"], r["lat"], r["lon"], r["published"], r["employer"]) == \
        ("Leipzig", "04109", 51.34, 12.37, "2026-10-09", "Firma D")


def test_missing_dates_fall_back_to_first_seen(tmp_path):
    import pandas as pd
    from skillmarker import analytics as A, db
    con = db.connect(tmp_path / "x.duckdb")
    row = {c: None for c in db.POSTING_COLUMNS}
    row.update(refnr="1", title="Data Analyst", family="Data Analyst", first_seen=pd.Timestamp("2026-10-14 09:00"))
    db.upsert(con, "postings", pd.DataFrame([row]), ["refnr"])
    con.close()
    p, _ = A.load(tmp_path / "x.duckdb")
    assert p["published"].iloc[0] == pd.Timestamp("2026-10-14")


def test_known_postings_get_missing_fields_repaired(tmp_path):
    path = tmp_path / "live.duckdb"
    c1 = FakeClient()
    C.run(["Data Analyst"], path=path, client=c1, verbose=False)
    con = duckdb.connect(str(path))
    con.execute("UPDATE postings SET city = NULL, published = NULL")      # as if an older parser missed them
    con.close()
    C.run(["Data Analyst"], path=path, client=FakeClient(), verbose=False)
    con = duckdb.connect(str(path), read_only=True)
    assert con.execute("SELECT city, published FROM postings WHERE refnr = '10001-1'").fetchone() == ("Berlin", __import__("datetime").date(2026, 10, 1))


def test_v6_layout_as_returned_by_the_api():
    """A real v6 search result (shortened), as printed by --show-raw."""
    payload = {"ergebnisliste": [{
        "stellenangebotsTitel": "Data Analyst (w/m/d)", "firma": "Bechtle AG", "hauptberuf": "Data-Analyst/in",
        "stellenlokationen": [{"adresse": {"plz": "74172", "ort": "Neckarsulm", "region": "BADEN_WUERTTEMBERG"},
                               "breite": 49.1988851, "laenge": 9.2378473}],
        "eintrittszeitraum": {"von": "2026-10-10"}, "datumErsteVeroeffentlichung": "2026-10-10",
        "aenderungsdatum": "2026-10-09T16:57:35.703", "referenznummer": "15086-44497982-109-S"}],
        "maxErgebnisse": 1, "page": 1, "size": 50}
    r = C.parse_search(payload)[0]
    assert (r["refnr"], r["title"], r["employer"], r["occupation"]) == \
        ("15086-44497982-109-S", "Data Analyst (w/m/d)", "Bechtle AG", "Data-Analyst/in")
    assert (r["city"], r["postcode"], r["lat"], r["lon"], r["published"]) == ("Neckarsulm", "74172", 49.1988851, 9.2378473, "2026-10-10")



def test_employment_and_work_model_from_v6_flags():
    full = {"arbeitszeitVollzeit": True, "homeofficemoeglich": True, "homeofficetyp": "NACH_VEREINBARUNG"}
    part = {"arbeitszeitVollzeit": False, "arbeitszeitTeilzeitFlexibel": True}
    assert (C.employment_of(full), C.work_model_of(full)) == ("full_time", "hybrid")
    assert C.employment_of(part) == "part_time" and C.employment_of({}, angebotsart=2) == "freelance"
    assert C.work_model_of({"homeofficetyp": "AUSSCHLIESSLICH"}) == "remote" and C.work_model_of({}) == "unknown"


def test_part_time_and_student_jobs_are_skipped(tmp_path):
    payload = {"ergebnisliste": [
        {"referenznummer": "A", "stellenangebotsTitel": "Data Analyst (m/w/d)", "arbeitszeitVollzeit": True},
        {"referenznummer": "B", "stellenangebotsTitel": "Data Analyst Teilzeit", "arbeitszeitTeilzeitFlexibel": True},
        {"referenznummer": "C", "stellenangebotsTitel": "Werkstudent Data Analytics", "arbeitszeitVollzeit": False},
        {"referenznummer": "D", "stellenangebotsTitel": "Freelance AI Engineer", "arbeitszeitVollzeit": False}]}

    class Client:
        requests, page_size = 0, 50
        def search(self, term, page=1, days=30, angebotsart=1):
            return payload if page == 1 else {"ergebnisliste": []}
        def details(self, refnr):
            return {"stellenangebotsBeschreibung": "SQL"}

    C.run(["Data Analyst"], path=tmp_path / "x.duckdb", client=Client(), verbose=False)
    con = duckdb.connect(str(tmp_path / "x.duckdb"), read_only=True)
    rows = dict(con.execute("SELECT refnr, employment FROM postings").fetchall())
    assert rows["A"] == "full_time" and "B" not in rows and "C" not in rows
    assert rows["D"] == "freelance"        # found in the self-employment search


def test_reclassify_stored_postings(tmp_path):
    import pandas as pd
    from skillmarker import db
    con = db.connect(tmp_path / "r.duckdb")
    rows = []
    for refnr, title, fam in [("1", "Teamleiter Business Intelligence", "Business Intelligence"),
                              ("2", "Werkstudent Data Analytics", "Data Analyst"), ("3", "Agentic AI Engineer", "AI Engineer")]:
        r = {c: None for c in db.POSTING_COLUMNS}
        r.update(refnr=refnr, title=title, family=fam, source="ba_api")
        rows.append(r)
    db.upsert(con, "postings", pd.DataFrame(rows), ["refnr"])
    assert C.reclassify(con) == 3
    fams = dict(con.execute("SELECT refnr, family FROM postings").fetchall())
    assert fams == {"1": "Analytics Leadership", "2": None, "3": "Agentic AI & GenAI"}
