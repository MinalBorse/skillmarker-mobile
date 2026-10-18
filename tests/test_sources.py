"""Career-page adapters, aggregators and de-duplication, with fake responses (no network)."""
import json

import duckdb
import pandas as pd
import pytest

from skillmarker import aggregators, careers, db, dedup
from skillmarker.roles import classify, employment_from_text, location_ok


class Resp:
    def __init__(self, status=200, data=None, text=None):
        self.status_code, self._data = status, data
        self.text = text if text is not None else json.dumps(data)
        self.content = self.text.encode()

    def json(self):
        return self._data


class Session:
    def __init__(self, routes):
        self.routes, self.headers, self.calls = routes, {}, []

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        for key, resp in self.routes.items():
            if key in url:
                return resp
        return Resp(404, {})


GREENHOUSE = {"jobs": [
    {"id": 1, "title": "Senior Data Analyst (m/w/d)", "absolute_url": "https://x/1", "location": {"name": "Berlin, Germany"},
     "updated_at": "2026-10-01T10:00:00Z", "content": "&lt;p&gt;We need &lt;b&gt;SQL&lt;/b&gt; and Python&lt;/p&gt;", "metadata": []},
    {"id": 2, "title": "Account Executive", "location": {"name": "Berlin"}, "content": ""},
    {"id": 3, "title": "Working Student Data Analytics", "location": {"name": "Berlin"}, "content": ""},
    {"id": 4, "title": "Data Scientist", "location": {"name": "New York, US"}, "content": ""},
    {"id": 5, "title": "ML Engineer", "location": {"name": "Remote - Europe"}, "content": "PyTorch"}]}
LEVER = [{"id": "a", "text": "Analytics Engineer", "hostedUrl": "https://jobs.lever.co/x/a",
          "categories": {"location": "Munich", "commitment": "Full-time"}, "workplaceType": "hybrid",
          "descriptionPlain": "dbt and Snowflake", "lists": [{"text": "Requirements", "content": "<li>SQL</li>"}], "createdAt": 1759312800000},
         {"id": "b", "text": "BI Developer", "categories": {"location": "Hamburg", "commitment": "Part-time"}}]
ASHBY = {"jobs": [{"id": "z", "title": "AI Engineer - Agents", "location": "Remote", "employmentType": "Contract",
                   "isRemote": True, "descriptionPlain": "LangGraph, RAG", "jobUrl": "https://ashby/z", "publishedAt": "2026-10-05"}]}
PERSONIO = """<?xml version="1.0"?><workzag-jobs><position><id>77</id><office>Köln</office><name>Data Engineer (m/w/d)</name>
<jobDescriptions><jobDescription><name>Dein Profil</name><value><![CDATA[<ul><li>Spark</li><li>Airflow</li></ul>]]></value></jobDescription></jobDescriptions>
<employmentType>permanent</employmentType><schedule>full-time</schedule><createdAt>2026-09-30T08:00:00+00:00</createdAt></position>
<position><id>78</id><office>Köln</office><name>Office Manager</name><employmentType>permanent</employmentType><schedule>full-time</schedule></position>
</workzag-jobs>"""


def test_greenhouse_filters_role_place_and_student_jobs():
    rows = careers.greenhouse(Session({"greenhouse.io/v1/boards/acme": Resp(200, GREENHOUSE)}), "acme", "Acme GmbH")
    titles = {r["title"]: r for r in rows}
    assert set(titles) == {"Senior Data Analyst (m/w/d)", "ML Engineer"}       # sales, student and US job dropped
    da = titles["Senior Data Analyst (m/w/d)"]
    assert "SQL" in da["description"] and "<" not in da["description"]         # HTML decoded and stripped
    assert (da["city"], da["lat"] is not None, da["family"], da["source"]) == ("Berlin", True, "Data Analyst", "careers:greenhouse")
    assert titles["ML Engineer"]["work_model"] == "remote"


def test_lever_tries_eu_instance_and_reads_commitment():
    s = Session({"api.eu.lever.co/v0/postings/acme": Resp(200, LEVER)})
    rows = careers.lever(s, "acme", "Acme")
    assert [r["title"] for r in rows] == ["Analytics Engineer"]                # part-time BI role dropped
    assert rows[0]["employment"] == "full_time" and rows[0]["work_model"] == "hybrid" and "SQL" in rows[0]["description"]
    assert rows[0]["published"] == "2025-10-01"


def test_ashby_contract_is_freelance_and_remote_is_kept():
    rows = careers.ashby(Session({"ashbyhq.com/posting-api/job-board/acme": Resp(200, ASHBY)}), "acme", "Acme")
    assert rows[0]["employment"] == "freelance" and rows[0]["work_model"] == "remote" and rows[0]["family"] == "Agentic AI & GenAI"


def test_personio_xml():
    rows = careers.personio(Session({"acme.jobs.personio.de/xml": Resp(200, None, PERSONIO)}), "acme", "Acme")
    assert len(rows) == 1 and rows[0]["title"] == "Data Engineer (m/w/d)" and "Airflow" in rows[0]["description"]
    assert rows[0]["external_url"] == "https://acme.jobs.personio.de/job/77" and rows[0]["city"] == "Köln"


def test_unknown_board_returns_none_for_discovery():
    assert careers.greenhouse(Session({}), "nope", "Nope") is None
    assert careers.personio(Session({"jobs.personio": Resp(200, None, "<html>not a job board</html>")}), "nope", "Nope") is None
    sr = Session({"smartrecruiters.com": Resp(200, {"totalFound": 0, "content": []})})
    assert careers.smartrecruiters(sr, "nope", "Nope") is None              # empty list, not 404, for unknown companies


def test_slug_candidates_and_discovery():
    assert careers.slug_candidates("Trade Republic GmbH") == ["traderepublic", "trade-republic", "trade"]
    assert careers.slug_candidates("Zühlke") == ["zuehlke"]
    s = Session({"api.ashbyhq.com/posting-api/job-board/trade-republic": Resp(200, {"jobs": []})})
    assert careers.discover("Trade Republic", s) == ("ashby", "trade-republic")
    assert careers.discover("Unknown Corp", Session({})) is None


def test_store_inserts_new_and_refreshes_known(tmp_path):
    rows = careers.greenhouse(Session({"greenhouse.io": Resp(200, GREENHOUSE)}), "acme", "Acme")
    path = tmp_path / "live.duckdb"
    assert careers.store(path, rows) == {"new": 2, "seen": 0}
    assert careers.store(path, rows) == {"new": 0, "seen": 2}


def test_arbeitnow_aggregator():
    data = {"data": [{"slug": "data-analyst-acme", "company_name": "Acme", "title": "Data Analyst", "location": "Berlin",
                      "description": "<p>SQL</p>", "remote": False, "url": "https://arbeitnow/x", "job_types": ["full time"],
                      "created_at": 1759312800},
                     {"slug": "koch", "company_name": "Restaurant", "title": "Koch", "location": "Berlin", "job_types": []}],
            "links": {"next": None}}
    rows = aggregators.arbeitnow(Session({"arbeitnow.com/api": Resp(200, data)}))
    assert [r["refnr"] for r in rows] == ["arbeitnow:data-analyst-acme"] and rows[0]["source"] == "arbeitnow"


def test_adzuna_needs_a_key(monkeypatch):
    from skillmarker import config
    monkeypatch.setattr(config, "ADZUNA_APP_ID", "")
    assert aggregators.adzuna(Session({})) == []


def _posting(refnr, employer, title, city, desc, source, work_model="unknown"):
    r = {c: None for c in db.POSTING_COLUMNS}
    r.update(refnr=refnr, employer=employer, title=title, city=city, description=desc, source=source,
             family=classify(title), work_model=work_model)
    return r


def test_dedup_keeps_the_richest_copy(tmp_path):
    path = tmp_path / "d.duckdb"
    con = db.connect(path)
    db.upsert(con, "postings", pd.DataFrame([
        _posting("ba:1", "Acme GmbH", "Data Analyst (m/w/d)", "Berlin", "short", "ba_api"),
        _posting("greenhouse:acme:1", "Acme", "Data Analyst", "Berlin, Germany", "a much longer description " * 20, "careers:greenhouse"),
        _posting("adzuna:9", "ACME GmbH", "Data Analyst (w/m/d)", "Berlin", "snippet", "adzuna"),
        _posting("ba:2", "Acme GmbH", "Data Engineer (m/w/d)", "Berlin", "other job", "ba_api"),
        _posting("ba:3", "Other AG", "Data Analyst (m/w/d)", "Berlin", "different employer", "ba_api")]), ["refnr"])
    con.close()
    assert dedup.run(path) == 2
    rows = dict(duckdb.connect(str(path), read_only=True).execute("SELECT refnr, dup_of FROM postings").fetchall())
    assert rows["ba:1"] == rows["adzuna:9"] == "greenhouse:acme:1"
    assert rows["greenhouse:acme:1"] is None and rows["ba:2"] is None and rows["ba:3"] is None


@pytest.mark.parametrize("loc, remote, ok", [("München", False, True), ("Remote, EU", True, True),
                                             ("Remote (US only)", True, False), ("Zurich", False, False)])
def test_location_rule(loc, remote, ok):
    assert location_ok(loc, remote) is ok


def test_employment_labels():
    assert employment_from_text("FullTime") == "full_time" and employment_from_text("Intern") == "internship"


# ---- Cloud storage (Hugging Face) with a fake remote --------------------------------------------
@pytest.fixture
def fake_remote(tmp_path, monkeypatch):
    import shutil
    from skillmarker import config, storage
    remote = tmp_path / "remote"
    remote.mkdir()
    monkeypatch.setenv("HF_TOKEN", "t")
    monkeypatch.setenv("HF_DATASET_REPO", "me/skillmarker-data")
    monkeypatch.setattr(storage, "USER_FILE", tmp_path / "user.json")
    monkeypatch.setattr(config, "LIVE_DB", tmp_path / "live.duckdb")

    def upload(path, name, message):
        shutil.copy(path, remote / name)

    def download(name):
        if not (remote / name).exists():
            return None
        target = (tmp_path / name)
        shutil.copy(remote / name, target)
        return target

    monkeypatch.setattr(storage, "upload", upload)
    monkeypatch.setattr(storage, "download", download)
    monkeypatch.setattr(storage, "ensure_repo", lambda: None)
    return remote


def test_saved_cv_has_no_contact_details(fake_remote):
    from skillmarker import storage
    storage.save_cv("Maria Beispiel, maria@example.com, +49 170 1234567\nSQL, Python, Power BI")
    data = json.loads((fake_remote / "user.json").read_text())
    assert "maria@example.com" not in data["cv"] and "1234567" not in data["cv"] and "Power BI" in data["cv"]
    storage.save_cv(None)
    assert json.loads((fake_remote / "user.json").read_text())["cv"] is None


def test_pasted_ads_survive_a_fresh_database(fake_remote, tmp_path):
    from skillmarker import config, extract as X, storage
    con = db.connect(config.LIVE_DB)
    row = _posting("manual:abc", "Example GmbH", "Data Analyst", "Berlin", "SQL and Python needed", "manual")
    db.upsert(con, "postings", pd.DataFrame([row]), ["refnr"])
    X.save_many(con, [("manual:abc", X.rule_extract("Data Analyst", "SQL and Python"), "rules", "h")])
    storage.remember_ad(con, "manual:abc")
    con.close()
    config.LIVE_DB.unlink()                                   # e.g. a new database from the collector
    db.connect(config.LIVE_DB).close()
    assert storage.apply_user_ads(config.LIVE_DB) == 1
    con = duckdb.connect(str(config.LIVE_DB), read_only=True)
    assert con.execute("SELECT employer FROM postings WHERE refnr = 'manual:abc'").fetchone() == ("Example GmbH",)
    assert {r[0] for r in con.execute("SELECT skill FROM posting_skills").fetchall()} >= {"SQL", "Python"}


def test_pull_applies_user_ads_to_downloaded_database(fake_remote):
    from skillmarker import config, storage
    db.connect(config.LIVE_DB).close()
    storage.push_live("x")
    assert (fake_remote / "live.duckdb").exists()
    assert storage.pull() == {"live": True, "user": False}


def test_storage_is_off_without_settings(monkeypatch):
    from skillmarker import storage
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert storage.remote_enabled() is False and storage.pull() == {"live": False, "user": False}
