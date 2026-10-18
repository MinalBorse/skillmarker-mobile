import json

import pandas as pd
import pytest

from skillmarker import analytics as A
from skillmarker import ask, evaluate
from skillmarker import matching as M
from skillmarker.sample import generate


def test_sample_is_deterministic():
    a, ta = generate(50, seed=3)
    b, tb = generate(50, seed=3)
    assert a.drop(columns=["first_seen", "last_seen"]).equals(b.drop(columns=["first_seen", "last_seen"])) and ta.equals(tb)


def test_skill_demand_and_rising(sample_db):
    p, s = A.load(sample_db)
    d = A.skill_demand(p, s, top=5)
    assert d["share"].is_monotonic_decreasing and d["share"].max() <= 1
    r = A.rising(p, s, min_postings=5)
    assert "LangGraph" in r.head(8)["skill"].tolist() or "LLMs" in r.head(8)["skill"].tolist()   # planted trends


def test_cooccurrence_lift():
    s = pd.DataFrame({"refnr": ["a", "a", "b", "b", "c"], "skill": ["dbt", "SQL", "dbt", "SQL", "Excel"]})
    c = A.cooccurring(s, "dbt")
    assert c.iloc[0]["skill"] == "SQL" and c.iloc[0]["lift"] == pytest.approx(1 / (2 / 3))


def test_unknown_skills_counted():
    p = pd.DataFrame({"other_skills": [json.dumps(["Superset", "Metabase"]), json.dumps(["Superset"]), None]})
    u = A.unknown_skills(p)
    assert u.iloc[0].tolist() == ["Superset", 2]


def test_salaries_report_sample_size(sample_db):
    p, _ = A.load(sample_db)
    sal = A.salaries(p)
    assert (sal["n"] >= 1).all() and (sal["low"] <= sal["high"]).all()


def test_match_coverage_and_gaps(sample_db):
    cv = "SQL, Python, Power BI, Excel"
    have = M.cv_skills(cv)
    every = M.match(sample_db, cv, have, families=["Data Analyst"], top=None)
    r = every.iloc[0]
    req = set(r["have"]) | set(r["missing"])
    assert r["coverage"] == pytest.approx(len(r["have"]) / len(req))
    assert set(every["family"]) == {"Data Analyst"}
    fit = M.fit_overview(every)
    assert fit["postings"] == len(every) and 0 <= fit["share"] <= 1
    g = M.gaps(sample_db, have, ["Data Analyst"], matches=every.head(50))
    assert not set(g["skill"]) & set(have)


def test_read_cv_text():
    assert M.read_cv("SQL und Python".encode(), "cv.txt") == "SQL und Python"


def test_learning_plan_offline_is_grounded(sample_db):
    g = M.gaps(sample_db, ["SQL"], ["Data Analyst"])
    plan = M.learning_plan(["SQL"], g, "Data Analyst")
    assert g.iloc[0]["skill"] in plan


def test_ask_retrieves_and_answers(sample_db):
    hits = ask.retrieve(sample_db, "LangGraph agents", k=4)
    assert len(hits) == 4
    assert "4 most relevant postings" in ask.answer("q", hits)


def test_eval_score_math():
    s = evaluate.score([["A", "B"], ["C"]], [["A"], ["C", "D"]])
    assert s["precision"] == pytest.approx(2 / 3) and s["recall"] == pytest.approx(2 / 3)


def test_rules_have_perfect_recall_but_imperfect_precision(sample_db):
    t = evaluate.evaluate(sample_db, n=300)
    assert t.loc["rules", "recall"] > .99 and t.loc["rules", "precision"] < .97


def test_duplicates_and_part_time_are_never_counted(tmp_path):
    from skillmarker import db
    path = tmp_path / "scope.duckdb"
    con = db.connect(path)
    rows = []
    for refnr, title, employment, dup, family in [("1", "Data Analyst", "full_time", None, "Data Analyst"),
                                                  ("2", "Data Analyst", "full_time", "1", "Data Analyst"),
                                                  ("3", "Data Analyst Teilzeit", "part_time", None, "Data Analyst"),
                                                  ("4", "Sales Manager", "full_time", None, None),
                                                  ("5", "Freelance AI Engineer", "freelance", None, "AI Engineer")]:
        r = {c: None for c in db.POSTING_COLUMNS}
        r.update(refnr=refnr, title=title, employment=employment, dup_of=dup, family=family, source="ba_api",
                 first_seen=pd.Timestamp("2026-10-01"))
        rows.append(r)
    db.upsert(con, "postings", pd.DataFrame(rows), ["refnr"])
    con.close()
    p, _ = A.load(path)
    assert sorted(p["refnr"]) == ["1", "5"]
