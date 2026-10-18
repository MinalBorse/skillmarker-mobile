import json
from types import SimpleNamespace as NS

import duckdb
import pytest

from skillmarker import config, extract as X


@pytest.mark.parametrize("text, lo, hi", [("Gehalt: 55.000 – 65.000 € brutto/Jahr", 55000, 65000),
                                           ("Salary: €60k–€70k per year", 60000, 70000),
                                           ("Vergütung: 58–64 T€ p. a.", 58000, 64000),
                                           ("An annual salary between €58,000 and €66,000", 58000, 66000)])
def test_salary_formats(text, lo, hi):
    e = X.rule_extract("Data Analyst", text)
    assert (e.salary_min, e.salary_max) == (lo, hi)


@pytest.mark.parametrize("title, level", [("Werkstudent Data Analytics (m/w/d)", "working_student"),
                                          ("Senior Data Engineer", "senior"), ("Lead AI Engineer", "lead"),
                                          ("Junior BI Developer", "junior"), ("Data Analyst", "mid")])
def test_seniority_from_title(title, level):
    assert X.rule_extract(title, "").seniority == level


def test_language_and_remote():
    e = X.rule_extract("x", "Verhandlungssichere Deutschkenntnisse. Hybrides Arbeiten. Gute Englischkenntnisse.")
    assert (e.german_level, e.remote, e.english_required) == ("fluent", "hybrid", True)
    e = X.rule_extract("x", "Basic German is a plus. Fully remote within Germany.")
    assert (e.german_level, e.remote) == ("basic", "remote")


def test_implausible_salary_rejected():
    assert X.Extraction(salary_min=5).salary_min is None


class FakeClient:
    def __init__(self, payload, cost=None):
        self.payload, self.cost, self.calls = payload, cost, []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        usage = NS(prompt_tokens=1000, completion_tokens=100, model_extra={"cost": self.cost} if self.cost else {})
        return NS(usage=usage, choices=[NS(message=NS(content=json.dumps(self.payload)))])


def test_llm_extract_normalises_and_keeps_unknown_skills():
    c = FakeClient({"skills": ["python", "PowerBI", "Superset"], "seniority": "senior", "german_level": "fluent",
                    "remote": "hybrid", "salary_min": 70000, "salary_max": 80000})
    ex, call = X.llm_extract("Senior BI", "Kontakt: Frau Anna Weber, anna@x.de", c)
    assert ex.skills == ["Python", "Power BI"] and ex.other_skills == ["Superset"]
    assert "anna@x.de" not in c.calls[0]["messages"][1]["content"]       # privacy
    assert call["cost_usd"] == pytest.approx((1000 * config.PRICE_INPUT_PER_M + 100 * config.PRICE_OUTPUT_PER_M) / 1e6)


def test_reported_cost_wins():
    _, call = X.llm_extract("x", "y", FakeClient({"skills": []}, cost=0.00042))
    assert call["cost_usd"] == 0.00042


def test_rules_run_and_cache(sample_db):
    stats = X.run(sample_db, method="rules", verbose=False)
    assert stats["todo"] == 0      # already extracted by the fixture, text unchanged: cached


def test_llm_budget_stops_the_run(sample_db, monkeypatch, tmp_path):
    import shutil
    db_copy = tmp_path / "copy.duckdb"
    shutil.copy(sample_db, db_copy)
    monkeypatch.setattr(config, "MOCK_LLM", False)
    monkeypatch.setattr(X, "_client", lambda: None)
    monkeypatch.setattr(X, "llm_extract", lambda t, d, c: (X.Extraction(skills=["SQL"]),
                        {"model": "m", "input_tokens": 1, "output_tokens": 1, "cost_usd": 0.05, "latency_ms": 1}))
    stats = X.run(db_copy, method="llm", max_cost=0.12, workers=2, verbose=False)
    assert stats["stopped_at_budget"] and stats["done"] < 600
    con = duckdb.connect(str(db_copy), read_only=True)
    assert con.execute("SELECT sum(cost_usd) FROM llm_calls").fetchone()[0] == pytest.approx(stats["cost_usd"])


def test_llm_without_key_explains(sample_db):
    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        X.run(sample_db, method="llm")
