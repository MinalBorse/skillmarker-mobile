import pytest

from skillmarker.privacy import scrub
from skillmarker.skills import CANONICAL, find_skills, normalise, spans


@pytest.mark.parametrize("text, skill", [("MS PowerBI", "Power BI"), ("PySpark", "Spark"), ("mit maschinellem Lernen", "Machine learning"),
                                         ("AI-Agenten bauen", "Agentic AI"), ("Vektordatenbanken", "Vector databases"),
                                         ("Claude Code", "AI coding assistants"), ("Statistik", "Statistics"), ("T-SQL", "SQL")])
def test_aliases(text, skill):
    assert skill in find_skills(text)


def test_r_only_as_a_word_and_case_sensitive():
    assert "R" in find_skills("Python oder R")
    assert "R" not in find_skills("R&D team, Herr Müller, für unsere Kunden")


def test_java_not_javascript():
    assert find_skills("JavaScript") == ["TypeScript"]


def test_normalise():
    assert normalise("power bi") == "Power BI" and normalise("Langgraph") == "LangGraph"
    assert normalise("Superset") is None and normalise("") is None


def test_spans_do_not_overlap():
    t = "SQL, Power BI, Python and Microsoft Power BI"
    s = spans(t)
    assert [n for _, _, n in s] == ["SQL", "Power BI", "Python", "Power BI"]
    assert all(a[1] <= b[0] for a, b in zip(s, s[1:]))


def test_taxonomy_names_unique():
    assert len(CANONICAL) == len(set(CANONICAL))


def test_scrub_removes_contact_details():
    text = ("Ihr Profil\n- SQL\n\nAnsprechpartnerin: Frau Anna Weber, anna.weber@firma.de\n"
            "Fragen? Herr Jonas Koch, Tel. +49 89 1234567, www.firma.de/jobs")
    out = scrub(text)
    for secret in ("Anna Weber", "anna.weber@firma.de", "Jonas Koch", "1234567", "www.firma.de"):
        assert secret not in out
    assert "SQL" in out
