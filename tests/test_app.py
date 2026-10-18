from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ["Skills in demand", "Trends", "Roles and pay", "Where the jobs are", "Match my CV", "Ask the market", "Data and costs"]


@pytest.fixture
def app(sample_db):
    return AppTest.from_file(APP, default_timeout=180).run()


def md(at):
    return " ".join(m.value for m in at.markdown)


def test_every_page_renders(app):
    for page in PAGES:
        app.sidebar.radio(key="page").set_value(page).run()
        assert not app.exception, f"{page}: {app.exception}"


def test_role_filter_changes_headline(app):
    before = md(app)
    app.radio(key="sk_family").set_value("AI Engineer").run()
    assert not app.exception and md(app) != before and "AI Engineer postings" in md(app)


def test_cv_match_flow_and_privacy(app):
    app.sidebar.radio(key="page").set_value("Match my CV").run()
    next(b for b in app.button if b.label == "Use a sample CV").click().run()
    text = md(app)
    assert not app.exception and "postings you're aiming for" in text and 'class="ad"' in text
    assert "maria.beispiel@example.com" not in text


def test_ask_flow(app):
    app.sidebar.radio(key="page").set_value("Ask the market").run()
    app.text_input(key="ask_q").set_value("Which jobs need dbt?").run()
    assert not app.exception and 'class="answer"' in md(app)


def test_real_postings_get_apply_links(sample_db, monkeypatch):
    import duckdb
    from skillmarker import config
    con = duckdb.connect(str(config.SAMPLE_DB))
    con.execute("UPDATE postings SET source = 'ba_api'")       # pretend these came from the Bundesagentur
    con.close()
    try:
        at = AppTest.from_file(APP, default_timeout=180).run()
        at.sidebar.radio(key="page").set_value("Match my CV").run()
        next(b for b in at.button if b.label == "Use a sample CV").click().run()
        text = md(at)
        assert "arbeitsagentur.de/jobsuche/jobdetail/" in text and "Read the full ad and apply" in text
        assert any(b.label.startswith("Download your top 100") for b in at.get("download_button"))
    finally:
        con = duckdb.connect(str(config.SAMPLE_DB))
        con.execute("UPDATE postings SET source = 'sample'")
        con.close()


def test_sample_postings_have_no_links(app):
    app.sidebar.radio(key="page").set_value("Match my CV").run()
    next(b for b in app.button if b.label == "Use a sample CV").click().run()
    assert "arbeitsagentur.de/jobsuche" not in md(app)


def test_filters_narrow_the_postings(app):
    before = md(app)
    app.sidebar.radio(key="f_work").set_value("Remote only").run()
    assert not app.exception and md(app) != before
    app.sidebar.multiselect(key="f_emp").set_value(["Freelance"]).run()     # the sample has no freelance roles
    assert not app.exception and "No postings match these filters" in md(app)


def test_add_a_job_ad(app, sample_db):
    app.sidebar.radio(key="page").set_value("Add a job ad").run()
    app.text_input(key="ad_title").set_value("Senior Data Analyst (m/w/d)").run()
    app.text_input(key="ad_company").set_value("Example GmbH").run()
    app.text_input(key="ad_city").set_value("Berlin").run()
    app.text_area(key="ad_text").set_value("Your profile: SQL, Python and Power BI. Kontakt: jobs@example.de").run()
    next(b for b in app.button if b.label == "Add and analyse").click().run()
    assert not app.exception and any("Saved as a Data Analyst posting" in s.value for s in app.success)
    text = md(app)
    assert '<mark class="any"' in text and "jobs@example.de" not in text          # highlighted, contact removed
    import duckdb
    from skillmarker import config
    con = duckdb.connect(str(config.LIVE_DB), read_only=True)
    row = con.execute("SELECT source, family, employer FROM postings WHERE source = 'manual'").fetchone()
    assert row == ("manual", "Data Analyst", "Example GmbH")
    assert con.execute("SELECT count(*) FROM posting_skills").fetchone()[0] >= 3


def test_password_gate(sample_db, monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "secret")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.sidebar.radio and at.text_input(key="pw")          # nothing visible before logging in
    at.text_input(key="pw").set_value("wrong").run()
    assert any("isn't right" in e.value for e in at.error)
    at.text_input(key="pw").set_value("secret").run()
    assert not at.exception and at.sidebar.radio(key="page")
