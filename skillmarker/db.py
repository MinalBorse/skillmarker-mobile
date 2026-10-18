"""DuckDB storage: one file per source (sample or live)."""
from pathlib import Path

import duckdb
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS postings (
    refnr VARCHAR PRIMARY KEY,      -- the Bundesagentur's reference number
    family VARCHAR,                 -- role family whose search found it first
    query VARCHAR,
    title VARCHAR,
    occupation VARCHAR,             -- the BA's occupation label ("beruf")
    employer VARCHAR,
    city VARCHAR,
    postcode VARCHAR,
    region VARCHAR,
    lat DOUBLE,
    lon DOUBLE,
    published DATE,
    external_url VARCHAR,
    description VARCHAR,
    source VARCHAR,                 -- 'sample', 'ba_api', 'careers:<system>', 'arbeitnow', 'adzuna', 'manual'
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,            -- last collection run that still found it online
    employment VARCHAR,             -- full_time, freelance, part_time, internship, unknown (from the source)
    work_model VARCHAR,             -- remote, hybrid, onsite, unknown (from the source)
    dup_of VARCHAR                  -- set when the same job was already found in another source
);
CREATE TABLE IF NOT EXISTS extractions (
    refnr VARCHAR PRIMARY KEY,
    method VARCHAR,                 -- 'llm' or 'rules'
    model VARCHAR,
    content_hash VARCHAR,           -- re-extract only when the posting text changes
    seniority VARCHAR,
    salary_min DOUBLE,              -- EUR per year
    salary_max DOUBLE,
    german_level VARCHAR,           -- none, basic, fluent
    english_required BOOLEAN,
    remote VARCHAR,                 -- onsite, hybrid, remote, unknown
    years_experience INTEGER,
    other_skills VARCHAR,           -- JSON list of skills outside the taxonomy
    extracted_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS posting_skills (
    refnr VARCHAR, skill VARCHAR, category VARCHAR, PRIMARY KEY (refnr, skill)
);
CREATE TABLE IF NOT EXISTS llm_calls (
    ts TIMESTAMP, refnr VARCHAR, purpose VARCHAR, model VARCHAR, input_tokens INTEGER,
    output_tokens INTEGER, cost_usd DOUBLE, latency_ms INTEGER, ok BOOLEAN
);
CREATE TABLE IF NOT EXISTS embeddings (refnr VARCHAR PRIMARY KEY, model VARCHAR, vector FLOAT[]);
CREATE TABLE IF NOT EXISTS truth (
    refnr VARCHAR PRIMARY KEY, skills VARCHAR, seniority VARCHAR, salary_min DOUBLE, salary_max DOUBLE,
    german_level VARCHAR, remote VARCHAR
);
CREATE TABLE IF NOT EXISTS collection_log (
    run_at TIMESTAMP, family VARCHAR, query VARCHAR, found INTEGER, new INTEGER, details INTEGER, status VARCHAR
);
"""

POSTING_COLUMNS = ["refnr", "family", "query", "title", "occupation", "employer", "city", "postcode", "region",
                   "lat", "lon", "published", "external_url", "description", "source", "first_seen", "last_seen",
                   "employment", "work_model", "dup_of"]


def connect(path, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        con.execute(SCHEMA)
        # Databases created by earlier versions get the new columns.
        for col in ("employment VARCHAR", "work_model VARCHAR", "dup_of VARCHAR"):
            con.execute(f"ALTER TABLE postings ADD COLUMN IF NOT EXISTS {col}")
        con.execute("ALTER TABLE extractions ADD COLUMN IF NOT EXISTS employment VARCHAR")
    return con


def upsert(con, table: str, df: pd.DataFrame, key: list[str]) -> int:
    if df.empty:
        return 0
    con.register("_in", df)
    on = " AND ".join(f"t.{k} = i.{k}" for k in key)
    con.execute(f"DELETE FROM {table} t USING _in i WHERE {on}")
    cols = ", ".join(df.columns)
    con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _in")
    con.unregister("_in")
    return len(df)


def has_data(path) -> bool:
    if not Path(path).exists():
        return False
    try:
        con = connect(path, read_only=True)
        n = con.execute("SELECT count(*) FROM postings").fetchone()[0]
        con.close()
        return n > 0
    except Exception:
        return False
