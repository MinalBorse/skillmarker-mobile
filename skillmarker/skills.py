"""Skill taxonomy: canonical names, categories and the ways postings write them.

One list drives everything: rule-based extraction, normalising the LLM's answers
("PowerBI", "MS Power BI" -> "Power BI"), and comparing a CV with postings.
Aliases are regular expressions, matched case-insensitively on word boundaries.
"""
import re

# (canonical name, category, [alias regexes])
TAXONOMY = [
    # Programming languages
    ("Python", "Languages", [r"python"]),
    ("SQL", "Languages", [r"sql", r"t-sql", r"pl/sql", r"postgresql", r"mysql"]),
    ("R", "Languages", [r"(?<![\w&-])R(?![\w&-])"]),
    ("Java", "Languages", [r"java(?!script)"]),
    ("Scala", "Languages", [r"scala"]),
    ("TypeScript", "Languages", [r"typescript", r"javascript"]),
    ("DAX", "Languages", [r"dax"]),
    # BI and visualisation
    ("Power BI", "BI & visualisation", [r"power\s?bi", r"powerbi"]),
    ("Tableau", "BI & visualisation", [r"tableau"]),
    ("Looker", "BI & visualisation", [r"looker"]),
    ("Qlik", "BI & visualisation", [r"qlik\w*"]),
    ("Excel", "BI & visualisation", [r"excel"]),
    ("Data visualisation", "BI & visualisation", [r"data visuali[sz]ation", r"datenvisualisierung", r"dashboards?"]),
    # Data platforms and engineering
    ("dbt", "Data platforms", [r"dbt"]),
    ("Airflow", "Data platforms", [r"airflow"]),
    ("Spark", "Data platforms", [r"spark", r"pyspark"]),
    ("Databricks", "Data platforms", [r"databricks"]),
    ("Snowflake", "Data platforms", [r"snowflake"]),
    ("BigQuery", "Data platforms", [r"bigquery"]),
    ("Kafka", "Data platforms", [r"kafka"]),
    ("ETL", "Data platforms", [r"etl", r"elt", r"data pipelines?", r"datenpipelines?"]),
    ("Data modelling", "Data platforms", [r"data model(?:l)?ing", r"datenmodellierung", r"star schema"]),
    ("SAP", "Data platforms", [r"sap(?:\s?(?:bw|hana|s/4))?"]),
    # Cloud and engineering
    ("AWS", "Cloud & engineering", [r"aws", r"amazon web services"]),
    ("Azure", "Cloud & engineering", [r"azure"]),
    ("GCP", "Cloud & engineering", [r"gcp", r"google cloud"]),
    ("Docker", "Cloud & engineering", [r"docker"]),
    ("Kubernetes", "Cloud & engineering", [r"kubernetes", r"k8s"]),
    ("Git", "Cloud & engineering", [r"git(?:hub|lab)?"]),
    ("CI/CD", "Cloud & engineering", [r"ci/cd", r"continuous integration"]),
    ("APIs", "Cloud & engineering", [r"rest[- ]?apis?", r"fastapi", r"apis"]),
    # Machine learning
    ("Machine learning", "ML & statistics", [r"machine learning", r"maschinelle[mns]? lernen", r"ml-?modell\w*"]),
    ("Deep learning", "ML & statistics", [r"deep learning"]),
    ("Statistics", "ML & statistics", [r"statisti\w+"]),
    ("A/B testing", "ML & statistics", [r"a/b[- ]test\w*", r"experimentation"]),
    ("Forecasting", "ML & statistics", [r"forecasting", r"prognose\w*", r"time series"]),
    ("pandas", "ML & statistics", [r"pandas"]),
    ("scikit-learn", "ML & statistics", [r"scikit-learn", r"sklearn"]),
    ("PyTorch", "ML & statistics", [r"pytorch"]),
    ("TensorFlow", "ML & statistics", [r"tensorflow"]),
    ("NLP", "ML & statistics", [r"nlp", r"natural language processing"]),
    ("MLOps", "ML & statistics", [r"mlops", r"mlflow"]),
    # Generative AI and agents
    ("LLMs", "GenAI & agents", [r"llms?", r"large language models?", r"generative ai", r"genai", r"generative ki"]),
    ("RAG", "GenAI & agents", [r"rag", r"retrieval[- ]augmented"]),
    ("LangChain", "GenAI & agents", [r"langchain"]),
    ("LangGraph", "GenAI & agents", [r"langgraph"]),
    ("Agentic AI", "GenAI & agents", [r"agentic", r"ai[- ]agent(?:s|en)?", r"ki-agenten", r"multi-agent\w*"]),
    ("Prompt engineering", "GenAI & agents", [r"prompt engineering", r"prompting"]),
    ("Vector databases", "GenAI & agents", [r"vector (?:databases?|stores?|db)", r"vektordatenbank\w*", r"pinecone",
                                            r"qdrant", r"weaviate", r"chroma(?:db)?", r"pgvector"]),
    ("LLM evaluation", "GenAI & agents", [r"llm[- ]eval\w*", r"evals", r"langsmith", r"langfuse"]),
    ("MCP", "GenAI & agents", [r"mcp", r"model context protocol"]),
    ("AI coding assistants", "GenAI & agents", [r"github copilot", r"cursor", r"claude code", r"ai[- ]assisted coding",
                                                 r"agentic coding"]),
    ("Hugging Face", "GenAI & agents", [r"hugging\s?face", r"transformers"]),
    # Product and business
    ("Product analytics", "Business & product", [r"product analytics", r"amplitude", r"mixpanel", r"funnel analys\w+"]),
    ("Google Analytics", "Business & product", [r"google analytics", r"ga4"]),
    ("KPIs", "Business & product", [r"kpis?", r"kennzahlen"]),
    ("Stakeholder management", "Business & product", [r"stakeholder\w*"]),
    ("Agile", "Business & product", [r"agil\w*", r"scrum", r"kanban"]),
    ("Data governance", "Business & product", [r"data governance", r"datenschutz", r"gdpr", r"dsgvo"]),
    ("Communication", "Business & product", [r"communication skills", r"kommunikationsstärke", r"kommunikationsfähigkeit"]),
]

CANONICAL = [t[0] for t in TAXONOMY]
CATEGORY = {t[0]: t[1] for t in TAXONOMY}
_LOOKUP = {name.lower(): name for name in CANONICAL}
_PATTERNS = [(name, re.compile(r"(?<![\w/])(?:" + "|".join(aliases) + r")(?![\w])", re.IGNORECASE | re.UNICODE))
             for name, _, aliases in TAXONOMY]
# "R" is case-sensitive: a lowercase "r" in German text is never the language.
_PATTERNS = [(n, re.compile(p.pattern, re.UNICODE) if n == "R" else p) for n, p in _PATTERNS]


def find_skills(text: str) -> list[str]:
    """All canonical skills mentioned in a text, in taxonomy order."""
    if not text:
        return []
    return [name for name, pat in _PATTERNS if pat.search(text)]


def normalise(raw: str) -> str | None:
    """Map a free-text skill (e.g. from an LLM) to its canonical name, or None if unknown."""
    if not raw:
        return None
    key = raw.strip().lower()
    if key in _LOOKUP:
        return _LOOKUP[key]
    found = find_skills(raw)
    return found[0] if len(found) == 1 else None


def spans(text: str) -> list[tuple[int, int, str]]:
    """Non-overlapping (start, end, skill) positions of skill mentions, for highlighting."""
    found = sorted(((m.start(), m.end(), name) for name, pat in _PATTERNS for m in pat.finditer(text or "")),
                   key=lambda x: (x[0], -(x[1] - x[0])))
    out, last_end = [], -1
    for s, e, name in found:
        if s >= last_end:
            out.append((s, e, name))
            last_end = e
    return out
