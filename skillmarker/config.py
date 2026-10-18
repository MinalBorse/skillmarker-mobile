"""Configuration: where data lives, which jobs we search for, and the AI settings."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("SKILLMARKER_DATA_DIR", ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DB = DATA_DIR / "sample.duckdb"   # realistic generated postings with known answers
LIVE_DB = DATA_DIR / "live.duckdb"       # real postings from the Bundesagentur für Arbeit

# ---- Bundesagentur für Arbeit job search (free, public client id, no registration) -------
BA_BASE_URL = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service"
BA_API_KEY = os.getenv("BA_API_KEY", "jobboerse-jobsuche")
BA_REQUESTS_PER_SECOND = float(os.getenv("BA_REQUESTS_PER_SECOND", 2))

# Role families and search terms live in roles.py (16 families, German and English titles).
# The old 8-family list below is kept only for the sample data.
SAMPLE_FAMILIES = {
    "Data Analyst": ["Data Analyst", "Datenanalyst", "Data Analytics"],
    "Business Intelligence": ["Business Intelligence", "BI Analyst", "BI Developer", "Power BI"],
    "Product Analyst": ["Product Analyst", "Product Analytics"],
    "Data Scientist": ["Data Scientist"],
    "Analytics Engineer": ["Analytics Engineer"],
    "Data Engineer": ["Data Engineer"],
    "AI Engineer": ["AI Engineer", "KI Engineer", "Machine Learning Engineer", "LLM Engineer"],
    "Agentic AI": ["Agentic AI", "AI Agents", "KI-Agenten", "Generative AI Engineer"],
}

# ---- LLM (any OpenAI-compatible API; OpenRouter by default) -------------------------------
LLM_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY")
                         else "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini" if os.getenv("OPENROUTER_API_KEY") else "gpt-4o-mini")
# USD per million tokens, used when the API doesn't report the cost itself. Check your model's page.
PRICE_INPUT_PER_M = float(os.getenv("LLM_PRICE_INPUT_PER_M", 0.15))
PRICE_OUTPUT_PER_M = float(os.getenv("LLM_PRICE_OUTPUT_PER_M", 0.60))
# Mock mode: a rule-based stand-in, so everything (tests, CI, demo) runs without a key or cost.
MOCK_LLM = os.getenv("SKILLMARKER_MOCK_LLM", "").lower() in ("1", "true", "yes") or not LLM_API_KEY

# ---- Embeddings (local, CPU, multilingual: postings are in German and English) -------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# "auto" uses fastembed if the model can be loaded, otherwise an offline hashing embedder.
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "auto")

from .roles import SEARCH_TERMS as ROLE_FAMILIES  # noqa: E402  (used by the collectors)

# Company career pages to follow (edit freely); see careers.py
COMPANIES_FILE = ROOT / "companies.json"
# Optional aggregator key (https://developer.adzuna.com, free tier)
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY", "")
