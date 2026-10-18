"""Generate realistic sample job postings, with the correct answers recorded.

    python -m skillmarker.sample                 # 2,400 postings over 120 days -> data/sample.duckdb
    python -m skillmarker.sample --n 600 --seed 3

Why sample data: the app works immediately, without network or API key, and because we
know what each posting really asks for (the `truth` table), the extractors can be scored
objectively (see `evaluate.py`). Employers are fictional.

Realism built in: German and English ads, skill aliases ("PowerBI", "PySpark"), salaries in
several formats, contact details to scrub, and traps such as "keine SAP-Kenntnisse
erforderlich" (no SAP needed), which keyword matching gets wrong.
Trends built in: agentic AI, LangGraph, MCP and AI coding assistants rise over the period;
Tableau and Qlik decline; Agentic AI postings grow in number.
"""
import argparse
import json
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from . import config, db

FAMILY_WEIGHT = {"Data Analyst": .24, "Business Intelligence": .16, "Product Analyst": .06, "Data Scientist": .14,
                 "Analytics Engineer": .05, "Data Engineer": .15, "AI Engineer": .13, "Agentic AI": .07}

# Probability that a posting of this family asks for the skill.
BASE = {
    "Data Analyst": {"SQL": .85, "Python": .6, "Excel": .55, "Power BI": .45, "Tableau": .25, "Statistics": .4,
                     "Data visualisation": .45, "KPIs": .35, "Stakeholder management": .35, "R": .15, "Looker": .08,
                     "A/B testing": .1, "Communication": .3, "Google Analytics": .1, "dbt": .08, "Agile": .15},
    "Business Intelligence": {"Power BI": .7, "SQL": .8, "DAX": .35, "Data modelling": .45, "Excel": .4, "Tableau": .25,
                              "Qlik": .15, "SAP": .25, "ETL": .35, "Azure": .25, "Data visualisation": .4,
                              "Stakeholder management": .3, "KPIs": .3, "Python": .25},
    "Product Analyst": {"SQL": .85, "Python": .55, "Product analytics": .6, "A/B testing": .55, "Statistics": .45,
                        "Looker": .2, "Tableau": .15, "Stakeholder management": .4, "Google Analytics": .25,
                        "KPIs": .35, "dbt": .15, "BigQuery": .15},
    "Data Scientist": {"Python": .9, "Machine learning": .75, "SQL": .7, "Statistics": .55, "pandas": .35,
                       "scikit-learn": .35, "PyTorch": .2, "TensorFlow": .12, "Deep learning": .2, "Forecasting": .2,
                       "NLP": .15, "LLMs": .25, "MLOps": .15, "AWS": .2, "Azure": .15, "Git": .3, "R": .2,
                       "Databricks": .15, "Spark": .15},
    "Analytics Engineer": {"dbt": .8, "SQL": .9, "Python": .5, "Snowflake": .4, "BigQuery": .3, "Airflow": .35,
                           "Data modelling": .55, "Git": .45, "Looker": .25, "CI/CD": .2, "ETL": .3},
    "Data Engineer": {"Python": .85, "SQL": .85, "Spark": .45, "Airflow": .4, "Kafka": .3, "AWS": .4, "Azure": .35,
                      "GCP": .15, "Databricks": .35, "Snowflake": .25, "Docker": .35, "Kubernetes": .2, "ETL": .55,
                      "Git": .4, "CI/CD": .3, "Scala": .12, "Java": .12, "dbt": .2, "Data modelling": .3},
    "AI Engineer": {"Python": .9, "LLMs": .65, "RAG": .4, "Machine learning": .5, "PyTorch": .3, "LangChain": .3,
                    "LangGraph": .15, "Vector databases": .3, "Docker": .4, "Kubernetes": .2, "AWS": .3, "Azure": .3,
                    "APIs": .45, "MLOps": .3, "Hugging Face": .2, "Prompt engineering": .25, "Git": .35, "CI/CD": .25,
                    "LLM evaluation": .15, "Agentic AI": .2, "TypeScript": .1},
    "Agentic AI": {"Python": .9, "LLMs": .85, "Agentic AI": .9, "LangGraph": .45, "LangChain": .4, "RAG": .55,
                   "Vector databases": .35, "MCP": .3, "Prompt engineering": .4, "LLM evaluation": .35, "APIs": .5,
                   "AI coding assistants": .3, "TypeScript": .25, "Docker": .3, "Azure": .25, "AWS": .25},
}
# Multiplier from the start (t=0) to the end (t=1) of the period.
TREND = {"Agentic AI": (.55, 1.6), "LangGraph": (.45, 1.75), "MCP": (.3, 1.9), "AI coding assistants": (.45, 1.8),
         "LLMs": (.8, 1.25), "RAG": (.8, 1.3), "LLM evaluation": (.6, 1.5), "dbt": (.85, 1.2),
         "Tableau": (1.2, .8), "Qlik": (1.3, .7)}
FAMILY_TREND = {"Agentic AI": (.5, 1.7), "AI Engineer": (.85, 1.2), "Business Intelligence": (1.05, .95)}

SENIORITY = [("working_student", .08), ("junior", .15), ("mid", .40), ("senior", .28), ("lead", .09)]
BASE_SALARY = {"Data Analyst": 55e3, "Business Intelligence": 60e3, "Product Analyst": 62e3, "Data Scientist": 66e3,
               "Analytics Engineer": 68e3, "Data Engineer": 68e3, "AI Engineer": 75e3, "Agentic AI": 78e3}
SENIORITY_FACTOR = {"junior": .8, "mid": 1.0, "senior": 1.25, "lead": 1.5}
GERMAN_P = {"Business Intelligence": .78, "Data Analyst": .68, "Product Analyst": .5, "Data Scientist": .5,
            "Analytics Engineer": .45, "Data Engineer": .52, "AI Engineer": .42, "Agentic AI": .35}

CITIES = [("Berlin", "10115", "Berlin", 52.52, 13.405, .22), ("München", "80331", "Bayern", 48.137, 11.575, .17),
          ("Hamburg", "20095", "Hamburg", 53.551, 9.993, .12), ("Frankfurt am Main", "60311", "Hessen", 50.11, 8.682, .10),
          ("Köln", "50667", "Nordrhein-Westfalen", 50.938, 6.96, .07), ("Stuttgart", "70173", "Baden-Württemberg", 48.776, 9.177, .06),
          ("Düsseldorf", "40213", "Nordrhein-Westfalen", 51.227, 6.773, .06), ("Leipzig", "04109", "Sachsen", 51.34, 12.375, .03),
          ("Hannover", "30159", "Niedersachsen", 52.375, 9.732, .03), ("Nürnberg", "90402", "Bayern", 49.452, 11.077, .03),
          ("Karlsruhe", "76133", "Baden-Württemberg", 49.007, 8.404, .03), ("Dresden", "01067", "Sachsen", 51.05, 13.738, .02),
          ("Bonn", "53111", "Nordrhein-Westfalen", 50.737, 7.098, .02), ("Essen", "45127", "Nordrhein-Westfalen", 51.456, 7.012, .02),
          ("Bremen", "28195", "Bremen", 53.079, 8.802, .02), ("Mannheim", "68159", "Baden-Württemberg", 49.488, 8.466, .02)]
EMPLOYERS = ["Nordwind Logistik GmbH", "Rheinblick Versicherung AG", "Alpenstrom Energie GmbH", "Kranich Mobility GmbH",
             "Elbe Commerce GmbH", "Spreeware Analytics GmbH", "Isar Health Tech GmbH", "Mainhafen Bank AG",
             "Kontor Retail Group", "Brückenwerk Consulting GmbH", "Taunus Pharma AG", "Hanse Media GmbH",
             "Schwarzwald Automotive GmbH", "Lindenhof Software GmbH", "Weserport Logistics", "Bodensee Insurance AG",
             "Neckar Industrie GmbH", "Saale Energy Solutions", "Ostsee Travel GmbH", "Feldberg Robotics GmbH",
             "Kiezbank Digital GmbH", "Harz Data Services GmbH", "Moselwein Commerce GmbH", "Allgäu Food Tech GmbH",
             "Ruhrgebiet Netze AG", "Spessart Systems GmbH", "Altmühl Analytics GmbH", "Wattenmeer Wind GmbH"]
ROLE_TITLES = {
    "Data Analyst": (["Data Analyst", "Datenanalyst", "Data Analyst Marketing", "Data Analyst Controlling"],
                     ["Data Analyst", "Data Analyst – Marketing", "Data Analyst Finance"]),
    "Business Intelligence": (["Business Intelligence Analyst", "BI Developer", "Power BI Entwickler", "BI Consultant"],
                              ["BI Developer", "Business Intelligence Engineer", "Power BI Developer"]),
    "Product Analyst": (["Product Analyst", "Product Data Analyst"], ["Product Analyst", "Senior Product Analytics"]),
    "Data Scientist": (["Data Scientist", "Data Scientist – Pricing"], ["Data Scientist", "Data Scientist – Forecasting"]),
    "Analytics Engineer": (["Analytics Engineer"], ["Analytics Engineer"]),
    "Data Engineer": (["Data Engineer", "Data Engineer – Cloud"], ["Data Engineer", "Cloud Data Engineer"]),
    "AI Engineer": (["KI-Engineer", "AI Engineer", "Machine Learning Engineer"], ["AI Engineer", "Machine Learning Engineer", "LLM Engineer"]),
    "Agentic AI": (["Engineer KI-Agenten", "Generative AI Engineer"], ["Agentic AI Engineer", "AI Agent Developer", "Generative AI Engineer"]),
}
# Alternative spellings, so extraction isn't trivially exact matching.
ALIASES = {"Power BI": ["Power BI", "PowerBI", "MS Power BI"], "Spark": ["Spark", "PySpark"], "GCP": ["GCP", "Google Cloud"],
           "LLMs": ["LLMs", "Large Language Models", "Generative AI"], "Agentic AI": ["Agentic AI", "AI-Agenten", "Multi-Agent-Systeme"],
           "Vector databases": ["Vektordatenbanken", "Vector Databases (z. B. Qdrant)", "pgvector"],
           "AI coding assistants": ["GitHub Copilot", "Claude Code", "Cursor"], "Excel": ["Excel", "MS Excel"],
           "Product analytics": ["Product Analytics", "Amplitude", "Mixpanel"], "Statistics": ["Statistik", "statistischen Methoden"],
           "Data visualisation": ["Datenvisualisierung", "Dashboards"], "Data modelling": ["Datenmodellierung", "Star Schema"],
           "Machine learning": ["Machine Learning", "maschinellem Lernen"], "A/B testing": ["A/B-Tests", "Experimentation"],
           "Stakeholder management": ["Stakeholder-Management", "Abstimmung mit Stakeholdern"],
           "LLM evaluation": ["LLM-Evaluation", "Evals (z. B. Langfuse)"], "ETL": ["ETL-Prozessen", "Datenpipelines"],
           "Communication": ["Kommunikationsstärke"], "KPIs": ["KPIs", "Kennzahlen"], "Agile": ["agilen Methoden", "Scrum"]}
EN_ALIASES = {"Statistics": ["statistics"], "Data visualisation": ["data visualisation", "dashboards"],
              "Data modelling": ["data modelling"], "Machine learning": ["machine learning"], "A/B testing": ["A/B testing"],
              "Stakeholder management": ["stakeholder management"], "ETL": ["ETL", "data pipelines"],
              "Communication": ["communication skills"], "Agile": ["agile methods", "Scrum"],
              "Vector databases": ["vector databases (e.g. Qdrant)", "pgvector"], "Agentic AI": ["agentic AI", "multi-agent systems"],
              "LLM evaluation": ["LLM evaluation", "evals (e.g. Langfuse)"]}
FIRST = ["Anna", "Lukas", "Sophie", "Jonas", "Lea", "Felix", "Marie", "Paul", "Mia", "Leon", "Emma", "Ben"]
LAST = ["Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner", "Becker", "Hoffmann", "Koch"]


def _pick_skills(rng, family: str, t: float) -> list[str]:
    out = []
    for skill, p in BASE[family].items():
        a, b = TREND.get(skill, (1, 1))
        if rng.random() < min(.97, p * (a + (b - a) * t)):
            out.append(skill)
    return out or ["SQL"]


def _salary_text(rng, lo, hi, english: bool) -> str:
    style = rng.integers(0, 3)
    if english:
        return [f"Salary: €{lo / 1000:.0f}k–€{hi / 1000:.0f}k per year", f"{lo:,.0f} – {hi:,.0f} EUR gross per year",
                f"An annual salary between €{lo:,.0f} and €{hi:,.0f}"][style]
    de = lambda v: f"{v:,.0f}".replace(",", ".")  # noqa: E731
    return [f"Gehalt: {de(lo)} – {de(hi)} € brutto/Jahr", f"Ein Jahresgehalt zwischen {de(lo)} und {de(hi)} Euro",
            f"Vergütung: {lo / 1000:.0f}–{hi / 1000:.0f} T€ p. a."][style]


def make_posting(rng, i: int, day: date, t: float, family: str) -> tuple[dict, dict]:
    seniority = rng.choice([s for s, _ in SENIORITY], p=[p for _, p in SENIORITY])
    german_level = "none"
    if rng.random() < GERMAN_P[family]:
        german_level = "fluent" if rng.random() < .72 else "basic"
    english = german_level != "fluent" and rng.random() < .7
    skills = _pick_skills(rng, family, t)
    remote = rng.choice(["onsite", "hybrid", "remote"], p=[.2, .6, .2])
    salary = None
    if seniority != "working_student" and rng.random() < .22:
        mid = BASE_SALARY[family] * SENIORITY_FACTOR[seniority] * rng.uniform(.9, 1.1)
        salary = (round(mid * .9 / 1000) * 1000, round(mid * 1.1 / 1000) * 1000)
    years = {"working_student": None, "junior": int(rng.integers(0, 2)), "mid": int(rng.integers(2, 4)),
             "senior": int(rng.integers(4, 7)), "lead": int(rng.integers(6, 10))}[seniority]
    employer = str(rng.choice(EMPLOYERS))
    city = CITIES[rng.choice(len(CITIES), p=np.array([c[5] for c in CITIES]) / sum(c[5] for c in CITIES))]
    de_titles, en_titles = ROLE_TITLES[family]
    role = str(rng.choice(en_titles if english else de_titles))
    prefix = {"junior": "Junior ", "senior": "Senior ", "lead": "Lead "}.get(seniority, "")
    if seniority == "working_student":
        title = f"Working Student {role} (all genders)" if english else f"Werkstudent {role} (m/w/d)"
    else:
        title = f"{prefix}{role} ({'all genders' if english else 'm/w/d'})"

    def alias(s):
        pool = (EN_ALIASES if english else ALIASES).get(s) or ALIASES.get(s) or [s]
        return str(rng.choice(pool))

    must, nice = skills[: max(1, int(len(skills) * .65))], skills[max(1, int(len(skills) * .65)):]
    trap = "SAP" not in skills and family in ("Business Intelligence", "Data Analyst") and rng.random() < .25
    # Mentions that are not requirements: technology partners in the company blurb and
    # training perks. Keyword matching counts them; a reader (or an LLM) shouldn't.
    pool = [x for x in ["AWS", "Azure", "Snowflake", "Databricks", "Tableau", "Kubernetes", "Power BI"] if x not in skills]
    partner = str(rng.choice(pool)) if pool and rng.random() < .25 else None
    perk = str(rng.choice([x for x in pool if x != partner])) if len(pool) > 1 and rng.random() < .25 else None
    contact = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    email = f"jobs@{employer.split()[0].lower()}.de"
    if english:
        lines = [f"About us\n{employer} is a growing company in {city[0]}." + (f" We are a certified {partner} partner." if partner else ""),
                 "Your tasks\n- Turn data into decisions together with our business teams\n- Build and improve our data products",
                 "Your profile\n- " + "\n- ".join([f"Very good knowledge of {', '.join(alias(s) for s in must)}"] +
                                                  ([f"Ideally experience with {', '.join(alias(s) for s in nice)}"] if nice else []) +
                                                  ([f"{years}+ years of relevant experience"] if years else ["Currently enrolled in a degree programme"]) +
                                                  (["Fluent English"] + (["Basic German is a plus"] if german_level == "basic" else [])))]
        offer = {"onsite": "Work on site in our office in " + city[0], "hybrid": "Hybrid working (2 days per week in the office)",
                 "remote": "Fully remote within Germany"}[remote]
        lines.append("What we offer\n- " + offer + (f"\n- {_salary_text(rng, *salary, True)}" if salary else "") + "\n- 30 days of holiday"
                     + (f"\n- Learning budget, e.g. for {perk} courses" if perk else ""))
        lines.append(f"Contact person: {contact}, {email}, +49 30 {rng.integers(1000000, 9999999)}")
    else:
        req = [f"Sehr gute Kenntnisse in {', '.join(alias(s) for s in must)}"]
        if nice:
            req.append(f"Idealerweise Erfahrung mit {', '.join(alias(s) for s in nice)}")
        if trap:
            req.append("Keine SAP-Kenntnisse erforderlich")
        req.append(f"Mindestens {years} Jahre Berufserfahrung" if years else "Eingeschriebenes Studium")
        if german_level == "fluent":
            req.append(str(rng.choice(["Verhandlungssichere Deutschkenntnisse", "Deutsch fließend (C1)", "Sehr gute Deutschkenntnisse"])))
        elif german_level == "basic":
            req.append("Grundkenntnisse in Deutsch (B1)")
        req.append("Gute Englischkenntnisse")
        offer = {"onsite": f"Arbeit vor Ort in {city[0]}", "hybrid": "Hybrides Arbeiten (2 Tage pro Woche im Büro)",
                 "remote": "100 % Remote innerhalb Deutschlands"}[remote]
        lines = [f"Über uns\n{employer} ist ein wachsendes Unternehmen mit Sitz in {city[0]}." + (f" Wir sind zertifizierter {partner}-Partner." if partner else ""),
                 "Ihre Aufgaben\n- Sie unterstützen unsere Fachbereiche datengetrieben\n- Sie entwickeln unsere Datenprodukte weiter",
                 "Ihr Profil\n- " + "\n- ".join(req),
                 "Wir bieten\n- " + offer + (f"\n- {_salary_text(rng, *salary, False)}" if salary else "") + "\n- 30 Tage Urlaub"
                 + (f"\n- Weiterbildungsbudget, z. B. für Kurse in {perk}" if perk else ""),
                 f"Ansprechpartnerin: Frau {contact}, {email}, Tel. +49 89 {rng.integers(1000000, 9999999)}"]
    refnr = f"10000-{1100000000 + i}-S"
    posting = {"refnr": refnr, "family": family, "query": family, "title": title, "occupation": family,
               "employer": employer, "city": city[0], "postcode": city[1], "region": city[2],
               "lat": city[3] + rng.normal(0, .03), "lon": city[4] + rng.normal(0, .03), "published": day,
               "external_url": None, "description": "\n\n".join(lines), "source": "sample",
               "first_seen": datetime.combine(day, datetime.min.time()),
               "last_seen": datetime.combine(day + timedelta(days=int(rng.integers(7, 60))), datetime.min.time())}
    truth = {"refnr": refnr, "skills": json.dumps(sorted(skills)), "seniority": seniority,
             "salary_min": salary[0] if salary else None, "salary_max": salary[1] if salary else None,
             "german_level": german_level, "remote": remote}
    return posting, truth


def generate(n: int = 2400, days: int = 120, seed: int = 42, end: date | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    end = end or date.today()
    fams = list(FAMILY_WEIGHT)
    postings, truths = [], []
    for i in range(n):
        t = float(rng.beta(1.3, 1.0))                       # slightly more recent postings
        day = end - timedelta(days=int((1 - t) * (days - 1)))
        w = np.array([FAMILY_WEIGHT[f] * (lambda a, b: a + (b - a) * t)(*FAMILY_TREND.get(f, (1, 1))) for f in fams])
        family = str(rng.choice(fams, p=w / w.sum()))
        p, tr = make_posting(rng, i, day, t, family)
        postings.append(p)
        truths.append(tr)
    return pd.DataFrame(postings, columns=db.POSTING_COLUMNS), pd.DataFrame(truths)


def write(postings: pd.DataFrame, truth: pd.DataFrame, path=None):
    path = path or config.SAMPLE_DB
    if path.exists():
        path.unlink()
    con = db.connect(path)
    db.upsert(con, "postings", postings, ["refnr"])
    db.upsert(con, "truth", truth, ["refnr"])
    con.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2400)
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    postings, truth = generate(args.n, args.days, args.seed)
    write(postings, truth)
    print(f"Generated {len(postings):,} sample postings -> {config.SAMPLE_DB}")
    print("Next: python -m skillmarker.extract   (extracts skills; uses the LLM if a key is set)")


if __name__ == "__main__":
    main()
