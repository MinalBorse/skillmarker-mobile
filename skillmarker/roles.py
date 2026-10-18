"""Role families: which analytics and AI roles we track, and how to recognise them by title.

Every source uses this. The Bundesagentur and aggregators are searched with SEARCH_TERMS;
company career pages return all of a company's jobs, so their titles are classified here
and everything that isn't an analytics or AI role is dropped.

Also decides what is out of scope: part-time, working-student, internship, apprenticeship
and mini-job postings (only full-time and freelance roles are kept).
"""
import re

# family -> (search terms for keyword-search sources, title patterns). Order matters:
# more specific families come first ("Analytics Engineer" before "Data Analyst").
FAMILIES = {
    "Agentic AI & GenAI": (["Agentic AI", "AI Agent", "KI-Agenten", "Generative AI", "Prompt Engineer"],
                           [r"agentic", r"\bai[- ]agents?\b", r"ki[- ]agent", r"\b(?:ai|ki)\b.*\bagent(?:s|en)?\b", r"gen(?:erative)?[- ]?ai", r"generative ki",
                            r"prompt engineer", r"\bllm\b.*(?:engineer|developer|entwickler)"]),
    "AI Product & Strategy": (["AI Product Manager", "AI Consultant", "KI-Berater", "AI Solutions Architect"],
                              [r"\b(?:ai|ki)\b.*(?:product (?:manager|owner)|produktmanager)", r"\b(?:ai|ki)[- ](?:consultant|berater)",
                               r"\b(?:ai|ki)\b.*(?:strateg|solutions? architect|transformation)", r"ai governance"]),
    "Analytics Leadership": (["Head of Data", "Head of Analytics", "Analytics Manager", "Teamleiter BI"],
                             [r"head of (?:data|analytics|bi|ai|business intelligence)", r"(?:analytics|data|bi) (?:manager|lead|director)",
                              r"(?:teamleit|abteilungsleit|leiter)\w*.*(?:data|daten|\bbi\b|business intelligence|analytics|\bki\b)", r"chief data"]),
    "ML Engineer & MLOps": (["Machine Learning Engineer", "MLOps Engineer"],
                            [r"machine[- ]learning[- ](?:engineer|developer|entwickler)", r"\bml[- ]?ops\b", r"\bml engineer"]),
    "AI Engineer": (["AI Engineer", "KI Engineer", "KI-Entwickler", "NLP Engineer", "Computer Vision Engineer"],
                    [r"\b(?:ai|ki)[- ](?:engineer|entwickler|developer|specialist|spezialist)", r"artificial intelligence",
                     r"künstliche intelligenz", r"\bnlp\b", r"computer vision", r"deep learning", r"\bllm\b"]),
    "Analytics Engineer": (["Analytics Engineer"], [r"analytics engineer", r"\bdbt\b"]),
    "Data Architect & Governance": (["Data Architect", "Data Governance", "Data Steward", "Data Quality"],
                                    [r"data architect", r"datenarchitekt", r"data governance", r"data steward",
                                     r"data quality", r"datenqualität", r"master data", r"stammdaten"]),
    "Data Engineer": (["Data Engineer", "ETL Developer", "Data Platform Engineer"],
                      [r"data engineer", r"data platform", r"\betl\b", r"data warehouse (?:engineer|developer|entwickler)",
                       r"daten(?:ingenieur|plattform)", r"big data"]),
    "Data Scientist": (["Data Scientist", "Decision Scientist"],
                       [r"data scien", r"decision scien", r"statistiker", r"\bstatistician", r"machine learning",
                        r"quantitative analyst"]),
    "SAP Analytics": (["SAP BW", "SAP Analytics Cloud", "SAP Datasphere"],
                      [r"sap[- ]?(?:bw|bi\b|analytics|datasphere|sac\b|hana)", r"bw/4hana", r"datasphere"]),
    "Process Mining": (["Process Mining", "Celonis"], [r"process mining", r"celonis", r"prozessanaly", r"process analyst"]),
    "Business Intelligence": (["Business Intelligence", "BI Developer", "BI Consultant", "Power BI", "Tableau"],
                              [r"business intelligence", r"\bbi[- ]", r"\bbi\b", r"power ?bi", r"tableau", r"qlik", r"reporting (?:developer|engineer)"]),
    "Product Analyst": (["Product Analyst", "Product Analytics"], [r"product (?:data )?analy"]),
    "Marketing & Web Analyst": (["Marketing Analyst", "Web Analyst", "Digital Analytics"],
                                [r"marketing (?:data )?analy", r"web ?analy", r"digital analy", r"crm analy", r"e-?commerce analy"]),
    "Finance & Controlling Analytics": (["Financial Analyst", "FP&A Analyst", "Controlling Analyst"],
                                       [r"financ\w* analy", r"fp&a", r"controlling.*(?:analy|data|bi)", r"(?:analy|data).*controlling"]),
    "Data Analyst": (["Data Analyst", "Datenanalyst", "Data Analytics", "Reporting Analyst"],
                     [r"data analy", r"datenanaly", r"daten-analy", r"reporting analy", r"insights? analy", r"business analy",
                      r"analytics"]),
}
FAMILY_NAMES = list(FAMILIES)
SEARCH_TERMS = {f: terms for f, (terms, _) in FAMILIES.items()}
_PATTERNS = [(f, re.compile("|".join(p), re.IGNORECASE)) for f, (_, p) in FAMILIES.items()]

# Out of scope: only full-time and freelance roles are kept.
EXCLUDED_TITLE = re.compile(
    r"werkstudent|working student|praktik|\bintern(?:ship)?\b|ausbildung|azubi|minijob|aushilfe|"
    r"\bteilzeit\b|part[- ]time|abschlussarbeit|masterarbeit|bachelorarbeit|thesis|duales studium|trainee",
    re.IGNORECASE)


def classify(title: str) -> str | None:
    """The role family of a job title, or None if it isn't an analytics or AI role."""
    t = re.sub(r"\((?:m|w|d|f|x|div|all genders?)[/|,\s\w]*\)", " ", title or "", flags=re.IGNORECASE)
    for family, pat in _PATTERNS:
        if pat.search(t):
            return family
    return None


def in_scope_title(title: str) -> bool:
    """False for student jobs, internships, apprenticeships, part-time and mini-jobs."""
    return not EXCLUDED_TITLE.search(title or "")


# ---- Employment type and work model, normalised across sources ----------------------------
FULL_TIME, FREELANCE, PART_TIME, INTERNSHIP, UNKNOWN = "full_time", "freelance", "part_time", "internship", "unknown"
KEEP_EMPLOYMENT = {FULL_TIME, FREELANCE, UNKNOWN}


def employment_from_text(*values) -> str:
    """Normalise an employment-type label from any source ("Full-time", "Vollzeit", "contract", ...)."""
    text = " ".join(str(v) for v in values if v).lower()
    if not text:
        return UNKNOWN
    if re.search(r"intern|praktik|werkstudent|working student|trainee|ausbildung|apprentice", text):
        return INTERNSHIP
    if re.search(r"freelanc|freiberuf|contract|contractor|selbst(?:ae|ä)ndig|interim|projektbasis", text):
        return FREELANCE
    if re.search(r"part[- _]?time|teilzeit|parttime|minijob", text):
        return PART_TIME
    if re.search(r"full[- _]?time|vollzeit|fulltime|permanent|unbefristet|festanstellung", text):
        return FULL_TIME
    return UNKNOWN


def work_model_from_text(*values) -> str:
    text = " ".join(str(v) for v in values if v).lower().replace("_", " ")
    if re.search(r"remote|home ?office|homeoffice|work from home|telearbeit|mobiles arbeiten|anywhere", text):
        return "hybrid" if re.search(r"hybrid|teilweise|nach vereinbarung|tage", text) else "remote"
    if re.search(r"hybrid", text):
        return "hybrid"
    if re.search(r"on-?site|vor ort|in office|präsenz", text):
        return "onsite"
    return "unknown"


GERMANY = re.compile(r"germany|deutschland|\bde\b|berlin|münchen|munich|hamburg|frankfurt|köln|cologne|stuttgart|"
                     r"düsseldorf|dusseldorf|leipzig|dresden|hannover|nürnberg|nuremberg|bonn|essen|dortmund|bremen|"
                     r"karlsruhe|mannheim|heidelberg|potsdam|darmstadt|wiesbaden|freiburg|münster|augsburg|regensburg",
                     re.IGNORECASE)
REMOTE_OPEN_TO_GERMANY = re.compile(r"remote.*(?:germany|deutschland|europe|eu\b|emea|dach|worldwide|global|anywhere)|"
                                    r"(?:europe|eu|emea|dach|worldwide|anywhere|global).*remote|^remote$", re.IGNORECASE)


def location_ok(location: str, remote: bool = False) -> bool:
    """Jobs in Germany, or remote jobs open to people living in Germany."""
    loc = (location or "").strip()
    if GERMANY.search(loc):
        return True
    if remote and (not loc or REMOTE_OPEN_TO_GERMANY.search(loc) or re.search(r"remote", loc, re.IGNORECASE)):
        return not re.search(r"\b(?:us|usa|united states|canada|uk only|india|singapore|australia)\b", loc, re.IGNORECASE)
    return False
