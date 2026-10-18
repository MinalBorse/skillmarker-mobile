"""Skillmarker: the German data and AI job market, highlighted.  Run: streamlit run app.py"""
import html

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import hmac
import os

from skillmarker import analytics as A
from skillmarker import ask, config, db, storage
from skillmarker import embeddings as E
from skillmarker import matching as M
from skillmarker.collect import ad_url
from skillmarker.privacy import scrub
from skillmarker.skills import spans

st.set_page_config(page_title="Skillmarker", page_icon="🖍️", layout="wide", initial_sidebar_state="auto")

PAPER, INK, PEN, YELLOW, GREEN, PINK, MUTED, RULE = ("#EEF1F4", "#16202B", "#2B3A8C", "#FFE45C", "#9BE7A6",
                                                     "#FF9EC7", "#5A6675", "#D5DBE3")
FAMILY_COLORS = ["#2B3A8C", "#4F7BD9", "#C9A400", "#2E8B57", "#D6457F", "#8A5CC2", "#E07A2E", "#5A6675"]

CSS = f"""
<style>
/* Archivo and Source Serif 4 are bundled in ./static/fonts (SIL Open Font License). */
@font-face {{ font-family: 'Archivo'; src: url('app/static/fonts/Archivo.ttf') format('truetype');
             font-weight: 100 900; font-stretch: 62% 125%; font-display: swap; }}
@font-face {{ font-family: 'Source Serif 4'; src: url('app/static/fonts/SourceSerif4.ttf') format('truetype');
             font-weight: 200 900; font-display: swap; }}
html, body, p, li, label, input, textarea, button, select,
div[data-testid="stMarkdownContainer"], div[data-testid="stWidgetLabel"] {{ font-family: 'Archivo', Arial, sans-serif; }}
[data-testid="stIconMaterial"], span.material-symbols-rounded {{ font-family: 'Material Symbols Rounded' !important; }}
h1, h2, h3 {{ font-family: 'Archivo', Arial, sans-serif !important; font-stretch: 110%; font-weight: 700 !important;
             color: {INK}; letter-spacing: -0.01em; }}
div[data-testid="stMarkdownContainer"] p.headline {{ font-size: clamp(1.9rem, 3.2vw, 2.9rem) !important;
    font-stretch: 112%; font-weight: 750; line-height: 1.08 !important; color: {INK}; max-width: 24ch; margin: .2rem 0 .6rem; }}
div[data-testid="stMarkdownContainer"] p.subline {{ font-size: 1.12rem !important; color: {MUTED}; max-width: 68ch;
    line-height: 1.5; margin-bottom: 1.3rem; }}
.note {{ color: {MUTED}; font-size: .95rem; max-width: 72ch; }}

/* The highlighter ladder: each bar is a marker stroke whose length is the share. */
.ladder {{ margin: .4rem 0 1.6rem; max-width: 900px; }}
.rung {{ display: grid; grid-template-columns: 11rem 1fr; align-items: center; column-gap: 14px; margin: 3px 0; }}
.rung .name {{ font-weight: 600; font-size: 1.02rem; color: {INK}; text-align: right; }}
.rung .track {{ position: relative; height: 1.75rem; }}
.stroke {{ position: absolute; left: 0; top: .18rem; height: 1.4rem; background: {YELLOW};
          border-radius: 6px 16px 7px 13px / 12px 7px 13px 9px; transform-origin: left center;
          animation: draw .7s cubic-bezier(.2,.7,.2,1) both; animation-delay: calc(var(--i) * 45ms); opacity: .92; }}
.stroke.have {{ background: {GREEN}; }} .stroke.miss {{ background: {PINK}; }}
.rung .val {{ position: absolute; top: .22rem; font-variant-numeric: tabular-nums; font-size: .98rem; color: {INK};
             padding-left: 8px; white-space: nowrap; }}
@keyframes draw {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}
@media (prefers-reduced-motion: reduce) {{ .stroke {{ animation: none; }} }}
@media (max-width: 640px) {{
  .rung {{ grid-template-columns: 6.5rem 1fr; column-gap: 8px; }}
  .rung .name {{ font-size: .9rem; }}
  .rung .val {{ font-size: .85rem; }}
  .ad {{ padding: 14px 14px 12px; }}
  .ad .fit {{ float: none; display: block; margin-bottom: 4px; }}
  .ad .links a {{ display: block; margin: 6px 0; }}
  div[data-testid="stMarkdownContainer"] p.headline {{ font-size: 1.7rem !important; }}
  .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
}}

/* Job ads as paper, marked up with highlighters. */
.ad {{ background: #fff; border: 1px solid {RULE}; border-radius: 3px; padding: 18px 22px 16px; margin: 0 0 14px;
      max-width: 860px; }}
.ad h4 {{ font-family: 'Archivo', Arial, sans-serif; font-size: 1.12rem; font-weight: 700; margin: 0 0 2px; color: {INK}; }}
.ad .meta {{ color: {MUTED}; font-size: .92rem; margin-bottom: 10px; }}
.ad .body {{ font-family: 'Source Serif 4', Georgia, serif; font-size: 1.02rem; line-height: 1.62; color: #22303F; }}
.ad .fit {{ float: right; font-weight: 700; font-size: 1.05rem; color: {PEN}; }}
mark {{ padding: 0 .18em; border-radius: .3em .55em .35em .5em; color: inherit; }}
mark.have {{ background: {GREEN}; }} mark.miss {{ background: {PINK}; }} mark.any {{ background: {YELLOW}; }}
.key {{ font-size: .95rem; color: {MUTED}; margin: -4px 0 14px; }}
.chips mark {{ display: inline-block; margin: 0 6px 6px 0; font-size: .98rem; font-weight: 500; }}
.ad .links {{ margin-top: 12px; font-size: .98rem; }}
.ad .links a {{ color: {PEN}; font-weight: 600; text-decoration: underline; text-underline-offset: 3px; margin-right: 18px; }}
.ad .links a:focus-visible {{ outline: 2px solid {PEN}; outline-offset: 2px; }}
.answer {{ background: #fff; border-left: 4px solid {PEN}; padding: 14px 18px; max-width: 860px; margin-bottom: 18px;
          font-size: 1.05rem; line-height: 1.55; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PLOT = dict(font=dict(family="Archivo, Arial, sans-serif", size=14, color=INK),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=54, b=10),
            colorway=FAMILY_COLORS, legend=dict(orientation="h", y=-0.2, title=None))


def style(fig, height=380, title=None):
    fig.update_layout(height=height, **PLOT)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(family="Archivo, Arial", size=20, color=INK)))
    fig.update_xaxes(gridcolor=RULE, zeroline=False)
    fig.update_yaxes(gridcolor=RULE, zeroline=False)
    return fig


def pct(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:.0f}%"


def esc(s) -> str:
    return html.escape(str(s or ""))


# ---- Access and cloud data ------------------------------------------------------------------
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
if APP_PASSWORD and not st.session_state.get("authed"):
    # A simple gate for a personal hosted app: the code repo is public, your data isn't.
    st.markdown('<p class="headline">Skillmarker</p>', unsafe_allow_html=True)
    pw = st.text_input("Password", type="password", key="pw")
    if st.button("Open", type="primary") or pw:
        if hmac.compare_digest(pw.encode(), APP_PASSWORD.encode()):
            st.session_state["authed"] = True
            st.rerun()
        elif pw:
            st.error("That password isn't right.")
    st.stop()


@st.cache_resource(ttl=6 * 3600, show_spinner="Fetching the latest postings ...")
def cloud_sync(_nonce: int = 0) -> dict:
    """Hosted mode: download the collector's database and your saved data (every 6 hours at most)."""
    return storage.pull() if storage.remote_enabled() else {}


SYNC = cloud_sync(st.session_state.get("sync_nonce", 0))


# ---- Data ---------------------------------------------------------------------------------
SOURCES = {"Sample postings": config.SAMPLE_DB, "Live postings (Bundesagentur)": config.LIVE_DB}


@st.cache_data(show_spinner="Loading postings ...")
def load(path: str, _v: float):
    return A.load(path)


@st.cache_resource(show_spinner="Loading the text model (first time: a one-off download) ...")
def embedder():
    return E.get_embedder()


def version(p) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


with st.sidebar:
    st.markdown("## Skillmarker")
    st.caption("Which skills Germany's data and AI employers ask for, and how your CV compares.")
    available = [k for k, p in SOURCES.items() if db.has_data(p)]
    source = st.radio("Data", available or ["Sample postings"], key="source")
    page = st.radio("View", ["Skills in demand", "Trends", "Roles and pay", "Where the jobs are", "Match my CV",
                             "Add a job ad", "Ask the market", "Data and costs"], key="page")

path = SOURCES[source]
if not db.has_data(path):
    st.markdown('<p class="headline">No job postings yet</p>', unsafe_allow_html=True)
    st.markdown('<p class="subline">Generate 2,400 realistic sample postings to explore the app now, or collect real '
                'postings from the Bundesagentur für Arbeit (see the README).</p>', unsafe_allow_html=True)
    if st.button("Generate sample postings", type="primary"):
        from skillmarker import extract, sample
        with st.spinner("Writing postings and extracting skills ..."):
            sample.write(*sample.generate())
            extract.run(config.SAMPLE_DB, method="rules", verbose=False)
        st.cache_data.clear()
        st.rerun()
    st.stop()

ver = version(path)
P_ALL, S_ALL = load(str(path), ver)
SOURCE_LABELS = {"ba_api": "Bundesagentur", "arbeitnow": "Arbeitnow", "adzuna": "Adzuna", "manual": "Added by you",
                 "sample": "Sample"}


def source_label(src: str) -> str:
    src = str(src or "")
    return f"Career page ({src.split(':', 1)[1].capitalize()})" if src.startswith("careers:") else SOURCE_LABELS.get(src, src)


with st.sidebar:
    st.markdown("#### Filters")
    emp_choice = st.multiselect("Employment", ["Full-time", "Freelance", "Not stated"],
                                default=["Full-time", "Freelance", "Not stated"], key="f_emp")
    work_choice = st.radio("Work model", ["Any", "Remote or hybrid", "Remote only"], key="f_work")
    P_ALL["source_label"] = P_ALL["source"].map(source_label)
    sources = sorted(P_ALL["source_label"].dropna().unique())
    src_choice = st.multiselect("Sources", sources, default=sources, key="f_src") if len(sources) > 1 else sources
emp_map = {"Full-time": "full_time", "Freelance": "freelance", "Not stated": "unknown"}
keep = P_ALL["employment"].isin([emp_map[e] for e in emp_choice]) & P_ALL["source_label"].isin(src_choice)
if work_choice == "Remote or hybrid":
    keep &= P_ALL["work_model"].isin(["remote", "hybrid"])
elif work_choice == "Remote only":
    keep &= P_ALL["work_model"] == "remote"
P_ALL = P_ALL[keep].reset_index(drop=True)
S_ALL = S_ALL[S_ALL["refnr"].isin(P_ALL["refnr"])].reset_index(drop=True)
ALLOWED = set(P_ALL["refnr"])
if P_ALL.empty:
    st.markdown('<p class="headline">No postings match these filters</p>', unsafe_allow_html=True)
    st.markdown('<p class="subline">Widen the filters in the sidebar, for example allow "Not stated" employment or any '
                'work model.</p>', unsafe_allow_html=True)
    st.stop()
FAMILIES = sorted(P_ALL["family"].dropna().unique())
with st.sidebar:
    if source.startswith("Sample"):
        st.caption("Sample postings: realistic but generated, with fictional employers.")
    else:
        st.caption("Data: Bundesagentur für Arbeit, company career pages, Arbeitnow, Adzuna.")
    if storage.remote_enabled() and st.button("Refresh data", width="stretch"):
        st.session_state["sync_nonce"] = st.session_state.get("sync_nonce", 0) + 1
        st.cache_data.clear()
        st.rerun()


# ---- Building blocks ------------------------------------------------------------------------
def ladder(df: pd.DataFrame, kind: str = "any", value=lambda r: pct(r.share), max_share: float | None = None):
    """Skill bars drawn as highlighter strokes; the stroke's length is the share."""
    top = max_share or float(df["share"].max() or 1)
    labels = [value(r) for r in df.itertuples()]
    # Leave room on the right for the label: long labels get shorter bars.
    room = 82 if max((len(x) for x in labels), default=0) <= 14 else 52
    rows = []
    for i, r in enumerate(df.itertuples()):
        w = max(2.0, r.share / top * room)
        cls = "" if kind == "any" else kind
        rows.append(f'<div class="rung"><span class="name">{esc(r.skill)}</span><span class="track">'
                    f'<span class="stroke {cls}" style="width:{w:.1f}%;--i:{i}"></span>'
                    f'<span class="val" style="left:{w:.1f}%">{esc(value(r))}</span></span></div>')
    st.markdown(f'<div class="ladder" role="list">{"".join(rows)}</div>', unsafe_allow_html=True)


def marked(text: str, have: set | None = None) -> str:
    """HTML of the text with skill mentions highlighted: green if you have it, pink if not."""
    out, pos = [], 0
    for s, e, name in spans(text):
        out.append(esc(text[pos:s]))
        cls = "any" if have is None else ("have" if name in have else "miss")
        out.append(f'<mark class="{cls}" title="{esc(name)}">{esc(text[s:e])}</mark>')
        pos = e
    out.append(esc(text[pos:]))
    # Real <br> tags: raw newlines (and blank lines) inside HTML would let Streamlit's
    # Markdown parser break the card apart.
    return "".join(out).replace("\n", "<br>")


SECTION_STARTS = ("ihr profil", "your profile", "anforderungen", "requirements", "das bringst du mit", "what you bring")
SECTION_ENDS = ("wir bieten", "what we offer", "unser angebot", "benefits", "das bieten wir", "ansprechpartner", "contact")


def profile_excerpt(description: str, n: int = 520) -> str:
    """The requirements section of an ad (where the skills are), ending at the next section."""
    text = scrub(description)
    low = text.lower()
    start = min((i for i in (low.find(k) for k in SECTION_STARTS) if i >= 0), default=0)
    ends = [i for i in (low.find(k, start + 10) for k in SECTION_ENDS) if i > start]
    end = min(ends + [start + n])
    cut = text[start:end].strip()
    while "\n\n" in cut:
        cut = cut.replace("\n\n", "\n")
    return cut + ("…" if end == start + n and len(text) > end else "")


def ad_card(r, have: set | None, fit: str | None = None):
    when = pd.Timestamp(r.published).strftime("%d %b %Y") if pd.notna(r.published) else ""
    fit_html = f'<span class="fit">{fit}</span>' if fit else ""
    links = ""
    src, ext = str(getattr(r, "source", "") or ""), getattr(r, "external_url", None)
    ext = ext if isinstance(ext, str) and ext.startswith("http") else None
    if src == "ba_api":   # sample postings are fictional: no links
        links = (f'<a href="{esc(ad_url(r.refnr))}" target="_blank" rel="noopener">Read the full ad and apply on '
                 'arbeitsagentur.de</a>' + (f'<a href="{esc(ext)}" target="_blank" rel="noopener">Original ad</a>' if ext else ""))
    elif ext and src != "sample":
        where = "the company's careers page" if src.startswith("careers:") else source_label(src)
        links = f'<a href="{esc(ext)}" target="_blank" rel="noopener">Read the full ad and apply on {esc(where)}</a>'
    links = f'<div class="links">{links}</div>' if links else ""
    extra = ", ".join(x for x in [{"full_time": "full-time", "freelance": "freelance"}.get(getattr(r, "employment", ""), ""),
                                  {"remote": "remote", "hybrid": "hybrid"}.get(getattr(r, "work_model", ""), "")] if x)
    st.markdown(f'<div class="ad">{fit_html}<h4>{esc(r.title)}</h4><div class="meta">{esc(r.employer)}, {esc(r.city)}, '
                f'{when}{", " + esc(extra) if extra else ""}</div><div class="body">{marked(profile_excerpt(r.description), have)}</div>{links}</div>',
                unsafe_allow_html=True)


def family_picker(key: str, label: str = "Role") -> str | None:
    choice = st.radio(label, ["All roles"] + FAMILIES, horizontal=True, key=key)
    return None if choice == "All roles" else choice


# ---- Pages ------------------------------------------------------------------------------------
def page_skills():
    top_slot = st.container()        # headline first, though it depends on the role picked below it
    fam = family_picker("sk_family")
    p = P_ALL if fam is None else P_ALL[P_ALL["family"] == fam]
    d = A.skill_demand(P_ALL, S_ALL, fam, top=15)
    who = "data and AI postings" if fam is None else f"{fam} postings"
    if not d.empty:
        top_slot.markdown(f'<p class="headline">{esc(d["skill"].iloc[0])} appears in {pct(d["share"].iloc[0])} of '
                    f'{len(p):,} {who}.</p>', unsafe_allow_html=True)
        top_slot.markdown(f'<p class="subline">From {p["employer"].nunique():,} employers, published between '
                    f'{p["published"].min():%d %b} and {p["published"].max():%d %b %Y}. The bars show the share of '
                    'postings that ask for each skill.</p>', unsafe_allow_html=True)
        ladder(d)
    r = A.rising(p, S_ALL[S_ALL["refnr"].isin(p["refnr"])])
    if not r.empty:
        st.markdown("### Rising fastest")
        st.markdown('<p class="note">Change in the share of postings asking for the skill: the last 4 weeks '
                    'compared with the weeks before.</p>', unsafe_allow_html=True)
        up = r.head(8).assign(share=lambda x: x["change_pts"].clip(lower=0.1) / 100)
        ladder(up, value=lambda row: f"+{row.change_pts:.1f} points  ({pct(row.share_before)} to {pct(row.share_recent)})")
    u = A.unknown_skills(p)
    if not u.empty:
        st.markdown("### New to the list")
        st.markdown('<p class="note">Skills the AI found that aren\'t in the skill list yet; candidates to add.</p>',
                    unsafe_allow_html=True)
        st.markdown('<div class="chips">' + "".join(f'<mark class="any">{esc(s)} ({n})</mark>' for s, n in
                    zip(u["skill"], u["postings"])) + "</div>", unsafe_allow_html=True)


def page_trends():
    st.markdown('<p class="headline">How demand is moving</p>', unsafe_allow_html=True)
    w = A.weekly(P_ALL)
    w = w[w["week"] < w["week"].max()]  # the current week is incomplete
    fig = px.line(w, x="week", y="postings", color="family", title="New postings per week",
                  labels={"week": "", "postings": "Postings", "family": ""})
    st.plotly_chart(style(fig, 400), width="stretch")
    r = A.rising(P_ALL, S_ALL)
    default = r["skill"].head(4).tolist() if not r.empty else []
    choice = st.multiselect("Skills", sorted(S_ALL["skill"].unique()), default=default, key="tr_skills")
    if choice:
        t = A.skill_trend(P_ALL, S_ALL, choice)
        t = t[t["month"] > t["month"].min()] if t["month"].nunique() > 2 else t   # first month is partial
        fig = px.line(t, x="month", y="share", color="skill", markers=True, title="Share of postings asking for each skill",
                      labels={"month": "", "share": "", "skill": ""})
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style(fig, 400), width="stretch")
    skill = st.selectbox("Skills that come together with", sorted(S_ALL["skill"].unique()),
                         index=sorted(S_ALL["skill"].unique()).index("Python") if "Python" in set(S_ALL["skill"]) else 0,
                         key="tr_cooc")
    c = A.cooccurring(S_ALL, skill)
    if not c.empty:
        st.markdown(f'<p class="note">Postings asking for {esc(skill)} also ask for these, more often than average '
                    '(the number is how many times more likely).</p>', unsafe_allow_html=True)
        ladder(c.assign(share=c["together"]), value=lambda row: f"{pct(row.together)} of them, {row.lift:.1f}× average")


def page_roles():
    st.markdown('<p class="headline">What the roles look like</p>', unsafe_allow_html=True)
    disclosed = A.salary_disclosure(P_ALL)
    st.markdown(f'<p class="subline">Only {pct(disclosed)} of postings state a salary, so salary figures rest on small '
                'samples; the number of postings is shown with each.</p>', unsafe_allow_html=True)
    sm = A.seniority_mix(P_ALL)
    names = {"working_student": "Working student", "junior": "Junior", "mid": "Mid-level", "senior": "Senior", "lead": "Lead"}
    sm["level"] = sm["seniority"].astype(str).map(names)
    fig = px.bar(sm, y="family", x="share", color="level", orientation="h", title="Seniority asked for",
                 labels={"share": "", "family": "", "level": ""},
                 color_discrete_sequence=["#BFD3F2", "#8FB0E8", "#4F7BD9", PEN, "#16205A"])
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(style(fig, 420), width="stretch")

    sal = A.salaries(P_ALL)
    sal = sal[sal["seniority"].isin(["junior", "mid", "senior", "lead"]) & (sal["n"] >= 3)]
    if not sal.empty:
        sal["level"] = sal["seniority"].map(names)
        fig = go.Figure()
        for i, (lvl, g) in enumerate(sal.groupby("level", sort=False)):
            fig.add_scatter(x=g["median"], y=g["family"], mode="markers", name=lvl,
                            marker=dict(size=13, color=["#8FB0E8", "#4F7BD9", PEN, "#16205A"][i % 4]),
                            error_x=dict(type="data", symmetric=False, array=g["high"] - g["median"],
                                         arrayminus=g["median"] - g["low"], color=RULE, thickness=6, width=0),
                            customdata=g["n"], hovertemplate="%{y}, " + lvl + ": €%{x:,.0f} (n=%{customdata})<extra></extra>")
        fig.update_xaxes(tickprefix="€", tickformat=",.0f")
        st.plotly_chart(style(fig, 440, "Stated salaries: median and middle half, per year"), width="stretch")

    lang = A.language(P_ALL)
    cols = [c for c in ("fluent", "basic", "none") if c in lang.columns]
    long = lang.melt(id_vars="family", value_vars=cols, var_name="German", value_name="share")
    long["German"] = long["German"].map({"fluent": "Fluent German", "basic": "Basic German", "none": "No German mentioned"})
    fig = px.bar(long, y="family", x="share", color="German", orientation="h", title="German language requirements",
                 labels={"share": "", "family": "", "German": ""}, color_discrete_sequence=[PEN, "#8FB0E8", "#D5DBE3"])
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(style(fig, 420), width="stretch")
    rm = A.remote_mix(P_ALL)
    st.markdown('<p class="note">Remote policy across all postings: ' + ", ".join(
        f"{esc(r.remote)} {pct(r.share)}" for r in rm.itertuples()) + ".</p>", unsafe_allow_html=True)


def page_where():
    st.markdown('<p class="headline">Where the jobs are</p>', unsafe_allow_html=True)
    fam = family_picker("wh_family")
    p = P_ALL if fam is None else P_ALL[P_ALL["family"] == fam]
    c = A.cities(p, top=16)
    if c.empty:
        st.info("These postings have no locations.")
        return
    fig = go.Figure(go.Scatter(x=c["lon"], y=c["lat"], mode="markers+text", text=c["city"], textposition="top center",
                               textfont=dict(family="Archivo, Arial", size=14, color=INK),
                               marker=dict(size=np.sqrt(c["postings"]) / np.sqrt(c["postings"]).max() * 46 + 6,
                                           color=YELLOW, line=dict(color=INK, width=1.5), opacity=1),
                               customdata=c["postings"], hovertemplate="%{text}: %{customdata} postings<extra></extra>"))
    fig.update_xaxes(visible=False, range=[5.6, 15.2])
    fig.update_yaxes(visible=False, range=[47.2, 55.2], scaleanchor="x", scaleratio=1 / np.cos(np.radians(51)))
    st.plotly_chart(style(fig, 640), width="stretch")
    total = len(p)
    c = c.assign(share=c["postings"] / total)
    ladder(c.rename(columns={"city": "skill"}).head(10), value=lambda r: f"{r.postings:,} postings ({pct(r.share)})")
    emp = p["employer"].value_counts().head(10)
    st.markdown("### Employers with the most postings")
    ladder(pd.DataFrame({"skill": emp.index, "share": emp.values / total, "n": emp.values}),
           value=lambda r: f"{r.n} postings")


SAMPLE_CV = """Maria Beispiel, maria.beispiel@example.com, +49 170 1234567

Data Analyst with 3 years of experience in e-commerce.
- Built Power BI dashboards and KPI reports for the sales team
- Wrote SQL queries and Python (pandas) scripts to automate reporting
- Ran A/B tests for the checkout and presented results to stakeholders
- Tools: Excel, Git, Google Analytics
Languages: English (fluent), German (B1)"""


def page_match():
    st.markdown('<p class="headline">How does my CV match the market?</p>', unsafe_allow_html=True)
    st.markdown('<p class="subline">Paste your CV or upload a PDF. Contact details are removed first; your CV is only '
                'sent to the AI if you allow it below.</p>', unsafe_allow_html=True)
    saved = storage.load_user()
    if saved.get("cv") and "cv_text" not in st.session_state:
        st.session_state["cv_text"] = saved["cv"]          # your remembered CV
    if st.button("Use a sample CV"):
        st.session_state["cv_text"] = SAMPLE_CV
    up = st.file_uploader("CV as PDF or text", type=["pdf", "txt", "md"], key="cv_file")
    if up is not None:
        st.session_state["cv_text"] = M.read_cv(up.getvalue(), up.name)
    cv = st.text_area("Your CV", key="cv_text", height=200, placeholder="Paste your CV here")
    c1, c2 = st.columns([3, 2])
    targets = c1.multiselect("Roles you're aiming for", FAMILIES,
                             default=[f for f in ("Data Analyst", "Product Analyst") if f in FAMILIES], key="cv_targets")
    use_ai = c2.toggle("Let the AI read my CV", value=False, key="cv_ai", disabled=config.MOCK_LLM,
                       help="Finds skills described in words, not only named tools. Needs an LLM key.")
    if not cv.strip():
        st.info("Paste a CV, upload one, or use the sample CV to see how matching works.")
        return
    r1, r2 = st.columns([1, 1])
    if r1.button("Remember my CV", help="Saved with contact details removed, so it's here next time"):
        storage.save_cv(cv)
        st.success("Saved. Your CV (without contact details) will be filled in next time.")
    if saved.get("cv") and r2.button("Forget my saved CV"):
        storage.save_cv(None)
        st.session_state.pop("cv_text", None)
        st.rerun()
    emb = embedder()
    E.ensure_embeddings(path, emb)
    have = M.cv_skills(cv, use_llm=use_ai, path=path)
    every = M.match(path, cv, have, families=targets or None, top=None, embedder=emb, allowed=ALLOWED)
    every = every.merge(P_ALL[["refnr", "employment", "work_model"]], on="refnr", how="left")
    m = every.head(10)
    g = M.gaps(path, have, targets or None, top=10, matches=every.head(50))
    target_txt = ", ".join(targets) if targets else "all roles"
    fit = M.fit_overview(every)
    st.markdown(f'<p class="headline">You cover 70% or more of the skills in {pct(fit["share"])} of the '
                f'{fit["postings"]:,} postings you\'re aiming for.</p>', unsafe_allow_html=True)
    st.markdown('<p class="key">Skills found in your CV:</p><div class="chips">' +
                "".join(f'<mark class="have">{esc(s)}</mark>' for s in have) + "</div>", unsafe_allow_html=True)
    if not g.empty:
        st.markdown("### Learn next")
        st.markdown(f'<p class="note">Skills you don\'t list yet, by how many {esc(target_txt)} postings ask for them.</p>',
                    unsafe_allow_html=True)
        ladder(g, kind="miss", value=lambda r: f"{pct(r.share)} of postings"
               + (f", makes {r.unlocks} of your top matches a strong fit" if r.unlocks else ""))
        with st.expander("A learning plan based on these gaps"):
            st.markdown(M.learning_plan(have, g, target_txt, path=path))
    if not m.empty:
        st.markdown("### Your best matches")
        st.markdown('<p class="key"><mark class="have">green</mark> you have it, '
                    '<mark class="miss">pink</mark> you don\'t (yet).</p>', unsafe_allow_html=True)
        for r in m.itertuples():
            ad_card(r, set(have), fit=f"{r.score * 100:.0f}% fit")
        top = every.head(100)
        sheet = pd.DataFrame({
            "fit": (top["score"] * 100).round().astype(int), "title": top["title"], "employer": top["employer"],
            "city": top["city"], "published": top["published"], "skills_you_have": top["have"].map(", ".join),
            "skills_missing": top["missing"].map(", ".join),
            "link": [ad_url(r) if s == "ba_api" else "" for r, s in zip(top["refnr"], top["source"])]})
        st.download_button("Download your top 100 matches (CSV)", sheet.to_csv(index=False).encode("utf-8-sig"),
                           file_name="my_matches.csv", mime="text/csv")
    if getattr(emb, "fallback_reason", None):
        st.markdown('<p class="note">Matching used the simple offline text model because the multilingual model '
                    'couldn\'t be downloaded. Connect to the internet and restart for better matches.</p>',
                    unsafe_allow_html=True)


def page_add_ad():
    """Ads found anywhere (LinkedIn, StepStone, XING, a company website): paste them in."""
    import hashlib
    from datetime import datetime as _dt

    from skillmarker import extract as X
    from skillmarker.roles import classify, employment_from_text, work_model_from_text

    st.markdown('<p class="headline">Add a job ad you found elsewhere</p>', unsafe_allow_html=True)
    st.markdown('<p class="subline">Paste an ad from LinkedIn, StepStone, XING or a company website. It is analysed like '
                'every other posting, compared with your CV, and saved to your live postings.</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    title = c1.text_input("Job title", key="ad_title")
    company = c2.text_input("Company", key="ad_company")
    city = c3.text_input("City, or Remote", key="ad_city")
    link = st.text_input("Link to the ad (optional)", key="ad_link")
    text = st.text_area("The ad text", height=220, key="ad_text", placeholder="Paste the full job description here")
    if not st.button("Add and analyse", type="primary"):
        return
    if not (title.strip() and text.strip()):
        st.error("Add at least the job title and the ad text.")
        return
    family = classify(title) or "Data Analyst"
    refnr = "manual:" + hashlib.sha1(f"{title}|{company}|{text[:500]}".encode()).hexdigest()[:12]
    from skillmarker.geo import locate
    found_city, lat, lon = locate(city)
    row = {c: None for c in db.POSTING_COLUMNS}
    row.update(refnr=refnr, family=family, title=title.strip(), employer=company.strip() or None,
               city=found_city or city.strip() or None, lat=lat, lon=lon, published=_dt.now().date(),
               external_url=link.strip() or None, description=text.strip(), source="manual",
               first_seen=_dt.now(), last_seen=_dt.now(), employment=employment_from_text(title, text[:2000]),
               work_model=work_model_from_text(city, text[:3000]))
    con = db.connect(config.LIVE_DB)
    db.upsert(con, "postings", pd.DataFrame([row]), ["refnr"])
    method = "rules" if config.MOCK_LLM else "llm"
    ex = X.rule_extract(title, text) if method == "rules" else X.llm_extract(title, text)[0]
    X.save_many(con, [(refnr, ex, method, X.content_hash(title, text, method))])
    try:
        storage.remember_ad(con, refnr)        # kept in your user data, survives restarts and daily updates
    except Exception as e:  # noqa: BLE001 - saving remotely must never lose the local result
        st.warning(f"Saved here, but not to cloud storage ({type(e).__name__}). It may be gone after a restart.")
    con.close()
    st.cache_data.clear()
    st.success(f"Saved as a {family} posting. It now appears in Match my CV when Live postings are selected.")
    cv = st.session_state.get("cv_text", "")
    have = set(M.cv_skills(cv)) if cv.strip() else None
    needed = set(ex.skills)
    if have is not None and needed:
        cover = len(needed & have) / len(needed)
        st.markdown(f'<p class="headline">You have {len(needed & have)} of the {len(needed)} skills this ad asks for '
                    f'({pct(cover)}).</p>', unsafe_allow_html=True)
    elif have is None:
        st.markdown('<p class="note">Paste your CV in Match my CV first to see which of these skills you have.</p>',
                    unsafe_allow_html=True)
    facts = [f"{s.replace('_', ' ')}" for s in [ex.seniority, ex.employment] if s and s != "unknown"]
    if ex.german_level != "none":
        facts.append(f"German: {ex.german_level}")
    st.markdown('<p class="note">' + esc(", ".join(facts) or "No level or contract details found.") + "</p>",
                unsafe_allow_html=True)
    st.markdown(f'<div class="ad"><h4>{esc(title)}</h4><div class="meta">{esc(company)}, {esc(city)}</div>'
                f'<div class="body">{marked(scrub(text)[:1500], have)}</div></div>', unsafe_allow_html=True)


def page_ask():
    st.markdown('<p class="headline">Ask the market</p>', unsafe_allow_html=True)
    st.markdown('<p class="subline">Questions are answered only from matching postings, with the postings cited, so you '
                'can check every claim.</p>', unsafe_allow_html=True)
    examples = ["Which AI engineer jobs ask for LangGraph or agents?", "Do BI developer jobs need SAP?",
                "Which data jobs don't require German?"]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, width="stretch"):
            st.session_state["ask_q"] = ex
    q = st.text_input("Your question", key="ask_q")
    fam = st.selectbox("Search in", ["All roles"] + FAMILIES, key="ask_family")
    if not q.strip():
        return
    emb = embedder()
    E.ensure_embeddings(path, emb)
    hits = ask.retrieve(path, q, k=6, family=None if fam == "All roles" else fam, embedder=emb, allowed=ALLOWED)
    st.markdown(f'<div class="answer">{esc(ask.answer(q, hits, path=path))}</div>', unsafe_allow_html=True)
    for i, r in enumerate(hits.itertuples(), 1):
        ad_card(r, None, fit=f"[{i}]")


def page_data():
    st.markdown('<p class="headline">Data and costs</p>', unsafe_allow_html=True)
    con = db.connect(path, read_only=True)
    methods = con.execute("SELECT method, count(*) n FROM extractions GROUP BY 1").df()
    calls = con.execute("""SELECT purpose, count(*) calls, sum(input_tokens) input_tokens, sum(output_tokens) output_tokens,
                                  sum(cost_usd) cost_usd FROM llm_calls GROUP BY 1""").df()
    log = con.execute("SELECT * FROM collection_log ORDER BY run_at DESC LIMIT 100").df()
    con.close()
    src = "generated sample postings with known answers" if source.startswith("Sample") else \
        "the Bundesagentur für Arbeit job search"
    st.markdown(f'<p class="subline">{len(P_ALL):,} postings from {src}. Extraction: ' +
                ", ".join(f"{r.n:,} by {'the LLM' if r.method == 'llm' else 'rules'}" for r in methods.itertuples()) +
                ".</p>", unsafe_allow_html=True)
    by_src = P_ALL.groupby("source_label").size().rename("postings").reset_index().rename(columns={"source_label": "source"})
    st.markdown("### Sources")
    st.dataframe(by_src, hide_index=True, width="content")
    st.markdown("### AI costs")
    if calls.empty:
        st.markdown('<p class="note">No LLM calls yet. With an OPENROUTER_API_KEY in .env, run '
                    '<code>python -m skillmarker.extract --max-cost 1</code>; every call\'s tokens and cost are logged here.</p>',
                    unsafe_allow_html=True)
    else:
        total = calls["cost_usd"].sum()
        n_llm = int(methods.loc[methods["method"] == "llm", "n"].sum()) if not methods.empty else 0
        st.markdown(f'<p class="subline">${total:.4f} in total' + (f", about ${total / n_llm * 1000:.2f} per 1,000 postings"
                    if n_llm else "") + ".</p>", unsafe_allow_html=True)
        st.dataframe(calls, hide_index=True, width="stretch")
    if source.startswith("Sample"):
        st.markdown("### How accurate is the extraction?")
        from skillmarker.evaluate import evaluate
        ev = evaluate(path, n=300).reset_index()
        st.dataframe(ev.rename(columns=lambda c: c.replace("_", " ").capitalize()), hide_index=True, width="stretch")
        st.markdown('<p class="note">Scored against the sample\'s known answers. Keyword rules never miss a listed tool, '
                    'but also count tools that are only mentioned (partners, perks, "no SAP needed"). Run '
                    '<code>python -m skillmarker.evaluate --llm</code> to measure the LLM on the same postings.</p>',
                    unsafe_allow_html=True)
    else:
        st.markdown("### Collection log")
        st.dataframe(log, hide_index=True, width="stretch")
    emb = E.get_embedder() if E._cached else None
    st.markdown(f'<p class="note">Text model for matching: {esc(emb.name if emb else config.EMBED_MODEL)}. '
                f'LLM: {esc("offline mode (no key)" if config.MOCK_LLM else config.LLM_MODEL)}.</p>', unsafe_allow_html=True)


{"Skills in demand": page_skills, "Trends": page_trends, "Roles and pay": page_roles, "Where the jobs are": page_where,
 "Match my CV": page_match, "Add a job ad": page_add_ad, "Ask the market": page_ask, "Data and costs": page_data}[page]()
