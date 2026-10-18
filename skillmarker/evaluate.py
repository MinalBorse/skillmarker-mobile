"""Score the extractors against the sample data's known answers.

    python -m skillmarker.evaluate                    # rules (free)
    python -m skillmarker.evaluate --llm --n 150      # rules vs LLM on 150 postings (a few cents)

Why: "the LLM extracts skills" is a claim; precision and recall against known answers make
it a measurement. On the sample data, keyword rules never miss a listed tool (high recall)
but count tools that are only mentioned (partner companies, training perks, "no SAP
needed"), which lowers precision. The LLM's job is to fix exactly that.
Writes docs/EVAL.md.
"""
import argparse
import json

import pandas as pd

from . import config, db
from .extract import llm_extract, rule_extract


def score(pred: list[list[str]], truth: list[list[str]]) -> dict:
    tp = fp = fn = 0
    for p, t in zip(pred, truth):
        p, t = set(p), set(t)
        tp += len(p & t)
        fp += len(p - t)
        fn += len(t - p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate(path=None, n: int = 300, with_llm: bool = False, seed: int = 0) -> pd.DataFrame:
    path = path or config.SAMPLE_DB
    con = db.connect(path, read_only=True)
    df = con.execute("SELECT p.refnr, p.title, p.description, t.* FROM postings p JOIN truth t USING (refnr)").df()
    con.close()
    if df.empty:
        raise SystemExit("No sample data with known answers. Run: python -m skillmarker.sample")
    df = df.sample(min(n, len(df)), random_state=seed)
    truth = [json.loads(s) for s in df["skills"]]
    results = {}
    methods = [("rules", lambda r: rule_extract(r.title, r.description))]
    if with_llm:
        if config.MOCK_LLM:
            raise SystemExit("--llm needs OPENROUTER_API_KEY in .env")
        methods.append(("llm", lambda r: llm_extract(r.title, r.description)[0]))
    rows = []
    for name, fn in methods:
        preds = [fn(r) for r in df.itertuples()]
        s = score([p.skills for p in preds], truth)
        sal = df["salary_min"].notna().values
        s.update({
            "method": name, "postings": len(df),
            "seniority_accuracy": sum(p.seniority == t for p, t in zip(preds, df["seniority"])) / len(df),
            "german_accuracy": sum(p.german_level == t for p, t in zip(preds, df["german_level"])) / len(df),
            "remote_accuracy": sum(p.remote == t for p, t in zip(preds, df["remote"])) / len(df),
            "salary_exact": (sum(p.salary_min == t for p, t, has in zip(preds, df["salary_min"], sal) if has) / max(sal.sum(), 1)),
        })
        rows.append(s)
        results[name] = preds
    return pd.DataFrame(rows).set_index("method")


def report(table: pd.DataFrame) -> str:
    pct = table.copy()
    for c in pct.columns:
        if c != "postings":
            pct[c] = (pct[c] * 100).round(1).astype(str) + "%"
    return ("# Extraction evaluation\n\nScored against the known answers of the sample postings.\n\n"
            + pct.reset_index().to_markdown(index=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--llm", action="store_true", help="also evaluate the LLM (costs a few cents)")
    args = ap.parse_args()
    table = evaluate(n=args.n, with_llm=args.llm)
    text = report(table)
    out = config.ROOT / "docs" / "EVAL.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
