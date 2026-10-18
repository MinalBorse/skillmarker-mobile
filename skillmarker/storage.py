"""Keep data in a private Hugging Face dataset repo, so the hosted app keeps it between restarts.

    python -m skillmarker.storage --push     # upload your local live.duckdb and user data (first setup)
    python -m skillmarker.storage --pull     # download them
    python -m skillmarker.storage --status

Why: free hosting (Streamlit Community Cloud) has a temporary disk that is wiped on every
restart. A private Hugging Face dataset repo is free, versioned file storage.

Two files, each with one owner, so they can never overwrite each other's changes:
- live.duckdb: written only by the daily collector (GitHub Actions); the app reads a copy.
- user.json: written only by the app: your saved CV (contact details removed) and the job
  ads you pasted, with their extraction. Applied on top of the downloaded database.

Set HF_TOKEN (a write token) and HF_DATASET_REPO (e.g. "your-name/skillmarker-data").
Without them, everything stays local, as before.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config, db

LIVE_NAME, USER_NAME = "live.duckdb", "user.json"
USER_FILE = config.DATA_DIR / USER_NAME


def remote_enabled() -> bool:
    return bool(os.getenv("HF_TOKEN") and os.getenv("HF_DATASET_REPO"))


def _api():
    from huggingface_hub import HfApi

    return HfApi(token=os.getenv("HF_TOKEN"))


def ensure_repo():
    _api().create_repo(os.getenv("HF_DATASET_REPO"), repo_type="dataset", private=True, exist_ok=True)


def download(name: str) -> Path | None:
    """Download one file into the data folder; None if it doesn't exist remotely yet."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError

    try:
        p = hf_hub_download(os.getenv("HF_DATASET_REPO"), name, repo_type="dataset", token=os.getenv("HF_TOKEN"),
                            local_dir=config.DATA_DIR, force_download=True)
        return Path(p)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    except HfHubHTTPError as e:            # some versions report a missing file as a plain 404
        if getattr(getattr(e, "response", None), "status_code", None) == 404:
            return None
        raise


def upload(path: Path, name: str, message: str):
    _api().upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=os.getenv("HF_DATASET_REPO"),
                       repo_type="dataset", commit_message=message)


# ---- User data (owned by the app) -------------------------------------------------------------
def load_user() -> dict:
    if USER_FILE.exists():
        try:
            return json.loads(USER_FILE.read_text(encoding="utf-8"))
        except ValueError:
            pass
    return {"cv": None, "cv_saved_at": None, "ads": []}


def save_user(data: dict, message: str = "Update user data"):
    USER_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if remote_enabled():
        upload(USER_FILE, USER_NAME, message)


def save_cv(text: str | None):
    """Remember the CV (contact details removed first), or forget it with None."""
    from .privacy import scrub

    data = load_user()
    data["cv"] = scrub(text) if text else None
    data["cv_saved_at"] = datetime.now().isoformat(timespec="seconds") if text else None
    save_user(data, "Save CV" if text else "Forget CV")


def remember_ad(con, refnr: str):
    """Copy a pasted ad (posting, extraction, skills) into user data, so it survives restarts."""
    data = load_user()
    rec = {t: con.execute(f"SELECT * FROM {t} WHERE refnr = ?", [refnr]).df().to_dict(orient="records")
           for t in ("postings", "extractions", "posting_skills")}
    data["ads"] = [a for a in data["ads"] if a.get("refnr") != refnr] + [{"refnr": refnr, **rec}]
    save_user(data, f"Save job ad {refnr}")


def apply_user_ads(path) -> int:
    """Put the pasted ads back into a (freshly downloaded) database."""
    ads = load_user().get("ads") or []
    if not ads:
        return 0
    con = db.connect(path)
    keys = {"postings": ["refnr"], "extractions": ["refnr"], "posting_skills": ["refnr", "skill"]}
    for ad in ads:
        for table, key in keys.items():
            rows = pd.DataFrame(ad.get(table) or [])
            if not rows.empty:
                for c in rows.columns:
                    if c.endswith(("_seen", "_at")) or c in ("published",):
                        rows[c] = pd.to_datetime(rows[c], errors="coerce")
                if "published" in rows:
                    rows["published"] = rows["published"].dt.date
                db.upsert(con, table, rows, key)
    con.close()
    return len(ads)


# ---- Whole-database sync (used by the app on start and by the collector) -----------------------
def pull() -> dict:
    out = {"live": False, "user": False}
    if not remote_enabled():
        return out
    out["live"] = download(LIVE_NAME) is not None
    out["user"] = download(USER_NAME) is not None
    if out["live"]:
        apply_user_ads(config.LIVE_DB)
    return out


def push_live(message: str = "Daily collection"):
    if remote_enabled() and config.LIVE_DB.exists():
        ensure_repo()
        upload(config.LIVE_DB, LIVE_NAME, message)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if not remote_enabled():
        print("Set HF_TOKEN and HF_DATASET_REPO (in .env, or as secrets when hosted) first.")
        return 1
    if args.push:
        ensure_repo()
        push_live("Initial upload")
        if USER_FILE.exists():
            upload(USER_FILE, USER_NAME, "Initial upload")
        print(f"Uploaded to the private dataset {os.getenv('HF_DATASET_REPO')}")
    elif args.pull:
        print(pull())
    else:
        files = _api().list_repo_files(os.getenv("HF_DATASET_REPO"), repo_type="dataset")
        print(f"{os.getenv('HF_DATASET_REPO')}: {files}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
