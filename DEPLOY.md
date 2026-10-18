# Use Skillmarker on your phone (free, laptop off)

After this setup, Skillmarker runs in the cloud, collects new postings every morning, keeps your
saved CV and pasted ads, and opens from your phone's home screen like an app.

| Part | Free service | Role |
|---|---|---|
| App | Streamlit Community Cloud | Runs the app at a web address |
| Data | Hugging Face (private dataset) | Stores the postings database and your saved data |
| Daily collection | GitHub Actions | Collects, extracts and uploads every morning |

Your code repository stays public (good for your portfolio); your data stays in a **private**
dataset, and the app is protected by a **password**.

Setup takes about 30 minutes. Do the steps in order.

---

## Step 1: Create the private data storage (Hugging Face)

1. Create a free account at **huggingface.co**.
2. Create an access token: profile picture → **Settings → Access Tokens → Create new token**.
   Choose **Write**, name it `skillmarker`, and copy it (it starts with `hf_`). Treat it like a password.
3. Pick a name for your private dataset, for example `MinalBorse/skillmarker-data`. You don't need to
   create it; the next step does that.

## Step 2: Upload your current data from your laptop (once)

In `C:\Sprin2Project\skillmarker_fresh\skillmarker`, add these two lines to your `.env`:

```
HF_TOKEN=hf_your_token_here
HF_DATASET_REPO=MinalBorse/skillmarker-data
```

Then upload the postings you've already collected:

```powershell
uv run python -m skillmarker.storage --push
uv run python -m skillmarker.storage --status
```

`--status` should list `live.duckdb`. On huggingface.co, your new dataset shows a lock icon: it's private.

Also run the career-page discovery once and push the result, so the daily job doesn't have to:

```powershell
uv run python -m skillmarker.careers --discover
git add companies.json
git commit -m "Discovered career-page systems"
git push
```

## Step 3: Set up the daily collection (GitHub Actions)

1. Open **github.com/MinalBorse/skillmarker → Settings → Secrets and variables → Actions**.
2. Click **New repository secret** for each of these:

   | Name | Value |
   |---|---|
   | `HF_TOKEN` | your Hugging Face token |
   | `HF_DATASET_REPO` | `MinalBorse/skillmarker-data` |
   | `OPENROUTER_API_KEY` | your OpenRouter key |
   | `ADZUNA_APP_ID`, `ADZUNA_APP_KEY` | optional |

3. Open the **Actions** tab → **Daily collection** → **Run workflow**. The first run takes 15–60 minutes.
   Green means done; the database on Hugging Face is now updated.

From now on it runs every morning at about 06:00–07:00 German time. LLM extraction is capped at
**$0.30 per day** (`--max-cost` in `.github/workflows/collect.yml`); change it there if you like.

## Step 4: Host the app (Streamlit Community Cloud)

1. Go to **share.streamlit.io** and sign in with GitHub.
2. **Create app → Deploy a public app from GitHub**:
   - Repository: `MinalBorse/skillmarker`
   - Branch: `main`
   - Main file: `app.py`
   - App URL: choose a name, e.g. `skillmarker-minal`
3. Open **Advanced settings**:
   - Python version: **3.12**
   - Secrets: paste the following, with your values:

   ```toml
   HF_TOKEN = "hf_your_token_here"
   HF_DATASET_REPO = "MinalBorse/skillmarker-data"
   OPENROUTER_API_KEY = "sk-or-v1-your-key"
   APP_PASSWORD = "choose-a-strong-password"
   EMBED_BACKEND = "auto"
   ```

4. Click **Deploy**. The first start takes a few minutes (installing packages).
5. Open the app, enter your password, and choose **Live postings** in the sidebar.

**If the app restarts or crashes when you open Match my CV,** the free server ran out of memory
loading the text model. In the app's **Settings → Secrets**, change `EMBED_BACKEND = "hash"`.
Matching then uses the lighter offline model; everything else is unchanged.

## Step 5: Add it to your phone's home screen

Open your app's address in the phone's browser and log in once.

- **iPhone (Safari):** Share button → **Add to Home Screen**
- **Android (Chrome):** ⋮ menu → **Add to Home screen**

It now opens from an icon like an app.

---

## How your data is kept

| Data | Where | Kept? |
|---|---|---|
| Collected postings, extractions, embeddings | `live.duckdb` in your private dataset | Yes, updated every morning |
| Your CV | Only if you tap **Remember my CV**: stored in `user.json` with contact details removed | Until you tap **Forget my saved CV** |
| Job ads you paste | `user.json` in your private dataset | Yes |

The collector only writes `live.duckdb`, and the app only writes `user.json`, so a morning update
can't overwrite something you saved, and saving can't overwrite the morning update. Hugging Face keeps
every version, so you can roll back from the dataset's **Files → History** page.

## Good to know

- **Sleeping:** free Streamlit apps sleep after a while without visits; the first visit then takes about
  a minute to wake up.
- **Fresh data:** the app fetches new data at most every 6 hours; tap **Refresh data** in the sidebar
  to fetch it now.
- **Costs:** everything here is free. The only cost is LLM extraction, capped per day.
- **Free-tier limits change.** If a service changes its free plan, check its pricing page.
- **Rotate keys** if you ever share your screen with secrets visible: create a new token, update the
  secret, delete the old one.
