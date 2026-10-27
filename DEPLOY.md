# Deploying the API + Dashboard

This deploys **one service** (the FastAPI backend, which also serves the
dashboard at its root URL) — not two separate frontend/backend deployments.
That keeps this simple and low-risk: one build, one URL, one thing that can
go wrong instead of two.

## Before you deploy: test it locally first

```bash
pip install -r requirements-api.txt
uvicorn src.api.main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser. You should see the
dashboard, your browser should ask for location permission, and (if you
allow it) it should show live weather for wherever you are. Try searching
"Kodagu" too — that should additionally show the flood-risk card.

**Only deploy once this works locally.** If something's wrong, it's much
faster to find and fix on your own machine than after a deploy.

## Deploying to Render (free, no credit card)

1. Push this repo to GitHub (already done — `main` branch).
2. Go to **render.com**, sign up / log in (GitHub login is easiest).
3. Click **New +** → **Blueprint**.
4. Connect your GitHub account if asked, then select the
   `Flood-Monitoring-Sentinel2-GEE` repository.
5. Render will detect `render.yaml` in this repo automatically and show you
   the `kodagu-flood-risk` service it's about to create. Click **Apply**.
6. Wait for the build to finish (the first build trains both models at
   startup, which takes a little longer than a normal web app — give it a
   few minutes, don't panic if it's not instant).
7. Once it says **Live**, Render gives you a URL like
   `https://kodagu-flood-risk.onrender.com` — that's your public,
   internet-reachable dashboard and API.

### If the build fails

Render shows build logs right there in the dashboard — read the last ~20
lines of red text, that's almost always the actual error. Common ones:

- **Out of memory during build**: the free tier has limited RAM. This
  shouldn't happen with `requirements-api.txt` (deliberately excludes
  TensorFlow/matplotlib, which aren't needed to serve the API), but if it
  does, it's worth checking you're using `requirements-api.txt` in the
  build command, not the full `requirements.txt`.
- **Port binding error**: make sure the start command uses `$PORT` (Render
  sets this automatically) — `render.yaml` already does this correctly.
- **Module not found**: a dependency is missing from `requirements-api.txt`
  — add it and push again, Render auto-redeploys on every push to `main`.

### Free tier behavior to know about

Render's free web services **sleep after 15 minutes of no traffic**, and
the next request takes ~30-60 seconds to wake it back up (it has to retrain
both models from scratch on wake). This is completely normal for a free
tier, not a bug — just something to mention if your teacher sees a slow
first load. If you need it always-instant for the review, open the URL
yourself 1-2 minutes before you present, so it's already awake.

## Testing the deployed version

Once it's live:
```bash
curl https://<your-app>.onrender.com/health
curl https://<your-app>.onrender.com/predict/historical/2023-07-05
```
Both should return the same JSON you saw testing locally. Then open the URL
in a browser to check the dashboard renders and the location prompt works.

**Report back here with the live URL (or any error) once you've tried it —
I can't deploy this myself (it needs your own hosting account), but I can
debug anything that goes wrong from the logs/errors you paste.**
