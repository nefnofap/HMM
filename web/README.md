# Regime — HMM market-state dashboard (web)

A Discord-gated Next.js dashboard that renders the output of the HMM regime
model in this repo. It does **not** run Python at request time. A GitHub Action
runs the model on a schedule, writes `web/data/regime.json`, and this app reads
that file.

```
        ┌─────────────── GitHub Action (cron) ─────────────┐
        │  scripts/export_regime.py  →  web/data/regime.json │
        └───────────────────────┬─────────────────────┘
                                │ commit
                                ▼
                  Vercel redeploy  →  Next.js dashboard
                         (Discord login + guild gate)
```

## Why precompute instead of running Python on Vercel
`hmmlearn` + `scipy` + `scikit-learn` are too heavy for a Vercel serverless
function, and Streamlit can't run there at all. Regimes are only locally
stationary, so a periodic refresh is the right fidelity for a dashboard.

## Local dev
```bash
cd web
cp .env.example .env.local   # fill in the Discord + AUTH values
npm install
npm run dev
```
With `DISCORD_GUILD_ID` blank, auth fails OPEN so you can develop without a
configured server.

## Deploy on Vercel
1. Import this repo. **Set the project Root Directory to `web/`.**
2. Add env vars from `.env.example` (use a real `AUTH_SECRET`, set `AUTH_URL`
   to your deployed URL).
3. In the Discord developer portal, add the OAuth redirect
   `https://<your-domain>/api/auth/callback/discord`.
4. Deploy. The login page is `/`; the dashboard is `/dashboard`.

## Data freshness
The dashboard shows whatever is in `web/data/regime.json` as of the last commit
+ deploy. The included file is a synthetic seed so the app builds out of the
box; the Action overwrites it with real BTC regimes on first run.

To refresh without redeploys later, move the JSON to Vercel Blob / KV and fetch
it in `app/dashboard/page.tsx` with `revalidate`.
