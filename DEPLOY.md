# Deploy Your Site for Free (so your community can use it)

Goal: a public web link anyone can open in a browser — **no Python, no install**
on their side. The easiest free option is **Streamlit Community Cloud**.

Your repo is already deploy-ready:
- `streamlit_app.py` — the app
- `requirements.txt` — the libraries it needs
- `.streamlit/config.toml` — locks in the dark theme on the hosted site
- `runtime.txt` — pins Python 3.13 (so `hmmlearn` installs from a prebuilt wheel)

---

## Option A — Streamlit Community Cloud (recommended, ~10 min)

1. Push the latest code to GitHub (already done at `nefnofap/hmm`, branch `main`).
2. Go to **https://share.streamlit.io** and sign in with your GitHub account.
3. Click **"Create app"** → **"Deploy a public app from GitHub"**.
4. Fill in:
   - **Repository:** `nefnofap/hmm`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
5. Click **Deploy**. First build takes a few minutes (it installs the libraries).
6. You get a public URL like `https://nefnofap-hmm.streamlit.app` — **share that
   link** with your community. Done.

### Updating the live site
Every time you `git push` to `main`, the site **auto-redeploys**. No extra steps.

### Free-tier notes
- Free apps may "sleep" after inactivity; the first visitor wakes it in ~30s.
- Resources are modest — fine for this app. To keep it snappy for many users,
  pre-train models offline (`python regime_detection.py --ticker BTC`) and commit
  the resulting `*_regime_model.json` so the app *loads* them instead of fitting
  on every visit.

---

## Option B — Hugging Face Spaces (also free)

1. Create an account at **https://huggingface.co**.
2. **New → Space** → SDK: **Streamlit** → name it (e.g. `regime-terminal`).
3. Point it at this repo (or push the files to the Space's git).
4. It builds and serves at `https://huggingface.co/spaces/<you>/regime-terminal`.

Good if you expect a larger audience or want a community "Files" tab.

---

## Option C — Your own server with Docker (full control, ~free on small tiers)

Add a `Dockerfile` (one-time):

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
```

Then deploy the container free/cheap on **Render**, **Railway**, or **Fly.io**.
These give you a custom domain option and no sleeping, but require a bit more
setup than Option A.

---

## Custom domain (optional, makes it look pro)

If you own a domain (e.g. from Namecheap/Cloudflare), you can point a subdomain
like `regime.yoursite.com` at a Render/Fly deployment (Option C). Streamlit
Community Cloud uses its own `*.streamlit.app` subdomain on the free tier.

---

## Quick comparison

| Option | Cost | Setup | Custom domain | Best for |
|--------|------|-------|---------------|----------|
| Streamlit Cloud | Free | Easiest | No (free tier) | Getting live fast ⭐ |
| HF Spaces | Free | Easy | No | Bigger audience |
| Docker + Render/Fly | Free tier | Moderate | Yes | Full control / pro look |

---

## Make it community-friendly (drives adoption)
- **Preset asset buttons** (BTC/ETH/Gold) so newcomers click, not type.
- A visible **"research only, not financial advice"** banner (already in the UI).
- A **Discord/Telegram alert bot** that posts "BTC flipped to BULL, 7/8 confirmed"
  — the single stickiest feature for bringing people back.
- A `LICENSE` (MIT is common) and `CONTRIBUTING.md` so others can contribute.

> Reminder: this is research/educational software. Markets carry risk of loss.
> Nothing here is financial advice.
