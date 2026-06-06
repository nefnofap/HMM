# Windows Setup Guide (Beginner-Friendly)

This guide walks you through running the HMM regime detector on **Windows 10/11**,
from zero to a working website in your browser. No prior Python experience needed.
Follow it top to bottom.

> Tip: Whenever you see a grey code box, you type that line and press **Enter**.

---

## Step 1 - Install Python

1. Go to <https://www.python.org/downloads/windows/>.
2. Click the big **"Download Python 3.12.x"** button (3.11 or 3.12 are both fine).
3. Run the downloaded installer.
4. **VERY IMPORTANT:** On the first installer screen, tick the box at the bottom
   that says **"Add python.exe to PATH"**. If you skip this, the commands below
   will not work.
5. Click **"Install Now"** and wait for it to finish, then click **Close**.

**Check it worked:** Press the **Windows key**, type `cmd`, and open
**Command Prompt**. Then type:

```
python --version
```

You should see something like `Python 3.12.4`. If you instead see an error or
the Microsoft Store opens, Python was not added to PATH - re-run the installer
and make sure that box is ticked.

---

## Step 2 - Install Git (to download the project)

1. Go to <https://git-scm.com/download/win> - the download starts automatically.
2. Run the installer. You can click **Next** through every screen (the defaults
   are fine).
3. Check it worked. In Command Prompt:

```
git --version
```

You should see something like `git version 2.45.0`.

---

## Step 3 - Download (clone) the project

We'll put it in your Documents folder. In Command Prompt, run these one at a time:

```
cd %USERPROFILE%\Documents
git clone https://github.com/nefnofap/hmm.git
cd hmm
```

- `cd %USERPROFILE%\Documents` moves you into your Documents folder.
- `git clone ...` downloads the project into a new folder called `hmm`.
- `cd hmm` moves you inside that project folder.

> If the project is on a branch (not `main`) and the clone looks empty, run:
> `git checkout init-hmm-toolkit`

---

## Step 4 - Create a virtual environment

A "virtual environment" is a private box for this project's libraries so they
don't clash with anything else on your PC. Create and activate it:

```
python -m venv venv
venv\Scripts\activate
```

After the second command your prompt line will start with **`(venv)`**. That
means the box is active. **You must run this `activate` line every time you open
a new Command Prompt to work on this project.**

> If you get a red error about "running scripts is disabled", that only happens
> in **PowerShell**. The simplest fix is to use **Command Prompt** (cmd) instead
> of PowerShell. To open it: Windows key -> type `cmd` -> Enter.

---

## Step 5 - Install the project's libraries

With `(venv)` showing, run:

```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This downloads numpy, pandas, hmmlearn, scikit-learn, yfinance, matplotlib, and
Streamlit. It may take a couple of minutes. A few yellow warning lines are
normal - only red **ERROR** lines matter.

---

## Step 6 - First test (no website yet)

Let's confirm the math works. Run:

```
python regime_detection.py --ticker BTC-USD
```

You should see it download Bitcoin data, print a **regime summary table**, and
save two files into the folder: `regime_plot.png` (a chart) and
`btc_regime_model.json` (the trained model). Open `regime_plot.png` by
double-clicking it in File Explorer.

> No internet or it fails to download? The script automatically falls back to
> built-in practice data so it still runs.

---

## Step 7 - Launch the website

This is the fun part. Run:

```
streamlit run streamlit_app.py
```

Your web browser opens automatically at **http://localhost:8501** showing the
dashboard. If it doesn't open, copy that address into your browser manually.

On the page:
1. In the left sidebar, set the **Ticker symbol** (e.g. `BTC-USD`, `ETH-USD`,
   `AAPL`).
2. Choose the number of regimes and history.
3. Click **Detect regimes**.

You'll see the price chart colour-coded by regime, a summary table, and a
short forecast.

**To stop the website:** go back to the Command Prompt window and press
**Ctrl + C**.

---

## Daily routine (after the first setup)

Every time you want to use it again, you only need:

```
cd %USERPROFILE%\Documents\hmm
venv\Scripts\activate
streamlit run streamlit_app.py
```

You do **not** repeat Steps 1-5 - those are one-time.

---

## Troubleshooting

| Problem | Fix |
|--------|-----|
| `'python' is not recognized` | Python wasn't added to PATH. Re-run the Python installer and tick **"Add python.exe to PATH"**, then reopen Command Prompt. |
| `'git' is not recognized` | Reinstall Git (Step 2), then reopen Command Prompt. |
| PowerShell error: *"running scripts is disabled"* | Use **Command Prompt (cmd)** instead of PowerShell. |
| `(venv)` is not showing | You didn't activate it. Run `venv\Scripts\activate` again. |
| `pip install` shows red ERRORs | Make sure `(venv)` is active and run `python -m pip install --upgrade pip` first, then retry. |
| Browser didn't open for Streamlit | Manually visit <http://localhost:8501>. |
| Streamlit page says download failed | It will use practice data automatically; or check your internet connection. |

---

## Putting the website online (optional, later)

Right now the site only runs on your own PC (`localhost`). When you're ready to
share it with a public link, the easiest free option is **Streamlit Community
Cloud**:

1. Push your project to your GitHub repo (`nefnofap/hmm`).
2. Go to <https://share.streamlit.io>, sign in with GitHub.
3. Pick the repo and `streamlit_app.py` as the main file, then **Deploy**.

It gives you a public URL anyone can open. (For a heavier React + FastAPI setup,
ask and I can scaffold that instead.)

> Reminder: this tool is for research and learning, not financial advice. Never
> connect real trading funds without your own risk controls.
