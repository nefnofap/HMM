# Discord Login Setup &mdash; In-Depth Tutorial

This walks you through, click-by-click, how to gate the Regime Terminal so
only members of your Discord server can use the site. Estimated time: **15
minutes**. No coding required &mdash; only configuration.

> **Important:** Discord OAuth requires a **public URL** for the redirect.
> The login flow does **not** work on `http://localhost:8501` unless you
> also register that exact URL as a redirect (see Step 5 below). The
> easiest path is to deploy first to Streamlit Community Cloud (free),
> then plug the credentials in.

---

## Big picture (the flow)

```
[User] -> clicks "Login with Discord"
       -> Discord asks them to authorize
       -> Discord redirects back to your site with ?code=...
       -> Your site swaps the code for an access token
       -> Your site asks Discord "what servers is this user in?"
       -> If your server is in the list -> ACCESS GRANTED
       -> Otherwise -> "Join the server" screen
```

You will need **four values**, all configured as Streamlit secrets:

| Key | What it is | Where it comes from |
|---|---|---|
| `DISCORD_CLIENT_ID` | Public ID of your Discord app | Developer Portal |
| `DISCORD_CLIENT_SECRET` | Private secret of your Discord app | Developer Portal |
| `DISCORD_REDIRECT_URI` | The URL Discord sends users back to | Your hosted Streamlit URL |
| `DISCORD_GUILD_ID` | The numeric ID of your server | Right-click your server in Discord |

---

## Step 1 - Deploy your site to get a public URL (5 min)

Skip this step only if you already have a public URL.

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Click **Create app -> Deploy a public app from GitHub**.
3. Pick:
   - **Repository:** `nefnofap/hmm`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
4. Click **Deploy**. After a few minutes you will get a URL like
   `https://nefnofap-hmm.streamlit.app`. **Copy this URL exactly** &mdash;
   you'll need it twice.

> The site will load **without** the gate at this point because no Discord
> secrets are configured yet. That's expected.

---

## Step 2 - Create a Discord application (3 min)

1. Go to <https://discord.com/developers/applications>.
2. Click **New Application** in the top right.
3. Name it something like **"Regime Terminal"**, accept the terms, click **Create**.
4. You're now on the app's **General Information** page.
   - Optional but recommended: upload an icon (drag a PNG into the App Icon area).
5. From the left sidebar, click **OAuth2**.
   - Copy the **Client ID** &mdash; this becomes `DISCORD_CLIENT_ID`.
   - Click **Reset Secret** (or the existing **Copy** button) to get the
     **Client Secret** &mdash; this becomes `DISCORD_CLIENT_SECRET`.
   - **Save both somewhere safe.** The secret is only shown once.

---

## Step 3 - Add the redirect URL (2 min)

This is the single most common reason logins fail. Get it exactly right.

1. Still inside **OAuth2** in the Developer Portal.
2. Under **Redirects**, click **Add Redirect**.
3. Paste the **exact** URL from Step 1, with **no trailing slash and no path**:

   ```
   https://nefnofap-hmm.streamlit.app
   ```

   - Wrong: `https://nefnofap-hmm.streamlit.app/`  (trailing slash)
   - Wrong: `https://nefnofap-hmm.streamlit.app/login`  (extra path)
   - Right: `https://nefnofap-hmm.streamlit.app`

4. Click **Save Changes** at the bottom.

> If you also want to test locally, add a second redirect for
> `http://localhost:8501` so you can sign in during development.

---

## Step 4 - Get your server ID (2 min)

1. Open Discord (desktop or web).
2. Open **User Settings** (gear icon, bottom-left next to your name).
3. Go to **Advanced** and turn on **Developer Mode**.
4. Close settings. **Right-click your server's icon** in the left sidebar.
5. Click **Copy Server ID** at the bottom of the menu.
6. Paste it somewhere &mdash; this becomes `DISCORD_GUILD_ID`. It will look like
   a long number, e.g. `1283902749384857345`.

---

## Step 5 - Add the secrets to Streamlit (3 min)

1. Go to your app at <https://share.streamlit.io> and click the **...**
   menu next to it -> **Settings -> Secrets**.
2. Paste in your four values, exactly like this (TOML syntax, **values must
   be in quotes**):

   ```toml
   DISCORD_CLIENT_ID = "1234567890123456789"
   DISCORD_CLIENT_SECRET = "abc123_long_secret_string_xyz"
   DISCORD_REDIRECT_URI = "https://nefnofap-hmm.streamlit.app"
   DISCORD_GUILD_ID = "1283902749384857345"

   # Optional: defaults to your existing invite if omitted
   # DISCORD_INVITE_URL = "https://discord.gg/MSXdaexYdH"
   ```

3. Click **Save**. Streamlit will automatically restart your app.
4. Visit your URL again. You should now see the **"Login with Discord"**
   landing screen instead of the dashboard.

---

## Step 6 - Test it end-to-end (1 min)

1. Open your site in a private/incognito window.
2. Click **Login with Discord**.
3. Discord shows the consent screen with the permissions
   (`identify` and `guilds`). Click **Authorize**.
4. You're redirected back to your site:
   - If you're in the server &rarr; **the dashboard loads.**
   - If you're not in the server &rarr; you get the **"Join the Discord"** screen.

To test the denied path, log in as a friend who isn't in your server (or
temporarily leave the server, retry, then rejoin).

---

## Local development

For testing on your own PC at `http://localhost:8501`:

1. Add `http://localhost:8501` as a **second** redirect URL in the Discord
   Developer Portal (Step 3).
2. Locally, **copy** `.streamlit/secrets.toml.example` to
   `.streamlit/secrets.toml` (note: that file is gitignored, never commit it).
3. Set `DISCORD_REDIRECT_URI = "http://localhost:8501"` for the local copy.
4. Run `streamlit run streamlit_app.py` and log in as usual.

> If you don't add any secrets at all, the gate is automatically **disabled**
> and the site runs free &mdash; perfect for hacking on the code.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Invalid OAuth2 redirect_uri` after consent | Your `DISCORD_REDIRECT_URI` doesn't **exactly** match the one in the Developer Portal. Check trailing slashes, http vs https, port numbers. |
| Login keeps looping back to the login screen | Streamlit may be running on a different port locally. Check the URL in your browser, register that exact URL. |
| `401 Unauthorized` from Discord | Your **Client Secret** is wrong or was rotated. Reset it in the portal and update your secrets. |
| "You're not a member" but you ARE in the server | Discord caches guild lists briefly. Click **Login** again, or remove + re-add the app authorization (your server icon -> Authorized Apps). |
| Site loads with no gate | Secrets aren't being read. Confirm all four keys are present and saved in **Settings -> Secrets**, then reboot the app. |

---

## Security notes
- The Client Secret is **private**. Never commit it. Never paste it in chat.
- This module asks Discord for the minimum scopes (`identify` and `guilds`).
  We never see passwords, DMs, or message content.
- If you suspect a leak, reset the secret in the Developer Portal &mdash; old
  values immediately stop working.
- The session lives only in Streamlit's in-memory `session_state` and
  expires when the user closes the tab.

---

> Reminder: this is research/educational software. Markets carry risk of
> loss. Nothing here is financial advice. The Discord gate is for community
> access control, not for compliance with any specific regulation.
