"""
discord_auth.py
===============

Discord OAuth2 gate for the Regime Terminal.

Flow
----
1. User clicks "Login with Discord" on the gated landing screen.
2. Discord prompts them to authorize, then redirects back to our app with a
   ``?code=...`` query parameter.
3. We exchange that code for an access token, then call Discord's
   ``/users/@me/guilds`` endpoint to check whether the user is a member of the
   configured guild (server). If yes, we grant access; if no, we show a
   "request access / join the server" screen.
4. The session is cached in ``st.session_state`` for the rest of the visit.

Configuration (via Streamlit secrets or environment variables)
--------------------------------------------------------------
Put these in ``.streamlit/secrets.toml`` (locally) or in the Streamlit Cloud
"Secrets" panel:

    DISCORD_CLIENT_ID = "your application id"
    DISCORD_CLIENT_SECRET = "your application secret"
    DISCORD_REDIRECT_URI = "https://your-app.streamlit.app"
    DISCORD_GUILD_ID = "the numeric id of your server"

If any of these are missing, ``require_login`` becomes a no-op (dev mode).
This is intentional so contributors can run the site locally without setting
up Discord credentials.

Discord Developer Portal setup (one time)
-----------------------------------------
1. Create an application at https://discord.com/developers/applications.
2. Under OAuth2 -> Redirects, add EXACTLY your app's URL (e.g.
   ``https://nefnofap-hmm.streamlit.app``).
3. Copy the Client ID and Client Secret into your secrets.
4. To get your guild ID: in Discord, enable Developer Mode (Settings ->
   Advanced), then right-click your server icon -> Copy Server ID.

Fixes vs original
-----------------
- 429 rate-limit: ``_CODE_USED`` session flag ensures ``exchange_code`` is
  called exactly ONCE per OAuth flow, never on every Streamlit rerun.
- Retry logic in ``exchange_code`` reads Discord's ``Retry-After`` header and
  waits before retrying (up to 3 attempts with exponential backoff).
- ``logout()`` now also clears the code-used flag so re-login works cleanly.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import requests
import streamlit as st

logger = logging.getLogger(__name__)

DISCORD_API   = "https://discord.com/api/v10"
AUTHORIZE_URL = "https://discord.com/api/oauth2/authorize"
TOKEN_URL     = "https://discord.com/api/oauth2/token"
SCOPES        = "identify guilds"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def _get(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read a config value from st.secrets first, then environment."""
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get(key, default)


def _config() -> Dict[str, Optional[str]]:
    return {
        "client_id":     _get("DISCORD_CLIENT_ID"),
        "client_secret": _get("DISCORD_CLIENT_SECRET"),
        "redirect_uri":  _get("DISCORD_REDIRECT_URI"),
        "guild_id":      _get("DISCORD_GUILD_ID"),
        "invite_url":    _get("DISCORD_INVITE_URL", "https://discord.gg/MSXdaexYdH"),
    }


def is_configured() -> bool:
    cfg = _config()
    return bool(
        cfg["client_id"] and cfg["client_secret"]
        and cfg["redirect_uri"] and cfg["guild_id"]
    )


# ─────────────────────────────────────────────────────────────────────────────
# OAuth helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_authorize_url() -> str:
    cfg = _config()
    params = {
        "client_id":     cfg["client_id"],
        "redirect_uri":  cfg["redirect_uri"],
        "response_type": "code",
        "scope":         SCOPES,
        "prompt":        "consent",
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def exchange_code(code: str) -> Dict[str, Any]:
    """
    Exchange an OAuth code for an access token.
    Retries up to 3 times, honouring Discord's Retry-After header on 429.
    Raises requests.HTTPError on final failure.
    """
    cfg = _config()
    data = {
        "client_id":     cfg["client_id"],
        "client_secret": cfg["client_secret"],
        "grant_type":    "authorization_code",
        "code":          code,
        "redirect_uri":  cfg["redirect_uri"],
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            r = requests.post(TOKEN_URL, data=data, headers=headers, timeout=15)

            if r.status_code == 429:
                # Respect Discord's rate-limit window
                retry_after = float(r.headers.get("Retry-After", 2 ** attempt))
                logger.warning("Discord 429 – waiting %.1f s (attempt %d/3)",
                               retry_after, attempt + 1)
                time.sleep(min(retry_after, 10))  # cap at 10 s
                continue

            r.raise_for_status()
            return r.json()

        except requests.HTTPError:
            raise   # non-429 HTTP errors are not retried
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(2 ** attempt)  # 1 s, 2 s backoff
            continue

    raise requests.RequestException(
        f"exchange_code failed after 3 attempts: {last_exc}"
    )


def fetch_user(token: str) -> Dict[str, Any]:
    r = requests.get(
        f"{DISCORD_API}/users/@me",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def fetch_guilds(token: str) -> list:
    r = requests.get(
        f"{DISCORD_API}/users/@me/guilds",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def is_member(token: str, guild_id: str) -> bool:
    return any(str(g.get("id")) == str(guild_id) for g in fetch_guilds(token))


# ─────────────────────────────────────────────────────────────────────────────
# Session state helpers
# ─────────────────────────────────────────────────────────────────────────────

SESSION_KEY = "_discord_user"
_CODE_USED  = "_discord_code_used"   # prevents re-exchange on reruns → fixes 429


def current_user() -> Optional[Dict[str, Any]]:
    return st.session_state.get(SESSION_KEY)


def logout() -> None:
    st.session_state.pop(SESSION_KEY, None)
    st.session_state.pop(_CODE_USED, None)   # allow fresh login after logout


def avatar_url(user: Dict[str, Any], size: int = 64) -> Optional[str]:
    if not user:
        return None
    if user.get("avatar"):
        return (
            f"https://cdn.discordapp.com/avatars/{user['id']}/"
            f"{user['avatar']}.png?size={size}"
        )
    return None


def display_name(user: Dict[str, Any]) -> str:
    return user.get("global_name") or user.get("username") or "user"


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit gate
# ─────────────────────────────────────────────────────────────────────────────

def require_login(
    render_login=None,
    render_denied=None,
) -> Optional[Dict[str, Any]]:
    """
    Streamlit gate. Returns the authenticated user dict (granted access), or
    renders the login/denied screens and stops the script.

    If Discord credentials are not configured, returns None and lets the app
    run unauthenticated (dev mode).

    FIX: The _CODE_USED session flag ensures exchange_code() is called exactly
    once per OAuth flow. Without this, Streamlit's rerun loop calls it on every
    interaction, hitting Discord's rate limit and causing 429 errors.

    Parameters
    ----------
    render_login  : callable(authorize_url, invite_url) – custom login UI.
    render_denied : callable(user, invite_url)          – custom denied UI.
    """
    if not is_configured():
        # Dev mode: no credentials set, skip gate entirely.
        return None

    # Already authenticated this session
    user = current_user()
    if user:
        return user

    cfg  = _config()
    code = st.query_params.get("code")
    if isinstance(code, list):
        code = code[0]

    if code and not st.session_state.get(_CODE_USED):
        # Set flag BEFORE the network call so that any rerun triggered
        # during the exchange (e.g. by Streamlit itself) doesn't retry.
        st.session_state[_CODE_USED] = True

        try:
            token_data = exchange_code(code)
            token      = token_data["access_token"]

            # Guild membership check (hard gate — must be in the server)
            if not is_member(token, cfg["guild_id"]):
                u = fetch_user(token)
                if render_denied:
                    render_denied(u, cfg["invite_url"])
                else:
                    _default_denied(u, cfg["invite_url"])
                st.stop()

            u = fetch_user(token)
            st.session_state[SESSION_KEY] = u
            # Clear ?code= from the URL so a page refresh doesn't re-exchange
            st.query_params.clear()
            st.rerun()

        except requests.HTTPError as exc:
            # Reset flag so the user can try logging in again
            st.session_state.pop(_CODE_USED, None)
            st.error(
                f"Discord login failed: {exc.response.status_code} "
                f"{exc.response.text[:200]}"
            )
        except Exception as exc:  # noqa: BLE001
            st.session_state.pop(_CODE_USED, None)
            st.error(f"Discord login failed: {exc}")

    # Not logged in → show login screen and stop the rest of the app
    if render_login:
        render_login(build_authorize_url(), cfg["invite_url"])
    else:
        _default_login(build_authorize_url(), cfg["invite_url"])
    st.stop()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Default login / denied UI (replaced by the app's themed versions)
# ─────────────────────────────────────────────────────────────────────────────

def _default_login(authorize_url: str, invite_url: str) -> None:
    st.markdown(
        f"### Login required\n"
        f"This terminal is gated to members of our Discord server.\n\n"
        f"[Login with Discord]({authorize_url})\n\n"
        f"Not a member yet? [Join here]({invite_url})."
    )


def _default_denied(user: Dict[str, Any], invite_url: str) -> None:
    name = display_name(user)
    st.error(
        f"Hi {name}, you're not a member of the required Discord server. "
        f"Join here: {invite_url}"
    )
