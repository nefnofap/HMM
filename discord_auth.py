"""
discord_auth.py
===============

Discord OAuth2 gate for the Regime Terminal.

Flow
----
1. User clicks "Login with Discord" on the gated landing screen.
2. Discord redirects back with a ``?code=...`` query parameter.
3. We exchange that code for an access token, check guild membership,
   and cache the session in ``st.session_state``.

Hardcoded config (no secrets needed):
  Client ID     : 1513610614549905588
  Guild ID      : 1478370205523775699
  Premium Role  : 1497947608616931428
  Invite URL    : https://discord.com/invite/MSXdaexYdH

DISCORD_CLIENT_SECRET and DISCORD_REDIRECT_URI still come from
Streamlit secrets (never hardcode the secret or redirect URI).
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

# ── Hardcoded public values ───────────────────────────────────────────────────
_CLIENT_ID   = "1513610614549905588"
_GUILD_ID    = "1478370205523775699"
_ROLE_ID     = "1497947608616931428"   # "premium" role — checked for access
_INVITE_URL  = "https://discord.com/invite/MSXdaexYdH"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (secret + redirect URI still come from Streamlit secrets)
# ─────────────────────────────────────────────────────────────────────────────

def _get(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


def _config() -> Dict[str, Optional[str]]:
    return {
        "client_id":     _CLIENT_ID,
        "client_secret": _get("DISCORD_CLIENT_SECRET", "TxobYzh5Ti7rYsmYVIs8Q0jgi3jrYSsj"),
        "redirect_uri":  _get("DISCORD_REDIRECT_URI"),
        "guild_id":      _GUILD_ID,
        "role_id":       _ROLE_ID,
        "invite_url":    _INVITE_URL,
    }


def is_configured() -> bool:
    cfg = _config()
    return bool(cfg["client_id"] and cfg["client_secret"] and cfg["redirect_uri"])


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
    Retries on 429 (respects Retry-After). Raises RuntimeError with the
    real Discord error body so the problem is immediately visible.
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

    for attempt in range(3):
        try:
            r = requests.post(TOKEN_URL, data=data, headers=headers, timeout=15)

            if r.status_code == 429:
                retry_after = float(r.headers.get("Retry-After", 2 ** attempt))
                logger.warning("Discord 429 – waiting %.1fs (attempt %d/3)",
                               retry_after, attempt + 1)
                time.sleep(min(retry_after, 10))
                continue

            if not r.ok:
                try:
                    body = r.json()
                except Exception:
                    body = r.text[:400]
                raise RuntimeError(
                    f"Discord token exchange failed — HTTP {r.status_code}: {body}\n\n"
                    f"Most common causes:\n"
                    f"  • DISCORD_REDIRECT_URI in Streamlit secrets doesn't match "
                    f"the redirect registered in the Discord Developer Portal\n"
                    f"  • DISCORD_CLIENT_SECRET is wrong or has been reset\n"
                    f"  • The OAuth code expired (>30 s between login click and redirect)\n"
                    f"\nRedirect URI used: {cfg['redirect_uri']}"
                )

            return r.json()

        except RuntimeError:
            raise
        except requests.RequestException as exc:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(
                f"Discord token exchange: network error after 3 attempts: {exc}"
            ) from exc

    raise RuntimeError(
        "Discord token exchange: failed after 3 attempts (all rate-limited)"
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


def fetch_member(token: str, guild_id: str) -> Dict[str, Any]:
    """Fetch the guild member object (includes roles)."""
    r = requests.get(
        f"{DISCORD_API}/users/@me/guilds/{guild_id}/member",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def is_member(token: str, guild_id: str) -> bool:
    """Return True if the user is in the guild."""
    try:
        guilds = fetch_guilds(token)
        return any(str(g.get("id")) == str(guild_id) for g in guilds)
    except Exception:
        return False


def has_premium_role(token: str, guild_id: str, role_id: str) -> bool:
    """Return True if the user has the premium role in the guild."""
    try:
        member = fetch_member(token, guild_id)
        return str(role_id) in [str(r) for r in member.get("roles", [])]
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Session state helpers
# ─────────────────────────────────────────────────────────────────────────────

SESSION_KEY  = "_discord_user"
_CODE_USED   = "_discord_code_used"
_HAS_PREMIUM = "_discord_has_premium"


def current_user() -> Optional[Dict[str, Any]]:
    return st.session_state.get(SESSION_KEY)


def logout() -> None:
    for k in (SESSION_KEY, _CODE_USED, _HAS_PREMIUM):
        st.session_state.pop(k, None)


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


def user_has_premium() -> bool:
    """Returns True if the logged-in user has the premium role."""
    return bool(st.session_state.get(_HAS_PREMIUM, False))


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit gate
# ─────────────────────────────────────────────────────────────────────────────

def require_login(
    render_login=None,
    render_denied=None,
) -> Optional[Dict[str, Any]]:
    """
    Streamlit gate.
    - Must be in the guild to access.
    - Premium role is checked and stored in session state (_HAS_PREMIUM)
      but does NOT block access — use user_has_premium() in the app
      to gate premium features if needed.
    - Dev mode: if DISCORD_REDIRECT_URI is not set, gate is skipped.
    - _CODE_USED flag prevents re-exchange on every rerun (fixes 429).
    """
    if not is_configured():
        # Dev mode — no redirect URI configured, skip gate
        return None

    user = current_user()
    if user:
        return user

    cfg  = _config()
    code = st.query_params.get("code")
    if isinstance(code, list):
        code = code[0]

    if code and not st.session_state.get(_CODE_USED):
        # Mark immediately — before network call — so reruns don't retry
        st.session_state[_CODE_USED] = True

        try:
            token_data   = exchange_code(code)
            token        = token_data["access_token"]

            # Must be in the guild
            if not is_member(token, cfg["guild_id"]):
                u = fetch_user(token)
                if render_denied:
                    render_denied(u, cfg["invite_url"])
                else:
                    _default_denied(u, cfg["invite_url"])
                st.stop()

            u = fetch_user(token)
            st.session_state[SESSION_KEY]  = u
            # Check and store premium role (non-blocking)
            st.session_state[_HAS_PREMIUM] = has_premium_role(
                token, cfg["guild_id"], cfg["role_id"]
            )
            st.query_params.clear()
            st.rerun()

        except Exception as exc:  # noqa: BLE001
            # Reset flag so the user can try again
            st.session_state.pop(_CODE_USED, None)
            st.error(str(exc))

    # Not logged in → show login screen and stop
    if render_login:
        render_login(build_authorize_url(), cfg["invite_url"])
    else:
        _default_login(build_authorize_url(), cfg["invite_url"])
    st.stop()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Default UI  (overridden by the themed versions in streamlit_app.py)
# ─────────────────────────────────────────────────────────────────────────────

def _default_login(authorize_url: str, invite_url: str) -> None:
    st.markdown(
        f"### Login required\n"
        f"[Login with Discord]({authorize_url})\n\n"
        f"Not a member yet? [Join here]({invite_url})."
    )


def _default_denied(user: Dict[str, Any], invite_url: str) -> None:
    name = display_name(user)
    st.error(
        f"Hi {name}, you're not a member of the required Discord server. "
        f"Join here: {invite_url}"
    )
