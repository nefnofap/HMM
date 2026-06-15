"""
discord_auth.py  —  Discord OAuth2 gate for Regime Terminal
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode, urlparse, urlunparse

import requests
import streamlit as st

logger = logging.getLogger(__name__)

DISCORD_API   = "https://discord.com/api/v10"
AUTHORIZE_URL = "https://discord.com/oauth2/authorize"
TOKEN_URL     = "https://discord.com/api/oauth2/token"
SCOPES        = "identify guilds"

_CLIENT_ID     = "1513610614549905588"
_CLIENT_SECRET = "TxobYzh5Ti7rYsmYVIs8Q0jgi3jrYSsj"
_REDIRECT_URI  = "https://luciiregime.streamlit.app/"
_GUILD_ID      = "1478370205523775699"
_ROLE_ID       = "1497947608616931428"
_INVITE_URL    = "https://discord.com/invite/MSXdaexYdH"


def _get(key: str, default: str) -> str:
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


def _config() -> Dict[str, str]:
    return {
        "client_id":     _CLIENT_ID,
        "client_secret": _get("DISCORD_CLIENT_SECRET", _CLIENT_SECRET),
        "redirect_uri":  _get("DISCORD_REDIRECT_URI",  _REDIRECT_URI),
        "guild_id":      _GUILD_ID,
        "role_id":       _ROLE_ID,
        "invite_url":    _INVITE_URL,
    }


def _normalize_uri(uri: str) -> str:
    """Ensure redirect URI always has a trailing slash."""
    if not uri.endswith("/"):
        uri = uri + "/"
    return uri


def _detect_redirect_uri() -> str:
    """
    Detect the actual redirect URI from the current page URL.
    This ensures what we send to Discord's token endpoint exactly matches
    what Streamlit actually redirected to (with or without trailing slash).
    """
    try:
        # Try to get the actual URL from query params context
        # Streamlit doesn't expose the full URL, so we use the hardcoded value
        # but normalise it to always have trailing slash
        cfg = _config()
        return _normalize_uri(cfg["redirect_uri"])
    except Exception:
        return _REDIRECT_URI


def build_authorize_url() -> str:
    cfg = _config()
    redirect = _normalize_uri(cfg["redirect_uri"])
    params = {
        "client_id":     cfg["client_id"],
        "redirect_uri":  redirect,
        "response_type": "code",
        "scope":         SCOPES,
        "prompt":        "consent",
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def exchange_code(code: str) -> Dict[str, Any]:
    """
    Try exchange with trailing slash first, then without.
    This handles cases where Streamlit changes the URI on redirect.
    """
    cfg = _config()
    base_uri = cfg["redirect_uri"].rstrip("/")
    
    # Try both variants — whichever matches what Discord has registered
    uris_to_try = [
        base_uri + "/",   # with trailing slash
        base_uri,         # without trailing slash
    ]

    last_error = None
    for redirect_uri in uris_to_try:
        data = {
            "client_id":     cfg["client_id"],
            "client_secret": cfg["client_secret"],
            "grant_type":    "authorization_code",
            "code":          code,
            "redirect_uri":  redirect_uri,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        for attempt in range(3):
            try:
                r = requests.post(TOKEN_URL, data=data, headers=headers, timeout=15)

                if r.status_code == 429:
                    retry_after = float(r.headers.get("Retry-After", 2 ** attempt))
                    time.sleep(min(retry_after, 10))
                    continue

                if r.status_code == 400:
                    # redirect_uri mismatch — try next variant
                    try:
                        body = r.json()
                    except Exception:
                        body = r.text[:200]
                    last_error = f"HTTP 400 with redirect_uri={redirect_uri}: {body}"
                    break  # break retry loop, try next URI variant

                if not r.ok:
                    try:
                        body = r.json()
                    except Exception:
                        body = r.text[:400]
                    raise RuntimeError(
                        f"Discord token exchange failed — HTTP {r.status_code}: {body}"
                    )

                return r.json()

            except RuntimeError:
                raise
            except requests.RequestException as exc:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                last_error = str(exc)
                break

    raise RuntimeError(
        f"Discord token exchange failed with both redirect URI variants.\n"
        f"Last error: {last_error}\n\n"
        f"Make sure BOTH of these are added in Discord Developer Portal "
        f"under OAuth2 → Redirects:\n"
        f"  • https://luciiregime.streamlit.app/\n"
        f"  • https://luciiregime.streamlit.app"
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
    r = requests.get(
        f"{DISCORD_API}/users/@me/guilds/{guild_id}/member",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def is_member(token: str, guild_id: str) -> bool:
    try:
        guilds = fetch_guilds(token)
        return any(str(g.get("id")) == str(guild_id) for g in guilds)
    except Exception:
        return False


def has_premium_role(token: str, guild_id: str, role_id: str) -> bool:
    try:
        member = fetch_member(token, guild_id)
        return str(role_id) in [str(r) for r in member.get("roles", [])]
    except Exception:
        return False


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
    return bool(st.session_state.get(_HAS_PREMIUM, False))


def require_login(
    render_login=None,
    render_denied=None,
) -> Optional[Dict[str, Any]]:
    user = current_user()
    if user:
        return user

    cfg  = _config()
    code = st.query_params.get("code")
    if isinstance(code, list):
        code = code[0]

    if code and not st.session_state.get(_CODE_USED):
        st.session_state[_CODE_USED] = True

        try:
            token_data = exchange_code(code)
            token      = token_data["access_token"]

            if not is_member(token, cfg["guild_id"]):
                u = fetch_user(token)
                if render_denied:
                    render_denied(u, cfg["invite_url"])
                else:
                    _default_denied(u, cfg["invite_url"])
                st.stop()

            u = fetch_user(token)
            st.session_state[SESSION_KEY]  = u
            st.session_state[_HAS_PREMIUM] = has_premium_role(
                token, cfg["guild_id"], cfg["role_id"]
            )
            st.query_params.clear()
            st.rerun()

        except Exception as exc:  # noqa: BLE001
            st.session_state.pop(_CODE_USED, None)
            st.error(str(exc))

    if render_login:
        render_login(build_authorize_url(), cfg["invite_url"])
    else:
        _default_login(build_authorize_url(), cfg["invite_url"])
    st.stop()
    return None


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
