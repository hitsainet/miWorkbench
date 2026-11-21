"""A stored provider key goes only to the base URL it was saved for (045 FPRD FR-19; CHAIN_PLAN A39, A43; FTDD §18a).

Before 045 the operator's OpenAI key (the encrypted `openai_api_key` setting, or the environment's
`OPENAI_API_KEY`) was handed to a client pointed at whatever endpoint the request named, so a judge run, spot-check
or prompt try naming `https://attacker.example/v1` received `Authorization: Bearer <operator's key>`.

Now EVERY read of that key is in this module (`test_provider_keys.py::test_every_key_read_goes_through_provider_key_for`
walks `src/` for any other), and `provider_key_for(endpoint, db)` returns a key only when `endpoint` has one of that
key's base URLs — same scheme, host and port, and a path under the base's path, after normalisation. A client for
any other host is built with NO key (a keyless local judge or miServer is unaffected).

Bindings (A43, V-3):
* the stored key is bound to the base URLs saved beside it (`openai_api_key_base_urls`, a JSON list; written by the
  settings page with any new key, and by the migration `e045_provider_key_hosts` for the key that existed before);
* with no saved list, and for the environment key (which has none), the binding is the provider default
  `https://api.openai.com/v1` plus the operator's stored `openai_compatible_endpoint` when set;
* a labelling job's own encrypted copy (the requester's key, `labeling_jobs.openai_api_key`) is bound the same way;
* a call naming no endpoint is a call to the provider default: OpenAI labelling passes no `base_url`, so
  `provider_key_for(None, db)` resolves `https://api.openai.com/v1` and the operator's labelling keeps its key.

A configuration that asks for the key BY PROVIDER (method `openai`) while it resolves to another host is refused
`422 key_not_for_this_host` before anything is queued (`require_provider_key*`).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
KEY_SETTING = "openai_api_key"
BASE_URLS_SETTING = "openai_api_key_base_urls"
COMPATIBLE_ENDPOINT_SETTING = "openai_compatible_endpoint"
KEY_NOT_FOR_THIS_HOST = "key_not_for_this_host"

_DEFAULT_PORTS = {"http": 80, "https": 443}


def normalise_base_url(url: Optional[str]) -> Optional[tuple[str, str, int, str]]:
    """(scheme, host, port, path) with the scheme and host lower-cased, the default port explicit and the path
    without a trailing slash; None for anything that is not an absolute http(s) URL."""
    if not url or not isinstance(url, str):
        return None
    try:
        parts = urlsplit(url.strip())
        scheme = (parts.scheme or "").lower()
        host = (parts.hostname or "").lower()
        port = parts.port or _DEFAULT_PORTS.get(scheme)
    except ValueError:
        return None
    if scheme not in _DEFAULT_PORTS or not host or port is None:
        return None
    path = parts.path or ""
    if _escapes_its_path(path):
        return None
    return scheme, host, port, path.rstrip("/")


#: Encoded forms of `.`, `/` and `\` that a client or server may decode into a path step.
_ENCODED_STEPS = ("%2e", "%2f", "%5c")


def _escapes_its_path(path: str) -> bool:
    """R3 review fix (OR3-045a0-6), 2026-11-16: a path with a `.` or `..` segment, a backslash, or an encoded dot or
    slash is REFUSED (no key for it), never compared raw. httpx removes dot segments before sending, so
    `http://judge.lan:8000/v1/../../x` passed the `/v1` prefix test and went out as `/x/…` with the key."""
    lowered = path.lower()
    if "\\" in path or any(code in lowered for code in _ENCODED_STEPS):
        return True
    return any(segment in (".", "..") for segment in path.split("/"))


def endpoint_has_base(endpoint: Optional[str], base: Optional[str]) -> bool:
    """Does `endpoint` (None = the provider default) live under `base`? Scheme, host and port equal; the path equal to
    the base's path or below it (`/v1` admits `/v1/chat`, never `/v1x`)."""
    e = normalise_base_url(endpoint or DEFAULT_OPENAI_BASE_URL)
    b = normalise_base_url(base)
    if e is None or b is None or e[:3] != b[:3]:
        return False
    return e[3] == b[3] or e[3].startswith(b[3] + "/")


def _clean(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _parse_base_urls(raw: Optional[str]) -> Optional[list[str]]:
    if not raw:
        return None
    try:
        urls = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("%s is not a JSON list; the stored key is bound to no host", BASE_URLS_SETTING)
        return []
    if not isinstance(urls, list):
        return []
    return [u for u in urls if normalise_base_url(u) is not None]


def default_base_urls(compatible_endpoint: Optional[str]) -> list[str]:
    """The binding of a key with no saved base URL: the provider default, plus the operator's compatible endpoint."""
    out = [DEFAULT_OPENAI_BASE_URL]
    if normalise_base_url(compatible_endpoint) is not None:
        out.append(compatible_endpoint.strip())
    return out


def _decrypt(row: Any) -> Optional[str]:
    if row is None or not row.value:
        return None
    if not row.is_sensitive:
        return _clean(row.value)
    from ..core.encryption import decrypt_value

    return _clean(decrypt_value(row.value, setting_key=row.key))


def _choose(endpoint: Optional[str], candidates: Iterable[tuple[Optional[str], list[str]]]) -> Optional[str]:
    for key, bases in candidates:
        if key and any(endpoint_has_base(endpoint, b) for b in bases):
            return key
    return None


def _environment_key() -> Optional[str]:
    from ..core.config import settings

    return _clean(getattr(settings, "openai_api_key", None))


def environment_key_for(endpoint: Optional[str]) -> Optional[str]:
    """The environment key for a client with no database at hand: bound to the provider default only."""
    return _environment_key() if endpoint_has_base(endpoint, DEFAULT_OPENAI_BASE_URL) else None


def decrypt_job_key(encrypted: Optional[str]) -> Optional[str]:
    """A labelling job's own copy of the requester's key (written encrypted at create)."""
    if not encrypted:
        return None
    from ..core.encryption import decrypt_value

    return _clean(decrypt_value(encrypted, setting_key=KEY_SETTING))


def _rows_sync(db: Any) -> dict[str, Any]:
    from ..models.app_setting import AppSetting

    rows = db.query(AppSetting).filter(
        AppSetting.key.in_((KEY_SETTING, BASE_URLS_SETTING, COMPATIBLE_ENDPOINT_SETTING))).all()
    return {r.key: r for r in rows}


async def _rows_async(db: Any) -> dict[str, Any]:
    from sqlalchemy import select

    from ..models.app_setting import AppSetting

    result = await db.execute(select(AppSetting).where(
        AppSetting.key.in_((KEY_SETTING, BASE_URLS_SETTING, COMPATIBLE_ENDPOINT_SETTING))))
    return {r.key: r for r in result.scalars().all()}


def _candidates(rows: dict[str, Any], job_key: Optional[str]) -> list[tuple[Optional[str], list[str]]]:
    compatible = rows.get(COMPATIBLE_ENDPOINT_SETTING)
    compatible_url = _decrypt(compatible) if compatible is not None else None
    defaults = default_base_urls(compatible_url)
    base_row = rows.get(BASE_URLS_SETTING)
    saved = _parse_base_urls(base_row.value if base_row is not None else None)
    stored_bases = defaults if saved is None else saved
    return [
        (job_key, defaults),
        (_decrypt(rows.get(KEY_SETTING)), stored_bases),
        (_environment_key(), defaults),
    ]


def provider_key_for(endpoint: Optional[str], db: Any, *, job_key: Optional[str] = None) -> Optional[str]:
    """The provider key for a request to `endpoint` (None = the provider default), or None (sync session)."""
    try:
        rows = _rows_sync(db)
    except Exception as exc:  # noqa: BLE001 - no settings readable: the environment key alone, still bound
        logger.warning("provider_key_for: settings unreadable (%s); using the environment key only", exc)
        rows = {}
    return _choose(endpoint, _candidates(rows, job_key))


async def provider_key_for_async(endpoint: Optional[str], db: Any) -> Optional[str]:
    """`provider_key_for` on an async session."""
    try:
        rows = await _rows_async(db)
    except Exception as exc:  # noqa: BLE001
        logger.warning("provider_key_for: settings unreadable (%s); using the environment key only", exc)
        rows = {}
    return _choose(endpoint, _candidates(rows, None))


def any_provider_key_sync(db: Any, *, job_key: Optional[str] = None) -> bool:
    try:
        rows = _rows_sync(db)
    except Exception:  # noqa: BLE001
        rows = {}
    return any(k for k, _ in _candidates(rows, job_key))


async def any_provider_key_async(db: Any) -> bool:
    try:
        rows = await _rows_async(db)
    except Exception:  # noqa: BLE001
        rows = {}
    return any(k for k, _ in _candidates(rows, None))


class KeyNotForThisHost(ValueError):
    """A configuration asked for the stored key by provider while naming a host it is not bound to."""

    code = KEY_NOT_FOR_THIS_HOST

    def __init__(self, endpoint: Optional[str]):
        super().__init__(
            f"{KEY_NOT_FOR_THIS_HOST}: the stored OpenAI key is bound to its saved base URL(s); "
            f"{endpoint or DEFAULT_OPENAI_BASE_URL} is not one of them. Save the key for that host in "
            f"Settings → API Keys, or use a keyless endpoint."
        )
        self.endpoint = endpoint


def http_refusal(exc: KeyNotForThisHost):
    from fastapi import HTTPException

    return HTTPException(422, {"code": KEY_NOT_FOR_THIS_HOST, "message": str(exc), "endpoint": exc.endpoint})


def require_provider_key(endpoint: Optional[str], db: Any, *, job_key: Optional[str] = None) -> Optional[str]:
    """For a configuration that asks for the key BY PROVIDER: the key, or `KeyNotForThisHost` when a key exists but
    is not bound to `endpoint`. None when no key is configured at all (the caller reports that as before)."""
    key = provider_key_for(endpoint, db, job_key=job_key)
    if key is None and any_provider_key_sync(db, job_key=job_key):
        raise KeyNotForThisHost(endpoint)
    return key


async def require_provider_key_async(endpoint: Optional[str], db: Any) -> Optional[str]:
    key = await provider_key_for_async(endpoint, db)
    if key is None and await any_provider_key_async(db):
        raise KeyNotForThisHost(endpoint)
    return key
