"""
Application Settings API endpoints.

Provides CRUD operations for persistent application settings with
transparent encryption for sensitive values (API keys, tokens).
"""

import hashlib
import hmac
import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.config import RUNTIME_KEYS, settings
from ....core.deps import get_db
from ....schemas.app_setting import (
    AppSettingUpsert,
    AppSettingResponse,
    AppSettingBulkUpsert,
    AppSettingBulkResponse,
    PinStatusResponse,
    PinVerifyRequest,
    PinVerifyResponse,
    PinSetRequest,
    PinRemoveRequest,
)
from ....services.app_setting_service import (
    _SENSITIVE_KEYS,
    AppSettingService,
    ProtectedSettingError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/settings", tags=["settings"])

# Key used to store the PIN hash in app_settings
_PIN_HASH_KEY = "settings_pin_hash"

#: The header every agent request carries (the MCP pod's `mcp`, miAssist's, any other).
AGENT_HEADER = "X-MiStudio-Agent"

#: Settings an agent may never write or delete (R1 review fix, OR1-045a0-6; OVN-mcp-workbench-39), 2026-11-16: the
#: provider key's host binding (045 FR-19, A39). The stored OpenAI key, or the environment key when no list is
#: saved, goes only to these hosts, so an agent that could rewrite them could re-point the operator's key at a host
#: of its choosing and undo A39. The PIN routes are refused to agents too (`_refuse_agent`).
#: R2 review fix (OR2-045a0-3; OVN-mcp-workbench-46), 2026-11-16: the stored provider keys themselves too (every key
#: the settings service encrypts, `_SENSITIVE_KEYS`): an agent that could REPLACE one would run the operator's
#: labelling and judge traffic on an account of its choosing, and one that could DELETE it would stop that work. A
#: write that marks itself sensitive or files under the page's `api_keys` category is refused the same way, and so is
#: deleting any row stored sensitive or under `api_keys` (`_refuse_agent_secret_row`).
AGENT_REFUSED_KEYS = frozenset({"openai_api_key_base_urls", "openai_compatible_endpoint"}) | _SENSITIVE_KEYS
SECRET_CATEGORY = "api_keys"


def _refuse_agent(request: Request, what: str) -> None:
    """403 `settings_operator_only` for any caller sending the agent header, as `environment_operator_only` does for
    runtime settings. The operator's own page sends no such header and is unaffected."""
    if request.headers.get(AGENT_HEADER) is not None:
        raise HTTPException(
            status_code=403,
            detail={"code": "settings_operator_only", "setting": what,
                    "message": f"{what} is the operator's, set from Settings in miWorkbench; an agent cannot "
                               "change it."},
        )


def _refuse_agent_key(request: Request, key: str, data: Optional[AppSettingUpsert] = None) -> None:
    if key in AGENT_REFUSED_KEYS:
        _refuse_agent(request, key)
    if data is not None and (data.is_sensitive or data.category == SECRET_CATEGORY):
        _refuse_agent(request, f"a stored secret ({key})")


async def _refuse_agent_secret_row(request: Request, db: AsyncSession, key: str) -> None:
    """An agent may not overwrite or remove a row STORED sensitive or under `api_keys`, whatever its name (DELETE since
    R2; PUT and bulk since R3 review fix (OR3-045a0-5), 2026-11-16: the request alone was checked, so an agent could
    overwrite a stored secret with `is_sensitive: false` and so downgrade it to plain text)."""
    if request.headers.get(AGENT_HEADER) is None:
        return
    row = await AppSettingService.get_by_key(db, key)
    if row is not None and (row.is_sensitive or row.category == SECRET_CATEGORY):
        _refuse_agent(request, f"a stored secret ({key})")


def _hash_pin(pin: str) -> str:
    """Hash a PIN with PBKDF2-SHA256 and a random salt."""
    salt = os.urandom(32)
    dk = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, 600_000)
    return f"pbkdf2:sha256:600000:{salt.hex()}:{dk.hex()}"


def _verify_pin(pin: str, stored: str) -> bool:
    """Verify a PIN against a stored PBKDF2 hash."""
    try:
        _, algo, iters_str, salt_hex, hash_hex = stored.split(":")
        if algo != "sha256":
            return False
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        dk = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, int(iters_str))
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


@router.get("/pin/status", response_model=PinStatusResponse)
async def get_pin_status(db: AsyncSession = Depends(get_db)):
    """Return whether a settings PIN is configured and whether the bypass flag is active."""
    existing = await AppSettingService.get_by_key(db, _PIN_HASH_KEY, _privileged=True)
    return PinStatusResponse(
        configured=existing is not None,
        bypass_active=settings.bypass_settings_pin,
    )


@router.post("/pin/verify", response_model=PinVerifyResponse)
async def verify_pin(body: PinVerifyRequest, db: AsyncSession = Depends(get_db)):
    """Verify the settings PIN. Returns {valid: true} on success."""
    existing = await AppSettingService.get_by_key(db, _PIN_HASH_KEY, unmask=True, _privileged=True)
    if not existing:
        # No PIN configured — trivially valid
        return PinVerifyResponse(valid=True)
    return PinVerifyResponse(valid=_verify_pin(body.pin, existing.value))


@router.post("/pin/set", status_code=200)
async def set_pin(body: PinSetRequest, request: Request, db: AsyncSession = Depends(get_db)):
    """Set or change the settings PIN.

    Requires the current PIN when one is already configured, unless
    MIWORKBENCH_BYPASS_PIN=true is active (the recovery path). Refused to agents (OR1-045a0-6).
    """
    _refuse_agent(request, "the settings PIN")
    existing = await AppSettingService.get_by_key(db, _PIN_HASH_KEY, unmask=True, _privileged=True)

    if existing and not settings.bypass_settings_pin:
        if not body.current_pin:
            raise HTTPException(status_code=400, detail="current_pin is required to change an existing PIN")
        if not _verify_pin(body.current_pin, existing.value):
            raise HTTPException(status_code=401, detail="Current PIN is incorrect")

    await AppSettingService.upsert(
        db,
        # The PIN endpoint OWNS this key — the only privileged write in the app.
        _privileged=True,
        data=AppSettingUpsert(
            key=_PIN_HASH_KEY,
            value=_hash_pin(body.pin),
            # Encrypted at rest. Belt and braces: _PROTECTED_KEYS already hides
            # this row from every generic read, but it was served in the clear
            # BECAUSE it was written is_sensitive=False, so if that guard is
            # ever removed the row should still not be readable.
            is_sensitive=True,
            category="system",
        ),
    )
    # Explicit commit before returning so the row is visible to any immediately-
    # following request (e.g. the frontend's set-then-GET /pin/status pattern).
    # get_db also commits after the endpoint returns, but that happens after the
    # HTTP response is sent — too late for rapid sequential callers.
    await db.commit()
    return {"success": True}


@router.post("/pin/remove", status_code=200)
async def remove_pin(body: PinRemoveRequest, request: Request, db: AsyncSession = Depends(get_db)):
    """Remove the settings PIN.

    Requires the current PIN, unless MIWORKBENCH_BYPASS_PIN=true is active (the
    recovery path) — the same rule as changing it in `/pin/set`.

    WHY A DEDICATED ROUTE. Settings → API Keys → "Remove" used to call the
    generic `DELETE /settings/settings_pin_hash`. MIS-E2E-055 rightly made that
    key unreachable through the generic CRUD (`_PROTECTED_KEYS`), because a
    delete with no PIN check removed the gate entirely. The button was never
    moved, so it answered 403 every time and always showed "Failed to remove
    PIN". This route is the privileged way in, and it asks for the PIN first.
    """
    _refuse_agent(request, "the settings PIN")
    existing = await AppSettingService.get_by_key(db, _PIN_HASH_KEY, unmask=True, _privileged=True)
    if not existing:
        return {"success": True, "removed": False}

    if not settings.bypass_settings_pin:
        if not body.current_pin:
            raise HTTPException(status_code=400, detail="current_pin is required to remove the PIN")
        if not _verify_pin(body.current_pin, existing.value):
            raise HTTPException(status_code=401, detail="Current PIN is incorrect")

    # The PIN endpoints OWN this key; the generic DELETE still refuses it.
    await AppSettingService.delete_by_key(db, _PIN_HASH_KEY, _privileged=True)
    # Commit before returning, as `/pin/set` does, so the frontend's immediate
    # GET /pin/status sees the row gone.
    await db.commit()
    return {"success": True, "removed": True}


@router.get("", response_model=list[AppSettingResponse])
async def list_settings(
    category: Optional[str] = Query(None, description="Filter by category"),
    db: AsyncSession = Depends(get_db),
):
    """List all settings, optionally filtered by category. Sensitive values are masked."""
    if category:
        return await AppSettingService.get_by_category(db, category)
    return await AppSettingService.list_all(db)


@router.get("/{key}", response_model=AppSettingResponse)
async def get_setting(
    key: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a single setting by key. Sensitive values are masked."""
    setting = await AppSettingService.get_by_key(db, key)
    if not setting:
        raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")
    return setting


_URL_VALIDATED_KEYS = frozenset({"openai_compatible_endpoint"})


def _refuse_runtime_key(key: str) -> None:
    """A runtime environment setting is never an `app_settings` row (Feature 036, ENH-003 D4).

    `ollama_url` used to live here; migration b36e0a7c1d42 moved it to `environment_overrides`,
    and its reader no longer looks here. Accepting the write would store a row nothing reads while
    the caller believed it took effect.
    """
    if key in RUNTIME_KEYS:
        raise HTTPException(
            status_code=409,
            detail=(
                f"{key!r} is a runtime environment setting: change it through "
                f"PUT /api/v1/environment/{key} (Settings → Environment)"
            ),
        )


@router.put("", response_model=AppSettingResponse, status_code=200)
async def upsert_setting(
    data: AppSettingUpsert,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Create or update a setting (upsert). Sensitive values are encrypted at rest."""
    _refuse_agent_key(request, data.key, data)
    await _refuse_agent_secret_row(request, db, data.key)  # OR3-045a0-5: the STORED row decides too
    _refuse_runtime_key(data.key)
    if data.key in _URL_VALIDATED_KEYS:
        from ....utils.url_validation import validate_llm_endpoint_url
        try:
            validate_llm_endpoint_url(data.value)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    try:
        setting, is_new = await AppSettingService.upsert(db, data)
    except ProtectedSettingError as e:
        raise HTTPException(status_code=403, detail=str(e))
    try:
        # Return masked response for sensitive values. CRITICAL: expunge the row from
        # the session before mutating its `value` field, otherwise SQLAlchemy will
        # autoflush/commit the masked string back over the encrypted ciphertext on
        # request teardown — silently corrupting the row each save.
        if setting.is_sensitive:
            from ....core.encryption import decrypt_value, mask_value
            db.expunge(setting)
            try:
                setting.value = mask_value(
                    decrypt_value(setting.value, setting_key=setting.key))
            except Exception:
                # A row we cannot authenticate is masked, never echoed back as
                # raw ciphertext. decrypt_value now raises on InvalidTag.
                setting.value = "***"
        return setting
    except HTTPException:
        raise  # see the note in bulk_upsert_settings
    except Exception as e:
        logger.exception(f"Failed to upsert setting '{data.key}'")
        raise HTTPException(status_code=500, detail="Failed to save setting")


@router.put("/bulk", response_model=AppSettingBulkResponse)
async def bulk_upsert_settings(
    data: AppSettingBulkUpsert,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Create or update multiple settings at once."""
    results = []
    created = 0
    updated = 0
    for item in data.settings:  # every item is checked before any is written
        _refuse_agent_key(request, item.key, item)
        await _refuse_agent_secret_row(request, db, item.key)  # OR3-045a0-5
        _refuse_runtime_key(item.key)
    try:
        for item in data.settings:
            try:
                setting, is_new = await AppSettingService.upsert(db, item)
            except ProtectedSettingError as e:
                # /bulk must not be the way around the guard on /settings —
                # MIS-E2E-073 is the same shape (bulk skipped the URL validation
                # the single-key route applied).
                raise HTTPException(status_code=403, detail=str(e))
            if setting.is_sensitive:
                from ....core.encryption import decrypt_value, mask_value
                # Expunge before mutating — otherwise the masked string overwrites
                # the encrypted ciphertext on session commit (see upsert_setting).
                db.expunge(setting)
                try:
                    setting.value = mask_value(
                        decrypt_value(setting.value, setting_key=setting.key))
                except Exception:
                    # A row we cannot authenticate is masked, never echoed back as
                    # raw ciphertext. decrypt_value now raises on InvalidTag.
                    setting.value = "***"
            results.append(setting)
            if is_new:
                created += 1
            else:
                updated += 1
        return AppSettingBulkResponse(data=results, created=created, updated=updated)
    except HTTPException:
        # Re-raise deliberate HTTP responses before the generic handler.
        # Without this, the 403 raised for a protected key is caught below and
        # re-raised as a 500 — the MIS-E2E-103 pattern, where a client mistake
        # is reported as a server error and the real reason is lost.
        raise
    except Exception as e:
        logger.exception("Failed to bulk upsert settings")
        raise HTTPException(status_code=500, detail="Failed to save settings")


@router.delete("/{key}", status_code=204)
async def delete_setting(
    key: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Delete a setting by key."""
    _refuse_agent_key(request, key)  # deleting a saved binding would widen it to the defaults
    await _refuse_agent_secret_row(request, db, key)  # OR2-045a0-3
    _refuse_runtime_key(key)
    try:
        deleted = await AppSettingService.delete_by_key(db, key)
    except ProtectedSettingError as e:
        raise HTTPException(status_code=403, detail=str(e))
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")
