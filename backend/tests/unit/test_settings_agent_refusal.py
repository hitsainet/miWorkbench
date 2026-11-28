"""R1 review fix (OR1-045a0-6; OVN-mcp-workbench-39), 2026-11-16: an agent may not rewrite the provider key's host
binding (045 FR-19, A39) or the settings PIN through the settings routes.

The stored OpenAI key, or the environment key when no list is saved, goes only to `openai_api_key_base_urls` (or, with
none saved, to the provider default plus `openai_compatible_endpoint`). An agent that could write either setting could
re-point the operator's key at a host of its choosing and undo A39. So any caller sending `X-MiStudio-Agent` (the MCP
pod included, as for `environment_operator_only`) is refused `403 settings_operator_only` on:
* `PUT /settings` and `PUT /settings/bulk` naming either key (bulk: before ANY item is written),
* `DELETE /settings/{key}` of either key (a deleted list widens the binding to the defaults),
* `POST /settings/pin/set` and `POST /settings/pin/remove`;
* since R2 (OR2-045a0-3): writing or deleting a stored provider key (`openai_api_key`, `hf_token`,
  `neuronpedia_api_key`), any write marked sensitive or filed under `api_keys`, and deleting any row stored so.
The operator's own page (no header) is unchanged, with the request shapes `SettingsPanel.tsx` sends.

Controls (record `0xcc/reviews/045_A0_implementation_controls_2026-11-16.md`, R1 fix section):
  drop `_refuse_agent_key` from `upsert_setting`  → test_an_agent_cannot_rewrite_the_binding red
  drop `_refuse_agent_key` from the bulk loop      → test_an_agent_cannot_rewrite_the_binding_in_bulk red
  drop `_refuse_agent` from `set_pin`             → test_an_agent_cannot_set_or_remove_the_pin red
"""
from __future__ import annotations

import json

import httpx
import pytest
from sqlalchemy import select

from src.models.app_setting import AppSetting

AGENTS = [{"X-MiStudio-Agent": "miassist"}, {"X-MiStudio-Agent": "agent:miassist"}, {"X-MiStudio-Agent": "mcp"}]
BINDING = "openai_api_key_base_urls"
COMPATIBLE = "openai_compatible_endpoint"
EVIL = "https://attacker.example/v1"


async def _call(method, url, headers=None, **kwargs):
    from src.main import app

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        return await c.request(method, url, headers=headers or {}, **kwargs)


async def _value(key):
    from src.core.database import AsyncSessionLocal

    async with AsyncSessionLocal() as fresh:
        row = (await fresh.execute(select(AppSetting).where(AppSetting.key == key))).scalar_one_or_none()
        return None if row is None else row.value


async def _operator_saves_binding(urls):
    """The request `SettingsPanel.tsx` `handleSave` sends for the binding (no agent header)."""
    r = await _call("PUT", "/api/v1/settings", json={"key": BINDING, "value": json.dumps(urls),
                                                     "is_sensitive": False, "category": "api_keys"})
    assert r.status_code == 200, r.text
    return r


def _refused(r):
    assert r.status_code == 403, r.text[:300]
    assert r.json()["detail"]["code"] == "settings_operator_only", r.text[:300]


@pytest.mark.parametrize("headers", AGENTS, ids=lambda h: h["X-MiStudio-Agent"])
async def test_an_agent_cannot_rewrite_the_binding(headers, async_session):
    await _operator_saves_binding(["https://api.openai.com/v1"])
    for key, value in ((BINDING, json.dumps([EVIL])), (COMPATIBLE, EVIL)):
        _refused(await _call("PUT", "/api/v1/settings", headers, json={"key": key, "value": value}))
        _refused(await _call("DELETE", f"/api/v1/settings/{key}", headers))
    assert await _value(BINDING) == json.dumps(["https://api.openai.com/v1"]), "the binding was changed by an agent"
    assert await _value(COMPATIBLE) is None


@pytest.mark.parametrize("headers", AGENTS, ids=lambda h: h["X-MiStudio-Agent"])
async def test_an_agent_cannot_rewrite_the_binding_in_bulk(headers, async_session):
    """Refused before ANY item is written: the harmless item beside it is not saved either."""
    r = await _call("PUT", "/api/v1/settings/bulk", headers, json={"settings": [
        {"key": "fixture_harmless", "value": "1"}, {"key": COMPATIBLE, "value": EVIL}]})
    _refused(r)
    assert await _value(COMPATIBLE) is None and await _value("fixture_harmless") is None


@pytest.mark.parametrize("headers", AGENTS, ids=lambda h: h["X-MiStudio-Agent"])
async def test_an_agent_cannot_set_or_remove_the_pin(headers, async_session):
    _refused(await _call("POST", "/api/v1/settings/pin/set", headers, json={"pin": "1234"}))
    status = (await _call("GET", "/api/v1/settings/pin/status")).json()
    assert status["configured"] is False, "an agent set the settings PIN"
    assert (await _call("POST", "/api/v1/settings/pin/set", json={"pin": "1234"})).status_code == 200
    _refused(await _call("POST", "/api/v1/settings/pin/remove", headers, json={"current_pin": "1234"}))
    assert (await _call("GET", "/api/v1/settings/pin/status")).json()["configured"] is True


SECRETS = ["openai_api_key", "hf_token", "neuronpedia_api_key"]


@pytest.mark.parametrize("key", SECRETS)
@pytest.mark.parametrize("headers", AGENTS, ids=lambda h: h["X-MiStudio-Agent"])
async def test_an_agent_cannot_replace_or_delete_a_stored_key(headers, key, async_session):
    """R2 review fix (OR2-045a0-3; OVN-mcp-workbench-46), 2026-11-16: an agent may neither REPLACE a stored provider
    key (the operator's traffic would run on the agent's account) nor DELETE it, through PUT, bulk or DELETE.
    Controls: drop `_SENSITIVE_KEYS` from `AGENT_REFUSED_KEYS` → red; drop `_refuse_agent_secret_row` → the
    custom-secret case of the next test red."""
    saved = await _call("PUT", "/api/v1/settings", json={"key": key, "value": "operator-secret-1234",
                                                         "is_sensitive": True, "category": "api_keys"})
    assert saved.status_code == 200, saved.text
    before = await _value(key)
    _refused(await _call("PUT", "/api/v1/settings", headers, json={"key": key, "value": "agents-own-key-5678"}))
    _refused(await _call("PUT", "/api/v1/settings/bulk", headers,
                         json={"settings": [{"key": key, "value": "agents-own-key-5678"}]}))
    _refused(await _call("DELETE", f"/api/v1/settings/{key}", headers))
    assert await _value(key) == before, "an agent replaced or deleted the operator's stored key"


@pytest.mark.parametrize("headers", AGENTS, ids=lambda h: h["X-MiStudio-Agent"])
async def test_an_agent_cannot_store_or_delete_any_secret(headers, async_session):
    """Any write that marks itself sensitive or files under `api_keys`, and any delete of a row stored that way, is
    the operator's (the key names above are not the only secrets the page can hold)."""
    for body in ({"key": "fixture_new_secret", "value": "x", "is_sensitive": True},
                 {"key": "fixture_new_secret", "value": "x", "category": "api_keys"}):
        _refused(await _call("PUT", "/api/v1/settings", headers, json=body))
    assert await _value("fixture_new_secret") is None
    made = await _call("PUT", "/api/v1/settings", json={"key": "fixture_custom_secret", "value": "s3cret",
                                                        "is_sensitive": True, "category": "api_keys"})
    assert made.status_code == 200, made.text
    _refused(await _call("DELETE", "/api/v1/settings/fixture_custom_secret", headers))
    assert await _value("fixture_custom_secret") is not None


@pytest.mark.parametrize("item", [{"key": "fixture_bulk_secret", "value": "x", "is_sensitive": True},
                                  {"key": "fixture_bulk_secret", "value": "x", "category": "api_keys"}],
                         ids=["sensitive", "api_keys"])
async def test_an_agent_cannot_store_a_secret_in_bulk(item, async_session):
    """R3 review fix (OR3-045a0-3), 2026-11-16 (the reviewer's candidate): the R2 rule for a write marked sensitive or
    filed under `api_keys` holds on `PUT /settings/bulk` too. Control: drop the `item` argument of the bulk
    `_refuse_agent_key` call → red."""
    _refused(await _call("PUT", "/api/v1/settings/bulk", AGENTS[0], json={"settings": [item]}))
    assert (await _call("GET", "/api/v1/settings/fixture_bulk_secret")).status_code == 404


async def test_an_agent_cannot_delete_an_api_keys_row_stored_in_plain_text(async_session):
    """R3 review fix (OR3-045a0-4, R3-8), 2026-11-16: the delete refusal covers a row filed under `api_keys` even when
    it is not marked sensitive. Control: drop the category half of `_refuse_agent_secret_row` → red."""
    made = await _call("PUT", "/api/v1/settings", json={"key": "fixture_plain_api_key_row", "value": "v",
                                                        "is_sensitive": False, "category": "api_keys"})
    assert made.status_code == 200, made.text
    _refused(await _call("DELETE", "/api/v1/settings/fixture_plain_api_key_row", AGENTS[0]))
    assert await _value("fixture_plain_api_key_row") == "v"


@pytest.mark.parametrize("headers", AGENTS, ids=lambda h: h["X-MiStudio-Agent"])
async def test_an_agent_cannot_overwrite_a_stored_secret_under_any_name(headers, async_session):
    """R3 review fix (OR3-045a0-5; OVN-mcp-workbench-48), 2026-11-16: whatever its name, a row STORED as a secret
    (sensitive, or under `api_keys`) is not an agent's to overwrite, by PUT or bulk, even with a request that calls
    itself plain text (which used to downgrade the stored secret to plain text and so make it deletable).
    Controls: drop the stored-row check from `upsert_setting` → red; from the bulk pre-check → red."""
    for stored in ({"key": "fixture_custom_secret", "value": "s3cret", "is_sensitive": True, "category": "general"},
                   {"key": "fixture_custom_api_row", "value": "plain", "is_sensitive": False,
                    "category": "api_keys"}):
        made = await _call("PUT", "/api/v1/settings", json=stored)
        assert made.status_code == 200, made.text
        before = await _value(stored["key"])
        downgrade = {"key": stored["key"], "value": "agent", "is_sensitive": False, "category": "general"}
        _refused(await _call("PUT", "/api/v1/settings", headers, json=downgrade))
        _refused(await _call("PUT", "/api/v1/settings/bulk", headers,
                             json={"settings": [{"key": "fixture_harmless", "value": "1"}, downgrade]}))
        assert await _value(stored["key"]) == before, "an agent overwrote a stored secret"
    assert await _value("fixture_harmless") is None, "bulk wrote an item before refusing"


async def test_an_agent_may_still_write_other_settings(async_session):
    """The refusal is narrow: any other setting is written as before (no new gate on the rest of the page)."""
    r = await _call("PUT", "/api/v1/settings", AGENTS[0], json={"key": "fixture_harmless", "value": "1"})
    assert r.status_code == 200, r.text
    assert await _value("fixture_harmless") == "1"


async def test_the_operator_page_still_saves_the_binding_and_the_pin(async_session):
    """The operator's own UI path (no agent header, `SettingsPanel.tsx`'s request shapes) is unchanged."""
    await _operator_saves_binding(["https://api.openai.com/v1", "http://judge.lan:8000/v1"])
    assert json.loads(await _value(BINDING)) == ["https://api.openai.com/v1", "http://judge.lan:8000/v1"]
    r = await _call("PUT", "/api/v1/settings", json={"key": COMPATIBLE, "value": "http://judge.lan:8000/v1",
                                                     "is_sensitive": False, "category": "endpoints"})
    assert r.status_code == 200, r.text
    bulk = await _call("PUT", "/api/v1/settings/bulk", json={"settings": [{"key": COMPATIBLE,
                                                                            "value": "http://judge2.lan:8000/v1"}]})
    assert bulk.status_code == 200, bulk.text
    assert await _value(COMPATIBLE) == "http://judge2.lan:8000/v1"
    assert (await _call("POST", "/api/v1/settings/pin/set", json={"pin": "1234"})).status_code == 200
    assert (await _call("POST", "/api/v1/settings/pin/remove", json={"current_pin": "1234"})).status_code == 200
    assert (await _call("DELETE", f"/api/v1/settings/{COMPATIBLE}")).status_code == 204
    # OR2-045a0-3: the page's own key save and delete (`ApiKeysTab.handleSave` / `handleDelete`) still work.
    for key in SECRETS:
        saved = await _call("PUT", "/api/v1/settings", json={"key": key, "value": "operator-secret-1234",
                                                             "is_sensitive": True, "category": "api_keys"})
        assert saved.status_code == 200, saved.text
        assert (await _call("DELETE", f"/api/v1/settings/{key}")).status_code == 204
