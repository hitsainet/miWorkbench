"""045 FPRD FR-19 (CHAIN_PLAN A39, A43 V-3; FTDD §18a; FTASKS 21.8): a stored provider key goes only to its own host.

* `provider_key_for(endpoint, db)` returns the stored key only when `endpoint` has one of its saved base URLs (scheme,
  host, port, path prefix), the environment key only for the provider default (and the stored compatible endpoint);
  a call naming no endpoint is a call to `https://api.openai.com/v1`.
* Wire: the OpenAI client's transport (beneath `openai`) sees `Authorization: Bearer <key>` at the key's own host and
  NO key at a request-named host, through the expression every judge call site uses
  (`_client_for(endpoint, resolve_api_key(db, endpoint))`).
* The call sites: every `_client_for(...)` passes the endpoint it connects to into `resolve_api_key` (AST), and every
  read of the key is in `services/provider_keys.py` (AST).
* Labelling: method `openai` with no endpoint keeps the operator's key (stored, or environment only); a key bound
  elsewhere is `422 key_not_for_this_host` before anything is created.
* The migration `e045_provider_key_hosts` binds the existing key to the default and the compatible endpoint.

Controls (0xcc/reviews/045_A0_implementation_controls_2026-11-16.md): pass the stored key to any host at one call
site → `test_every_judge_call_site_binds_its_own_endpoint` red; read `openai_api_key` directly in one service →
`test_every_key_read_goes_through_provider_key_for` red; `provider_key_for` returns None for a missing endpoint →
`test_operator_openai_labelling_keeps_its_key` red; drop the host comparison → the wire test red.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from src.services import provider_keys as pk
from tests.support.hub_spy import blocked_sockets  # noqa: F401 - fixture

# The `openai` client speaks httpx2 (a separate package) at this pin; the spy patches ITS transports.
import httpx2  # noqa: E402


@pytest.fixture(autouse=True)
def _no_network(blocked_sockets):  # noqa: F811
    """Nothing in this file may leave the machine, whatever a broken spy would let through."""

STORED = "sk-STORED-operator-key"
ENVKEY = "sk-ENV-deployment-key"
SRC = Path(__file__).resolve().parents[2] / "src"


def _rows(key=STORED, bases=None, compatible=None):
    rows = {}
    if key:
        rows[pk.KEY_SETTING] = SimpleNamespace(key=pk.KEY_SETTING, value=key, is_sensitive=False)
    if bases is not None:
        rows[pk.BASE_URLS_SETTING] = SimpleNamespace(key=pk.BASE_URLS_SETTING, value=json.dumps(bases),
                                                     is_sensitive=False)
    if compatible:
        rows[pk.COMPATIBLE_ENDPOINT_SETTING] = SimpleNamespace(key=pk.COMPATIBLE_ENDPOINT_SETTING, value=compatible,
                                                               is_sensitive=False)
    return rows


@pytest.fixture
def settings_rows(monkeypatch):
    """The settings rows `provider_keys` reads, without writing the shared test database."""
    state = {"rows": _rows(), "env": None}

    async def _async(db):
        return state["rows"]

    monkeypatch.setattr(pk, "_rows_sync", lambda db: state["rows"])
    monkeypatch.setattr(pk, "_rows_async", _async)
    monkeypatch.setattr(pk, "_environment_key", lambda: state["env"])
    return state


@pytest.mark.parametrize("endpoint,expected", [
    (None, True),
    ("https://api.openai.com/v1", True),
    ("https://API.OpenAI.com:443/v1/", True),
    ("https://api.openai.com/v1/chat/completions", True),
    ("https://api.openai.com/v1x", False),
    ("http://api.openai.com/v1", False),
    ("https://api.openai.com:8443/v1", False),
    ("https://api.openai.com.evil.tld/v1", False),
    ("https://evil.tld/api.openai.com/v1", False),
    ("https://attacker.example/v1", False),
    ("not a url", False),
])
def test_endpoint_has_base(endpoint, expected):
    assert pk.endpoint_has_base(endpoint, pk.DEFAULT_OPENAI_BASE_URL) is expected


def test_the_stored_key_is_bound_to_its_saved_base_urls(settings_rows):
    settings_rows["rows"] = _rows(bases=["http://judge.lan:8000/v1"])
    assert pk.provider_key_for("http://judge.lan:8000/v1", None) == STORED
    assert pk.provider_key_for("https://api.openai.com/v1", None) is None, "saved list replaces the default"
    assert pk.provider_key_for("https://attacker.example/v1", None) is None


def test_no_saved_list_binds_the_default_and_the_compatible_endpoint(settings_rows):
    settings_rows["rows"] = _rows(compatible="http://ollama.lan:11434/v1")
    assert pk.provider_key_for(None, None) == STORED
    assert pk.provider_key_for("http://ollama.lan:11434/v1", None) == STORED
    assert pk.provider_key_for("http://ollama.lan:11435/v1", None) is None


def test_the_environment_key_is_bound_to_the_default(settings_rows):
    settings_rows["rows"], settings_rows["env"] = {}, ENVKEY
    assert pk.provider_key_for(None, None) == ENVKEY
    assert pk.provider_key_for("https://attacker.example/v1", None) is None


# ── the wire ────────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def openai_wire(monkeypatch):
    """A spy beneath the OpenAI client: every request's host and Authorization, answered with a chat completion."""
    seen: list[tuple[str, str | None]] = []

    def handle(self, request):
        seen.append((request.url.host, request.headers.get("authorization")))
        return httpx2.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}]})

    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", handle)
    return seen


def _judge_call(endpoint):
    """The expression every judge call site uses (judge run, spot-check, prompt try)."""
    from src.services.probe_monitor_judge import _client_for, resolve_api_key

    client = _client_for(endpoint, resolve_api_key(None, endpoint))
    client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])


@pytest.mark.parametrize("endpoint", ["https://attacker.example/v1", "http://10.9.8.7:8000/v1",
                                      "https://api.openai.com.evil.tld/v1"])
def test_key_never_goes_to_a_request_named_host(settings_rows, openai_wire, endpoint):
    settings_rows["env"] = ENVKEY
    _judge_call(endpoint)
    assert openai_wire, "the spy saw nothing: the test proves nothing"
    assert all(STORED not in (a or "") and ENVKEY not in (a or "") for _, a in openai_wire), openai_wire


DOT_FORMS = ["http://judge.lan:8000/v1/../../x", "http://judge.lan:8000/v1/..", "http://judge.lan:8000/v1/./chat",
             "http://judge.lan:8000/v1/%2e%2e/x", "http://judge.lan:8000/v1/%2E%2E/x", "http://judge.lan:8000/v1/.%2e/x",
             "http://judge.lan:8000/v1%2f..%2fx", "http://judge.lan:8000/v1/..\\x"]


@pytest.mark.parametrize("endpoint", DOT_FORMS)
def test_a_path_that_steps_out_of_its_base_gets_no_key(settings_rows, openai_wire, endpoint):
    """R3 review fix (OR3-045a0-6), 2026-11-16: dot segments, a backslash and encoded dots or slashes are refused
    before anything is sent (httpx would remove the dot segments and send the key outside the bound path).
    Control: drop `_escapes_its_path` from `normalise_base_url` → red."""
    settings_rows["rows"] = _rows(compatible="http://judge.lan:8000/v1")
    assert pk.endpoint_has_base(endpoint, "http://judge.lan:8000/v1") is False
    assert pk.provider_key_for(endpoint, None) is None
    try:
        _judge_call(endpoint)
    except Exception:  # noqa: BLE001 - a URL the client itself refuses sends nothing either
        pass
    assert all(STORED not in (a or "") for _, a in openai_wire), openai_wire


def test_the_bound_path_itself_still_gets_its_key(settings_rows):
    settings_rows["rows"] = _rows(compatible="http://judge.lan:8000/v1")
    assert pk.provider_key_for("http://judge.lan:8000/v1/chat/completions", None) == STORED


def test_a_corrupt_binding_binds_the_key_to_no_host(settings_rows):
    """R3 review fix (OR3-045a0-4, R3-7), 2026-11-16 (the reviewer's candidate, on the in-memory rows): a saved binding
    that is not a JSON list binds the stored key to NO host, never to the defaults (the provider plus the compatible
    endpoint). Control: fall back to the defaults on a corrupt list → red."""
    rows = _rows(compatible="http://judge.lan:8000/v1")
    rows[pk.BASE_URLS_SETTING] = SimpleNamespace(key=pk.BASE_URLS_SETTING, value="not json", is_sensitive=False)
    settings_rows["rows"] = rows
    assert pk.provider_key_for(None, None) is None
    assert pk.provider_key_for("http://judge.lan:8000/v1", None) is None


def test_key_goes_to_its_own_host(settings_rows, openai_wire):
    _judge_call("https://api.openai.com/v1")
    assert openai_wire == [("api.openai.com", f"Bearer {STORED}")]


# ── the call sites ───────────────────────────────────────────────────────────────────────────────────────────────
def _calls(name):
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == name:
                yield path.relative_to(SRC.parent), node


def test_every_judge_call_site_binds_its_own_endpoint():
    """Each `_client_for(endpoint, key)` that carries a key takes it from `resolve_api_key(db, <that endpoint>)`."""
    sites = []
    for path, call in _calls("_client_for"):
        if "mcp_server" in str(path) or len(call.args) < 2:
            continue
        key = call.args[1]
        assert isinstance(key, ast.Call) and getattr(key.func, "id", None) == "resolve_api_key", (path, call.lineno)
        assert len(key.args) == 2 and ast.dump(key.args[1]) == ast.dump(call.args[0]), \
            f"{path}:{call.lineno} resolves the key for another endpoint than it connects to"
        sites.append(str(path))
    assert sorted(sites) == sorted([
        "src/services/probe_monitor_judge.py", "src/services/probe_spot_check.py",
        "src/api/v1/endpoints/probe_judge_prompts.py"]), sites


# Files that DECLARE the key (schema fields, the ORM column, settings definitions and lists), never read it.
DECLARATIONS = {
    "src/schemas/labeling.py", "src/models/labeling_job.py", "src/core/config.py",
    "src/services/app_setting_service.py", "src/services/provider_keys.py",
}
# Lines that touch the name but are not a read of the stored or environment key, each with its reason.
NOT_A_READ = {
    ("src/services/labeling_service.py", 'encrypt_value(config["openai_api_key"])'):
        "the create path ENCRYPTS the requester's own key into the job row",
    ("src/services/labeling_service.py", 'if config.get("openai_api_key")'): "the same create path",
    ("src/services/labeling_service.py", "decrypt_job_key(labeling_job.openai_api_key)"):
        "the job row's copy goes to provider_keys, which decrypts and binds it",
    ("src/api/v1/endpoints/labeling.py", "or config.openai_api_key:"):
        "whether the REQUEST brings its own key (then the stored key is not asked for)",
}


def key_reads(root: Path = SRC):
    """(file, line) of every read of the key outside `provider_keys.py`: an attribute load `.openai_api_key`, or the
    string "openai_api_key" compared with, or passed to, anything."""
    out = []
    for path in sorted(root.rglob("*.py")):
        rel = str(path.relative_to(root.parent))
        if rel in DECLARATIONS:
            continue
        tree = ast.parse(path.read_text(), str(path))
        parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
        for node in ast.walk(tree):
            hit = False
            if isinstance(node, ast.Attribute) and node.attr == "openai_api_key" and isinstance(node.ctx, ast.Load):
                hit = True
            elif isinstance(node, ast.Constant) and node.value == "openai_api_key":
                parent = parents.get(node)
                if isinstance(parent, ast.keyword):
                    parent = parents.get(parent)
                hit = isinstance(parent, (ast.Compare, ast.Call, ast.Subscript, ast.Assign))
            if not hit:
                continue
            line = path.read_text().splitlines()[node.lineno - 1]
            if any(rel == f and marker in line for f, marker in NOT_A_READ):
                continue
            out.append((rel, node.lineno))
    return sorted(set(out))


def test_every_key_read_goes_through_provider_key_for():
    assert key_reads() == []


def test_the_key_read_guard_sees_a_direct_read(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text(
        "x = settings.openai_api_key\n"
        "row = db.query(S).filter(S.key == 'openai_api_key').first()\n"
        "k = get_decrypted_value(db, 'openai_api_key')\n")
    assert [line for _, line in key_reads(tmp_path / "src")] == [1, 2, 3]


# ── labelling ────────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("which", ["stored", "environment"])
def test_operator_openai_labelling_keeps_its_key(settings_rows, which):
    """A43, V-3: OpenAI labelling names no endpoint; the worker's resolution keeps the operator's key."""
    if which == "environment":
        settings_rows["rows"], settings_rows["env"] = {}, ENVKEY
    assert pk.require_provider_key(None, None) == (STORED if which == "stored" else ENVKEY)


def test_openai_labelling_keeps_its_key_on_the_wire(settings_rows, monkeypatch):
    """The labelling client built from that key, as the worker builds it (no base_url), sends it to OpenAI."""
    import asyncio

    from src.services.openai_labeling_service import OpenAILabelingService

    seen = []

    async def handle(self, request):
        seen.append((request.url.host, request.headers.get("authorization")))
        cls = httpx2.Response if isinstance(request, httpx2.Request) else httpx.Response
        return cls(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                                         "choices": [{"index": 0, "finish_reason": "stop",
                                                      "message": {"role": "assistant", "content": "{}"}}]})

    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", handle)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle)
    service = OpenAILabelingService(api_key=pk.require_provider_key(None, None), model="m")
    asyncio.run(service.client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}]))
    assert seen == [("api.openai.com", f"Bearer {STORED}")]


def test_openai_method_with_another_host_is_refused(settings_rows):
    settings_rows["rows"] = _rows(bases=["http://judge.lan:8000/v1"])
    with pytest.raises(pk.KeyNotForThisHost) as refused:
        pk.require_provider_key(None, None)
    assert refused.value.code == "key_not_for_this_host"
    settings_rows["rows"], settings_rows["env"] = {}, None
    assert pk.require_provider_key(None, None) is None, "no key at all is reported as before, not as a host refusal"


async def test_the_labelling_route_refuses_before_creating(settings_rows, monkeypatch):
    """`POST /labeling` with method `openai` and a key bound elsewhere: 422 `key_not_for_this_host`, no job."""
    from src.main import app
    from src.services import labeling_service

    settings_rows["rows"] = _rows(bases=["http://judge.lan:8000/v1"])
    monkeypatch.setattr(labeling_service.LabelingService, "create_labeling_job",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("a job was created")), raising=False)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.post("/api/v1/labeling", json={"extraction_job_id": "ext_nope", "labeling_method": "openai"})
    assert r.status_code == 422, r.text[:300]
    assert r.json()["detail"]["code"] == "key_not_for_this_host"


# ── the migration ───────────────────────────────────────────────────────────────────────────────────────────────
def test_the_migration_binds_the_existing_key():
    from sqlalchemy import text

    from tests.unit.test_e008_drain_migration import _load, _run, _scratch_engine

    module = _load("e045_provider_key_hosts.py", "e045b7c1d2f3", "e045a1b2c3d4")
    engine = _scratch_engine("e045_provider_keys")
    try:
        with engine.begin() as conn:
            import src.models  # noqa: F401
            from src.core.database import Base

            conn.execute(text('DROP TABLE IF EXISTS "app_settings" CASCADE'))
            Base.metadata.tables["app_settings"].create(conn)

            def put(key, value, sensitive=False):
                conn.execute(text("INSERT INTO app_settings (id, key, value, is_sensitive, category) VALUES "
                                  "(gen_random_uuid(), :k, :v, :s, 'x')"), {"k": key, "v": value, "s": sensitive})

            def bases():
                return conn.execute(text("SELECT value FROM app_settings WHERE key = 'openai_api_key_base_urls'")
                                    ).scalar()

            _run(conn, module.upgrade)
            assert bases() is None, "no stored key: nothing to bind"
            put("openai_api_key", "enc", True)
            put("openai_compatible_endpoint", "http://ollama.lan:11434/v1")
            _run(conn, module.upgrade)
            assert json.loads(bases()) == ["https://api.openai.com/v1", "http://ollama.lan:11434/v1"]
            conn.execute(text("UPDATE app_settings SET value = '[\"http://x:1/v1\"]' "
                              "WHERE key = 'openai_api_key_base_urls'"))
            _run(conn, module.upgrade)
            assert json.loads(bases()) == ["http://x:1/v1"], "a saved binding is never overwritten"
            _run(conn, module.downgrade)
            assert bases() is None
            conn.execute(text('DROP TABLE "app_settings" CASCADE'))
    finally:
        engine.dispose()
