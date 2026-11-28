"""The answer middleware is INSTALLED by `tests/conftest.py` and BITES on a real route (046 FR-046.12, WT2; FTASKS 7.1).

Asserted on the live session, not on the module's existence:
1. `tests/conftest.py` imported it (the module object conftest holds is this one), and `FastAPI.build_middleware_stack`
   is its wrapper, so a fresh app's built stack is an `AnswerCheck`;
2. a real covered route of `src.main.app` (`GET /api/v1/system/gpu-list`), answered through the real stack, is checked:
   against a contract entry that requires a field the route does not send, and against one that does not declare a
   key the route sends, a failure names the key and the field; against the committed entry, none;
3. the conftest gate turns a recorded failure into a red test, even when nothing raised (a planted run).

Controls (0xcc/reviews/046_controls_<date>.md):
  M1  remove the import from tests/conftest.py         → test_conftest_installs_it red (and the gate's planted run)
  M2  the wrapper returns the original stack           → test_a_fresh_apps_stack_is_the_check red
  M3  the endpoint map keys by the router-relative path → test_a_real_route_is_checked_* red
  M4  the check validates the published form           → test_an_undeclared_key_from_a_real_route_fails red
  M40 the check validates with no format checker       → test_formats_are_checked_as_the_production_client_* red
"""
from __future__ import annotations

import asyncio
import copy
import sys
from pathlib import Path

import httpx
import pytest

KEY = "GET /api/v1/system/gpu-list"


def test_conftest_installs_it():
    conftest = sys.modules.get("tests.conftest")
    assert conftest is not None, "tests/conftest.py is not loaded as tests.conftest"
    import tests.support.answer_middleware as middleware

    assert getattr(conftest, "_answer_middleware", None) is middleware, "tests/conftest.py does not install it"
    assert middleware.installed()


def test_a_fresh_apps_stack_is_the_check():
    from fastapi import FastAPI

    from tests.support.answer_middleware import AnswerCheck

    assert isinstance(FastAPI().build_middleware_stack(), AnswerCheck)


@pytest.fixture
def gpus(monkeypatch):
    """`get_gpu_list` answers two GPUs without touching a device."""
    from src.api.v1.endpoints import system

    class Info:
        def __init__(self, i):
            self.i = i

        def to_dict(self):
            return {"index": self.i, "name": f"card {self.i}"}

    class Service:
        def is_available(self):
            return True

        def get_device_count(self):
            return 2

        def get_all_gpu_info(self):
            return [Info(0), Info(1)]

    monkeypatch.setattr(system, "get_gpu_monitor_service", lambda: Service())


@pytest.fixture
def middleware(monkeypatch):
    import tests.support.answer_middleware as middleware

    yield middleware
    middleware.FAILURES.clear()  # this module's planted failures are expected; the conftest gate must not see them


def _with_entry(middleware, monkeypatch, edit) -> None:
    committed = middleware.contract().contract
    doc = copy.deepcopy(committed)
    edit(doc)
    monkeypatch.setitem(middleware._STATE, "contract", middleware.Contract(doc))  # noqa: SLF001


def _call() -> httpx.Response:
    from src.main import app

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.get("/api/v1/system/gpu-list")

    return asyncio.run(go())


def test_a_real_route_is_checked_and_a_missing_field_is_named(gpus, middleware, monkeypatch):
    def require_more(doc):
        target = doc["$defs"]["GPUListResponse"]
        target["properties"]["driver"] = {"type": "string"}
        target["required"] = [*target["required"], "driver"]

    _with_entry(middleware, monkeypatch, require_more)
    middleware.FAILURES.clear()
    response = _call()
    assert response.status_code == 200 and response.json()["gpu_count"] == 2  # the answer itself is untouched
    assert [f.key for f in middleware.FAILURES] == [KEY], middleware.FAILURES
    assert "driver" in str(middleware.FAILURES[0])


def test_an_undeclared_key_from_a_real_route_fails(gpus, middleware, monkeypatch):
    """The STRICT form: the published entry allows extra keys; the check does not (FR-046.12 AC)."""
    def forget_gpus(doc):
        target = doc["$defs"]["GPUListResponse"]
        target["properties"].pop("gpus")
        target["required"] = [r for r in target["required"] if r != "gpus"]

    _with_entry(middleware, monkeypatch, forget_gpus)
    middleware.FAILURES.clear()
    _call()
    assert len(middleware.FAILURES) == 1 and "undeclared key 'gpus'" in str(middleware.FAILURES[0]), middleware.FAILURES


def test_a_real_route_is_checked_clean_against_the_committed_entry(gpus, middleware):
    middleware.FAILURES.clear()
    before = middleware.CHECKED.get(KEY, 0)
    _call()
    assert middleware.FAILURES == []
    assert middleware.CHECKED.get(KEY, 0) == before + 1  # it ran, and found nothing


def test_formats_are_checked_as_the_production_client_checks_them(gpus, middleware, monkeypatch):
    """046 P7 finding (2026-11-26): the check judges `format` with `tests/support/strict_formats.STRICT`, so an entry
    promising `date-time` for a value the route sends without an offset fails (the live `list_cluster_profiles`
    regression), and so does a format the checker does not know (fail closed). Here `name` ("card 0") is declared
    each way in turn."""
    for fmt, says in (("date-time", "is not a 'date-time'"), ("hostname", "unknown format 'hostname'")):
        def promise(doc, fmt=fmt):
            doc["$defs"]["GPUListResponse"]["properties"]["gpus"]["items"] = {
                "type": "object", "additionalProperties": True,
                "properties": {"name": {"type": "string", "format": fmt}}}

        _with_entry(middleware, monkeypatch, promise)
        middleware.FAILURES.clear()
        _call()
        assert len(middleware.FAILURES) == 2 and all(says in str(f) for f in middleware.FAILURES), (
            fmt, middleware.FAILURES)


def test_an_undeclared_status_fails(gpus, middleware, monkeypatch):
    def move_status(doc):
        doc["responses"][KEY] = {"201": doc["responses"][KEY]["200"]}

    _with_entry(middleware, monkeypatch, move_status)
    middleware.FAILURES.clear()
    _call()
    assert len(middleware.FAILURES) == 1 and "status 200 is not declared" in str(middleware.FAILURES[0])


def test_the_conftest_gate_fails_a_test_whose_answer_was_off_contract(tmp_path):
    """The autouse gate in `tests/conftest.py`, unmodified, on a planted run: a recorded failure is a red test."""
    import os
    import subprocess

    backend = Path(__file__).resolve().parents[3]
    (tmp_path / "conftest.py").write_text((backend / "tests" / "conftest.py").read_text(encoding="utf-8"))
    (tmp_path / "test_planted.py").write_text(
        "import tests.support.answer_middleware as am\n\n\n"
        "def test_off_contract():\n"
        "    am.FAILURES.append(am.AnswerContractError('GET /api/x', 200, 'a', 'planted'))\n\n\n"
        "def test_clean():\n"
        "    pass\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env["PYTHONPATH"] = str(backend)
    run = subprocess.run([sys.executable, "-m", "pytest", "test_planted.py", "-q", "-p", "no:cacheprovider",
                          "-p", "no:xdist", "--no-cov"], cwd=tmp_path, env=env, capture_output=True, text=True,
                         timeout=900)
    out = run.stdout + run.stderr
    assert "answers off their contract" in out and "planted" in out, out[-2000:]
    assert "2 passed" in out and "1 error" in out, out[-2000:]


#: What turning a check off would have to touch: the installed hook, or a check function of either module.
_OFF_TARGETS = {"build_middleware_stack", "install", "check", "check_json", "check_other", "strict_form", "_judge_requests",
                "FAILURES"}


def _switch_findings(source: str, name: str) -> list[str]:
    """An assignment, `setattr`/`monkeypatch.setattr` or `del` naming one of `_OFF_TARGETS` on the two check modules or
    on `FastAPI` (046 MW3): the one way a test could quietly turn the undeclared-key check off."""
    import ast

    tree = ast.parse(source)
    owners = {"FastAPI", "answer_middleware", "contract_fakes", "contract_tools"}
    for node in ast.walk(tree):  # the names this file binds the check modules (or FastAPI) to, aliases included
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                full = f"{getattr(node, 'module', '') or ''}.{alias.name}"
                if any(word in full for word in owners):
                    owners.add(alias.asname or alias.name.split(".")[0])

    def watched(owner: str) -> bool:
        return any(part in owners for part in owner.replace("(", ".").split("."))

    out = []
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
        for target in targets:
            if isinstance(target, ast.Attribute) and target.attr in _OFF_TARGETS:
                if watched(ast.unparse(target.value)):
                    out.append(f"{name}:{node.lineno} {ast.unparse(target)}")
        if isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) == "setattr" \
                and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value in _OFF_TARGETS:
            owner = ast.unparse(node.args[0])
            if watched(owner):
                out.append(f"{name}:{node.lineno} setattr({owner}, {node.args[1].value!r})")
    return out


def test_nothing_in_the_tree_turns_the_check_off():
    """FTASKS 7.7 (MW3): no module in `tests/` or `src/` other than the two check modules themselves (which install
    them) reassigns, patches or deletes the hook or a check function. A planted file is found (the guard's control)."""
    backend = Path(__file__).resolve().parents[3]
    own = {backend / "tests" / "support" / "answer_middleware.py", backend / "tests" / "support" / "contract_fakes.py"}
    files = [p for root in ("tests", "src") for p in sorted((backend / root).rglob("*.py")) if p not in own]
    assert len(files) > 500, len(files)
    found = [f for p in files for f in _switch_findings(p.read_text(encoding="utf-8"), str(p.relative_to(backend)))]
    assert not found, found
    planted = ("import tests.support.answer_middleware as am\nimport fastapi\n"
               "fastapi.FastAPI.build_middleware_stack = am.ORIGINAL_BUILD\n"
               "def t(monkeypatch):\n    monkeypatch.setattr(am, 'check', lambda *a: None)\n")
    assert len(_switch_findings(planted, "planted")) == 2


def test_there_is_no_switch():
    """MW3 (FTASKS 7.7): nothing in the middleware reads the environment, so nothing can turn the check off."""
    source = (Path(__file__).resolve().parents[2] / "support" / "answer_middleware.py").read_text(encoding="utf-8")
    import ast

    names = {n.attr if isinstance(n, ast.Attribute) else n.id for n in ast.walk(ast.parse(source))
             if isinstance(n, (ast.Attribute, ast.Name))}
    assert not names & {"environ", "getenv"}, sorted(names & {"environ", "getenv"})


# ── 046-2 review round 1: excluded routes checked per status, unknown endpoints, embedded documents ─────────────

EXTRACTION_DELETE = "DELETE /api/v1/extractions/{extraction_id}"


@pytest.mark.parametrize(("status", "media", "size", "says"), [
    (204, None, 0, None),                       # deleted at once: no body, as declared
    (202, "text/plain", 40, None),              # queued: a text note, as declared
    (202, None, 40, "no content type"),         # the route's queued 202 as it is served today (a T1 body: debt)
    (204, "text/plain", 0, None),               # a content type with no bytes is no body (round 2, O3)
    (204, "text/plain", 3, "declared None"),    # a body where the status declares none
    (200, None, 0, "status 200 is not declared"),
])
def test_an_excluded_route_is_checked_per_status(middleware, status, media, size, says):
    """Finding 4: an excluded row records each declared 2xx with its own media type, and the check reads it: no body,
    a body of the declared type, or a failure naming why."""
    found = middleware.contract()
    assert found.contract["excluded"][EXTRACTION_DELETE]["statuses"] == {"202": "text/plain", "204": None}
    failures = found.check_other(EXTRACTION_DELETE, status, media, size)
    if says is None:
        assert failures == [], failures
    else:
        assert len(failures) == 1 and says in str(failures[0]), failures


def test_a_src_endpoint_the_app_does_not_serve_fails(middleware):
    """Finding 7: an answer from a `src.` endpoint that is no route of `src.main.app` is a failure naming it, never
    a silent pass (the branch a test's own route takes is the `None` key)."""
    async def stray():  # pragma: no cover - never served
        return {}

    stray.__module__ = "src.api.v1.endpoints.not_a_router"
    middleware.FAILURES.clear()
    middleware.check("GET", stray, 200, "application/json", b"{}", 2)
    assert len(middleware.FAILURES) == 1, middleware.FAILURES
    assert "not a route of src.main.app" in str(middleware.FAILURES[0])
    assert "the contract does not list this route" in str(middleware.FAILURES[0])

    async def own():  # a test's own route: not judged
        return {}

    own.__module__ = "tests.unit.support.test_answer_middleware_wiring"
    middleware.FAILURES.clear()
    middleware.check("GET", own, 200, "application/json", b"{}", 2)
    assert middleware.FAILURES == []


@pytest.mark.parametrize("key", ["GET /api/v1/circuits/{circuit_id}/export",
                                 "GET /api/v1/cluster-profiles/{profile_id}/export",
                                 "GET /api/v1/jlens/artifacts/{slug}/document",
                                 "GET /api/v1/probe-monitors/probes/{probe_id}/definition"])
def test_a_planted_key_in_an_exported_document_fails_once(middleware, key):
    """Finding 1: each document-export route's answer is held to the document's STRICT form: one planted key, at the
    top level or nested one level down, fails."""
    from tests.support.answer_samples import sample

    found = middleware.contract()
    good = sample(found.contract["responses"][key]["200"], found.contract)
    assert found.check_json(key, 200, good) == [], found.check_json(key, 200, good)
    # A document that is a `oneOf` of kinds (the cluster bundle, the J-space kinds) names the best branch's error,
    # which may be another kind's; the answer still fails, once.
    top = {**good, "planted_top": 1}
    assert len(found.check_json(key, 200, top)) == 1, found.check_json(key, 200, top)
    nested_field = next(k for k, v in good.items() if isinstance(v, dict) and v and k != "$schema")
    nested = {**good, nested_field: {**good[nested_field], "planted_nested": 1}}
    assert found.check_json(key, 200, nested), nested_field


def test_a_date_must_exist_in_its_month():
    """Review round 1 (minor): ajv-formats' full mode refuses a day its month does not have."""
    from tests.support.strict_formats import STRICT

    for good in ("2024-02-29", "2026-01-31", "2026-12-31T23:59:59Z"):
        assert STRICT.conforms(good, "date-time" if "T" in good else "date"), good
    for bad in ("2026-02-29", "2026-02-30", "2026-04-31", "2026-04-31T00:00:00Z"):
        assert not STRICT.conforms(bad, "date-time" if "T" in bad else "date"), bad


def test_a_json_routes_no_body_status_passes_only_when_declared(middleware):
    """Round 2 (O5 survived): a JSON route answering a status with no body passes only when its entry lists that status
    in `x-no-body`."""
    doc = copy.deepcopy(middleware.contract().contract)
    found = middleware.Contract(doc)
    assert "no body from a route the contract lists as JSON" in str(found.check_other(KEY, 204, None, 0)[0])
    doc["responses"][KEY]["x-no-body"] = ["204"]
    found = middleware.Contract(doc)
    assert found.check_other(KEY, 204, None, 0) == []
    assert found.check_other(KEY, 204, "application/json", 0) == []  # a content type, no bytes


def test_an_off_contract_answer_from_a_shared_fixture_is_reported_not_wiped(tmp_path):
    """The middleware's own shared-fixture gate (`tests/conftest.py`, `_answers_match_their_contract`): an off-contract
    answer recorded while a module-scoped fixture was built, before the test's setup, is reported, never wiped. The
    fake check's twin is `test_contract_fakes_wiring.py::test_a_failure_from_a_shared_fixture_is_reported_not_wiped`;
    this gate had none, and turning it off left every test green (the rebase onto 0b64145e6, control R2e'')."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    backend = Path(__file__).resolve().parents[3]
    (tmp_path / "conftest.py").write_text((backend / "tests" / "conftest.py").read_text(encoding="utf-8"))
    (tmp_path / "test_planted.py").write_text(
        "import pytest\nimport tests.support.answer_middleware as am\n\n\n"
        "@pytest.fixture(scope='module')\ndef world():\n"
        "    am.FAILURES.append(am.AnswerContractError('GET /api/v1/x', 200, '', 'from a shared fixture'))\n\n\n"
        "def test_uses_the_world(world):\n    pass\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env["PYTHONPATH"] = str(backend)
    run = subprocess.run([sys.executable, "-m", "pytest", "test_planted.py", "-q", "-p", "no:cacheprovider",
                          "-p", "no:xdist", "--no-cov"], cwd=tmp_path, env=env, capture_output=True, text=True,
                         timeout=900)
    out = run.stdout + run.stderr
    assert run.returncode != 0, out[-2000:]
    assert "answers off their contract, recorded while a shared fixture was built" in out, out[-2000:]
    assert "from a shared fixture" in out, out[-2000:]
