"""The fake check is INSTALLED and BITES (046 FR-046.24, WT3; FTASKS 3.2).

The wiring is asserted on the live classes before this module imports the plugin itself (it is imported lazily
below), so only `tests/conftest.py`'s installing import can have wrapped them.

Controls (0xcc/reviews/046_controls_2026-11-25.md):
  M20  remove `import tests.support.contract_fakes` from tests/conftest.py → every test errors (the conftest gate
       names it) and test_the_three_clients_are_wrapped red
  M21  the check validates against the contract, not its strict form     → test_an_undeclared_key_in_a_fake_fails red
"""
from __future__ import annotations

import asyncio

import httpx
import pytest


def test_the_three_clients_are_wrapped():
    from src.mcp_server.client import MiWorkbenchClient
    from src.mcp_server.dataworks_client import DataworksClient
    from src.mcp_server.miserver_client import MiServerClient

    assert getattr(MiWorkbenchClient.request, "__contract_fakes__", None) == "miworkbench"
    assert getattr(MiServerClient.request_full, "__contract_fakes__", None) == "miserver"
    assert getattr(MiServerClient.raw_get, "__contract_fakes__", None) == "miserver"
    assert getattr(MiServerClient._get_jsonl_full, "__contract_fakes__", None) == "miserver"  # noqa: SLF001
    assert getattr(DataworksClient.request, "__contract_fakes__", None) == "dataworks"


def _miserver(handler) -> "object":
    from src.mcp_server.miserver_client import MiServerClient

    client = MiServerClient("http://miserver.invalid")
    client._http = httpx.AsyncClient(base_url="http://miserver.invalid", transport=httpx.MockTransport(handler))
    return client


def _status_body(**extra) -> dict:
    """`GET /api/circuit-sensing/events`' real envelope (a page of events), plus `extra` in its `data`."""
    return {"success": True, "error": None, "data": {"events": [], "count": 0, **extra}}


def _run(client, method="GET", path="/api/approvals/actions"):
    return asyncio.run(client.request_full(method, path))


@pytest.fixture
def fakes():
    import tests.support.contract_fakes as contract_fakes

    yield contract_fakes
    contract_fakes.FAILURES.clear()  # this module's planted failures are expected; the conftest gate must not see them


def _actions(extra_entry: dict | None = None) -> dict:
    from tests.support.miserver_fakes import approval_actions, envelope

    body = approval_actions(["model_load"])
    if extra_entry:
        body["actions"][0].update(extra_entry)
    return envelope(body)


def test_a_fake_in_the_contracts_shape_passes_and_is_counted(fakes):
    before = fakes.CHECKED["miserver"]
    client = _miserver(lambda request: httpx.Response(200, json=_actions()))
    assert _run(client).body == {"actions": _actions()["data"]["actions"]}
    assert fakes.CHECKED["miserver"] == before + 1


def test_an_undeclared_key_in_a_fake_fails(fakes):
    """The planted key: an entry the contract's `ApprovalActionEntry` does not declare."""
    client = _miserver(lambda request: httpx.Response(200, json=_actions({"undeclared": 1})))
    with pytest.raises(fakes.FakeAnswerError, match=r"GET /api/approvals/actions.*undeclared"):
        _run(client)
    assert fakes.FAILURES and fakes.FAILURES[-1].key == "GET /api/approvals/actions"


def test_a_wrong_typed_field_in_a_fake_fails(fakes):
    client = _miserver(lambda request: httpx.Response(200, json=_actions({"label": 7})))
    with pytest.raises(fakes.FakeAnswerError, match="label|7"):
        _run(client)


def test_a_route_the_contract_does_not_list_fails(fakes):
    client = _miserver(lambda request: httpx.Response(200, json={"success": True, "data": {}, "error": None}))
    with pytest.raises(fakes.FakeAnswerError, match="GET /api/no-such-route.*not in the contract"):
        _run(client, path="/api/no-such-route")


def test_a_status_the_contract_does_not_declare_fails(fakes):
    client = _miserver(lambda request: httpx.Response(201, json=_actions()))
    with pytest.raises(fakes.FakeAnswerError, match="status is not declared"):
        _run(client)


def test_an_error_answer_is_not_judged(fakes):
    from src.mcp_server.client import BackendError

    client = _miserver(lambda request: httpx.Response(404, json={"detail": "Not Found"}))
    with pytest.raises(BackendError):
        _run(client)
    assert not fakes.FAILURES


def test_a_real_transport_is_never_judged(fakes):
    """Only `MockTransport` traffic is a fake; any other transport (a real call) passes untouched."""

    class Real(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, json=_actions({"undeclared": 1}), request=request)

    from src.mcp_server.miserver_client import MiServerClient

    client = MiServerClient("http://miserver.invalid")
    client._http = httpx.AsyncClient(base_url="http://miserver.invalid", transport=Real())
    assert _run(client).body["actions"][0]["undeclared"] == 1


def test_a_product_with_no_contract_yet_is_skipped_and_counted(fakes, monkeypatch):
    """miWorkbench's own contract arrives in 046-2 (FTASKS 7.3): until then its fakes are counted as skipped."""
    from src.mcp_server.client import MiWorkbenchClient

    monkeypatch.setitem(fakes.CONTRACTS, "miworkbench", fakes.CONTRACTS["miworkbench"].with_name("absent.json"))
    fakes._contract.cache_clear()
    try:
        before = fakes.SKIPPED["miworkbench"]
        client = MiWorkbenchClient("http://miworkbench.invalid")
        client._http = httpx.AsyncClient(base_url="http://miworkbench.invalid/api/v1",
                                         transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"a": 1})))
        assert asyncio.run(client.request("GET", "/anything")) == {"a": 1}
        assert fakes.SKIPPED["miworkbench"] == before + 1
        assert "skipped (no contract yet)" in fakes.summary()
    finally:
        fakes._contract.cache_clear()


def test_a_failure_the_tool_swallows_still_reaches_the_gate(fakes, monkeypatch, request):
    """Driven through the SDK, a tool turns the check's raise into an error RESULT; the failure is still recorded,
    and the conftest's autouse gate (in every test's fixtures) fails the test at teardown."""
    from tests.support import mcp_drive

    assert "_fake_answers_match_their_contracts" in request.fixturenames
    backend = mcp_drive.RouteBackend().answer("GET", "/api/circuit-sensing/status",
                                              {"success": True, "error": None, "data": {"planted_key": 1}})
    with mcp_drive.patched_routes(monkeypatch, backend):
        mcp = mcp_drive.build_category("miserver_circuits")
        result = mcp_drive.call(mcp, "miserver_circuit_sensing_status", {})
    assert result.is_error
    assert fakes.FAILURES and fakes.FAILURES[-1].key == "GET /api/circuit-sensing/status"


def test_a_raw_document_export_is_judged(fakes):
    """review round 1, finding 2a: `raw_get` does not go through `request_full`; its answer is judged too."""
    client = _miserver(lambda request: httpx.Response(200, json={"not": "a circuit document"}))
    with pytest.raises(fakes.FakeAnswerError, match="GET /api/circuits/{circuit_id}/export"):
        asyncio.run(client.raw_get("/api/circuits/c1/export"))


def test_a_json_body_sent_as_text_is_judged(fakes):
    """review round 1, finding 2b: the clients parse any body, so a JSON answer with no JSON content type is judged."""
    import json

    client = _miserver(lambda request: httpx.Response(200, content=json.dumps(_actions({"undeclared": 1})).encode()))
    with pytest.raises(fakes.FakeAnswerError, match="undeclared"):
        _run(client)


def test_a_2xx_the_client_refuses_is_still_judged(fakes):
    """review round 1, finding 2c: judged as it arrives, before the client raises on it."""
    client = _miserver(lambda request: httpx.Response(200, json={"success": False, "data": None,
                                                                 "error": {"code": "X", "message": "m"},
                                                                 "undeclared": 1}))
    with pytest.raises(fakes.FakeAnswerError, match="undeclared"):
        _run(client)
    assert fakes.FAILURES


def test_the_conftest_gate_fails_a_test_whose_failure_was_swallowed(tmp_path):
    """review round 1, finding 3: the autouse gate in `tests/conftest.py` is what turns a recorded failure into a red
    test when a tool swallowed the raise; it is driven here, unmodified, on a planted test in a separate run (the
    gate runs at teardown, so it cannot fail the test that observes it)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    backend = Path(__file__).resolve().parents[3]
    (tmp_path / "conftest.py").write_text((backend / "tests" / "conftest.py").read_text(encoding="utf-8"))
    (tmp_path / "test_planted.py").write_text(
        "import tests.support.contract_fakes as cf\n\n\n"
        "def test_swallowed():\n"
        "    cf.FAILURES.append(cf.FakeAnswerError('miserver', 'GET /api/x', 200, '', 'planted'))\n\n\n"
        "def test_clean():\n"
        "    pass\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env["PYTHONPATH"] = str(backend)
    run = subprocess.run([sys.executable, "-m", "pytest", "test_planted.py", "-q", "-p", "no:cacheprovider",
                          "-p", "no:xdist", "--no-cov"], cwd=tmp_path, env=env, capture_output=True, text=True,
                         timeout=900)
    out = run.stdout + run.stderr
    assert "fake backend answers that their contracts do not allow" in out, out[-2000:]
    assert "2 passed" in out and "1 error" in out, out[-2000:]  # the gate errors the swallowed one at teardown


def test_a_failure_from_a_shared_fixture_is_reported_not_wiped(tmp_path):
    """review round 1, finding 3: a module-scoped fixture's fake failure is recorded before the gate's setup; the
    gate reports it instead of clearing it."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    backend = Path(__file__).resolve().parents[3]
    (tmp_path / "conftest.py").write_text((backend / "tests" / "conftest.py").read_text(encoding="utf-8"))
    (tmp_path / "test_planted.py").write_text(
        "import pytest\nimport tests.support.contract_fakes as cf\n\n\n"
        "@pytest.fixture(scope='module')\ndef world():\n"
        "    cf.FAILURES.append(cf.FakeAnswerError('miserver', 'GET /api/x', 200, '', 'from a shared fixture'))\n\n\n"
        "def test_uses_the_world(world):\n    pass\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env["PYTHONPATH"] = str(backend)
    run = subprocess.run([sys.executable, "-m", "pytest", "test_planted.py", "-q", "-p", "no:cacheprovider",
                          "-p", "no:xdist", "--no-cov"], cwd=tmp_path, env=env, capture_output=True, text=True,
                         timeout=900)
    out = run.stdout + run.stderr
    assert "recorded while a shared fixture was built" in out and "from a shared fixture" in out, out[-2000:]


def test_a_miworkbench_fake_is_judged_against_its_own_contract(fakes):
    """046 FTASKS 7.3: miWorkbench's product is on: a fake of a typed miWorkbench route with an undeclared key fails,
    naming the route; the same answer without it passes and is counted."""
    from src.mcp_server.client import MiWorkbenchClient
    from tests.support.fake_clients import workbench_answer

    good = workbench_answer("GET", "/api/v1/system/gpu-list", {"gpu_count": 1, "gpus": []})

    def client(body):
        c = MiWorkbenchClient("http://miworkbench.invalid")
        c._http = httpx.AsyncClient(base_url="http://miworkbench.invalid/api/v1",  # noqa: SLF001
                                    transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)))
        return c

    before = fakes.CHECKED["miworkbench"]
    assert asyncio.run(client(good).get("/system/gpu-list")) == good
    assert fakes.CHECKED["miworkbench"] == before + 1
    with pytest.raises(fakes.FakeAnswerError, match="GET /api/v1/system/gpu-list"):
        asyncio.run(client({**good, "undeclared": 1}).get("/system/gpu-list"))


# ── 046-2 review round 1, findings 1 to 3 ──────────────────────────────────────────────────────────────────────


def _workbench(handler):
    from src.mcp_server.client import MiWorkbenchClient

    c = MiWorkbenchClient("http://miworkbench.invalid")
    c._http = httpx.AsyncClient(base_url="http://miworkbench.invalid/api/v1",  # noqa: SLF001
                                transport=httpx.MockTransport(handler))
    return c


def test_a_document_inside_an_answer_is_held_to_its_strict_form(fakes):
    """Finding 1: a published document's objects are open as published; a fake's undeclared key inside one fails, at
    the top level and nested."""
    from tests.support.answer_samples import published_sample

    document = published_sample("circuit-definition-v1.json")
    for planted in ({**document, "undeclared_top": 1},
                    {**document, "provenance": {**document["provenance"], "undeclared_nested": 1}}):
        with pytest.raises(fakes.FakeAnswerError, match="GET /api/v1/circuits/{circuit_id}/export.*undeclared"):
            asyncio.run(_workbench(lambda request, body=planted: httpx.Response(200, json=body)).get(
                "/circuits/c1/export"))
    fakes.FAILURES.clear()
    before = fakes.CHECKED["miworkbench"]
    asyncio.run(_workbench(lambda request: httpx.Response(200, json=document)).get("/circuits/c1/export"))
    assert fakes.CHECKED["miworkbench"] == before + 1


def test_the_fake_check_judges_shape_not_formats(fakes):
    """Finding 2, declined with its reason: a fake is judged on its SHAPE (keys and types, strictly); formats are judged
    on real served answers (the answer middleware). The typed-output harnesses sample fakes from the tool schemas, which
    carry no offset formats (`without_offset_formats`), so a format check here would judge the sampler, not a route."""
    from tests.support.answer_samples import published_sample

    document = published_sample("circuit-definition-v1.json")
    naive = {**document, "provenance": {**document["provenance"], "created_at": "2026-11-25T00:00:00"}}
    asyncio.run(_workbench(lambda request: httpx.Response(200, json=naive)).get("/circuits/c1/export"))
    assert not fakes.FAILURES


@pytest.mark.parametrize("answer", [lambda request: httpx.Response(200, text="OK"),
                                    lambda request: httpx.Response(200, content=b"")],
                         ids=["text body", "empty body"])
def test_a_json_route_answered_with_no_json_fails(fakes, answer):
    """Finding 3: a 2xx body that is not JSON, or empty, on a route the contract types as JSON fails, not skipped."""
    before = fakes.CHECKED["miworkbench"]
    with pytest.raises(Exception):
        asyncio.run(_workbench(answer).get("/system/gpu-list"))
    assert fakes.FAILURES and fakes.FAILURES[-1].key == "GET /api/v1/system/gpu-list", fakes.FAILURES[-1:]
    assert fakes.CHECKED["miworkbench"] == before


def test_a_nullable_routes_neutral_fake_answers_the_json_null():
    """Round 2, finding 1: a `ContractBackend` on a nullable route samples `None` and must send the bytes `null` (an
    empty body is what the fake check refuses), and a scripted `None` likewise; a 204 stays bodiless."""
    from tests.support.mcp_drive import ContractBackend, json_response

    backend = ContractBackend()
    response = backend.handler(httpx.Request("GET", "http://miworkbench.invalid/api/v1/jlens/artifacts/s1/band-report"))
    assert response.status_code == 200 and response.content == b"null", (response.status_code, response.content)
    assert response.headers["content-type"] == "application/json"
    assert json_response(200, None).content == b"null" and json_response(204, None).content == b""


def test_an_empty_body_passes_only_on_a_declared_no_body_status(fakes, monkeypatch):
    """Round 2 (O4 survived): an empty 2xx on a JSON route is refused unless the entry's `x-no-body` lists the status,
    and then it is counted."""
    import copy

    real = fakes._contract("miworkbench")
    doc = copy.deepcopy(real)
    doc["responses"]["GET /api/v1/system/gpu-list"]["x-no-body"] = ["200"]
    monkeypatch.setattr(fakes, "_contract", lambda product: doc if product == "miworkbench" else real)
    before = fakes.CHECKED["miworkbench"]
    asyncio.run(_workbench(lambda request: httpx.Response(200, content=b"")).get("/system/gpu-list"))
    assert not fakes.FAILURES and fakes.CHECKED["miworkbench"] == before + 1
