"""The fake check (046 WT3, FR-046.24): every FAKE backend answer an MCP test gives is checked against its route's entry
in the relevant answer contract, so a fake cannot agree with the code by construction.

IMPORTING THIS MODULE INSTALLS IT (`tests/conftest.py` imports it before the app): it wraps the three clients'
body-returning methods,

* `MiWorkbenchClient.request` (`src/mcp_server/client.py`),
* `MiServerClient.request_full` (`src/mcp_server/miserver_client.py`; every verb, `request` and `upload` go through
  it), and the two miServer paths that do not: `raw_get` (a document export) and `_get_jsonl_full` (JSON Lines),
* `DataworksClient.request` (`src/mcp_server/dataworks_client.py`),

and, ONLY when the client's `httpx` transport is a `MockTransport` (a fake; a real cross-repo call is never judged
here), captures the HTTP answer the fake gave and checks it before the client unwraps or parses it: a 2xx JSON answer
must be a route and a status the product's contract lists, and must validate against that entry's STRICT form
(`src.api.contract_tools.strict_form`: an undeclared key fails, MW3). A failure raises `FakeAnswerError` naming the
product, the route, the status and the JSON path.

The contracts:

* miServer: `src/mcp_server/vendored/miserver-answers-v1.json` (046-1);
* miWorkbench: `docs/schemas/miworkbench-answers-v1.json`, from 046-2 (FTASKS 7.3);
* miDataworks: `src/mcp_server/vendored/midataworks-answers-v1.json`, from 046-3 (FTASKS 11.2).

A product whose contract is not there yet is SKIPPED AND COUNTED: the session summary prints each product's checked
and skipped counts, so a skip is never silent. There is no switch that turns the check off (MW3; FTASKS 7.7).
"""
from __future__ import annotations

import functools
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import httpx

BACKEND = Path(__file__).resolve().parents[2]
CONTRACTS: dict[str, Path] = {
    "miworkbench": BACKEND.parent / "docs" / "schemas" / "miworkbench-answers-v1.json",
    "miserver": BACKEND / "src" / "mcp_server" / "vendored" / "miserver-answers-v1.json",
    "dataworks": BACKEND / "src" / "mcp_server" / "vendored" / "midataworks-answers-v1.json",
}
#: Relayed calls that are not contract routes (FR-046.26): faking them is not checked.
NOT_CONTRACT_ROUTES: dict[tuple[str, str, str], str] = {
    ("miserver", "GET", "/openapi.json"): "miserver_recalibrate_probe reads miServer's OpenAPI as a capability probe",
}

CHECKED: Counter = Counter()
SKIPPED: Counter = Counter()
#: Every failure of the current test. A tool driven through the SDK turns an exception into an error RESULT, so a
#: raise alone could be swallowed: `tests/conftest.py`'s autouse fixture fails the test when this is not empty.
FAILURES: list["FakeAnswerError"] = []


class FakeAnswerError(AssertionError):
    """A fake backend answer its route's contract entry does not allow."""

    def __init__(self, product: str, key: str, status: int, path: str, message: str) -> None:
        self.product, self.key, self.status, self.path = product, key, status, path
        super().__init__(f"fake {product} answer for {key} ({status}) does not match its contract at {path or '/'}: "
                         f"{message}")


@functools.lru_cache(maxsize=None)
def _contract(product: str) -> Optional[dict[str, Any]]:
    path = CONTRACTS[product]
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


@functools.lru_cache(maxsize=None)
def _templates(product: str) -> dict[tuple[str, ...], tuple[str, str]]:
    """`{(METHOD, *segments): (section, key)}`, a parameter segment as `{}`."""
    from src.mcp_server.route_map import normalise

    contract = _contract(product) or {}
    out = {}
    for section in ("responses", "excluded", "not_covered"):
        for key in contract.get(section) or {}:
            method, path = key.split(" ", 1)
            out[(method, *normalise(path).split("/"))] = (section, key)
    return out


def _key_for(product: str, method: str, path: str) -> Optional[tuple[str, str]]:
    from src.mcp_server.route_map import normalise

    segments = normalise(path).split("/")
    best = None
    for (m, *parts), found in _templates(product).items():
        if m == method and len(parts) == len(segments) and all(p == "{}" or p == s for p, s in zip(parts, segments, strict=True)):
            score = parts.count("{}")
            if best is None or score < best[0]:
                best = (score, found)
    return best[1] if best else None


def product_of(host: str) -> str:
    """The product a fake request is addressed to, by the placeholder host its client was built with."""
    if host == "miserver.invalid":
        return "miserver"
    if host in ("midataworks.invalid", "dataworks.invalid"):
        return "dataworks"
    return "miworkbench"


def own_status(product: str, method: str, path: str, body: Any = None) -> int:
    """The 2xx a route answers, for a fake that does not say (046 FTASKS 7.3): its one declared 2xx; for a route that
    declares the pending `202` beside its own answer, `202` for a pending body and its own status otherwise. A route
    no contract covers (or a product with no contract) answers `200`."""
    if _contract(product) is None:
        return 200
    found = _key_for(product, method, path)
    if found is None or found[0] != "responses":
        return 200
    entry = _contract(product)["responses"][found[1]]
    declared = sorted(int(s) for s in entry if s[:1].isdigit())
    if not declared:
        return 200
    if len(declared) == 1:
        return declared[0]
    pending = isinstance(body, dict) and "approval_request_id" in body
    if 202 in declared and (pending or declared == [202]):
        return 202
    own = [s for s in declared if s != 202]
    return own[0] if own else declared[0]


@functools.lru_cache(maxsize=None)
def _validator(product: str, key: str, status: str):
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    from src.api.contract_tools import strict_form

    contract = _contract(product)
    resources = [(contract["$id"], Resource.from_contents(strict_form(contract), default_specification=DRAFT202012))]
    for sid, row in (contract.get("external") or {}).items():
        document = json.loads((BACKEND.parent / row["file"]).read_text(encoding="utf-8"))
        # strict as the contract is (046-2 review round 1, finding 1: an embedded document's undeclared key passed
        # here). Formats are NOT checked (finding 2, declined): a fake is judged on its SHAPE; a real answer's formats
        # are judged where it is served (the answer middleware for miWorkbench's routes, each sibling's own validator
        # for its own).
        resources.append((sid, Resource.from_contents(strict_form(document), default_specification=DRAFT202012)))
    pointer = "/responses/" + key.replace("~", "~0").replace("/", "~1") + "/" + status
    return Draft202012Validator({"$ref": contract["$id"] + "#" + pointer},
                                registry=Registry().with_resources(resources))


def check(product: str, response: httpx.Response) -> None:
    """Judge one fake answer. 2xx JSON only; anything else is the client's own business."""
    if not 200 <= response.status_code < 300:
        return
    method, path = response.request.method, response.request.url.path
    if (product, method, path) in NOT_CONTRACT_ROUTES:
        return
    if _contract(product) is None:
        SKIPPED[product] += 1
        return
    found = _key_for(product, method, path)
    if found is None:
        raise FakeAnswerError(product, f"{method} {path}", response.status_code, "",
                              "the route is not in the contract")
    section, key = found
    if section != "responses":
        CHECKED[product] += 1
        return  # an excluded (non-JSON) or not-covered route
    entry = _contract(product)["responses"][key]
    status = str(response.status_code)
    if status not in entry:
        raise FakeAnswerError(product, key, response.status_code, "",
                              f"the status is not declared (declared: {sorted(s for s in entry if s[:1].isdigit())})")
    if not response.content:
        if status in entry.get("x-no-body", ()):
            CHECKED[product] += 1
            return
        raise FakeAnswerError(product, key, response.status_code, "", "an empty body: the route answers JSON")
    try:
        # Whatever the content type says: the clients parse any body (`http_common.parse_body`), so a JSON body sent
        # as text is judged too (review round 1, finding 2b).
        body = response.json()
    except ValueError:
        # 046-2 review round 1, finding 3: a body that is not JSON passed here unjudged and uncounted
        raise FakeAnswerError(product, key, response.status_code, "", "a body that is not JSON: the route answers "
                                                                      "JSON") from None
    from jsonschema.exceptions import best_match

    error = best_match(_validator(product, key, status).iter_errors(body))
    if error is not None:
        while error.context:
            error = best_match(error.context)
        raise FakeAnswerError(product, key, response.status_code,
                              "/" + "/".join(str(p) for p in error.absolute_path), error.message[:500])
    CHECKED[product] += 1


def _judge_requests(http: httpx.AsyncClient, product: str) -> None:
    """Shadow `http.request` (once per client, so concurrent and nested calls share one judge) with a version that
    judges each answer AS IT ARRIVES, before the client parses, unwraps or raises on it (review round 1, finding 2c:
    a 2xx the client then refused was never judged). `AsyncClient.get` goes through `request`, so `raw_get` and the
    JSON Lines read are seen here too."""
    if getattr(http.__dict__.get("request"), "__judged_for__", None) == product:
        return
    real = type(http).request.__get__(http)

    async def judged(*a: Any, **k: Any) -> httpx.Response:
        response = await real(*a, **k)
        try:
            check(product, response)
        except FakeAnswerError as failure:
            FAILURES.append(failure)
            raise
        return response

    judged.__judged_for__ = product  # type: ignore[attr-defined]
    http.request = judged  # an instance attribute: only this (fake) client is judged


def _wrapped(method: Any, product: str) -> Any:
    @functools.wraps(method)
    async def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        http = getattr(self, "_http", None)
        if isinstance(getattr(http, "_transport", None), httpx.MockTransport):
            _judge_requests(http, product)
        return await method(self, *args, **kwargs)

    wrapper.__contract_fakes__ = product  # type: ignore[attr-defined]
    return wrapper


def install() -> None:
    from src.mcp_server.client import MiWorkbenchClient
    from src.mcp_server.dataworks_client import DataworksClient
    from src.mcp_server.miserver_client import MiServerClient

    for cls, name, product in ((MiWorkbenchClient, "request", "miworkbench"),
                               (MiServerClient, "request_full", "miserver"),
                               # review round 1, finding 2a: the two miServer paths that do not go through
                               # `request_full` (the raw document export, the JSON Lines read)
                               (MiServerClient, "raw_get", "miserver"),
                               (MiServerClient, "_get_jsonl_full", "miserver"),
                               (DataworksClient, "request", "dataworks")):
        current = getattr(cls, name)
        if getattr(current, "__contract_fakes__", None) is None:
            setattr(cls, name, _wrapped(current, product))


def summary() -> str:
    """One line per product: fakes checked, and fakes skipped because the product has no contract yet."""
    parts = []
    for product in CONTRACTS:
        skipped = f", {SKIPPED[product]} skipped (no contract yet)" if SKIPPED[product] else ""
        parts.append(f"{product} {CHECKED[product]} checked{skipped}")
    return "CONTRACT FAKES: " + "; ".join(parts)


install()
