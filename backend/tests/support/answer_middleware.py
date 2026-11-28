"""The answer middleware (046 WT2, FR-046.12; FTDD §6.3, FTID §8.1): every 2xx answer a COVERED route gives during
the test session is checked against the committed `docs/schemas/miworkbench-answers-v1.json`, in its STRICT form.

Ported from miDataworks 023's `tests/support/answer_validator.py`, the suite's precedent, with miWorkbench's
differences: a route is identified by its ENDPOINT (FastAPI 0.159's `scope["route"].path` is router-relative here,
measured 2026-11-25), and a `not_covered` route's answers are not checked (MW1).

**How it is attached.** Importing this module (`tests/conftest.py`, before `from src.main import app`) replaces
`fastapi.FastAPI.build_middleware_stack`, so every FastAPI app's stack is wrapped, outermost, by `AnswerCheck`:
`src.main.app` and every app a test builds from the real routers. FastAPI builds a stack on an app's first ASGI call,
so the order of the import is not load-bearing; the app objects stay `FastAPI` instances.

**What a check does,** for an answer from a route whose endpoint lives in `src.` (a test's own route is skipped):

1. the contract key is the endpoint's (`<module>.<qualname>` -> `<METHOD> <full path>`, from the live app's routes);
   a `src.` endpoint the contract does not know FAILS;
2. a `not_covered` key is not checked;
3. a 2xx JSON answer: its status must be declared for the key (an `excluded` key answers no JSON), and the body must
   validate against the entry's strict form (`contract_tools.strict_form`): an object with named properties refuses
   a key it does not declare; a map accepts any key. The published file stays permissive (SH7);
4. a 2xx non-JSON answer, or a 204: the key must be `excluded` with that media type (`null` for no body).

The middleware forwards every message as it comes and only copies JSON body bytes, so it changes no answer. A failure
is recorded and charged to the test that produced it (`pytest_runtest_makereport`, registered by `tests/conftest.py`),
naming the key, the status, the JSON path and the message, so a failure inside a call whose exception a caller
swallowed still fails its test.

**Strict, always (MW3).** There is no switch: every failure fails its test.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

JSON = "application/json"
_INSTALLED = "__miworkbench_answer_check__"


class AnswerContractError(AssertionError):
    """An answer its contract entry does not describe (FTID §12)."""

    def __init__(self, key: str, status: int | str, json_path: str, message: str) -> None:
        self.key, self.status, self.json_path, self.message = key, str(status), json_path, message
        super().__init__(f"{key} answered {status} off its contract at {json_path or '(body)'}: {message}")


#: Failures not yet charged to a test (the conftest hooks clear and read it).
FAILURES: list[AnswerContractError] = []
#: Answers checked, per key, this session (the summary line).
CHECKED: dict[str, int] = {}
_STATE: dict[str, Any] = {}


def _pointer(*parts: str) -> str:
    return "/" + "/".join(quote(p.replace("~", "~0").replace("/", "~1"), safe="~") for p in parts)


class Contract:
    """The committed contract, compiled lazily: one validator per (key, status), strict form."""

    def __init__(self, contract: dict[str, Any]) -> None:
        from referencing import Registry, Resource
        from referencing.jsonschema import DRAFT202012

        from src.api.answers_contract import SCHEMAS_DIR
        from src.api.contract_tools import strict_form

        self.contract = contract
        resources = [(contract["$id"], Resource.from_contents(strict_form(contract), default_specification=DRAFT202012))]
        for sid, row in (contract.get("external") or {}).items():
            doc = json.loads((SCHEMAS_DIR.parents[1] / row["file"]).read_text(encoding="utf-8"))
            # The published documents leave objects open; an answer is held to their STRICT form as to the contract's
            # (046-2 review round 1, finding 1: an export's undeclared key passed at any depth).
            resources.append((sid, Resource.from_contents(strict_form(doc), default_specification=DRAFT202012)))
        self.registry = Registry().with_resources(resources)
        self._validators: dict[Any, Any] = {}

    def section(self, key: str) -> str | None:
        return next((s for s in ("responses", "excluded", "not_covered") if key in self.contract[s]), None)

    def validator(self, key: str, status: str) -> Any:
        if (key, status) not in self._validators:
            from jsonschema import Draft202012Validator

            uri = self.contract["$id"] + "#" + _pointer("responses", key, status)
            from tests.support.strict_formats import STRICT

            # Formats checked as the production client checks them (P7 finding, 2026-11-26): a contract promising
            # an offset the route never sends fails here, not in a client.
            self._validators[(key, status)] = Draft202012Validator({"$ref": uri}, registry=self.registry,
                                                                   format_checker=STRICT)
        return self._validators[(key, status)]

    def check_json(self, key: str, status: int, body: Any) -> list[AnswerContractError]:
        entry = self.contract["responses"].get(key)
        if entry is None:
            return [AnswerContractError(key, status, "", "a JSON answer from a route the contract lists as excluded")]
        if str(status) not in entry:
            declared = sorted(k for k in entry if k[:1].isdigit())
            return [AnswerContractError(key, status, "", f"status {status} is not declared (declared: {declared})")]
        return [AnswerContractError(key, status, _json_path(e), _message(e))
                for e in _leaves(self.validator(key, str(status)).iter_errors(body))]

    def check_other(self, key: str, status: int, media: str | None, size: int = 0) -> list[AnswerContractError]:
        """A 2xx answer that is not JSON, or has no body: the key is `excluded` and declares this status with this
        media type (null: no body at all). A body with no content type is neither (046-2 review round 1, finding 4:
        `DELETE /extractions/{id}`'s queued 202)."""
        served = media if size else None
        row = self.contract["excluded"].get(key)
        if row is None and not size and str(status) in (self.contract["responses"].get(key) or {}).get(
                "x-no-body", ()):
            return []  # a JSON route that answers this status with no content, as it declares
        if row is None:
            what = "no body" if not size else f"a {media} answer"
            return [AnswerContractError(key, status, "", f"{what} from a route the contract lists as JSON")]
        if size and media is None:
            return [AnswerContractError(key, status, "", "a body with no content type")]
        statuses = row.get("statuses", {str(status): row["media_type"]})
        if str(status) not in statuses:
            return [AnswerContractError(key, status, "", f"status {status} is not declared (declared: "
                                                         f"{sorted(statuses)})")]
        if statuses[str(status)] != served:
            return [AnswerContractError(key, status, "", f"served {served!r}, declared {statuses[str(status)]!r}")]
        return []


def _leaves(errors: Any) -> list[Any]:
    """The errors worth naming: inside a union, the deepest branch errors."""
    out: list[Any] = []
    for error in errors:
        inner = _leaves(error.context or [])
        if not inner:
            out.append(error)
            continue
        relevant = [leaf for leaf in inner if not (leaf.validator == "type" and leaf.validator_value == "null")]
        out.append(max(relevant or inner, key=lambda e: len(list(e.absolute_path))))
    return out


def _json_path(error: Any) -> str:
    out = ""
    for part in error.absolute_path:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out


def _message(error: Any) -> str:
    if error.validator == "additionalProperties" and isinstance(error.instance, dict):
        declared = set((error.schema or {}).get("properties", {}))
        extra = sorted(set(error.instance) - declared)
        return f"undeclared key {', '.join(map(repr, extra))} (model {(error.schema or {}).get('title', '?')})"
    return str(error.message)


def contract() -> Contract:
    """The session's contract, built on first use from the committed file."""
    if "contract" not in _STATE:
        from src.api.answers_contract import DEFAULT_PATH

        _STATE["contract"] = Contract(json.loads(DEFAULT_PATH.read_text(encoding="utf-8")))
    found: Contract = _STATE["contract"]
    return found


def endpoint_name(endpoint: Callable[..., Any]) -> str:
    return f"{endpoint.__module__}.{endpoint.__qualname__}"


def endpoint_keys() -> dict[str, dict[str, str]]:
    """`module.qualname` of each real endpoint -> {METHOD: contract key}, from `src.main.app`'s routes. Two endpoints
    with one name, or a route with no endpoint, is an error (FTID §8.1)."""
    if "endpoints" not in _STATE:
        from src.api.answers_contract import operations
        from src.main import app

        out: dict[str, dict[str, str]] = {}
        for method, path, route in operations(app):
            if route.endpoint is None:
                raise LookupError(f"{method} {path} has no endpoint")
            name = endpoint_name(route.endpoint)
            if method in out.setdefault(name, {}) and out[name][method] != f"{method} {path}":
                raise LookupError(f"two {method} routes share the endpoint {name}: {out[name][method]}, {path}")
            out[name][method] = f"{method} {path}"
        _STATE["endpoints"] = out
    found: dict[str, dict[str, str]] = _STATE["endpoints"]
    return found


def key_of(method: str, endpoint: Any) -> str | None:
    """The contract key of a `src.` endpoint's answer to `method`, or None for a test's own route. A `src.` endpoint
    the live app does not serve under `method` gets a key no section holds, so it fails."""
    if endpoint is None or not (getattr(endpoint, "__module__", "") or "").startswith("src."):
        return None
    name = endpoint_name(endpoint)
    return endpoint_keys().get(name, {}).get(method) or f"{method} ? ({name}, not a route of src.main.app)"


def check(method: str, endpoint: Any, status: int, media: str | None, raw: bytes, size: int) -> None:
    """Check one finished answer; record each failure."""
    if not 200 <= status < 300:
        return
    key = key_of(method, endpoint)
    if key is None:
        return
    found = contract()
    section = found.section(key)
    if section == "not_covered":
        return
    if section is None:
        FAILURES.append(AnswerContractError(key, status, "", "the contract does not list this route"))
        return
    CHECKED[key] = CHECKED.get(key, 0) + 1
    if media == JSON and size:
        FAILURES.extend(found.check_json(key, status, json.loads(raw)))
    else:
        FAILURES.extend(found.check_other(key, status, media, size))


class AnswerCheck:
    """Wraps an ASGI app; checks each 2xx answer once its body is complete. Every message is forwarded unchanged and
    at once; only JSON body bytes are copied."""

    def __init__(self, app: Any) -> None:
        self.app = app

    def __getattr__(self, name: str) -> Any:
        return getattr(self.app, name)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        state: dict[str, Any] = {"status": 0, "media": None, "body": bytearray(), "size": 0, "done": False}

        async def watched(message: Any) -> None:
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
                headers = {k.lower(): v for k, v in message.get("headers") or ()}
                media = headers.get(b"content-type", b"").split(b";")[0].strip()
                state["media"] = media.decode("latin-1") or None
            elif message["type"] == "http.response.body":
                chunk = message.get("body", b"")
                state["size"] += len(chunk)
                if state["media"] == JSON:
                    state["body"] += chunk
                if not message.get("more_body", False) and not state["done"]:
                    state["done"] = True
                    check(scope.get("method", ""), scope.get("endpoint"), state["status"], state["media"],
                          bytes(state["body"]), state["size"])
            await send(message)

        await self.app(scope, receive, watched)


ORIGINAL_BUILD: Callable[..., Any] | None = None


def install() -> None:
    """Wrap every FastAPI app's middleware stack in `AnswerCheck` (idempotent)."""
    global ORIGINAL_BUILD
    from fastapi import FastAPI

    if getattr(FastAPI.build_middleware_stack, _INSTALLED, False):
        return
    original = FastAPI.build_middleware_stack
    ORIGINAL_BUILD = original

    def build_middleware_stack(self: Any) -> Any:
        return AnswerCheck(original(self))

    setattr(build_middleware_stack, _INSTALLED, True)
    FastAPI.build_middleware_stack = build_middleware_stack  # type: ignore[method-assign]


def installed() -> bool:
    from fastapi import FastAPI

    return bool(getattr(FastAPI.build_middleware_stack, _INSTALLED, False))


def summary() -> str:
    return f"ANSWER CHECK: {sum(CHECKED.values())} answers of {len(CHECKED)} covered routes checked"


install()
