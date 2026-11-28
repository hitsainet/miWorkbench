"""`docs/schemas/miworkbench-answers-v1.json` is what the live routes declare (046 FR-046.1, .2, .14; FTASKS 6.3).

The contract is GENERATED (`src/api/answers_contract.py`) from `src.main.app`'s OpenAPI document and the live MCP
registry, so it moves when a route, an answer model or a tool's route moves, and this test goes red until the
committed file is regenerated (`MIWORKBENCH_WRITE_CONTRACTS=1 pytest tests/unit/test_answers_contract.py`). The writer
refuses a breaking change (FR-046.13): that needs `miworkbench-answers-v2.json` beside v1. Regeneration is the moment a
consumer's copy (miAssist, `tests/unit/test_answers_contract_consumers.py`) must move too.

Controls (0xcc/reviews/046_controls_<date>.md): M16 (a field added to one covered model without regenerating), and a
timestamp in `description` (the determinism test).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.api import contract_tools
from src.api.answers_contract import (
    CONTRACT_FILENAME,
    CONTRACT_ID,
    DEFAULT_PATH,
    KIND,
    answers_contract,
    body_of,
    problems,
    render,
    write_contract,
)
from tests.support.cross_repo import test_env

BACKEND = Path(__file__).resolve().parents[2]
CONTRACT = DEFAULT_PATH
REGENERATE = "cd backend && MIWORKBENCH_WRITE_CONTRACTS=1 pytest tests/unit/test_answers_contract.py"


@pytest.fixture(scope="module")
def generated() -> dict:
    body = answers_contract()
    if test_env("MIWORKBENCH_WRITE_CONTRACTS", "MISTUDIO_WRITE_CONTRACTS") == "1":
        write_contract(CONTRACT, body)
    return body


@pytest.fixture(scope="module")
def committed(generated: dict) -> dict:  # noqa: ARG001 (written first, when asked)
    assert CONTRACT.is_file(), f"{CONTRACT} is missing; run {REGENERATE}"
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_the_committed_contract_is_the_generated_one(generated: dict, committed: dict):
    """FR-046.2: the committed file equals the rendered contract of the live app and registry."""
    assert body_of(committed) == generated, (
        f"{CONTRACT.name} no longer matches the registered routes and tools. Regenerate it ({REGENERATE}); then "
        "miAssist re-vendors it.")


def test_the_committed_bytes_are_the_rendered_form(committed: dict):
    assert CONTRACT.read_text(encoding="utf-8") == render(committed), (
        f"{CONTRACT.name} is not in its rendered form (sorted keys, two-space indent, newline): {REGENERATE}")


def test_its_digest_and_revision_hold(committed: dict):
    """FR-046.14 (TD11): the embedded digest is recomputed; a hand edit without the writer goes red."""
    assert problems(committed) == []


def test_a_hand_edit_without_a_revision_change_is_caught(committed: dict):
    """FR-046.14 AC: one schema edited by hand, `revision` and `content_sha256` left as they were."""
    edited = json.loads(json.dumps(committed))
    name = sorted(edited["$defs"])[0]
    edited["$defs"][name]["description"] = "edited by hand"
    assert any("content_sha256" in p for p in problems(edited)), problems(edited)


def test_rendering_is_deterministic_across_hash_seeds():
    """FR-046.1 AC: two processes with different hash seeds render byte-identical text."""
    code = ("import hashlib, sys\n"
            "from src.api.answers_contract import answers_contract, render\n"
            "sys.stdout.write('DIGEST=' + hashlib.sha256(render(answers_contract()).encode()).hexdigest())\n")
    digests = []
    for seed in ("1", "2"):
        proc = subprocess.run([sys.executable, "-c", code], cwd=BACKEND, env={**os.environ, "PYTHONHASHSEED": seed},
                              capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, proc.stderr[-3000:]
        digests.append(proc.stdout.rsplit("DIGEST=", 1)[1].strip())
    assert digests[0] == digests[1]
    assert len(digests[0]) == 64


def test_its_identity(committed: dict):
    """FR-046.1 AC: the `$id`, `kind` and file name exactly; no new name carries an old product name."""
    assert CONTRACT_FILENAME == "miworkbench-answers-v1.json"
    assert committed["$id"] == CONTRACT_ID == "https://hitsai.net/schemas/miworkbench-answers-v1.json"
    assert committed["kind"] == KIND == "miworkbench.answer-contract/v1"
    assert isinstance(committed["revision"], int) and committed["revision"] >= 1
    names = [CONTRACT_FILENAME, KIND, CONTRACT_ID, committed["title"], *committed["$defs"], *committed["errors"]]
    old = [n for n in names if "mistudio" in n.lower() or "millm" in n.lower()]
    assert not old, old


def test_the_file_name_is_written_in_the_writer_only():
    """FR-046.1: `CONTRACT_FILENAME` is the one place the name is written in `src/`. (The test-side fake check names
    the path it reads, which is outside `src/`.)"""
    hits = sorted(p.relative_to(BACKEND).as_posix() for p in (BACKEND / "src").rglob("*.py")
                  if CONTRACT_FILENAME in p.read_text(encoding="utf-8"))
    assert hits == ["src/api/answers_contract.py"], hits


def test_the_digest_is_the_one_contract_tools_computes(committed: dict):
    body = {k: v for k, v in committed.items() if k not in ("revision", "content_sha256")}
    expected = hashlib.sha256(render(body).encode("utf-8")).hexdigest()
    assert committed["content_sha256"] == expected == contract_tools.content_digest(committed)


def test_the_writer_does_not_transform_the_served_document():
    """FTID §3.3: FastAPI caches the served document on `app.openapi_schema`; building the contract leaves it as
    it was (a `$ref` still names `#/components/schemas/`, and no `required` was rewritten)."""
    from src.main import app

    before = json.dumps(app.openapi(), sort_keys=True)
    answers_contract()
    assert json.dumps(app.openapi(), sort_keys=True) == before


def test_external_names_only_the_documents_an_answer_reaches():
    """Review round 1 (A-F6), pinned in round 2 (R9 survived): a document referenced only by a component no covered
    answer reaches (here, a not-covered route's model) is not pinned in `external`; the covered one is."""
    from typing import Any

    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    from src.api.answers_contract import contract_from
    from src.schemas.jlens_answers import JSPACE_CONTRACTS_ID

    circuit = json.loads((BACKEND.parent / "docs" / "schemas" / "circuit-definition-v1.json").read_text())["$id"]

    class Covered(BaseModel):
        document: dict[str, Any] = Field(json_schema_extra={"$ref": JSPACE_CONTRACTS_ID})

    class Uncovered(BaseModel):
        document: dict[str, Any] = Field(json_schema_extra={"$ref": circuit})

    app = FastAPI()

    @app.get("/api/v1/a", responses={200: {"model": Covered}})
    async def a(q: int = 0):  # a parameter, so the app declares the 422 the writer requires
        return {}

    @app.get("/api/v1/b", responses={200: {"model": Uncovered}})
    async def b():
        return {}

    body = contract_from(app, app.openapi(), frozenset({"GET /api/v1/a"}))
    assert set(body["external"]) == {JSPACE_CONTRACTS_ID}, body["external"]
