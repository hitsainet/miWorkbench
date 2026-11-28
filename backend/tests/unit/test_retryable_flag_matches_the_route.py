"""The listing's `retryable` flag is exactly what `POST /task-queue/{id}/retry` accepts, for every task type (2026-11-27).

Failed Operations offered Retry on every task-queue row that was not a wedged-worker report, and the retry route
refused every pair it has no dispatch branch for with a 400: every J-Lens task, every probe definition build, and
since the probe-progress work every `probe_evaluation` and `wsp_heldout_capture`. Setting `can_retry` false would also
have hidden Delete, so the rows now carry `retryable`, computed by `task_queue.retry_refusal`, the route's own check.

How the cases are chosen, so a new task type cannot slip past by not being listed here:
* the ACCEPTED pairs are read out of `retry_task`'s own dispatch chain (its AST), never typed in this file;
* the REFUSED side is every `task_type` the backend writes: each `task_type="..."` keyword and `*TASK_TYPE` constant
  in `src/`, and the task-queue types `jlens_progress` declares, each under every entity type the dispatch uses plus
  `probe_monitor`; plus a dispatch task type under a foreign entity, and a tokenization row with no `model_id`.
For each row the REAL route is called (Celery `.delay` patched and recorded), and the flag must equal "the route
answered 2xx". A refused row must still be `failed` with its error message (MIS-E2E-098).

Controls (0xcc/reviews/eval_card_and_retry_2026-11-27.md):
  R1  `_serialize_task` sends `retryable: can_retry` again                       -> red (J-Lens / probe rows)
  R2  `retry_refusal` drops the RETRYABLE_TASK_TYPES check (route keeps its else) -> red
  R3  `retry_refusal` drops the tokenization model_id check                      -> red (flag true, route 422)
  R4  the route raises the model_id 422 after `increment_retry_count` again      -> red (evidence wiped)
  R5  `get_task` builds its own dict again (no gpu_uuids / retryable)            -> the single-row test
"""
from __future__ import annotations

import ast
import inspect
import pathlib
import re
import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.models.task_queue import TaskQueue

pytestmark = pytest.mark.asyncio

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"


def dispatch_pairs() -> set:
    """`(task_type, entity_type)` of every branch of `retry_task`'s dispatch chain, from its AST."""
    from src.api.v1.endpoints import task_queue

    tree = ast.parse(inspect.getsource(task_queue))
    body = next(n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "retry_task")
    pairs = set()
    for node in ast.walk(body):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            found = {}
            for cmp_node in node.values:
                if (isinstance(cmp_node, ast.Compare) and isinstance(cmp_node.left, ast.Attribute)
                        and isinstance(cmp_node.comparators[0], ast.Constant)):
                    found[cmp_node.left.attr] = cmp_node.comparators[0].value
            if "task_type" in found and "entity_type" in found:
                pairs.add((found["task_type"], found["entity_type"]))
    assert pairs, "no dispatch branch found in retry_task: the scan broke"
    return pairs


def written_task_types() -> set:
    """Every task type the backend writes into `task_queue` (or federates into the listing)."""
    from src.workers import jlens_progress

    types_ = set()
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.keyword) and node.arg == "task_type"
                    and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                types_.add(node.value.value)
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id.endswith("TASK_TYPE") and isinstance(node.value, ast.Constant)):
                types_.add(node.value.value)
    for name, value in vars(jlens_progress).items():
        if name.isupper() and isinstance(value, str) and re.fullmatch(r"(jlens|probe|wsp)_[a-z_]+", value):
            types_.add(value)
    # The scan must have seen the types this fix is about; a scan that finds nothing asserts nothing.
    for must in ("probe_evaluation", "wsp_heldout_capture", "probe_definition_build", "jlens_fit",
                 "gpu_worker_wedged", "download", "tokenization"):
        assert must in types_, f"the task-type scan missed {must!r}: {sorted(types_)}"
    return types_


def cases() -> list:
    accepted = dispatch_pairs()
    entities = sorted({e for _t, e in accepted} | {"probe_monitor"})
    out = []
    for task_type in sorted(written_task_types()):
        for entity in entities:
            params = {"repo_id": "org/x", "model_id": "m_x"} if task_type == "tokenization" else {"repo_id": "org/x"}
            out.append((task_type, entity, params))
    # A tokenization row written before model_id was recorded: the route refuses it (422).
    out.append(("tokenization", "dataset", {"repo_id": "org/x"}))
    # Every accepted pair is in, whatever the scan found.
    for task_type, entity in sorted(accepted):
        params = {"repo_id": "org/x", "model_id": "m_x"} if task_type == "tokenization" else {"repo_id": "org/x"}
        if (task_type, entity, params) not in out:
            out.append((task_type, entity, params))
    return out


async def test_the_flag_equals_what_the_retry_route_accepts_for_every_task_type(client, async_session):
    rows = {}
    for task_type, entity, params in cases():
        row = TaskQueue(id=f"tq_{uuid.uuid4().hex[:12]}", task_id=f"old-{uuid.uuid4().hex[:8]}",
                        task_type=task_type, entity_id=str(uuid.uuid4()), entity_type=entity, status="failed",
                        progress=12.0, error_message=f"boom {task_type}", retry_params=dict(params), retry_count=0)
        async_session.add(row)
        rows[row.id] = (task_type, entity, params)
    await async_session.commit()

    listed = (await client.get("/api/v1/task-queue/failed")).json()["data"]
    flags = {t["id"]: t for t in listed if t["id"] in rows}
    assert set(flags) == set(rows), "a failed task-queue row is missing from /failed"

    accepted_pairs = dispatch_pairs()
    sends = {}
    with patch("src.workers.model_tasks.download_and_load_model.delay") as model_dl, \
            patch("src.workers.dataset_tasks.download_dataset_task.delay") as dataset_dl, \
            patch("src.workers.dataset_tasks.tokenize_dataset_task.delay") as tokenize:
        for mock in (model_dl, dataset_dl, tokenize):
            mock.return_value = MagicMock(id=f"new-{uuid.uuid4().hex[:8]}")
        disagree = []
        for row_id, (task_type, entity, params) in rows.items():
            before = model_dl.call_count + dataset_dl.call_count + tokenize.call_count
            response = await client.post(f"/api/v1/task-queue/{row_id}/retry", json={})
            sent = model_dl.call_count + dataset_dl.call_count + tokenize.call_count - before
            ok = 200 <= response.status_code < 300
            sends[row_id] = sent
            if flags[row_id]["retryable"] is not ok:
                disagree.append((task_type, entity, sorted(params), flags[row_id]["retryable"],
                                 response.status_code, response.text[:160]))
            assert sent == (1 if ok else 0), (task_type, entity, response.status_code, sent)
            # The flag's own sanity: it is true only for a pair the dispatch chain names.
            if flags[row_id]["retryable"]:
                assert (task_type, entity) in accepted_pairs
            # Delete is still offered for every row this API owns, except the wedged-worker report.
            assert flags[row_id]["can_retry"] is (task_type != "gpu_worker_wedged"), (task_type, flags[row_id])
    assert not disagree, "the retryable flag and the retry route disagree:\n" + "\n".join(map(str, disagree))

    # Both sides were exercised: some rows retried, the four this fix names refused.
    retried = {rows[r][:2] for r, n in sends.items() if n == 1}
    assert retried == accepted_pairs, (retried, accepted_pairs)
    refused_types = {rows[r][0] for r, t in flags.items() if not t["retryable"]}
    assert {"probe_evaluation", "wsp_heldout_capture", "probe_definition_build", "jlens_fit"} <= refused_types

    # A tokenization row with no recorded model_id is refused (422: the worker would die on a TypeError) and is
    # not offered; the one that recorded its model is retried.
    for row_id, (task_type, entity, params) in rows.items():
        if (task_type, entity) == ("tokenization", "dataset"):
            assert flags[row_id]["retryable"] is bool(params.get("model_id")), (params, flags[row_id])
            assert sends[row_id] == (1 if params.get("model_id") else 0), params

    # A refused row keeps its failure evidence (MIS-E2E-098), including the tokenization 422.
    async_session.expire_all()
    for row_id, (task_type, entity, params) in rows.items():
        if sends[row_id]:
            continue
        row = await async_session.get(TaskQueue, row_id)
        assert (row.status, row.error_message, row.retry_count) == ("failed", f"boom {task_type}", 0), (
            task_type, entity, params, row.status, row.error_message)


async def test_one_row_answers_its_cards_and_its_flag(client, async_session):
    """`GET /task-queue/{id}` (what `get_task_status` reads) names every card a job used, and the same flag."""
    split = ["GPU-aaaaaaaa-0000-0000-0000-00000000000a", "GPU-bbbbbbbb-0000-0000-0000-00000000000b"]
    row = TaskQueue(id=f"tq_{uuid.uuid4().hex[:12]}", task_id="c1", task_type="probe_evaluation",
                    entity_id="pm_x", entity_type="probe_monitor", status="failed", progress=40.0,
                    error_message="RuntimeError: boom", retry_params={}, retry_count=0,
                    gpu_uuid=split[0], gpu_uuids=split)
    ok = TaskQueue(id=f"tq_{uuid.uuid4().hex[:12]}", task_id="c2", task_type="download", entity_id="m_x",
                   entity_type="model", status="failed", progress=0.0, error_message="reset",
                   retry_params={"repo_id": "org/x"}, retry_count=0)
    async_session.add_all([row, ok])
    await async_session.commit()

    data = (await client.get(f"/api/v1/task-queue/{row.id}")).json()["data"]
    assert (data["gpu_uuid"], data["gpu_uuids"], data["retryable"]) == (split[0], split, False), data
    assert (await client.get(f"/api/v1/task-queue/{ok.id}")).json()["data"]["retryable"] is True
