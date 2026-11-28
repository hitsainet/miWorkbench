"""`GET /probe-monitors/probes/{probe_id}/evaluations/{evaluation_id}/rows` and `get_probe_evaluation_rows`
(ENH-010 FU-7a gap G-1, 2026-11-26).

THE GAP: an evaluation's rows could not be read by position in bulk. `get_probe_evaluation_scores` gives positions
without text, `get_probe_dataset_rows` pages a view by offset, so the 5,000 rows of `pme_9ced7db5a06a`, scattered over
1,000,000 positions, took one call per row.

THE PROPERTY THAT MATTERS: each served text is the text AT the stored position, beside THAT position's score. The
fixture's positions are scattered and not in order of the download's rows (as an evaluation's are after filtering), its
inputs all differ, and its labels alternate — so a text read at the page's offset instead of the stored position, or one
row off, fails here (a fixture whose positions were 0..n-1 would agree with either by construction).

Driven through the REAL app, with the arrays written under `evaluation_array_paths` (the writer's own naming) and the
download written by `save_to_disk`, as production writes both.

Controls (recorded in the hand-back; each restored from saved bytes, sha256-checked):
  delete the tool's registration (`@mcp.tool()` above `get_probe_evaluation_rows`) → reachability, contract and the
      probe-monitor registry tests red
  drop `offset=offset` from the tool's route call → `EXPECTED_CALLS` and `test_the_tool_sends_its_defaults` red
  read `positions` as the page's offsets in the route → the text-at-position tests red
  drop the label check in `inputs_at` → `test_a_row_that_maps_to_another_label_is_refused` red
"""
from __future__ import annotations

import asyncio
import uuid

import numpy as np
import pytest

from src.core.config import settings
from src.services.probe_monitor_run import artifact_dir_for, evaluation_array_paths

datasets = pytest.importorskip("datasets")

COMMIT = "e" * 40
N_DOWNLOAD = 60
LONG = "y" * 25_000


def _download_rows():
    """60 rows, each with a distinct input; label alternates 1/0 by row; row 59's input is over the cell cap."""
    rows = [{"i": f"row {p} says {p * 7}", "l": str(p % 2)} for p in range(N_DOWNLOAD)]
    rows[59]["i"] = LONG
    return rows


#: The evaluation's stored order: scattered, NOT ascending, gaps (filtered rows) — and label = the row's own.
POSITIONS = [41, 3, 17, 58, 0, 29, 12, 59, 33, 8, 50, 21, 46]
LABELS = [p % 2 for p in POSITIONS]


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    return tmp_path


async def _seed(session, tmp_path, *, positions=POSITIONS, labels=LABELS, positions_recorded=True,
                download_rows=None):
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import (
        ProbeMonitor,
        ProbeMonitorDataset,
        ProbeMonitorEvaluation,
        ProbeMonitorRun,
    )

    path = tmp_path / f"download-{uuid.uuid4().hex[:6]}"
    datasets.DatasetDict({"train": datasets.Dataset.from_list(download_rows or _download_rows())}).save_to_disk(
        str(path))
    model = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny", status=ModelStatus.READY,
                  file_path="/nonexistent", architecture="LlamaForCausalLM", params_count=1_000)
    dataset = Dataset(id=uuid.uuid4(), name="corpus", source="HuggingFace", hf_repo_id="org/ds",
                      status=DatasetStatus.READY, raw_path=str(path), extra_metadata={}, revision=COMMIT)
    session.add_all([model, dataset])
    await session.flush()
    view = ProbeMonitorDataset(name="held-out", dataset_id=dataset.id, split="train", input_column="i",
                               label_column="l", label_mapping={"1": "positive", "0": "negative"}, role="eval",
                               counts={})
    session.add(view)
    await session.flush()
    run = ProbeMonitorRun(model_id=model.id, train_dataset_id=view.id, eval_dataset_ids=[view.id],
                          config={}, status="completed", environment={})
    session.add(run)
    await session.flush()
    probe = ProbeMonitor(run_id=run.id, layer=1, rule="mean", rule_params={}, variant="dense",
                         val_metrics={}, selected=True)
    other = ProbeMonitor(run_id=run.id, layer=1, rule="max", rule_params={}, variant="dense",
                         val_metrics={}, selected=False)
    session.add_all([probe, other])
    await session.flush()

    n = len(positions)
    scores = (np.arange(n, dtype=np.float32) * 0.25 - 1.0).astype(np.float32)  # distinct per stored index
    n_scored = (np.arange(n) + 5).astype(np.int32)
    truncated = np.array([1 if i % 4 == 0 else 0 for i in range(n)], dtype=np.int8)
    evaluation = ProbeMonitorEvaluation(
        id=f"pme_{uuid.uuid4().hex[:12]}", probe_id=probe.id, dataset_id=view.id, status="completed",
        metrics={"auroc": 0.7},
    )
    directory = artifact_dir_for(run.id)
    directory.mkdir(parents=True, exist_ok=True)
    paths = evaluation_array_paths(directory, probe.id, view.id, evaluation.id)
    for column, values in (("scores_path", scores), ("positions_path", np.asarray(positions, dtype=np.int64)),
                           ("n_scored_path", n_scored), ("labels_path", np.asarray(labels, dtype=np.int8)),
                           ("truncated_path", truncated)):
        np.save(paths[column], values)
        setattr(evaluation, column, str(paths[column]))
    if not positions_recorded:
        evaluation.positions_path = None
    session.add(evaluation)
    await session.commit()
    return {"probe": probe.id, "other": other.id, "evaluation": evaluation.id, "view": view,
            "dataset": dataset, "scores": scores, "n_scored": n_scored, "truncated": truncated,
            "inputs": [r["i"] for r in (download_rows or _download_rows())]}


def _url(probe, evaluation):
    return f"/api/v1/probe-monitors/probes/{probe}/evaluations/{evaluation}/rows"


async def _all_pages(client, probe, evaluation, limit):
    rows, offset, pages = [], 0, 0
    while offset is not None:
        r = await client.get(_url(probe, evaluation), params={"offset": offset, "limit": limit})
        assert r.status_code == 200, r.text
        body = r.json()
        rows += body["rows"]
        offset = body["next_offset"]
        pages += 1
    return rows, body, pages


@pytest.mark.asyncio
class TestEachTextIsTheTextAtItsStoredPosition:
    async def test_pages_concatenate_to_the_stored_rows_with_their_texts(self, client, async_session, tmp_path):
        seeded = await _seed(async_session, tmp_path)
        rows, last, pages = await _all_pages(client, seeded["probe"], seeded["evaluation"], limit=5)
        assert pages == 3  # 5 + 5 + 3
        assert [r["position"] for r in rows] == POSITIONS
        assert [r["label"] for r in rows] == LABELS
        assert np.allclose([r["score"] for r in rows], seeded["scores"])
        assert [r["n_scored"] for r in rows] == seeded["n_scored"].tolist()
        assert [r["truncated"] for r in rows] == [bool(t) for t in seeded["truncated"]]
        expected_text = [seeded["inputs"][p] if p != 59 else LONG[:20_000] for p in POSITIONS]
        assert [r["text"] for r in rows] == expected_text
        assert last["total"] == len(POSITIONS) and last["next_offset"] is None

    async def test_the_rows_match_the_scores_route_row_for_row(self, client, async_session, tmp_path):
        """The score half is the 035 scores route's own slice — not a second reading that could drift."""
        seeded = await _seed(async_session, tmp_path)
        scores = (await client.get(
            f"/api/v1/probe-monitors/probes/{seeded['probe']}/evaluations/{seeded['evaluation']}/scores",
            params={"offset": 4, "limit": 6})).json()
        rows = (await client.get(_url(seeded["probe"], seeded["evaluation"]),
                                 params={"offset": 4, "limit": 6})).json()
        assert [{k: r[k] for k in scores["rows"][0]} for r in rows["rows"]] == scores["rows"]
        assert rows["next_offset"] == scores["next_offset"] == 10

    async def test_a_cell_over_the_cap_is_cut_and_says_so(self, client, async_session, tmp_path):
        seeded = await _seed(async_session, tmp_path)
        body = (await client.get(_url(seeded["probe"], seeded["evaluation"]), params={"offset": 7, "limit": 1})).json()
        (row,) = body["rows"]
        assert row["position"] == 59 and row["text_cut"] is True and len(row["text"]) == 20_000
        other = (await client.get(_url(seeded["probe"], seeded["evaluation"]), params={"limit": 1})).json()
        assert other["rows"][0]["text_cut"] is False

    async def test_a_conversation_input_comes_back_as_stored(self, client, async_session, tmp_path):
        chat = [{"i": [{"role": "user", "content": f"q{p}"}, {"role": "assistant", "content": f"a{p}"}],
                 "l": str(p % 2)} for p in range(4)]
        seeded = await _seed(async_session, tmp_path, positions=[2, 1], labels=[0, 1], download_rows=chat)
        body = (await client.get(_url(seeded["probe"], seeded["evaluation"]))).json()
        assert [r["text"] for r in body["rows"]] == [chat[2]["i"], chat[1]["i"]]


@pytest.mark.asyncio
class TestTheEnvelope:
    async def test_it_names_its_sources_and_adds_only_text(self, client, async_session, tmp_path):
        seeded = await _seed(async_session, tmp_path)
        body = (await client.get(_url(seeded["probe"], seeded["evaluation"]))).json()
        assert body["evaluation_id"] == seeded["evaluation"] and body["probe_id"] == seeded["probe"]
        assert body["dataset_id"] == seeded["view"].id
        assert body["source_dataset_id"] == str(seeded["dataset"].id) and body["revision"] == COMMIT
        assert body["status"] == "completed" and body["input_column"] == "i"
        assert body["offset"] == 0 and body["limit"] == 100 and body["total"] == len(POSITIONS)
        assert set(body["rows"][0]) == {"position", "score", "label", "n_scored", "truncated", "text", "text_cut"}

    async def test_the_page_is_bounded(self, client, async_session, tmp_path):
        seeded = await _seed(async_session, tmp_path)
        url = _url(seeded["probe"], seeded["evaluation"])
        assert (await client.get(url, params={"limit": 200})).status_code == 200
        assert (await client.get(url, params={"limit": 201})).status_code == 422
        assert (await client.get(url, params={"limit": 0})).status_code == 422
        past = (await client.get(url, params={"offset": 500})).json()
        assert past["rows"] == [] and past["next_offset"] is None


@pytest.mark.asyncio
class TestRefusals:
    async def test_another_probes_evaluation_is_404(self, client, async_session, tmp_path):
        seeded = await _seed(async_session, tmp_path)
        assert (await client.get(_url(seeded["other"], seeded["evaluation"]))).status_code == 404

    async def test_no_recorded_positions_is_409(self, client, async_session, tmp_path):
        seeded = await _seed(async_session, tmp_path, positions_recorded=False)
        r = await client.get(_url(seeded["probe"], seeded["evaluation"]))
        assert r.status_code == 409 and r.json()["detail"]["code"] == "positions_not_recorded"

    async def test_a_position_past_the_download_is_refused(self, client, async_session, tmp_path):
        seeded = await _seed(async_session, tmp_path, positions=[3, N_DOWNLOAD + 4], labels=[1, 0])
        r = await client.get(_url(seeded["probe"], seeded["evaluation"]))
        assert r.status_code == 409, r.text
        detail = r.json()["detail"]
        assert detail["code"] == "rows_moved" and detail["out_of_range"] == [N_DOWNLOAD + 4]
        assert "rows" not in r.json()

    async def test_a_row_that_maps_to_another_label_is_refused(self, client, async_session, tmp_path):
        """Position 3's row maps to positive (label "1"); the evaluation stored negative there — the download is not
        the one scored, and serving its text beside that score would pair the wrong row with it."""
        seeded = await _seed(async_session, tmp_path, positions=[2, 3, 4], labels=[0, 0, 0])
        r = await client.get(_url(seeded["probe"], seeded["evaluation"]))
        assert r.status_code == 409, r.text
        assert r.json()["detail"]["code"] == "rows_moved" and r.json()["detail"]["disagree"] == [3]

    async def test_a_purged_view_takes_its_evaluation_with_it(self, client, async_session, tmp_path):
        """The evaluation's view (and the view's download) are CASCADE foreign keys, so a purge leaves no
        evaluation to page: 404, never a page of scores with no text. (`view_gone` / `download_gone` in the route
        are the defensive branches for a row the database no longer joins.)"""
        seeded = await _seed(async_session, tmp_path)
        await async_session.delete(seeded["view"])
        await async_session.commit()
        r = await client.get(_url(seeded["probe"], seeded["evaluation"]))
        assert r.status_code == 404

    async def test_the_download_gone_from_disk_is_409(self, client, async_session, tmp_path):
        import shutil

        seeded = await _seed(async_session, tmp_path)
        shutil.rmtree(seeded["dataset"].raw_path)
        r = await client.get(_url(seeded["probe"], seeded["evaluation"]))
        assert r.status_code == 409 and r.json()["detail"]["code"] == "download_gone"


class TestTheService:
    """`inputs_at` alone, so its bound and its order hold for any caller."""

    def _spec(self):
        from src.services.probe_dataset_rows import ViewSpec

        return ViewSpec(id="pmd_x", split="train", input_column="i", label_column="l", pair_column=None,
                        label_mapping={"1": "positive", "0": "negative"}, keyword_filter=None)

    def _path(self, tmp_path):
        path = tmp_path / "svc"
        datasets.DatasetDict({"train": datasets.Dataset.from_list(_download_rows())}).save_to_disk(str(path))
        return path

    def test_order_is_the_order_given(self, tmp_path):
        from src.services.probe_dataset_rows import inputs_at

        out = inputs_at(self._spec(), self._path(tmp_path), [9, 2, 5], [1, 0, 1])
        assert [r["text"] for r in out] == ["row 9 says 63", "row 2 says 14", "row 5 says 35"]

    def test_more_than_a_page_is_refused(self, tmp_path):
        from src.services.probe_dataset_rows import MAX_PAGE_SIZE, RowsRefused, inputs_at

        with pytest.raises(RowsRefused):
            inputs_at(self._spec(), self._path(tmp_path), [0] * (MAX_PAGE_SIZE + 1), [0] * (MAX_PAGE_SIZE + 1))


def test_the_tool_sends_its_defaults():
    """PAYLOAD AND COUNT with only the two ids: one GET, the route's first page at the tool's default bound."""
    from tests.unit.test_reachability import _tools_by_name

    manager, client = _tools_by_name()
    asyncio.run(manager.call_tool("get_probe_evaluation_rows",
                                  {"probe_id": "pm_1", "evaluation_id": "pme_1"}, context=None))
    assert client.calls == [("GET", "/probe-monitors/probes/pm_1/evaluations/pme_1/rows",
                             {"offset": 0, "limit": 100})]
