"""ENH-010 FU-7a: read a DOWNLOADED dataset's rows with no probe dataset view.

FU-7a had to register a new set as a view to read its rows (to prove it disjoint), then archive the view when it
failed. `get_dataset_rows` (category `read`, beside `get_dataset`) -> `GET /datasets/{id}/rows`
(`DatasetRows`) -> `probe_dataset_rows.download_rows`: positions in the split, at most 200 rows a page (the bound of
`get_probe_evaluation_rows`), each cell cut at 20,000 characters and named in `truncated`, from the files on disk only.

Why a new tool and not an option on `get_dataset`: that tool answers `DatasetResponse` (typed from its route); an
option would give one tool two answer shapes, a page bound and failures (missing files, splits) a metadata read never
has. The UI's `/samples` route is not reused either: page numbers, no cell cap, a Hub fallback and token decoding.

Controls (recorded in the commit): the tool's `@mcp.tool()` removed -> red (live registry, payload); the page bound
loosened at the tool, the route and the service, each -> red here.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest

from src.core.config import settings
from src.services import probe_dataset_rows as R

LONG = "x" * (R.MAX_CELL_CHARS + 500)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    return tmp_path


def _one_split(root, rows=450):
    from tests.support.detector_fixtures import write_split

    texts = [f"row {i}" for i in range(rows)]
    texts[3] = LONG
    return write_split(root, {"text": texts, "kind": [f"k{i % 7}" for i in range(rows)],
                              "meta": [{"i": i, "blob": LONG if i == 5 else ""} for i in range(rows)]})


def _two_splits(root):
    from datasets import Dataset as HF
    from datasets import DatasetDict

    path = root / "datasets" / f"dd_{uuid.uuid4().hex[:8]}"
    path.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": HF.from_dict({"text": ["a", "b", "c"]}),
                 "test": HF.from_dict({"text": ["t0", "t1"]})}).save_to_disk(str(path))
    return str(path)


# ── the service ──────────────────────────────────────────────────────────────────────────────────


class TestDownloadRows:
    def test_a_page_by_position_with_cut_cells(self, tmp_path):
        page = R.download_rows(_one_split(tmp_path), offset=0, limit=10)
        assert page["split"] is None and page["splits"] == [] and page["total"] == 450
        assert page["columns"] == page["returned_columns"] == ["text", "kind", "meta"]
        assert [r["position"] for r in page["rows"]] == list(range(10)) and page["next_offset"] == 10
        assert page["rows"][3]["values"]["text"] == LONG[:R.MAX_CELL_CHARS] and page["rows"][3]["truncated"] == ["text"]
        assert page["rows"][5]["truncated"] == ["meta"] and page["rows"][0]["truncated"] == []
        assert page["max_cell_chars"] == 20_000

    def test_paging_to_the_end(self, tmp_path):
        path = _one_split(tmp_path)
        page = R.download_rows(path, offset=400, limit=200)
        assert len(page["rows"]) == 50 and page["next_offset"] is None
        assert R.download_rows(path, offset=999)["rows"] == [] and R.download_rows(path, offset=999)["next_offset"] is None

    def test_the_page_bound_is_200(self, tmp_path):
        path = _one_split(tmp_path)
        assert len(R.download_rows(path, limit=200)["rows"]) == 200
        with pytest.raises(R.RowsRefused, match="200"):
            R.download_rows(path, limit=201)

    def test_columns_project_and_unknown_ones_are_refused(self, tmp_path):
        path = _one_split(tmp_path)
        page = R.download_rows(path, limit=2, columns=["kind"])
        assert page["returned_columns"] == ["kind"] and page["rows"][0]["values"] == {"kind": "k0"}
        with pytest.raises(R.RowsRefused, match="nope"):
            R.download_rows(path, columns=["nope"])

    def test_several_splits_need_one_named(self, tmp_path):
        path = _two_splits(tmp_path)
        with pytest.raises(R.RowsRefused, match="several splits"):
            R.download_rows(path)
        page = R.download_rows(path, split="test")
        assert page["split"] == "test" and sorted(page["splits"]) == ["test", "train"]
        assert [r["values"]["text"] for r in page["rows"]] == ["t0", "t1"]
        with pytest.raises(R.RowsRefused, match="validation"):
            R.download_rows(path, split="validation")


# ── the route ────────────────────────────────────────────────────────────────────────────────────


class TestTheRoute:
    async def test_reads_a_download_with_no_view(self, client, async_session, data_dir):
        from sqlalchemy import func, select

        from src.models.probe_monitor import ProbeMonitorDataset
        from tests.support.detector_fixtures import dataset_row

        ds = await dataset_row(async_session, _one_split(data_dir))
        await async_session.commit()
        views = (await async_session.execute(select(func.count()).select_from(ProbeMonitorDataset))).scalar_one()
        response = await client.get(f"/api/v1/datasets/{ds.id}/rows",
                                    params={"offset": 2, "limit": 3, "columns": "text,kind"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["dataset_id"] == str(ds.id) and body["total"] == 450 and body["next_offset"] == 5
        assert body["returned_columns"] == ["text", "kind"] and [r["position"] for r in body["rows"]] == [2, 3, 4]
        assert body["rows"][1]["truncated"] == ["text"] and len(body["rows"][1]["values"]["text"]) == 20_000
        after = (await async_session.execute(select(func.count()).select_from(ProbeMonitorDataset))).scalar_one()
        assert after == views                                                # nothing registered

    async def test_the_route_bounds_the_page_itself(self, client, async_session, data_dir):
        from tests.support.detector_fixtures import dataset_row

        ds = await dataset_row(async_session, _one_split(data_dir))
        await async_session.commit()
        ok = await client.get(f"/api/v1/datasets/{ds.id}/rows", params={"limit": 200})
        assert ok.status_code == 200 and len(ok.json()["rows"]) == 200
        over = await client.get(f"/api/v1/datasets/{ds.id}/rows", params={"limit": 201})
        assert over.status_code == 422
        # FastAPI's own query validation, not the service's refusal: the route's bound holds on its own.
        assert isinstance(over.json()["detail"], list) and over.json()["detail"][0]["loc"] == ["query", "limit"]

    async def test_refusals(self, client, async_session, data_dir):
        from tests.support.detector_fixtures import dataset_row

        assert (await client.get(f"/api/v1/datasets/{uuid.uuid4()}/rows")).status_code == 404
        multi = await dataset_row(async_session, _two_splits(data_dir))
        gone = await dataset_row(async_session, str(data_dir / "datasets" / "never_written"))
        await async_session.commit()
        several = await client.get(f"/api/v1/datasets/{multi.id}/rows")
        assert several.status_code == 422 and "several splits" in several.text
        assert (await client.get(f"/api/v1/datasets/{multi.id}/rows", params={"split": "train"})).json()["total"] == 3
        missing = await client.get(f"/api/v1/datasets/{gone.id}/rows")
        assert missing.status_code == 409 and missing.json()["detail"]["code"] == "rows_not_on_disk"

    async def test_the_route_is_in_the_live_app_and_typed(self, client):
        spec = (await client.get("/api/openapi.json")).json()
        route = spec["paths"]["/api/v1/datasets/{dataset_id}/rows"]["get"]
        assert route["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("/DatasetRows")


# ── the MCP tool ─────────────────────────────────────────────────────────────────────────────────


def _calls(arguments):
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.tools import discovery
    from tests.unit.test_reachability import RecordingClient

    mcp = MCPServer("t")
    client = RecordingClient()
    discovery.register(mcp, client, MCPSettings(auth_token="x" * 32))
    asyncio.run(mcp.call_tool("get_dataset_rows", arguments))
    return client.calls


def test_the_tool_sends_its_defaults_once():
    assert _calls({"dataset_id": "d-1"}) == [("GET", "/datasets/d-1/rows", {"offset": 0, "limit": 100})]


def test_the_tool_bounds_the_page_and_calls_nothing_past_it():
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.tools import discovery
    from tests.unit.test_reachability import RecordingClient

    mcp = MCPServer("t")
    client = RecordingClient()
    discovery.register(mcp, client, MCPSettings(auth_token="x" * 32))
    with pytest.raises(ToolError, match="limit"):
        asyncio.run(mcp.call_tool("get_dataset_rows", {"dataset_id": "d-1", "limit": 201}))
    assert client.calls == []
    asyncio.run(mcp.call_tool("get_dataset_rows", {"dataset_id": "d-1", "limit": 200}))
    assert client.calls == [("GET", "/datasets/d-1/rows", {"offset": 0, "limit": 200})]


def test_the_tool_is_in_the_live_registry_with_a_typed_answer():
    from src.mcp_server.tool_classes import TOOL_CLASSES
    from tests.support.next_calls import live_input_schemas

    schema = live_input_schemas()["get_dataset_rows"]
    assert set(schema["properties"]) == {"dataset_id", "split", "offset", "limit", "columns"}
    assert schema["properties"]["limit"]["maximum"] == 200
    assert TOOL_CLASSES["get_dataset_rows"].kind == "read"
    import json
    from pathlib import Path

    import src.mcp_server as pkg

    typed = json.loads((Path(pkg.__file__).parent / "output_schemas.json").read_text())["tools"]
    # 046 (ENH-013) advertises a typed answer as `anyOf` the route's answer and the shared alternatives (approval,
    # refusal); the route's alternative carries the rows (train W2c, 2026-11-26).
    schema = typed["get_dataset_rows"]
    alternatives = schema.get("anyOf") or [schema]
    assert any("rows" in a.get("properties", {}) for a in alternatives), alternatives
