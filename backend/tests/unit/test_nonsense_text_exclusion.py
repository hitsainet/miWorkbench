"""FU-7a (operator, 2026-11-25): a fresh nonsense-text set shares NO row with earlier sets.

`create_nonsense_text_dataset(exclude_view_ids=[...])` → `POST /probe-monitors/nonsense-text` →
`nonsense_text.generate(..., exclude=<text keys>)`. The key is FU-7a C1's own (`wsp_bundle.text_key`: sha256 of the
RFC 8785 bytes of the text after NFC, CRLF→LF and strip), the key `heldout_reused` compares, so a set made here is
one C1 accepts as unused. The generator's short rows repeat across seeds (`!`, `0`, `typeof`, `console.log`,
`###`, `N N N N`, one shuffled sentence): without exclusion seed 1 shares rows with seed 0. With it, every excluded
candidate is drawn again, so the set has EXACTLY the rows asked for, in the same kind shares; a kind that cannot
reach its quota fails the call naming the kind, and no dataset is written. Gap G-2: the manifest records the prose
corpus (id, column, documents read), the excluded views and the candidates dropped for them.

Wiring controls (recorded in the commit): the tool's pass-through of `exclude_view_ids` removed -> red here; the
route's `exclude=excluded` removed -> red here. The tool's payload is asserted HERE, not in test_reachability's
`EXPECTED_CALLS`: that entry is pinned by the P3 cut's digest (`test_mcp_cut.py`), which records the payloads the cut
left unchanged, so it keeps its pre-cut arguments.
"""
from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import unicodedata

import pytest
import rfc8785

from src.core.config import settings
from src.services import nonsense_text as N
from src.services.wsp_bundle import text_key

API = "/api/v1/probe-monitors"
PROSE = [f"The {w} river ran past the old mill at dawn. Number {i} 123-45-6789 was on a card. "
         f"Then 987654321 was read aloud by the {w} keeper!" for i, w in enumerate(N.COMMON_WORDS[:60])]


def _independent_key(value: str) -> str:
    """The key as FU-7a states it, written out here so a change to the shared function cannot pass unnoticed."""
    norm = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n").strip()
    return hashlib.sha256(bytes(rfc8785.dumps(norm))).hexdigest()


def _shared(texts, keys) -> int:
    return sum(1 for t in texts if _independent_key(t) in keys)


# ── the key ──────────────────────────────────────────────────────────────────────────────────────


class TestTheKeyIsC1s:
    def test_text_keys_is_wsp_bundle_text_key(self):
        values = ["typeof", "  typeof\r\n", "é", "é", "N N N N"]
        assert N.text_keys(values) == {text_key(v) for v in values} == {_independent_key(v) for v in values}

    def test_the_key_normalises_as_c1_does(self):
        assert N.text_keys(["typeof"]) == N.text_keys(["  typeof\r\n"])
        assert N.text_keys(["é"]) == N.text_keys(["é"])          # NFC
        assert N.text_keys(["a\r\nb"]) == N.text_keys(["a\nb"])            # CRLF -> LF


# ── the generator ────────────────────────────────────────────────────────────────────────────────


class TestExclusion:
    @pytest.mark.parametrize("prose", [None, PROSE], ids=["no_prose", "prose"])
    def test_seed_1_shares_no_row_with_seed_0_and_keeps_its_count_and_kinds(self, prose):
        old = N.generate(2000, 0, prose=prose)
        keys = {_independent_key(t) for t in old.texts}
        base = N.generate(4000, 1, prose=prose)
        # The fixture's precondition: WITHOUT exclusion the base generator collides (else this proves nothing).
        assert _shared(base.texts, keys) > 0
        fresh = N.generate(4000, 1, prose=prose, exclude=N.text_keys(old.texts))
        assert len(fresh.texts) == len(fresh.kinds) == 4000
        assert _shared(fresh.texts, keys) == 0
        assert fresh.counts == base.counts and collections.Counter(fresh.kinds) == collections.Counter(base.kinds)
        assert max(fresh.counts.values()) - min(fresh.counts.values()) <= 1        # every kind balanced
        assert sum(fresh.dropped.values()) >= 1
        manifest = fresh.manifest(exclude_view_ids=["pmd_old"])
        assert manifest["dropped_by_exclusion"] == sum(fresh.dropped.values())
        assert manifest["dropped_by_exclusion_kinds"] == fresh.dropped and manifest["rows"] == 4000

    def test_the_degenerate_rows_that_collided_are_caught(self):
        """The strings FU-7a saw collide, excluded outright: none of them is emitted."""
        bad = ["!", "0", "typeof", "console.log", "###", "N N N N"]
        base = N.generate(4000, 1, prose=PROSE)
        sentences = N.prose_sentences(PROSE)
        bad += [s for s in sentences if s in set(base.texts)]                    # one-sentence shuffled prose
        keys = {_independent_key(t) for t in bad}
        fresh = N.generate(4000, 1, prose=PROSE, exclude=N.text_keys(bad))
        assert _shared(fresh.texts, keys) == 0 and len(fresh.texts) == 4000

    def test_exclusion_is_deterministic(self):
        keys = N.text_keys(N.generate(500, 0).texts)
        assert N.generate(1000, 1, exclude=keys).texts == N.generate(1000, 1, exclude=keys).texts

    def test_no_exclusion_gives_the_rows_of_every_earlier_version(self):
        """An earlier set must still be reproducible from its seed: the same digest as before exclusion existed."""
        for (rows, seed, prose), digest in (
                ((2000, 0, None), "e41200c0ce69ef4ee64edade01d8e89c501896849e5baba79a7de965db38130f"),
                ((700, 1, PROSE), "ef1eb19f09be9e7b48f75287b688a6bde07151204678f089016edc497791f247")):
            for exclude in (None, set()):
                made = N.generate(rows, seed, prose=prose, exclude=exclude)
                assert hashlib.sha256(json.dumps([made.texts, made.kinds]).encode()).hexdigest() == digest
                assert made.dropped == {}


class TestShortfall:
    def test_a_kind_that_cannot_reach_its_quota_fails_naming_it(self, monkeypatch):
        monkeypatch.setattr(N, "_repeated_token", lambda rng, tokens: "!")
        with pytest.raises(N.ExclusionShortfall) as raised:
            N.generate(700, 1, exclude=N.text_keys(["!"]))
        short = raised.value
        assert short.kind == "repeated_token" and "repeated_token" in str(short)
        # 700 rows over the six kinds without prose: `repeated_token` (third) gets 117; its first slot fails.
        assert short.made == 0 and short.quota == 117
        assert short.draws == N.MAX_DRAWS_PER_ROW

    def test_without_the_exclusion_the_same_generator_is_fine(self, monkeypatch):
        """Control: the shortfall is the exclusion's, not the patched kind's."""
        monkeypatch.setattr(N, "_repeated_token", lambda rng, tokens: "!")
        assert len(N.generate(700, 1).texts) == 700


# ── the route ────────────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    return tmp_path


def _rows(raw_path):
    from src.services.probe_monitor_service import open_split, resolve_dataset_path

    data = open_split(resolve_dataset_path(raw_path))
    return list(data["text"]), list(data["kind"])


class TestTheRoute:
    async def test_seed_1_4000_excluding_the_old_view_shares_no_key(self, client, async_session, data_dir):
        from tests.support.detector_fixtures import dataset_row, write_split

        prose_ds = await dataset_row(async_session, write_split(data_dir, {"body": PROSE}))
        await async_session.commit()
        old = await client.post(f"{API}/nonsense-text", json={
            "rows": 2000, "seed": 0, "prose_dataset_id": str(prose_ds.id), "prose_column": "body"})
        assert old.status_code == 201, old.text
        old_view = old.json()["view"]["id"]
        old_keys = {_independent_key(t) for t in _rows(old.json()["dataset"]["raw_path"])[0]}

        base = N.generate(4000, 1, prose=PROSE)
        assert _shared(base.texts, old_keys) > 0                              # it WOULD collide

        response = await client.post(f"{API}/nonsense-text", json={
            "rows": 4000, "seed": 1, "prose_dataset_id": str(prose_ds.id), "prose_column": "body",
            "exclude_view_ids": [old_view]})
        assert response.status_code == 201, response.text
        body = response.json()
        texts, kinds = _rows(body["dataset"]["raw_path"])
        assert len(texts) == 4000 and body["dataset"]["num_samples"] == 4000
        assert _shared(texts, old_keys) == 0
        assert collections.Counter(kinds) == collections.Counter(base.kinds)
        assert body["view"]["counts"]["negative"] == 4000

        # G-2 and the exclusion, in the response AND on the stored dataset row.
        manifest = body["manifest"]
        assert manifest["prose_dataset_id"] == str(prose_ds.id)
        assert manifest["prose_column"] == "body" and manifest["prose_rows"] == len(PROSE)
        assert manifest["exclude_view_ids"] == [old_view]
        assert manifest["dropped_by_exclusion"] >= 1
        assert manifest["dropped_by_exclusion"] == sum(manifest["dropped_by_exclusion_kinds"].values())
        stored = (await client.get(f"/api/v1/datasets/{body['dataset']['id']}")).json()
        assert stored["metadata"]["nonsense_text"] == manifest

    async def test_the_manifest_without_prose_or_exclusion(self, client, data_dir):
        response = await client.post(f"{API}/nonsense-text", json={"rows": 300, "seed": 9, "register_view": False})
        assert response.status_code == 201, response.text
        manifest = response.json()["manifest"]
        assert manifest["prose_dataset_id"] is None and manifest["prose_column"] is None
        assert manifest["prose_rows"] is None and manifest["exclude_view_ids"] == []
        assert manifest["dropped_by_exclusion"] == 0 and manifest["dropped_by_exclusion_kinds"] == {}

    async def test_a_short_kind_is_refused_and_nothing_is_written(self, client, async_session, data_dir,
                                                                 monkeypatch):
        from sqlalchemy import func, select

        from src.models.dataset import Dataset
        from src.models.probe_monitor import ProbeMonitorDataset
        from tests.support.detector_fixtures import dataset_row, write_split

        ds = await dataset_row(async_session, write_split(data_dir, {"text": ["!", "ordinary words"]}))
        view = ProbeMonitorDataset(name="old", dataset_id=ds.id, input_column="text", label_column="text",
                                   label_mapping={"*": "negative"}, role="nonsense_text", counts={"negative": 2})
        async_session.add(view)
        await async_session.commit()
        before = (await async_session.execute(select(func.count()).select_from(Dataset))).scalar_one()

        monkeypatch.setattr(N, "_repeated_token", lambda rng, tokens: "!")
        response = await client.post(f"{API}/nonsense-text", json={
            "rows": 700, "seed": 1, "exclude_view_ids": [view.id]})
        assert response.status_code == 422, response.text
        detail = response.json()["detail"]
        assert detail["code"] == "exclusion_shortfall" and detail["kind"] == "repeated_token"
        assert "repeated_token" in detail["message"] and detail["exclude_view_ids"] == [view.id]
        after = (await async_session.execute(select(func.count()).select_from(Dataset))).scalar_one()
        assert after == before                                               # never a short set, silently or not

    async def test_an_unknown_view_is_refused(self, client, data_dir):
        response = await client.post(f"{API}/nonsense-text", json={
            "rows": 300, "exclude_view_ids": ["pmd_doesnotexist"]})
        assert response.status_code == 404, response.text
        assert response.json()["detail"]["views"] == ["pmd_doesnotexist"]


# ── the MCP tool ─────────────────────────────────────────────────────────────────────────────────


def _tool_calls(arguments):
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.tools import probe_monitors
    from tests.unit.test_reachability import RecordingClient

    mcp = MCPServer("t")
    client = RecordingClient()
    probe_monitors.register(mcp, client, MCPSettings(auth_token="x" * 32))
    asyncio.run(mcp.call_tool("create_nonsense_text_dataset", arguments))
    return client.calls


def test_the_tool_sends_exclude_view_ids_to_the_route():
    """Payload and count: the ids travel under the route's name, once."""
    assert _tool_calls({"rows": 4000, "seed": 1, "exclude_view_ids": ["pmd_03e7d9c82e6b"]}) == [
        ("POST", "/probe-monitors/nonsense-text",
         {"json_body": {"rows": 4000, "seed": 1, "prose_column": "text", "register_view": True,
                        "exclude_view_ids": ["pmd_03e7d9c82e6b"]}})]


def test_the_tool_declares_the_argument_in_the_live_registry():
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.tools import probe_monitors
    from tests.unit.test_reachability import RecordingClient

    mcp = MCPServer("t")
    probe_monitors.register(mcp, RecordingClient(), MCPSettings(auth_token="x" * 32))
    tool = {t.name: t for t in asyncio.run(mcp.list_tools())}["create_nonsense_text_dataset"]
    assert "exclude_view_ids" in tool.input_schema["properties"]
