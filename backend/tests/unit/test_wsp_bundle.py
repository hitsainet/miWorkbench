"""ENH-010 044 (task 3.3): WSP files live under the parent run's directory, built only from minted ids."""
from __future__ import annotations

import pytest

from src.core.config import settings
from src.services import wsp_layout
from src.services.probe_monitor_run import artifact_dir_for

RUN, CAPTURE, BUNDLE = "pmr_0123456789ab", "wac_aaaaaaaaaaaa", "wsb_bbbbbbbbbbbb"


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    return tmp_path


def test_everything_lives_under_the_run_dir():
    run_dir = artifact_dir_for(RUN).resolve()
    for path in (wsp_layout.wsp_root(RUN), wsp_layout.capture_dir(RUN, CAPTURE), wsp_layout.bundle_dir(RUN, BUNDLE),
                 wsp_layout.baselines_dir(RUN, BUNDLE)):
        assert path.resolve().is_relative_to(run_dir), path
    assert wsp_layout.capture_dir(RUN, CAPTURE).parts[-3:] == ("wsp", "captures", CAPTURE)
    assert wsp_layout.bundle_dir(RUN, BUNDLE).parts[-3:] == ("wsp", "bundles", BUNDLE)


@pytest.mark.parametrize("bad", ["../x", "wac_../../etc", "wac_AAAAAAAAAAAA", "", "wsb_bbbbbbbbbbbb"])
def test_a_path_is_never_built_from_an_unminted_id(bad):
    with pytest.raises(ValueError):
        wsp_layout.capture_dir(RUN, bad)
    with pytest.raises(ValueError):
        wsp_layout.wsp_root("../" + RUN)


@pytest.mark.asyncio
async def test_run_delete_reclaims_wsp_files(client, async_session):
    """F44-D18: `DELETE /runs/{id}` removes the run directory, and with it every capture, bundle and baseline file —
    no second janitor, no orphan bytes. Driven through the real route (FTDD §4.4)."""
    from src.models.probe_monitor import ProbeMonitorRun
    from tests.unit.test_gpu_worker_watch import _run_row

    run_id = await _run_row(async_session, "completed")
    row = await async_session.get(ProbeMonitorRun, run_id)
    row.artifact_dir = str(artifact_dir_for(run_id))
    await async_session.commit()
    files = [
        wsp_layout.capture_dir(run_id, CAPTURE) / "tokens.f16",
        wsp_layout.capture_dir(run_id, CAPTURE) / "offsets.npy",
        wsp_layout.bundle_dir(run_id, BUNDLE) / "manifest.json",
        wsp_layout.baselines_dir(run_id, BUNDLE) / "lda.safetensors",
    ]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    assert all(p.is_file() for p in files)

    response = await client.delete(f"/api/v1/probe-monitors/runs/{run_id}")
    assert response.status_code == 204, response.text
    assert not artifact_dir_for(run_id).exists()
    assert not any(p.exists() for p in files)


# ── 5.1: no held-out row is a practice row, by any key (P45, P57, P67; FTID I44-D5) ─────────────────────────────
#
# Real Arrow datasets on disk and real Dataset / view rows; the keys are read the way a bundle build reads them.
# Controls (044 controls record): compare by `view_id` instead of the location → the shared-split test red; the
# location key only → the two-downloads test red; `_dw_row_key` "else" the text key → the mixed test red.

import uuid  # noqa: E402

from src.services.wsp_bundle import BundleRefused, check_overlap, row_keys  # noqa: E402
from tests.support.wsp_world import sync_db  # noqa: E402,F401 (fixture)

TEXTS = ["risk1 calm2 filler3", "calm1 calm4 filler9", "risk7 risk2 filler1", "calm3 risk5 filler0",
         "risk0 calm6 filler5", "calm7 calm0 filler2"]


def _download(tmp_path, name, texts, *, dw_keys=None):
    from datasets import Dataset as HFDataset

    columns = {"inputs": list(texts), "stakes": ["high" if i % 2 == 0 else "low" for i in range(len(texts))]}
    if dw_keys is not None:
        columns["_dw_row_key"] = list(dw_keys)
    path = tmp_path / name
    HFDataset.from_dict(columns).save_to_disk(str(path))
    return path


def _view(db, path, *, role="train"):
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.probe_monitor import ProbeMonitorDataset

    dataset = Dataset(id=uuid.uuid4(), name=path.name, source="local", status=DatasetStatus.READY, raw_path=str(path),
                      extra_metadata={})
    db.add(dataset)
    db.flush()
    return _view_over(db, dataset.id, role=role)


def _view_over(db, dataset_id, *, role):
    from src.models.probe_monitor import ProbeMonitorDataset

    view = ProbeMonitorDataset(name=f"v-{role}", dataset_id=dataset_id, input_column="inputs", label_column="stakes",
                               label_mapping={"high": "positive", "low": "negative"}, role=role, counts={})
    db.add(view)
    db.commit()
    return view


def test_no_heldout_row_is_in_any_bundle_file(sync_db, tmp_path):
    """A training view and an eval view over ONE shared split (I-E's fixture): a (view_id, position) key agrees by
    construction that they are different rows; the LOCATION key sees they are the same."""
    path = _download(tmp_path, "shared", TEXTS)
    train = _view(sync_db, path)
    heldout = _view_over(sync_db, train.dataset_id, role="eval")
    with pytest.raises(BundleRefused) as refused:
        check_overlap(row_keys(sync_db, train, [0, 1, 2, 3]), [row_keys(sync_db, heldout, [3, 4, 5])])
    assert (refused.value.code, refused.value.extra["key"], refused.value.extra["rows"]) == (
        "heldout_overlaps_practice", "location", 1)
    check_overlap(row_keys(sync_db, train, [0, 1, 2]), [row_keys(sync_db, heldout, [3, 4, 5])])  # disjoint passes


def test_two_downloads_of_one_split_overlap_by_content(sync_db, tmp_path):
    """Two downloads of one split: different dataset ids, so no location is shared; the TEXT key sees the rows,
    after the dw.rowkey/v1 normalisation (a CRLF and trailing space do not hide a row)."""
    first = _view(sync_db, _download(tmp_path, "first", TEXTS))
    second = _view(sync_db, _download(tmp_path, "second", [t.replace(" ", " ") + "\r\n " for t in TEXTS]), role="eval")
    with pytest.raises(BundleRefused) as refused:
        check_overlap(row_keys(sync_db, first, [0, 1, 2]), [row_keys(sync_db, second, [2, 3])])
    assert (refused.value.extra["key"], refused.value.extra["rows"]) == ("text", 1)


def test_dataworks_split_against_plain_download_overlaps(sync_db, tmp_path):
    """A miDataworks split (rows carry `_dw_row_key`) against a plain Hub download of the same rows (no key): the
    text key still applies — the dw key is IN ADDITION to it, never instead of it."""
    dataworks = _view(sync_db, _download(tmp_path, "dw", TEXTS, dw_keys=[f"k{i}" for i in range(len(TEXTS))]))
    plain = _view(sync_db, _download(tmp_path, "plain", TEXTS), role="eval")
    with pytest.raises(BundleRefused) as refused:
        check_overlap(row_keys(sync_db, dataworks, [0, 1]), [row_keys(sync_db, plain, [1, 5])])
    assert refused.value.extra["key"] == "text"


def test_two_dataworks_versions_with_different_row_keys_still_overlap_by_text(sync_db, tmp_path):
    """Both sides carry `_dw_row_key` but from different versions (the keys differ while the rows are the same):
    using the dw key INSTEAD of the text key would miss these rows."""
    old = _view(sync_db, _download(tmp_path, "v1", TEXTS, dw_keys=[f"a{i}" for i in range(len(TEXTS))]))
    new = _view(sync_db, _download(tmp_path, "v2", TEXTS, dw_keys=[f"b{i}" for i in range(len(TEXTS))]),
                role="eval")
    with pytest.raises(BundleRefused) as refused:
        check_overlap(row_keys(sync_db, old, [0, 1, 2]), [row_keys(sync_db, new, [2])])
    assert refused.value.extra["key"] == "text"


def test_a_shared_dw_row_key_refuses_even_when_the_text_differs(sync_db, tmp_path):
    """The dw key is a key of its own: two rows miDataworks says are one row overlap whatever their text."""
    left = _view(sync_db, _download(tmp_path, "l", TEXTS[:3], dw_keys=["x0", "x1", "x2"]))
    right = _view(sync_db, _download(tmp_path, "r", TEXTS[3:], dw_keys=["y0", "x1", "y2"]), role="eval")
    with pytest.raises(BundleRefused) as refused:
        check_overlap(row_keys(sync_db, left, [0, 1, 2]), [row_keys(sync_db, right, [0, 1, 2])])
    assert refused.value.extra["key"] == "dw_row_key"


# ── 5.5b: miDataworks' split roles are never crossed ──────────────────────────


def test_partition_roles_refuse_crossovers():
    """Control: drop either check → red."""
    from types import SimpleNamespace

    from src.services.wsp_bundle import check_partition_roles

    def view(view_id, role):
        return SimpleNamespace(id=view_id, dataset_version=None if role is None else {"wsp_role": role})

    with pytest.raises(BundleRefused) as practice_is_heldout:
        check_partition_roles(view("p", "wsp_heldout"), [view("h", None)])
    with pytest.raises(BundleRefused) as heldout_is_practice:
        check_partition_roles(view("p", "wsp_practice"), [view("h", "wsp_heldout"), view("x", "wsp_practice")])
    assert practice_is_heldout.value.extra["key"] == heldout_is_practice.value.extra["key"] == "partition_role"
    assert heldout_is_practice.value.extra["heldout_view"] == "x"
    check_partition_roles(view("p", "wsp_practice"), [view("h", "wsp_heldout"), view("u", None)])
    check_partition_roles(view("p", None), [view("h", None)])
