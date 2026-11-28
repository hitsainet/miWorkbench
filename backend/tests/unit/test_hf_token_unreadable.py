"""An unreadable saved Hugging Face token says so (ENH-011 WB-1, N4).

`resolve_hf_token` read the encrypted `hf_token` setting inside `except Exception: logger.debug(...)`: when the
database could not be reached, or the row could not be decrypted (a changed `SECRET_KEY`), it logged at DEBUG and
returned None. Since 045 the operator's downloads reach it through `hub_token("stored", …)`, so a gated download then
ran anonymously and the Hub's 401 read as "this repository is gated", not "the saved token could not be read".

Now:
* the failure is one WARNING carrying the stable reason code `hf_token_unreadable`, and never the token, its base64
  form or the stored ciphertext (no traceback either: an exception's text can hold the value);
* `resolve_hf_token_detail` carries the reason out; `resolve_hf_token` keeps its signature for its many callers;
* the operator's model and dataset downloads, refused by the Hub (401/403), say the saved token could not be read;
* an agent (`"anonymous"`, 045 A36) never reads the stored token, so it never sees the reason
  (`test_hub_auth_for_agents.py` pins that the anonymous path resolves nothing).
"""
from __future__ import annotations

import base64
import contextlib
import logging
import os
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

TOKEN = "hf_SAVED_TOKEN_must_never_be_logged"
B64 = base64.b64encode(TOKEN.encode()).decode()
REASON = "hf_token_unreadable"
NOTE = "saved Hugging Face token could not be read"


@pytest.fixture
def no_other_token(monkeypatch):
    """Only the database tier can supply a token: no explicit, no `HF_TOKEN`, no settings value."""
    from src.core.config import settings

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(settings, "hf_token", None, raising=False)


class _Unreachable:
    """`SyncSessionLocal()` for a database that cannot be read; the error's text holds the token, as a driver's or a
    decryption error's can."""

    def __call__(self):
        return self

    def __enter__(self):
        raise RuntimeError(f"could not read app_settings: value {TOKEN} / {B64}")

    def __exit__(self, *exc):
        return False


@pytest.fixture
def unreachable(monkeypatch, no_other_token):
    monkeypatch.setattr("src.core.database.SyncSessionLocal", _Unreachable())


def _leaks(records, *secrets) -> list[str]:
    formatter = logging.Formatter("%(name)s %(levelname)s %(message)s")
    out = []
    for record in records:
        text = formatter.format(record) + str(record.args) + str(getattr(record, "__dict__", {}))
        out.extend(f"{record.name}: {s[:12]}…" for s in secrets if s in text)
    return out


def test_an_unreadable_saved_token_is_one_warning_with_a_reason_and_no_token(unreachable, caplog):
    """(a) One WARNING with the reason code; neither the token nor its base64 form in ANY record."""
    from src.services.huggingface_sae_service import resolve_hf_token

    with caplog.at_level(logging.DEBUG):
        assert resolve_hf_token(None) is None, "an unreadable token still means no token"
    warned = [r for r in caplog.records if r.levelno >= logging.WARNING and REASON in r.getMessage()]
    assert len(warned) == 1, [(r.levelname, r.getMessage()) for r in caplog.records]
    assert warned[0].name == "src.services.huggingface_sae_service"
    assert not warned[0].exc_info, "a traceback can carry the value the error was about"
    assert _leaks(caplog.records, TOKEN, B64) == []


def test_the_detail_carries_the_reason_and_the_old_signature_is_kept(unreachable):
    from src.services.huggingface_sae_service import resolve_hf_token, resolve_hf_token_detail

    detail = resolve_hf_token_detail(None)
    assert (detail.token, detail.reason) == (None, REASON)
    assert resolve_hf_token(None) is None


def test_a_readable_or_absent_token_carries_no_reason(no_other_token, monkeypatch):
    from src.core.config import settings
    from src.services.huggingface_sae_service import resolve_hf_token_detail

    assert tuple(resolve_hf_token_detail("hf_explicit")) == ("hf_explicit", None)
    monkeypatch.setattr(settings, "hf_token", "hf_env", raising=False)  # the deployment tier (`HF_TOKEN`)
    assert tuple(resolve_hf_token_detail(None)) == ("hf_env", None)


def test_the_agent_path_never_reads_the_stored_token_so_never_sees_the_reason(unreachable, caplog):
    """045 A36: `"anonymous"` resolves nothing, so the database is never touched and nothing is logged."""
    from src.core.hub_auth import hub_token, hub_token_detail

    with caplog.at_level(logging.DEBUG):
        assert tuple(hub_token_detail("anonymous", "hf_x")) == (False, None)
        assert hub_token_detail("not-a-policy").token is False
        assert hub_token("anonymous", "hf_x") is False
    assert not [r for r in caplog.records if REASON in r.getMessage()], "an agent's call read the stored token"
    assert tuple(hub_token_detail("stored")) == (None, REASON)


# ── (b) the operator's download, refused by the Hub, says why ──────────────────────────────────────────────────────


def _hub_refusal(status: int):
    """The error huggingface_hub raises for a gated repository read without a token."""
    import httpx2
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    response = httpx2.Response(status, request=httpx2.Request("GET", "https://huggingface.co/api/x"))
    cls = GatedRepoError if status == 401 else HfHubHTTPError
    return cls(f"{status} Client Error. Cannot access gated repo for url https://huggingface.co/api/x.",
               response=response)


@pytest.mark.parametrize("status", [401, 403])
def test_the_operators_model_download_says_the_saved_token_could_not_be_read(unreachable, tmp_path, status):
    """The REAL task body under `"stored"` with the real token resolution (no stub of it); the load is refused."""
    import torch

    from src.models.model import Model
    from src.workers import model_tasks
    from tests.unit.test_gpu_model_download_placement import CARDS, Placement, _Query

    model_row = SimpleNamespace(status=None, progress=None, error_message=None)
    added = []
    db = MagicMock()
    db.add.side_effect = added.append
    db.query.side_effect = lambda cls: _Query(first=model_row) if cls is Model else _Query(rows=())
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=db)
    ctx.__exit__ = MagicMock(return_value=False)
    sent = MagicMock()
    seen = {}

    def load(**kwargs):
        seen["token"] = kwargs["hf_token"]
        raise _hub_refusal(status)

    with patch("src.workers.base_task.DatabaseTask.get_db", return_value=ctx), \
         patch("src.services.gpu_placement.place_job",
               MagicMock(return_value=Placement(card=CARDS[1], device=torch.device("cuda", 1)))), \
         patch.object(model_tasks, "required_mb_for_load", MagicMock(return_value=None)), \
         patch("src.workers.base_task.mark_task_queue_entries_completed", MagicMock()), \
         patch("src.ml.transformers_compat.patch_transformers_compatibility", MagicMock()), \
         patch.object(model_tasks, "load_model_from_hf", load), \
         patch.object(model_tasks, "send_progress_update", sent), \
         patch.object(model_tasks, "DownloadProgressMonitor", MagicMock()), \
         patch.object(model_tasks, "clear_cancel_request", MagicMock()), \
         patch.object(model_tasks, "cancel_checker", MagicMock()), \
         patch.object(model_tasks, "settings", SimpleNamespace(models_dir=tmp_path)):
        with pytest.raises(OSError):  # huggingface_hub's HTTP errors are OSErrors
            model_tasks.download_and_load_model.run("m_1", "vendor/gated", "FP16", gpu_request=None,
                                                    hub_auth="stored")
    assert seen["token"] is None, "the test must reach the load with no token"
    assert NOTE in model_row.error_message and REASON in model_row.error_message, model_row.error_message
    assert "Cannot access gated repo" in model_row.error_message, "the Hub's own words are kept"
    queued = [r for r in added if getattr(r, "task_type", None) == "download"]
    assert len(queued) == 1 and NOTE in queued[0].error_message
    last = sent.call_args_list[-1].kwargs
    assert last["status"] == "error" and NOTE in last["message"]


def test_a_refused_model_download_with_a_readable_token_is_not_blamed_on_it(no_other_token, tmp_path):
    """The note is about the saved token only: an explicit token refused by the Hub reads as before."""
    from src.core.hub_auth import explain_hub_refusal

    exc = _hub_refusal(401)
    assert explain_hub_refusal("Failed: x", exc, None) == "Failed: x"
    assert explain_hub_refusal("Failed: x", RuntimeError("CUDA out of memory"), REASON) == "Failed: x"
    assert explain_hub_refusal("Failed: x", exc, REASON).startswith("The saved Hugging Face token could not be read")


@pytest.fixture
def undecryptable_saved_token(async_engine, no_other_token):
    """A stored `hf_token` row that fails authentication (written under another key): a real cause of the failure,
    with the task's own database untouched."""
    from src.core.database import SyncSessionLocal
    from src.models.app_setting import AppSetting

    envelope = base64.b64encode(os.urandom(64)).decode()
    with SyncSessionLocal() as db:
        db.query(AppSetting).filter(AppSetting.key == "hf_token").delete()
        db.add(AppSetting(key="hf_token", value=envelope, is_sensitive=True, category="api_keys"))
        db.commit()
    yield envelope
    with SyncSessionLocal() as db:
        db.query(AppSetting).filter(AppSetting.key == "hf_token").delete()
        db.commit()


@pytest.mark.parametrize("status", [401, 403])
def test_the_operators_dataset_download_says_the_saved_token_could_not_be_read(undecryptable_saved_token, tmp_path,
                                                                               monkeypatch, caplog, status):
    """The REAL `download_dataset_task` body against a real row; `datasets.load_dataset` is refused by the Hub."""
    from src.core import cancellation
    from src.core.config import settings
    from src.core.database import SyncSessionLocal
    from src.models.dataset import Dataset, DatasetStatus
    from src.workers import dataset_tasks
    from tests.support.hub_spy import SHA

    seen = {}

    def refused(*args, **kwargs):
        seen["token"] = kwargs.get("token")
        raise _hub_refusal(status)

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data", raising=False)
    monkeypatch.setattr(dataset_tasks, "emit_dataset_progress", lambda *a, **k: None)
    monkeypatch.setattr(cancellation, "CancelWatchdog", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr("datasets.load_dataset", refused)
    with SyncSessionLocal() as db:
        row = Dataset(id=uuid.uuid4(), name="ds", source="HuggingFace", hf_repo_id="fixture-org/gated",
                      status=DatasetStatus.DOWNLOADING, revision=SHA, extra_metadata={"split": "train"})
        db.add(row)
        db.commit()
        dataset_id = str(row.id)
    with caplog.at_level(logging.DEBUG):
        try:
            dataset_tasks.download_dataset_task(dataset_id=dataset_id, repo_id="fixture-org/gated", split="train",
                                                revision=SHA, hub_auth="stored")
        except Exception:  # noqa: BLE001 - the row is what the operator reads
            pass
    assert seen.get("token") is None, "the test must reach the load with no token"
    with SyncSessionLocal() as db:
        done = db.get(Dataset, uuid.UUID(dataset_id))
        assert done.status == DatasetStatus.ERROR
        assert NOTE in done.error_message and REASON in done.error_message, done.error_message
        from src.models.task_queue import TaskQueue

        queued = db.query(TaskQueue).filter(TaskQueue.entity_id == dataset_id).all()
        assert queued and all(NOTE in q.error_message for q in queued)
    assert _leaks(caplog.records, TOKEN, B64, undecryptable_saved_token) == []
