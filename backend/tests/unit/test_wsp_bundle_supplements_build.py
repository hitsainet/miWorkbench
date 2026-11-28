"""ENH-010 FU-7a C1 and C2 in the build, on a REAL tiny run with real captures (FTID I44-D1; 2026-11-25).

The world: the pipeline's practice corpus, its held-out eval and calibration sets, and two `supplement` views the run
is evaluated on (its evaluation never counts towards the rung), each captured in the same job as the held-out sets:

* `clean`: a third corpus, negatives only, seven words a row (no practice or held-out text), with a `carrier_id`;
* `dirty`: the HELD-OUT corpus's negatives registered as a supplement, so every one of its rows is a held-out row.

What is pinned:

* a bundle with `supplement_view_ids=[clean]` carries `practice.supplements` exactly as miForge's published schema
  (`MIFORGE_REPO`) states it: its rows, digests and parity reproduce from its own fields and files, the five
  `supplement_*` files hold exactly the capture's rows, and the supplement is in no held-out set;
* a bundle without supplements is unchanged (no key, no file);
* `dirty` is refused `heldout_overlaps_practice` (miForge would read held-out rows);
* `heldout_reused`: a set an earlier bundle was sealed on and miForge scored (`used_heldout`) refuses a new bundle
  over the same rows, a set bundled but never scored does not, and the calibration capture is exempt;
* `shared_key`: blank `carrier_id` / `pair_id` cells never match, and a column one side lacks is no key.

Negative controls (`0xcc/reviews/fu7a_c1c2_2026-11-25.md`): drop `check_supplements_unseen`, drop
`check_heldout_unused`, drop `write_supplement_files`, drop the supplements from the manifest, or let a blank carrier
match → red.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("datasets")

from tests.support.wsp_pipeline import (  # noqa: E402,F401 (fixtures)
    arrow_dataset,
    capture_all,
    heldout_arrow,
    make_world,
    monkeypatch_module,
    new_bundle,
    pipeline_db,
    run_probe,
    tiny_lm,
    tiny_tokenizer,
    wsp_data_dir,
)

SUPPLEMENT_ROWS = 40


@pytest.fixture(scope="module")
def clean_arrow(tmp_path_factory):
    """Negatives from 'another generator': seven words a row, a vocabulary neither other corpus uses."""
    from datasets import Dataset as HFDataset

    rng = np.random.default_rng(23)
    inputs = [" ".join([f"calm{rng.integers(0, 8)}" for _ in range(3)] + [f"gen{rng.integers(0, 32)}"
                                                                           for _ in range(4)])
              for _ in range(SUPPLEMENT_ROWS)]
    path = tmp_path_factory.mktemp("wsp_supplement") / "arrow"
    HFDataset.from_dict({"inputs": inputs, "stakes": ["low"] * SUPPLEMENT_ROWS,
                         "carrier_id": [f"carrier-{i}" for i in range(SUPPLEMENT_ROWS)]}).save_to_disk(str(path))
    return path


@pytest.fixture(scope="module")
def calm_arrow(tmp_path_factory):
    """The calibration corpus: negatives on their OWN corpus (eight words a row), so no held-out eval row is a
    calibration row and the calibration exemption is seen on its own."""
    from datasets import Dataset as HFDataset

    rng = np.random.default_rng(29)
    inputs = [" ".join([f"calm{rng.integers(0, 8)}" for _ in range(3)] + [f"cal{rng.integers(0, 32)}"
                                                                           for _ in range(5)])
              for _ in range(60)]
    path = tmp_path_factory.mktemp("wsp_calm") / "arrow"
    HFDataset.from_dict({"inputs": inputs, "stakes": ["low"] * 60}).save_to_disk(str(path))
    return path


@pytest.fixture(scope="module")
def sp_world(pipeline_db, arrow_dataset, heldout_arrow, clean_arrow, calm_arrow):
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.probe_monitor import ProbeMonitorDataset

    world = make_world(pipeline_db, arrow_dataset, heldout_arrow, name="unseen-supplemented",
                       calibration_path=calm_arrow)
    clean_ds = Dataset(id=uuid.uuid4(), name="supplement-corpus", source="local", status=DatasetStatus.READY,
                       raw_path=str(clean_arrow), extra_metadata={})
    pipeline_db.add(clean_ds)
    pipeline_db.commit()
    clean = ProbeMonitorDataset(name="clean-supplement", dataset_id=clean_ds.id, input_column="inputs",
                                label_column="stakes", label_mapping={"low": "negative"}, role="supplement", counts={})
    dirty = ProbeMonitorDataset(name="dirty-supplement", dataset_id=world["heldout_ds"].id, input_column="inputs",
                                label_column="stakes", label_mapping={"low": "negative", "high": "excluded"},
                                role="supplement", counts={})
    pipeline_db.add_all([clean, dirty])
    pipeline_db.commit()
    world["run"].eval_dataset_ids = [world["eval"].id, clean.id, dirty.id]
    pipeline_db.commit()
    return {**world, "clean": clean, "dirty": dirty}


@pytest.fixture(scope="module")
def sp_probe(pipeline_db, sp_world, tiny_lm, tiny_tokenizer, wsp_data_dir):
    return run_probe(pipeline_db, sp_world, tiny_lm, tiny_tokenizer)


@pytest.fixture(scope="module")
def sp_captures(pipeline_db, sp_probe, tiny_lm, tiny_tokenizer):
    rows = capture_all(pipeline_db, sp_probe, tiny_lm, tiny_tokenizer)
    roles = sorted(r.role for r in rows)
    assert roles == ["calibration", "eval", "supplement", "supplement"], roles
    return {r.dataset_id: r for r in rows}


def _build(db, probe, tokenizer, **kwargs):
    from src.services.wsp_bundle import build_bundle_row

    bundle = new_bundle(db, probe)
    result = build_bundle_row(db, bundle.id, tokenizer=tokenizer, max_seconds=120, **kwargs)
    db.refresh(bundle)
    return bundle, result


@pytest.fixture(scope="module")
def supplemented(pipeline_db, sp_world, sp_probe, sp_captures, tiny_tokenizer):
    bundle, result = _build(pipeline_db, sp_probe, tiny_tokenizer, heldout_view_ids=[sp_world["eval"].id],
                            supplement_view_ids=[sp_world["clean"].id])
    assert bundle.status == "ready", (result, bundle.error)
    return bundle


def test_the_rung_never_counts_a_supplement_evaluation(pipeline_db, sp_world, sp_probe):
    from src.api.v1.endpoints.probe_monitors import evaluation_eligibility

    eligibility = evaluation_eligibility(sp_probe, sp_world["clean"])
    assert eligibility.counts is False and eligibility.note == "role supplement"


def test_the_manifest_validates_against_miforges_published_schema(supplemented):
    from jsonschema.validators import validator_for

    from src.services.wsp_bundle import schema_errors
    from tests.support.cross_repo import REQUIRE_NEW, require_cross_repo
    from tests.support.sibling_repos import miforge_repo

    assert schema_errors(supplemented.manifest) == []
    published = miforge_repo() / "docs" / "schemas" / "miforge-wsp-bundle-v1.json"
    if not published.is_file():
        message = (f"miForge's bundle schema is not at {published} (set MIFORGE_REPO). Skipping is expected on the "
                   f"public mirror; set {REQUIRE_NEW}=1 to make this a failure.")
        if require_cross_repo():
            pytest.fail(message)
        pytest.skip(message)
    schema = json.loads(published.read_text())
    assert list(validator_for(schema)(schema).iter_errors(supplemented.manifest)) == []
    on_disk = json.loads((Path(supplemented.dir_path) / "manifest.json").read_text())
    assert on_disk == supplemented.manifest


def test_the_supplement_is_its_capture_row_for_row(supplemented, sp_world, sp_captures):
    from src.services.wsp_digests import row_ids_sha256

    capture = sp_captures[sp_world["clean"].id]
    [entry] = supplemented.manifest["practice"]["supplements"]
    directory = Path(supplemented.dir_path)
    positions = np.load(Path(capture.dir_path) / "positions.npy")
    assert entry == {**entry, "view_id": sp_world["clean"].id, "first_row": 0, "rows": capture.rows,
                     "tokens": capture.tokens, "validation_only": True,
                     "label_counts": {"positive": 0, "negative": capture.rows},
                     "row_ids_sha256": capture.row_ids_sha256}
    # Reproducible from its own fields: the digest from the positions file the bundle ships.
    shipped = np.load(directory / "supplement_positions.npy")
    assert np.array_equal(shipped, positions)
    assert row_ids_sha256(entry["view_id"], shipped) == entry["row_ids_sha256"]
    assert np.array_equal(np.load(directory / "supplement_offsets.npy"), np.load(Path(capture.dir_path) / "offsets.npy"))
    tokens = next(Path(capture.dir_path).glob("tokens.f*"))
    activations = next(directory.glob("supplement_activations.f*"))
    assert activations.read_bytes() == tokens.read_bytes()
    roles = {f["role"]: f for f in supplemented.manifest["files"]}
    for role in ("supplement_activations", "supplement_offsets", "supplement_positions", "supplement_labels",
                 "supplement_truncated_mask"):
        import hashlib

        path = directory / roles[role]["name"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == roles[role]["sha256"], role


def test_the_supplement_parity_reproduces_from_its_files(pipeline_db, supplemented, sp_probe):
    from src.services.probe_monitor_run import _aggregate_rows, load_probe

    [entry] = supplemented.manifest["practice"]["supplements"]
    practice = supplemented.manifest["practice"]
    directory = Path(supplemented.dir_path)
    offsets = np.load(directory / "supplement_offsets.npy")
    activations = next(directory.glob("supplement_activations.f*"))
    memmap = np.memmap(activations, dtype=np.dtype(practice["dtype"]), mode="r",
                       shape=(int(offsets[-1]), practice["d_model"]))
    head, trained = load_probe(pipeline_db, sp_probe.id)
    rows = entry["parity"]["rows"]
    scores = _aggregate_rows(head, trained.rule, dict(trained.rule_params or {}),
                             [memmap[int(offsets[i]):int(offsets[i + 1])] for i in rows])
    assert len(rows) == 16 and np.allclose(scores, entry["parity"]["scores"], atol=entry["parity"]["tolerance"],
                                           rtol=0)


def test_the_supplement_is_no_heldout_set(supplemented, sp_world, sp_captures):
    sets = supplemented.manifest["heldout_ref"]["sets"]
    assert {s["view_id"] for s in sets} == {sp_world["eval"].id, sp_world["calibration"].id}
    assert sp_captures[sp_world["clean"].id].id not in supplemented.capture_ids
    assert sp_captures[sp_world["clean"].id].id in supplemented.capture_seals


def test_the_sensitivity_sources_include_the_supplement(pipeline_db, supplemented, sp_world):
    from src.api.v1.endpoints.wsp import bundle_sensitivity_now

    gone = SimpleNamespace(id=supplemented.id, capture_ids=list(supplemented.capture_ids),
                           manifest={**supplemented.manifest,
                                     "practice": {**supplemented.manifest["practice"],
                                                  "supplements": [{"view_id": "pmd_gone00000000"}]}})
    answer = bundle_sensitivity_now(pipeline_db, gone)
    assert answer["value"] == "unknown" and "view pmd_gone00000000" in answer["unresolved"]


def test_a_bundle_without_supplements_is_unchanged(pipeline_db, sp_world, sp_probe, sp_captures, tiny_tokenizer):
    bundle, result = _build(pipeline_db, sp_probe, tiny_tokenizer, heldout_view_ids=[sp_world["eval"].id])
    assert bundle.status == "ready", (result, bundle.error)
    assert "supplements" not in bundle.manifest["practice"]
    assert not any(f["role"].startswith("supplement_") for f in bundle.manifest["files"])


def test_a_supplement_of_heldout_rows_is_refused(pipeline_db, sp_world, sp_probe, sp_captures, tiny_tokenizer):
    bundle, result = _build(pipeline_db, sp_probe, tiny_tokenizer, heldout_view_ids=[sp_world["eval"].id],
                            supplement_view_ids=[sp_world["dirty"].id])
    assert bundle.status == "failed" and result["refused"]["code"] == "heldout_overlaps_practice", result
    assert result["refused"]["practice_view"] == sp_world["dirty"].id
    assert result["refused"]["heldout_view"] == sp_world["eval"].id


def test_a_scored_set_of_an_earlier_bundle_refuses_a_new_one(pipeline_db, sp_world, sp_probe, sp_captures,
                                                             supplemented, tiny_tokenizer):
    """`supplemented` was sealed on the eval capture; once miForge counts it used, the same rows refuse
    `heldout_reused`; unscored, they build (a bundle that failed to import may be built again)."""
    from src.services.wsp_bundle import heldout_set_name

    eval_capture = sp_captures[sp_world["eval"].id]
    used = [{"set_name": heldout_set_name(eval_capture), "row_ids_sha256": eval_capture.row_ids_sha256,
             "heldout_uses": 18}]
    bundle, result = _build(pipeline_db, sp_probe, tiny_tokenizer, heldout_view_ids=[sp_world["eval"].id],
                            used_heldout=used)
    assert bundle.status == "failed" and result["refused"]["code"] == "heldout_reused", result
    assert result["refused"]["heldout_view"] == sp_world["eval"].id
    again, result = _build(pipeline_db, sp_probe, tiny_tokenizer, heldout_view_ids=[sp_world["eval"].id],
                           used_heldout=[])
    assert again.status == "ready", (result, again.error)


def test_the_calibration_capture_is_exempt_in_the_build(pipeline_db, sp_world, sp_probe, sp_captures, supplemented,
                                                        tiny_tokenizer):
    from src.services.wsp_bundle import heldout_set_name

    cal = sp_captures[sp_world["calibration"].id]
    used = [{"set_name": heldout_set_name(cal), "row_ids_sha256": cal.row_ids_sha256, "heldout_uses": 18}]
    bundle, result = _build(pipeline_db, sp_probe, tiny_tokenizer, heldout_view_ids=[sp_world["eval"].id],
                            used_heldout=used)
    assert bundle.status == "ready", (result, bundle.error)


def test_an_earlier_bundles_supplement_rows_count_as_practice(pipeline_db, sp_probe, supplemented, sp_world):
    from src.services.wsp_bundle import earlier_bundle_keys

    heldout, practice = earlier_bundle_keys(pipeline_db, sp_probe, "wsb_notabundle0", [])
    assert sp_world["clean"].id in {k.view_id for k in practice}
    assert heldout == []   # nothing was scored


# ── the keys (pure) ───────────────────────────────────────────────────────────


def _keys(view, texts, carrier=None, pair=None, dw=None):
    from src.services.wsp_bundle import RowKeys

    return RowKeys(view_id=view, location=[(view, None, None, "s", i) for i in range(len(texts))], text=list(texts),
                   dw=dw, carrier=carrier, pair=pair)


def test_shared_key_reads_every_kind_and_never_a_blank():
    from src.services.wsp_bundle import shared_key

    a = _keys("a", ["t1", "t2"], carrier=["c1", None], pair=[None, "p1"])
    assert shared_key(a, _keys("b", ["t3"], carrier=[None])) is None, "two blank carriers are no key"
    assert shared_key(a, _keys("b", ["t3"], carrier=["c1"])) == ("carrier_id", 1)
    assert shared_key(a, _keys("b", ["t3"], pair=["p1"])) == ("pair_id", 1)
    assert shared_key(a, _keys("b", ["t2"])) == ("text", 1)
    assert shared_key(a, _keys("b", ["t3"])) is None, "a column one side lacks is no key"
    assert shared_key(_keys("a", ["x"], dw=["k"]), _keys("b", ["y"], dw=["k"])) == ("dw_row_key", 1)


def test_a_new_set_sharing_rows_with_a_scored_calibration_set_is_reused():
    """The calibration capture itself may be bundled again (FU-7a D4), but its rows were scored: another held-out
    set holding them is not fresh."""
    from src.services.wsp_bundle import BundleRefused, check_heldout_unused

    with pytest.raises(BundleRefused) as caught:
        check_heldout_unused([(SimpleNamespace(role="eval", dataset_id="new"), _keys("new", ["cal-row"]))],
                             [_keys("old-calibration", ["cal-row"])], [])
    assert caught.value.code == "heldout_reused"


def test_check_heldout_unused_names_the_key_and_skips_calibration():
    from src.services.wsp_bundle import BundleRefused, check_heldout_unused

    new = _keys("new", ["fresh", "old-row"])
    with pytest.raises(BundleRefused) as caught:
        check_heldout_unused([(SimpleNamespace(role="eval", dataset_id="new"), new)], [_keys("old", ["old-row"])],
                             [])
    assert caught.value.code == "heldout_reused" and caught.value.extra["key"] == "text"
    check_heldout_unused([(SimpleNamespace(role="calibration", dataset_id="new"), new)], [_keys("old", ["old-row"])],
                         [])
    with pytest.raises(BundleRefused):
        check_heldout_unused([(SimpleNamespace(role="eval", dataset_id="new"), new)], [],
                             [_keys("old-practice", ["old-row"])])


# ── cross-repo: miForge's own code on both halves (loud skip without miForge; fails under the switch) ──────────────


from tests.unit.test_bundle_route_cross_repo import BASE, IntoTheApp, miforge_backend, route_session  # noqa: E402,F401


def test_miforge_imports_a_supplemented_bundle_through_the_routes(pipeline_db, supplemented, route_session, tmp_path):
    """miForge's import task body (`request_import`, `import_bundle`), from the miForge checkout `MIFORGE_REPO`
    names, pulls the supplemented bundle through this app's live routes and checks every supplement: its files, its
    row-id digest and its own parity, with miForge's evaluator. Control: drop `write_supplement_files` → red."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from src.main import app

    transport = IntoTheApp(app)
    with miforge_backend() as mf:
        engine = create_engine(f"sqlite:///{tmp_path / 'miforge.sqlite'}")
        mf.models.WspBundle.__table__.create(engine)
        settings = mf.settings.WSPSettings(miworkbench_api_url=BASE, wsp_bundle_root=tmp_path / "wsp",
                                           wsp_providers_available="")
        with sessionmaker(bind=engine, expire_on_commit=False)() as session:
            row, _ = mf.bundles.request_import(
                session, {"owner": "miworkbench", "bundle_id": supplemented.id,
                          "bundle_sha256": supplemented.bundle_sha256}, received_by="operator", settings=settings)
            session.commit()
            outcome = mf.bundles.import_bundle(session, row.bundle_sha256, settings=settings, transport=transport)
            session.commit()
            assert outcome == "imported", (outcome, row.import_failure)
            assert row.manifest["practice"]["supplements"] == supplemented.manifest["practice"]["supplements"]
            stored = mf.bundles.store_dir(settings, supplemented.bundle_sha256)
            assert {p.name for p in stored.iterdir()} >= {"supplement_offsets.npy", "supplement_labels.npy"}
        engine.dispose()


def test_miforges_own_uses_route_feeds_the_reader(tmp_path):
    """The shape the route reads is the one miForge's OWN `GET /wsp/heldout-uses` answers (its function, on its own
    table), never a picture of it: the real reader parses it and `refuse_used` refuses the set it counts."""
    import datetime as dt

    import httpx
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from src.services.wsp_bundle import BundleRefused
    from src.services.wsp_heldout_uses import read_heldout_uses, refuse_used, used_by_us

    with miforge_backend() as mf:
        engine = create_engine(f"sqlite:///{tmp_path / 'miforge.sqlite'}")
        mf.models.WspHeldoutUse.__table__.create(engine)
        with sessionmaker(bind=engine)() as session:
            session.add(mf.models.WspHeldoutUse(owner="miworkbench", set_name="eval-pmd_0df63ef937a4",
                                                row_ids_sha256="c" * 64, heldout_uses=18,
                                                last_used_at=dt.datetime(2026, 11, 24, 1, 33, tzinfo=dt.UTC),
                                                last_candidate_id="wspc_093e79561559"))
            session.commit()
            answer = mf.candidates.list_heldout_uses(session)
        engine.dispose()

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=answer))
    uses = read_heldout_uses("http://miforge.test", transport=transport)
    assert used_by_us(uses) == [{"set_name": "eval-pmd_0df63ef937a4", "row_ids_sha256": "c" * 64,
                                 "heldout_uses": 18}]
    capture = SimpleNamespace(role="eval", dataset_id="pmd_0df63ef937a4", row_ids_sha256="d" * 64)
    with pytest.raises(BundleRefused) as caught:
        refuse_used([capture], [], uses)
    assert caught.value.code == "heldout_reused"


# ── the refits fit on practice-train AND the supplement (operator decision 2026-11-27) ──────────────────────────────
#
# In the phase-2 verdicts the LR baseline was the parent bit for bit (the same scores_sha256 and bar on every set): it
# re-ran the parent's own trainer on the parent's own rows with the parent's seed. Now LR, LDA and DoM fit on every
# practice row miForge's editor may use, the supplement included (train-only: practice-validation is untouched), and
# a score answer that asks says which baseline heads are still the parent (`identical_to_parent`). Record and
# controls: 0xcc/reviews/wsp_baselines_supplement_2026-11-27.md.

REFITS = ("logistic_regression", "lda", "diff_of_means")


@pytest.fixture(scope="module")
def plain(pipeline_db, sp_world, sp_probe, sp_captures, tiny_tokenizer):
    """The same probe and held-out sets, bundled with NO supplement."""
    bundle, result = _build(pipeline_db, sp_probe, tiny_tokenizer, heldout_view_ids=[sp_world["eval"].id])
    assert bundle.status == "ready", (result, bundle.error)
    assert "supplements" not in bundle.manifest["practice"]
    return bundle


def _score(db, bundle, labels, **extra):
    from src.schemas.wsp_scoring import WspScoreRequest
    from src.services import wsp_score

    body = {"kind": "miworkbench.wsp-score-request/v1", "bundle_id": bundle.id, "bundle_sha256": bundle.bundle_sha256,
            "run_ref": {"miforge_run_id": "wsr_baselines", "patch_sha256": None, "max_changed_entries": None},
            "heldout_ref": json.loads(json.dumps(bundle.manifest["heldout_ref"])),
            "heads": [{"label": label, "tensors": None} for label in labels], "primary_sets": [], **extra}
    return wsp_score.score(db, WspScoreRequest.model_validate(body))


def _contract_checked(answer):
    """Every answer here is the published contract's: the response model and, independently, the committed file."""
    from jsonschema import Draft202012Validator

    from src.schemas.wsp_scoring import WspScoreResponse

    WspScoreResponse.model_validate(answer)
    contract = json.loads((Path(__file__).resolve().parents[3] / "docs" / "schemas"
                           / "miworkbench-wsp-scoring-v1.json").read_text())
    errors = [e.message for e in Draft202012Validator(contract["response"]).iter_errors(answer)]
    assert errors == [], errors[:5]
    return answer


def _scored_sets(answer):
    sets = [s for s in answer["sets"] if s["role"] != "calibration"]
    assert sets, "fixture: the bundle has a set that carries scores"
    return sets


def _sha(values):
    import hashlib

    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()


def _baseline_rows(db, bundle):
    from src.models.wsp import WspBaseline

    return {r.kind: r for r in db.query(WspBaseline).filter(WspBaseline.bundle_id == bundle.id).all()}


def test_without_a_supplement_the_lr_refit_is_the_parent_and_the_answer_says_so(pipeline_db, plain):
    """The defect, reproduced on the real tiny run: practice-train alone gives the parent back bit for bit, and the
    flag reads true (so a verdict can say the bar adds nothing)."""
    fits = _baseline_rows(pipeline_db, plain)
    assert fits["logistic_regression"].fit["supplement_rows"] == 0
    answer = _contract_checked(_score(pipeline_db, plain, ["parent", "baseline:logistic_regression"],
                                      report_identical_to_parent=True))
    for one in _scored_sets(answer):
        assert _sha(one["scores"]["baseline:logistic_regression"]) == _sha(one["scores"]["parent"]), one["name"]
        assert one["identical_to_parent"] == {"baseline:logistic_regression": True}, one["name"]


def test_with_a_supplement_the_refits_are_not_the_parent_and_the_flag_is_false(pipeline_db, supplemented, sp_captures,
                                                                               sp_world):
    """The fix: each refit fitted on practice-train plus the supplement's 40 rows, so LR is no longer the parent and
    `identical_to_parent` is false for every real refit; `threshold_recut` (the parent's scores, its own bar) reads
    true by construction."""
    supplement = sp_captures[sp_world["clean"].id]
    fits = _baseline_rows(pipeline_db, supplemented)
    for kind in REFITS:
        assert fits[kind].fit["supplement_rows"] == supplement.rows == SUPPLEMENT_ROWS, (kind, fits[kind].fit)
    heads = ["parent"] + [f"baseline:{k}" for k in REFITS] + ["baseline:threshold_recut"]
    answer = _contract_checked(_score(pipeline_db, supplemented, heads, report_identical_to_parent=True))
    for one in _scored_sets(answer):
        flags = one["identical_to_parent"]
        assert set(flags) == set(heads) - {"parent"}, flags
        for kind in REFITS:
            label = f"baseline:{kind}"
            assert flags[label] is False, (one["name"], label)
            assert _sha(one["scores"][label]) != _sha(one["scores"]["parent"]), (one["name"], label)
        assert flags["baseline:threshold_recut"] is True
    assert "identical_to_parent" not in next(s for s in answer["sets"] if s["role"] == "calibration")


def test_an_answer_nobody_asked_for_the_flag_is_the_old_shape(pipeline_db, supplemented):
    """Opt-in (`report_identical_to_parent`, default false): miForge validates every answer against its vendored copy of
    the contract, where a new key in a set is refused, so a caller that does not ask gets exactly the old key set."""
    answer = _contract_checked(_score(pipeline_db, supplemented, ["parent", "baseline:logistic_regression"]))
    for one in answer["sets"]:
        assert "identical_to_parent" not in one, one["name"]


def test_a_default_answer_validates_against_the_contract_miforge_vendors(pipeline_db, supplemented):
    """Cross-repo: the answer to a request without the flag is valid under miForge's vendored copy at `MIFORGE_REPO`
    (the old revision until miForge re-vendors), so this push breaks no live `evaluate`."""
    from jsonschema import Draft202012Validator

    from tests.support.cross_repo import REQUIRE_NEW, require_cross_repo
    from tests.support.sibling_repos import miforge_repo

    vendored = miforge_repo() / "docs" / "schemas" / "miworkbench-wsp-scoring-v1.json"
    if not vendored.is_file():
        message = (f"miForge's vendored scoring contract is not at {vendored} (set MIFORGE_REPO). Skipping is expected "
                   f"on the public mirror; set {REQUIRE_NEW}=1 to make this a failure.")
        if require_cross_repo():
            pytest.fail(message)
        pytest.skip(message)
    contract = json.loads(vendored.read_text())
    answer = _score(pipeline_db, supplemented, ["parent"] + [f"baseline:{k}" for k in REFITS])
    errors = [e.message for e in Draft202012Validator(contract["response"]).iter_errors(answer)]
    assert errors == [], errors[:5]


def test_the_refits_see_practice_train_and_the_supplement_and_no_heldout_row(pipeline_db, sp_world, sp_probe,
                                                                             sp_captures, tiny_tokenizer, monkeypatch):
    """6.5 with a supplement: a spy on every registered fit. The three refits fit on practice-train plus EXACTLY the
    supplement capture's rows; the supplement never enters practice-validation (the LR early-stopping basis) or the
    operating-point kinds' rows; and no held-out capture row (eval, calibration) reaches any fit."""
    from dataclasses import replace

    from src.services import wsp_baselines

    def as_bytes(rows):
        return [np.asarray(r, dtype=np.float32).tobytes() for r in rows]

    seen = {}

    def spy(kind, fit):
        def wrapped(parent, practice, **kwargs):
            seen[kind] = {"fit": as_bytes(practice.fit_rows()[0]), "validation": as_bytes(practice.split(True)[0]),
                          "practice": as_bytes(practice.rows), "supplement": as_bytes(practice.supplement_rows)}
            return fit(parent, practice, **kwargs)
        return wrapped

    monkeypatch.setattr(wsp_baselines, "BASELINE_KINDS", {
        kind: replace(spec, fit=spy(kind, spec.fit)) for kind, spec in wsp_baselines.BASELINE_KINDS.items()})
    bundle, result = _build(pipeline_db, sp_probe, tiny_tokenizer, heldout_view_ids=[sp_world["eval"].id],
                            supplement_view_ids=[sp_world["clean"].id])
    assert bundle.status == "ready", (result, bundle.error)

    def capture_rows(capture):
        directory = Path(capture.dir_path)
        offsets = np.load(directory / "offsets.npy")
        memmap = np.memmap(next(directory.glob("tokens.f*")), dtype=np.dtype(capture.dtype), mode="r",
                           shape=(capture.tokens, capture.d_model))
        return as_bytes(memmap[int(offsets[i]):int(offsets[i + 1])] for i in range(len(offsets) - 1))

    supplement = capture_rows(sp_captures[sp_world["clean"].id])
    heldout = set()
    for capture in sp_captures.values():
        if capture.role != "supplement":
            heldout |= set(capture_rows(capture))
    assert heldout and set(seen) == set(wsp_baselines.BASELINE_KINDS)
    for kind, rows in seen.items():
        assert rows["supplement"] == supplement, kind
        if kind in REFITS:
            assert rows["fit"][-len(supplement):] == supplement, kind
        assert not (set(rows["validation"]) & set(supplement)), kind
        assert not (set(rows["practice"]) & set(supplement)), kind
        everything = set(rows["practice"]) | set(rows["supplement"])
        assert not (everything & heldout), kind
