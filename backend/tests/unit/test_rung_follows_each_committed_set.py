"""A probe's rung follows each committed evaluation set, not only the whole group (2026-11-17).

Run `pmr_b63a7e7c94ca` (2026-11-03) was killed partway through its evaluation sets. ENH-006 G15 had
made each set commit on its own, so the finished sets survived, but the rung was recomputed only
after every set of the group: two of the run's probes kept `rung = 0` beside in-distribution AUROC
intervals whose lower bounds were above 0.7, until the G5 backfill recomputed them on 2026-11-17.

`_evaluate_probes_on_sets` runs for real here, over the real Arrow world of
`test_probe_evaluation_arrays.py`; only the model forward and the renderer are faked. The second
set's forward raises, as a killed worker's job ends. The test asserts the first set's commit was
followed by a rung recompute for every probe of the group, in that order.

Control: delete the per-set `recompute_rung` loop in `_evaluate_probes_on_sets` and this goes red.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_probe_evaluation_arrays import FakeDB, world  # noqa: F401  (the fixture is used by name)

from src.services import probe_monitor_capture, probe_monitor_run


class RecordingDB(FakeDB):
    def __init__(self, world, events):  # noqa: F811  (the parameter, not the fixture)
        super().__init__(world)
        self.events = events

    def commit(self):
        super().commit()
        self.events.append(("commit",))


def test_a_committed_set_promotes_the_rung_before_a_later_set_fails(world, monkeypatch):  # noqa: F811
    events: list[tuple] = []
    monkeypatch.setattr(
        probe_monitor_run, "recompute_rung", lambda _db, probe_id: events.append(("rung", probe_id))
    )
    real_forward = probe_monitor_capture.forward_scores_many

    def forward_then_die(model, rendered, specs, **kwargs):
        if world["state"]["forwards"] >= 1:
            world["state"]["forwards"] += 1
            raise MemoryError("the kernel killed the worker on the second set")
        return real_forward(model, rendered, specs, **kwargs)

    monkeypatch.setattr(probe_monitor_capture, "forward_scores_many", forward_then_die)

    probes = [
        SimpleNamespace(id=pid, target_fpr=None, variant="dense", sae_id=None, threshold=None)
        for pid in ("pm_a", "pm_b")
    ]
    trained = SimpleNamespace(rule="mean", rule_params={})
    context = SimpleNamespace(artifact_dir=world["dir"], scope="all", max_length=64,
                              skip_template_prefix=False, target_fpr=0.1, seed=0)
    db = RecordingDB(world, events)
    with pytest.raises(MemoryError):
        probe_monitor_run._evaluate_probes_on_sets(
            db, [(p, object(), trained, None, None) for p in probes], context, None,
            SimpleNamespace(pad_token_id=0), "LlamaForCausalLM", ["pmd_eval", "pmd_eval_2"],
        )

    assert db.commits == 1, "precondition: the first set committed and the second did not"
    first_commit = events.index(("commit",))
    after = events[first_commit + 1:]
    assert ("rung", "pm_a") in after and ("rung", "pm_b") in after, (
        "the first set's evaluations committed but no probe's rung was recomputed before the job "
        f"died: a killed run leaves rung 0 beside passing evidence (events {events})"
    )
