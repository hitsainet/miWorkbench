"""The WSP worker is deployed (ENH-010 044 task 3.6; FTDD §11; T44-D2): k8s and compose both run a container consuming
ONLY `wsp`, solo, one at a time, with no GPU environment and miDataworks' URL (a bundle reads sensitivity live, P41);
the CPU tasks route to `wsp` and the held-out capture to `extraction` (the GPU queue of every probe stage).

Control (0xcc/reviews/044_implementation_controls_2026-11-16.md): remove `wsp` from the container's CELERY_QUEUES → red.
R3 review fix (OR3-044a-M1), 2026-11-17: drop `src.workers.wsp_tasks` from `autodiscover_tasks` →
`test_a_worker_registers_the_wsp_tasks` red (review survivor X20: the routes import the module lazily, so every
in-process test stayed green while a worker would answer each queued task "unregistered").
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
CPU_TASKS = ("src.workers.wsp_tasks.build_wsp_bundle", "src.workers.wsp_tasks.import_wsp_candidate")
GPU_TASKS = ("src.workers.wsp_tasks.capture_wsp_heldout",)


def _k8s_containers():
    out = {}
    for doc in yaml.safe_load_all((REPO / "k8s" / "base" / "backend.yaml").read_text()):
        if doc and doc.get("kind") == "Deployment":
            for container in doc["spec"]["template"]["spec"]["containers"]:
                out[container["name"]] = container
    return out


def _queues(env_value) -> set[str]:
    return {q for q in str(env_value or "").split(",") if q}


def test_k8s_runs_the_wsp_worker():
    container = _k8s_containers().get("celery-worker-wsp")
    assert container is not None, "no celery-worker-wsp container in k8s/base/backend.yaml"
    env = {e["name"]: e.get("value") for e in container["env"]}
    assert _queues(env["CELERY_QUEUES"]) == {"wsp"} and env["CELERY_WORKER_NAME"] == "wsp"
    assert (env["CELERY_POOL"], env["CELERY_CONCURRENCY"]) == ("solo", "1")
    assert not any(k.startswith(("NVIDIA", "CUDA")) for k in env), "the WSP worker sees no card"
    assert env["GPU_WORKER_MODE"] == "per_card"
    assert env["MIDATAWORKS_API_URL"], "a bundle reads each source's sensitivity live (P41)"
    assert {m["mountPath"] for m in container["volumeMounts"]} == {"/data"}


def test_no_other_k8s_worker_consumes_wsp():
    consumers = [name for name, c in _k8s_containers().items()
                 if "wsp" in _queues({e["name"]: e.get("value") for e in c.get("env", [])}.get("CELERY_QUEUES"))]
    assert consumers == ["celery-worker-wsp"]


def test_compose_runs_the_wsp_worker():
    service = yaml.safe_load((REPO / "docker-compose.yml").read_text())["services"]["celery-worker-wsp"]
    env = service["environment"]
    assert _queues(env["CELERY_QUEUES"]) == {"wsp"} and env["CELERY_POOL"] == "solo"
    assert str(env["CELERY_CONCURRENCY"]) == "1" and "MIDATAWORKS_API_URL" in env


def test_the_tasks_route_to_their_queues():
    """Resolved through Celery's own router (what a `.delay()` uses), by exact task name."""
    from src.core.celery_app import celery_app

    def resolved(name):
        queue = celery_app.amqp.router.route({}, name, (), {}, {}).get("queue")
        return getattr(queue, "name", queue)

    assert {name: resolved(name) for name in CPU_TASKS + GPU_TASKS} == {
        **{name: "wsp" for name in CPU_TASKS}, **{name: "extraction" for name in GPU_TASKS}}


def test_a_worker_registers_the_wsp_tasks():
    """A worker registers only what `celery_app` autodiscovers; read in a fresh interpreter that imports nothing but
    the Celery app (the janitor's twin, `test_wsp_janitor.py::test_a_worker_registers_the_janitor`). Without it the
    `wsp` worker would leave every bundle `building` and the extraction worker every capture `pending`."""
    from tests.unit.test_training_evaluation_task_registered_in_worker import _worker_registry

    registry = _worker_registry()
    assert "train_sae" in registry, "the probe read an empty registry"
    missing = [name for name in CPU_TASKS + GPU_TASKS if name not in registry]
    assert missing == [], f"a worker does not register {missing}"
