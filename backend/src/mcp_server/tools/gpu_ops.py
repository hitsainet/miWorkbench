"""Is the GPU free? (category: jobs) — MCP gap G9, 2026-10-21.

Before a GPU run, the end-to-end probe run of 2026-10-20 needed `ssh` + `nvidia-smi` on the node, `curl` to
miServer and miForge, and six miWorkbench MCP calls over five tools — and `list_gpus` answered only
static card facts. Three tools replace that:

* `get_gpu_status` — every card by UUID (or one): memory used/total, utilisation, compute
  processes, miWorkbench's lease on the card and the miWorkbench jobs that recorded it;
* `list_active_jobs` — every queued or running miWorkbench job, of EVERY kind, in one read;
* `gpu_preflight(gpu)` — both, plus miServer's resident model, lease and per-card memory, and
  miForge's active runs, with a verdict and the reasons for it.

STANDALONE FIRST. miServer and miForge are optional siblings: with no URL configured the preflight
says `configured: false` for that product and carries on — never an error. A configured product
that does not answer is `reachable: false` with the reason, and makes the verdict `uncertain`,
because "I could not ask" is not "it is idle".

THE PREFLIGHT IS NATIVE, NOT A `miserver_*` TOOL, deliberately: it must work on a server with no miServer
categories enabled. It builds its own short-timeout miServer client from `MISERVER_API_URL` and reads
two read-only routes (`/api/health/detailed`, `/api/health/gpus`) that `miserver_status` already
relies on; nothing here changes any product's state.

GPUs ARE NAMED BY UUID. Indices follow PCI bus order and moved on this estate in 2026-09; a tool
that accepted one could report on a different card than the caller meant.
"""

from typing import Annotated, Any, Optional

import httpx
from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import BackendError, MiWorkbenchClient
from ..config import MCPSettings
from ..miserver_client import MiServerClient

#: Sibling reads must not hold the agent up: a product that cannot answer in this long is
#: reported unreachable, and the verdict says so.
SIBLING_TIMEOUT_S = 8.0

GPU_UUID_PARAM = (
    "The card's UUID (GPU-…), from list_gpus or get_gpu_status. Indices are refused: they follow "
    "PCI bus order and can name a different card after a reboot"
)


def _miserver_client(url: str) -> MiServerClient:
    """The miServer client the preflight reads with. A function so a test can hand in a transport."""
    return MiServerClient(url, timeout=SIBLING_TIMEOUT_S)


def _forge_client(url: str) -> httpx.AsyncClient:
    """The miForge HTTP client. miForge has no client class here; two read-only GETs need none."""
    return httpx.AsyncClient(base_url=url, timeout=SIBLING_TIMEOUT_S)


def _norm(uuid: Optional[str]) -> str:
    text = (uuid or "").strip().lower()
    return text if text.startswith("gpu-") else f"gpu-{text}"


def _refusal(exc: BackendError) -> dict[str, Any]:
    return {"refused": True, "status_code": exc.status_code, "detail": exc.detail}


async def miserver_block(settings: MCPSettings, gpu: str) -> dict[str, Any]:
    """What miServer is doing with this card: resident model, lease, its memory on the card."""
    url = settings.miserver_api_url
    if not url:
        return {"configured": False,
                "detail": "MISERVER_API_URL is not set on this MCP server; miServer was not consulted"}
    miserver = _miserver_client(url)
    try:
        detailed = await miserver.get("/api/health/detailed")
        memory = await miserver.get("/api/health/gpus")
    except BackendError as exc:
        return {"configured": True, "reachable": False, "url": url, "reason": exc.detail}
    finally:
        await miserver.close()
    card = next((c for c in (memory or {}).get("cards") or []
                 if _norm(c.get("uuid")) == _norm(gpu)), None)
    return {
        "configured": True,
        "reachable": True,
        "model_loaded": (detailed or {}).get("model_loaded"),
        "model_name": (detailed or {}).get("model_name"),
        "model_placement": (detailed or {}).get("model_placement"),
        "lease": (detailed or {}).get("lease"),
        "weights": (detailed or {}).get("weights"),
        "card": card,
        "resident_on_this_card": resident_on_card(detailed or {}, card),
        "holds_context_on_this_card": holds_context(card),
    }


def resident_on_card(detailed: dict[str, Any], card: Optional[dict[str, Any]]) -> bool:
    """Is miServer's LOADED model on this card? Gap (ag), 2026-11-27.

    Only when miServer says a model is loaded (`model_loaded`, `GET /api/health/detailed`). Memory alone is not a
    model: torch keeps a few MB allocated after an unload (16 MB on the 3090, live), so a non-zero
    `millm_allocated_mb` with `model_loaded: false` is an idle server, not a resident model. Which card(s) the model
    is on comes from miServer's own answer: `engine_memory` on a card a GGUF model occupies, else the placement's
    `gpu_indices` (torch indices, which `/api/health/gpus` gives per card as `torch_index`). Only when a loaded
    model's placement is unreadable does this fall back to miServer's torch memory on the card."""
    if not card or detailed.get("model_loaded") is not True:
        return False
    if card.get("engine_memory") is not None:
        return True
    placement = detailed.get("model_placement")
    indices = placement.get("gpu_indices") if isinstance(placement, dict) else None
    if isinstance(indices, list):
        return card.get("torch_index") is not None and card.get("torch_index") in indices
    return bool(card.get("millm_allocated_mb"))


def holds_context(card: Optional[dict[str, Any]]) -> bool:
    """Does miServer's own process hold a CUDA context on this card? True when torch measured the card (miServer
    placed a model there in this process: the context outlives an unload) or a resident GGUF model occupies it.
    miServer's `processes` list is nvidia-smi's list for the WHOLE card, every pod's processes, so being listed
    there says nothing about who owns a pid; this is what says miServer has one."""
    return bool(card) and (card.get("torch_measured") is True or card.get("engine_memory") is not None)


def attribute_processes(studio_card: Optional[dict[str, Any]],
                        miserver_state: dict[str, Any]) -> dict[str, Any]:
    """Which of the card's processes miWorkbench could not attribute are miServer's own. Gap (ah), 2026-11-27.

    miServer runs in another pod, so miWorkbench's card reads its server process with `owner: null`. miServer's
    answer does not name its own pid (its list is the whole card's, and its own pid is in another PID namespace),
    so the attribution is by elimination: when miServer holds a context on this card it accounts for ONE process,
    and when that leaves exactly one unattributed process which miServer's list also shows, that pid is
    miServer's. With more than one unattributed process miServer still accounts for one, and the rest stay unknown
    (busy). Returns `{miserver_pid, miserver_state, unknown_count, unknown_pids}`; `miserver_state` is `idle_context`
    (no model loaded on this card) or `model_resident`, or null when miServer accounts for none."""
    processes = (studio_card or {}).get("processes") or []
    unknown = [p for p in processes if isinstance(p, dict) and p.get("owner") is None]
    card = miserver_state.get("card") if miserver_state.get("reachable") else None
    listed = {p.get("pid") for p in (card or {}).get("processes") or [] if isinstance(p, dict)}
    share = 1 if unknown and holds_context(card) and any(p.get("pid") in listed for p in unknown) else 0
    mine = unknown[0].get("pid") if share and len(unknown) == 1 else None
    state = None
    if share:
        state = "model_resident" if miserver_state.get("resident_on_this_card") else "idle_context"
    rest = [p.get("pid") for p in unknown if mine is None or p.get("pid") != mine]
    return {"miserver_pid": mine, "miserver_state": state, "unknown_count": len(unknown) - share,
            "unknown_pids": rest}


def _card_reasons(studio_card: dict[str, Any], attribution: dict[str, Any]) -> list[str]:
    """The card's own busy reasons, with its process line re-counted after miServer's share.

    `gpu_status.assemble` writes the lease line, then the job lines, then ONE process line LAST, present exactly when
    a process has `owner: null` (a test runs the real `assemble` through this to pin that order)."""
    reasons = list(studio_card.get("in_use_by") or [])
    processes = studio_card.get("processes") or []
    if not any(isinstance(p, dict) and p.get("owner") is None for p in processes):
        return reasons
    reasons = reasons[:-1]
    if attribution["unknown_count"] > 0:
        extra = " beside miServer's own CUDA context" if attribution["miserver_state"] else ""
        reasons.append(
            f"{attribution['unknown_count']} compute process(es) not attributable to a miWorkbench job or to "
            f"miServer{extra} (pids among {attribution['unknown_pids']}; another pod or app, e.g. miForge)")
    return reasons


async def miforge_block(settings: MCPSettings) -> dict[str, Any]:
    """miForge's pending and running training runs (miForge's run list names no card)."""
    url = settings.miforge_api_url
    if not url:
        return {"configured": False,
                "detail": "MIFORGE_API_URL is not set on this MCP server; miForge was not consulted"}
    forge = _forge_client(url)
    runs: list[dict[str, Any]] = []
    try:
        for status in ("running", "pending"):
            response = await forge.get("/api/v1/runs", params={"status": status, "limit": 200})
            if response.status_code >= 400:
                return {"configured": True, "reachable": False, "url": url,
                        "reason": f"miForge answered {response.status_code} on /api/v1/runs"}
            try:
                body = response.json()
            except ValueError:
                return {"configured": True, "reachable": False, "url": url,
                        "reason": "miForge returned non-JSON from /api/v1/runs (a misrouted "
                                  "ingress or proxy page) — NOT an empty run list"}
            listed = body.get("runs") if isinstance(body, dict) else None
            if not isinstance(listed, list):
                # miForge's GET /api/v1/runs answers `{runs: [...], total, ...}`. Any other shape
                # is not "no runs": say so rather than read it as empty.
                return {"configured": True, "reachable": False, "url": url,
                        "reason": "miForge's /api/v1/runs answer has no `runs` list"}
            runs += [r for r in listed if isinstance(r, dict)]
    except httpx.HTTPError as exc:
        return {"configured": True, "reachable": False, "url": url,
                "reason": f"miForge unreachable: {type(exc).__name__}: {exc}"}
    finally:
        await forge.aclose()
    return {"configured": True, "reachable": True, "active_runs": runs,
            "active_run_count": len(runs)}


def verdict(studio_card: Optional[dict[str, Any]], studio_reason: Optional[str],
            queued_auto: list[dict[str, Any]], miserver_state: dict[str, Any],
            forge_state: dict[str, Any],
            worker_watch: Optional[dict[str, Any]] = None) -> tuple[str, list[str], list[str]]:
    """`(verdict, busy_reasons, uncertain_reasons)`. `clear` only when nothing says otherwise."""
    attribution = attribute_processes(studio_card, miserver_state)
    busy: list[str] = []
    uncertain: list[str] = []
    # A wedged worker takes nothing, so a job submitted to this card may never start; an unreadable watch is not "fine".
    if (worker_watch or {}).get("state") == "wedged":
        uncertain.append(str(worker_watch.get("message")))
    elif (worker_watch or {}).get("state") == "unknown":
        uncertain.append(f"miWorkbench could not read its GPU queues: {worker_watch.get('reason')}")
    if studio_card is None:
        uncertain.append(studio_reason or "miWorkbench cannot see this card")
    else:
        busy += _card_reasons(studio_card, attribution)
    if queued_auto:
        uncertain.append(
            f"{len(queued_auto)} queued miWorkbench job(s) have no card yet and may take this one "
            "when they start: " + ", ".join(f"{j['kind']} {j['id']}" for j in queued_auto[:5]))
    if miserver_state.get("configured"):
        if not miserver_state.get("reachable"):
            uncertain.append(f"miServer is configured but did not answer: {miserver_state.get('reason')}")
        elif miserver_state.get("resident_on_this_card"):
            busy.append(f"miServer's resident model {miserver_state.get('model_name')!r} is on this card")
    if forge_state.get("configured"):
        if not forge_state.get("reachable"):
            uncertain.append(f"miForge is configured but did not answer: {forge_state.get('reason')}")
        elif forge_state.get("active_run_count"):
            uncertain.append(
                f"miForge has {forge_state['active_run_count']} pending/running run(s); its run list "
                "does not name a card, so check them before assuming this one is free")
    if busy:
        return "busy", busy, uncertain
    return ("uncertain" if uncertain else "clear"), busy, uncertain


def register(mcp: MCPServer, client: MiWorkbenchClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def get_gpu_status(
        gpu: Annotated[Optional[str], Field(description=f"Only this card. {GPU_UUID_PARAM}. Omit for every card")] = None,
    ) -> Any:
        """Live state of every GPU miWorkbench can see (one or many), by UUID: memory used / free /
        total (MiB), utilisation, compute processes, miWorkbench's GPU lease on the card and the
        miWorkbench jobs that recorded it, plus `in_use` and the reasons.

        A process is attributed only when it is a miWorkbench steering worker; miServer, miForge and
        miWorkbench's own worker run in other pods, so their PIDs show `owner: null`. For "can I
        start a GPU job on this card?" use gpu_preflight, which also asks miServer and miForge.

        `worker_watch.state` says whether a worker has stopped taking work: `wedged` means a task ended and was never
        acked, messages are waiting and no card is held, so nothing will start until the operator restarts that worker
        (`message` says what to do; nothing restarts it automatically). `suspect` is the same with less certainty, `ok`
        is what a healthy worker reads however long its task runs, and `unknown` means the broker could not be read."""
        return await client.get("/operations/gpu-status", gpu=gpu)

    @mcp.tool()
    async def list_active_jobs(
        kind: Annotated[Optional[str], Field(description="Only this kind, e.g. probe_training, probe_judge, sae_training, feature_extraction, circuit_capture, task_queue, steering_worker. An unknown kind is refused with the list")] = None,
        gpu: Annotated[Optional[str], Field(description=f"Only jobs that recorded this card. {GPU_UUID_PARAM}")] = None,
    ) -> Any:
        """Every miWorkbench job that is queued or running, of EVERY kind, in one call: SAE training,
        extraction, activation extraction, labeling (all three), tokenization, downloads, probe
        training / evaluation / judge, circuit capture / discovery / attribution / validation /
        faithfulness / calibration, steering recordings and workers, Neuronpedia, grouping, and
        the generic task queue. A probe evaluation or WSP held-out capture is a `task_queue` job (`task_type`
        probe_evaluation or wsp_heldout_capture, `entity_id` the probe) with `progress` and, while running,
        `progress_detail` (set_index of set_count, rows_done of rows_total, rate, ETA). Work on the GPU queues with no
        row of its own (a re-cut, a score) reads
        `kind: gpu_queue_message`, named by its message, `state` queued, parked or delivered. A waiting message whose
        own row is gone (a deleted run) or has ended reads `status: "orphaned"` with `orphan_reason` (`row_missing`,
        `row_cancelled`, …): the worker will refuse it, so it is not work waiting for a card. `worker_watch` is as
        in get_gpu_status.

        `errors` names any source that could not be read — then the list may be incomplete, which
        is not the same as nothing running."""
        return await client.get("/operations/active-jobs", kind=kind, gpu=gpu)

    @mcp.tool()
    async def gpu_preflight(
        gpu: Annotated[str, Field(description=GPU_UUID_PARAM, min_length=4)],
    ) -> Any:
        """"Is this card free for a GPU job?" in one call, across the suite.

        Combines miWorkbench's live card state and every active miWorkbench job with miServer's resident
        model, lease and per-card memory (when MISERVER_API_URL is set) and miForge's pending and
        running runs (when MIFORGE_API_URL is set). Returns `verdict`:

        * `clear` — nothing holds the card and every configured product answered;
        * `busy` — `busy_reasons` names what holds it (a lease, a running job, miServer's LOADED
          model, an unattributed process using memory). miServer's idle server, holding only its CUDA
          context with no model loaded, is attributed to miServer (its block's `process_attribution`) and
          does not make a card busy;
        * `uncertain` — nothing is known to hold it, but `uncertain_reasons` says what could not
          be ruled out (a product that did not answer, queued jobs with no card yet, miForge runs).

        A sibling that is not configured is reported `configured: false` and does not affect the
        verdict: miWorkbench works without miServer and miForge. Read-only; starts nothing."""
        if not gpu.strip().lower().startswith("gpu-"):
            return {"refused": True, "detail": f"{gpu!r} is not a GPU UUID. {GPU_UUID_PARAM}."}
        try:
            studio = await client.get("/operations/gpu-status", gpu=gpu)
            jobs = await client.get("/operations/active-jobs")
        except BackendError as exc:
            return {"gpu": gpu, **_refusal(exc)}
        cards = studio.get("cards") or []
        card = cards[0] if cards else None
        queued_auto = [
            j for j in jobs.get("jobs") or []
            if j.get("status") in ("queued", "pending") and not j.get("gpu_uuids")
        ]
        miserver_state = await miserver_block(settings, gpu)
        if miserver_state.get("reachable"):
            miserver_state["process_attribution"] = attribute_processes(card, miserver_state)
        forge_state = await miforge_block(settings)
        result, busy, uncertain = verdict(card, studio.get("reason"), queued_auto, miserver_state,
                                          forge_state, studio.get("worker_watch"))
        return {
            "gpu": gpu,
            "verdict": result,
            "busy_reasons": busy,
            "uncertain_reasons": uncertain,
            "mistudio": {
                "card": card,
                "reason": studio.get("reason"),
                "active_jobs": jobs.get("jobs") or [],
                "job_read_errors": jobs.get("errors") or [],
                "worker_watch": studio.get("worker_watch"),
            },
            "millm": miserver_state,
            "miforge": forge_state,
            "read_at": jobs.get("read_at"),
        }
