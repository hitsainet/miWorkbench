"""
miServer model management (category: miserver_models) — acquisition, residency and
disposal of the weights miServer SERVES.

Written because the workflow that prompted it was unreachable over MCP:
"unload the current gguf, download and load the IQ4 version, test it." Every
step of that existed as a REST route and none of it as a tool, so the whole
sequence had to be done with `curl` inside the pod.

The asymmetry with miWorkbench's `models` category is deliberate and load-bearing:
**miServer holds ONE model resident at a time.** Loading evicts whatever is there.
miWorkbench has no residency at all — it loads per task — so it has no
load/unload tools and no eviction to warn about.

All tools are health-gated: when miServer is unreachable they return
{"unavailable": MISERVER.name, "reason": …} rather than raising (contract §3).
"""

from typing import Annotated, Any, Literal, Optional, Union

from mcp.server.mcpserver import MCPServer
from pydantic import Field, StrictInt

from ..client import BackendError
from ..context import ExternalContext
from ..health_gate import HealthGate, gated
from ..miserver_client import MiServerClient, error_fields
from ..miserver_send import refusal
from .miserver_approvals import RELAYED_ACTIONS, ApprovalQueueCheck, moved_refusal, relayed
from ..products import MISERVER

#: miServer's lease TTL bound (miServer FR-29.1.2, contract §4e): also its default.
MAX_LEASE_TTL_S = 7200

# ── The GGUF context ceiling (miServer contract v1.30, 2026-11-21) ──────────────────────────────────
#
# A token count, or "max" for as large a window as fits on the card(s) the model is placed on, up to
# the length the model declares. miServer also reads 0 as max; the tools take the one spelling, so a
# 0 is refused here with that fix rather than meaning two things. A RECORD reports max as 0.

#: The smallest ceiling miServer accepts (its `context_ceiling.MIN_CONTEXT`, equal to the loader's
#: `GGUF_MIN_CONTEXT`); pinned to miServer's value by the cross-repo test in
#: test_miserver_context_ceiling.py.
MIN_CONTEXT_CEILING = 2048
CEILING_MAX = "max"

#: The approval action a v1.30 miServer lists in `GET /api/approvals/actions`. An older miServer
#: IGNORES an unknown body field and would load at its own default with nobody noticing, so a load
#: that names a ceiling is refused unless this is listed (contract v1.30, `miserver_load_model` row).
CONTEXT_CEILING_MARKER = "model_context_write"

#: The argument type: a whole number of tokens or the word "max" (strict: `true` and "65536" are
#: refused by the schema, never read as 1 or as a number).
ContextCeiling = Union[StrictInt, Literal["max"]]


def context_ceiling_refusal(value: Any, *, allow_none: bool = True) -> Optional[dict[str, Any]]:
    """`None` when `value` is a ceiling miServer can take; otherwise the refusal (nothing was sent).

    miServer refuses a ceiling above the model's declared length, or on a model that is not GGUF,
    itself (422 before anything is filed); this checks only what needs no model."""
    if value is None and allow_none:
        return None
    if value == CEILING_MAX and isinstance(value, str):
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value >= MIN_CONTEXT_CEILING:
        return None
    fix = (f'Pass a number of tokens of at least {MIN_CONTEXT_CEILING} (e.g. 65536), "max" for as '
           "large a window as fits on the card, up to the model's declared length, or null for the "
           "model's own setting (else miServer's GGUF_CONTEXT_LENGTH).")
    if value == 0 and not isinstance(value, bool):
        fix = '0 means "as large as fits" to miServer; pass "max" to say that. ' + fix
    return {
        "refused": "invalid_context_ceiling",
        "context_ceiling": value,
        "reason": (f"context_ceiling must be a whole number of tokens >= {MIN_CONTEXT_CEILING} or "
                   f'"max", not {value!r}. Nothing was sent.'),
        "fix": fix,
    }


def register(
    mcp: MCPServer, miserver: MiServerClient, gate: HealthGate, ctx: ExternalContext
) -> None:
    #: ENH-008: the fail-closed check that this miServer has its own approval queue.
    queue = ApprovalQueueCheck(miserver, gate)

    async def _actual_context() -> Optional[int]:
        """The context the engine ACTUALLY got, not the one that was asked for.

        A GGUF model loads at the largest context that fits: 8192 requested and
        4096 obtained on both quantizations tried on the 24 GB card here. A
        4703-token prompt against a 4096 window is the next failure a user hits,
        and nothing else on the system explains it — so every tool that reports
        a loaded model reports this beside it.
        """
        try:
            info = await miserver.get("/api/health/inference")
        except Exception:
            return None
        if not isinstance(info, dict):
            return None
        value = info.get("context_length")
        return int(value) if isinstance(value, int) and value > 0 else None

    @mcp.tool()
    @gated(gate, MISERVER.name)
    async def miserver_list_models(
        loaded_only: Annotated[bool, Field(description="Return only the resident model")] = False,
    ) -> Any:
        """What models miServer has, and which one is serving right now.

        Start here. The answer to "why won't it work?" is usually in this
        output: a model that is `error`, a download stuck at an unknown
        percentage, or a resident model with a smaller context window than the
        prompt being sent.

        Statuses are `ready` (on disk), `loaded` (resident and serving),
        `downloading`, `loading` and `error`. Exactly one row can be `loaded`.

        Each row carries `gguf_label` (Q4_K_M, IQ4_XS) where the model is a
        GGUF quantization, and the model's `name` embeds it as `repo:QUANT` —
        two quantizations of one repository are DIFFERENT MODELS with different
        sizes and quality, and both are separately selectable.

        `context_length` on the resident model is the context the engine
        ACTUALLY obtained, which is often smaller than the file declares.

        Since miServer contract v1.30 each row also carries, passed through unchanged:
        `context_ceiling` (the model's stored ceiling in tokens; 0 = max, as large as fits;
        null = miServer's GGUF_CONTEXT_LENGTH) and, on the resident model only,
        `kv_cache_type` (the KV-cache type the context was created with: q8_0, f16, …; f16 when
        the configured type was refused), `supports_embeddings` (false when a GGUF model was
        loaded without them) and `gguf_load` (why: the cache and embeddings choices with their
        reasons, the ceiling and its source, the declared length, the predicted fit and every
        construction tried). Change the stored ceiling with
        miserver_set_model_context_ceiling; name one for a single load with miserver_load_model.
        """
        result = await miserver.get("/api/models")
        rows = result.get("data") if isinstance(result, dict) else result
        if not isinstance(rows, list):
            return result

        # Read the live context ONLY if a resident row does not carry it already. It is a
        # property of the running engine, so with nothing loaded there is nothing to ask about
        # and the extra round trip buys a null. Since miServer contract v1.30 the record itself
        # carries `context_length` for the loaded model (the same figure, from the same loaded
        # instance); an older miServer's record does not, and /api/health/inference answers.
        need_engine = any(
            isinstance(r, dict) and r.get("status") == "loaded" and not _positive_int(r.get("context_length"))
            for r in rows
        )
        context = await _actual_context() if need_engine else None
        out = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if loaded_only and row.get("status") != "loaded":
                continue
            row = dict(row)
            if row.get("status") == "loaded":
                # Reported ONLY for the resident model: it is a property of the
                # live engine, not of the row, and attaching it to a model that
                # is merely on disk would state a context nothing has measured.
                own = row.get("context_length")
                row["context_length"] = own if _positive_int(own) else context
                if row["context_length"] is None:
                    row["context_length_note"] = (
                        "could not be read from /api/health/inference; the "
                        "model is loaded but its actual window is unknown"
                    )
            out.append(_with_progress_note(row))
        return {"models": out, "count": len(out)}

    @mcp.tool()
    @gated(gate, MISERVER.name)
    async def miserver_get_model(
        model_id: Annotated[int, Field(description="Model row id from miserver_list_models")],
    ) -> Any:
        """One model's full record — status, size, quantization, and download
        progress if it is still downloading.

        Poll this after miserver_download_model. `download_progress` is in-process
        and reads null when the download is running in a different worker, so a
        null percentage means "progress unknown", NOT "no progress" — this tool
        says which.

        Since miServer contract v1.30 the record carries, unchanged: `context_ceiling` (stored;
        0 = max, null = miServer's GGUF_CONTEXT_LENGTH) and, when this model is loaded,
        `context_length` (the window it was ACTUALLY created with), `kv_cache_type`,
        `supports_embeddings` and `gguf_load` (why the load got its cache type, embeddings and
        window: the ceiling and its source, the declared length, the prediction, every attempt).
        """
        result = await miserver.get(f"/api/models/{model_id}")
        row = result.get("data") if isinstance(result, dict) else result
        return _with_progress_note(row) if isinstance(row, dict) else result

    @mcp.tool()
    @gated(gate, MISERVER.name)
    async def miserver_preview_model_repo(
        repo_id: Annotated[str, Field(description="HuggingFace repo id, e.g. mradermacher/gemma-4-31b-it-GGUF")],
        hf_token: Annotated[Optional[str], Field(description="Token for a gated repo; never logged. Usually OMIT it: miServer then uses the token saved on its Settings page (then its HF_TOKEN env), so no credential passes through the agent transcript")] = None,
    ) -> Any:
        """Inspect a HuggingFace repo WITHOUT downloading it.

        **Credentials:** omit `hf_token` and miServer uses the token the operator SAVED on its
        Settings page (falling back to its `HF_TOKEN` environment); no route returns that
        token, and this server never asks for one. Pass `hf_token` only to override it.

        **Call this before miserver_download_model on any GGUF repo.** A GGUF
        repository holds many mutually exclusive quantizations, and downloading
        the repo means downloading all of them: 121 GB where the one you want
        is 2.78 GB. That is measured, not illustrative.

        `gguf_quants` lists each selectable quantization with its MEASURED
        total size. Pass a chosen entry's `label` and its files' `path` values
        straight into miserver_download_model — they are the input that tool
        expects, so no filename has to be guessed.

        A quantization may be SPLIT across numbered parts (`is_split`). Its
        `files` list is then several entries and EVERY one is required;
        downloading a subset produces a directory that looks complete and a
        model that cannot load.

        `gguf_companions` are projectors and similar side files. They are NOT
        quantizations — an `mmproj-f16.gguf` is a vision projector, and on a
        31B repo it is a 1.2 GB file that would read like a bargain and serve
        nothing. Download one only alongside the quantization it accompanies.
        """
        body: dict[str, Any] = {"repo_id": repo_id}
        if hf_token:
            body["hf_token"] = hf_token
        return await miserver.post("/api/models/preview", json_body=body)

    @mcp.tool()
    @gated(gate, MISERVER.name)
    async def miserver_download_model(
        repo_id: Annotated[str, Field(description="HuggingFace repo id")],
        quantization: Annotated[str, Field(description="Coarse bucket: Q2, Q4, Q8, FP16, FP32. For a GGUF repo pass the bucket matching gguf_label")] = "Q4",
        gguf_label: Annotated[Optional[str], Field(description="Exact quantization label from miserver_preview_model_repo, e.g. Q4_K_M or IQ4_XS. Required to pick ONE quant from a GGUF repo")] = None,
        gguf_files: Annotated[Optional[list[str]], Field(description="EVERY file path of that one quantization, from the preview. A subset yields a model that cannot load")] = None,
        custom_name: Annotated[Optional[str], Field(description="Display name. Omit to get `repo:QUANT`, which keeps quantizations distinguishable")] = None,
        revision: Annotated[Optional[str], Field(description="Git revision to pin")] = None,
        trust_remote_code: Annotated[bool, Field(description="Required by some architectures; executes repo code at load time")] = False,
        hf_token: Annotated[Optional[str], Field(description="Token for a gated repo; never logged. Usually OMIT it: miServer then uses the token saved on its Settings page (then its HF_TOKEN env)")] = None,
    ) -> Any:
        """Start downloading a model. Returns immediately with the new row's id.

        **The served name is tagged by precision** (miServer D56, 2026-10-18): a non-GGUF
        registration at any precision but the native FP16 is served as `repo:Q8`, `repo:Q4` or
        `repo:FP32` — two precisions of one repository are different models — and a GGUF one as
        `repo:QUANT` (`repo:Q4_K_M`). The native registration keeps the bare name, unless a
        legacy untagged row already holds it, in which case the NEW row is tagged (`repo:FP16`)
        so the existing name never moves under its clients. A name clash is REFUSED, never
        silently suffixed: `{"refused": "model_name_taken", "hint", …}` — re-call with a
        `custom_name` of your own.

        **Credentials:** omit `hf_token` and miServer uses the token saved on its Settings page
        (then its `HF_TOKEN` environment); no credential passes through the agent transcript.

        **This is long and large.** Poll `miserver_get_model(model_id)` rather
        than waiting; a quantization of a 31B model is ~16 GB and the whole of
        a GGUF repo can be 121 GB.

        For a GGUF repo, call miserver_preview_model_repo first and pass the
        chosen quantization's `label` as `gguf_label` and its file `path`s as
        `gguf_files`. Omitting them downloads EVERY quantization in the repo.

        Leave `custom_name` unset unless you have a reason. The default name is
        `repo:QUANT`, which is what makes two quantizations of one repository
        separately addressable — a custom name is not tagged, so two custom
        names that collide are indistinguishable to every OpenAI client.

        Downloading does not load. Follow with miserver_load_model.
        """
        body: dict[str, Any] = {
            "source": "huggingface",
            "repo_id": repo_id,
            "quantization": quantization,
            "trust_remote_code": bool(trust_remote_code),
        }
        if gguf_label:
            body["gguf_label"] = gguf_label
        if gguf_files:
            body["gguf_files"] = list(gguf_files)
        if custom_name:
            body["custom_name"] = custom_name
        if revision:
            body["revision"] = revision
        if hf_token:
            body["hf_token"] = hf_token
        # ⚠ "/api/models", NOT "/api/models/download". miServer serves the download as
        # POST on the collection; there is no /download sub-path and never was, so this
        # tool 404'd on every call an agent ever made. Found 2026-09-27 by pointing the
        # cross-repo path guard at this module for the first time — miWorkbench's own caller
        # assertion could not see it, because the tool and its documented path agreed
        # with each other and both were wrong.
        try:
            return await miserver.post("/api/models", json_body=body)
        except BackendError as exc:
            # 409 MODEL_NAME_TAKEN (miServer `_unique_served_name`) is a request refusal with a
            # one-line remedy, not a fault — shaped like every other refusal (miserver_send).
            refused = refusal(exc)
            if refused is not None and refused["refused"] == "model_name_taken":
                return refused
            raise

    @mcp.tool()
    @gated(gate, MISERVER.name)
    async def miserver_cancel_download(
        model_id: Annotated[int, Field(description="Model row id of the download to stop")],
    ) -> Any:
        """Stop an in-progress download and clean up its partial files."""
        return await miserver.post(f"/api/models/{model_id}/cancel")

    @mcp.tool()
    async def miserver_load_model(
        model_id: Annotated[int, Field(description="Model row id from miserver_list_models")],
        approval_request_id: Annotated[Optional[str], Field(description="DEPRECATED (ENH-008): miServer now decides its own requests. Passing it is refused with approval_moved_to_millm; poll miserver_get_approval_status instead")] = None,
        lease_id: Annotated[Optional[str], Field(description="Your lease id from miserver_acquire_lease, when you hold the lease; sent as the X-miLLM-Lease header. miServer keeps it server-side until the operator decides and never returns it")] = None,
        context_ceiling: Annotated[Optional[ContextCeiling], Field(description=f'GGUF models only: the most context to load with, for THIS load only: a number of tokens (at least {MIN_CONTEXT_CEILING}) or "max" for as large a window as fits on the card(s), up to the length the model declares. Omit (null) for the model\'s own setting (miserver_set_model_context_ceiling), else miServer\'s GGUF_CONTEXT_LENGTH. Shown on the operator\'s approval card and covered by its request_digest; needs miServer contract v1.30')] = None,
    ) -> Any:
        """Make a model the one miServer serves. **This EVICTS whatever is resident**, so it waits for
        the operator in **miServer's** Approvals page.

        The call files the request in miServer and returns miServer's pending answer unchanged, plus
        `"decide_in": "miserver"`: `{approval_id, status: "pending", action: "model_load", summary,
        request_digest, expires_at, hint, decide_in}`. Nothing is loaded yet. When the operator
        approves, miServer runs exactly this load ONCE, by itself — poll
        `miserver_get_approval_status(approval_id)` (`executed`, `failed`, `rejected` or `expired`),
        then miserver_list_models until the model is `loaded`. Never call this again to "use" an
        approval; a second call files a second request.

        A miServer without its own approval queue is REFUSED here (`millm_has_no_approval_queue`) and
        nothing is sent: it would load with nobody approving.

        miServer serves one model at a time, so loading is also an unload of the current one, and any
        SAE attached to it goes with it. miServer can still refuse the approved load when it runs — a
        model LOCKED for steering, or a LEASE held by someone else (pass `lease_id` if it is yours)
        — and History then shows the error.

        **The context ceiling** (`context_ceiling`, GGUF models only, miServer contract v1.30): the most
        context this one load may use, in tokens, or "max" for as large a window as fits on the
        card(s) the model is placed on (miServer starts at its predicted fit, walks down if that
        fails and probes upward if it loads). Omit it and the model's stored ceiling applies
        (miserver_set_model_context_ceiling), else miServer's GGUF_CONTEXT_LENGTH. It is sent as the
        JSON body `{"context_ceiling": …}`, so the operator's card shows a "Context ceiling" line and
        the `request_digest` covers it; the load still waits for the operator. A larger window
        reserves more of the card, which miWorkbench's own GPU work shares. Refusals: a value below
        2048, 0, or anything that is not a number or "max" (`invalid_context_ceiling`, nothing
        sent); a miServer that predates v1.30, which would IGNORE the field and load at its own
        default (`miserver_context_ceiling_unsupported` with a `fix`, nothing sent); miServer's own
        422 for a ceiling above the model's declared length or on a transformers model
        (`millm_refused`, nothing filed).

        The context the engine obtains may be smaller than the file declares (or than the ceiling);
        once loaded, miserver_get_model reports `context_length`, `kv_cache_type`,
        `supports_embeddings` and `gguf_load` (why), and miserver_list_models the actual window.
        """
        if approval_request_id is not None:
            return moved_refusal("miserver_load_model")
        refused = context_ceiling_refusal(context_ceiling)
        if refused is not None:
            return refused
        ok, reason = await gate.check(MISERVER.name)
        if not ok:
            return {"unavailable": MISERVER.name, "reason": reason}
        refused = await queue.refusal(RELAYED_ACTIONS["miserver_load_model"])
        if refused is not None:
            return refused
        if context_ceiling is not None:
            # Read from the same cached action list as the queue check above: one GET.
            refused = await queue.capability_refusal(
                CONTEXT_CEILING_MARKER, code="miserver_context_ceiling_unsupported",
                feature="contract v1.30 (a context ceiling on a load)",
                fix=("Load without context_ceiling (miServer then uses the model's stored ceiling, "
                     "else its GGUF_CONTEXT_LENGTH), or upgrade miServer to a release whose "
                     "GET /api/approvals/actions lists model_context_write (contract v1.30)."),
            )
            if refused is not None:
                return refused
        # The lease id travels ONLY in the header: miServer keeps it in the request's secret payload
        # until the decision and never returns it (ENH-008 §3.1, L8). The ceiling travels ONLY in the
        # body, and only when named: no body is miServer's default, exactly as before v1.30.
        headers = {"X-miLLM-Lease": lease_id} if lease_id else None
        body = {"json_body": {"context_ceiling": context_ceiling}} if context_ceiling is not None else {}
        try:
            response = await miserver.post(
                f"/api/models/{model_id}/load", headers=headers, full=True, **body
            )
        except BackendError as exc:
            if exc.status_code == 0:
                gate.invalidate(MISERVER.name)
                return {"unavailable": MISERVER.name, "reason": str(exc)}
            if exc.status_code == 422:
                # v1.30: miServer refuses a ceiling it cannot apply before anything is filed.
                return {"refused": "millm_refused", "status_code": exc.status_code,
                        **error_fields(exc.status_code, exc.detail)}
            raise
        pending = relayed(response)
        if pending is not None:
            return pending
        result = response.body
        if isinstance(result, dict) and result.get("error"):
            return result
        return {
            "load_started": True,
            "model": result.get("data") if isinstance(result, dict) else result,
            "next": (
                "poll miserver_list_models until status is 'loaded', then read "
                "context_length before sending long prompts"
            ),
        }

    @mcp.tool()
    async def miserver_set_model_context_ceiling(
        model_id: Annotated[int, Field(description="Model row id from miserver_list_models (a GGUF model)")],
        context_ceiling: Annotated[Optional[ContextCeiling], Field(description=f'The most context this model is loaded with at EVERY later load: a number of tokens (at least {MIN_CONTEXT_CEILING}), "max" for as large a window as fits on the card(s), up to the declared length, or null to go back to miServer\'s GGUF_CONTEXT_LENGTH')],
    ) -> Any:
        """Ask to change a GGUF model's stored context ceiling in miServer (contract v1.30): the most
        context it is loaded with at every later load, the OpenAI path's auto-loads included.

        **This waits for the operator.** The call files the change in **miServer's** Approvals page
        (action `model_context_write`; "Approve all settings changes" may approve it with others) and
        returns miServer's pending answer unchanged plus `"decide_in": "miserver"`; nothing changes
        yet. When the operator approves, miServer stores exactly this value once, by itself. Poll
        `miserver_get_approval_status(approval_id)`; never call this again to "use" an approval.

        Nothing is loaded or unloaded: the loaded window changes at the model's NEXT load, which
        is its own approval (miserver_load_model). To change the window for one load only, pass
        `context_ceiling` to miserver_load_model instead. "max" asks for as large a window as fits,
        which reserves most of the card. The record (miserver_get_model) reports the stored value
        as `context_ceiling`, with max as 0.

        Refusals: a value below 2048, 0, or anything that is not a number, "max" or null
        (`invalid_context_ceiling`, nothing sent); a miServer whose queue does not gate
        `model_context_write`, i.e. one before v1.30 (`millm_has_no_approval_queue`, nothing sent);
        miServer's own 4xx (a transformers model, a ceiling above the declared length, an unknown
        model) as `{"refused": "millm_refused", "status_code", "code", "message", "details"}`."""
        refused = context_ceiling_refusal(context_ceiling)
        if refused is not None:
            return refused
        ok, why = await gate.check(MISERVER.name)
        if not ok:
            return {"unavailable": MISERVER.name, "reason": why}
        refused = await queue.refusal(RELAYED_ACTIONS["miserver_set_model_context_ceiling"])
        if refused is not None:
            return refused
        try:
            response = await miserver.put(
                f"/api/models/{model_id}/context-ceiling",
                json_body={"context_ceiling": context_ceiling}, full=True,
            )
        except BackendError as exc:
            if exc.status_code == 0:
                gate.invalidate(MISERVER.name)
                return {"unavailable": MISERVER.name, "reason": str(exc)}
            if 400 <= exc.status_code < 500:
                return {"refused": "millm_refused", "status_code": exc.status_code,
                        **error_fields(exc.status_code, exc.detail)}
            raise
        return relayed(response) or response.body

    # ── Leases (Feature 034, FR-18; miServer Feature 29) ─────────────────────────────────────
    #
    # A lease pins the RESIDENT model for a holder: while it is live, nobody else's load evicts
    # it and nobody else's request auto-loads over it. It loads nothing, so it needs no approval
    # (P-05). The lease id is the only proof of holding: it is returned once, by the grant, and
    # travels afterwards ONLY in the `X-miLLM-Lease` header — never a path, a body, an approval
    # payload or a log line.

    def _ttl_refusal(ttl_seconds: Optional[int]) -> Optional[dict]:
        """1-7200, validated before the gate and REFUSED, never clamped (miServer FR-29.1.2)."""
        if ttl_seconds is not None and not 1 <= ttl_seconds <= MAX_LEASE_TTL_S:
            return {"error": "invalid_ttl",
                    "message": f"ttl_seconds must be 1-{MAX_LEASE_TTL_S} (got {ttl_seconds}); "
                               "it is refused, never clamped. Omit it for the maximum."}
        return None

    @mcp.tool()
    async def miserver_acquire_lease(
        model_id: Annotated[int, Field(description="Model row id of the RESIDENT model (miserver_list_models: status 'loaded')")],
        reason: Annotated[str, Field(description="Why you are pinning it, shown to anyone refused because of your lease (1-512 chars)", min_length=1, max_length=512)],
        ttl_seconds: Annotated[Optional[int], Field(description=f"Seconds until the lease expires, 1-{MAX_LEASE_TTL_S}. Omit for {MAX_LEASE_TTL_S}")] = None,
    ) -> Any:
        """Pin the model miServer is serving so nobody else's load or request swaps it out.

        Returns the lease, including `lease_id` — **the only time it is ever returned**. Keep
        it: pass it as `lease_id` to miserver_load_model and every scoring, generation and batch
        tool, and to miserver_renew_lease / miserver_release_lease. It cannot be retrieved later.

        One lease per server, and only on the RESIDENT model (this loads nothing). A lease ends
        at its expiry, on release, when the model stops being resident, and at EVERY miServer
        restart. Holder is this MCP path's identity. No operator approval is needed: a lease
        evicts nothing. Refusals: `MODEL_NOT_RESIDENT`, `MODEL_LEASED` (someone else holds it),
        `INVALID_LEASE_REQUEST`.
        """
        refused = _ttl_refusal(ttl_seconds)
        if refused:
            return refused
        ok, reason_unavailable = await gate.check(MISERVER.name)
        if not ok:
            return {"unavailable": MISERVER.name, "reason": reason_unavailable}
        body: dict[str, Any] = {"holder": ctx.identity, "reason": reason}
        if ttl_seconds is not None:
            body["ttl_seconds"] = ttl_seconds
        lease = await miserver.post(f"/api/models/{model_id}/lease", json_body=body)
        return {
            "lease": lease,
            "hint": "lease_id is shown ONCE. Pass it as lease_id to keep using this model; "
                    "renew before expires_at, release when done.",
        }

    @mcp.tool()
    async def miserver_renew_lease(
        model_id: Annotated[int, Field(description="Model row id the lease is on")],
        lease_id: Annotated[str, Field(description="The lease_id miserver_acquire_lease returned")],
        ttl_seconds: Annotated[Optional[int], Field(description=f"New TTL counted from NOW, 1-{MAX_LEASE_TTL_S}. Omit for {MAX_LEASE_TTL_S}")] = None,
    ) -> Any:
        """Extend your lease: the new expiry is now + ttl_seconds.

        After a miServer restart every lease is gone and this answers `LEASE_NOT_FOUND` — acquire a
        new one. An expired lease answers `LEASE_EXPIRED`.
        """
        refused = _ttl_refusal(ttl_seconds)
        if refused:
            return refused
        ok, reason_unavailable = await gate.check(MISERVER.name)
        if not ok:
            return {"unavailable": MISERVER.name, "reason": reason_unavailable}
        body = {"ttl_seconds": ttl_seconds} if ttl_seconds is not None else None
        return await miserver.post(
            f"/api/models/{model_id}/lease/renew", json_body=body,
            headers={"X-miLLM-Lease": lease_id},
        )

    @mcp.tool()
    @gated(gate, MISERVER.name)
    async def miserver_release_lease(
        model_id: Annotated[int, Field(description="Model row id the lease is on")],
        lease_id: Annotated[str, Field(description="The lease_id miserver_acquire_lease returned")],
    ) -> Any:
        """End your lease now, so others may load or swap the model again."""
        return await miserver.delete(
            f"/api/models/{model_id}/lease", headers={"X-miLLM-Lease": lease_id}
        )

    @mcp.tool()
    @gated(gate, MISERVER.name)
    async def miserver_lease_status(
        model_id: Annotated[Optional[int], Field(description="Model row id. Omit to ask whether ANY lease is held (a drain check): miServer leases only the resident model")] = None,
    ) -> Any:
        """Who holds a model's lease, why and until when — and the last lease that ended.

        Without `model_id`: whether any lease is held at all, for a drain or a pre-flight. miServer keeps
        one lease per model and only the RESIDENT model can be leased (its X-08), so the resident model's
        lease, read from `GET /api/health/detailed`, is every lease it holds: `{"any_lease_held",
        "lease", "model_loaded", "model_name"}`.

        Never returns a lease id: a lease you did not acquire cannot be used, only waited for.
        """
        if model_id is None:
            health = await miserver.get("/api/health/detailed")
            health = health if isinstance(health, dict) else {}
            lease = health.get("lease")
            return {"any_lease_held": lease is not None, "lease": lease,
                    "model_loaded": health.get("model_loaded"), "model_name": health.get("model_name"),
                    "scope": "miServer leases only the resident model (one lease per model), so this is "
                             "every lease it holds"}
        return await miserver.get(f"/api/models/{model_id}/lease")

    @mcp.tool()
    @gated(gate, MISERVER.name)
    async def miserver_unload_model(
        model_id: Annotated[int, Field(description="Model row id to evict from GPU memory")],
    ) -> Any:
        """Evict a model from GPU memory, freeing the card. Waits for the operator in miServer's
        Approvals page (ENH-008).

        Returns miServer's pending answer plus `"decide_in": "miserver"`; nothing is unloaded until the
        operator approves, and then miServer runs it once, by itself. Poll
        `miserver_get_approval_status(approval_id)`. A miServer without its own approval queue is
        refused (`millm_has_no_approval_queue`) and nothing is sent.

        Once run, it waits for pending inference to finish first. After this, miServer serves
        nothing until something is loaded — an OpenAI request naming a model
        auto-loads it, so this is about freeing VRAM, not about blocking use.
        """
        refused = await queue.refusal(RELAYED_ACTIONS["miserver_unload_model"])
        if refused is not None:
            return refused
        response = await miserver.post(f"/api/models/{model_id}/unload", full=True)
        return relayed(response) or response.body

    @mcp.tool()
    @gated(gate, MISERVER.name)
    async def miserver_delete_model(
        model_id: Annotated[int, Field(description="Model row id to delete permanently")],
        acknowledge_permanent: Annotated[bool, Field(description="Delete the row AND its files from disk. Irreversible; re-downloading a large quantization takes tens of GB and a long wait")] = False,
    ) -> Any:
        """Delete a model permanently — the database row and the files on disk. Waits for the
        operator in miServer's Approvals page (ENH-008): an acknowledged call returns miServer's
        pending answer plus `"decide_in": "miserver"`, and miServer deletes once the operator approves.
        Poll `miserver_get_approval_status(approval_id)`. A miServer without its own approval queue is
        refused (`millm_has_no_approval_queue`) and nothing is sent.

        Refuses unless `acknowledge_permanent=True`, and the refusal names the
        model, its quantization and its size so a stale id is visible before it
        costs anything. There is no undo: re-acquiring a 16 GB quantization is
        a fresh download.

        miServer already refuses to delete a LOADED model outright, with no
        override — unload it first. What was unprotected was everything else: a
        READY 16 GB download had nothing between it and a mistyped id, which is
        precisely the case an agent asked to "clean up the old model" hits.
        This mirrors `miserver_delete_circuit`'s gate.

        The size lookup is best-effort and deliberately FAILS OPEN: if the row
        cannot be read the delete still proceeds once acknowledged, so the tool
        stays usable during an outage. It guards against the plausible mistake,
        not against a determined caller.
        """
        if not acknowledge_permanent:
            row = None
            try:
                result = await miserver.get(f"/api/models/{model_id}")
                row = result.get("data") if isinstance(result, dict) else result
            except Exception:
                row = None

            described = "this model"
            if isinstance(row, dict):
                label = row.get("gguf_label")
                size = row.get("disk_size_mb")
                described = str(row.get("name") or f"model {model_id}")
                if label:
                    described += f" ({label})"
                if isinstance(size, (int, float)) and size > 0:
                    described += f", {size / 1024:.1f} GB on disk"
            return {
                "refused": "delete_not_acknowledged",
                "model_id": model_id,
                "would_delete": described,
                "status": row.get("status") if isinstance(row, dict) else None,
                "reason": (
                    f"Deleting {described} removes the row AND its files. This "
                    "cannot be undone — getting it back means downloading it "
                    "again. Confirm this is the right model, then pass "
                    "acknowledge_permanent=true."
                ),
            }
        refused = await queue.refusal(RELAYED_ACTIONS["miserver_delete_model"])
        if refused is not None:
            return refused
        response = await miserver.delete(f"/api/models/{model_id}", full=True)
        return relayed(response) or response.body


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _with_progress_note(row: dict) -> dict:
    """Say whether a download has no progress or merely UNREPORTED progress.

    `download_progress` is written in-process. A download running in another
    worker — or one that has just restarted — reads null, and a client that
    renders null as 0% shows a stalled bar over a download that is proceeding
    normally. The two cases need different reactions from an agent (wait, vs
    investigate), so they must not render identically.
    """
    if row.get("status") != "downloading":
        return row
    row = dict(row)
    if row.get("download_progress") is None:
        row["progress_note"] = (
            "downloading; progress is UNKNOWN, not zero — the percentage is "
            "tracked in-process and is unreadable from another worker. Poll "
            "miserver_get_model and watch `status` rather than the percentage."
        )
    return row
