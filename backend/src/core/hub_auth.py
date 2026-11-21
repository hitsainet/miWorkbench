"""Hugging Face Hub auth policy per caller (045 FR-18; CHAIN_PLAN A17, A28, A36, A42; FTDD §18).

An agent other than the MCP pod reads the Hub ANONYMOUSLY: it never borrows the operator's stored token or the
deployment's `HF_TOKEN`. The operator (no agent header) and the MCP pod (`mcp`, A15) keep today's resolution.

* `hub_auth_for(request)` → `"anonymous"` for an agent other than `MCP_POD_AGENTS`, else `"stored"`.
* `hub_token(policy, explicit)` → **`False`** for `"anonymous"`, without resolving anything (A28: in huggingface_hub
  `token=None` means "use `HF_TOKEN` or the cached token"; only `False` is anonymous). For `"stored"` it is today's
  `resolve_hf_token(explicit)`. An unknown policy fails closed (anonymous).

* `hub_load(kind, repo_id, policy, …)` → the A36 two-step load. Under `"anonymous"`: (1) `hub_snapshot`, a
  `snapshot_download(token=False, allow_patterns=PATTERNS[kind])` into the normal cache, then (2) the load from that
  local directory with `local_files_only=True`. `AutoTokenizer.from_pretrained(name, token=False)` is never used for an
  agent: transformers 5.20.2 lists the repository's files through a token-less `HfApi` (round 2 measured it sending
  `HF_TOKEN`). Under `"stored"` the load is today's, with `token=hub_token("stored", explicit)`.
* `hub_model_source(repo_id, policy, …)` → where a model load reads from: the local snapshot directory under
  `"anonymous"` (weights, config and tokenizer files only), the repository id under `"stored"`.

The policy travels to a worker as the task argument `hub_auth` (default `"anonymous"`: an enqueue that omits it gets no
token, A28/R1A-5), never as a token.
"""
from __future__ import annotations

from typing import Any, Literal, NamedTuple, Optional, Union

Policy = Literal["anonymous", "stored"]
ANONYMOUS: Policy = "anonymous"
STORED: Policy = "stored"


def hub_auth_for(request: Any) -> Policy:
    """The policy for one request: anonymous for an agent other than the MCP pod (fail closed on any other value)."""
    from src.core.approval_gate import AGENT_HEADER, MCP_POD_AGENTS

    agent = request.headers.get(AGENT_HEADER)
    if agent is None or agent in MCP_POD_AGENTS:
        return STORED
    return ANONYMOUS


def hub_token(policy: str, explicit: Optional[str] = None) -> Union[str, Literal[False], None]:
    """The `token=` to pass a Hub call under `policy`: `False` (anonymous) unless the policy is exactly `"stored"`."""
    if policy != STORED:
        return False
    from src.services.huggingface_sae_service import resolve_hf_token

    return resolve_hf_token(explicit)


class ModelFilesNotOnDisk(Exception):
    """A42: an agent's read needs model files that are not on disk; it is never fetched from the Hub for an agent."""

    code = "model_files_not_on_disk"


# A36: the files each kind of load needs. A tokenizer never pulls weights (a GET preview must not pull a model).
TOKENIZER_PATTERNS: tuple[str, ...] = (
    "tokenizer*", "special_tokens_map.json", "*.json", "*.txt", "*.model", "*.jinja", "*.tiktoken",
)
_WEIGHTS = {"safetensors": ("*.safetensors",), "bin": ("*.bin",)}
DATASET_PATTERNS: tuple[str, ...] = (
    "*.parquet", "*.arrow", "*.json", "*.jsonl", "*.json.gz", "*.jsonl.gz", "*.csv", "*.tsv", "*.txt", "README.md",
)
# Checkpoint copies some repositories carry beside the transformers files (e.g. `original/consolidated.pth`).
IGNORE_PATTERNS: tuple[str, ...] = ("original/*", "*/original/*", "*.pth", "*.pt", "*.gguf", "*.onnx", "*.h5",
                                    "*.msgpack", "*.ckpt")
PATTERNS: dict[str, tuple[str, ...]] = {
    "tokenizer": TOKENIZER_PATTERNS,
    "model": TOKENIZER_PATTERNS + ("*.safetensors", "*.bin"),
    "dataset": DATASET_PATTERNS,
}


def _model_patterns(repo_id: str, revision: Optional[str]) -> tuple[str, ...]:
    """Config, tokenizer and ONE weight format: safetensors when the repository has them, else `.bin`.

    Listed anonymously (`token=False`); a repository that cannot be listed falls back to both formats.
    """
    try:
        from huggingface_hub import HfApi

        files = HfApi(token=False).list_repo_files(repo_id, revision=revision, token=False)
    except Exception:  # noqa: BLE001 - the snapshot itself reports a refused or missing repository
        return PATTERNS["model"]
    fmt = "safetensors" if any(f.endswith(".safetensors") for f in files) else "bin"
    return TOKENIZER_PATTERNS + _WEIGHTS[fmt]


def hub_snapshot(kind: str, repo_id: str, *, revision: Optional[str] = None,
                 cache_dir: Optional[str] = None) -> str:
    """Step 1 of A36: an ANONYMOUS `snapshot_download` of the files a `kind` load needs; returns the local directory."""
    from huggingface_hub import snapshot_download

    if kind not in PATTERNS:
        raise ValueError(f"unknown hub load kind {kind!r}")
    patterns = _model_patterns(repo_id, revision) if kind == "model" else PATTERNS[kind]
    return snapshot_download(
        repo_id,
        revision=revision,
        repo_type="dataset" if kind == "dataset" else "model",
        cache_dir=str(cache_dir) if cache_dir else None,
        token=False,
        allow_patterns=list(patterns),
        ignore_patterns=list(IGNORE_PATTERNS),
    )


def hub_load(kind: Literal["tokenizer", "dataset"], repo_id: str, policy: str, *, revision: Optional[str] = None,
             cache_dir: Optional[str] = None, explicit: Optional[str] = None, split: Optional[str] = None,
             **kwargs: Any) -> Any:
    """A tokenizer or dataset load under `policy` (A36). The only place an agent's tokenizer is loaded from the Hub.

    Under anything but `"stored"` (fails closed): snapshot anonymously, then load the local directory with
    `local_files_only=True`; `trust_remote_code` is dropped (an agent never runs repository code).
    """
    if kind == "tokenizer":
        from transformers import AutoTokenizer

        if policy != STORED:
            kwargs.pop("trust_remote_code", None)
            kwargs.pop("local_files_only", None)
            local = hub_snapshot("tokenizer", repo_id, revision=revision, cache_dir=cache_dir)
            return AutoTokenizer.from_pretrained(local, local_files_only=True, **kwargs)
        return AutoTokenizer.from_pretrained(repo_id, revision=revision, cache_dir=cache_dir,
                                             token=hub_token(STORED, explicit), **kwargs)
    if kind == "dataset":
        from datasets import load_dataset

        if policy != STORED:
            kwargs.pop("trust_remote_code", None)
            local = hub_snapshot("dataset", repo_id, revision=revision, cache_dir=cache_dir)
            return load_dataset(local, split=split, token=False, **kwargs)
        return load_dataset(repo_id, split=split, revision=revision, cache_dir=cache_dir,
                            token=hub_token(STORED, explicit), **kwargs)
    raise ValueError(f"unknown hub load kind {kind!r}")


class ModelSource(NamedTuple):
    """Where a model load reads from, and how (`hub_model_source`)."""

    source: str
    token: Union[str, Literal[False], None]
    local_files_only: bool


def hub_model_source(repo_id: str, policy: str, *, explicit: Optional[str] = None,
                     cache_dir: Optional[str] = None, revision: Optional[str] = None) -> ModelSource:
    """A model load's source under `policy` (A36): the anonymous local snapshot, or the repository with the stored token."""
    if policy != STORED:
        local = hub_snapshot("model", repo_id, revision=revision, cache_dir=cache_dir)
        return ModelSource(local, False, True)
    return ModelSource(repo_id, hub_token(STORED, explicit), False)


def refuse_agent_hub_token(policy: str, **fields: Any) -> None:
    """FR-18: an agent's own Hub token (`access_token`, `hf_token`, `token`) is `422 agent_token_not_accepted` before
    anything runs. The gated routes refuse it in the gate (V-9); the ungated Hub routes call this."""
    if policy == STORED:
        return
    from fastapi import HTTPException

    for field, value in fields.items():
        if value not in (None, ""):
            raise HTTPException(422, {"code": "agent_token_not_accepted", "field": field})
