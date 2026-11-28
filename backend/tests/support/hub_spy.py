"""A fake Hugging Face Hub beneath `huggingface_hub`'s own HTTP client (045 FR-18; FTASKS 21.7, 21.9).

`huggingface_hub` 2.x sends every request through one shared client built by a factory
(`huggingface_hub.set_client_factory`); `transformers` and `datasets` reach the Hub through it. Installing an
`httpx.MockTransport` there puts this fake BELOW all three libraries, so a test sees the exact headers each request
carried — the `Authorization` header in particular — whichever library built it.

The fake answers the endpoints `snapshot_download`, `hf_hub_download`, `HfApi.repo_info` / `model_info` /
`dataset_info` and `list_repo_tree` / `list_repo_files` use, for the repositories it is given; anything else is a 404
with the Hub's `EntryNotFound` code (an optional file transformers probes for).

Used with `blocked_sockets` (no real connection to anything but the local test database).
"""
from __future__ import annotations

import hashlib
import json
import re
import socket
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import pytest

# huggingface_hub 2.x speaks httpx2 (a separate package), not httpx: a plain `httpx.Client` here would raise
# `httpx.HTTPStatusError`, which the Hub client does not map, so a 404 for an optional file would abort the load.
import httpx2 as httpx

SHA = "0123456789abcdef0123456789abcdef01234567"

_REVISION = re.compile(r"^/api/(models|datasets)/(.+?)/revision/([^/]+)$")
_TREE = re.compile(r"^/api/(models|datasets)/(.+?)/tree/([^/]+)(?:/(.*))?$")
_INFO = re.compile(r"^/api/(models|datasets)/([^/]+/[^/]+|[^/]+)$")
_RESOLVE = re.compile(r"^/(datasets/)?(.+?)/resolve/([^/]+)/(.+)$")


@dataclass
class HubRequest:
    method: str
    path: str
    authorization: str | None


@dataclass
class FakeHub:
    """`repos` maps `(kind, repo_id)` (kind `"model"` or `"dataset"`) to `{filename: bytes}`."""

    repos: dict[tuple[str, str], dict[str, bytes]]
    requests: list[HubRequest] = field(default_factory=list)

    # ------------------------------------------------------------------ what a test reads
    @property
    def authorized(self) -> list[HubRequest]:
        return [r for r in self.requests if r.authorization]

    def fetched(self) -> set[str]:
        """The file names whose CONTENT was requested (a GET on a resolve URL)."""
        out = set()
        for r in self.requests:
            m = _RESOLVE.match(r.path)
            if m and r.method == "GET":
                out.add(m.group(4))
        return out

    # ------------------------------------------------------------------ the transport
    def _files(self, kind: str, repo: str) -> dict[str, bytes] | None:
        return self.repos.get(("dataset" if kind.startswith("dataset") else "model", repo))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.requests.append(HubRequest(request.method, path, request.headers.get("authorization")))
        if m := _REVISION.match(path):
            return self._info(m.group(1), m.group(2))
        if m := _TREE.match(path):
            files = self._files(m.group(1), m.group(2))
            if files is None:
                return self._missing("RepoNotFound")
            return httpx.Response(200, json=self._tree(files, m.group(4) or "",
                                                       request.url.params.get("recursive") in ("1", "true", "True")))
        if m := _INFO.match(path):
            return self._info(m.group(1), m.group(2))
        if m := _RESOLVE.match(path):
            files = self._files("datasets" if m.group(1) else "models", m.group(2))
            if files is None:
                return self._missing("RepoNotFound")
            body = files.get(m.group(4))
            if body is None:
                return self._missing("EntryNotFound")
            headers = {"X-Repo-Commit": SHA, "ETag": f'"{hashlib.sha1(body).hexdigest()}"',
                       "Content-Length": str(len(body)), "Content-Type": "application/octet-stream"}
            return httpx.Response(200, headers=headers, content=b"" if request.method == "HEAD" else body)
        return self._missing("EntryNotFound")

    @staticmethod
    def _tree(files: dict[str, bytes], under: str, recursive: bool) -> list[dict]:
        """The Hub's tree listing of `under`, as the real one answers it: its files and its sub-directories (a
        `directory` entry each), one level deep unless `recursive`. R1 review fix (OR1-045a0-7), 2026-11-16: the
        listing used to be every file of the repository, flat, so `HfFileSystem` never saw `data/` and
        `load_dataset` found no data file."""
        prefix = under.strip("/") + "/" if under.strip("/") else ""
        entries: dict[str, dict] = {}
        for name, body in sorted(files.items()):
            if not name.startswith(prefix):
                continue
            parts = name[len(prefix):].split("/")
            for depth in range(1, len(parts)):
                folder = prefix + "/".join(parts[:depth])
                if recursive or depth == 1:
                    entries.setdefault(folder, {"type": "directory", "path": folder,
                                                "oid": hashlib.sha1(folder.encode()).hexdigest()})
            if recursive or len(parts) == 1:
                entries[name] = {"type": "file", "path": name, "size": len(body), "oid": hashlib.sha1(body).hexdigest()}
        return list(entries.values())

    def _info(self, kind: str, repo: str) -> httpx.Response:
        files = self._files(kind, repo)
        if files is None:
            return self._missing("RepoNotFound")
        return httpx.Response(200, json={
            "id": repo, "sha": SHA, "private": False, "gated": False,
            "siblings": [{"rfilename": f} for f in sorted(files)],
        })

    @staticmethod
    def _missing(code: str) -> httpx.Response:
        return httpx.Response(404, headers={"X-Error-Code": code}, json={"error": code})


def tiny_tokenizer_files() -> dict[str, bytes]:
    """A real, loadable fast tokenizer (word level, eight tokens) as the Hub would serve it."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {w: i for i, w in enumerate(["[UNK]", "[PAD]", "<eos>", "hello", "world", "the", "cat", "sat"])}
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    return {
        "tokenizer.json": tok.to_str().encode(),
        "tokenizer_config.json": json.dumps({
            "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]", "pad_token": "[PAD]",
            "eos_token": "<eos>", "model_max_length": 64,
        }).encode(),
        "special_tokens_map.json": json.dumps({"unk_token": "[UNK]", "pad_token": "[PAD]", "eos_token": "<eos>"}).encode(),
        "config.json": json.dumps({"model_type": "gpt2", "vocab_size": 8, "n_embd": 8, "n_layer": 1, "n_head": 1,
                                   "n_positions": 64, "architectures": ["GPT2LMHeadModel"]}).encode(),
        # Weights a tokenizer load must never fetch (A36's allow_patterns).
        "model.safetensors": b"\0" * 64,
        "pytorch_model.bin": b"\0" * 64,
    }


def tiny_dataset_files() -> dict[str, bytes]:
    return {"data/train.jsonl": b'{"text": "hello world"}\n{"text": "the cat sat"}\n', "README.md": b"# fixture\n"}


@pytest.fixture
def blocked_sockets(monkeypatch):
    """No connection leaves this machine: only the local database and Redis hosts resolve (FTASKS 21.7)."""
    real = socket.getaddrinfo

    def guarded(host, *args, **kwargs):
        if host not in ("localhost", "127.0.0.1", "::1", None):
            raise OSError(f"blocked by the test: no network to {host!r}")
        return real(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", guarded)


@contextmanager
def installed_fake_hub(monkeypatch, tmp_path, repos) -> Iterator[FakeHub]:
    """The fake Hub installed beneath huggingface_hub for `repos`, with `HF_TOKEN` SET (the deployment's token), a
    fresh cache, and a stored token in the settings path (`resolve_hf_token`'s database tier is never reached for
    an agent)."""
    import huggingface_hub
    from huggingface_hub import constants
    from huggingface_hub.utils import _http

    hub = FakeHub(repos)
    monkeypatch.setenv("HF_TOKEN", DEPLOYMENT_TOKEN)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False, raising=False)
    monkeypatch.setattr(constants, "HF_TOKEN_PATH", str(tmp_path / "token"), raising=False)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"), raising=False)
    monkeypatch.setattr(constants, "HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"), raising=False)
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", True, raising=False)
    monkeypatch.setattr("src.core.config.settings.hf_token", DEPLOYMENT_TOKEN, raising=False)
    try:
        import datasets.config as dconfig

        monkeypatch.setattr(dconfig, "HF_DATASETS_CACHE", tmp_path / "datasets", raising=False)
        monkeypatch.setattr(dconfig, "HF_HUB_OFFLINE", False, raising=False)
    except ImportError:  # pragma: no cover
        pass
    previous = _http._GLOBAL_CLIENT_FACTORY
    huggingface_hub.set_client_factory(lambda: httpx.Client(transport=httpx.MockTransport(hub),
                                                            follow_redirects=True))
    # R3 review (044-A I1), 2026-11-17: a tokenizer loaded from THIS fake Hub must not outlive it. `tokenize-preview`
    # keeps loads in a process-wide `lru_cache` (`load_tokenizer_cached`); the hub sweep left the fixture's tokenizer
    # there, so a later test in the same worker (A0's write sweep, which lists the route as refusing early with the
    # network blocked) was answered 200 from the cache. Cleared on the way in and on the way out.
    from src.api.v1.endpoints.datasets import load_tokenizer_cached

    load_tokenizer_cached.cache_clear()
    try:
        yield hub
    finally:
        huggingface_hub.set_client_factory(previous)
        load_tokenizer_cached.cache_clear()


#: The deployment's token in every fake-Hub test: set in `HF_TOKEN` and the settings, so a request that carries it
#: is a request that used the pod's credential.
DEPLOYMENT_TOKEN = "hf_DEPLOYMENT_TOKEN_must_not_leave"


@pytest.fixture
def fake_hub(monkeypatch, tmp_path, blocked_sockets) -> Iterator[FakeHub]:
    """The fake Hub with a tokenizer repository and a dataset repository (see `installed_fake_hub`)."""
    with installed_fake_hub(monkeypatch, tmp_path, {("model", "fixture-org/tok"): tiny_tokenizer_files(),
                                                    ("dataset", "fixture-org/ds"): tiny_dataset_files()}) as hub:
        yield hub
