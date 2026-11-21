"""No deploy, dev or tooling default names a host ENH-009 P7 retired (043 FR-043.17; P9-D18; FTASKS 7.9 - 7.11).

P7 removed the old ingress aliases, yet compose still defaulted `OLLAMA_URL` to the old miServer host and the dev
setup still served `dev-mistudio.hitsai.local`: a default that names a retired host fails only when it is used.
The old hosts are read from the export (DR-29's literals and the `host` records), never typed here; the files are
discovered by `git ls-files` per kind of default.
"""
from __future__ import annotations

import re
import subprocess

from tests.support import legacy_export as le

ROOT = le.REPO_ROOT
GLOBS = ("docker-compose*.yml", "docker/*", "nginx/*", "frontend/vite.config.ts", "start-*.sh", "stop-*.sh",
         "restart-*.sh", ".env.example", "backend/.env.example", "scripts/**", "backend/scripts/**",
         "manual/scripts/**", ".claude/commands/*")
#: The dev host names of the reference deployment before P9 (`dev-`/`dkr-` + the old app name).
OLD_DEV = re.compile(r"\b(dev|dkr)-mi(studio|llm)\.", re.IGNORECASE)


def _files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", "--", *[f":(glob){g}" for g in GLOBS]], cwd=ROOT,
                         capture_output=True, check=True).stdout.decode()
    return sorted(p for p in out.split("\0") if p and (not p.endswith(".md") or p.startswith(".claude/")))


def retired_hosts(export: le.Export) -> list[str]:
    hosts = {r["old"] for r in export.hosts() if r["status"] == "retired"}
    try:
        hosts |= {h for h in export.entry("DR-29")["literals"]}
    except KeyError:  # the public export withholds DR-29 (R23)
        pass
    return sorted(hosts, key=len, reverse=True)


def offenders(text: str, hosts: list[str]) -> list[str]:
    found = []
    for n, line in enumerate(text.split("\n"), 1):
        low = line.lower()
        # A path segment (`/home/x/<host>/…`, a folder named after a host) is a path, not a host default.
        hit = [h for h in hosts if re.search(r"(?:(?<=://)|(?<![\w./-]))" + re.escape(h.lower()) + r"(?![\w-])", low)]
        if hit or OLD_DEV.search(line):
            found.append(f"{n}: {line.strip()[:120]}")
    return found


def test_no_default_names_a_retired_host():
    export = le.load()
    hosts = retired_hosts(export)
    assert hosts, "the export lists no retired host"
    files = _files()
    assert len(files) > 20, f"only {len(files)} files discovered: the scan reads nothing"
    bad = {f: o for f in files if (o := offenders((ROOT / f).read_text(encoding="utf-8", errors="replace"), hosts))}
    assert not bad, bad


def test_a_retired_host_default_is_found():
    hosts = ["k8s-" + "mi" + "llm.hitsai.local"]
    assert offenders("OLLAMA_URL: ${OLLAMA_URL:-http://k8s-mi" "llm.hitsai.local}", hosts)
    assert offenders('DOMAIN="dev-mi' 'studio.hitsai.local"', hosts)
    assert not offenders("OLLAMA_URL: ${OLLAMA_URL:-http://k8s-miserver.hitsai.local}", hosts)
