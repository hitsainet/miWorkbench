"""ENH-010 044 task 3.5 (R1A-W12): the BACKEND container gets `MIFORGE_API_URL`, in k8s and in compose.

Without it production answers `unavailable: miforge` forever, which reads exactly like correct standalone behaviour —
nothing else would fail. So this test reads the rendered manifests, not the settings class. Live proof is 044 17.3
(the deployed backend reports `miforge: reachable`).
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
MIFORGE = "http://miforge-backend.miforge.svc.cluster.local:8000"


def _k8s_backend_env() -> dict:
    for doc in yaml.safe_load_all((REPO / "k8s" / "base" / "backend.yaml").read_text()):
        if not doc or doc.get("kind") != "Deployment":
            continue
        for container in doc["spec"]["template"]["spec"]["containers"]:
            env = {e["name"]: e.get("value") for e in container.get("env", [])}
            if env.get("SERVICE_TYPE") == "api":
                return env
    raise AssertionError("no API container (SERVICE_TYPE=api) in k8s/base/backend.yaml")


def test_k8s_backend_sets_miforge_url():
    env = _k8s_backend_env()
    assert env.get("MIFORGE_API_URL") == MIFORGE, env.get("MIFORGE_API_URL")
    # The same in-cluster address the MCP pod already uses.
    mcp = [d for d in yaml.safe_load_all((REPO / "k8s" / "base" / "mcp.yaml").read_text()) if d and d.get("kind") == "Deployment"]
    mcp_env = {e["name"]: e.get("value") for c in mcp[0]["spec"]["template"]["spec"]["containers"] for e in c.get("env", [])}
    assert mcp_env.get("MIFORGE_API_URL") == env["MIFORGE_API_URL"]


def test_compose_backend_sets_miforge_url():
    env = yaml.safe_load((REPO / "docker-compose.yml").read_text())["services"]["backend"]["environment"]
    assert env.get("MIFORGE_API_URL") == "${MIFORGE_API_URL:-}"


def test_the_setting_reads_the_variable(monkeypatch):
    from src.core.config import Settings

    monkeypatch.setenv("MIFORGE_API_URL", "http://forge.example:8000")
    assert Settings().miforge_api_url == "http://forge.example:8000"
    monkeypatch.delenv("MIFORGE_API_URL")
    assert Settings().miforge_api_url == ""
