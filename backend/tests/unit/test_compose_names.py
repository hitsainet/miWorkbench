"""Compose names: containers and this app's images carry the new brand, volume names stay (042 FR-042.31;
TI-24; DR-41; FTASKS 15.1, 15.6; control K-15.13).

Read from the PARSED YAML of every tracked `docker-compose*.yml`, never a line search: a new service or a
moved line cannot slip past. Three rules:

* every `container_name` starts with `miworkbench-`;
* every image this repository publishes (the old-brand stem the export's DR-27 record freezes on Docker
  Hub, or its new twin) is referenced under the new name `hitsai/miworkbench-*`. Third-party images
  (postgres, redis, nginx, the Neuronpedia webapp) are not this app's and are not judged;
* every named volume keeps the exact `name:` it had before the rebrand. A renamed volume is an EMPTY
  volume: compose would start Postgres on a fresh data directory and the old data would sit unused
  under the old name (DR-41). The frozen names are pinned here by value and must match the export's
  DR-41 pattern, so the pin cannot drift from the register.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

from tests.support import legacy_export as le

REPO = le.REPO_ROOT
NEW_PREFIX = "miworkbench-"

#: file -> {volume key -> frozen `name:`}. Values as they stood at the P3 close (`976865f5d`).
FROZEN_VOLUME_NAMES: dict[str, dict[str, str]] = {
    "docker-compose.yml": {
        "postgres_data": "mistudio_postgres_data",
        "redis_data": "mistudio_redis_data",
        "mistudio_data": "mistudio_data",
        "neuronpedia_postgres_data": "mistudio_neuronpedia_postgres_data",
    },
    "docker-compose.hub.yml": {
        "postgres_data": "mistudio_postgres_data",
        "redis_data": "mistudio_redis_data",
    },
    # The dev file named no volume until 2026-11-26, so compose derived `<project>_<key>` from the FOLDER: the
    # checkout's folder rename at P5 silently pointed it at new, empty volumes while the dev database stayed in the
    # volume named below (043 acceptance F-9). Pinned to the names it had before P5, which are also the main
    # compose file's.
    "docker-compose.dev.yml": {
        "postgres_data": "mistudio_postgres_data",
        "neuronpedia_postgres_data": "mistudio_neuronpedia_postgres_data",
    },
    # The GCP example: no explicit volume name either (its volume KEYS carry the old stem and stay).
    "docker/docker-compose.gcp.yml": {},
}


def compose_files(root: Path = REPO) -> dict[str, dict]:
    out = subprocess.run(["git", "ls-files", "-z", "--", "docker-compose*.yml", "docker/docker-compose*.yml"], cwd=root,
                         capture_output=True, check=True).stdout.decode()
    names = sorted(n for n in out.split("\0") if n)
    return {n: yaml.safe_load((root / n).read_text(encoding="utf-8")) for n in names}


def container_names(doc: dict) -> dict[str, str]:
    return {svc: spec["container_name"] for svc, spec in (doc.get("services") or {}).items()
            if "container_name" in spec}


OLD_STEM = re.compile(r"mi(?:studio|llm)", re.IGNORECASE)


def not_renamed(docs: dict[str, dict]) -> list[str]:
    """The stack's own compose files (repository root): every container starts `miworkbench-`. An example
    under a sub-folder may also run a third-party container under its own name (the GCP example's
    Neuronpedia pair), so there the rule is: no container keeps an old-brand name."""
    out = []
    for name, doc in docs.items():
        for svc, cn in container_names(doc).items():
            bad = OLD_STEM.search(cn) if "/" in name else not cn.startswith(NEW_PREFIX)
            if bad:
                out.append(f"{name} service {svc}: container_name {cn!r}")
    return out


def test_container_names_start_miworkbench():
    docs = compose_files()
    # Fail closed: every compose file was read (the three at the root and the GCP example) and each
    # names containers.
    assert set(docs) == set(FROZEN_VOLUME_NAMES), sorted(docs)
    per_file = {name: container_names(doc) for name, doc in docs.items()}
    assert all(per_file.values()), per_file
    assert len(per_file["docker-compose.yml"]) >= 14, per_file["docker-compose.yml"]
    # Every service that runs a container says its name (compose would otherwise derive one from the
    # project directory, which is the old brand until P5).
    for name, doc in docs.items():
        unnamed = [s for s, spec in doc["services"].items() if "container_name" not in spec]
        assert unnamed == [], (name, unnamed)
    assert not_renamed(docs) == []
    # The rule bites on a fixture: one old name is found, the right one.
    planted = {"x.yml": {"services": {"a": {"container_name": "miworkbench-a"},
                                      "b": {"container_name": "mi" + "studio-b"}}}}
    assert not_renamed(planted) == ["x.yml service b: container_name 'mistudio-b'"]
    nested = {"docker/x.yml": {"services": {"a": {"container_name": "third-party"},
                                            "b": {"container_name": "mi" + "studio-b"}}}}
    assert not_renamed(nested) == ["docker/x.yml service b: container_name 'mistudio-b'"]
    assert not_renamed({"x.yml": {"services": {"a": {"container_name": "third-party"}}}})


def test_own_images_named_miworkbench():
    """TI-24: `image:` lines for this app's images name `hitsai/miworkbench-*`. The old names (pushed until P7,
    frozen on Docker Hub) are assembled from the three images and must match the export's DR-27 pattern; they are
    never typed here. (Not the DR-27 `literals`: the census lists only the spellings still present in this tree,
    which shrank as P7 removed the old pushes and comments.)"""
    export = le.load()
    pattern = re.compile(export.entry("DR-27")["pattern"])
    old_images = {"hitsai/mi" + "studio-" + x for x in ("backend", "frontend", "rule-runner")}
    assert all(pattern.search(o) for o in old_images), old_images
    new_images = {o.replace("/mi" + "studio-", "/" + NEW_PREFIX) for o in old_images}
    seen: list[str] = []
    for name, doc in compose_files().items():
        for svc, spec in doc["services"].items():
            image = str(spec.get("image", "")).split(":", 1)[0]
            assert image not in old_images, f"{name} service {svc}: image {image!r} is the old name"
            if image in new_images:
                seen.append(image)
    # Fail closed: the backend, frontend and rule-runner images are each referenced under the new name.
    assert set(seen) == new_images, (sorted(set(seen)), sorted(new_images))


def test_volume_names_unchanged():
    export = le.load()
    frozen = re.compile(export.entry("DR-41")["pattern"], re.MULTILINE)
    docs = compose_files()
    assert set(docs) == set(FROZEN_VOLUME_NAMES)
    for name, doc in docs.items():
        named = {key: spec["name"] for key, spec in (doc.get("volumes") or {}).items()
                 if isinstance(spec, dict) and "name" in spec}
        assert named == FROZEN_VOLUME_NAMES[name], name
        for value in named.values():
            assert frozen.search(f"    name: {value}\n"), f"{value} is not covered by the export's DR-41"
    assert sum(len(v) for v in FROZEN_VOLUME_NAMES.values()) == 8
    # The GCP example's volume KEYS are what compose names its volumes from (`<project>_<key>`).
    gcp = docs["docker/docker-compose.gcp.yml"]["volumes"]
    assert {"mistudio_postgres_data", "mistudio_redis_data", "mistudio_data"} <= set(gcp), sorted(gcp)


@pytest.mark.parametrize("name", sorted(FROZEN_VOLUME_NAMES))
def test_services_mount_the_same_volume_keys(name):
    """A frozen `name:` is only half of DR-41: a service must still mount the KEY that carries it."""
    doc = compose_files()[name]
    declared = set(doc.get("volumes") or {})
    used = {str(v).split(":", 1)[0] for spec in doc["services"].values() for v in spec.get("volumes", [])
            if isinstance(v, str) and not str(v).startswith((".", "/", "$", "~"))}
    assert used <= declared, (name, sorted(used - declared))
    # The database volume is the one a rename would empty: Postgres still mounts its key.
    if "postgres_data" in declared:
        assert any(str(v).startswith("postgres_data:") for v in doc["services"]["postgres"]["volumes"]), name


# ── 043 FR-043.17 (P9-D7): networks carry the new name; GCP operator variables are read under both ──────────
def test_network_names_start_miworkbench():
    """Every compose network key and explicit `name:` starts with `miworkbench-` (a network holds no data, so it is
    renamed outright; volumes above are not)."""
    bad = []
    for name, doc in compose_files().items():
        for key, spec in (doc.get("networks") or {}).items():
            for value in (key, (spec or {}).get("name", key)):
                if not str(value).startswith(NEW_PREFIX):
                    bad.append(f"{name}: network {value}")
        for svc, spec in (doc.get("services") or {}).items():
            nets = spec.get("networks") or []
            for net in (nets if isinstance(nets, list) else list(nets)):
                if not str(net).startswith(NEW_PREFIX):
                    bad.append(f"{name}: service {svc} joins {net}")
    assert compose_files(), "no compose file discovered"
    assert not bad, bad


_VAR = re.compile(r"\$\{([A-Z0-9_]+)(:-|:\?)")


def interpolate(text: str, env: dict[str, str]) -> str:
    """Compose's `${VAR:-default}` and `${VAR:?message}`, nested (the subset the GCP file uses)."""
    def one(s: str, i: int) -> tuple[str, int]:
        m = _VAR.match(s, i)
        assert m, s[i:i + 40]
        var, op, j, depth, start = m.group(1), m.group(2), m.end(), 1, m.end()
        while depth:
            if s.startswith("${", j):
                depth, j = depth + 1, j + 2
                continue
            depth -= s[j] == "}"
            j += 1
        inner = s[start:j - 1]
        if env.get(var):
            return env[var], j
        if op == ":-":
            return interpolate(inner, env), j
        raise LookupError(f"{var}: {inner}")

    out, i = [], 0
    while i < len(text):
        if text.startswith("${", i):
            value, i = one(text, i)
            out.append(value)
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


@pytest.mark.parametrize("spellings", [("MIWORKBENCH",), ("MISTUDIO",), ("MIWORKBENCH", "MISTUDIO")])
def test_gcp_compose_reads_both_spellings(spellings):
    """The GCP example reads the new operator variables, and the old ones for one release (FTASKS 7.3; 8.1 drops
    them); with both set the new one wins. Evaluated on the parsed file's values by compose's own interpolation
    rules. Values are generated per run (no credential literal in a test file, 042 FR-042.40)."""
    import secrets

    doc = compose_files()["docker/docker-compose.gcp.yml"]
    env, want = {}, {}
    for prefix in spellings:
        for suffix in ("_POSTGRES_PASSWORD", "_SECRET_KEY"):
            env[prefix + suffix] = secrets.token_hex(6)
    for suffix in ("_POSTGRES_PASSWORD", "_SECRET_KEY"):
        want[suffix] = env[spellings[0] + suffix]
    values = {}
    for svc, spec in doc["services"].items():
        env_block = spec.get("environment") or {}
        for key in ("POSTGRES_PASSWORD", "DATABASE_URL", "SECRET_KEY"):
            raw = env_block.get(key) if isinstance(env_block, dict) else None
            if raw and ("MISTUDIO" in raw or "MIWORKBENCH" in raw):
                values[(svc, key)] = interpolate(raw, env)
    assert {k[1] for k in values} == {"POSTGRES_PASSWORD", "DATABASE_URL", "SECRET_KEY"}, values
    for (svc, key), value in values.items():
        assert want["_SECRET_KEY" if key == "SECRET_KEY" else "_POSTGRES_PASSWORD"] in value, (svc, key)


def test_gcp_compose_refuses_with_neither_spelling():
    doc = compose_files()["docker/docker-compose.gcp.yml"]
    raw = doc["services"]["postgres"]["environment"]["POSTGRES_PASSWORD"]
    with pytest.raises(LookupError):
        interpolate(raw, {})


def test_mcp_openapi_ttl_reads_both_spellings(monkeypatch):
    from src.mcp_server.config import MCPSettings

    for name, value in (("MCP_MISERVER_OPENAPI_TTL_SECONDS", "9"), ("MCP_MILLM_OPENAPI_TTL_SECONDS", "7")):
        monkeypatch.delenv("MCP_MISERVER_OPENAPI_TTL_SECONDS", raising=False)
        monkeypatch.delenv("MCP_MILLM_OPENAPI_TTL_SECONDS", raising=False)
        monkeypatch.setenv(name, value)
        assert MCPSettings(allow_anonymous=True).miserver_openapi_ttl_seconds == int(value), name
