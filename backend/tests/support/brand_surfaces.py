"""The brand guard, backend half: surface discovery and the scanner (042 FR-042.13, FR-042.34; FTDD §6.7;
FTID §3.14; TPD-15, TPD-16, TPD-31, TPD-33; CF-13, CF-15, CF-20 – CF-23).

WHAT IS A SURFACE (PD-3). Discovered, never listed by file: README-family H1–H3 headings
(`README*.md`, `CONTRIBUTING.md`, `SECURITY.md` anywhere tracked, outside the record trees `0xcc/` and
`.claude/`), every manual doc's front-matter `title`/`sidebar_label` and first heading, the Docusaurus config's
`title`/`tagline`/navbar/footer and `sidebars.ts` labels, SVG `<title>`/`<text>` under `frontend/public` and
`frontend/src/assets`, `frontend/index.html`'s `<title>`, the FastAPI title and health `service` from the REAL
app object, and the MCP server name, instructions, every tool's name and description and the howto topics'
text from the REAL built server (`build_server()`, every category on). A report-only class (TPD-16) lists
backend string constants under `src/` naming an old brand; it is printed, never enforced.

THE P9 CLASSES (043 FR-043.2; FTDD §6.1; FTID §3.1). ENH-009 P9 adds five classes that reach what no P4 surface
does: `comments` (comments and docstrings), `docs_text` (documentation and agent-instruction prose), `test_text`
(test code, test strings and test file names), `identifiers` (source code names and string values, and source file
names) and `deploy_text` (deploy, CI and tooling values). Each is report-only until its residue group closes:
`CLASS_CLOSED[cls]` turns True in the group's last commit, and only then does `effective_mode(export, cls=cls)`
enforce it (and, as for every class, only where the full export exists, R23). Their record trees are `0xcc/` and
`.claude/context/` only: `.claude/commands/` and `.claude/*.md` are agent instructions, so `docs_text` scans them.

WHAT IS ALLOWED. Only what the census export says (FTID §3.14), read through `tests/support/legacy_export.py`:
(a) a match of a non-retired `entry` record's `pattern`, or one of its `literals`; (b) the `old` of a
`transition` `env`/`host`/`history_tool` record; (c) the WHOLE `<owner>/<name>` token of a `transition` `repo`
record (so `<owner>/<name>-backup` is not covered; TPD-33 — this module holds no repository name); (d) a
match of an ACTIVE `allow` guard record. An active `deny` guard's match is a finding unless (a)–(d) cover it.
Guard activity keys to the header's `phase`; enforcement keys to its `closed` (CF-21).

The export is never a surface (CF-20): discovery always passes BOTH glob-correct exclusions, because on
git 2.43 `:(exclude,glob)**/docs/legacy-identifiers*.md` does not exclude the root-level file under a `docs/`
positive pathspec (miForge P0 review, finding 13).
"""
from __future__ import annotations

import ast
import asyncio
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from tests.support import legacy_export as le

#: P0 – P4: findings are printed and the test passes. Flipped to "enforce" in this repository's first commit
#: after S001 closes P4 (CF-26). `test_brand_guard_mode.py` ties it to the export's `closed` and to the
#: frontend mirror `frontend/src/test/brandMode.ts`. What a run USES is `effective_mode(export)`: this
#: constant with the full export (the private repository), always "report" with the public one (R23).
BRAND_GUARD_MODE = "enforce"

REPO_ROOT = le.REPO_ROOT

#: The hand-kept old-brand stems (the brand list is not a register fact; FTID §3.14). `miSuite` is NOT here:  # p9: keep, brand-guard plant or rule
#: it is the suite's name until P5, when the export's `G-MISUITE` deny record makes it a finding (CF-15).  # p9: keep, brand-guard plant or rule
STEM = re.compile(r"mi(?:studio|llm)|mechinterp[ -]?studio", re.IGNORECASE)
#: `_` and `.` are word characters for a token, so `millm_probe_id` is reported (and allowed) whole.
TOKEN_CHAR = re.compile(r"[\w.]")

#: CF-20, both forms (see the module docstring).
EXPORT_EXCLUDES = (":(exclude,glob)**/docs/legacy-identifiers*.md", ":(exclude,glob)docs/legacy-identifiers*.md")
#: Record trees (dated history and agent context, stripped from the mirror) are never a brand surface. They
#: are dropped in Python, NOT by pathspec: on git 2.43 an extra `:(exclude,glob)0xcc/**` (or `…/**/*`) beside
#: the export exclusions made unrelated positive pathspecs (`frontend/index.html`, `manual/docs/**/*.md`) match
#: NOTHING (measured 2026-11-08), which would have emptied whole classes.
_EXPORT_NAME = re.compile(r"(^|/)docs/legacy-identifiers[^/]*\.md$")
_RECORD_PREFIXES = ("0xcc/", ".claude/")

#: Surface classes enforced at P4 (PD-3), and the one report-only class (TPD-16).
ENFORCED_CLASSES = ("readme", "manual_docs", "manual_config", "svg", "index_html", "api", "mcp")
REPORT_ONLY_CLASSES = ("backend_strings",)
#: ENH-009 P9 (043 FR-043.2): one class per fixable residue group, each report-only until `CLASS_CLOSED`.
P9_CLASSES = ("comments", "docs_text", "test_text", "test_values", "identifiers", "deploy_text")
FILE_CLASSES = ("readme", "manual_docs", "manual_config", "svg", "index_html", "backend_strings", *P9_CLASSES)
LIVE_CLASSES = ("api", "mcp")


@dataclass(frozen=True)
class Item:
    cls: str
    where: str
    text: str
    #: Where `text` starts in its source line (P9 classes: the rewrite tool maps a finding back to the file).
    col: int = 0
    #: The source line carries a keep mark (`p9` + `: keep, <reason>`; P9 classes): its findings are reported as kept and
    #: never fail. `test_keep_marks_keep_something` holds every mark to a reason and to a kept finding.
    kept: bool = False
    #: Only old names that START inside one of these `(start, end)` spans of `text` count (None: anywhere).
    #: `test_text` and `test_values` read the same test line, split at its string literals (P9-D6).
    regions: tuple[tuple[int, int], ...] | None = None


@dataclass
class Finding:
    cls: str
    where: str
    token: str
    rules: list[str] = field(default_factory=list)
    kept: bool = False

    def line(self) -> str:
        kept = " [kept]" if self.kept else ""
        return f"[{self.cls}] {self.where}: {self.token} ({', '.join(self.rules)}){kept}"


# ── discovery ──────────────────────────────────────────────────────────────────────────────────────────
def ls_files(root: Path, globs: Iterable[str], records: tuple[str, ...] = _RECORD_PREFIXES) -> list[str]:
    """Tracked files matching `globs`, never the export and never a record tree (`records`: path prefixes)."""
    specs = [f":(glob){g}" for g in globs]
    out = subprocess.run(["git", "ls-files", "-z", "--", *specs, *EXPORT_EXCLUDES],
                         cwd=root, capture_output=True, check=True).stdout.decode("utf-8")
    paths = sorted(p for p in out.split("\0") if p)
    # Defence in depth: git pathspec magic has failed in both directions on 2.43, so the exclusions are
    # re-applied here. `test_export_files_are_never_surfaces` plants both and checks each layer.
    return [p for p in paths if not _EXPORT_NAME.search(p) and not p.startswith(records)]


GLOBS: dict[str, tuple[str, ...]] = {
    "readme": ("README*.md", "**/README*.md", "CONTRIBUTING.md", "**/CONTRIBUTING.md", "SECURITY.md",
               "**/SECURITY.md"),
    "manual_docs": ("manual/docs/**/*.md", "manual/docs/**/*.mdx"),
    "manual_config": ("manual/docusaurus.config.ts", "manual/sidebars.ts"),
    "svg": ("frontend/public/**/*.svg", "frontend/src/assets/**/*.svg"),
    "index_html": ("frontend/index.html",),
    "backend_strings": ("backend/src/**/*.py",),
}

_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
_FRONT = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_FRONT_KEY = re.compile(r"^(title|sidebar_label)\s*:\s*(.+?)\s*$", re.MULTILINE)
_CONFIG_KEY = re.compile(r"\b(title|tagline|label|copyright)\s*:\s*(['\"`])((?:\\.|(?!\2).)*)\2", re.DOTALL)
_SVG_TEXT = re.compile(r"<(title|text)\b[^>]*>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
_HTML_TITLE = re.compile(r"<title\b[^>]*>(.*?)</title>", re.DOTALL | re.IGNORECASE)


def _strip_code_fences(text: str) -> list[tuple[int, str]]:
    out, fenced = [], False
    for n, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if not fenced:
            out.append((n, line))
    return out


def readme_items(path: str, text: str) -> list[Item]:
    items = []
    for n, line in _strip_code_fences(text):
        m = _HEADING.match(line)
        if m:
            items.append(Item("readme", f"{path}:{n}", m.group(2)))
    return items


def manual_doc_items(path: str, text: str) -> list[Item]:
    items = []
    body_start = 0
    fm = _FRONT.match(text)
    if fm:
        body_start = fm.end()
        for m in _FRONT_KEY.finditer(fm.group(1)):
            items.append(Item("manual_docs", f"{path} front matter {m.group(1)}", m.group(2).strip("'\"")))
    offset = text[:body_start].count("\n")
    for n, line in _strip_code_fences(text[body_start:]):
        m = _HEADING.match(line)
        if m and len(m.group(1)) == 1:
            items.append(Item("manual_docs", f"{path}:{n + offset} first heading", m.group(2)))
            break
    return items


def manual_config_items(path: str, text: str) -> list[Item]:
    """Known config shape: `title`, `tagline`, `label`, `copyright` string values (navbar, footer, sidebars)."""
    items = []
    for m in _CONFIG_KEY.finditer(text):
        line = text[: m.start()].count("\n") + 1
        items.append(Item("manual_config", f"{path}:{line} {m.group(1)}", m.group(3)))
    return items


def svg_items(path: str, text: str) -> list[Item]:
    return [Item("svg", f"{path} <{m.group(1)}>", re.sub(r"\s+", " ", m.group(2)).strip())
            for m in _SVG_TEXT.finditer(text)]


def index_html_items(path: str, text: str) -> list[Item]:
    return [Item("index_html", f"{path} <title>", m.group(1).strip()) for m in _HTML_TITLE.finditer(text)]


def backend_string_items(path: str, text: str) -> list[Item]:
    """String constants of a module (AST), docstrings excluded (TPD-16; never comments)."""
    tree = ast.parse(text, filename=path)
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))
    items = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            # Wrapped in quotes, the literal's source form: register patterns anchor on them (DR-01's
            # quoted kinds), and a quote is never a token character.
            items.append(Item("backend_strings", f"{path}:{node.lineno}", f'"{node.value}"'))
    return items


EXTRACTORS: dict[str, Callable[[str, str], list[Item]]] = {
    "readme": readme_items,
    "manual_docs": manual_doc_items,
    "manual_config": manual_config_items,
    "svg": svg_items,
    "index_html": index_html_items,
    "backend_strings": backend_string_items,
}


def file_items(cls: str, root: Path = REPO_ROOT) -> tuple[list[str], list[Item]]:
    """`(files discovered, items extracted)` for one file-backed surface class."""
    if cls in P9_CLASSES:
        files = [f for f in ls_files(root, P9_GLOBS[cls], P9_RECORD_PREFIXES) if not _p9_excluded(cls, f)]
    else:
        files = ls_files(root, GLOBS[cls])
    items: list[Item] = []
    for rel in files:
        raw = (root / rel).read_bytes()
        if b"\0" in raw[:8192]:  # binary (an image, a font): only its NAME can be a finding
            text = ""
        else:
            text = raw.decode("utf-8", errors="replace")
        items.extend(EXTRACTORS[cls](rel, text))
    return files, items


def api_items(app: Any) -> list[Item]:
    """The FastAPI title and the health `service`, read from the app object and its served route."""
    from fastapi.testclient import TestClient

    items = [Item("api", "FastAPI title", app.title)]
    body = TestClient(app).get("/api/health").json()
    items.append(Item("api", "GET /api/health service", body["service"]))
    return items


def _run(coro: Any) -> Any:
    # `asyncio.run` cannot nest inside a running loop (async tests): run in a worker thread.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _tool_text(result: Any) -> str:
    """The text a tool call answered with, whatever the mcp version's return shape."""
    parts: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for v in value:
                walk(v)
        elif hasattr(value, "text") and isinstance(value.text, str):
            parts.append(value.text)
        elif hasattr(value, "content"):
            walk(value.content)
        elif isinstance(value, dict):
            parts.append(json.dumps(value))
        elif isinstance(value, str):
            parts.append(value)

    walk(result)
    return "\n".join(parts)


def _decoded(text: str) -> str:
    """A tool answer as the agent READS it. A tool returns a JSON document, so the raw text holds
    `\\n` and `\\"` escapes: scanning it reported the token `nmiStudio` for a line that starts with the  (p9: keep, brand-guard plant or rule)
    word, and could never match a quoted document kind (`"mistudio.circuit-definition"`, which the
    register's DR-01 record allows only with its quotes) because the closing quote arrived as `\\"`.
    The strings of the parsed document are scanned instead, each re-quoted the way JSON shows a string
    value; text that is not JSON is scanned as it is."""
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return text
    parts: list[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, dict):
            for k, item in v.items():
                parts.append(str(k))
                walk(item)
        elif isinstance(v, (list, tuple)):
            for item in v:
                walk(item)

    walk(value)
    return "\n".join(parts) if parts else text


def mcp_items(server: Any) -> list[Item]:
    """Server name, instructions, every tool's name and description, and the howto topics' text — all from
    the built server object, never a tool module."""
    items = [Item("mcp", "server name", server.name), Item("mcp", "server instructions", server.instructions or "")]
    tools = _run(server.list_tools())
    for tool in tools:
        items.append(Item("mcp", f"tool name {tool.name}", tool.name))
        items.append(Item("mcp", f"tool description {tool.name}", tool.description or ""))
    howto = [t.name for t in tools if t.name.endswith("_howto")]
    for name in howto:
        index = json.loads(_tool_text(_run(server.call_tool(name, {}))))
        for topic in index.get("topics", []):
            if topic == "tools":  # derived from the registry: the tool names/descriptions above
                continue
            text = _decoded(_tool_text(_run(server.call_tool(name, {"topic": topic}))))
            items.append(Item("mcp", f"howto topic {topic}", text))
    return items


def live_mcp_server() -> Any:
    """The real server, every category on and every product URL set (the `next_calls` recipe)."""
    import pytest

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.products import PRODUCTS
    from src.mcp_server.server import build_server
    from src.mcp_server.tools import all_categories

    mp = pytest.MonkeyPatch()
    try:
        for spec in PRODUCTS:
            mp.setenv(spec.url_env, f"http://{spec.name}.test")
        settings = MCPSettings(tool_categories=",".join(sorted(all_categories())), allow_anonymous=True)
        server, _client = build_server(settings, stdio=True)
    finally:
        mp.undo()
    return server


_LIVE: dict[str, list[Item]] = {}


def live_items(cls: str) -> list[Item]:
    """Built once per test process (the server takes ~20 s to build and list)."""
    if cls not in _LIVE:
        _LIVE[cls] = _live_items(cls)
    return list(_LIVE[cls])


def _live_items(cls: str) -> list[Item]:
    if cls == "api":
        from src.main import app

        return api_items(app)
    if cls == "mcp":
        return mcp_items(live_mcp_server())
    raise KeyError(cls)


# ── the scanner ────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Rules:
    patterns: tuple[re.Pattern[str], ...]          # (a) entry patterns, (d) active allow guards
    literals: tuple[str, ...]                       # (a) entry literals (exact)
    olds_exact: tuple[str, ...]                     # (b) env, history_tool olds
    olds_ci: tuple[str, ...]                        # (b) host olds (DNS is case-insensitive)
    repos: tuple[re.Pattern[str], ...]              # (c) whole <owner>/<name> tokens
    deny: tuple[tuple[str, re.Pattern[str]], ...]   # active deny guards


def rules_for(export: le.Export) -> Rules:
    entries = [e for e in export.entries() if e["status"] != "retired"]
    active = export.active_guards()
    return Rules(
        patterns=tuple([re.compile(e["pattern"]) for e in entries]
                       + [re.compile(g["pattern"]) for g in active if g["effect"] == "allow"]),
        literals=tuple(lit for e in entries for lit in e["literals"]),
        olds_exact=tuple([r["old"] for r in export.env() if r["status"] == "transition"]
                         + [r["old"] for r in export.history_tools()]),
        olds_ci=tuple(r["old"] for r in export.hosts() if r["status"] == "transition"),
        repos=tuple(re.compile(r"(?<![\w-])" + re.escape(r["old"]) + r"(?![\w-])", re.IGNORECASE)
                    for r in export.repos("transition")),
        deny=tuple((g["id"], re.compile(g["pattern"])) for g in active if g["effect"] == "deny"),
    )


def _allowed_spans(text: str, rules: Rules) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for pattern in (*rules.patterns, *rules.repos):
        spans.extend(m.span() for m in pattern.finditer(text) if m.end() > m.start())
    low = text.lower()
    for needle, hay in ([(x, text) for x in (*rules.literals, *rules.olds_exact)]
                        + [(x.lower(), low) for x in rules.olds_ci]):
        start = hay.find(needle)
        while start != -1:
            spans.append((start, start + len(needle)))
            start = hay.find(needle, start + 1)
    return spans


def _covered(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(s <= start and end <= e for s, e in spans)


def _token(text: str, start: int, end: int) -> str:
    while start > 0 and TOKEN_CHAR.match(text[start - 1]):
        start -= 1
    while end < len(text) and TOKEN_CHAR.match(text[end]):
        end += 1
    return text[start:end].strip(".")


def scan(items: Iterable[Item], rules: Rules) -> list[Finding]:
    findings: list[Finding] = []
    for item in items:
        text = item.text
        # Fast path (P9 classes hand in every line of a file): a text with no old-brand stem and no deny-guard
        # match can produce no finding below, so its allowances are not computed.
        if not STEM.search(text) and not any(p.search(text) for _g, p in rules.deny):
            continue
        spans = _allowed_spans(text, rules)
        brand: list[tuple[int, int, Finding]] = []
        for m in STEM.finditer(text):
            if _covered(spans, *m.span()):
                continue
            if item.regions is not None and not any(a <= m.start() < b for a, b in item.regions):
                continue
            f = Finding(item.cls, item.where, _token(text, *m.span()), [f"brand:{m.group(0).lower()}"], item.kept)
            brand.append((m.start(), m.end(), f))
            findings.append(f)
        for guard_id, pattern in rules.deny:
            for m in pattern.finditer(text):
                if m.end() == m.start() or _covered(spans, *m.span()):
                    continue
                hit = [f for s, e, f in brand if s < m.end() and m.start() < e]
                if hit:
                    for f in hit:
                        if guard_id not in f.rules:
                            f.rules.append(guard_id)
                elif item.regions is None or any(a <= m.start() < b for a, b in item.regions):
                    findings.append(Finding(item.cls, item.where, m.group(0), [guard_id], item.kept))
    return findings


def enforced(mode: str = BRAND_GUARD_MODE) -> bool:
    if mode not in ("report", "enforce"):
        raise ValueError(f"BRAND_GUARD_MODE must be 'report' or 'enforce', not {mode!r}")
    return mode == "enforce"


def mode_required_by(export: le.Export) -> str:
    """The mode the export's `closed` requires: `enforce` once P4 is closed (CF-21), else `report`."""
    return "enforce" if export.closed_at_least("P4") else "report"


def effective_mode(export: le.Export, mode: str | None = None, cls: str | None = None) -> str:
    """The mode a run against `export` actually uses (R23, operator 2026-11-11).

    ENFORCED where the full export exists: the private repository, whose CI a commit must pass before it
    is mirrored. REPORTED on the public mirror, whatever the constant says: the mirror carries the public
    export alone, which withholds the old host names (DR-29) and the private repository names, so a
    string those records allow would be a finding there and an enforced mirror run would fail on text
    the private run has already passed. The private CI is the enforcement point; nothing withheld is
    published to make the mirror able to enforce.

    `mode` defaults to the module constant READ AT CALL TIME (so the P4-close flip needs no other edit).

    `cls` (043 FR-043.2): a P9 class reports until its group closes (`CLASS_CLOSED[cls]`, read at call time);
    a report-only class always reports. The public-export rule comes first and holds for every class.
    """
    mode = BRAND_GUARD_MODE if mode is None else mode
    enforced(mode)  # refuses anything but the two modes
    if export.is_public:
        return "report"
    if cls in REPORT_ONLY_CLASSES:
        return "report"
    if cls in P9_CLASSES and not CLASS_CLOSED[cls]:
        return "report"
    return mode


def report(label: str, findings: list[Finding], mode: str = BRAND_GUARD_MODE) -> list[str]:
    """Print every finding; in enforce mode return them as failures (the caller asserts the list empty)."""
    lines = sorted(f.line() for f in findings if not f.kept)
    kept = sum(1 for f in findings if f.kept)
    print(f"\n[brand guard {mode}] {label}: {len(lines)} finding(s), {kept} kept by a keep mark")
    for line in lines:
        print("  " + line)
    return lines if enforced(mode) else []



# ── ENH-009 P9 classes (043 FR-043.2; FTDD §6.1; FTID §3.1) ─────────────────────────────────────────────
#: Turned True in the LAST commit of each residue group (RG-1 comments, RG-2 docs_text, RG-3 test_text, RG-5
#: identifiers, RG-7 deploy_text). `test_closed_groups_are_enforced` holds a True flag to an enforcing mode, and
#: `frontend/src/test/brandMode.ts` mirrors the map (`brandMode.test.ts` holds the two equal).
CLASS_CLOSED: dict[str, bool] = {
    "comments": True,      # RG-1 closed 2026-11-14 (043): residue left is marked `p9` + `: keep, <reason>`
    "docs_text": True,     # RG-2 closed 2026-11-14
    "test_text": True,     # RG-3 closed 2026-11-14
    "test_values": False,  # RG-6: never closed (P9-D6, NEVER_CLOSED)
    "identifiers": True,   # RG-5 closed 2026-11-14
    "deploy_text": True,   # RG-7 closed 2026-11-14
}

#: P9-D6: a test value changes only with the code value it asserts, so RG-6 never closes as a class: a test
#: may assert an old value on purpose (a frozen value, a retired name). `test_closed_groups_are_enforced` holds it.
NEVER_CLOSED = frozenset({"test_values"})

#: A line that keeps an old name on purpose says so, in its file type's comment form, with a reason
#: (`# p9` + `: keep, a sentence about the rename`, written whole). The guard reports its findings as kept.
KEEP_MARK = re.compile(r"\bp9: keep\b")
KEEP_MARK_WITH_REASON = re.compile(r"\bp9: keep, \S")

#: Record trees for the P9 classes: dated records and dated agent sessions. `.claude/commands/` and
#: `.claude/*.md` are live agent instructions and are scanned (`docs_text`).
P9_RECORD_PREFIXES = ("0xcc/", ".claude/context/")

_PY = ("**/*.py",)
P9_GLOBS: dict[str, tuple[str, ...]] = {
    "comments": (
        "backend/**/*.py", "backend/Dockerfile", "backend/docker-entrypoint.sh", "backend/requirements*.txt",
        "backend/constraints.txt", "backend/pyproject.toml", "backend/systemd/**", "backend/.env.example",
        "rule_runner/**", "frontend/src/**/*.ts", "frontend/src/**/*.tsx", "frontend/src/**/*.css",
        "frontend/*.ts", "frontend/*.js", "frontend/Dockerfile", "k8s/**", "docker/**", "docker-compose*.yml",
        ".github/**", "scripts/**", "nginx/**", "manual/scripts/**", "manual/src/**", "manual/*.ts",
        "*.sh", ".env.example", "miworkbench",
    ),
    "docs_text": (
        "docs/**/*.md", "manual/**/*.md", "manual/**/*.mdx", "README*.md",
        "**/README*.md", "CODE_OF_CONDUCT.md", "CONTRIBUTING.md", "SECURITY.md", "CLAUDE.md",
        ".claude/commands/**", ".claude/*.md", "scripts/**/*.md",
    ),
    "test_text": (
        "backend/tests/**", "rule_runner/tests/**", "frontend/src/**/*.test.ts", "frontend/src/**/*.test.tsx",
        "frontend/src/test/**", "frontend/src/**/__tests__/**",
    ),
    "identifiers": ("backend/src/**", "rule_runner/**/*.py", "frontend/src/**/*.ts", "frontend/src/**/*.tsx"),
    "test_values": (
        "backend/tests/**", "rule_runner/tests/**", "frontend/src/**/*.test.ts", "frontend/src/**/*.test.tsx",
        "frontend/src/test/**", "frontend/src/**/__tests__/**",
    ),
    "deploy_text": (
        "docker-compose*.yml", "docker/**", "backend/Dockerfile", "backend/docker-entrypoint.sh",
        "backend/pyproject.toml", "backend/systemd/**", ".env.example", "backend/.env.example", ".github/**",
        "scripts/**", "backend/scripts/**", "*.sh", "miworkbench", "nginx/**", "k8s/**", "frontend/Dockerfile",
        "frontend/vite.config.ts", "manual/scripts/**", "manual/package.json", "rule_runner/Dockerfile",
    ),
}

#: History inside a class's scope (FTDD §6.2: file-level keeps are class scope only). Applied migrations run on
#: every fresh database and are pinned byte for byte (RG-8, P9-D8); the MCP rename map is the old tool names kept
#: to read history (RG-9, 042 FTASKS 18.12); `docs/schemas/` holds generated and vendored contracts.
_P9_EXCLUDE = re.compile(r"^(backend/alembic/versions/|backend/tests/support/mcp_rename_map\.json$|docs/schemas/)")


def _p9_excluded(cls: str, path: str) -> bool:
    if _P9_EXCLUDE.match(path):
        return True
    if cls == "deploy_text" and path.endswith(".md"):
        return True  # prose beside deploy files is docs_text
    if cls == "identifiers" and (path.startswith("rule_runner/tests/") or _is_frontend_test(path)):
        return True
    return False


def _is_frontend_test(path: str) -> bool:
    return bool(re.search(r"\.(test|spec)\.tsx?$", path)) or path.startswith("frontend/src/test/") \
        or "/__tests__/" in path


# -- lexing: which characters of a file are comments, which are code -------------------------------------
def _kind(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    if name.endswith(".py"):
        return "py"
    if name.endswith((".ts", ".tsx", ".js", ".mjs", ".cjs", ".jsx")):
        return "js"
    if name.endswith(".css"):
        return "css"
    if name.endswith((".md", ".mdx", ".json", ".txt.md")):
        return "text"
    return "hash"  # YAML, shell, Dockerfile, nginx, ini, toml, requirements, .env, .gitattributes, systemd


def _py_spans(text: str) -> list[tuple[int, int, int, int]]:
    """(start_row, start_col, end_row, end_col) of every comment and docstring of a Python file (1-based rows)."""
    import io
    import tokenize

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    doc_starts: set[tuple[int, int]] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                doc_starts.add((body[0].value.lineno, body[0].value.col_offset))
    spans = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT or (tok.type == tokenize.STRING and tok.start in doc_starts):
            spans.append((*tok.start, *tok.end))
    return spans


def _py_string_spans(text: str) -> list[tuple[int, int, int, int]]:
    """Every string literal of a Python file that is not a docstring (an f-string whole, start to end)."""
    import io
    import tokenize

    docs = {(sr, sc) for sr, sc, _er, _ec in _py_spans(text)}
    spans, fstart = [], []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.STRING and tok.start not in docs:
                spans.append((*tok.start, *tok.end))
            elif tok.type == getattr(tokenize, "FSTRING_START", -1):
                fstart.append(tok.start)
            elif tok.type == getattr(tokenize, "FSTRING_END", -1) and fstart:
                spans.append((*fstart.pop(), *tok.end))
    except (tokenize.TokenError, SyntaxError):
        return []
    return spans


def _js_spans(text: str, css: bool = False) -> list[tuple[int, int, int, int]]:
    """Comment spans of a JavaScript/TypeScript/CSS file: `//` (not CSS) and `/* */`, skipping string and
    template literals. Regular-expression literals are not recognised; a `//` inside one is written with escaped slashes."""
    return _js_lex(text, css)[0]


def _js_string_spans(text: str) -> list[tuple[int, int, int, int]]:
    """The string and template-literal spans of a JavaScript/TypeScript file (a template's `${…}` is code)."""
    return _js_lex(text)[1]


def _js_lex(text: str, css: bool = False) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    spans: list[tuple[int, int, int, int]] = []
    strings: list[tuple[int, int, int, int]] = []
    sstart = (0, 0)
    i, n, row, col = 0, len(text), 1, 0
    quote: str | None = None
    tmpl_depth: list[int] = []  # brace depth at each `${` of a template literal
    depth = 0

    def advance(k: int) -> None:
        nonlocal i, row, col
        for _ in range(k):
            if i < n and text[i] == "\n":
                row, col = row + 1, 0
            else:
                col += 1
            i += 1

    while i < n:
        c = text[i]
        if quote:
            if c == "\\":
                advance(2)
                continue
            if quote == "`" and text.startswith("${", i):
                strings.append((*sstart, row, col))
                tmpl_depth.append(depth)
                quote = None
                advance(2)
                continue
            if c == quote or (c == "\n" and quote != "`"):
                quote = None
                advance(1)
                strings.append((*sstart, row, col))
                continue
            advance(1)
            continue
        if text.startswith("/*", i):
            start = (row, col)
            end = text.find("*/", i + 2)
            end = n if end == -1 else end + 2
            advance(end - i)
            spans.append((*start, row, col))
            continue
        if not css and text.startswith("//", i):
            start = (row, col)
            end = text.find("\n", i)
            end = n if end == -1 else end
            advance(end - i)
            spans.append((*start, row, col))
            continue
        if c in "'\"`":
            quote = c
            sstart = (row, col)
        elif c == "{":
            depth += 1
        elif c == "}":
            if tmpl_depth and tmpl_depth[-1] == depth:
                tmpl_depth.pop()
                quote = "`"
                sstart = (row, col)
            else:
                depth -= 1
        advance(1)
    return spans, strings


_HASH_QUOTE = re.compile(r"""'[^'\n]*'|"(?:\\.|[^"\\\n])*\"""")


def _hash_comment_col(line: str) -> int:
    """Column of the line's comment: the first `#` outside quotes that starts the line or follows whitespace."""
    masked = _HASH_QUOTE.sub(lambda m: " " * len(m.group(0)), line)
    for m in re.finditer("#", masked):
        if m.start() == 0 or masked[m.start() - 1] in " \t":
            return m.start()
    return -1


def _split(path: str, text: str) -> tuple[list[tuple[int, int, str]], list[str]]:
    """`(comment segments as (row, col, text), code lines with comments blanked)` for one file."""
    lines = text.split("\n")
    kind = _kind(path)
    comments: list[tuple[int, int, str]] = []
    code = list(lines)
    if kind == "text":
        return comments, code
    if kind == "hash":
        for r, line in enumerate(lines, 1):
            c = _hash_comment_col(line)
            if c >= 0:
                comments.append((r, c, line[c:]))
                code[r - 1] = line[:c]
        return comments, code
    spans = _py_spans(text) if kind == "py" else _js_spans(text, css=(kind == "css"))
    for sr, sc, er, ec in spans:
        for r in range(sr, er + 1):
            line = lines[r - 1]
            a = sc if r == sr else 0
            b = ec if r == er else len(line)
            comments.append((r, a, line[a:b]))
            code[r - 1] = code[r - 1][:a] + " " * (b - a) + code[r - 1][b:]
    return comments, code


def _marked(text: str) -> set[int]:
    """Rows (1-based) whose source line carries a keep mark."""
    return {r for r, line in enumerate(text.split("\n"), 1) if KEEP_MARK.search(line)}


def comment_items(path: str, text: str) -> list[Item]:
    """RG-1: every comment and docstring, one item per physical line (FTID §3.1)."""
    marked = _marked(text)
    return [Item("comments", f"{path}:{r}", seg, c, r in marked) for r, c, seg in _split(path, text)[0]
            if seg.strip()]


#: The dated regions of `CLAUDE.md` (P9-D2: dated lessons stay as written). Each is a heading and the next
#: heading after it; the extractor refuses a file where either is missing, so a reorganised file fails loudly
#: instead of scanning nothing or everything.
CLAUDE_DATED_REGIONS = (("## Current Status", "## PRIMARY UI/UX REFERENCE"),
                        ("## Session History Log", "## Research Integration"))


def _dated_rows(text: str) -> set[int]:
    lines = text.split("\n")
    rows: set[int] = set()
    for start, end in CLAUDE_DATED_REGIONS:
        a = [i for i, line in enumerate(lines, 1) if line.strip() == start]
        b = [i for i, line in enumerate(lines, 1) if line.strip() == end]
        if len(a) != 1 or len(b) != 1 or b[0] <= a[0]:
            raise ValueError(f"CLAUDE.md: dated region {start!r} .. {end!r} not found exactly once in order")
        rows.update(range(a[0], b[0]))
    return rows


def docs_text_items(path: str, text: str) -> list[Item]:
    """RG-2: every line of a document, code fences included (a command with an old name is still an
    instruction), except `CLAUDE.md`'s dated regions."""
    skip = _dated_rows(text) if path == "CLAUDE.md" else set()
    marked = _marked(text)
    return [Item("docs_text", f"{path}:{r}", line, 0, r in marked) for r, line in enumerate(text.split("\n"), 1)
            if r not in skip and line.strip()]


def _code_items(cls: str, path: str, text: str) -> list[Item]:
    marked = _marked(text)
    items = [Item(cls, f"{path}:{r}", line, 0, r in marked) for r, line in enumerate(_split(path, text)[1], 1)
             if line.strip()]
    if STEM.search(path):
        items.append(Item(cls, f"{path} (file name)", path))
    return items


def _string_rows(path: str, text: str) -> dict[int, list[tuple[int, int]]] | None:
    """Per row, the column spans of string literals; None for a file with no code (JSON, Markdown, YAML):
    every character of it is a value."""
    kind = _kind(path)
    if kind == "py":
        spans = _py_string_spans(text)
    elif kind == "js":
        spans = _js_string_spans(text)
    else:
        return None
    lines = text.split("\n")
    rows: dict[int, list[tuple[int, int]]] = {}
    for sr, sc, er, ec in spans:
        for r in range(sr, er + 1):
            a = sc if r == sr else 0
            b = ec if r == er else len(lines[r - 1])
            rows.setdefault(r, []).append((a, b))
    return rows


def _complement(spans: list[tuple[int, int]], width: int) -> tuple[tuple[int, int], ...]:
    out, at = [], 0
    for a, b in sorted(spans):
        if a > at:
            out.append((at, a))
        at = max(at, b)
    out.append((at, max(width, at) + 1))
    return tuple(out)


def test_text_items(path: str, text: str) -> list[Item]:
    """RG-3: test code with comments, docstrings and string literals left out (names: functions, classes,
    variables, attributes), and the file name. A test file with no code (a fixture) has no names."""
    rows = _string_rows(path, text)
    items = []
    for item in _code_items("test_text", path, text):
        if item.where.endswith(" (file name)"):
            items.append(item)
        elif rows is not None:
            r = int(item.where.rsplit(":", 1)[1])
            items.append(Item("test_text", item.where, item.text, 0, item.kept,
                              _complement(rows.get(r, []), len(item.text))))
    return items


def test_values_items(path: str, text: str) -> list[Item]:
    """RG-6: the string literals of test code, and every line of a fixture file (P9-D6: never a closed class)."""
    rows = _string_rows(path, text)
    items = []
    for item in _code_items("test_values", path, text):
        if item.where.endswith(" (file name)"):
            continue
        if rows is None:
            items.append(item)
            continue
        r = int(item.where.rsplit(":", 1)[1])
        if rows.get(r):
            items.append(Item("test_values", item.where, item.text, 0, item.kept, tuple(rows[r])))
    return items


def identifier_items(path: str, text: str) -> list[Item]:
    """RG-5: source code with comments and docstrings blanked (names and string values); and the file name."""
    return _code_items("identifiers", path, text)


def deploy_text_items(path: str, text: str) -> list[Item]:
    """RG-7: deploy, CI and tooling values outside comments (compose services, networks, variables, paths)."""
    marked = _marked(text)
    return [Item("deploy_text", f"{path}:{r}", line, 0, r in marked)
            for r, line in enumerate(_split(path, text)[1], 1) if line.strip()]


EXTRACTORS.update({
    "comments": comment_items,
    "docs_text": docs_text_items,
    "test_text": test_text_items,
    "test_values": test_values_items,
    "identifiers": identifier_items,
    "deploy_text": deploy_text_items,
})
