"""The brand guard, backend half (042 FR-042.13, FR-042.34; FTDD §6.7; FTID §3.14; FTASKS 7.8, 7.9; T-16).

Report mode until S001 closes P4 (`BRAND_GUARD_MODE`, tied to the export by `test_brand_guard_mode.py`):
`test_real_tree_report` prints every finding and passes; in enforce mode it fails on any. Everything else
here is enforced in both modes: the scanner finds one plant per surface class, discovery is never empty,
the MCP surface is the live registry, old repository names pass only through `transition` repo records,
the guard modules carry no repository name, the suite-phrase rules come from the export's guard records by
phase, and the export itself is never a surface (CF-20).

Nothing here re-types a register fact: allowances come from `docs/legacy-identifiers*.md` through
`tests/support/legacy_export.py`; repository names are taken from its `repo` records.
"""
from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from tests.support import brand_surfaces as bs
from tests.support import legacy_export as le

ROOT = le.REPO_ROOT


@pytest.fixture(scope="module")
def export() -> le.Export:
    return le.load()


def _with(export: le.Export, *, header: dict | None = None, records=None) -> le.Export:
    """A copy of the committed export with header keys or records replaced (never re-typed)."""
    return le.Export(export.path, {**export.header, **(header or {})},
                     tuple(export.records if records is None else records))


def _git_tree(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "tree"
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    return root


# One plant per surface class. Each text carries the plant once, beside text the guard must ignore.
FIXTURE_FILES = {
    "README.md": "# Plant miStudio here\n\nBody text naming miLLM is not a heading.\n",
    "manual/docs/intro.md": "---\ntitle: Planted millm page\n---\n\nBody miStudio text.\n",
    "manual/docusaurus.config.ts": "export default {\n  title: 'Planted MechInterp Studio',\n  url: 'x',\n};\n",
    "frontend/public/icon.svg": "<svg><title>miLLM plant</title><desc>miStudio not a surface</desc></svg>\n",
    "frontend/index.html": "<html><head><title>miStudio plant</title></head><body>miLLM</body></html>\n",
    "backend/src/planted.py": '"""Docstring naming miStudio is not scanned."""\nLABEL = "Open miLLM now"\n# miStudio comment\n',
}


def _tiny_app():
    from fastapi import FastAPI

    app = FastAPI(title="Planted app")

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "service": "miStudio planted service"}

    return app


def _tiny_mcp():
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("planted", instructions="Instructions with nothing old.")

    @server.tool()
    async def plain_tool() -> str:
        """Description naming miLLM once."""  # p9: keep, brand-guard plant or rule
        return "x"

    return server


def test_fixture_tree_one_finding_per_surface_class(tmp_path, export):
    root = _git_tree(tmp_path, FIXTURE_FILES)
    rules = bs.rules_for(export)
    per_class: dict[str, list[bs.Finding]] = {}
    for cls in bs.FILE_CLASSES:
        if cls in bs.P9_CLASSES:  # their own plants: test_p9_class_finds_its_plant
            continue
        _files, items = bs.file_items(cls, root)
        per_class[cls] = bs.scan(items, rules)
    per_class["api"] = bs.scan(bs.api_items(_tiny_app()), rules)
    per_class["mcp"] = bs.scan(bs.mcp_items(_tiny_mcp()), rules)
    assert set(per_class) == set(bs.ENFORCED_CLASSES) | set(bs.REPORT_ONLY_CLASSES)
    counts = {cls: len(f) for cls, f in per_class.items()}
    assert counts == {cls: 1 for cls in per_class}, {c: [f.line() for f in fs] for c, fs in per_class.items()}


@pytest.mark.parametrize("cls", bs.FILE_CLASSES)
def test_discovery_not_empty_per_class(cls):
    files, items = bs.file_items(cls)
    assert files, f"surface class {cls!r} discovered no file: a guard that discovers nothing asserts nothing"
    assert items, f"surface class {cls!r} extracted no text from {len(files)} file(s)"
    records = bs.P9_RECORD_PREFIXES if cls in bs.P9_CLASSES else ("0xcc/", ".claude/")
    assert not [f for f in files if f.startswith(records)]


@pytest.mark.parametrize("cls", bs.LIVE_CLASSES)
def test_discovery_not_empty_per_live_class(cls):
    items = bs.live_items(cls)
    assert items and all(i.text for i in items if i.where != "server instructions"), cls


def test_real_tree_report(export):
    """Prints every finding per class; fails only where the guard is enforced: the constant says
    `enforce` AND this tree holds the full export (R23: the private repository enforces, the public
    mirror reports). The report-only class (TPD-16) is printed and never enforced."""
    rules = bs.rules_for(export)
    failures: list[str] = []
    for cls in (*bs.FILE_CLASSES, *bs.LIVE_CLASSES):
        items = bs.file_items(cls)[1] if cls in bs.FILE_CLASSES else bs.live_items(cls)
        findings = bs.scan(items, rules)
        mode = bs.effective_mode(export, cls=cls)  # report-only and open P9 classes report (043 FR-043.2)
        failures += bs.report(cls, findings, mode)
    assert not failures, "\n".join(failures)


def test_mcp_surface_read_from_build_server(monkeypatch):
    """The MCP surface is the BUILT server's registry: the same tool set the live-registry recipe sees."""
    from src.mcp_server.tools import howto
    from tests.support.next_calls import live_input_schemas

    items = bs.live_items("mcp")
    names = {i.text for i in items if i.where.startswith("tool name ")}
    assert names == set(live_input_schemas()), "brand guard's tool list differs from the live registry"
    assert len(names) > 300
    described = {i.where.removeprefix("tool description ") for i in items if i.where.startswith("tool description ")}
    assert described == names
    server = [i for i in items if i.where == "server name"]
    assert len(server) == 1 and server[0].text  # the server's own name, read from the object
    topics = [i for i in items if i.where.startswith("howto topic ")]
    assert len(topics) >= 10 and all(len(t.text) > 100 for t in topics), "howto topics were not read"
    # A PLANT through the real howto path. Before 2026-11-11 this line asked only that SOME topic held a
    # finding, which was true while the real guidance still named an old brand and would have gone red,
    # proving nothing, the day the guidance was clean. Now one old brand is put into the text the real
    # tool serves for one topic, the server is built again, and exactly that topic gains exactly one finding.
    rules = bs.rules_for(le.load())
    topic = sorted(howto.TOPICS)[-1]
    where = f"howto topic {topic}"
    before = [f.line() for f in bs.scan([i for i in topics if i.where == where], rules)]
    marker = "Planted: open mi" + "Studio now."
    monkeypatch.setitem(howto.TOPICS, topic, howto.TOPICS[topic] + "\n" + marker + "\n")
    planted = [i for i in bs.mcp_items(bs.live_mcp_server()) if i.where == where]
    assert len(planted) == 1 and marker in planted[0].text, "the planted topic text was not read from the server"
    after = [f.line() for f in bs.scan(planted, rules)]
    gained = list(after)
    for line in before:
        gained.remove(line)
    assert len(gained) == 1 and gained[0].startswith(f"[mcp] {where}: mi" + "Studio "), (before, after)


def _howto_mcp(text: str):
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("planted", instructions="Nothing old here.")

    @server.tool()
    async def planted_howto(topic: str | None = None) -> dict:
        """A guide with one topic."""
        return {"topics": ["shapes"]} if topic is None else {"topic": topic, "text": text}

    return server


def test_howto_text_is_scanned_as_the_agent_reads_it(export):
    """A tool answers with a JSON document. Scanned raw, a line starting with an old brand was reported
    as the token `n<brand>` (the `\\n` escape), and a quoted document kind the register allows
    (DR-01, with its quotes) was a finding because its closing quote arrived escaped. The parsed strings
    are what an agent reads, so they are what is scanned."""
    kind = '"mi' + 'studio.circuit-definition"'
    old = "mi" + "Studio"
    text = f"The kind is {kind} exactly.\n{old} starts this line."
    items = [i for i in bs.mcp_items(_howto_mcp(text)) if i.where == "howto topic shapes"]
    assert len(items) == 1 and kind in items[0].text and "\\n" not in items[0].text, items
    findings = bs.scan(items, bs.rules_for(export))
    assert [f.token for f in findings] == [old], [f.line() for f in findings]
    # The raw form is what the two defects came from (so the decode is doing the work here).
    raw = bs.scan([bs.Item("mcp", "raw", json.dumps({"text": text}))], bs.rules_for(export))
    assert sorted(f.token for f in raw) != [old], [f.line() for f in raw]


def _repo_names(export: le.Export) -> list[str]:
    names = {r[k] for r in export.repos() for k in ("old", "new")}
    assert names, "the export lists no repo record"
    return sorted(names)


def test_old_repo_name_allowed_only_by_transition_record(export):
    # Since ENH-009 P7 every repo record is `retired`; the allowance rule is exercised on the committed
    # records put back into `transition` (a copy, never re-typed), and the retired export itself is checked
    # below and by `test_retired_repo_records_allow_nothing`.
    if not export.repos():
        pytest.fail("the export has no repo record to test with")
    if not export.repos("transition"):
        export = _with(export, records=[{**r, "status": "transition"} if r["section"] == "repo" else r
                                        for r in export.records])
    olds = [r["old"] for r in export.repos("transition")]
    rules = bs.rules_for(export)
    for old in olds:
        url = f"Source: https://github.com/{old}.git and {old}/issues"
        assert bs.scan([bs.Item("readme", "x", url)], rules) == [], old
        # Only the WHOLE <owner>/<name> token: a longer name is not covered (TPD-33, miForge N31).
        longer = bs.scan([bs.Item("readme", "x", f"see {old}-backup")], rules)
        assert len(longer) == 1, (old, [f.line() for f in longer])
        # Not the name part alone, under another owner.
        other = bs.scan([bs.Item("readme", "x", "see someone/" + old.split("/", 1)[1])], rules)
        assert len(other) == 1, old
    # P7: the record turns `retired`; every remaining spelling is then a finding.
    retired = _with(export, records=[{**r, "status": "retired"} if r["section"] == "repo" else r
                                     for r in export.records])
    for old in olds:
        found = bs.scan([bs.Item("readme", "x", f"https://github.com/{old}")], bs.rules_for(retired))
        assert found, f"a retired repo record still allows {old}"


def test_retired_repo_records_allow_nothing(export):
    """ENH-009 P7 (FTASKS 18.12): the committed export retires every repo record, so no old repository name
    passes the guard any more."""
    retired = export.repos("retired")
    if export.header.get("phase") in ("P7", "P8") and not export.header.get("withheld"):
        assert retired and not export.repos("transition"), "the P7 export still holds a transition repo record"
    rules = bs.rules_for(export)
    for rec in retired:
        found = bs.scan([bs.Item("readme", "x", f"https://github.com/{rec['old']}")], rules)
        assert found, f"the retired repo record still allows {rec['old']}"


def test_guard_modules_hold_no_repo_literal(export):
    """TPD-33: no hand-kept allow list. The guard modules name no repository (old or new); every allowance
    comes from the export's `repo` records."""
    names = _repo_names(export)
    modules = [ROOT / "backend/tests/support/brand_surfaces.py", ROOT / "frontend/src/test/brandNames.test.ts",
               ROOT / "frontend/src/test/brandMode.ts", ROOT / "frontend/src/test/legacyExport.ts"]
    offenders = []
    for path in modules:
        assert path.is_file(), path
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            strings = [n.value for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
            text = text + "\n" + "\n".join(strings)
        for name in names:
            if re.search(re.escape(name), text, re.IGNORECASE):
                offenders.append(f"{path.relative_to(ROOT)} names {name}")
    assert not offenders, offenders


def _at(export: le.Export, phase: str) -> bs.Rules:
    closed = {"P4": "P3", "P5": "P4"}[phase]
    return bs.rules_for(_with(export, header={"phase": phase, "closed": closed}))


def _found(rules: bs.Rules, text: str) -> list[bs.Finding]:
    return bs.scan([bs.Item("readme", "plant", text)], rules)


def test_suite_phrase_rules_by_phase(export):
    """S001 FR-S001.69 / CF-15 / CF-22 / CF-23 (selectable with `-k suite_phrase`). Phase rules come only
    from the export's guard records: neither guard module holds a phrase or a phase."""
    p4, p5 = _at(export, "P4"), _at(export, "P5")
    suite_phrase, open_app = "part of the miStudio suite", "Open miStudio"
    assert _found(p4, suite_phrase) and not _found(p5, suite_phrase)
    assert _found(p4, open_app) and _found(p5, open_app)
    # miSuite: the suite's name until P5 (allowed), then a G-MISUITE finding.  # p9: keep, brand-guard plant or rule
    assert _found(p4, "miSuite") == []
    p5_misuite = _found(p5, "miSuite")
    assert [f.rules for f in p5_misuite] == [["G-MISUITE"]], [f.line() for f in p5_misuite]
    # CF-22: the suite's paths through G-STUDIO-PATH from P5. Withheld from the public export (it names a
    # workstation path), so the mirror skips this part BY NAME.
    le.require(export, "guard", "G-STUDIO-PATH")
    for path in ("~/app/miStudio/scripts", "-home-x-sean-app-miStudio"):
        assert _found(p4, path), path
        assert _found(p5, path) == [], path


def test_export_files_are_never_surfaces(tmp_path):
    """CF-20: the export names every old name by design, so it is never scanned — root-level and nested
    copies, under a `docs/` positive pathspec and a repository-wide one. Both layers are checked: git's
    glob exclusions (both forms; the `**/` form alone leaks the root-level file on git 2.43) and the
    module's own filter. The finding count is unchanged by the export's lines."""
    export_text = (le.paths()[0] if le.paths()[0].exists() else le.paths()[1]).read_text(encoding="utf-8")
    plant = "# Heading naming miStudio\n"
    base = {"README.md": plant, "docs/notes.md": plant}
    copies = {"docs/legacy-identifiers.md": export_text + plant,
              "docs/legacy-identifiers.public.md": export_text + plant,
              "sub/docs/legacy-identifiers.md": export_text + plant}
    with_export = _git_tree(tmp_path / "a", {**base, **copies})
    without = _git_tree(tmp_path / "b", base)
    for globs in (("docs/*.md",), ("**/*.md",), ("docs/**",), ("*.md", "**/*.md")):
        specs = [f":(glob){g}" for g in globs]
        raw = subprocess.run(["git", "ls-files", "--", *specs, *bs.EXPORT_EXCLUDES], cwd=with_export,
                             capture_output=True, text=True, check=True).stdout.split()
        assert not [p for p in raw if "legacy-identifiers" in p], (globs, raw)  # git layer
        listed = bs.ls_files(with_export, globs)
        assert listed == bs.ls_files(without, globs), globs                      # both layers
    for p in copies:
        assert bs._EXPORT_NAME.search(p), p
    # The module's own filter alone (git's exclusions switched off) still drops every copy.
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(bs, "EXPORT_EXCLUDES", ())
        for globs in (("docs/*.md",), ("**/*.md",)):
            assert bs.ls_files(with_export, globs) == bs.ls_files(without, globs), globs
    finally:
        mp.undo()
    count = lambda root: len(bs.scan([i for f in bs.ls_files(root, ("**/*.md",))
                                      for i in bs.readme_items(f, (root / f).read_text())],
                                     bs.rules_for(le.load())))
    assert count(with_export) == count(without) == 2


def _allowed_elsewhere(export: le.Export, entry: dict, text: str) -> bool:
    """An INDEPENDENT reference (never the scanner's rules): is every brand stem of `text` covered by an
    export record other than `entry`? A scanner allowance that the export does not hold cannot hide here."""
    spans: list[tuple[int, int]] = []
    for other in export.entries():
        if other is entry or other["status"] == "retired":
            continue
        spans += [m.span() for m in re.finditer(other["pattern"], text)]
        for lit in other["literals"]:
            spans += [(m.start(), m.end()) for m in re.finditer(re.escape(lit), text)]
    for r in [*export.env(), *export.hosts(), *export.history_tools()]:
        if r.get("status", "transition") == "transition":
            spans += [(m.start(), m.end()) for m in re.finditer(re.escape(r["old"]), text, re.IGNORECASE)]
    for r in export.repos("transition"):
        spans += [m.span() for m in re.finditer(r"(?<![\w-])" + re.escape(r["old"]) + r"(?![\w-])", text, re.I)]
    for g in export.active_guards():
        if g["effect"] == "allow":
            spans += [m.span() for m in re.finditer(g["pattern"], text)]
    stems = [m.span() for m in re.finditer(r"mi(?:studio|llm)|mechinterp[ -]?studio", text, re.IGNORECASE)]
    return all(any(s <= a and b <= e for s, e in spans) for a, b in stems)


def test_deleted_entry_is_reported_in_enforce_mode(export):
    """FTASKS 6.22 (FR-042.7): remove one entry from the export while the code keeps its value, and the brand
    guard reports the value — a register entry cannot be dropped silently, and the scanner holds no allowance
    of its own. For every entry literal naming an old brand that NO OTHER export record covers (decided by an
    independent reference, not the scanner), the scanner must report it once the entry is gone."""
    checked = []
    rules_full = bs.rules_for(export)
    for entry in export.entries():
        if entry["status"] == "retired":
            continue
        rules_without = bs.rules_for(_with(export, records=[r for r in export.records if r is not entry]))
        for lit in entry["literals"]:
            text = f'"{lit}"'
            if not bs.STEM.search(lit):
                continue
            item = [bs.Item("readme", f"{entry['id']} literal", text)]
            assert bs.scan(item, rules_full) == [], (entry["id"], lit)  # allowed while the entry exists
            if _allowed_elsewhere(export, entry, text):
                continue
            failures = bs.report(f"without {entry['id']}", bs.scan(item, rules_without), "enforce")
            assert failures, f"{entry['id']} removed, yet {lit!r} is still allowed: an allowance outside the export"
            checked.append(entry["id"])
    assert len(set(checked)) >= 5, f"too few entries checked: {sorted(set(checked))}"


# ── ENH-009 P9 classes (043 FR-043.2; FTASKS 1.3, 1.4) ─────────────────────────────────────────────────
#: One plant per P9 class, beside text each class must leave to another class (a comment is never test or
#: identifier text, and the reverse), a dated `CLAUDE.md` region, a record tree and an applied migration.
P9_FIXTURE = {
    "backend/src/p9.py": '"""Docstring with nothing old."""\nVALUE = "plain"  # mi' 'LLM serves this\n'
                         'NAME = "open mi' 'Studio"\n',
    "backend/tests/unit/test_p9.py": '"""Nothing old."""\n\n\ndef test_x():\n    assert "hello mi' 'studio-x" != ""\n\n\n'
                                     'def test_mi' 'llm_y():\n    assert "kept mi' 'LLM"  # p9' ': keep, a planted keep\n',
    "docs/p9.md": "Body text naming mi" "Studio.\n",
    "docker-compose.p9.yml": "services:\n  x:\n    networks: [mi" "llm-net]\n",
    "CLAUDE.md": "# Title\n\n## Current Status\nold mi" "Studio\n## PRIMARY UI/UX REFERENCE\nnothing\n"
                 "## Session History Log\nold mi" "LLM\n## Research Integration\nnothing\n",
    ".claude/context/2026-01-01.md": "mi" "Studio in a dated session\n",
    "backend/alembic/versions/a1_old.py": '"""mi' 'LLM in an applied migration."""\n# mi' 'Studio\n',
    "frontend/src/p9.ts": "const a = 'plain'; // plain\n",
}


@pytest.mark.parametrize("cls", bs.P9_CLASSES)
def test_p9_class_finds_its_plant(tmp_path, export, cls):
    root = _git_tree(tmp_path, P9_FIXTURE)
    files, items = bs.file_items(cls, root)
    findings = [f for f in bs.scan(items, bs.rules_for(export)) if not f.kept]
    assert len(findings) == 1, (cls, [f.line() for f in findings])
    assert not [f for f in files if f.startswith(("0xcc/", ".claude/context/", "backend/alembic/versions/"))]


def test_p9_lexer_splits_comments_from_code():
    """The extractors' split, per file kind: what a comment class sees is never what a code class sees."""
    py = 'X = "a # not a comment"  # a comment\n"""not a docstring: not first"""\n'
    comments, code = bs._split("m.py", '"""Doc."""\n' + py)
    assert [c[2] for c in comments] == ['"""Doc."""', "# a comment"]
    assert '"a # not a comment"' in code[1] and "# a comment" not in code[1]
    ts = "const u = 'http://x'; // note\nconst t = `a ${b // not code\n} c`; /* block\nmore */ const z = 1;\n"
    comments, code = bs._split("m.ts", ts)
    assert [c[2] for c in comments] == ["// note", "// not code", "/* block", "more */"]  # `${…}` is code
    assert "'http://x'" in code[0] and "const z = 1;" in code[3]
    sh = 'echo "a # b" # real\nURL=http://x#frag\n  # indented\n'
    comments, code = bs._split("s.sh", sh)
    assert [c[2] for c in comments] == ["# real", "# indented"]
    assert code[1] == "URL=http://x#frag"
    css = "a { background: url(http://x); } /* c */\n"
    assert [c[2] for c in bs._split("s.css", css)[0]] == ["/* c */"]


def test_claude_md_dated_regions_are_found_or_refused():
    text = P9_FIXTURE["CLAUDE.md"]
    items = bs.docs_text_items("CLAUDE.md", text)
    assert [i.text for i in items] == ["# Title", "## PRIMARY UI/UX REFERENCE", "nothing", "## Research Integration",
                                       "nothing"]  # each region stops at its end heading
    with pytest.raises(ValueError, match="dated region"):
        bs.docs_text_items("CLAUDE.md", text.replace("## Session History Log", "## Renamed"))


def test_p9_record_trees_are_dated_records_only(tmp_path):
    """`.claude/commands/` and `.claude/*.md` are live agent instructions (scanned); `.claude/context/` and
    `0xcc/` are dated records (never scanned)."""
    root = _git_tree(tmp_path, {".claude/commands/start.md": "x\n", ".claude/spec.md": "x\n",
                                ".claude/context/s.md": "x\n", "0xcc/notes.md": "x\n"})
    files, _ = bs.file_items("docs_text", root)
    assert files == [".claude/commands/start.md", ".claude/spec.md"]


def test_closed_groups_are_enforced(export, monkeypatch):
    """FR-043.2 / FTASKS 1.4: a class whose group is closed is ENFORCED where the full export exists, and one
    still open only reports; on the public mirror every class reports (R23)."""
    private = _with(export, header={"variant": "full"}) if export.is_public else export
    for cls in bs.P9_CLASSES:
        want = "enforce" if bs.CLASS_CLOSED[cls] else "report"
        assert bs.effective_mode(private, "enforce", cls=cls) == want, cls
        monkeypatch.setitem(bs.CLASS_CLOSED, cls, True)
        assert bs.effective_mode(private, "enforce", cls=cls) == "enforce", cls
        assert bs.effective_mode(_with(export, header={"variant": "public"}), "enforce", cls=cls) == "report"
        monkeypatch.setitem(bs.CLASS_CLOSED, cls, False)
        assert bs.effective_mode(private, "enforce", cls=cls) == "report", cls
    assert bs.effective_mode(private, "enforce", cls="backend_strings") == "report"
    assert set(bs.CLASS_CLOSED) == set(bs.P9_CLASSES)
    # P9-D6: a test value changes only with the code value it asserts, so RG-6 never closes as a class.
    assert bs.NEVER_CLOSED <= set(bs.P9_CLASSES) and not [c for c in bs.NEVER_CLOSED if bs.CLASS_CLOSED[c]]


#: The residue groups 043 CLOSED on 2026-11-14 (P9 record §8: RG-1, RG-2, RG-3, RG-5, RG-7), typed here BY VALUE.
#: `test_closed_groups_are_enforced` reads its expectation from `CLASS_CLOSED` itself, and `brandMode.test.ts` only
#: holds the frontend map equal to the backend's, so re-opening a group (one flag, or both maps together) passed
#: every test (043 acceptance 2026-11-26, F-1, control B2). A closed group never re-opens: changing this set is a
#: decision recorded in the 043 documents, not an edit to the flag.
CLOSED_AT_P9 = frozenset({"comments", "docs_text", "test_text", "identifiers", "deploy_text"})


def test_the_groups_closed_at_p9_stay_closed_and_enforced(export):
    """FR-043.2 (043 acceptance F-1): the closed set is pinned independently of the flag it checks, and each closed
    class is ENFORCED through the mode path the real-tree test uses (`effective_mode(export, cls=…)`, the module
    constant read at call time). Control: `CLASS_CLOSED["comments"] = False` → red here."""
    closed = {cls for cls, flag in bs.CLASS_CLOSED.items() if flag}
    assert closed == CLOSED_AT_P9, f"closed P9 classes changed: re-opened {sorted(CLOSED_AT_P9 - closed)}, " \
                                   f"newly closed {sorted(closed - CLOSED_AT_P9)}"
    private = _with(export, header={"variant": "full"}) if export.is_public else export
    for cls in sorted(CLOSED_AT_P9):
        assert bs.effective_mode(private, "enforce", cls=cls) == "enforce", cls
    if not export.is_public:
        assert {cls: bs.effective_mode(export, cls=cls) for cls in CLOSED_AT_P9} == dict.fromkeys(CLOSED_AT_P9,
                                                                                                    "enforce")


def test_real_tree_report_enforces_a_closed_p9_class(export, monkeypatch):
    """The WIRING of the per-class mode: with one P9 class closed and a plant on it, the real-tree test
    FAILS on the private export; with the class open it reports and passes."""
    if export.is_public:
        pytest.skip("the public mirror reports every class (R23); this wiring is private-repo only")
    plant = [bs.Item("comments", "planted.py:1", "# mi" "LLM plant")]
    monkeypatch.setattr(bs, "file_items", lambda cls, root=None: (["planted.py"], plant if cls == "comments" else []))
    monkeypatch.setattr(bs, "live_items", lambda cls: [])
    monkeypatch.setitem(bs.CLASS_CLOSED, "comments", False)
    test_real_tree_report(export)
    monkeypatch.setitem(bs.CLASS_CLOSED, "comments", True)
    with pytest.raises(AssertionError, match="planted.py:1"):
        test_real_tree_report(export)


def test_a_keep_mark_keeps_its_line_and_only_its_line(tmp_path, export):
    """A keep mark (`p9` + `: keep, <reason>`) makes its own line's findings kept (reported, never failing); the next
    line's are not. Control: `_marked` returning an empty set turns this red."""
    root = _git_tree(tmp_path, {"backend/src/k.py": '# mi' 'LLM here  # p9' ': keep, a planted keep\n# mi' 'LLM there\n'})
    findings = bs.scan(bs.file_items("comments", root)[1], bs.rules_for(export))
    assert [(f.where, f.kept) for f in findings] == [("backend/src/k.py:1", True), ("backend/src/k.py:2", False)]
    lines = bs.report("plant", findings, "enforce")
    assert len(lines) == 1 and lines[0].startswith("[comments] backend/src/k.py:2:")


def test_test_text_and_test_values_split_a_test_line(tmp_path, export):
    """RG-3 reads a test's names, RG-6 its string values (P9-D6): one line, two classes, no double count."""
    root = _git_tree(tmp_path, {"backend/tests/unit/test_s.py":
                                'def test_mi' 'llm_z():\n    assert f("mi' 'LLM")\n',
                                "frontend/src/test/s.test.ts": "const miStudioX = 'mi' + 'llm';\nexpect(`a ${miLLM} b`);\n"})
    rules = bs.rules_for(export)
    names = sorted(f.where for f in bs.scan(bs.file_items("test_text", root)[1], rules))
    values = sorted(f.where for f in bs.scan(bs.file_items("test_values", root)[1], rules))
    assert names == ["backend/tests/unit/test_s.py:1", "frontend/src/test/s.test.ts:1", "frontend/src/test/s.test.ts:2"]
    assert values == ["backend/tests/unit/test_s.py:2"], values


#: The real tree's scan of every P9 class, shared by the keep-mark test (one scan, not two).
_P9_SCAN: dict[str, list[bs.Finding]] = {}


def _p9_scan(export: le.Export) -> dict[str, list[bs.Finding]]:
    if not _P9_SCAN:
        rules = bs.rules_for(export)
        for cls in bs.P9_CLASSES:
            _P9_SCAN[cls] = bs.scan(bs.file_items(cls)[1], rules)
    return _P9_SCAN


def test_keep_marks_carry_a_reason_and_keep_something(export):
    """Every keep mark (`p9` + `: keep, <reason>`) in a P9 surface carries a reason, and sits on a line where some P9 class found
    an old name it then kept: a mark that keeps nothing is stale and is removed, so marks cannot pile up
    unread (miServer 042's `test_keep_marks_keep_something` is the model)."""
    kept = {f.where for fs in _p9_scan(export).values() for f in fs if f.kept}
    files = sorted({f for cls in bs.P9_CLASSES for f in bs.file_items(cls)[0]})
    marks, bare = set(), []
    for rel in files:
        text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        if not bs.KEEP_MARK.search(text):
            continue
        for number, line in enumerate(text.split("\n"), 1):
            if bs.KEEP_MARK.search(line):
                marks.add(f"{rel}:{number}")
                if not bs.KEEP_MARK_WITH_REASON.search(line):
                    bare.append(f"{rel}:{number}")
    assert len(marks) > 100, "too few keep marks found: the scan reads nothing"
    assert not bare, f"a keep mark without its reason: {bare}"
    stale = sorted(marks - kept)
    assert not stale, f"keep marks that keep nothing (remove them): {stale[:40]}"
