"""No test writes an old- or new-name runtime variable into the PROCESS environment (042 FR-042.8, FR-042.38).

Since 042 P0, a `MILLM_API_URL`/`MISERVER_API_URL` in `os.environ` is configuration with effects: it gives the  (p9: keep, an old variable name, still read or refused by name)
judge gate a configured miServer address (so an unrelated endpoint is resolved and, unresolvable, gated), and
it puts an `env` use into the process's old-name evidence. Four tests set `MILLM_API_URL` with  (p9: keep, an old variable name, still read or refused by name)
`os.environ.setdefault(...)` (the MIS-E2E-115 shape); under `--dist loadfile` that leaked into every later test
on the worker and turned three files red in the merged P0 run. A test that needs an address passes it per
instance (`MCPSettings(miserver_api_url_override=...)`) or uses `monkeypatch.setenv`, which is undone.

Authority: the names are `src/core/legacy_names.ENV` (old and new), read from the module, never re-typed.
The scan is AST over every file under `backend/tests/` (writes through `os.environ[...] =`, `.setdefault`,
`.update`, `os.putenv`); `monkeypatch.setenv` is the allowed form and is not a write to `os.environ` here.
"""
from __future__ import annotations

import ast
from pathlib import Path

from src.core import legacy_names
from tests.support.source_tree import python_sources

TESTS = Path(__file__).resolve().parents[1]
NAMES = frozenset(n for row in legacy_names.ENV for n in (row.old, row.new))


def _str(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _is_os_environ(node: ast.AST) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "environ"
            and isinstance(node.value, ast.Name) and node.value.id in ("os", "_os"))


def writes(text: str) -> list[tuple[int, str]]:
    """`(line, name)` of each process-wide write of a NAMES variable in `text`."""
    out = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Subscript) and _is_os_environ(t.value) and _str(t.slice) in NAMES:
                    out.append((node.lineno, _str(t.slice)))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            f = node.func
            if f.attr in ("setdefault", "__setitem__") and _is_os_environ(f.value) and node.args \
                    and _str(node.args[0]) in NAMES:
                out.append((node.lineno, _str(node.args[0])))
            elif f.attr == "update" and _is_os_environ(f.value):
                for arg in node.args:
                    if isinstance(arg, ast.Dict):
                        out += [(node.lineno, _str(k)) for k in arg.keys if k is not None and _str(k) in NAMES]
                out += [(node.lineno, kw.arg) for kw in node.keywords if kw.arg in NAMES]
            elif f.attr == "putenv" and isinstance(f.value, ast.Name) and f.value.id == "os" and node.args \
                    and _str(node.args[0]) in NAMES:
                out.append((node.lineno, _str(node.args[0])))
    return out


def scan(root: Path = TESTS) -> tuple[int, list[str]]:
    """`(files read, findings)` over every `.py` under `root`. A file another xdist worker deletes between the listing
    and the read (`test_suite_reports_its_totals.py`'s `test_zz_totals_probe_<pid>.py`) is skipped
    (`tests/support/source_tree.py`); it raised `FileNotFoundError` here before (2026-11-28)."""
    sources = python_sources(root)
    return len(sources), [f"{p.relative_to(root.parent)}:{line}: {name}"
                          for p, text in sources for line, name in writes(text)]


def test_no_test_writes_a_legacy_env_name_process_wide():
    count, found = scan()
    assert count > 500 and NAMES, "the scan discovered nothing"
    assert not found, ("a test writes a runtime variable into os.environ (it leaks to every later test on the "
                       "worker); use monkeypatch.setenv or a per-instance override:\n" + "\n".join(found))


def test_the_scanner_finds_each_write_form():
    old = sorted(NAMES)[0]
    plants = [f"import os\nos.environ.setdefault({old!r}, 'x')\n",
              f"import os\nos.environ[{old!r}] = 'x'\n",
              f"import os\nos.environ.update({{{old!r}: 'x'}})\n",
              f"import os\nos.environ.update({old}='x')\n",
              f"import os\nos.putenv({old!r}, 'x')\n",
              f"import os as _o\nimport os\nos.environ.setdefault({old!r}, 'x')\n"]
    for src in plants:
        assert writes(src) == [(2 if src.count("\n") == 2 else 3, old)], src
    allowed = [f"def t(monkeypatch):\n    monkeypatch.setenv({old!r}, 'x')\n",
               f"import os\nv = os.environ.get({old!r})\n",
               "import os\nos.environ.setdefault('UNRELATED_NAME', 'x')\n"]
    for src in allowed:
        assert writes(src) == [], src


def test_a_file_deleted_after_the_listing_is_skipped_and_the_rest_still_scanned(tmp_path, monkeypatch):
    """The xdist race (2026-11-28): a file listed, then deleted by another worker before it is read. Simulated by a
    listed file whose read raises `FileNotFoundError`; the scan goes on and still finds the planted write."""
    old = sorted(NAMES)[0]
    tests = tmp_path / "tests"
    (tests / "unit").mkdir(parents=True)
    (tests / "unit" / "test_planted.py").write_text(f"import os\nos.environ.setdefault({old!r}, 'x')\n")
    gone = tests / "unit" / "test_zz_totals_probe_1.py"
    gone.write_text("")
    real_read = Path.read_text

    def read_text(self, *args, **kwargs):
        if self == gone:
            raise FileNotFoundError(2, "No such file or directory", str(self))
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    count, found = scan(tests)
    assert count == 1
    assert found == [f"tests/unit/test_planted.py:2: {old}"]
