"""Frontend CI type-checks every TypeScript project, the referenced ones included (042 acceptance 2026-11-26, F-3).

`frontend/tsconfig.json` references `tsconfig.node.json`, the project that holds `vite.config.ts`. CI ran
`npm run type-check` = `tsc --noEmit`, which checks the referencing project alone and never builds a reference, so
`npx tsc -b --noEmit` (042 FTASKS 20.3) showed six errors in `vite.config.ts` and its project file that no CI job had
ever reported. The fix runs the type check in build mode, and this file pins the WIRING from the workflow step down
to the file the old command never read:

* the `Type check` step of `frontend-ci.yml` runs the `type-check` script;
* that script is `tsc` in build mode (`-b`/`--build`), which checks each referenced project;
* every top-level TypeScript file of the frontend belongs to a project the build reaches;
* no `*.tsbuildinfo` is tracked: build mode writes one per project, and a tracked one dirties the tree on every run.
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
from fnmatch import fnmatch
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
FRONTEND = REPO / "frontend"


def jsonc(text: str) -> dict:
    """A tsconfig file: JSON with `//` and `/* */` comments and trailing commas. Strings are copied whole, so a
    `/*` inside one (`"@/*"`) is not read as a comment."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        elif text.startswith("//", i):
            i = text.find("\n", i) if "\n" in text[i:] else n
        elif text.startswith("/*", i):
            i = text.index("*/", i) + 2
        else:
            out.append(c)
            i += 1
    return json.loads(re.sub(r",(\s*[}\]])", r"\1", "".join(out)))  # trailing commas


def projects(entry: Path = FRONTEND / "tsconfig.json") -> list[tuple[Path, dict]]:
    """The project `tsc -b` starts from and every project it reaches by `references`."""
    seen: list[tuple[Path, dict]] = []
    todo = [entry]
    while todo:
        path = todo.pop(0).resolve()
        if path.is_dir():
            path = path / "tsconfig.json"
        if any(p == path for p, _ in seen):
            continue
        config = jsonc(path.read_text(encoding="utf-8"))
        seen.append((path, config))
        todo += [path.parent / ref["path"] for ref in config.get("references", [])]
    return seen


def _tracked(*globs: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", "--", *globs], cwd=REPO, capture_output=True, check=True).stdout
    return sorted(p for p in out.decode().split("\0") if p)


def _type_check_command() -> list[str]:
    workflow = yaml.safe_load((REPO / ".github" / "workflows" / "frontend-ci.yml").read_text())
    steps = [s for job in workflow["jobs"].values() for s in job.get("steps", []) if s.get("name") == "Type check"]
    assert len(steps) == 1, f"expected one 'Type check' step in frontend-ci.yml, found {len(steps)}"
    step = steps[0]
    assert step.get("working-directory") == "frontend", step
    assert shlex.split(step["run"]) == ["npm", "run", "type-check"], step["run"]
    script = json.loads((FRONTEND / "package.json").read_text())["scripts"]["type-check"]
    return shlex.split(script)


def test_ci_type_check_is_tsc_build_mode():
    words = _type_check_command()
    assert words[0] == "tsc", words
    assert "-b" in words or "--build" in words, f"the CI type check is not build mode, so references are never " \
                                                 f"checked: {' '.join(words)}"


def test_every_top_level_typescript_file_is_in_a_project_the_build_reaches():
    reached = projects()
    assert len(reached) >= 2, "tsconfig.json references no project: build mode would check one project only"
    files = [p for p in _tracked("frontend/*.ts", "frontend/*.tsx", "frontend/*.mts") if "/" not in p[len("frontend/"):]]
    assert "frontend/vite.config.ts" in files, files
    for rel in files:
        name = Path(rel).name
        owners = [path.name for path, config in reached
                  if path.parent == FRONTEND and any(fnmatch(name, g) for g in config.get("include", []))]
        assert owners, f"{rel} belongs to no project `tsc -b` checks"


def test_no_build_info_file_is_tracked():
    assert _tracked("*.tsbuildinfo", "**/*.tsbuildinfo") == []


def test_the_reader_handles_the_comment_shapes_tsconfig_uses():
    text = '{\n  // a line comment\n  "a": "@/*", /* block */\n  "b": ["x/*", "y",],\n}\n'
    assert jsonc(text) == {"a": "@/*", "b": ["x/*", "y"]}
