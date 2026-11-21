"""Validate exported probe definitions with miServer's OWN contract model (`millm.api.schemas.probe.ProbeDefinitionV1`,
the model miServer's import path validates with). Executed by miWorkbench's test in miServer's venv, cwd = the miServer
checkout, as a subprocess: nothing of miServer is imported into miWorkbench. argv[1] is miServer's package name
(`millm`, or `miserver` after its ENH-009 P8 rename); the rest are definition file paths. Prints  (p9: keep, rename history)
{path: "ok" | the validation error}. Run without the package argument it exits 2 and never guesses."""
import importlib
import json
import sys

if len(sys.argv) < 2 or sys.argv[1] not in ("millm", "miserver"):
    sys.stderr.write("usage: miserver_definition_driver <millm|miserver> <definition file>... (042 FR-042.36)\n")
    raise SystemExit(2)
ProbeDefinitionV1 = importlib.import_module(f"{sys.argv[1]}.api.schemas.probe").ProbeDefinitionV1

out = {}
for path in sys.argv[2:]:
    try:
        ProbeDefinitionV1.model_validate(json.loads(open(path, "rb").read()))
        out[path] = "ok"
    except Exception as exc:  # noqa: BLE001 - reported per file
        out[path] = f"{type(exc).__name__}: {exc}"[:500]
print(json.dumps(out))
