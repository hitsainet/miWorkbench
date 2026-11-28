"""The vendored `miforge.wsp-bundle/v1` schema, read-only (ENH-010 044 task 5.2; FTDD §5.7.4, §5.7.5).

Its own module so the module that WRITES bundle files never names a vendored schema file
(`test_vendored_wsp_schemas.py::test_nothing_in_src_WRITES_any_copy`): vendored, never generated.
"""
from __future__ import annotations

import functools
import json
from importlib import resources
from typing import List

SCHEMA_PACKAGE = "src.schemas.vendored"
SCHEMA_FILE = "miforge-wsp-bundle-v1.json"


@functools.cache
def load_schema() -> dict:
    return json.loads(resources.files(SCHEMA_PACKAGE).joinpath(SCHEMA_FILE).read_text(encoding="utf-8"))


@functools.cache
def load_validator():
    from jsonschema.validators import validator_for

    schema = load_schema()
    cls = validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)


def schema_errors(manifest: dict) -> List[str]:
    return [f"{'/'.join(str(p) for p in e.absolute_path) or '$'}: {e.message}"
            for e in load_validator().iter_errors(manifest)]


