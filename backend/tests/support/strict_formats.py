"""A format checker that refuses what the production MCP client refuses (ajv with ajv-formats, `full` mode).

`date-time` and `time` need an RFC 3339 offset (ajv-formats 3's strictTimeZone default); `date` and `uuid` are checked;
any other format is unknown and FAILS (fail closed: a format this checker cannot judge could still fail in the client).
The answer middleware validates every covered answer with it (046 P7 finding, 2026-11-26), so a contract that promises
an offset the route never sends is red here before a client refuses it in production.
"""
from __future__ import annotations

import calendar
import re

import jsonschema

_DATE = re.compile(r"^(\d\d\d\d)-(\d\d)-(\d\d)$")
_TIME = re.compile(r"^(\d\d):(\d\d):(\d\d(?:\.\d+)?)(z|[+-]\d\d(?::?\d\d)?)$", re.IGNORECASE)
_UUID = re.compile(r"^(?:urn:uuid:)?[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$", re.IGNORECASE)


def _date(value: str) -> bool:
    """A calendar date, as ajv-formats' full mode reads one: the day must exist in its month (leap years included;
    046-2 review round 1: every month allowed 31 days)."""
    found = _DATE.match(value)
    if not found or not 1 <= int(found[2]) <= 12:
        return False
    return 1 <= int(found[3]) <= calendar.monthrange(int(found[1]), int(found[2]))[1]


def _time(value: str) -> bool:
    found = _TIME.match(value)
    return bool(found) and int(found[1]) <= 23 and int(found[2]) <= 59 and float(found[3]) < 61


def _date_time(value: str) -> bool:
    parts = re.split(r"t|\s", value, maxsplit=1, flags=re.IGNORECASE)
    return len(parts) == 2 and _date(parts[0]) and _time(parts[1])


CHECKS = {"date-time": _date_time, "time": _time, "date": _date, "uuid": lambda v: bool(_UUID.match(v))}


class _FailClosed(jsonschema.FormatChecker):
    def check(self, instance, format):  # noqa: A002 - jsonschema's own parameter name
        if format not in self.checkers:
            raise jsonschema.FormatError(f"unknown format {format!r}: the production client may refuse it")
        return super().check(instance, format)


STRICT = _FailClosed(formats=())
for _name, _check in CHECKS.items():
    STRICT.checks(_name)(lambda v, _c=_check: not isinstance(v, str) or _c(v))
