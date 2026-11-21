"""Read / write CAAC-Pool v1 rows (JSONL, JSONL.gz, CSV) and meta.json.

Readers are deliberately strict about types but never guess values: a field that
cannot be parsed is reported by the validator, not silently fixed. Parsing errors
are collected as (line number, message) so a user sees every bad line at once.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
from dataclasses import asdict
from pathlib import Path

from caac.pool_v1.schema import OPTIONAL, REQUIRED, Meta, Row

_INT = ("draw", "out_tokens", "in_tokens")
_FLOAT = ("gpu_s", "cost_usd")
_TRUE = {"1", "true", "True", "TRUE", "yes"}
_FALSE = {"0", "false", "False", "FALSE", "no"}


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else \
        open(path, "r", encoding="utf-8", newline="")


def _bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.strip() in _TRUE:
        return True
    if isinstance(v, str) and v.strip() in _FALSE:
        return False
    raise ValueError(f"not a 0/1 value: {v!r}")


def _row_from_dict(d: dict) -> Row:
    missing = [k for k in REQUIRED if k not in d or d[k] in (None, "")]
    if missing:
        raise ValueError(f"missing required field(s): {', '.join(missing)}")
    kw = {}
    for k in REQUIRED + OPTIONAL:
        v = d.get(k)
        if v is None or v == "":
            continue
        if k in _INT:
            if isinstance(v, float) and not v.is_integer():
                raise ValueError(f"{k} must be an integer, got {v!r}")
            v = int(v)
        elif k in _FLOAT:
            v = float(v)
        elif k in ("correct", "is_greedy"):
            v = _bool(v)
        elif k == "flags":
            v = list(v) if isinstance(v, list) else [x for x in str(v).split(";") if x]
        else:
            v = str(v)
        kw[k] = v
    unknown = set(d) - set(REQUIRED) - set(OPTIONAL)
    row = Row(**kw)
    if unknown:  # kept visible, never dropped silently
        row.flags = row.flags + [f"unknown_field:{u}" for u in sorted(unknown)]
    return row


def read_rows(path: str | Path) -> tuple[list[Row], list[tuple[int, str]]]:
    """Return (rows, errors). errors = [(1-based line, message)]."""
    path = Path(path)
    rows: list[Row] = []
    errors: list[tuple[int, str]] = []
    name = path.name[:-3] if path.suffix == ".gz" else path.name
    with _open(path) as f:
        if name.endswith(".csv"):
            reader = csv.DictReader(f)
            for i, d in enumerate(reader, start=2):
                try:
                    rows.append(_row_from_dict(d))
                except (ValueError, TypeError) as e:
                    errors.append((i, str(e)))
        else:
            for i, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(_row_from_dict(json.loads(line)))
                except (ValueError, TypeError) as e:
                    errors.append((i, str(e)))
    return rows, errors


def write_rows(rows: list[Row], path: str | Path) -> None:
    """JSONL (optionally .gz); None fields and empty flags are omitted; key order fixed."""
    path = Path(path)
    buf = io.StringIO()
    for r in rows:
        d = {k: v for k, v in asdict(r).items() if v is not None and v != []}
        buf.write(json.dumps(d, ensure_ascii=False, sort_keys=False) + "\n")
    data = buf.getvalue().encode("utf-8")
    if path.suffix == ".gz":
        with open(path, "wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as g:
            g.write(data)   # no name, mtime=0 in the header: same rows -> same bytes
    else:
        path.write_bytes(data)


def read_meta(path: str | Path) -> tuple[Meta | None, list[str]]:
    """Return (meta, errors)."""
    errs: list[str] = []
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return None, [f"meta.json unreadable: {e}"]
    for k in ("source", "grader", "answer_equivalence"):
        if k not in d:
            errs.append(f"meta.json: missing '{k}'")
    if errs:
        return None, errs
    return Meta(source=d["source"], grader=d["grader"],
                answer_equivalence=d["answer_equivalence"],
                caps=d.get("caps", {}), prices=d.get("prices", {}),
                notes=d.get("notes", ""), answer_keys=bool(d.get("answer_keys", False)),
                format=d.get("format", ""),
                version=str(d.get("version", ""))), []


def load_pool_v1(directory: str | Path):
    """Find rows.* and meta.json in a directory. Returns (rows, row_errors, meta, meta_errors)."""
    d = Path(directory)
    cands = [d / n for n in ("rows.jsonl.gz", "rows.jsonl", "rows.csv", "rows.csv.gz")]
    found = [p for p in cands if p.exists()]
    if not found:
        return [], [(0, f"no rows.jsonl[.gz] / rows.csv in {d}")], None, []
    if len(found) > 1:
        return [], [(0, f"more than one rows file in {d}: {[p.name for p in found]}")], None, []
    rows, rerr = read_rows(found[0])
    meta, merr = read_meta(d / "meta.json") if (d / "meta.json").exists() else (None, ["meta.json missing"])
    return rows, rerr, meta, merr
