#!/usr/bin/env python3
"""C1 step-4 (joint提速改造 2026-10-17): canonical-equivalence comparator.

Decision ① formalized: after the Sys calendar surgery, same-tick cross-rank
delivery order is not a byte-identity contract anymore.  The canonical gate
is: (a) double fresh-process runs of the SAME binary must be identical
modulo *wall_ns* fields, and (b) against a reference produced by a different
build, order-sensitive audit arrays are compared under a stable sort within
their (tick, id) family keys before diffing.

Normalized surfaces (the §10.3 enumeration):
  - decision rows: "events" / "completed_groups" / "event_seqs" /
    "completed_event_seqs" style arrays are stably sorted by (tick, id-ish)
    keys; everything else must match exactly (wall_ns exempt).
  - train_ledger.jsonl / work_ledger_events.jsonl rows: any array field
    named "events" or "node_transitions" is stably sorted by its (tick, id)
    prefix keys.

Usage:
  joint_canonical_compare.py RUN_A RUN_B [--strict]
    RUN_A/RUN_B are run directories containing either bridge/
    online_decision_log.jsonl (harvested shape) or results/
    online_decision_log.jsonl (natural-end shape); ledger files are picked
    up from the same locations when present.
  --strict: no normalization (pure wall_ns-exempt deep diff).

Exit 0 = PASS, 2 = differences, 1 = usage/IO error.
"""
import json
import sys
from pathlib import Path

WALL_NS_MARK = "<WALL_NS>"
ORDER_SENSITIVE_KEYS = (
    "events",
    "completed_groups",
    "event_seqs",
    "completed_event_seqs",
    "node_transitions",
)
LEDGER_ARRAY_KEYS = ("events", "node_transitions")


def _sortable(item):
    return json.dumps(item, sort_keys=True)


def _family_key(item):
    """(tick, id)-family sort key: numeric fields named tick*/seq*/id* first,
    then the canonical text as tiebreaker (stable for equal family keys)."""
    if not isinstance(item, dict):
        return (0, 0, _sortable(item))
    tick = None
    ident = None
    for key in ("tick", "tick_ns", "sim_tick", "delivery_tick", "event_tick"):
        value = item.get(key)
        if isinstance(value, (int, float)):
            tick = value
            break
    for key in ("id", "event_id", "node_id", "seq", "event_seq", "request_id",
                "generation", "stage"):
        value = item.get(key)
        if isinstance(value, (int, float)):
            ident = value
            break
        if isinstance(value, str):
            ident = value
            break
    tick = tick if isinstance(tick, (int, float)) else 0
    ident = ident if ident is not None else 0
    if not isinstance(ident, (int, float)):
        ident = str(ident)
    return (tick, ident, _sortable(item))


def normalize(obj, normalize_arrays):
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if "wall_ns" in key:
                out[key] = WALL_NS_MARK
            elif (normalize_arrays and isinstance(value, list)
                    and key in ORDER_SENSITIVE_KEYS
                    and all(isinstance(x, dict) for x in value)):
                out[key] = sorted(
                    (normalize(x, normalize_arrays) for x in value),
                    key=_family_key)
            else:
                out[key] = normalize(value, normalize_arrays)
        return out
    if isinstance(obj, list):
        return [normalize(x, normalize_arrays) for x in obj]
    return obj


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compare_rows(name, rows_a, rows_b, normalize_arrays, report):
    ok = True
    if len(rows_a) != len(rows_b):
        # Common-prefix comparison: a harvested run ends mid-stream.
        common = min(len(rows_a), len(rows_b))
        report.append(
            f"[{name}] row count {len(rows_a)} vs {len(rows_b)}; "
            f"comparing common prefix of {common} (harvest-depth difference)")
        if common == 0:
            return False
        rows_a, rows_b = rows_a[:common], rows_b[:common]
    for index, (raw_a, raw_b) in enumerate(zip(rows_a, rows_b)):
        a = normalize(raw_a, normalize_arrays)
        b = normalize(raw_b, normalize_arrays)
        if _sortable(a) != _sortable(b):
            report.append(f"[{name}] row {index} DIFFERS")
            ok = False
    return ok


def ledger_path(run_dir):
    for candidate in (
            Path(run_dir) / "results" / "train_ledger.jsonl",
            Path(run_dir) / "bridge" / "train_ledger.jsonl",
            Path(run_dir) / "results" / "work_ledger_events.jsonl",
            Path(run_dir) / "bridge" / "work_ledger_events.jsonl"):
        if candidate.exists():
            yield candidate


def decision_path(run_dir):
    for candidate in (Path(run_dir) / "results" / "online_decision_log.jsonl",
                      Path(run_dir) / "bridge" /
                      "online_decision_log.jsonl"):
        if candidate.exists():
            return candidate
    return None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    strict = "--strict" in sys.argv
    if len(args) != 2:
        print(__doc__)
        sys.exit(1)
    run_a, run_b = args
    report = []
    ok = True
    dec_a, dec_b = decision_path(run_a), decision_path(run_b)
    if dec_a is None or dec_b is None:
        print(f"FATAL: decision log missing: {dec_a} / {dec_b}")
        sys.exit(1)
    ok &= compare_rows("decision_log", load_rows(dec_a), load_rows(dec_b),
                       not strict, report)
    ledgers_a = {p.name: p for p in ledger_path(run_a)}
    for name, path_b in ((p.name, p) for p in ledger_path(run_b)):
        if name in ledgers_a:
            ok &= compare_rows(name, load_rows(ledgers_a[name]),
                               load_rows(path_b), not strict, report)
    for line in report:
        print(line)
    verdict = "PASS" if ok else "DIFF"
    mode = "strict" if strict else "normalized"
    print(f"RESULT: {verdict} (mode={mode})")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
