#!/usr/bin/env python3
"""Recheck saved actual Runtime allocations with the independent grouped oracle.

No Runtime executable or compiler certificate is produced by this checker.
The eight pure-compute cases and four canonical-Q cases retain their separate
original raw outputs; only this comparison report is written.
"""
import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "die/validation/grouped_gemm_oracle.py").is_file())
sys.path.insert(0, str(ROOT / "die/validation"))
import grouped_gemm_oracle as oracle
import compiled_gemm_oracle as reference


def number(value):
    return float(Fraction(str(value)))


def verify(name, definitions, observed, capacities):
    active, quantities = [], []
    for key, definition in definitions.items():
        active.append(SimpleNamespace(request=key[0], work=definition["work"],
                                      private_time_s=definition["private"]))
        quantities.append(definition["q"])
    expected = oracle.grouped_rates(active, capacities, quantities)
    owner_q = {}
    for (key, definition), rate in zip(definitions.items(), expected):
        if rate > 0:
            owner_q[key[0]] = owner_q.get(key[0], 0) + definition["q"]
    for (key, definition), rate in zip(definitions.items(), expected):
        actual_rate, actual_alpha = observed[key]
        alpha = definition["q"] / owner_q[key[0]] if rate > 0 else 0
        if not (math.isclose(actual_rate, rate, rel_tol=1e-9, abs_tol=1e-12)
                and math.isclose(actual_alpha, alpha, rel_tol=1e-9, abs_tol=1e-12)):
            raise AssertionError((name, key, (actual_rate, actual_alpha), (rate, alpha)))
    # LOCAL/SRAM canonical-Q fixtures also carry nonbinding physical work;
    # confirm all thirteen resource inequalities, not just the finite three.
    for r, capacity in enumerate(capacities):
        used = math.fsum(a.work[r] * rate for a, rate in zip(active, expected))
        if used > capacity * (1 + 1e-9):
            raise AssertionError((name, reference.RESOURCE_NAMES[r], used, capacity))
    return {"scenario": name, "activity_count": len(active), "rate_and_alpha": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eight", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--four", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    actual_eight = {row["case"]: row for row in
                    map(json.loads, args.eight.read_text().splitlines())}
    results = []
    for case in json.loads(args.cases.read_text())["cases"]:
        definitions = {}
        for a in case["activities"]:
            work = [0.] * 13
            work[0], work[4], work[5] = (number(a.get(k, 0))
                                        for k in ("compute", "row0", "row1"))
            definitions[(a["owner"], a["id"])] = {
                "work": work, "private": number(a.get("private_time", 0)),
                "q": 1. if a["stage"] == "C" else work[4] + work[5]}
        capacities = [1e30] * 13
        for r, cap in case["capacity"].items():
            capacities[reference.RESOURCE_IDS[r]] = cap
        observed = {(a["owner"], a["id"]): (a["rate"], a["alpha"])
                    for a in actual_eight[case["name"]]["activities"]}
        results.append(verify(case["name"], definitions, observed, capacities))
    for case in json.loads(args.four.read_text())["scenarios"]:
        definitions, observed = {}, {}
        for owner in case["owner_definitions"]:
            for a in owner["activities"]:
                work = [a["resource_work"].get(r, 0) for r in reference.RESOURCE_NAMES]
                definitions[(owner["node_id"], a["activity_id"])] = {
                    "work": work, "private": a["private_time_s"], "q": a["canonical_q"]}
        for owner in case["actual_query_progress_rates"]:
            for a in owner["activities"]:
                observed[(owner["node_id"], a["activity_id"])] = (a["actual_rate"], a["alpha"])
        capacities = [case["current_runtime_capacities"][r] for r in reference.RESOURCE_NAMES]
        results.append(verify(case["scenario"], definitions, observed, capacities))
    if len(results) != 12:
        raise AssertionError(f"expected twelve actual Runtime cases, got {len(results)}")
    args.output.write_text(json.dumps({"status": "passed", "cases": 12, "results": results,
        "scope": "saved synthetic Runtime rate/alpha queries; no signed credits or compiled lifecycle certificate"},
        indent=2) + "\n")


if __name__ == "__main__":
    main()
