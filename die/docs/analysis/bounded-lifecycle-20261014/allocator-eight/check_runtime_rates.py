#!/usr/bin/env python3
"""Independent Fraction LP check for real Runtime synthetic rate snapshots."""

from __future__ import annotations

import json
import math
import sys
from fractions import Fraction as F
from pathlib import Path


CAP_KEYS = ("COMPUTE_GATE", "HBM0", "HBM1")


def exact(value: object) -> F:
    return F(str(value))


def load_records(path: Path) -> dict[str, dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    out = {str(row["case"]): row for row in rows}
    if len(out) != len(rows):
        raise AssertionError("duplicate Runtime case output")
    return out


def describe(case: dict[str, object]) -> tuple[dict[str, F], list[dict[str, object]]]:
    cap_in = case["capacity"]
    caps = {resource: exact(cap_in[resource]) for resource in CAP_KEYS}
    activities: list[dict[str, object]] = []
    for raw in case["activities"]:
        stage = str(raw["stage"])
        if stage == "C":
            work = {"COMPUTE_GATE": exact(raw["compute"]), "HBM0": F(0), "HBM1": F(0)}
            q = F(1)
            private = exact(raw["private_time"])
        else:
            work = {"COMPUTE_GATE": F(0), "HBM0": exact(raw["row0"]),
                    "HBM1": exact(raw["row1"])}
            q = work["HBM0"] + work["HBM1"]
            private = F(0)
        if q <= 0:
            raise AssertionError("synthetic physical Q must be positive")
        activities.append({"owner": int(raw["owner"]), "id": int(raw["id"]),
                           "stage": stage, "q": q, "work": work,
                           "private": private})
    return caps, activities


def solve_exact(caps: dict[str, F], acts: list[dict[str, object]]) -> tuple[list[F], list[dict[str, object]], dict[int, F]]:
    """Solve the same grouped max-min tiers as Runtime using exact rationals.

    If a zero-rate tier excludes activities, Runtime recomputes owner Q for the
    remaining set. These fixtures intentionally use positive rates for all
    resource domains, but the restart is implemented for complete semantics.
    """
    rates = [F(0) for _ in acts]
    excluded: set[int] = set()
    certificates: list[dict[str, object]] = []
    owner_q: dict[int, F] = {}
    while True:
        owner_q = {}
        for i, activity in enumerate(acts):
            if i in excluded:
                continue
            owner = int(activity["owner"])
            owner_q[owner] = owner_q.get(owner, F(0)) + activity["q"]
        if not owner_q:
            break
        free = set(range(len(acts))) - excluded
        restart = False
        while free:
            residual = {
                resource: caps[resource] - sum(
                    acts[i]["work"][resource] * rates[i] for i in range(len(acts)))
                for resource in CAP_KEYS
            }
            slopes = {
                resource: sum(
                    acts[i]["work"][resource] / owner_q[int(acts[i]["owner"])]
                    for i in free)
                for resource in CAP_KEYS
            }
            bounds: dict[str, F] = {
                resource: residual[resource] / slopes[resource]
                for resource in CAP_KEYS if slopes[resource] > 0
            }
            for i in free:
                private = acts[i]["private"]
                if private > 0:
                    bounds[f"PRIVATE:{i}"] = (
                        F(1) / private - rates[i]) * owner_q[int(acts[i]["owner"])]
            if not bounds:
                raise AssertionError("tier has no finite resource/private bound")
            increment = min(bounds.values())
            if increment < 0:
                raise AssertionError("tier became infeasible")
            binding = sorted(key for key, value in bounds.items() if value == increment)
            primal = list(rates)
            for i in free:
                primal[i] += increment / owner_q[int(acts[i]["owner"])]
            for resource in CAP_KEYS:
                used = sum(acts[i]["work"][resource] * primal[i]
                           for i in range(len(acts)))
                if used > caps[resource]:
                    raise AssertionError(f"primal violates {resource}: {used}>{caps[resource]}")
            for i, activity in enumerate(acts):
                if activity["private"] > 0 and primal[i] > F(1) / activity["private"]:
                    raise AssertionError(f"private cap violated by id={activity['id']}")

            freeze: set[int] = set()
            for item in binding:
                if item.startswith("PRIVATE:"):
                    freeze.add(int(item.split(":", 1)[1]))
                else:
                    resource = item
                    freeze.update(i for i in free if acts[i]["work"][resource] > 0)
            if not freeze:
                raise AssertionError(f"binding tier freezes no activity: {binding}")

            resource = next((key for key in binding if not key.startswith("PRIVATE:")), None)
            if resource is not None:
                dual_lambda = F(1) / slopes[resource]
                dual_objective = dual_lambda * residual[resource]
            else:
                dual_lambda = F(1)
                dual_objective = increment
            if dual_objective != increment:
                raise AssertionError("exact LP primal/dual scalar bounds differ")
            certificates.append({
                "increment": str(increment), "binding": binding,
                "free": [acts[i]["id"] for i in sorted(free)],
                "freeze": [acts[i]["id"] for i in sorted(freeze)],
                "residual": {k: str(v) for k, v in residual.items()},
                "slope": {k: str(v) for k, v in slopes.items()},
                "dual_constraint": resource or binding[0],
                "dual_lambda": str(dual_lambda),
                "primal_dual_objective": str(increment),
            })
            rates = primal
            if increment == 0:
                excluded.update(freeze)
                restart = True
                break
            free.difference_update(freeze)
        if not restart:
            break
    return rates, certificates, owner_q


def check_close(label: str, actual: float, expected: F) -> None:
    target = float(expected)
    if not math.isclose(actual, target, rel_tol=2.0e-10, abs_tol=2.0e-12):
        raise AssertionError(f"{label}: Runtime={actual:.17g}, exact LP={expected}")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: check_runtime_rates.py CASES_JSON RUNTIME_RATES_JSONL")
    case_path, runtime_path = map(Path, sys.argv[1:])
    cases = json.loads(case_path.read_text())["cases"]
    records = load_records(runtime_path)
    if set(records) != {row["name"] for row in cases}:
        raise AssertionError("Runtime output case set differs from fixture case set")

    reports = []
    for case in cases:
        name = case["name"]
        caps, activities = describe(case)
        exact_rates, certs, owner_q = solve_exact(caps, activities)
        actuals = records[name]["activities"]
        actual_by_id = {int(row["id"]): row for row in actuals}
        if set(actual_by_id) != {int(a["id"]) for a in activities}:
            raise AssertionError(f"{name}: active IDs differ")
        rates_out = []
        for i, activity in enumerate(activities):
            raw = actual_by_id[int(activity["id"])]
            actual_q = exact(raw["q"])
            expected_alpha = activity["q"] / owner_q[int(activity["owner"])]
            check_close(f"{name}/{activity['id']}/q", float(raw["q"]), activity["q"])
            check_close(f"{name}/{activity['id']}/alpha", float(raw["alpha"]), expected_alpha)
            check_close(f"{name}/{activity['id']}/rate", float(raw["rate"]), exact_rates[i])
            if raw["status"] not in ("running", "blocked"):
                raise AssertionError(f"{name}/{activity['id']}: unexpected status {raw['status']}")
            rates_out.append({"id": activity["id"], "owner": activity["owner"],
                              "stage": activity["stage"], "q": str(activity["q"]),
                              "alpha": str(expected_alpha), "rate": str(exact_rates[i]),
                              "runtime_rate": raw["rate"], "runtime_alpha": raw["alpha"]})
        reports.append({"case": name, "status": "exact_lp_matches_runtime",
                         "owner_q": {str(k): str(v) for k, v in sorted(owner_q.items())},
                         "basis": [cert["binding"] for cert in certs],
                         "activities": rates_out, "tier_dual_witnesses": certs})
    print(json.dumps({
        "status": "passed_exact_fraction_lp_vs_runtime_probe_only",
        "scope": "8 synthetic simultaneous Runtime allocations; physical L/S Q=HBM payload bytes, C Q=1; no production oracle certification",
        "cases": reports,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
