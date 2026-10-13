#!/usr/bin/env python3
"""Independent exact LP check for the C++ grouped Q/alpha allocator probe."""

from __future__ import annotations

import json
import shlex
import sys
from dataclasses import dataclass
from fractions import Fraction as F
from pathlib import Path


CAPACITY = {"COMPUTE_GATE": F(1), "HBM0": F(50_000_000), "HBM1": F(50_000_000)}
RESOURCES = tuple(CAPACITY)


@dataclass(frozen=True)
class Activity:
    owner: int
    stage: str
    q: F
    work: dict[str, F]
    private_time: F
    cpp_rate: F
    local_bytes: int

    @property
    def key(self) -> str:
        return f"{self.owner}.{self.stage}"


def fields(line: str) -> dict[str, str]:
    return dict(field.split("=", 1) for field in shlex.split(line)[1:])


def parse_cases(path: Path) -> dict[str, tuple[dict[str, str], list[Activity]]]:
    headers: dict[str, dict[str, str]] = {}
    activities: dict[str, list[Activity]] = {}
    for line in path.read_text().splitlines():
        if line.startswith("lp-case "):
            candidate = fields(line)
            name = candidate.get("name", "")
            if name in ("three_stage_four_active_q1", "three_stage_four_active_qlocal"):
                headers[name] = candidate
                activities[name] = []
        elif line.startswith("lp-activity "):
            row = fields(line)
            name = row.get("case", "")
            if name not in activities:
                continue
            activities[name].append(Activity(
                owner=int(row["owner"]), stage=row["stage"], q=F(row["q"]),
                work={"COMPUTE_GATE": F(row["compute"]),
                      "HBM0": F(row["hbm0"]), "HBM1": F(row["hbm1"])},
                private_time=F(row["private"]), cpp_rate=F(row["rate"]),
                local_bytes=int(row["local"])))
    required = {"three_stage_four_active_q1", "three_stage_four_active_qlocal"}
    if set(headers) != required:
        raise AssertionError(f"missing three-stage LP cases: {set(headers)}")
    return {name: (headers[name], activities[name]) for name in sorted(required)}


def solve_lp(activities: list[Activity]) -> tuple[dict[str, F], list[dict[str, object]]]:
    """Solve exact weighted max-min LPs tier by tier.

    At each tier, frozen rates remain fixed. For each free activity i,
    y_i/alpha_owner is the normalized objective, so the LP's common floor z
    imposes rate_i >= z / Q_owner. Eliminating rates leaves exact constraints
    slope_r*z <= residual_r and one private-rate upper bound per compute
    activity. A binding constraint gives a matching dual multiplier.
    """
    owner_q: dict[int, F] = {}
    for a in activities:
        owner_q[a.owner] = owner_q.get(a.owner, F(0)) + a.q
    if not activities or any(q <= 0 for q in owner_q.values()):
        raise AssertionError("invalid LP activity or owner weight")

    rates = [F(0) for _ in activities]
    free = set(range(len(activities)))
    certs: list[dict[str, object]] = []
    while free:
        residual = {
            resource: CAPACITY[resource] - sum(
                activities[i].work[resource] * rates[i] for i in range(len(activities)))
            for resource in RESOURCES
        }
        slopes = {
            resource: sum(activities[i].work[resource] / owner_q[activities[i].owner]
                          for i in free)
            for resource in RESOURCES
        }
        bounds: dict[str, F] = {
            resource: residual[resource] / slopes[resource]
            for resource in RESOURCES if slopes[resource] > 0
        }
        for i in free:
            a = activities[i]
            if a.private_time > 0:
                bounds[f"PRIVATE:{i}"] = (F(1, 1) / a.private_time - rates[i]) * owner_q[a.owner]
        if not bounds:
            raise AssertionError("LP tier has no finite resource/private constraint")
        increment = min(bounds.values())
        if increment < 0:
            raise AssertionError("LP tier became infeasible")
        binding = tuple(sorted(k for k, value in bounds.items() if value == increment))

        primal = list(rates)
        for i in free:
            primal[i] += increment / owner_q[activities[i].owner]
        for resource in RESOURCES:
            used = sum(activities[i].work[resource] * primal[i]
                       for i in range(len(activities)))
            if used > CAPACITY[resource]:
                raise AssertionError(f"LP primal violates {resource}: {used}")
        for i, a in enumerate(activities):
            if a.private_time > 0 and primal[i] > F(1, 1) / a.private_time:
                raise AssertionError(f"LP primal violates private cap for {a.key}")

        # One exact dual witness for the optimal scalar floor.
        dual: dict[str, str] = {}
        first_resource = next((r for r in RESOURCES if r in binding), None)
        if first_resource is not None:
            lam = F(1, 1) / slopes[first_resource]
            if lam * residual[first_resource] != increment:
                raise AssertionError("resource dual does not equal primal optimum")
            dual = {"constraint": first_resource, "lambda": str(lam)}
        else:
            private_index = int(binding[0].split(":", 1)[1])
            dual = {"constraint": f"PRIVATE:{activities[private_index].key}",
                    "lambda": "1"}

        freeze: set[int] = set()
        for item in binding:
            if item.startswith("PRIVATE:"):
                freeze.add(int(item.split(":", 1)[1]))
            else:
                resource = item
                freeze.update(i for i in free if activities[i].work[resource] > 0)
        if not freeze:
            raise AssertionError("binding LP constraint freezes no free activity")
        certs.append({
            "increment": str(increment),
            "free": sorted(activities[i].key for i in free),
            "binding": list(binding),
            "freeze": sorted(activities[i].key for i in freeze),
            "residual": {r: str(v) for r, v in residual.items()},
            "slope": {r: str(v) for r, v in slopes.items()},
            "dual": dual,
        })
        rates = primal
        free.difference_update(freeze)

    return {activities[i].key: rates[i] for i in range(len(activities))}, certs


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_grouped_lp.py CPP_OUTPUT")
    all_cases = parse_cases(Path(sys.argv[1]))
    results: dict[str, object] = {}
    for name, (header, activities) in all_cases.items():
        if int(header["active"]) != 4 or len(activities) != 4:
            raise AssertionError(f"{name}: probe must contain exactly four live activities")
        stages = {a.stage for a in activities}
        if stages != {"L", "C", "S"}:
            raise AssertionError(f"{name}: probe omitted a lifecycle stage: {stages}")
        per_owner: dict[int, int] = {}
        for activity in activities:
            per_owner[activity.owner] = per_owner.get(activity.owner, 0) + 1
        if max(per_owner.values()) > 3:
            raise AssertionError(f"{name}: probe exceeds three active activities per owner")
        for activity in activities:
            if activity.stage == "C":
                expected_q = F(1) if name.endswith("_q1") else F(activity.local_bytes)
                if activity.q != expected_q:
                    raise AssertionError(f"{name}/{activity.key}: unexpected compute canonical Q {activity.q}")
            else:
                expected_q = activity.work["HBM0"] + activity.work["HBM1"]
                if activity.q != expected_q or activity.local_bytes != 0:
                    raise AssertionError(f"{name}/{activity.key}: unexpected HBM canonical Q or local ledger")

        rates, certificates = solve_lp(activities)
        cpp_rates = {a.key: a.cpp_rate for a in activities}
        if rates != cpp_rates:
            raise AssertionError(f"{name}: C++ grouped rates differ from exact LP: {cpp_rates} != {rates}")
        basis = ">".join("+".join(r for r in RESOURCES if r in tier["binding"]) +
                         ("+PRIVATE" if any(str(x).startswith("PRIVATE:") for x in tier["binding"])
                          else "") for tier in certificates)
        if basis != header["basis"]:
            raise AssertionError(f"{name}: C++/LP tier basis differs: {header['basis']} != {basis}")
        results[name] = {
            "activities": [{
                "key": a.key, "stage": a.stage, "q": str(a.q),
                "work": {r: str(a.work[r]) for r in RESOURCES},
                "local_bytes_ledger": a.local_bytes,
                "private_time": str(a.private_time), "cpp_rate": str(a.cpp_rate),
                "lp_rate": str(rates[a.key]),
            } for a in activities],
            "owner_q": {str(owner): str(sum(a.q for a in activities if a.owner == owner))
                        for owner in sorted(per_owner)},
            "basis": basis,
            "exact_lp_certificates": certificates,
        }
    print(json.dumps({
        "status": "passed_exact_lp_probes_only",
        "scope": "two four-activity states from the three-stage lifecycle; q1 is a pure-compute toy, qlocal is GEMM canonical-Q with LOCAL ledger-only and nonbinding; no production allocator certification",
        "cases": results,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
