#!/usr/bin/env python3
"""Exact-rational grouped max-min K-full stream experiment.

Scratch only. This is an independent small model of the design's Q/alpha
rule, with an exact rational LP primal/dual oracle. It models two owners, each
with at most Load+Compute active activities, and no K-final store in this
non-final K-full category. It is not a production Runtime implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction as F
from typing import Optional

PAYLOAD = F(2)
LOAD_HBM_WORK = PAYLOAD
COMPUTE_GATE_WORK = F(10)
LOCAL_PAYLOAD = F(8)
CAP = {"HBM0": F(1), "COMPUTE_GATE": F(1), "LOCAL": F(1000)}
MAX_PROBES = 256
MAX_WAVES = 10**12


@dataclass(frozen=True)
class Activity:
    owner: str
    name: str
    q: F
    work: tuple[tuple[str, F], ...]

    def work_at(self, resource: str) -> F:
        for key, value in self.work:
            if key == resource:
                return value
        return F(0)


@dataclass
class Owner:
    name: str
    cprog: Optional[F] = None
    lprog: Optional[F] = None
    ready_inputs: int = 0
    completed: int = 0
    started: bool = False


@dataclass
class World:
    q_compute: F
    local_work: F
    now: F = F(0)
    owners: dict[str, Owner] = field(default_factory=dict)
    stats: dict[str, F] = field(default_factory=lambda: {
        "hbm_busy": F(0), "hbm_shared": F(0), "compute_busy": F(0),
        "hbm_work": F(0), "compute_work": F(0), "local_work": F(0),
    })
    event_intervals: int = 0
    lp_oracle_checks: int = 0
    skipped_periods: int = 0
    capacity_generation: int = 0
    detected: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)


def active_activities(world: World) -> list[Activity]:
    rows: list[Activity] = []
    for name in sorted(world.owners):
        owner = world.owners[name]
        if owner.lprog is not None:
            rows.append(Activity(name, "L", PAYLOAD,
                                 (("HBM0", LOAD_HBM_WORK),)))
        if owner.cprog is not None:
            rows.append(Activity(
                name, "C", world.q_compute,
                (("COMPUTE_GATE", COMPUTE_GATE_WORK),
                 ("LOCAL", world.local_work) if world.local_work else ("LOCAL", F(0)))))
    if len(rows) > 4:
        raise RuntimeError("fixture exceeded the four-activity bound")
    return rows


def grouped_rates(world: World) -> tuple[dict[tuple[str, str], F], dict]:
    """Fill exact y/alpha tiers, holding alpha fixed until next event.

    At allocation entry, alpha_i=Q_i/sum_owner(Q). Let owner_q be that sum.
    The normalized objective y_i/alpha_i equals owner_q * progress_rate_i.
    All free activities rise at a common normalized level z. When a resource
    saturates, activities using it freeze at their positive current rates;
    other activities continue at higher z without renormalizing owner weight.
    """
    acts = active_activities(world)
    if not acts:
        return {}, {"basis": (), "owner_q": {}}
    owner_q: dict[str, F] = {}
    for act in acts:
        owner_q[act.owner] = owner_q.get(act.owner, F(0)) + act.q
    if any(value <= 0 for value in owner_q.values()):
        raise RuntimeError("invalid owner canonical weight")

    free = set(range(len(acts)))
    frozen: set[int] = set()
    rates: dict[int, F] = {}
    level = F(0)
    basis: list[tuple[str, tuple[int, ...]]] = []
    resources = tuple(CAP)

    while free:
        candidates: dict[str, F] = {}
        for resource in resources:
            used_frozen = sum(acts[i].work_at(resource) * rates[i] for i in frozen)
            used_free = sum(acts[i].work_at(resource) * level / owner_q[acts[i].owner]
                            for i in free)
            slope = sum(acts[i].work_at(resource) / owner_q[acts[i].owner]
                        for i in free)
            if slope == 0:
                continue
            available = CAP[resource] - used_frozen - used_free
            if available < 0:
                raise RuntimeError(f"negative available capacity for {resource}")
            candidates[resource] = available / slope
        if not candidates:
            raise RuntimeError("active set has no finite resource tier")
        step = min(candidates.values())
        limiting = tuple(sorted(resource for resource, value in candidates.items()
                                if value == step))
        level += step
        to_freeze = {i for i in free
                     if any(acts[i].work_at(resource) > 0 for resource in limiting)}
        if not to_freeze:
            raise RuntimeError("max-min tier did not freeze a limiting activity")
        for i in to_freeze:
            rates[i] = level / owner_q[acts[i].owner]
        frozen.update(to_freeze)
        free.difference_update(to_freeze)
        basis.append(("+".join(limiting), tuple(sorted(to_freeze))))

    result = {(acts[i].owner, acts[i].name): rates[i] for i in range(len(acts))}
    # Independently check every real resource constraint after the fill.
    for resource, capacity in CAP.items():
        used = sum(acts[i].work_at(resource) * rates[i] for i in range(len(acts)))
        if used > capacity:
            raise RuntimeError(f"grouped fill exceeded {resource}: {used}>{capacity}")
    return result, {"basis": tuple(basis), "owner_q": owner_q,
                    "activities": tuple((a.owner, a.name, a.q, a.work) for a in acts)}


def exact_lp_oracle(world: World) -> tuple[dict[tuple[str, str], F], list[dict]]:
    """Solve the tiny weighted max-min LP tier by tier with exact certificates.

    At a tier, frozen activities have fixed rates. For every free activity,
    y/alpha = owner_q * v, so the LP raises a common floor z with
    v_i >= z/owner_q[i]. Eliminating v leaves one variable and one constraint
    per resource: slope[r]*z <= residual[r]. The primal optimum is the
    smallest residual/slope. A binding resource gives a dual certificate
    lambda[r]=1/slope[r]; all arithmetic is Fraction-exact.

    This derives the LP from its inequalities rather than calling the
    progressive-fill implementation above. Each tied binding resource freezes
    its positive-rate users; the remaining variables continue at the same
    owner_q until the next membership event.
    """
    acts = active_activities(world)
    if not acts:
        return {}, []
    owner_q: dict[str, F] = {}
    for act in acts:
        owner_q[act.owner] = owner_q.get(act.owner, F(0)) + act.q
    free = set(range(len(acts)))
    frozen: dict[int, F] = {}
    level = F(0)
    certificates: list[dict] = []

    while free:
        residual: dict[str, F] = {}
        slopes: dict[str, F] = {}
        bounds: dict[str, F] = {}
        for resource, capacity in CAP.items():
            residual[resource] = capacity - sum(
                acts[i].work_at(resource) * rate for i, rate in frozen.items())
            slopes[resource] = sum(
                acts[i].work_at(resource) / owner_q[acts[i].owner] for i in free)
            if slopes[resource] > 0:
                bounds[resource] = residual[resource] / slopes[resource]
        if not bounds:
            raise RuntimeError("LP oracle found an unbounded tier")
        z_star = min(bounds.values())
        if z_star < level:
            raise RuntimeError("LP oracle lost primal feasibility")
        binding = tuple(sorted(r for r, bound in bounds.items() if bound == z_star))
        freeze = {i for i in free if any(acts[i].work_at(r) > 0 for r in binding)}
        if not freeze:
            raise RuntimeError("LP oracle optimum has no forced activity")

        # Exact primal feasibility and a matching one-resource dual witness.
        primal = {i: z_star / owner_q[acts[i].owner] for i in free}
        for resource, capacity in CAP.items():
            used = sum(acts[i].work_at(resource) * rate for i, rate in frozen.items())
            used += sum(acts[i].work_at(resource) * rate for i, rate in primal.items())
            if used > capacity:
                raise RuntimeError(f"LP oracle primal violates {resource}")
        dual_resource = binding[0]
        dual_lambda = F(1) / slopes[dual_resource]
        if dual_lambda * residual[dual_resource] != z_star:
            raise RuntimeError("LP oracle primal/dual objective mismatch")

        certificates.append({
            "floor": z_star,
            "free": tuple(sorted((acts[i].owner, acts[i].name) for i in free)),
            "freeze": tuple(sorted((acts[i].owner, acts[i].name) for i in freeze)),
            "residual": dict(residual),
            "slope": dict(slopes),
            "binding": binding,
            "dual_resource": dual_resource,
            "dual_lambda": dual_lambda,
            "primal_free_rates": {
                f"{acts[i].owner}.{acts[i].name}": rate
                for i, rate in primal.items()},
        })
        for i in freeze:
            frozen[i] = primal[i]
        free.difference_update(freeze)
        level = z_star

    rates = {(acts[i].owner, acts[i].name): frozen[i] for i in range(len(acts))}
    return rates, certificates


def start_cold_owner(world: World, name: str) -> None:
    owner = world.owners[name]
    owner.started = True
    owner.lprog = F(0)  # first K-full load; reserves input slot on admission


def settle(world: World) -> None:
    changed = True
    while changed:
        changed = False
        for owner in world.owners.values():
            if owner.lprog == 1:
                owner.lprog = None
                owner.ready_inputs += 1
                changed = True
        for owner in world.owners.values():
            if owner.cprog == 1:
                owner.cprog = None
                owner.completed += 1
                changed = True
        for owner in world.owners.values():
            # Compute takes only a fully loaded exact K wave.
            if owner.cprog is None and owner.ready_inputs:
                owner.ready_inputs -= 1
                owner.cprog = F(0)
                changed = True
            occupancy = (int(owner.cprog is not None) + owner.ready_inputs +
                         int(owner.lprog is not None))
            # A load reserves a slot at start and may run while C is active.
            if (owner.started and owner.cprog is not None and owner.lprog is None and
                    owner.ready_inputs == 0 and owner.completed < MAX_WAVES and
                    occupancy < 2):
                owner.lprog = F(0)
                changed = True


def state_key(world: World, rates: dict[tuple[str, str], F], solver: dict) -> tuple:
    """Exact state modulo completed homogeneous full-K waves."""
    owner_states = []
    for name in sorted(world.owners):
        o = world.owners[name]
        owner_states.append((name, o.started, "K_FULL_NONFINAL",
                             o.cprog, o.lprog, o.ready_inputs))
    return ("grouped-maxmin-v1", world.capacity_generation,
            tuple(sorted(CAP.items())), world.q_compute, world.local_work,
            tuple(owner_states), tuple(sorted(rates.items())), solver["basis"])


def owner_counts(world: World) -> dict[str, int]:
    return {name: owner.completed for name, owner in world.owners.items()}


def snapshot(world: World) -> tuple[F, dict[str, F], dict[str, int]]:
    return world.now, dict(world.stats), owner_counts(world)


def interval_delta(world: World) -> F:
    rates, _ = grouped_rates(world)
    candidates = []
    for name, owner in world.owners.items():
        cr = rates.get((name, "C"), F(0))
        lr = rates.get((name, "L"), F(0))
        if owner.cprog is not None and cr > 0:
            candidates.append((1 - owner.cprog) / cr)
        if owner.lprog is not None and lr > 0:
            candidates.append((1 - owner.lprog) / lr)
    if not candidates:
        raise RuntimeError("deadlock: no positive-rate stage")
    return min(candidates)


def advance(world: World, dt: F) -> None:
    rates, solver = grouped_rates(world)
    lp_rates, _ = exact_lp_oracle(world)
    if rates != lp_rates:
        raise RuntimeError(f"progressive fill differs from independent LP: {rates} != {lp_rates}")
    world.lp_oracle_checks += 1
    active_hbm = [name for name, owner in world.owners.items()
                  if owner.lprog is not None and rates.get((name, "L"), F(0)) > 0]
    active_comp = [name for name, owner in world.owners.items()
                   if owner.cprog is not None and rates.get((name, "C"), F(0)) > 0]
    if active_hbm:
        world.stats["hbm_busy"] += dt
        if len(active_hbm) >= 2:
            world.stats["hbm_shared"] += dt
        world.stats["hbm_work"] += sum(
            LOAD_HBM_WORK * rates[(name, "L")] * dt for name in active_hbm)
    if active_comp:
        world.stats["compute_busy"] += dt
        world.stats["compute_work"] += sum(
            COMPUTE_GATE_WORK * rates[(name, "C")] * dt for name in active_comp)
        world.stats["local_work"] += sum(
            world.local_work * rates[(name, "C")] * dt for name in active_comp)
    for name, owner in world.owners.items():
        if owner.cprog is not None:
            owner.cprog += rates[(name, "C")] * dt
        if owner.lprog is not None:
            owner.lprog += rates[(name, "L")] * dt
    world.now += dt
    world.event_intervals += 1
    settle(world)
    # Keep a bounded diagnostic trace of relative compute phase after arrival.
    if len(world.owners) == 2:
        a, b = (world.owners[k] for k in sorted(world.owners))
        if a.cprog is not None and b.cprog is not None:
            world.trace.append({
                "time": world.now,
                "phase_diff": (a.cprog - b.cprog) % 1,
                "cprog": (a.cprog, b.cprog),
                "lprog": (a.lprog, b.lprog),
                "ready": (a.ready_inputs, b.ready_inputs),
            })
            if len(world.trace) > 512:
                world.trace.pop(0)


def difference(old: tuple[F, dict[str, F], dict[str, int]], world: World):
    t0, stats0, counts0 = old
    stats_delta = {key: world.stats[key] - stats0[key] for key in stats0}
    count_delta = {key: owner_counts(world)[key] - counts0[key] for key in counts0}
    return world.now - t0, stats_delta, count_delta


def run(arrival: F, target: F, q_compute: F, fast: bool) -> tuple[World, str, int]:
    world = World(q_compute, LOCAL_PAYLOAD if q_compute == LOCAL_PAYLOAD else F(0))
    world.owners["A"] = Owner("A")
    start_cold_owner(world, "A")
    settle(world)
    joined = False
    seen: dict[tuple, tuple[F, dict[str, F], dict[str, int]]] = {}
    probes = 0
    max_probes = 0
    status = "completed"

    while world.now < target:
        if not joined and world.now == arrival:
            world.owners["B"] = Owner("B")
            start_cold_owner(world, "B")
            settle(world)
            joined = True
            seen.clear()
            probes = 0

        deadline = target if joined else min(target, arrival)
        if world.now >= deadline:
            if not joined and deadline == arrival and arrival < target:
                continue
            break

        if fast:
            rates, solver = grouped_rates(world)
            key = state_key(world, rates, solver)
            prev = seen.get(key)
            if prev is None:
                seen[key] = snapshot(world)
            else:
                period, stats_delta, count_delta = difference(prev, world)
                if period <= 0:
                    status = "fail_closed_zero_time_repeat"
                    break
                k = int((deadline - world.now) // period)
                for name, delta_count in count_delta.items():
                    if delta_count > 0:
                        k = min(k, (MAX_WAVES - world.owners[name].completed) // delta_count)
                if k > 0:
                    record = {
                        "owners": tuple(sorted(world.owners)),
                        "period": period,
                        "stats_per_period": stats_delta,
                        "counts_per_period": count_delta,
                        "solver_basis": solver["basis"],
                        "owner_q": solver["owner_q"],
                        "phase_state": tuple((name, world.owners[name].cprog,
                                              world.owners[name].lprog,
                                              world.owners[name].ready_inputs)
                                             for name in sorted(world.owners)),
                        "probe_steps": probes,
                    }
                    if not any(x["owners"] == record["owners"] for x in world.detected):
                        world.detected.append(record)
                    for name, value in stats_delta.items():
                        world.stats[name] += k * value
                    for name, value in count_delta.items():
                        world.owners[name].completed += k * value
                    world.now += k * period
                    world.skipped_periods += k
                    seen.clear()
                    probes = 0
                    continue
                # The remaining prefix is shorter than a proven period.
                seen.clear()
                probes = 0

            # The key at the state reached by probe 256 is checked above.
            # Fail before advancing a 257th interval if it has no exact match.
            if probes >= MAX_PROBES:
                status = f"fail_closed_no_exact_repeat_within_{MAX_PROBES}"
                return world, status, max_probes

        dt = min(interval_delta(world), deadline - world.now)
        advance(world, dt)
        if fast:
            probes += 1
            max_probes = max(max_probes, probes)

    if not joined and arrival <= target and world.now == arrival:
        world.owners["B"] = Owner("B")
        start_cold_owner(world, "B")
        settle(world)
    return world, status, max_probes


def key_for_final(world: World) -> tuple:
    rates, solver = grouped_rates(world)
    return state_key(world, rates, solver)


def fractional(v: F):
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def count_phase_drift(trace: list[dict]) -> dict:
    values = [row["phase_diff"] for row in trace]
    if not values:
        return {"samples": 0, "distinct_phase_differences": 0,
                "first": None, "last": None, "changed": False}
    return {
        "samples": len(values),
        "distinct_phase_differences": len(set(values)),
        "first": fractional(values[0]),
        "last": fractional(values[-1]),
        "changed": any(a != b for a, b in zip(values, values[1:])),
    }


def run_case(arrival: F, q_compute: F) -> dict:
    target = arrival + F(500)  # long enough to expose drift, small for the oracle
    naive, naive_status, _ = run(arrival, target, q_compute, fast=False)
    fast, fast_status, probes = run(arrival, target, q_compute, fast=True)
    joint = [row for row in fast.detected if row["owners"] == ("A", "B")]
    if fast_status == "completed":
        exact_match = (fast.stats == naive.stats and
                       owner_counts(fast) == owner_counts(naive) and
                       key_for_final(fast) == key_for_final(naive))
        prefix_equal = True
        gap = F(0)
    else:
        naive_prefix, _, _ = run(arrival, fast.now, q_compute, fast=False)
        prefix_equal = (fast.stats == naive_prefix.stats and
                        owner_counts(fast) == owner_counts(naive_prefix) and
                        key_for_final(fast) == key_for_final(naive_prefix))
        exact_match = None
        gap = target - fast.now
    report = {
        "arrival": fractional(arrival),
        "q_compute": fractional(q_compute),
        "local_payload_q": (fractional(LOCAL_PAYLOAD)
                             if q_compute == LOCAL_PAYLOAD else None),
        "target": fractional(target),
        "naive_event_intervals": naive.event_intervals,
        "naive_lp_oracle_checks": naive.lp_oracle_checks,
        "accelerated_event_intervals_before_stop": fast.event_intervals,
        "accelerated_lp_oracle_checks": fast.lp_oracle_checks,
        "accelerated_probe_steps": probes,
        "accelerated_status": fast_status,
        "exact_joint_repeat_found": bool(joint),
        "first_joint_repeat": ({
            "period": fractional(joint[0]["period"]),
            "stats_per_period": {k: fractional(v) for k, v in
                                 joint[0]["stats_per_period"].items()},
            "counts_per_period": joint[0]["counts_per_period"],
            "probe_steps": joint[0]["probe_steps"],
            "owner_q": {k: fractional(v) for k, v in joint[0]["owner_q"].items()},
            "phase_state": [(n, fractional(c) if c is not None else None,
                             fractional(l) if l is not None else None, ready)
                            for n, c, l, ready in joint[0]["phase_state"]],
        } if joint else None),
        "skipped_periods": fast.skipped_periods,
        "phase_drift_trace": count_phase_drift(fast.trace),
        "prefix_matches_naive": prefix_equal,
        "final_matches_naive": exact_match,
        "unadvanced_horizon_if_fail_closed": fractional(gap),
        "fast_prefix_counters": {k: fractional(v) for k, v in fast.stats.items()},
        "naive_final_counters": {k: fractional(v) for k, v in naive.stats.items()},
    }
    return report


def sample_lp_certificate(q_compute: F) -> list[dict]:
    """Export the four-active-activity L/C overlap's primal/dual LP rows."""
    world = World(q_compute, LOCAL_PAYLOAD if q_compute == LOCAL_PAYLOAD else F(0))
    world.owners["A"] = Owner("A", cprog=F(1, 5), lprog=F(0))
    world.owners["B"] = Owner("B", cprog=F(1, 7), lprog=F(0))
    rates, certs = exact_lp_oracle(world)
    prod_rates, _ = grouped_rates(world)
    if rates != prod_rates:
        raise RuntimeError("sample LP certificate disagrees with fill")
    return [{
        **{k: v for k, v in cert.items()},
        "floor": fractional(cert["floor"]),
        "residual": {k: fractional(v) for k, v in cert["residual"].items()},
        "slope": {k: fractional(v) for k, v in cert["slope"].items()},
        "dual_lambda": fractional(cert["dual_lambda"]),
        "primal_free_rates": {k: fractional(v) for k, v in
                               cert["primal_free_rates"].items()},
    } for cert in certs]


def stress_offsets() -> list[F]:
    """Deterministic arbitrary-rational phase sample independent of base rows."""
    import random
    rng = random.Random(2837)
    denominators = (97, 101, 127, 251, 257, 509, 997, 1009, 65521)
    offsets = []
    for _ in range(120):
        denominator = rng.choice(denominators)
        offsets.append(F(rng.randrange(denominator), denominator))
    return offsets


def fail_closed_prefix_test() -> dict:
    """Exercise the no-repeat exit with a deliberately lowered local cap."""
    global MAX_PROBES
    original_limit = MAX_PROBES
    MAX_PROBES = 1
    arrival = F(1, 7)
    target = arrival + F(500)
    try:
        fast, status, probes = run(arrival, target, F(1), fast=True)
    finally:
        MAX_PROBES = original_limit
    oracle, _, _ = run(arrival, fast.now, F(1), fast=False)
    return {
        "diagnostic_probe_limit": 1,
        "status": status,
        "probes": probes,
        "prefix_gap": fractional(target - fast.now),
        "prefix_stats_equal": fast.stats == oracle.stats,
        "prefix_counts_equal": owner_counts(fast) == owner_counts(oracle),
        "prefix_exact_state_equal": key_for_final(fast) == key_for_final(oracle),
    }


def main() -> int:
    arrivals = [F(0), F(1, 7), F(1, 13), F(1), F(3, 2), F(7, 3),
                F(19, 10), F(101, 10), F(1000, 7), F(17, 31), F(997, 101)]
    reports = []
    for q_compute in (F(1), LOCAL_PAYLOAD):
        for arrival in arrivals:
            reports.append(run_case(arrival, q_compute))
    stress = []
    for q_compute in (F(1), LOCAL_PAYLOAD):
        for arrival in stress_offsets():
            row = run_case(arrival, q_compute)
            stress.append({
                "arrival": row["arrival"], "q_compute": row["q_compute"],
                "status": row["accelerated_status"],
                "probes": row["accelerated_probe_steps"],
                "repeat": (row["first_joint_repeat"]["period"]
                           if row["first_joint_repeat"] else None),
                "skipped_periods": row["skipped_periods"],
                "prefix_matches_naive": row["prefix_matches_naive"],
                "final_matches_naive": row["final_matches_naive"],
            })
    result = {
        "scope": "two identical cold-start K-full non-final streams; 2 activities/owner; exact Fraction rates",
        "q_policy": "Q_load=payload=2; Q_compute=1 or LOCAL_payload=8; alpha=Q/sum(feasible Q per owner) fixed within each progressive-fill allocation; owner total weight stays 1",
        "tier_rule": "weighted max-min y/alpha; freeze users of first saturated COMPUTE_GATE, then continue remaining HBM tier without increasing frozen owner's weight; recompute alpha only after a discrete membership event",
        "capacities": {k: fractional(v) for k, v in CAP.items()},
        "probe_limit": MAX_PROBES,
        "offset_count_per_q": len(arrivals),
        "reports": reports,
        "lp_certificates_four_active": {
            fractional(q): sample_lp_certificate(q) for q in (F(1), LOCAL_PAYLOAD)},
        "stress_seed": 2837,
        "stress_denominators": [97, 101, 127, 251, 257, 509, 997, 1009, 65521],
        "stress_offsets_per_q": 120,
        "stress_reports": stress,
        "forced_fail_closed_prefix_check": fail_closed_prefix_test(),
        "status": "exploratory_not_certificate",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
