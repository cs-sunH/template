#!/usr/bin/env python3
"""Tiny exact-rational K-full cycle detector vs event-by-event reference.

This is a scratch differential, not production code. Each owner is in a warmed
OS interior: one K-full compute consumes the current ready input while one
load prefetches the next K-full input. There is no output store in a non-final
K wave. The shared resources are one HBM row and one compute gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction as F
from typing import Optional

L_WORK = F(2)       # bytes-equivalent for one K-full load
C_WORK = F(10)      # compute-equivalent for one K-full compute
HBM_CAP = F(1)
COMP_CAP = F(1)
MAX_PROBES = 256
WAVE_LIMIT = 10**12


@dataclass
class Owner:
    name: str
    cprog: Optional[F] = None
    lprog: Optional[F] = None
    ready: int = 0
    completed: int = 0
    started: bool = False


@dataclass
class World:
    now: F = F(0)
    owners: dict[str, Owner] = field(default_factory=dict)
    stats: dict[str, F] = field(default_factory=lambda: {
        "hbm_busy": F(0), "hbm_shared": F(0), "compute_busy": F(0),
        "hbm_work": F(0), "compute_work": F(0),
    })
    intervals: int = 0
    skipped_periods: int = 0
    detected: list[dict] = field(default_factory=list)


def start_warm_owner(world: World, name: str) -> None:
    """Join at a cycle boundary with current input ready, then prefetch next."""
    owner = world.owners[name]
    owner.started = True
    owner.ready = 1
    owner.cprog = F(0)
    owner.ready = 0  # the ready current input is consumed by this compute
    owner.lprog = F(0)  # reserve the second input slot and start next load


def exact_key(world: World) -> tuple:
    """State modulo completed homogeneous waves; absolute counters are omitted."""
    rows = []
    for name in sorted(world.owners):
        o = world.owners[name]
        rows.append((name, o.started, o.cprog, o.lprog, o.ready))
    return tuple(rows)


def owner_counts(world: World) -> dict[str, int]:
    return {name: o.completed for name, o in world.owners.items()}


def snapshot(world: World) -> tuple[F, dict[str, F], dict[str, int]]:
    return world.now, dict(world.stats), owner_counts(world)


def add_scaled(target: dict[str, F], delta: dict[str, F], count: int) -> None:
    for key, value in delta.items():
        target[key] += count * value


def compute_rates(world: World) -> tuple[dict[str, F], dict[str, F]]:
    """Exact equal-priority per-resource progressive rates for this fixture."""
    active_c = [o for o in world.owners.values() if o.cprog is not None]
    active_l = [o for o in world.owners.values() if o.lprog is not None]
    cr = F(0) if not active_c else COMP_CAP / (C_WORK * len(active_c))
    lr = F(0) if not active_l else HBM_CAP / (L_WORK * len(active_l))
    return ({o.name: cr for o in active_c}, {o.name: lr for o in active_l})


def settle(world: World) -> None:
    """Close simultaneous completions, then start newly ready work."""
    changed = True
    while changed:
        changed = False
        # Load completion first: it may make the next input ready for a C end
        # at the same exact timestamp.
        for o in world.owners.values():
            if o.lprog == 1:
                o.lprog = None
                o.ready += 1
                changed = True
        for o in world.owners.values():
            if o.cprog == 1:
                o.cprog = None
                o.completed += 1
                changed = True
        for o in world.owners.values():
            # A ready input is consumed only when a compute slot is free.
            if o.cprog is None and o.ready and o.completed < WAVE_LIMIT:
                o.ready -= 1
                o.cprog = F(0)
                changed = True
            # The current compute occupies one slot; a ready input occupies
            # another. A new load reserves a slot at start and remains active
            # through completion.
            occupancy = int(o.cprog is not None) + o.ready + int(o.lprog is not None)
            if (o.started and o.cprog is not None and o.lprog is None and
                    o.ready == 0 and o.completed < WAVE_LIMIT and occupancy < 2):
                o.lprog = F(0)
                changed = True


def next_local_delta(world: World) -> F:
    cr, lr = compute_rates(world)
    candidates: list[F] = []
    for o in world.owners.values():
        if o.cprog is not None and cr[o.name] > 0:
            candidates.append((1 - o.cprog) / cr[o.name])
        if o.lprog is not None and lr[o.name] > 0:
            candidates.append((1 - o.lprog) / lr[o.name])
    if not candidates:
        raise RuntimeError("deadlock: no active stage before target")
    return min(candidates)


def advance_interval(world: World, dt: F) -> None:
    if dt <= 0:
        raise RuntimeError("nonpositive event interval")
    cr, lr = compute_rates(world)
    active_l = [o for o in world.owners.values() if o.lprog is not None and lr[o.name] > 0]
    active_c = [o for o in world.owners.values() if o.cprog is not None and cr[o.name] > 0]
    if active_l:
        world.stats["hbm_busy"] += dt
        if len(active_l) >= 2:
            world.stats["hbm_shared"] += dt
        world.stats["hbm_work"] += sum(L_WORK * lr[o.name] * dt for o in active_l)
    if active_c:
        world.stats["compute_busy"] += dt
        world.stats["compute_work"] += sum(C_WORK * cr[o.name] * dt for o in active_c)
    for o in world.owners.values():
        if o.cprog is not None:
            o.cprog += cr[o.name] * dt
        if o.lprog is not None:
            o.lprog += lr[o.name] * dt
    world.now += dt
    world.intervals += 1
    settle(world)


def state_delta(old: tuple[F, dict[str, F], dict[str, int]],
                world: World) -> tuple[F, dict[str, F], dict[str, int]]:
    t0, s0, c0 = old
    return (world.now - t0,
            {k: world.stats[k] - s0[k] for k in world.stats},
            {k: owner_counts(world)[k] - c0[k] for k in c0})


def run(arrival: F, target: F, accelerated: bool) -> World:
    world = World()
    world.owners["A"] = Owner("A")
    start_warm_owner(world, "A")
    settle(world)
    joined = False
    seen: dict[tuple, tuple[F, dict[str, F], dict[str, int]]] = {}
    probes = 0

    while world.now < target:
        if not joined and world.now == arrival:
            world.owners["B"] = Owner("B")
            start_warm_owner(world, "B")
            settle(world)
            joined = True
            seen.clear()
            probes = 0

        # One external arrival deadline divides solo and two-owner windows.
        deadline = target if joined else min(target, arrival)
        if world.now >= deadline:
            if not joined and deadline == arrival and arrival < target:
                continue
            break

        if accelerated:
            key = exact_key(world)
            previous = seen.get(key)
            if previous is None:
                seen[key] = snapshot(world)
            else:
                period, stats_delta, count_delta = state_delta(previous, world)
                if period <= 0:
                    raise RuntimeError("zero-time repeated state")
                max_loops = int((deadline - world.now) // period)
                for name, delta_count in count_delta.items():
                    if delta_count > 0:
                        max_loops = min(max_loops,
                                        (WAVE_LIMIT - world.owners[name].completed) // delta_count)
                if max_loops > 0:
                    loop_record = {
                        "owners": tuple(sorted(world.owners)),
                        "period": period,
                        "stats_per_period": stats_delta,
                        "count_per_period": count_delta,
                        "probes_before_repeat": probes,
                    }
                    if (len(world.owners) == 2 and
                            all(o.cprog is not None for o in world.owners.values())):
                        loop_record["phase_distance"] = phase_distance_seconds(world, period)
                    if not any(x["owners"] == loop_record["owners"] for x in world.detected):
                        world.detected.append(loop_record)
                    add_scaled(world.stats, stats_delta, max_loops)
                    for name, delta_count in count_delta.items():
                        world.owners[name].completed += max_loops * delta_count
                    world.now += max_loops * period
                    world.skipped_periods += max_loops
                    seen.clear()
                    probes = 0
                    continue
                # Keep the remaining sub-period as an exact ordinary prefix.
                seen.clear()
                probes = 0

        local = next_local_delta(world)
        dt = min(local, deadline - world.now)
        advance_interval(world, dt)
        if accelerated and world.now < deadline:
            probes += 1
            if probes > MAX_PROBES:
                raise RuntimeError("fail_closed: no certified repeat within 256 probes")

    if not joined and arrival <= target and world.now == arrival:
        world.owners["B"] = Owner("B")
        start_warm_owner(world, "B")
        settle(world)
    return world


def phase_distance_seconds(world: World, period: F) -> F:
    a = world.owners["A"].cprog
    b = world.owners["B"].cprog
    if a is None or b is None:
        raise RuntimeError("analytic check requires both compute stages active")
    phase = (a - b) % 1
    return min(phase, 1 - phase) * period


def analytical_stable_cycle(world: World) -> dict[str, F]:
    """Closed-form check for this fixture after its joint state is periodic.

    Two compute streams share unit capacity, so each wave cycle is 20 s.
    Each owner launches one 2-unit HBM prefetch at its compute-cycle start.
    Two such bursts overlap only when their circular phase distance is <2 s.
    Processor sharing then gives a 4 s HBM union and overlap 4-2*d.
    """
    period = F(20)
    d = phase_distance_seconds(world, period)
    shared = max(F(0), F(4) - 2 * d) if d < 2 else F(0)
    return {"period": period, "hbm_busy": F(4),
            "hbm_shared": shared, "compute_busy": period,
            "hbm_work": F(4), "compute_work": F(20)}


def main() -> int:
    offsets = [F(0), F(1, 7), F(1), F(3, 2), F(19, 10),
               F(3), F(9, 2), F(10), F(101, 10), F(1000, 7)]
    checks = []
    for arrival in offsets:
        target = arrival + 73 * 20 + F(7, 13)
        naive = run(arrival, target, accelerated=False)
        fast = run(arrival, target, accelerated=True)
        if (naive.stats != fast.stats or exact_key(naive) != exact_key(fast) or
                owner_counts(naive) != owner_counts(fast)):
            raise AssertionError(f"event and skipped-cycle results differ at {arrival}")
        joint = [item for item in fast.detected if item["owners"] == ("A", "B")]
        if not joint:
            raise AssertionError(f"no joint repeat certified at offset {arrival}")
        cycle = joint[0]
        if cycle["period"] != 20:
            raise AssertionError(f"unexpected joint period {cycle['period']} at {arrival}")
        # At a repeated key, the exact one-period integral must match the
        # independently derived phase-mask formula.
        # Reconstruct the phase separation from the repeated exact state.
        d = cycle["phase_distance"]
        expected_shared = max(F(0), F(4) - 2 * d) if d < 2 else F(0)
        expected = {"period": F(20), "hbm_busy": F(4),
                    "hbm_shared": expected_shared, "compute_busy": F(20),
                    "hbm_work": F(4), "compute_work": F(20)}
        actual = cycle["stats_per_period"]
        actual_named = {"period": cycle["period"], **actual}
        if actual_named != expected:
            raise AssertionError(
                f"analytic mask integral differs at {arrival}: "
                f"actual={actual_named}, expected={expected}, d={d}")
        checks.append({
            "arrival": str(arrival), "horizon": str(target),
            "naive_event_intervals": naive.intervals,
            "fast_probe_intervals": fast.intervals,
            "skipped_periods": fast.skipped_periods,
            "solo_period": (str(next(x["period"] for x in fast.detected
                                      if x["owners"] == ("A",)))
                            if any(x["owners"] == ("A",) for x in fast.detected)
                            else None),
            "joint_period": str(cycle["period"]),
            "probe_steps_to_joint_repeat": cycle["probes_before_repeat"],
            "joint_busy": {k: str(v) for k, v in cycle["stats_per_period"].items()},
            "joint_phase_distance": str(d),
            "status": "exact_match",
        })
    print({"fixture": "two identical warmed K-full owners; HBM=1, compute=1; 2 input slots",
           "offsets_checked": len(checks), "max_probe_steps": MAX_PROBES,
           "checks": checks, "status": "passed"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
