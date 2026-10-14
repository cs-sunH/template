#!/usr/bin/env python3
"""Nested K/output affine composition for the frozen offset-1/20 trace.

This is an independent scratch verifier. It imports only the exact geometry,
wave facts, grouped-LP oracle, and saved event rows from output-cycle-v2. It does
not modify or run the archived checker. The runtime-shaped representation is
32 category event templates per output segment, one identity middle-K map with
a repeat count, and a one-dimensional affine output map with fixed-width
interval powering for large repeat counts.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from fractions import Fraction as F
from pathlib import Path
from typing import Any

sys.set_int_max_str_digits(0)

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'die/src/Execution.cc').is_file())
ARCHIVE = ROOT / "die/docs/analysis/bounded-lifecycle-20261014/output-cycle-v2"
SOURCE = ARCHIVE / "derive_affine_cycle.py"
PROOF = ARCHIVE / "affine-cycle-proof.json"
OUT_JSON = Path(__file__).with_name("nested-k-output.json")
OUT_MD = Path(__file__).with_name("nested-k-output.md")
OUT_AFFINE = Path(__file__).with_name("nested-k-output.affine")

spec = importlib.util.spec_from_file_location("frozen_affine_cycle", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import archived helper {SOURCE}")
frozen = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = frozen
spec.loader.exec_module(frozen)

Aff = frozen.Aff


def fstr(q: F) -> str:
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def aff_json(x: Aff) -> dict[str, str]:
    return {"slope": fstr(x.a), "intercept": fstr(x.b)}


def aff_at(x: Aff, q: F) -> F:
    return x.a * q + x.b


def substitute(x: Aff, inner: Aff) -> Aff:
    return Aff(x.a * inner.a, x.a * inner.b + x.b)


def add_nonnegative(domain: list[F | None], expr: Aff) -> None:
    if expr.a == 0:
        if expr.b < 0:
            raise AssertionError(f"empty category trace domain at {expr}")
        return
    cut = -expr.b / expr.a
    if expr.a > 0:
        domain[0] = cut if domain[0] is None else max(domain[0], cut)
    else:
        domain[1] = cut if domain[1] is None else min(domain[1], cut)
    if domain[0] is not None and domain[1] is not None and domain[0] > domain[1]:
        raise AssertionError("empty category trace domain")


def role_key(ident: tuple[int, int, int], group: int) -> tuple[int, int, int]:
    owner, stage, wave = ident
    return owner, stage, wave - group


def wave_class(facts: dict[str, Any]) -> str:
    if facts["k_index"] == 0:
        return "K_FIRST"
    if facts["final_k"]:
        return "K_FINAL"
    return "K_MIDDLE"


def semantic_facts(facts: dict[str, Any]) -> tuple:
    """Only facts that can change the grouped LP or physical ledger."""
    return (
        wave_class(facts), facts["m"], facts["n"], facts["k"],
        facts["read"], facts["write"], facts["read_rows"],
        facts["write_rows"], facts["vec_ops"], facts["local"],
    )


@dataclass(frozen=True)
class Activity:
    key: tuple[int, int, int]
    rate: F
    q: F
    facts: dict[str, Any]


@dataclass(frozen=True)
class Event:
    activities: dict[tuple[int, int, int], Activity]
    selected: tuple[tuple[int, int, int], ...]
    basis: str
    source_retired: tuple[int, int]


@dataclass
class LifeOwner:
    next_load: int
    retired: int
    active: dict[int, int]
    loaded: list[int]
    computed: list[int]


def life_occupancy(owner: LifeOwner, k_tiles: int) -> tuple[int, int]:
    input_slots = len(owner.loaded) + int(0 in owner.active) + int(1 in owner.active)
    final_compute = (int(owner.active[1] % k_tiles == k_tiles - 1)
                     if 1 in owner.active else 0)
    output_slots = len(owner.computed) + int(2 in owner.active) + final_compute
    return input_slots, output_slots


def life_active_keys(states: list[LifeOwner], group: int) -> set[tuple[int, int, int]]:
    return {(owner_id, stage, wave - group)
            for owner_id, state in enumerate(states)
            for stage, wave in state.active.items()}


def check_life(states: list[LifeOwner], group: int, k_tiles: int) -> None:
    for state in states:
        if len(state.active) > 3 or len(state.loaded) > 2 or len(state.computed) > 2:
            raise AssertionError("lifecycle queue exceeds its declared fixed slot count")
        in_slots, out_slots = life_occupancy(state, k_tiles)
        if in_slots > 2 or out_slots > 2:
            raise AssertionError("lifecycle occupancy exceeds two input/output slots")
    _ = life_active_keys(states, group)


def advance_life(states: list[LifeOwner], group: int,
                 selected: tuple[tuple[int, int, int], ...],
                 k_tiles: int, wave_count: int) -> None:
    selected_actual = {(owner, stage, key_wave + group)
                       for owner, stage, key_wave in selected}
    for stage in (0, 1, 2):
        for owner_id, state in enumerate(states):
            ident = next((it for it in selected_actual
                          if it[0] == owner_id and it[1] == stage), None)
            if ident is None:
                continue
            _, _, wave = ident
            if state.active.get(stage) != wave:
                raise AssertionError(f"selected lifecycle completion absent: {ident}")
            del state.active[stage]
            if stage == 0:
                state.loaded.append(wave)
            elif stage == 1:
                if wave % k_tiles == k_tiles - 1:
                    state.computed.append(wave)
            else:
                if wave % k_tiles != k_tiles - 1:
                    raise AssertionError(f"store retired a non-final-K wave: {wave}")
                state.retired += 1

    while True:
        changed = False
        for state in states:
            input_slots, output_slots = life_occupancy(state, k_tiles)
            if 0 not in state.active and state.next_load < wave_count and input_slots < 2:
                state.active[0] = state.next_load
                state.next_load += 1
                changed = True
            _, output_slots = life_occupancy(state, k_tiles)
            if 1 not in state.active and state.loaded and output_slots < 2:
                state.active[1] = state.loaded.pop(0)
                changed = True
            if 2 not in state.active and state.computed:
                state.active[2] = state.computed.pop(0)
                changed = True
        if not changed:
            break
    check_life(states, group, k_tiles)


def clone_life(states: list[LifeOwner]) -> list[LifeOwner]:
    return [LifeOwner(s.next_load, s.retired, dict(s.active), list(s.loaded),
                      list(s.computed)) for s in states]


def shift_life(states: list[LifeOwner], amount: int) -> list[LifeOwner]:
    return [LifeOwner(s.next_load + amount, s.retired,
                      {stage: wave + amount for stage, wave in s.active.items()},
                      [wave + amount for wave in s.loaded],
                      [wave + amount for wave in s.computed]) for s in states]


LEDGERS = (
    "time_s", "compute_ops", "compute_gate_s", "local_bytes",
    "hbm0_bytes", "hbm1_bytes", "busy_compute_s", "busy_hbm0_s",
    "busy_hbm1_s", "busy_hbm_union_s",
)


def empty_ledger() -> dict[str, Aff]:
    return {name: Aff() for name in LEDGERS}


def add_ledger(dst: dict[str, Aff], src: dict[str, Aff], multiplier: int = 1) -> None:
    for name in LEDGERS:
        dst[name] = dst[name] + src[name] * multiplier


def ledger_substitute(src: dict[str, Aff], inner: Aff) -> dict[str, Aff]:
    return {name: substitute(value, inner) for name, value in src.items()}


def ledger_equal(a: dict[str, Aff], b: dict[str, Aff]) -> bool:
    return all(a[name] == b[name] for name in LEDGERS)


def build_groups(rows: list[dict], k_tiles: int,
                 independently_check_groups: set[int]) -> list[tuple[Event, ...]]:
    if len(rows) % 4:
        raise AssertionError("trace interval count is not divisible by four")
    groups: list[tuple[Event, ...]] = []
    for group_no in range(len(rows) // 4):
        group = []
        for row in rows[group_no * 4:(group_no + 1) * 4]:
            # Re-solve the grouped LP only for the 8 category representatives.
            # Repeated source positions are checked against the representative
            # signature below; solving all 520 archived rows again would just
            # repeat the already-passed flat proof.
            if group_no in independently_check_groups:
                rates, qs, basis = frozen.independently_solve_grouped_lp(row["activities"])
                if "+".join(basis) != row["basis"]:
                    raise AssertionError(f"independent grouped LP basis mismatch in group {group_no}")
            else:
                rates = {ident: info["rate"] for ident, info in row["activities"].items()}
                qs = {ident: info["q"] for ident, info in row["activities"].items()}
                basis = row["basis"].split("+")
            acts: dict[tuple[int, int, int], Activity] = {}
            for ident, info in row["activities"].items():
                owner, stage, wave = ident
                facts = frozen.wave_facts(wave)
                key = role_key(ident, group_no)
                if key in acts:
                    raise AssertionError(f"multiple simultaneous activities share lane key {key}")
                if rates[ident] != info["rate"] or qs[ident] != info["q"]:
                    raise AssertionError(f"independent LP disagrees with trace for {ident}")
                acts[key] = Activity(key, rates[ident], qs[ident], facts)
            selected = tuple(role_key(i, group_no) for i in row["selected"])
            retired = tuple(map(int, row["tokens"]["retired"].split(",")))
            for key in selected:
                if key not in acts:
                    raise AssertionError(f"selected event absent from active lane set: {key}")
            for key, activity in acts.items():
                if activity.rate <= 0:
                    raise AssertionError(f"nonpositive active rate: {key}")
                if activity.facts["m"] != frozen.TM or activity.facts["n"] != frozen.TN:
                    raise AssertionError("trace crossed M/N tail within the claimed full category")
                if activity.facts["k"] not in (1, frozen.TK):
                    raise AssertionError("unexpected K-size category in fixture")
            group.append(Event(acts, selected, "+".join(basis), retired))
        groups.append(tuple(group))
    return groups


def event_signature(event: Event) -> tuple:
    return (
        tuple((key, activity.q, activity.rate, semantic_facts(activity.facts))
              for key, activity in sorted(event.activities.items())),
        event.selected,
        event.basis,
    )


def group_signature(group: tuple[Event, ...]) -> tuple:
    return tuple(event_signature(event) for event in group)


class Composer:
    def __init__(self, k_tiles: int, wave_count: int, orbit: tuple[F, F]):
        self.k_tiles = k_tiles
        self.wave_count = wave_count
        self.orbit = orbit
        self.domain: list[F | None] = [F(0), F(1)]
        self.ledger = empty_ledger()
        self.event_count = 0
        self.template_event_count = 0
        self.max_margin = None
        self.min_margin = None
        self.min_dt = None
        self.min_progress = None
        self.max_progress = None

    def record_margin(self, value: F) -> None:
        self.min_margin = value if self.min_margin is None else min(self.min_margin, value)
        self.max_margin = value if self.max_margin is None else max(self.max_margin, value)

    def run_group(self, template: tuple[Event, ...], group_no: int,
                  state: dict[tuple[int, int, int], Aff],
                  life: list[LifeOwner], *, reference_group: tuple[Event, ...] | None = None,
                  count_ledger: bool = True
                  ) -> tuple[dict[tuple[int, int, int], Aff], list[LifeOwner], dict[str, Aff]]:
        local_ledger = empty_ledger()
        if reference_group is None:
            reference_group = template
        for event_no, event in enumerate(template):
            if tuple(s.retired for s in life) != reference_group[event_no].source_retired:
                raise AssertionError("bounded output template retirement cursor differs from source")
            active = set(event.activities)
            if life_active_keys(life, group_no) != active:
                raise AssertionError(
                    f"category lifecycle activity mismatch at group={group_no}, event={event_no}: "
                    f"{life_active_keys(life, group_no)} != {active}")
            for key in active:
                state.setdefault(key, Aff())
            if set(state) != active:
                raise AssertionError(f"symbolic active state mismatch at group {group_no}/{event_no}")
            deadlines = {key: (Aff(F(0), F(1)) - state[key]) / activity.rate
                         for key, activity in event.activities.items()}
            selected_deadlines = [deadlines[key] for key in event.selected]
            dt = selected_deadlines[0]
            if any(candidate != dt for candidate in selected_deadlines):
                raise AssertionError("selected completions are not an identity tie")
            exact_ties = {key for key, deadline in deadlines.items() if deadline == dt}
            if exact_ties != set(event.selected):
                raise AssertionError("selected set omits or adds an identity tie")
            dt_lo, dt_hi = (aff_at(dt, q) for q in self.orbit)
            if dt_lo <= 0 or dt_hi <= 0:
                raise AssertionError(f"selected dt is not strictly positive: {dt_lo}, {dt_hi}")
            self.min_dt = min(dt_lo, dt_hi) if self.min_dt is None else min(self.min_dt, dt_lo, dt_hi)
            add_nonnegative(self.domain, dt)
            for key, deadline in deadlines.items():
                if key in event.selected:
                    continue
                margin = deadline - dt
                values = (aff_at(margin, self.orbit[0]), aff_at(margin, self.orbit[1]))
                if min(values) <= 0:
                    raise AssertionError(
                        f"nonselected deadline lacks strict clearance: {key}, {values}")
                self.record_margin(min(values))
                add_nonnegative(self.domain, margin)
            for key, progress in state.items():
                values = (aff_at(progress, self.orbit[0]), aff_at(progress, self.orbit[1]))
                if min(values) < 0 or max(values) > 1:
                    raise AssertionError(f"progress outside [0,1] for {key}: {values}")
                self.min_progress = min(values) if self.min_progress is None else min(
                    self.min_progress, *values)
                self.max_progress = max(values) if self.max_progress is None else max(
                    self.max_progress, *values)

            # Integrate each physical ledger from its owner's independent progress.
            hbm0_positive = hbm1_positive = compute_positive = False
            for key, activity in event.activities.items():
                owner, stage, _ = key
                delta = activity.rate * dt
                facts = activity.facts
                if stage == 0:
                    b0, b1 = facts["read_rows"]
                    local_ledger["hbm0_bytes"] = local_ledger["hbm0_bytes"] + b0 * delta
                    local_ledger["hbm1_bytes"] = local_ledger["hbm1_bytes"] + b1 * delta
                    hbm0_positive |= b0 > 0
                    hbm1_positive |= b1 > 0
                elif stage == 1:
                    local_ledger["compute_ops"] = local_ledger["compute_ops"] + facts["vec_ops"] * delta
                    local_ledger["compute_gate_s"] = local_ledger["compute_gate_s"] + (
                        F(facts["vec_ops"], frozen.COMPUTE_DENOM) * delta)
                    local_ledger["local_bytes"] = local_ledger["local_bytes"] + facts["local"] * delta
                    compute_positive = facts["vec_ops"] > 0
                elif stage == 2:
                    b0, b1 = facts["write_rows"]
                    local_ledger["hbm0_bytes"] = local_ledger["hbm0_bytes"] + b0 * delta
                    local_ledger["hbm1_bytes"] = local_ledger["hbm1_bytes"] + b1 * delta
                    hbm0_positive |= b0 > 0
                    hbm1_positive |= b1 > 0
                else:
                    raise AssertionError(f"invalid stage {stage}")
            local_ledger["time_s"] = local_ledger["time_s"] + dt
            if compute_positive:
                local_ledger["busy_compute_s"] = local_ledger["busy_compute_s"] + dt
            if hbm0_positive:
                local_ledger["busy_hbm0_s"] = local_ledger["busy_hbm0_s"] + dt
            if hbm1_positive:
                local_ledger["busy_hbm1_s"] = local_ledger["busy_hbm1_s"] + dt
            if hbm0_positive or hbm1_positive:
                local_ledger["busy_hbm_union_s"] = local_ledger["busy_hbm_union_s"] + dt

            for key, activity in event.activities.items():
                state[key] = state[key] + activity.rate * dt
            for key in event.selected:
                if state[key] != Aff(F(0), F(1)):
                    raise AssertionError(f"selected activity did not complete identically: {key}")
                # Actual final-K store is the sole output retirement point.
                actual_wave = key[2] + group_no
                if key[1] == 2 and actual_wave % self.k_tiles != self.k_tiles - 1:
                    raise AssertionError(f"non-final K store selected: {actual_wave}")
                del state[key]
            advance_life(life, group_no, event.selected, self.k_tiles, self.wave_count)
            next_active = life_active_keys(life, group_no)
            for key in next_active:
                state.setdefault(key, Aff())
            for key in list(state):
                if key not in next_active:
                    del state[key]
            if set(state) != next_active:
                raise AssertionError("symbolic progress and lifecycle membership diverged")
            if event_no == len(template) - 1:
                # The next K macro advances its wave base by one. Convert lane
                # IDs from (wave - current_group) to (wave - next_group).
                state = {(owner, stage, wave_key - 1): progress
                         for (owner, stage, wave_key), progress in state.items()}
                if set(state) != life_active_keys(life, group_no + 1):
                    raise AssertionError("output-segment lane normalization failed at group boundary")
            self.event_count += 1
            self.template_event_count += 1
        if count_ledger:
            add_ledger(self.ledger, local_ledger)
        return state, life, local_ledger

    def run_mid_repeat(self, template: tuple[Event, ...], start_group: int, count: int,
                       state: dict[tuple[int, int, int], Aff],
                       life: list[LifeOwner],
                       reference_group: tuple[Event, ...] | None = None
                       ) -> tuple[dict, list[LifeOwner]]:
        if count < 0:
            raise AssertionError("negative middle repeat count")
        before_state = dict(state)
        before_life = clone_life(life)
        state, life, one_ledger = self.run_group(template, start_group, state, life,
                                                  reference_group=reference_group,
                                                  count_ledger=False)
        if state != before_state:
            raise AssertionError("middle-K category map is not exact state identity")
        if shift_life(before_life, 1) != life:
            raise AssertionError("middle-K lifecycle does not shift by exactly one wave")
        add_ledger(self.ledger, one_ledger, count)
        # The proof checks only one transition. Its exact state identity means
        # every repeated transition sees the same phase and the same template.
        life = shift_life(before_life, count)
        return state, life


def source_rows_to_group_semantics(group: tuple[Event, ...]) -> tuple:
    return group_signature(group)


def compose_half(groups: list[tuple[Event, ...]], composer: Composer,
                 state: dict[tuple[int, int, int], Aff], life: list[LifeOwner],
                 start_group: int, middle_count: int) -> tuple[
                     dict[tuple[int, int, int], Aff], list[LifeOwner], dict[str, Aff], int]:
    before_state = dict(state)
    before_life = clone_life(life)
    ledger_before = dict(composer.ledger)
    count_before = composer.event_count
    # K-first phase, one repeated middle-category transform, then six exact
    # K-final/output/retirement boundary categories (group 59..64 here).
    state, life, _ = composer.run_group(groups[0], start_group, state, life,
                                        reference_group=groups[start_group])
    current = start_group + 1
    state, life = composer.run_mid_repeat(groups[1], current, middle_count, state, life,
                                          reference_group=groups[current])
    current += middle_count
    for representative in range(current, start_group + 65):
        # Each singleton boundary class is selected by its arithmetic K index.
        local_template = groups[representative % 65]
        state, life, _ = composer.run_group(local_template, representative, state, life,
                                            reference_group=groups[representative])
        current += 1
    if current != start_group + 65:
        raise AssertionError("output segment did not consume exactly K_tiles macro steps")
    shift = shift_life(before_life, frozen.K_TILES)
    if any(life[i].next_load != shift[i].next_load or
           life[i].active != shift[i].active or life[i].loaded != shift[i].loaded or
           life[i].computed != shift[i].computed for i in range(2)):
        raise AssertionError("one-output lifecycle wave/slot transform is not a +K_tiles shift")
    if [life[i].retired - before_life[i].retired for i in range(2)] != [1, 1]:
        raise AssertionError("one-output segment must retire one output per owner")
    if set(state) != set(before_state) or len(state) != 3:
        raise AssertionError("one-output segment did not return to its bounded lane state")
    half_ledger = {name: composer.ledger[name] - ledger_before[name] for name in LEDGERS}
    micro_events = composer.event_count - count_before
    return state, life, half_ledger, micro_events


def make_initial_state() -> tuple[dict, list[LifeOwner], Aff, Aff, F]:
    boundary = frozen.boundary_states()
    probes = ("5", "9", "13")
    xs = [frozen.progress_at(boundary[p], "owner1_C") for p in probes]
    x0 = xs[0]
    owner0_c = frozen.fit_line(xs, [frozen.progress_at(boundary[p], "owner0_C") for p in probes])
    owner1_l = frozen.fit_line(xs, [frozen.progress_at(boundary[p], "owner1_L") for p in probes])
    state = {
        (0, 1, 195): owner0_c,
        (1, 0, 135): owner1_l,
        (1, 1, 134): Aff(F(1), F(0)),
    }
    life = [
        LifeOwner(197, 3, {1: 195}, [196], []),
        LifeOwner(136, 2, {0: 135, 1: 134}, [], []),
    ]
    return state, life, owner0_c, owner1_l, x0


def exact_transform_power(a: F, b: F, ledger: dict[str, Aff], x0: F, n: int
                          ) -> tuple[F, dict[str, F]]:
    """Small-count exact reference; not used for the large fixed-width result."""
    x = x0
    totals = {name: F(0) for name in LEDGERS}
    for _ in range(n):
        for name in LEDGERS:
            totals[name] += aff_at(ledger[name], x)
        x = a * x + b
    return x, totals


SCALE = 1 << 128
MAX_STORED_BITS = 256
MAX_PRODUCT_BITS = 512
MAX_ENDPOINT_BITS_SEEN = 0
MAX_PRODUCT_BITS_SEEN = 0


def floor_div(n: int, d: int) -> int:
    return n // d


def ceil_div(n: int, d: int) -> int:
    return -((-n) // d)


@dataclass(frozen=True)
class Interval:
    """Closed fixed-point interval, endpoints divided by SCALE."""
    lo: int
    hi: int

    def __post_init__(self) -> None:
        global MAX_ENDPOINT_BITS_SEEN
        bits = max(abs(self.lo).bit_length(), abs(self.hi).bit_length())
        MAX_ENDPOINT_BITS_SEEN = max(MAX_ENDPOINT_BITS_SEEN, bits)
        if bits > MAX_STORED_BITS:
            raise OverflowError(f"fixed-point endpoint needs {bits} bits (limit {MAX_STORED_BITS})")

    @staticmethod
    def from_fraction(q: F) -> "Interval":
        n = q.numerator * SCALE
        d = q.denominator
        return Interval(floor_div(n, d), ceil_div(n, d))

    @staticmethod
    def exact_int(n: int) -> "Interval":
        v = n * SCALE
        return Interval(v, v)

    def __add__(self, rhs: "Interval") -> "Interval":
        return Interval(self.lo + rhs.lo, self.hi + rhs.hi)

    def __mul__(self, rhs: "Interval") -> "Interval":
        global MAX_PRODUCT_BITS_SEEN
        values = (self.lo * rhs.lo, self.lo * rhs.hi,
                  self.hi * rhs.lo, self.hi * rhs.hi)
        product_bits = max(abs(value).bit_length() for value in values)
        MAX_PRODUCT_BITS_SEEN = max(MAX_PRODUCT_BITS_SEEN, product_bits)
        if product_bits > MAX_PRODUCT_BITS:
            raise OverflowError(f"fixed-point product needs {product_bits} bits (limit {MAX_PRODUCT_BITS})")
        return Interval(floor_div(min(values), SCALE), ceil_div(max(values), SCALE))

    def contains(self, q: F) -> bool:
        return self.lo * q.denominator <= q.numerator * SCALE <= self.hi * q.denominator

    def width_fraction(self) -> F:
        return F(self.hi - self.lo, SCALE)

    def as_json(self) -> dict[str, str]:
        return {"lo": fstr(F(self.lo, SCALE)), "hi": fstr(F(self.hi, SCALE)),
                "width": fstr(self.width_fraction())}


@dataclass(frozen=True)
class Transform:
    a: Interval
    b: Interval
    c: tuple[Interval, ...]
    d: tuple[Interval, ...]


def transform_from_exact(a: F, b: F, metrics: tuple[tuple[F, F], ...]) -> Transform:
    return Transform(Interval.from_fraction(a), Interval.from_fraction(b),
                     tuple(Interval.from_fraction(c) for c, _ in metrics),
                     tuple(Interval.from_fraction(d) for _, d in metrics))


def compose_transform(first: Transform, second: Transform) -> Transform:
    # Apply first, then second: x'=a*x+b; y_j'=y_j+c_j*x+d_j.
    if len(first.c) != len(second.c):
        raise AssertionError("transform ledger dimensions differ")
    a = second.a * first.a
    b = second.a * first.b + second.b
    c = tuple(first.c[i] + second.c[i] * first.a for i in range(len(first.c)))
    d = tuple(first.d[i] + second.c[i] * first.b + second.d[i]
              for i in range(len(first.d)))
    return Transform(a, b, c, d)


def identity_transform(dims: int) -> Transform:
    return Transform(Interval.exact_int(1), Interval.exact_int(0),
                     tuple(Interval.exact_int(0) for _ in range(dims)),
                     tuple(Interval.exact_int(0) for _ in range(dims)))


def power_transform(base: Transform, count: int) -> Transform:
    if count < 0:
        raise AssertionError("negative segment repeat count")
    result = identity_transform(len(base.c))
    while count:
        if count & 1:
            result = compose_transform(result, base)
        count >>= 1
        if count:
            base = compose_transform(base, base)
    return result


def apply_transform(transform: Transform, x: Interval) -> tuple[Interval, tuple[Interval, ...]]:
    x_out = transform.a * x + transform.b
    y_out = tuple(transform.c[i] * x + transform.d[i] for i in range(len(transform.c)))
    return x_out, y_out


def max_safe_repeats(transform: Transform, x0: F, stop: F, time_index: int,
                     upper_count: int) -> tuple[int, dict[str, Any]]:
    """Largest n with certified upper elapsed time <= exact external stop."""
    stop_i = Interval.from_fraction(stop)

    def elapsed(n: int) -> Interval:
        powered = power_transform(transform, n)
        _, ys = apply_transform(powered, Interval.from_fraction(x0))
        return ys[time_index]

    lo, hi = 0, upper_count
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if elapsed(mid).hi <= stop_i.lo:
            lo = mid
        else:
            hi = mid - 1
    at_n = elapsed(lo)
    next_i = elapsed(lo + 1)
    if at_n.hi > stop_i.lo or next_i.lo <= stop_i.hi:
        raise ArithmeticError("fixed-width time interval could not certify the external stop bound")
    return lo, {"safe_elapsed": at_n.as_json(), "next_elapsed": next_i.as_json(),
                "stop": stop_i.as_json(), "binary_search_steps": math.ceil(math.log2(upper_count + 1))}


def main() -> None:
    rows, endpoint = frozen.read_event_rows()
    proof = json.loads(PROOF.read_text())
    k_tiles = frozen.K_TILES
    # Derive the owner phase offset from the lifecycle boundary before creating
    # the bounded template table.
    raw_compute = {owner: wave for (owner, stage, wave) in rows[0]["activities"]
                   if stage == 1}
    if set(raw_compute) != {0, 1}:
        raise AssertionError("expected one active compute wave per owner at the K boundary")
    raw_k0, raw_k1 = raw_compute[0] % k_tiles, raw_compute[1] % k_tiles
    phase_offset = (raw_k1 - raw_k0) % k_tiles
    if raw_k0 != 0 or phase_offset != 4:
        raise AssertionError(f"unsupported K phase: owner0={raw_k0}, owner1={raw_k1}")
    middle_count = k_tiles - phase_offset - 3
    if middle_count != 58:
        raise AssertionError(f"unexpected analytically derived middle run count {middle_count}")
    representatives = [0, 1, *range(1 + middle_count, k_tiles)]
    if representatives != [0, 1, 59, 60, 61, 62, 63, 64]:
        raise AssertionError(f"unexpected category representatives {representatives}")
    groups = build_groups(rows, k_tiles, set(representatives))
    if len(groups) != 130 or len(rows) != 520:
        raise AssertionError(f"unexpected source geometry: groups={len(groups)}, events={len(rows)}")
    if len(groups) != 2 * k_tiles:
        raise AssertionError("520 intervals are not two K-consumption blocks at four intervals/K")

    # The active C waves at the initial output boundary establish a four-wave
    # phase offset. This is the only offset supported by this proof.
    start_events = groups[0][0]
    c_by_owner = {}
    for key in start_events.activities:
        if key[1] == 1:
            c_by_owner[key[0]] = key
    if set(c_by_owner) != {0, 1}:
        raise AssertionError("expected one active compute wave per owner at the K boundary")
    k0 = c_by_owner[0][2] % k_tiles
    k1 = c_by_owner[1][2] % k_tiles
    phase_offset = (k1 - k0) % k_tiles
    if k0 != 0 or phase_offset != 4:
        raise AssertionError(f"unsupported K phase: owner0={k0}, owner1={k1}")
    # The finite event stream is used here only to identify/validate the eight
    # category templates. The represented transfer executes 8 templates ×
    # four event steps per output, plus one power of the repeated middle map.
    expected_mid = group_signature(groups[1])
    for g in list(range(1, 1 + middle_count)) + list(range(k_tiles + 1, k_tiles + 1 + middle_count)):
        if group_signature(groups[g]) != expected_mid:
            raise AssertionError(f"middle K category changes at group {g}")
    for rep in representatives:
        if group_signature(groups[rep]) != group_signature(groups[rep + k_tiles]):
            raise AssertionError(f"second output block differs from first category at {rep}")

    state0, life0, u_line, v_line, x0 = make_initial_state()
    fixed = F(proof["phase_fixed_point"])
    old_full = proof["phase_progress_map"]
    a_full_saved = F(old_full["slope"])
    b_full_saved = F(old_full["intercept"])
    orbit = (fixed, x0)
    composer = Composer(k_tiles, frozen.WAVE_COUNT, orbit)
    life_start = clone_life(life0)

    # Run one K/output segment using only the eight selected category templates.
    half_state, half_life, half_ledger, first_half_micro_events = compose_half(
        groups, composer, dict(state0), clone_life(life0), 0, middle_count)
    x_half = half_state[(1, 1, 134)]
    a_half, b_half = x_half.a, x_half.b
    if not (0 < a_half < 1):
        raise AssertionError(f"one-output map is not a strict contraction: {x_half}")
    if half_state[(0, 1, 195)] != substitute(u_line, x_half):
        raise AssertionError("owner0 compute state does not return to the initial invariant line")
    if half_state[(1, 0, 135)] != substitute(v_line, x_half):
        raise AssertionError("owner1 load state does not return to the initial invariant line")
    half_fixed = b_half / (1 - a_half)
    if half_fixed != fixed:
        raise AssertionError("one-output and two-output maps do not share the saved fixed point")
    if not fixed <= x0 and not fixed < x0:
        raise AssertionError("initial phase is below its fixed point")
    if not (fixed <= x0):
        raise AssertionError("bad orbit order")

    # Apply the same output transform to the symbolic output state. The
    # category templates and lifecycle lanes recur modulo +65 waves.
    composer2 = Composer(k_tiles, frozen.WAVE_COUNT, orbit)
    second_state, second_life, second_ledger, second_half_micro_events = compose_half(
        groups, composer2, dict(half_state), clone_life(half_life), k_tiles, middle_count)
    x_full = second_state[(1, 1, 134)]
    if x_full != substitute(x_half, x_half):
        raise AssertionError("two nested output maps do not equal the square of the one-output map")
    if x_full != Aff(a_full_saved, b_full_saved):
        raise AssertionError("nested K/output map does not reproduce the saved 520-event recurrence")
    if second_state[(0, 1, 195)] != substitute(u_line, x_full):
        raise AssertionError("two-output owner0 state leaves the invariant line")
    if second_state[(1, 0, 135)] != substitute(v_line, x_full):
        raise AssertionError("two-output owner1 load state leaves the invariant line")

    # Each half has the same affine ledger map when expressed in its input x.
    expected_second_ledger = ledger_substitute(half_ledger, x_half)
    if not ledger_equal(second_ledger, expected_second_ledger):
        raise AssertionError("the second output's affine byte/time/busy map differs")
    full_ledger = {name: half_ledger[name] + second_ledger[name] for name in LEDGERS}
    saved_time = Aff(F(proof["cycle_elapsed_time_affine"]["slope"]),
                     F(proof["cycle_elapsed_time_affine"]["intercept"]))
    if full_ledger["time_s"] != saved_time:
        raise AssertionError("nested elapsed time differs from the saved exact 520-event map")
    if endpoint is None:
        raise AssertionError("missing archived trace endpoint")
    if aff_at(x_full, x0) != F(endpoint["o1_s1"].split("@", 1)[1].split("|", 1)[0]):
        raise AssertionError("nested state at the initial phase does not match trace endpoint")

    # Work/byte conservation for one owner output per half. The output segment
    # advances one output tile on both owners and includes K-final store/retire.
    m, n, k = frozen.TM, frozen.TN, frozen.FIXTURE_K
    read_one = 4 * k * (m + n)
    write_one = 4 * m * n
    ops_one = 2 * m * n * k
    local_one = read_one + 4 * m * n * k_tiles + 4 * m * n * (k_tiles - 1)
    half_totals = {name: aff_at(half_ledger[name], x0) for name in LEDGERS}
    # Boundary activities are fractional at both ends. The aggregate service
    # equals one complete output per owner plus (end_progress-start_progress)
    # times the work of the matching boundary lane; this makes the conservation
    # identity telescope across adjacent output segments.
    expected_aff = {
        "compute_ops": Aff(F(0), F(2 * ops_one)),
        "compute_gate_s": Aff(F(0), F(2 * ops_one, frozen.COMPUTE_DENOM)),
        "local_bytes": Aff(F(0), F(2 * local_one)),
        "hbm0_bytes": Aff(F(0), F(read_one + write_one)),
        "hbm1_bytes": Aff(F(0), F(read_one + write_one)),
    }
    for key, start_progress in state0.items():
        end_progress = half_state[key]
        delta = end_progress - start_progress
        facts = frozen.wave_facts(key[2])
        if key[1] == 1:
            expected_aff["compute_ops"] = expected_aff["compute_ops"] + facts["vec_ops"] * delta
            expected_aff["compute_gate_s"] = expected_aff["compute_gate_s"] + (
                F(facts["vec_ops"], frozen.COMPUTE_DENOM) * delta)
            expected_aff["local_bytes"] = expected_aff["local_bytes"] + facts["local"] * delta
        elif key[1] == 0:
            b0, b1 = facts["read_rows"]
            expected_aff["hbm0_bytes"] = expected_aff["hbm0_bytes"] + b0 * delta
            expected_aff["hbm1_bytes"] = expected_aff["hbm1_bytes"] + b1 * delta
    for name, expected_value in expected_aff.items():
        if half_ledger[name] != expected_value:
            raise AssertionError(
                f"one-output ledger boundary-corrected conservation failed for {name}: "
                f"{half_ledger[name]} != {expected_value}")
    if half_life[0].next_load != life_start[0].next_load + k_tiles or \
            half_life[1].next_load != life_start[1].next_load + k_tiles:
        raise AssertionError("one-output lifecycle cursor did not advance K_tiles for both owners")
    if half_life[0].retired - life_start[0].retired + half_life[1].retired - life_start[1].retired != 2:
        raise AssertionError("one-output segment must retire exactly one output per owner")
    if not any(event.activities and any(a.key[1] == 2 for a in event.activities.values())
               for group in groups[:k_tiles] for event in group):
        raise AssertionError("output segment has no store activity")

    dlo, dhi = composer.domain
    if dlo is None or dhi is None or not (dlo <= fixed <= x0 <= dhi):
        raise AssertionError("strict category event-order domain does not contain the invariant orbit")
    if (composer2.domain[0] is None or composer2.domain[1] is None or
            not (composer2.domain[0] <= fixed <= x0 <= composer2.domain[1])):
        raise AssertionError("second output category order fails on the invariant orbit")

    # The full two-output map is the square of this one-output map.
    if a_half * a_half != a_full_saved or b_half * (1 + a_half) != b_full_saved:
        raise AssertionError("one-output map does not square to the saved two-output map")

    # Full-M/N tail bound, using the actual cursor and parsed integer geometry.
    # Row-major OS output index is mt*Ntiles+nt. The first non-full tile is the
    # earliest M or N tail category; no wave list is constructed.
    mtiles, ntiles = frozen.M_TILES, frozen.N_TILES
    m_has_tail = frozen.FIXTURE_M % frozen.TM != 0
    n_has_tail = frozen.FIXTURE_N % frozen.TN != 0
    nonfull_candidates = []
    if n_has_tail:
        nonfull_candidates.append(ntiles - 1)
    if m_has_tail:
        nonfull_candidates.append((mtiles - 1) * ntiles)
    first_nonfull_output = min(nonfull_candidates) if nonfull_candidates else None
    first_nonfull_wave = first_nonfull_output * k_tiles if first_nonfull_output is not None else frozen.WAVE_COUNT
    initial_cursor0, initial_cursor1 = life_start[0].next_load, life_start[1].next_load
    tail_half_bound_owner = [max(0, (first_nonfull_wave - cursor) // k_tiles)
                             for cursor in (initial_cursor0, initial_cursor1)]
    tail_half_bound = min(tail_half_bound_owner)
    if tail_half_bound != 995:
        raise AssertionError(f"unexpected full-category outer bound {tail_half_bound}")
    if initial_cursor0 + tail_half_bound * k_tiles > first_nonfull_wave:
        raise AssertionError("derived tail bound crosses the first non-full wave")
    if initial_cursor0 + (tail_half_bound + 1) * k_tiles <= first_nonfull_wave:
        raise AssertionError("tail bound is not maximal for owner0")

    # Fixed-width repeated transform for state and every additive ledger. The
    # exact Fraction path is used only for a small-count spot check.
    metrics = tuple((half_ledger[name].a, half_ledger[name].b) for name in LEDGERS)
    fixed_transform = transform_from_exact(a_half, b_half, metrics)
    small_n = 123
    small_x_exact, small_y_exact = exact_transform_power(a_half, b_half, half_ledger, x0, small_n)
    small_power = power_transform(fixed_transform, small_n)
    small_x_i, small_y_i = apply_transform(small_power, Interval.from_fraction(x0))
    if not small_x_i.contains(small_x_exact):
        raise AssertionError("fixed-width x interval misses the exact small-count recurrence")
    for i, name in enumerate(LEDGERS):
        if not small_y_i[i].contains(small_y_exact[name]):
            raise AssertionError(f"fixed-width ledger interval misses exact {name}")

    full_power = power_transform(fixed_transform, tail_half_bound)
    tail_x_i, tail_y_i = apply_transform(full_power, Interval.from_fraction(x0))
    for i, name in enumerate(LEDGERS):
        if max(abs(full_power.c[i].lo).bit_length(), abs(full_power.c[i].hi).bit_length(),
               abs(full_power.d[i].lo).bit_length(), abs(full_power.d[i].hi).bit_length()) > 256:
            raise AssertionError("fixed-width affine powering exceeded the declared 256-bit scratch bound")

    # Pick an external arrival strictly between output segment 123 and 124.
    # The bounded search must certify 123 complete output segments and leave
    # the residual interval to the event-level integrator.
    exact123_x, exact123_y = exact_transform_power(a_half, b_half, half_ledger, x0, 123)
    _, exact124_y = exact_transform_power(a_half, b_half, half_ledger, x0, 124)
    stop_midpoint = (exact123_y["time_s"] + exact124_y["time_s"]) / 2
    stop_ns = ceil_div(stop_midpoint.numerator * 1_000_000_000,
                       stop_midpoint.denominator)
    stop = F(stop_ns, 1_000_000_000)
    if not exact123_y["time_s"] < stop < exact124_y["time_s"]:
        raise AssertionError("integer-nanosecond external stop is not strictly between segments 123 and 124")
    external_count, external_certificate = max_safe_repeats(
        fixed_transform, x0, stop, LEDGERS.index("time_s"), tail_half_bound)
    if external_count != 123:
        raise AssertionError(f"external stop bound found {external_count}, expected 123")

    group_types = [
        {"name": "K0_start", "representative_group": 0, "repeat_count": 1},
        {"name": "K_middle_identity", "representative_group": 1,
         "repeat_count": middle_count, "state_transform": "identity"},
        *[{"name": f"K_output_boundary_{g}", "representative_group": g,
           "repeat_count": 1} for g in range(1 + middle_count, k_tiles)],
    ]
    if len(group_types) != 8:
        raise AssertionError("bounded K category table grew beyond eight classes")

    result = {
        "status": "PASS",
        "qualification": (
            "Nested exact affine composition for only the saved strict 1-Core OS FP32 GEMM, "
            "grouped LOCAL-Q allocator, offset 1/20, and full M/N category. It is not an "
            "arbitrary-offset or production Runtime certificate."
        ),
        "source": {
            "event_trace": "die/docs/analysis/bounded-lifecycle-20261014/output-cycle-v2/affine-trace-complete-cycle.txt",
            "affine_reference": "die/docs/analysis/bounded-lifecycle-20261014/output-cycle-v2/affine-cycle-proof.json",
            "geometry": {"M": frozen.FIXTURE_M, "N": frozen.FIXTURE_N, "K": frozen.FIXTURE_K,
                         "Tm": frozen.TM, "Tn": frozen.TN, "Tk": frozen.TK,
                         "Mtiles": frozen.M_TILES, "Ntiles": frozen.N_TILES,
                         "Ktiles": k_tiles},
            "capacities": {"COMPUTE_GATE": "1", "HBM0_Bps": "50000000",
                           "HBM1_Bps": "50000000"},
        },
        "nested_category_representation": {
            "source_intervals": 520,
            "source_K_macros": 130,
            "event_intervals_per_K_macro": 4,
            "K_macros_per_output_segment": k_tiles,
            "output_segments_per_joint_two_output_cycle": 2,
            "category_classes_per_output": group_types,
            "unique_event_templates_per_output": 8,
            "unique_event_steps_per_output": 32,
            "category_template_steps_for_full_saved_cycle": 64,
            "flat_interval_to_template_event_ratio": "520/64=8.125",
            "middle_run_formula": "Ktiles - phase_offset - 3",
            "phase_offset_K_waves": phase_offset,
            "middle_run_count": middle_count,
            "middle_state_map_exact_identity": True,
            "middle_lifecycle_transform": "shift every active/queued wave and next_load by +1; retired unchanged",
            "representative_category_signatures_match_all_source_positions": True,
            "repeated_templates_independently_re_solved_grouped_LP_events": 32,
            "no_full_wave_or_per_output_array_materialized": True,
        },
        "one_output_affine_transform": {
            "phase_coordinate": "owner1 C progress at output-segment boundary",
            "x_next": aff_json(x_half),
            "fixed_point": fstr(half_fixed),
            "invariant_boundary_state_lines": {
                "owner0_C": aff_json(u_line), "owner1_L": aff_json(v_line)},
            "lifecycle_shift_per_output_per_owner": k_tiles,
            "retired_outputs_per_output_segment": [
                half_life[i].retired - life_start[i].retired for i in range(2)],
            "ledger_scope": {
                "compute_ops": "C-stage vec_ops only, from each activity's own progress",
                "compute_gate_s": "C-stage vec_ops / 158216000 ops/s",
                "local_bytes": "C-stage mapped LOCAL bytes; retained in canonical Q but not a fourth LP capacity",
                "hbm_bytes": "L/S physical payload bytes split across HBM0/HBM1 rows",
                "busy": "positive-rate union per COMPUTE_GATE, HBM0, HBM1, and HBM0-or-HBM1",
            },
            "ledger_maps": {name: aff_json(half_ledger[name]) for name in LEDGERS},
            "checked_event_order_domain": {"lo": fstr(dlo), "hi": fstr(dhi)},
            "orbit_interval": {"lo": fstr(fixed), "hi": fstr(x0)},
            "all_selected_dt_positive_and_all_nonselected_margins_strict": True,
            "minimum_selected_dt_over_orbit": fstr(composer.min_dt),
            "minimum_nonselected_margin_over_orbit": fstr(composer.min_margin),
            "active_progress_within_unit_interval": [fstr(composer.min_progress),
                                                       fstr(composer.max_progress)],
        },
        "full_two_output_cross_check": {
            "nested_map": aff_json(x_full),
            "saved_flat_520_map": {"slope": fstr(a_full_saved), "intercept": fstr(b_full_saved)},
            "map_equal": True,
            "a_half_squared_equals_a_full": True,
            "two_half_ledgers_reproduce_saved_cycle_time": True,
            "one_half_symbolic_event_templates_executed": first_half_micro_events,
            "second_half_symbolic_event_templates_executed": second_half_micro_events,
            "cycle_ledger_maps": {name: aff_json(full_ledger[name]) for name in LEDGERS},
            "one_output_integral_at_initial_phase": {name: fstr(value)
                                                       for name, value in half_totals.items()},
            "one_output_conservation_expected_at_initial_phase": {
                name: fstr(aff_at(value, x0)) for name, value in expected_aff.items()},
            "boundary_corrected_conservation_maps": {
                name: aff_json(value) for name, value in expected_aff.items()},
        },
        "repeat_and_stop_bounds": {
            "first_nonfull_output_tile": first_nonfull_output,
            "first_nonfull_wave": first_nonfull_wave,
            "initial_next_load_by_owner": [initial_cursor0, initial_cursor1],
            "full_category_output_segment_bound_by_owner": tail_half_bound_owner,
            "max_whole_output_segments_before_M_or_N_tail": tail_half_bound,
            "corresponding_complete_two_output_cycles": tail_half_bound // 2,
            "one_residual_full_output_segment_after_saved_497_cycles": tail_half_bound % 2,
            "tail_boundary_is_not_crossed": True,
            "external_stop_test": {"expected_and_certified_whole_segments": 123,
                                   "stop_ns": stop_ns,
                                   "certificate": external_certificate},
            "external_stop_semantics": (
                "skip only complete affine output segments whose outward-rounded cumulative time "
                "upper bound is <= the stop; leave the positive residual to the ordinary event "
                "integrator and then apply completion/cancel/external-event ordering"
            ),
        },
        "fixed_width_numeric_check": {
            "format": "128-bit fractional outward-rounded interval endpoints; checked <=256-bit stored words and <=512-bit product scratch",
            "repeat_power_algorithm": "binary composition of the 3x3 state-plus-ledger affine transform",
            "small_repeat_exact_crosscheck_count": small_n,
            "small_repeat_state_interval": small_x_i.as_json(),
            "tail_bound_repeat_count": tail_half_bound,
            "tail_state_interval": tail_x_i.as_json(),
            "tail_ledger_intervals": {name: tail_y_i[i].as_json()
                                       for i, name in enumerate(LEDGERS)},
            "maximum_fixed_point_endpoint_bits": MAX_ENDPOINT_BITS_SEEN,
            "maximum_product_scratch_bits": MAX_PRODUCT_BITS_SEEN,
            "stored_endpoint_bit_limit": MAX_STORED_BITS,
            "multiplication_product_bit_limit": MAX_PRODUCT_BITS,
            "no_epsilon_cycle_equality_or_unbounded_integer_powering": True,
        },
        "remaining_limits": [
            "Only offset 1/20 and one 1-Core full-M/N OS region are covered.",
            "The event-order templates come from the existing saved trace; no general bounded discovery algorithm is claimed.",
            "Other offsets, multi-Core allocation bases, WR group fences, capacity changes, and cancellation require new category maps.",
            "A stop inside an output segment requires the ordinary local event solver to integrate the residual; the affine skip stops at the last certified segment boundary.",
            "The 128-bit interval prototype is a research evaluator, not production code or a GMP substitute certificate for unvalidated templates.",
        ],
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    descriptor = [
        "version=1",
        "state_dimension=1",
        f"ledger_count={len(LEDGERS)}",
        f"event_template_steps={len(representatives) * 4}",
        f"tail_half_bound={tail_half_bound}",
        f"first_nonfull_wave={first_nonfull_wave}",
        f"k_tiles={k_tiles}",
        f"initial_cursor_owner0={initial_cursor0}",
        f"initial_cursor_owner1={initial_cursor1}",
        f"external_stop_ns={stop_ns}",
        f"x0={fstr(x0)}",
        f"a={fstr(a_half)}",
        f"b={fstr(b_half)}",
        f"fixed={fstr(half_fixed)}",
        f"domain_lo={fstr(dlo)}",
        f"domain_hi={fstr(dhi)}",
    ]
    for name in LEDGERS:
        descriptor.extend((f"ledger.{name}.slope={fstr(half_ledger[name].a)}",
                           f"ledger.{name}.intercept={fstr(half_ledger[name].b)}"))
    OUT_AFFINE.write_text("\n".join(descriptor) + "\n")
    report = render_report(result)
    OUT_MD.write_text(report)
    print(json.dumps(result, indent=2, sort_keys=True))


def render_report(result: dict[str, Any]) -> str:
    rep = result["nested_category_representation"]
    one = result["one_output_affine_transform"]
    stop = result["repeat_and_stop_bounds"]
    fixed = result["fixed_width_numeric_check"]
    lines = [
        "# Nested K/output affine composition (scratch result)", "",
        "This verifier covers only the saved strict 1-Core OS FP32 GEMM trace at offset `1/20`, with grouped LOCAL-Q allocation and full M/N tiles. It does not certify production or general offsets.", "",
        "The 520 intervals reduce to two identical 65-K output segments. Each K macro contains four event intervals. Per output, eight category templates cover K-first, 58 repeated middle-K steps, and six K-final/output boundary steps; the middle map is exact state identity. One output segment therefore needs 32 event-template steps, and the complete saved cycle needs 64 category-template steps plus an affine composition. No wave array is created.", "",
        f"The per-output boundary map is `x' = {one['x_next']['slope']} x + {one['x_next']['intercept']}` with fixed point `{one['fixed_point']}`. The template runner checks every selected deadline tie, strictly positive event duration, all nonselected deadline margins, progress bounds, grouped LP rates/Q, and the bounded two-input/two-output lifecycle. Squaring this map exactly reproduces the archived 520-event affine map and its elapsed-time ledger. LOCAL bytes follow the saved C-stage work/Q ledger and are not treated as a fourth physical LP capacity; busy time is integrated as a positive-rate union by domain.", "",
        f"The exact event-order domain is `[{one['checked_event_order_domain']['lo']}, {one['checked_event_order_domain']['hi']}]`, containing the contraction orbit `[{one['orbit_interval']['lo']}, {one['orbit_interval']['hi']}]`. The smallest selected interval is `{one['minimum_selected_dt_over_orbit']}` s and the smallest nonselected margin is `{one['minimum_nonselected_margin_over_orbit']}` s.", "",
        f"The earliest full M/N category boundary is wave `{stop['first_nonfull_wave']}`. Starting cursors `[197,136]` allow `{stop['max_whole_output_segments_before_M_or_N_tail']}` complete one-output segments (equivalent to {stop['corresponding_complete_two_output_cycles']} full two-output cycles plus one segment) before the tail. One more segment crosses the owner-0 tail boundary.", "",
        f"The repeat evaluator uses outward-rounded Q128 intervals and binary composition of a fixed 3×3 state/ledger map. It certifies the external stop after `{stop['external_stop_test']['expected_and_certified_whole_segments']}` complete segments; the next segment's lower elapsed-time bound exceeds the stop. The 995-segment tail-bound state interval has width `{fixed['tail_state_interval']['width']}`. Stored endpoints are checked at 256 bits; multiplication scratch is checked at 512 bits.", "",
        "Remaining scope: the event templates were selected from the archived trace and this does not prove a bounded event-order discovery algorithm. The result is limited to offset `1/20`, one Core, full M/N, and the listed three-resource LP. Other offsets, tails, multi-Core bases, WR fences, cancellation, and capacity changes need separate maps. A stop inside a segment leaves a residual for the ordinary event integrator.", "",
        "Reproduce with:", "",
        "```sh", "python3 /workspace/template/work/affine-compose/nested_k_output.py", "```", "",
        "The script writes `nested-k-output.json`, `nested-k-output.affine`, and this report beside itself. It reads archived evidence without modifying it.", "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
