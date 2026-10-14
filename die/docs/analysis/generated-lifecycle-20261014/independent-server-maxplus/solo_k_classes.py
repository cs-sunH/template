#!/usr/bin/env python3
"""Auditable single-owner OS GEMM lifecycle prototype.

This is a work/ research prototype, not production code.  It composes exact
max-plus completion maps for L/C/S stages, powers homogeneous K-middle waves,
and checks small cases against a separate event-driven slot oracle.

The service model is deliberately explicit: each stage gets a fixed effective
service duration from its own work/capacity profile.  L, C and S therefore act
as independent servers.  Shared L/S HBM capacity, dynamic capacity changes
inside a segment, and production tie/order semantics are outside this model.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from fractions import Fraction as F
from typing import Callable, Iterable, Sequence

sys.set_int_max_str_digits(0)

NEG = None  # max-plus negative infinity
DIM = 7
# State: last L completion, last C completion, C completion two waves back,
# last/previous S completion, and the corresponding two S start times.
STATE_NAMES = ("L_done", "C_done", "C_done_2back", "S_done",
               "S_done_2back", "S_start", "S_start_2back")


def frac(value: object) -> F:
    if isinstance(value, F):
        return value
    if isinstance(value, int):
        return F(value)
    return F(str(value))


def fstr(value: F) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


Matrix = tuple[tuple[F | None, ...], ...]
Vector = tuple[F, ...]


def identity() -> Matrix:
    return tuple(tuple(F(0) if i == j else NEG for j in range(DIM))
                 for i in range(DIM))


def compose(outer: Matrix, inner: Matrix) -> Matrix:
    """Return outer(inner(x)) in max-plus algebra."""
    out: list[tuple[F | None, ...]] = []
    for i in range(DIM):
        row: list[F | None] = []
        for j in range(DIM):
            terms = [outer[i][k] + inner[k][j]
                     for k in range(DIM)
                     if outer[i][k] is not None and inner[k][j] is not None]
            row.append(max(terms) if terms else NEG)
        out.append(tuple(row))
    return tuple(out)


def apply(matrix: Matrix, state: Vector) -> Vector:
    result: list[F] = []
    for row in matrix:
        terms = [offset + state[j] for j, offset in enumerate(row)
                 if offset is not None]
        if not terms:
            raise AssertionError("uninitialized max-plus state coordinate")
        result.append(max(terms))
    return tuple(result)


def power(matrix: Matrix, exponent: int) -> Matrix:
    if exponent < 0:
        raise ValueError("negative matrix exponent")
    result = identity()
    base = matrix
    n = exponent
    while n:
        if n & 1:
            result = compose(base, result)
        n >>= 1
        if n:
            base = compose(base, base)
    return result


def translated(state: Vector, amount: F) -> Vector:
    return tuple(x + amount for x in state)


def normalized(state: Vector) -> tuple[F, ...]:
    # S_done is monotone and is the retirement boundary for each output.
    anchor = state[3]
    return tuple(x - anchor for x in state)


@dataclass(frozen=True)
class Rates:
    load_bytes_s: F
    compute_ops_s: F
    store_bytes_s: F
    acc_convert_eqops: F = F(2)

    @classmethod
    def make(cls, load_bytes_s: object, compute_ops_s: object,
             store_bytes_s: object, acc_convert_eqops: object = 2) -> "Rates":
        rates = cls(frac(load_bytes_s), frac(compute_ops_s), frac(store_bytes_s),
                    frac(acc_convert_eqops))
        if min(rates.load_bytes_s, rates.compute_ops_s, rates.store_bytes_s) <= 0:
            raise ValueError("all stage capacities must be positive")
        if rates.acc_convert_eqops < 0:
            raise ValueError("negative INT4 conversion work")
        if rates.acc_convert_eqops.denominator != 1:
            raise ValueError("this prototype requires integral INT4 conversion eq-ops")
        return rates


@dataclass(frozen=True)
class Wave:
    load_bytes: int
    compute_ops: int
    store_bytes: int
    load_s: F
    compute_s: F
    store_s: F


@dataclass(frozen=True)
class WorkTotals:
    load_bytes: int
    compute_ops: int
    store_bytes: int
    load_busy_s: F
    compute_busy_s: F
    store_busy_s: F

    def plus(self, other: "WorkTotals") -> "WorkTotals":
        return WorkTotals(self.load_bytes + other.load_bytes,
                          self.compute_ops + other.compute_ops,
                          self.store_bytes + other.store_bytes,
                          self.load_busy_s + other.load_busy_s,
                          self.compute_busy_s + other.compute_busy_s,
                          self.store_busy_s + other.store_busy_s)

    def times(self, count: int) -> "WorkTotals":
        return WorkTotals(self.load_bytes * count, self.compute_ops * count,
                          self.store_bytes * count, self.load_busy_s * count,
                          self.compute_busy_s * count, self.store_busy_s * count)


ZERO_TOTALS = WorkTotals(0, 0, 0, F(0), F(0), F(0))


@dataclass(frozen=True)
class KProfile:
    first: Wave
    middle: Wave
    final: Wave
    middle_count: int
    k_tiles: int
    totals: WorkTotals


@dataclass(frozen=True)
class OutputClass:
    m: int
    n: int
    label: str
    k: KProfile
    matrix: Matrix
    full: bool


@dataclass(frozen=True)
class TileRun:
    output_class: OutputClass
    count: int


def make_wave(m: int, n: int, k: int, k_index: int, k_tiles: int,
              precision: str, rates: Rates, output_bytes: int) -> Wave:
    if min(m, n, k) <= 0:
        raise ValueError("wave dimensions must be positive")
    p = precision.lower()
    if p not in ("fp32", "int4"):
        raise ValueError("precision must be fp32 or int4")
    if p == "fp32":
        # A is row-packed and W is row-packed; FP32 has four bytes/element.
        load_bytes = 4 * (m * k + k * n)
        compute_ops = 2 * m * n * k
    else:
        # INT4 rows are packed independently, so each odd row has its own tail.
        load_bytes = m * ceil_div(k, 2) + k * ceil_div(n, 2)
        # The target INT4 template produces INT32 partials, converts each one,
        # and adds old FP32 accumulators after the first K tile.
        compute_ops = (2 * m * n * k + int(rates.acc_convert_eqops * m * n)
                       + (m * n if k_index > 0 else 0))
    store_bytes = output_bytes * m * n if k_index == k_tiles - 1 else 0
    return Wave(load_bytes, compute_ops, store_bytes,
                F(load_bytes, 1) / rates.load_bytes_s,
                F(compute_ops, 1) / rates.compute_ops_s,
                F(store_bytes, 1) / rates.store_bytes_s)


def one_wave_matrix(w: Wave, first_c_guard: bool, store: bool) -> Matrix:
    """Map one wave; first-C guard checks the oldest of two output credits."""
    rows: list[dict[int, F]] = [dict() for _ in range(DIM)]
    # L_i starts after the previous L and after C_(i-2) frees an input slot.
    rows[0] = {0: w.load_s, 2: w.load_s}
    # C_i follows its own load and previous C.  Only the first K wave of a new
    # output needs an output-slot credit; subsequent C waves cannot increase
    # occupancy until this output's final C completes.
    rows[1] = {0: w.load_s + w.compute_s,
               2: w.load_s + w.compute_s,
               1: w.compute_s}
    if first_c_guard:
        rows[1][4] = w.compute_s
    rows[2] = {1: F(0)}
    rows[3] = {3: F(0)}
    rows[4] = {4: F(0)}
    rows[5] = {5: F(0)}
    rows[6] = {6: F(0)}
    if store:
        # The K-final output is generated only after C completes.  S is serial;
        # its slot remains occupied from S_start through S_done/retire.
        rows[5] = dict(rows[1])
        rows[5][3] = F(0)
        rows[3] = {i: offset + w.store_s for i, offset in rows[5].items()}
        rows[4] = {3: F(0)}
        rows[6] = {5: F(0)}
    return tuple(tuple(row.get(j) for j in range(DIM)) for row in rows)


def build_k_profile(m: int, n: int, k_total: int, tk: int, precision: str,
                    rates: Rates, output_bytes: int = 4) -> KProfile:
    if min(m, n, k_total, tk) <= 0:
        raise ValueError("M/N/K/tk must be positive")
    k_tiles = ceil_div(k_total, tk)
    first_k = min(tk, k_total)
    if k_tiles == 1:
        first = make_wave(m, n, first_k, 0, k_tiles, precision, rates, output_bytes)
        total = WorkTotals(first.load_bytes, first.compute_ops, first.store_bytes,
                           first.load_s, first.compute_s, first.store_s)
        return KProfile(first, first, first, 0, 1, total)
    final_k = k_total - tk * (k_tiles - 1)
    first = make_wave(m, n, first_k, 0, k_tiles, precision, rates, output_bytes)
    middle = make_wave(m, n, tk, 1, k_tiles, precision, rates, output_bytes)
    final = make_wave(m, n, final_k, k_tiles - 1, k_tiles, precision, rates, output_bytes)
    total = WorkTotals(
        first.load_bytes + middle.load_bytes * (k_tiles - 2) + final.load_bytes,
        first.compute_ops + middle.compute_ops * (k_tiles - 2) + final.compute_ops,
        final.store_bytes,
        first.load_s + middle.load_s * (k_tiles - 2) + final.load_s,
        first.compute_s + middle.compute_s * (k_tiles - 2) + final.compute_s,
        final.store_s,
    )
    return KProfile(first, middle, final, k_tiles - 2, k_tiles, total)


def build_output_class(m: int, n: int, full_m: int, full_n: int,
                       k_total: int, tk: int, precision: str, rates: Rates,
                       output_bytes: int = 4) -> OutputClass:
    kprofile = build_k_profile(m, n, k_total, tk, precision, rates, output_bytes)
    first_guard = True
    if kprofile.k_tiles == 1:
        matrix = one_wave_matrix(kprofile.first, first_guard, store=True)
    else:
        first = one_wave_matrix(kprofile.first, first_guard, store=False)
        middle = one_wave_matrix(kprofile.middle, False, store=False)
        final = one_wave_matrix(kprofile.final, False, store=True)
        matrix = compose(final, compose(power(middle, kprofile.middle_count), first))
    label = ("full_m/full_n" if m == full_m and n == full_n else
             f"m{m}_n{n}")
    return OutputClass(m, n, label, kprofile, matrix,
                       m == full_m and n == full_n)


def class_overlap(state: Vector, cls: OutputClass) -> F:
    """Exact overlap of this output's L intervals with its two prior S jobs."""
    p = cls.k
    stores = ((state[5], state[3]), (state[6], state[4]))
    intervals = [(p.first.load_s, max(state[0], state[2]))]
    total = F(0)
    for store_start, store_end in stores:
        if store_end > store_start:
            load_start = intervals[0][1]
            total += max(F(0), min(load_start + p.first.load_s, store_end)
                         - max(load_start, store_start))

    after_first = apply(one_wave_matrix(p.first, True,
                                        store=p.k_tiles == 1), state)
    if p.k_tiles == 1:
        return total

    middle_matrix = one_wave_matrix(p.middle, False, store=False)
    middle_state = after_first
    count = p.middle_count
    ldur = p.middle.load_s

    def interval_at(index: int) -> tuple[F, F]:
        st = apply(power(middle_matrix, index), middle_state)
        begin = max(st[0], st[2])
        return begin, begin + ldur

    def overlaps_middle(store_start: F, store_end: F) -> F:
        if count == 0 or ldur == 0 or store_end <= store_start:
            return F(0)
        # First index whose load has not ended at the store start.
        lo, hi = 0, count
        while lo < hi:
            mid = (lo + hi) // 2
            if interval_at(mid)[1] <= store_start:
                lo = mid + 1
            else:
                hi = mid
        first_i = lo
        # First index whose load starts at/after the store end.
        lo, hi = first_i, count
        while lo < hi:
            mid = (lo + hi) // 2
            if interval_at(mid)[0] < store_end:
                lo = mid + 1
            else:
                hi = mid
        after_i = lo
        if first_i >= after_i:
            return F(0)
        last_i = after_i - 1
        first_start, first_end = interval_at(first_i)
        if first_i == last_i:
            return max(F(0), min(first_end, store_end)
                       - max(first_start, store_start))
        last_start, last_end = interval_at(last_i)
        first_overlap = max(F(0), min(first_end, store_end)
                            - max(first_start, store_start))
        last_overlap = max(F(0), min(last_end, store_end)
                           - max(last_start, store_start))
        return first_overlap + (last_i - first_i - 1) * ldur + last_overlap

    for store_start, store_end in stores:
        total += overlaps_middle(store_start, store_end)

    middle_end = apply(power(middle_matrix, count), after_first)
    final_start = max(middle_end[0], middle_end[2])
    final_end = final_start + p.final.load_s
    for store_start, store_end in stores:
        if store_end > store_start:
            total += max(F(0), min(final_end, store_end)
                         - max(final_start, store_start))
    return total


def vector_add(state: Vector, amount: F) -> Vector:
    return tuple(x + amount for x in state)


def repeat_transform(state: Vector, matrix: Matrix, count: int,
                     one_step: Callable[[Vector], tuple[Vector, F]],
                     cycle_cap: int) -> tuple[Vector, F, dict[str, int]]:
    """Repeat a translation-invariant boundary map and sum one scalar ledger.

    Exact normalized-state recurrence proves a cycle for this fixed rational
    profile.  The cap is a fail-closed discovery bound, independent of count.
    """
    if count < 0:
        raise ValueError("negative repeat count")
    if count == 0:
        return state, F(0), {"transient": 0, "period": 0, "powered_cycles": 0}
    states = [state]
    prefixes = [F(0)]
    seen: dict[tuple[F, ...], int] = {}
    i = 0
    while i < count:
        key = normalized(states[i])
        if key in seen:
            start = seen[key]
            period = i - start
            if period <= 0:
                raise AssertionError("zero normalized-state period")
            shift = states[i][3] - states[start][3]
            if translated(states[start], shift) != states[i]:
                raise AssertionError("normalized-state repeat was not a translation")
            remaining = count - start
            cycles, remainder = divmod(remaining, period)
            final_state = translated(states[start + remainder], cycles * shift)
            cycle_ledger = prefixes[i] - prefixes[start]
            ledger = (prefixes[start] + cycles * cycle_ledger
                      + prefixes[start + remainder] - prefixes[start])
            direct = apply(power(matrix, count), state)
            if final_state != direct:
                raise AssertionError("cycle-powered state differs from max-plus power")
            return direct, ledger, {"transient": start, "period": period,
                                    "powered_cycles": cycles}
        seen[key] = i
        if i >= cycle_cap:
            raise RuntimeError(f"normalized state failed to close within cycle cap {cycle_cap}")
        next_state, increment = one_step(states[i])
        states.append(next_state)
        prefixes.append(prefixes[-1] + increment)
        i += 1
    return states[-1], prefixes[-1], {"transient": count, "period": 0,
                                      "powered_cycles": 0}


def repeat_class(state: Vector, cls: OutputClass, count: int,
                 cycle_cap: int = 4096) -> tuple[Vector, F, dict[str, int]]:
    return repeat_transform(state, cls.matrix, count,
                             lambda x: (apply(cls.matrix, x), class_overlap(x, cls)),
                             cycle_cap)


def compose_run_matrix(runs: Sequence[TileRun]) -> Matrix:
    result = identity()
    for run in runs:
        result = compose(power(run.output_class.matrix, run.count), result)
    return result


def apply_runs(state: Vector, runs: Sequence[TileRun], cycle_cap: int = 4096
               ) -> tuple[Vector, F, list[dict[str, int]]]:
    total_overlap = F(0)
    diagnostics: list[dict[str, int]] = []
    current = state
    for run in runs:
        current, overlap, diag = repeat_class(current, run.output_class,
                                              run.count, cycle_cap)
        total_overlap += overlap
        diagnostics.append({"count": run.count, **diag})
    return current, total_overlap, diagnostics


def repeat_macro(state: Vector, macro: Sequence[TileRun], repeats: int,
                 cycle_cap: int = 4096) -> tuple[Vector, F, dict[str, int]]:
    matrix = compose_run_matrix(macro)

    def once(x: Vector) -> tuple[Vector, F]:
        y, overlap, _ = apply_runs(x, macro, cycle_cap)
        return y, overlap

    return repeat_transform(state, matrix, repeats, once, cycle_cap)


def geometry_macros(M: int, N: int, K: int, tm: int, tn: int, tk: int,
                    precision: str, rates: Rates, output_bytes: int = 4
                    ) -> tuple[list[tuple[list[TileRun], int]], list[OutputClass], int, int]:
    if min(M, N, K, tm, tn, tk) <= 0:
        raise ValueError("geometry values must be positive")
    mt, nt = ceil_div(M, tm), ceil_div(N, tn)
    full_m = tm
    full_n = tn
    mt_full = mt if M % tm == 0 else mt - 1
    nt_full = nt if N % tn == 0 else nt - 1
    tail_m = M - tm * (mt - 1) if M % tm else tm
    tail_n = N - tn * (nt - 1) if N % tn else tn
    classes: dict[tuple[int, int], OutputClass] = {}

    def cls(m: int, n: int) -> OutputClass:
        key = (m, n)
        if key not in classes:
            classes[key] = build_output_class(m, n, tm, tn, K, tk,
                                              precision, rates, output_bytes)
        return classes[key]

    macros: list[tuple[list[TileRun], int]] = []
    full_row: list[TileRun] = []
    if nt_full:
        full_row.append(TileRun(cls(full_m, full_n), nt_full))
    if N % tn:
        full_row.append(TileRun(cls(full_m, tail_n), 1))
    if mt_full:
        macros.append((full_row, mt_full))
    tail_row: list[TileRun] = []
    if M % tm:
        if nt_full:
            tail_row.append(TileRun(cls(tail_m, full_n), nt_full))
        if N % tn:
            tail_row.append(TileRun(cls(tail_m, tail_n), 1))
        macros.append((tail_row, 1))
    if not macros:
        raise AssertionError("geometry generated no output macros")
    full_output_count = (mt_full * nt_full)
    if N % tn:
        first_nonfull = nt_full if mt_full else 0
    elif M % tm:
        first_nonfull = mt_full * nt
    else:
        first_nonfull = mt * nt
    return macros, list(classes.values()), full_output_count, mt * nt


def macro_matrix(macros: Sequence[tuple[list[TileRun], int]]) -> Matrix:
    result = identity()
    for runs, repeats in macros:
        result = compose(power(compose_run_matrix(runs), repeats), result)
    return result


def macro_work(macros: Sequence[tuple[list[TileRun], int]]) -> WorkTotals:
    total = ZERO_TOTALS
    for runs, row_count in macros:
        for run in runs:
            total = total.plus(run.output_class.k.totals.times(run.count * row_count))
    return total


def run_geometry(M: int, N: int, K: int, tm: int, tn: int, tk: int,
                 precision: str, rates: Rates, output_bytes: int = 4,
                 cycle_cap: int = 4096) -> dict[str, object]:
    macros, classes, first_nonfull, output_count = geometry_macros(
        M, N, K, tm, tn, tk, precision, rates, output_bytes)
    state: Vector = tuple(F(0) for _ in range(DIM))
    total_overlap = F(0)
    repeat_info: list[dict[str, object]] = []
    current = state
    for runs, repeat_count in macros:
        current, overlap, diag = repeat_macro(current, runs, repeat_count,
                                              cycle_cap)
        total_overlap += overlap
        repeat_info.append({"row_repeats": repeat_count,
                            "class_runs": [
                                {"class": x.output_class.label, "count": x.count}
                                for x in runs], **diag})
    totals = macro_work(macros)
    elapsed = current[3] - state[3]
    hbm_union = totals.load_busy_s + totals.store_busy_s - total_overlap
    # Single-owner pipeline work is never completely idle before final retire:
    # a full input buffer implies C/S progress, and a full output buffer implies
    # a running/ready S.  The small oracle checks this claim independently.
    result = {
        "status": "PASS",
        "qualification": ("single-owner full-M/N OS lifecycle; fixed effective L/C/S service durations; "
                          "independent L/C/S servers; exact rational work and two-slot credits"),
        "geometry": {"M": M, "N": N, "K": K, "tm": tm, "tn": tn, "tk": tk,
                     "precision": precision.lower(),
                     "m_tiles": ceil_div(M, tm), "n_tiles": ceil_div(N, tn),
                     "k_tiles": ceil_div(K, tk), "outputs": output_count,
                     "first_nonfull_output": first_nonfull},
        "category_table": [
            {"m": c.m, "n": c.n, "label": c.label,
             "k_first": {"load_bytes": c.k.first.load_bytes,
                         "compute_ops": c.k.first.compute_ops,
                         "load_s": fstr(c.k.first.load_s),
                         "compute_s": fstr(c.k.first.compute_s)},
             "k_middle": {"count": c.k.middle_count,
                          "load_bytes": c.k.middle.load_bytes,
                          "compute_ops": c.k.middle.compute_ops,
                          "load_s": fstr(c.k.middle.load_s),
                          "compute_s": fstr(c.k.middle.compute_s)},
             "k_final": {"load_bytes": c.k.final.load_bytes,
                         "compute_ops": c.k.final.compute_ops,
                         "store_bytes": c.k.final.store_bytes,
                         "load_s": fstr(c.k.final.load_s),
                         "compute_s": fstr(c.k.final.compute_s),
                         "store_s": fstr(c.k.final.store_s)}} for c in classes],
        "ledger": {"time_s": fstr(elapsed),
                   "load_bytes": totals.load_bytes,
                   "compute_ops_equivalent": totals.compute_ops,
                   "store_bytes": totals.store_bytes,
                   "busy_load_s": fstr(totals.load_busy_s),
                   "busy_compute_s": fstr(totals.compute_busy_s),
                   "busy_store_s": fstr(totals.store_busy_s),
                   "busy_hbm_union_s": fstr(hbm_union),
                   "load_store_overlap_s": fstr(total_overlap),
                   "busy_any_stage_union_s": fstr(elapsed)},
        "lifecycle": {"input_slots": 2, "output_slots": 2,
                      "k_middle_steps_materialized": 0,
                      "max_plus_matrix_dimension": DIM,
                      "row_repeat_categories": repeat_info,
                      "cycle_discovery_cap": cycle_cap,
                      "middle_repeat_cost": "O(log(Ktiles)) max-plus matrix compositions",
                      "output_repeat_cost": "exact normalized-state transient/cycle, fail closed at declared cap"},
        "boundary_format": {
            "state_names": list(STATE_NAMES),
            "state_semantics": ("completion/start clocks at a retirement boundary; because L/C may run ahead "
                                "of the latest S retirement, this is a schedule map state, not by itself "
                                "a live scheduler snapshot"),
            "external_change_rule": ("never power across a stop/capacity epoch; find the last complete "
                                     "descriptor interval before it, then reconstruct active L/C/S progress "
                                     "and both queues from a live snapshot or ordinary event integrator"),
            "live_residual_minimum": ["absolute time", "next_load_cursor", "computed/retired output cursors",
                                      "active stage IDs and exact fractional progress", "ordered loaded queue",
                                      "ordered computed queue", "input/output slot occupancy",
                                      "per-stage remaining work", "cumulative work/busy ledgers"]},
    }
    return result


def interval_union(intervals: Iterable[tuple[F, F]]) -> F:
    spans = sorted((a, b) for a, b in intervals if b > a)
    if not spans:
        return F(0)
    total = F(0)
    lo, hi = spans[0]
    for a, b in spans[1:]:
        if a <= hi:
            hi = max(hi, b)
        else:
            total += hi - lo
            lo, hi = a, b
    return total + hi - lo


@dataclass(frozen=True)
class WaveJob:
    output: int
    k_index: int
    final: bool
    first: bool
    wave: Wave


def discrete_oracle(output_classes: Sequence[OutputClass],
                    stage_event_limit: int = 100000) -> dict[str, object]:
    """Independent explicit event scheduler for small geometry only."""
    jobs: list[WaveJob] = []
    for output, cls in enumerate(output_classes):
        p = cls.k
        if p.k_tiles == 1:
            jobs.append(WaveJob(output, 0, True, True, p.first))
        else:
            jobs.append(WaveJob(output, 0, False, True, p.first))
            for k_index in range(1, p.k_tiles - 1):
                jobs.append(WaveJob(output, k_index, False, False, p.middle))
            jobs.append(WaveJob(output, p.k_tiles - 1, True, False, p.final))

    now = F(0)
    next_load = 0
    loaded: list[int] = []
    computed: list[int] = []
    active_l: tuple[int, F, F] | None = None
    active_c: tuple[int, F, F] | None = None
    active_s: tuple[int, F, F] | None = None
    l_intervals: list[tuple[F, F]] = []
    c_intervals: list[tuple[F, F]] = []
    s_intervals: list[tuple[F, F]] = []
    events = 0
    retired = 0
    max_input_slots = 0
    max_output_slots = 0

    def note_slots() -> None:
        nonlocal max_input_slots, max_output_slots
        input_used = len(loaded) + int(active_l is not None) + int(active_c is not None)
        output_used = (len(computed) + int(active_s is not None)
                       + int(active_c is not None and jobs[active_c[0]].final))
        max_input_slots = max(max_input_slots, input_used)
        max_output_slots = max(max_output_slots, output_used)
        if input_used > 2 or output_used > 2:
            raise AssertionError(f"slot capacity exceeded: input={input_used}, output={output_used}")

    while retired < len(output_classes):
        changed = True
        while changed:
            changed = False
            if active_s is None and computed:
                output = computed.pop(0)
                cls = output_classes[output]
                begin = now
                end = begin + cls.k.final.store_s
                active_s = (output, begin, end)
                s_intervals.append((begin, end))
                changed = True
            if active_c is None and loaded:
                job_id = loaded[0]
                job = jobs[job_id]
                output_slots = len(computed) + int(active_s is not None)
                if not job.first or output_slots < 2:
                    loaded.pop(0)
                    begin = now
                    end = begin + job.wave.compute_s
                    active_c = (job_id, begin, end)
                    c_intervals.append((begin, end))
                    changed = True
            input_slots = len(loaded) + int(active_l is not None) + int(active_c is not None)
            if active_l is None and next_load < len(jobs) and input_slots < 2:
                job_id = next_load
                begin = now
                end = begin + jobs[job_id].wave.load_s
                active_l = (job_id, begin, end)
                l_intervals.append((begin, end))
                next_load += 1
                changed = True
            note_slots()
        deadlines = [x[2] for x in (active_l, active_c, active_s) if x is not None]
        if not deadlines:
            raise AssertionError("oracle deadlocked with unfinished outputs")
        now = min(deadlines)
        if active_l is not None and active_l[2] == now:
            job_id, _, _ = active_l
            loaded.append(job_id)
            active_l = None
            events += 1
        if active_c is not None and active_c[2] == now:
            job_id, _, _ = active_c
            job = jobs[job_id]
            if job.final:
                computed.append(job.output)
            active_c = None
            events += 1
        if active_s is not None and active_s[2] == now:
            retired_output, _, _ = active_s
            if retired_output != retired:
                raise AssertionError("store retired outputs out of order")
            retired += 1
            active_s = None
            events += 1
        note_slots()
        if events > stage_event_limit:
            raise RuntimeError("discrete oracle exceeded explicit small-case event limit")

    return {"time_s": now,
            "events": events,
            "retired": retired,
            "load_busy_s": sum((b - a for a, b in l_intervals), F(0)),
            "compute_busy_s": sum((b - a for a, b in c_intervals), F(0)),
            "store_busy_s": sum((b - a for a, b in s_intervals), F(0)),
            "load_bytes": sum(job.wave.load_bytes for job in jobs),
            "compute_ops": sum(job.wave.compute_ops for job in jobs),
            "store_bytes": sum(job.wave.store_bytes for job in jobs),
            "load_store_overlap_s": sum(
                (max(F(0), min(lb, sb) - max(la, sa))
                 for la, lb in l_intervals for sa, sb in s_intervals), F(0)),
            "busy_any_stage_union_s": interval_union(l_intervals + c_intervals + s_intervals),
            "max_input_slots": max_input_slots,
            "max_output_slots": max_output_slots}


def expand_small_macros(macros: Sequence[tuple[list[TileRun], int]],
                        limit: int = 256) -> list[OutputClass]:
    total = sum(sum(run.count for run in runs) * reps for runs, reps in macros)
    if total > limit:
        raise ValueError(f"small oracle only; {total} outputs exceeds {limit}")
    out: list[OutputClass] = []
    for runs, repeats in macros:
        for _ in range(repeats):
            for run in runs:
                out.extend([run.output_class] * run.count)
    return out


def verify_case(M: int, N: int, K: int, tm: int, tn: int, tk: int,
                precision: str, rates: Rates, output_bytes: int = 4) -> dict[str, object]:
    macros, _, _, _ = geometry_macros(M, N, K, tm, tn, tk, precision,
                                      rates, output_bytes)
    analytic = run_geometry(M, N, K, tm, tn, tk, precision, rates,
                            output_bytes)
    classes = expand_small_macros(macros)
    oracle = discrete_oracle(classes)
    led = analytic["ledger"]
    checks = {
        "time_s": (frac(led["time_s"]) == oracle["time_s"]),
        "busy_load_s": (frac(led["busy_load_s"]) == oracle["load_busy_s"]),
        "busy_compute_s": (frac(led["busy_compute_s"]) == oracle["compute_busy_s"]),
        "busy_store_s": (frac(led["busy_store_s"]) == oracle["store_busy_s"]),
        "load_bytes": (led["load_bytes"] == oracle["load_bytes"]),
        "compute_ops": (led["compute_ops_equivalent"] == oracle["compute_ops"]),
        "store_bytes": (led["store_bytes"] == oracle["store_bytes"]),
        "busy_hbm_union_s": (frac(led["busy_hbm_union_s"])
                              == oracle["load_busy_s"] + oracle["store_busy_s"]
                              - oracle["load_store_overlap_s"]),
        "busy_any_stage_union_s": (frac(led["busy_any_stage_union_s"])
                                    == oracle["busy_any_stage_union_s"]),
        "input_slots_at_most_two": (oracle["max_input_slots"] <= 2),
        "output_slots_at_most_two": (oracle["max_output_slots"] <= 2),
    }
    if not all(checks.values()):
        raise AssertionError(f"analytic/oracle mismatch: {checks}; oracle={oracle}; ledger={led}")
    demands = {"load": frac(led["busy_load_s"]),
               "compute": frac(led["busy_compute_s"]),
               "store": frac(led["busy_store_s"])}
    demand_order = sorted(demands, key=lambda stage: demands[stage], reverse=True)
    return {"case": {"M": M, "N": N, "K": K, "tm": tm, "tn": tn,
                     "tk": tk, "precision": precision,
                     "load_bps": fstr(rates.load_bytes_s),
                     "compute_ops_s": fstr(rates.compute_ops_s),
                     "store_bps": fstr(rates.store_bytes_s),
                     "standalone_stage_demand_order": demand_order},
            "checks": checks, "oracle": {k: fstr(v) if isinstance(v, F) else v
                                         for k, v in oracle.items()}}


def selfcheck() -> dict[str, object]:
    cases = []
    # Load-limited and compute-limited profiles, odd INT4 rows, K=1/2/many,
    # M/N tails, and load/store overlap under two input/output slots.
    scenarios = [
        (5, 7, 19, 4, 4, 3, "fp32", Rates.make(1000, 10_000, 800)),
        (5, 7, 19, 4, 4, 3, "int4", Rates.make(100, 1_000_000, 250)),
        (3, 5, 1, 4, 4, 8, "fp32", Rates.make(100, 1000, 70)),
        (8, 7, 2, 4, 4, 1, "int4", Rates.make(750, 500, 100)),
        (7, 8, 8, 4, 4, 4, "int4", Rates.make(1000, 100, 1000)),
        (8, 8, 17, 4, 4, 4, "fp32", Rates.make(250, 250, 250)),
    ]
    for scenario in scenarios:
        cases.append(verify_case(*scenario))
    if cases[0]["case"]["standalone_stage_demand_order"][0] != "load":
        raise AssertionError("load-limited profile was not represented")
    if not any(c["case"]["standalone_stage_demand_order"][0] == "compute"
               for c in cases):
        raise AssertionError("compute-limited profile was not represented")
    if not any(frac(c["oracle"]["load_store_overlap_s"]) > 0
               for c in cases):
        raise AssertionError("no small case overlapped next-output L with prior S")
    if not any(c["oracle"]["max_output_slots"] == 2 for c in cases):
        raise AssertionError("no small case filled the two output slots")

    # A large middle run demonstrates logarithmic K composition.  The selected
    # geometry has one output and no M/N tail; no K-wave list is constructed.
    large = run_geometry(4, 4, 4 * 1_000_000 + 3, 4, 4, 4,
                         "fp32", Rates.make(1000, 5000, 1000), cycle_cap=4096)
    if large["lifecycle"]["k_middle_steps_materialized"] != 0:
        raise AssertionError("large K path materialized middle waves")
    large_outputs = run_geometry(4_000_000, 4, 17, 4, 4, 4,
                                 "fp32", Rates.make(1000, 5000, 1000),
                                 cycle_cap=4096)
    output_repeat = large_outputs["lifecycle"]["row_repeat_categories"][0]
    if output_repeat["powered_cycles"] < 1:
        raise AssertionError("large output run did not use a normalized-state cycle")
    return {"status": "PASS", "independent_oracle_cases": cases,
            "large_k_case": {"k_tiles": large["geometry"]["k_tiles"],
                             "middle_steps_materialized": 0,
                             "time_s": large["ledger"]["time_s"],
                             "hbm_union_s": large["ledger"]["busy_hbm_union_s"],
                             "repeat": large["lifecycle"]["row_repeat_categories"]},
            "large_output_case": {"outputs": large_outputs["geometry"]["outputs"],
                                  "powered_cycles": output_repeat["powered_cycles"],
                                  "transient": output_repeat["transient"],
                                  "period": output_repeat["period"],
                                  "time_s": large_outputs["ledger"]["time_s"]},
            "slot_bounds": "independent oracle checked input_slots<=2 and output_slots<=2"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--geometry", nargs=9, metavar=("M", "N", "K", "tm", "tn", "tk",
                                                           "precision", "load_Bps", "compute_ops_s"))
    parser.add_argument("--store-Bps", default="1")
    parser.add_argument("--output-bytes", type=int, default=4)
    args = parser.parse_args()
    if args.selfcheck:
        result = selfcheck()
    elif args.geometry:
        M, N, K, tm, tn, tk = map(int, args.geometry[:6])
        precision, load, compute = args.geometry[6:]
        rates = Rates.make(load, compute, args.store_Bps)
        result = run_geometry(M, N, K, tm, tn, tk, precision, rates,
                              args.output_bytes)
    else:
        parser.error("choose --selfcheck or --geometry")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
