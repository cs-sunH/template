#!/usr/bin/env python3
"""Exact, offline single-owner lifecycle model with shared HBM rows.

This is a bounded descriptor-discovery prototype, not production code.  It
uses Fraction arithmetic, a three-domain grouped-Q allocator, two input and
two output credits, and exact translation proofs for middle-K and repeated
output-row words.  No K-wave or output history is materialized.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction as F
from typing import Any, Iterable
import argparse
import json


@dataclass(frozen=True)
class GemmFP32:
    M: int
    N: int
    K: int
    tm: int = 16
    tn: int = 64
    tk: int = 64
    compute_rate: F = F(158_216_000)
    capacities: tuple[F, F, F] = (F(1), F(50_000_000), F(50_000_000))

    def __post_init__(self) -> None:
        if any(x <= 0 for x in (self.M, self.N, self.K, self.tm, self.tn,
                                self.tk, self.compute_rate, *self.capacities)):
            raise ValueError("geometry and capacities must be positive")

    @property
    def mt(self) -> int:
        return (self.M + self.tm - 1) // self.tm

    @property
    def nt(self) -> int:
        return (self.N + self.tn - 1) // self.tn

    @property
    def kt(self) -> int:
        return (self.K + self.tk - 1) // self.tk

    @property
    def outputs(self) -> int:
        return self.mt * self.nt

    @property
    def waves(self) -> int:
        return self.outputs * self.kt

    def facts(self, w: int) -> tuple[int, int, int, bool, bool, int, int, int, int]:
        if not 0 <= w < self.waves:
            raise ValueError("wave index out of range")
        q, ki = divmod(w, self.kt)
        mi, ni = divmod(q, self.nt)
        m = min(self.tm, self.M - mi * self.tm)
        n = min(self.tn, self.N - ni * self.tn)
        k = min(self.tk, self.K - ki * self.tk)
        final = ki == self.kt - 1
        first = ki == 0
        read = 4 * k * (m + n)
        write = 4 * m * n if final else 0
        ops = 2 * m * n * k
        # Root's three-domain GEMM-Q model accounts for A/B payload, one
        # accumulator write each K-wave, and one previous-accumulator read on
        # noninitial K-waves. LOCAL is a Q/ledger term, not a fourth domain.
        local = read + 4 * m * n + (4 * m * n if not first else 0)
        return m, n, k, final, first, read, write, ops, local

    def work(self, stage: int, w: int) -> tuple[tuple[F, F, F], F, F]:
        m, n, k, final, first, read, write, ops, local = self.facts(w)
        if stage == 0:  # load: one common stream split across both HBM rows
            return (F(0), F(read, 2), F(read, 2)), F(read), F(0)
        if stage == 1:  # compute: private upper rate and compute-gate work
            c = F(ops, 1) / self.compute_rate
            return (c, F(0), F(0)), F(local), c
        if stage == 2 and final:  # store only after final-K compute retires
            return (F(0), F(write, 2), F(write, 2)), F(write), F(0)
        raise ValueError("invalid store wave")

    def output_shape(self, q: int) -> tuple[int, int]:
        if not 0 <= q < self.outputs:
            raise ValueError("output tile index out of range")
        mi, ni = divmod(q, self.nt)
        return (min(self.tm, self.M - mi * self.tm),
                min(self.tn, self.N - ni * self.tn))

    def same_shape_run_end(self, q: int) -> int:
        """Exclusive end of the contiguous equal-(m,n)-shape output run."""
        m, n = self.output_shape(q)
        mi, ni = divmod(q, self.nt)
        end = self.outputs
        # The final N tile is a distinct class only if a full N tile precedes it.
        if self.N % self.tn and self.nt > 1:
            if n == self.tn:
                end = min(end, mi * self.nt + self.nt - 1)
            else:
                end = min(end, (mi + 1) * self.nt)
        # A short final M tile begins a new class at its first output.
        if self.M % self.tm and self.mt > 1:
            m_tail_start = (self.mt - 1) * self.nt
            if m == self.tm:
                end = min(end, m_tail_start)
        return max(q + 1, end)

    def expected(self) -> dict[str, int]:
        read = 4 * self.K * (self.nt * self.M + self.mt * self.N)
        write = 4 * self.M * self.N
        ops = 2 * self.M * self.N * self.K
        local = read + (2 * self.kt - 1) * write
        return {"read": read, "write": write, "ops": ops, "local": local,
                "outputs": self.outputs, "waves": self.waves}


def grouped_q_alpha(g: GemmFP32, active: dict[tuple[int, int], F]) -> dict[tuple[int, int], F]:
    """Exact weighted progressive filling for one owner's active activities.

    The resource set is COMPUTE_GATE, HBM0, HBM1.  Local bytes contribute to
    the owner's canonical-Q sum but do not form a fourth capacity.
    """
    works: dict[tuple[int, int], tuple[F, F, F]] = {}
    qs: dict[tuple[int, int], F] = {}
    privates: dict[tuple[int, int], F] = {}
    q_owner = F(0)
    for ident in active:
        work, q, private = g.work(*ident)
        works[ident], qs[ident], privates[ident] = work, q, private
        q_owner += q
    if not active or q_owner <= 0:
        raise ValueError("empty activity set or nonpositive canonical Q")
    rates = {i: F(0) for i in active}
    free = set(active)
    for _ in range(len(active)):
        if not free:
            break
        bounds: dict[tuple[str, Any], F] = {}
        for r in range(3):
            used = sum(works[i][r] * rates[i] for i in active)
            slope = sum(works[i][r] / q_owner for i in free)
            if slope:
                bounds[("r", r)] = (g.capacities[r] - used) / slope
        for i in free:
            if privates[i]:
                bounds[("p", i)] = (1 / privates[i] - rates[i]) * q_owner
        if not bounds:
            raise ValueError("unbounded progressive-fill tier")
        step = min(bounds.values())
        if step < 0:
            raise ValueError("infeasible resource state")
        for i in free:
            rates[i] += step / q_owner
        frozen: set[tuple[int, int]] = set()
        for (kind, key), value in bounds.items():
            if value != step:
                continue
            if kind == "r":
                frozen.update(i for i in free if works[i][key] > 0)
            else:
                frozen.add(key)
        if not frozen:
            raise ValueError("progressive fill did not freeze a limiter")
        free.difference_update(frozen)
    if free:
        raise ValueError("progressive-fill tier bound exhausted")
    for r in range(3):
        if sum(works[i][r] * rates[i] for i in active) > g.capacities[r]:
            raise ValueError("exact rate allocation exceeds capacity")
    return rates


def equal_progress_rates(g: GemmFP32,
                         active: dict[tuple[int, int], F]) -> dict[tuple[int, int], F]:
    """Independent-oracle policy: all active normalized progress rates tie."""
    works = {i: g.work(*i)[0] for i in active}
    privates = {i: g.work(*i)[2] for i in active}
    rates = {i: F(0) for i in active}
    free = set(active)
    for _ in range(len(active)):
        if not free:
            break
        bounds: list[tuple[str, Any, F]] = []
        for r, cap in enumerate(g.capacities):
            used = sum(works[i][r] * rates[i] for i in active)
            slope = sum(works[i][r] for i in free)
            if slope:
                bounds.append(("r", r, (cap - used) / slope))
        for i in free:
            if privates[i]:
                bounds.append(("p", i, 1 / privates[i] - rates[i]))
        if not bounds:
            raise ValueError("unbounded equal-progress tier")
        step = min(b[2] for b in bounds)
        if step < 0:
            raise ValueError("infeasible equal-progress state")
        for i in free:
            rates[i] += step
        freeze: set[tuple[int, int]] = set()
        for kind, key, value in bounds:
            if value != step:
                continue
            if kind == "r":
                freeze.update(i for i in free if works[i][key] > 0)
            else:
                freeze.add(key)
        if not freeze:
            raise ValueError("equal-progress fill did not freeze a limiter")
        free.difference_update(freeze)
    if free:
        raise ValueError("equal-progress tier bound exhausted")
    return rates


LEDGERS = ("elapsed", "vec_ops", "local_bytes", "hbm0", "hbm1",
           "compute_service", "compute_busy", "hbm0_busy", "hbm1_busy",
           "hbm_busy", "hbm_shared")


@dataclass
class Owner:
    next_load: int = 0
    retired: int = 0
    active: dict[int, tuple[int, F]] = field(default_factory=dict)  # stage -> (wave, progress)
    loaded: list[int] = field(default_factory=list)
    computed: list[int] = field(default_factory=list)
    finish: F | None = None


class Solo:
    def __init__(self, g: GemmFP32, *, cap: int = 256, accelerate: bool = True,
                 policy: str = "grouped_q_alpha"):
        self.g, self.cap, self.accelerate = g, cap, accelerate
        if policy not in ("grouped_q_alpha", "equal_progress"):
            raise ValueError("unknown policy")
        self.policy = policy
        self.o = Owner()
        self.now = F(0)
        self.ledger = {n: F(0) for n in LEDGERS}
        self.events = 0
        self.deadline_trace: list[dict[str, Any]] = []
        self.skipped_waves = 0
        self.skipped_outputs = 0
        self.descriptors: list[dict[str, Any]] = []
        self.seen: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        self.peak_input = self.peak_output = self.peak_active = 0

    def occupancy(self) -> tuple[int, int]:
        o = self.o
        inp = len(o.loaded) + int(0 in o.active) + int(1 in o.active)
        out = len(o.computed) + int(2 in o.active) + int(
            1 in o.active and self.g.facts(o.active[1][0])[3])
        return inp, out

    def settle(self) -> None:
        while True:
            changed = False
            o = self.o
            if o.finish is not None:
                return
            inp, out = self.occupancy()
            self.peak_input = max(self.peak_input, inp)
            self.peak_output = max(self.peak_output, out)
            if inp > 2 or out > 2:
                raise ValueError("credit capacity violated")
            if 0 not in o.active and o.next_load < self.g.waves and inp < 2:
                o.active[0] = (o.next_load, F(0))
                o.next_load += 1
                changed = True
            _, out = self.occupancy()
            if 1 not in o.active and o.loaded and out < 2:
                o.active[1] = (o.loaded.pop(0), F(0))
                changed = True
            if 2 not in o.active and o.computed:
                o.active[2] = (o.computed.pop(0), F(0))
                changed = True
            if (o.retired == self.g.outputs and not o.active and
                    not o.loaded and not o.computed):
                o.finish = self.now
                return
            if not changed:
                return

    def refs(self, o: Owner | None = None, *, include_next: bool = True) -> list[int]:
        o = self.o if o is None else o
        r = [w for w, _ in o.active.values()] + list(o.loaded) + list(o.computed)
        if include_next and o.next_load < self.g.waves:
            r.append(o.next_load)
        return r

    def anchor(self, o: Owner | None = None) -> int:
        o = self.o if o is None else o
        return min(self.refs(o), default=o.next_load)

    def key(self) -> tuple[Any, ...] | None:
        o, g = self.o, self.g
        if o.finish is not None:
            return None
        refs = self.refs(o)
        base = min(refs, default=o.next_load)
        def tag(w: int) -> tuple[Any, ...]:
            return w - base, g.facts(w)
        return (tuple((s, *tag(w)) for s, (w, _) in sorted(o.active.items())),
                tuple(tag(w) for w in o.loaded),
                tuple(tag(w) for w in o.computed),
                o.next_load - base, o.retired - base // g.kt,
                g.facts(o.next_load) if o.next_load < g.waves else None)

    def snapshot(self) -> dict[str, Any]:
        o = self.o
        return {"active": dict(o.active), "loaded": list(o.loaded),
                "computed": list(o.computed), "next_load": o.next_load,
                "retired": o.retired, "anchor": self.anchor(o),
                "now": self.now, "ledger": dict(self.ledger),
                "events": self.events, "trace_start": len(self.deadline_trace)}

    def output_copy_bound(self, old: dict[str, Any], d: int) -> int:
        """Bound repeats by the exact full/tail output class word.

        Supported accelerated words are a single equal-shape run (p=1 or a
        short contiguous block) and a row-aligned word (p multiple of N tiles).
        These cover repeated full tiles, N-tail-separated rows, and M-tail
        exclusion without scanning output tiles.
        """
        g, o = self.g, self.o
        if d % g.kt:
            return 0
        p = d // g.kt
        if p <= 0:
            return 0
        if o.next_load - old["next_load"] != d:
            return 0
        if o.retired - old["retired"] != p:
            return 0
        q0 = old["next_load"] // g.kt
        qcur = o.next_load // g.kt
        if qcur != q0 + p:
            return 0
        live_refs = self.refs(o)
        wave_bounds = [(g.waves - 1 - w) // d for w in live_refs
                       if w < g.waves]
        wave_bounds.append((g.waves - o.next_load) // d)
        wave_bound = max(0, min(wave_bounds, default=0))
        if wave_bound == 0:
            return 0

        # An aligned row word preserves N-full/N-tail positions. Its M-class
        # can repeat only inside the contiguous full-M or tail-M region.
        if p % g.nt == 0:
            rows_per_cycle = p // g.nt
            qlast = q0 + p - 1
            source_rows = {g.output_shape(q)[0]
                           for q in (q0, qlast)}
            if len(source_rows) != 1:
                return 0
            source_m = next(iter(source_rows))
            # Since the block is row-aligned, endpoint checks characterize
            # every M tile row inside it (only the final M row can differ).
            current_m = {g.output_shape(w // g.kt)[0]
                         for w in live_refs if w < g.waves}
            if current_m and any(m != source_m for m in current_m):
                return 0
            row_bound = wave_bound
            tail_row = g.mt - 1 if g.M % g.tm else g.mt
            if source_m != g.tm:
                # The short M-tail is a terminal class.
                return 0
            if g.M % g.tm:
                for w in live_refs:
                    if w >= g.waves:
                        continue
                    row = (w // g.kt) // g.nt
                    row_bound = min(row_bound,
                                    max(0, (tail_row - 1 - row) // rows_per_cycle))
            return min(wave_bound, row_bound)

        # For other periods, only accept a block that stays in one tile-shape
        # run. Arithmetic run ends avoid walking a potentially large block.
        if not (q0 < g.outputs and qcur < g.outputs):
            return 0
        shape = g.output_shape(q0)
        if g.same_shape_run_end(q0) < q0 + p:
            return 0
        if g.output_shape(qcur) != shape or g.same_shape_run_end(qcur) < qcur + p:
            return 0
        class_bound = wave_bound
        for w in live_refs:
            if w >= g.waves:
                continue
            q = w // g.kt
            run_end = g.same_shape_run_end(q)
            class_bound = min(class_bound,
                              max(0, (run_end - 1 - q) // p))
        # If N has a short final column, the same-shape run ends before it;
        # a subsequent aligned row-word return can resume after that tail.
        return min(wave_bound, class_bound)

    def try_skip(self, old: dict[str, Any]) -> bool:
        o, g = self.o, self.g
        old_refs = self.refs(Owner(old["next_load"], old["retired"],
                                   dict(old["active"]), list(old["loaded"]),
                                   list(old["computed"])))
        new_refs = self.refs(o)
        d = self.anchor(o) - old["anchor"]
        if d <= 0 or not old_refs or not new_refs:
            return False
        if len(old["active"]) != len(o.active) or set(old["active"]) != set(o.active):
            return False
        for s, (w0, p0) in old["active"].items():
            w1, p1 = o.active[s]
            if w1 - w0 != d or p1 != p0 or g.facts(w1) != g.facts(w0):
                return False
        if len(old["loaded"]) != len(o.loaded) or len(old["computed"]) != len(o.computed):
            return False
        if any(b - a != d for a, b in zip(old["loaded"], o.loaded)):
            return False
        if any(b - a != d for a, b in zip(old["computed"], o.computed)):
            return False
        if o.next_load - old["next_load"] != d:
            return False

        # K-middle return: one or more consecutive full K waves in the same
        # output; first/final K and all credit transitions are expanded.
        if d < g.kt:
            if any(w // g.kt != old["anchor"] // g.kt or
                   w % g.kt == 0 or w % g.kt >= g.kt - 1
                   for w in old_refs + new_refs):
                return False
            max_k = max(w % g.kt for w in old_refs + new_refs)
            count = (g.kt - 2 - max_k) // d
            if count <= 0:
                return False
            kind = "Kmiddle_identity"
            outputs = 0
            category_word = "same output tile; interior full-K waves only; k-first and K-final are explicit"
        else:
            count = self.output_copy_bound(old, d)
            if count <= 0:
                return False
            kind = "output_classword_identity"
            outputs = (d // g.kt) * count
            p = d // g.kt
            q0 = old["next_load"] // g.kt
            if p % g.nt == 0:
                category_word = (f"row-major word, {p // g.nt} output rows per period; "
                                 f"N-full={g.nt - int(bool(g.N % g.tn and g.nt > 1))}, "
                                 f"N-tail={g.N % g.tn if g.N % g.tn and g.nt > 1 else 0}; "
                                 f"M-class={g.output_shape(q0)[0]}")
            else:
                category_word = (f"constant tile class m={g.output_shape(q0)[0]}, "
                                 f"n={g.output_shape(q0)[1]} within its arithmetic run")

        # This solo prototype has a zero-dimensional exact state. Equality of
        # all live progress coordinates is the complete return guard; there
        # are no external arrival, cancellation, or changing-capacity times.
        before = old["ledger"]
        unit = {n: self.ledger[n] - before[n] for n in LEDGERS}
        deadline_template = self.deadline_trace[old["trace_start"]:]
        for n, value in unit.items():
            self.ledger[n] += value * count
        self.now = self.ledger["elapsed"]
        shift = d * count
        o.active = {s: (w + shift, p) for s, (w, p) in o.active.items()}
        o.loaded = [w + shift for w in o.loaded]
        o.computed = [w + shift for w in o.computed]
        o.next_load += shift
        if outputs:
            o.retired += outputs
        else:
            o.retired += (shift // g.kt)  # middle-K waves remain in same output
        self.skipped_waves += shift
        self.skipped_outputs += outputs
        self.descriptors.append({
            "kind": kind,
            "step_waves": d,
            "repeat_count": count,
            "step_outputs": d // g.kt if d % g.kt == 0 else 0,
            "category_word": category_word,
            "template_events": self.events - old["events"],
            "return_map": "exact identity on every live progress coordinate",
            "guard": "same active/queue topology, exact translated wave roles and work classes, two-input/two-output occupancy; solo fixed capacities and no external stop",
            "ledger_per_step": {k: str(v) for k, v in unit.items()},
            "deadline_template": deadline_template,
            "skip_waves": shift,
            "skip_outputs": outputs,
        })
        # Retain the old occurrence: a K-return can be nested in an output
        # row return, and a full-N run may lead into a row-aligned return.
        return True

    def event(self) -> None:
        if self.events >= self.cap:
            raise ValueError("global exact-event discovery cap reached")
        self.settle()
        o, g = self.o, self.g
        active = {(s, w): p for s, (w, p) in o.active.items()}
        rates = (grouped_q_alpha(g, active) if self.policy == "grouped_q_alpha"
                 else equal_progress_rates(g, active))
        self.peak_active = max(self.peak_active, len(active))
        deadlines = {(s, w): (1 - p) / rates[(s, w)]
                     for (s, w), p in active.items() if rates[(s, w)] > 0}
        if not deadlines:
            raise ValueError("deadlock: no active positive-rate stage")
        delta = min(deadlines.values())
        selected = {i for i, d in deadlines.items() if d == delta}
        base = self.anchor()
        self.deadline_trace.append({
            "delta": str(delta),
            "active": [{"stage": s, "wave_offset": w - base,
                        "progress": str(p), "rate": str(rates[(s, w)])}
                       for (s, w), p in sorted(active.items())],
            "deadline_margin": [{"stage": s, "wave_offset": w - base,
                                 "margin": str(deadlines[(s, w)] - delta)}
                                for s, w in sorted(deadlines)],
            "selected": [{"stage": s, "wave_offset": w - base}
                         for s, w in sorted(selected)],
        })
        self.ledger["elapsed"] += delta
        busy = [False, False, False]
        for (s, w), p in active.items():
            rate = rates[(s, w)]
            work, _, _ = g.work(s, w)
            inc = delta * rate
            self.o.active[s] = (w, p + inc)
            f = g.facts(w)
            if s == 1:
                self.ledger["vec_ops"] += inc * f[7]
                self.ledger["local_bytes"] += inc * f[8]
                self.ledger["compute_service"] += inc * work[0]
            else:
                self.ledger["hbm0"] += inc * work[1]
                self.ledger["hbm1"] += inc * work[2]
            for r in range(3):
                if rate > 0 and work[r] > 0:
                    busy[r] = True
        if busy[0]:
            self.ledger["compute_busy"] += delta
        if busy[1]:
            self.ledger["hbm0_busy"] += delta
        if busy[2]:
            self.ledger["hbm1_busy"] += delta
        if busy[1] or busy[2]:
            self.ledger["hbm_busy"] += delta
        self.now += delta
        # Complete tied stages L, C, S in stable order; the final-K C release
        # creates output occupancy before its K-final store can retire.
        for s, w in sorted(selected):
            del o.active[s]
            if s == 0:
                o.loaded.append(w)
            elif s == 1:
                if g.facts(w)[3]:
                    o.computed.append(w)
            else:
                o.retired += 1
        self.events += 1
        self.settle()

    def run(self) -> dict[str, Any]:
        self.settle()
        while self.o.finish is None:
            if self.accelerate:
                key = self.key()
                if key is not None:
                    history = self.seen.setdefault(key, [])
                    for old in reversed(history):
                        if self.try_skip(old):
                            break
                    history.append(self.snapshot())
                    if len(history) > 8:
                        del history[:-8]
            if self.o.finish is None:
                self.event()
        expected = self.g.expected()
        got = {"read": self.ledger["hbm0"] + self.ledger["hbm1"] - expected["write"],
               "write": self.ledger["hbm0"] + self.ledger["hbm1"] - (self.ledger["hbm0"] + self.ledger["hbm1"] - expected["write"]),
               "ops": self.ledger["vec_ops"], "local": self.ledger["local_bytes"]}
        # Exact global totals: row totals are balanced for strict FP32 payloads.
        assert self.ledger["vec_ops"] == expected["ops"]
        assert self.ledger["local_bytes"] == expected["local"]
        assert self.ledger["hbm0"] + self.ledger["hbm1"] == expected["read"] + expected["write"]
        assert self.ledger["hbm_busy"] == self.ledger["hbm0_busy"] == self.ledger["hbm1_busy"]
        assert self.o.retired == expected["outputs"]
        return {
            "status": "passed",
            "scope": "offline exact single-owner FP32 OS model; shared HBM0/HBM1 grouped-Q; not a production or hardware certificate",
            "rate_policy": self.policy,
            "geometry": {k: getattr(self.g, k) for k in ("M", "N", "K", "tm", "tn", "tk")},
            "events": self.events, "global_event_cap": self.cap,
            "skipped_waves": self.skipped_waves, "skipped_outputs": self.skipped_outputs,
            "descriptor_count": len(self.descriptors), "descriptors": self.descriptors,
            "peak_active": self.peak_active, "peak_input_slots": self.peak_input,
            "peak_output_slots": self.peak_output,
            "finish_s": str(self.o.finish),
            "ledger": {k: str(v) for k, v in self.ledger.items()},
            "expected_work": expected,
            "hbm_busy_is_both_row_union": self.ledger["hbm_busy"] == self.ledger["hbm0_busy"] == self.ledger["hbm1_busy"],
            "external_stop_boundary": "not modeled; skip is valid only while capacities, geometry, and owner admission remain unchanged",
        }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=65)
    p.add_argument("--n", type=int, default=65)
    p.add_argument("--k", type=int, default=4097)
    p.add_argument("--tm", type=int, default=16)
    p.add_argument("--tn", type=int, default=64)
    p.add_argument("--tk", type=int, default=64)
    p.add_argument("--cap", type=int, default=256)
    p.add_argument("--no-accelerate", action="store_true")
    p.add_argument("--output")
    a = p.parse_args()
    try:
        result = Solo(GemmFP32(a.m, a.n, a.k, a.tm, a.tn, a.tk),
                      cap=a.cap, accelerate=not a.no_accelerate).run()
    except (ValueError, AssertionError) as e:
        result = {"status": "failed_closed", "reason": str(e),
                  "production": "not_implemented"}
    data = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if a.output:
        with open(a.output, "w") as f:
            f.write(data)
    else:
        print(data, end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
