#!/usr/bin/env python3
"""Bounded exact one-owner lifecycle prototype driven by compiler facts.

Facts are read from the standalone C++ Runtime driver's class report. This
prototype has no byte, operation, or resource-work formula: every stage uses
the emitted 13-domain Activity facts, private bound, canonical Q, typed work,
and HBM row payload. Identity and output-classword skips use exact Fractions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction as F
from typing import Any
import argparse
import json
from pathlib import Path

RESOURCE_NAMES = (
    "COMPUTE_GATE", "SRAM_READ", "SRAM_WRITE", "LOCAL", "HBM0", "HBM1",
    "INJECT0", "INJECT1", "XPOS", "XNEG", "YPOS", "YNEG", "HOP",
)
STAGES = ("load", "compute", "store")
NS = 10**9


class FailClosed(ValueError):
    pass


def fraction_from_dyadic(value: dict[str, Any] | None) -> F:
    if value is None:
        return F(0)
    significand = int(value["significand"])
    exponent = int(value["exponent"])
    return F(significand * 2**exponent) if exponent >= 0 else F(significand, 2**-exponent)


def fraction_from_hex(value: str) -> F:
    text = value.strip().lower()
    sign = -1 if text.startswith("-") else 1
    if text[:1] in ("-", "+"):
        text = text[1:]
    mantissa, exponent_text = text.split("p", 1)
    exponent = int(exponent_text)
    mantissa = mantissa.removeprefix("0x")
    whole, dot, frac = mantissa.partition(".")
    digits = (whole or "0") + frac
    significand = int(digits or "0", 16) * sign
    exponent -= 4 * len(frac) if dot else 0
    return F(significand * 2**exponent) if exponent >= 0 else F(significand, 2**-exponent)


@dataclass(frozen=True)
class ActivityFact:
    work: tuple[F, ...]
    canonical_q: F
    private_time_s: F
    hbm_payload: tuple[tuple[int, int], tuple[int, int]]
    int4_ops: F
    vec_eqops: F


class FactsGeometry:
    """O(class_count) lazy tile lookup; retains no tile/core arrays."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        self.geometry = report["geometry"]
        self.M = int(self.geometry["M"])
        self.N = int(self.geometry["N"])
        self.K = int(self.geometry["K"])
        self.tm = int(self.geometry["tm"])
        self.tn = int(self.geometry["tn"])
        self.tk = int(self.geometry["tk"])
        self.mt = int(self.geometry["m_tiles"])
        self.nt = int(self.geometry["n_tiles"])
        self.kt = int(self.geometry["k_tiles"])
        self.outputs = int(self.geometry["output_tile_count"])
        self.waves = int(self.geometry["consumption_wave_count"])
        self.path = self.geometry["path"]
        self.dtypes = (self.geometry["a_storage"], self.geometry["w_storage"],
                       self.geometry["output_storage"])
        self.kappa = (F(self.geometry["kappa_quant"]),
                      F(self.geometry["kappa_acc"]))
        if int(self.geometry["core_count"]) != 1:
            raise FailClosed("single-owner 13-row model requires core_count=1")
        if not 0 < len(self.geometry["classes"]) <= 32:
            raise FailClosed("access-class facts outside fixed 1..32 cache")
        self.classes = list(self.geometry["classes"])
        self.capacities = tuple(F(value) for value in
                                report["synthetic_hardware"]["hardware_capacities_13"])
        if len(self.capacities) != 13:
            raise FailClosed("hardware capacity snapshot is not 13 domains")
        if tuple(report["synthetic_hardware"]["resource_names"]) != RESOURCE_NAMES:
            raise FailClosed("13-resource capacity order differs from the Activity schema")
        self.class_stages: dict[int, dict[str, ActivityFact]] = {}
        self._validate_classes()
        self.expected = self._aggregate_class_facts()
        self._validate_source_ledger()

    def _activity(self, activity: dict[str, Any]) -> ActivityFact:
        work = activity.get("resource_work_13", activity.get("work_by_resource"))
        if not isinstance(work, list) or len(work) != 13:
            raise FailClosed("stage Activity omitted its 13 resource-work domains")
        payload = activity.get("hbm_row_payload_bytes",
                               activity.get("hbm_bytes_by_row_direction"))
        if not isinstance(payload, list) or len(payload) != 2 or any(
                not isinstance(row, list) or len(row) != 2 for row in payload):
            raise FailClosed("stage Activity omitted its two-row/two-direction HBM payload")
        typed = activity["typed_ops"]
        vec = F(int(typed["vec_eqops_whole"])) + fraction_from_dyadic(
            typed.get("vec_eqops_fraction"))
        return ActivityFact(
            work=tuple(F(value) for value in work),
            canonical_q=F(activity["canonical_q"]),
            private_time_s=F(activity["private_time_s"]),
            hbm_payload=tuple(tuple(int(x) for x in row) for row in payload),
            int4_ops=F(int(typed["int4_ops"])),
            vec_eqops=vec,
        )

    def _validate_classes(self) -> None:
        total_repetitions = 0
        for cls in self.classes:
            cid = int(cls["index"])
            if cid in self.class_stages:
                raise FailClosed("duplicate access-class index")
            if len(cls.get("hardware_capacities_13", [])) != 13:
                raise FailClosed("access class omitted its 13 hardware capacities")
            if tuple(F(v) for v in cls["hardware_capacities_13"]) != self.capacities:
                raise FailClosed("class and Runtime capacity snapshots differ")
            repetitions = int(cls["repeat_count"])
            ranges = (int(cls["m_tile_count"]), int(cls["n_tile_count"]),
                      int(cls["k_tile_count"]))
            if repetitions != ranges[0] * ranges[1] * ranges[2]:
                raise FailClosed("class repetition count is not its rectangular coverage")
            total_repetitions += repetitions
            self.class_stages[cid] = {
                stage: self._activity(cls[stage + "_fact"]) for stage in STAGES
            }
        if total_repetitions != self.waves:
            raise FailClosed("class repetition counts do not cover all K consumptions")
        if self.outputs * self.kt != self.waves:
            raise FailClosed("geometry output/K products disagree with the facts report")

    def class_for_tiles(self, mi: int, ni: int, ki: int) -> dict[str, Any]:
        matches = []
        for cls in self.classes:
            if (int(cls["m_tile_begin"]) <= mi <
                    int(cls["m_tile_begin"]) + int(cls["m_tile_count"]) and
                int(cls["n_tile_begin"]) <= ni <
                    int(cls["n_tile_begin"]) + int(cls["n_tile_count"]) and
                int(cls["k_tile_begin"]) <= ki <
                    int(cls["k_tile_begin"]) + int(cls["k_tile_count"])):
                matches.append(cls)
        if len(matches) != 1:
            raise FailClosed(f"lazy geometry lookup matched {len(matches)} classes")
        return matches[0]

    def class_for_wave(self, wave: int) -> dict[str, Any]:
        if not 0 <= wave < self.waves:
            raise FailClosed("wave index outside lazy geometry")
        q, ki = divmod(wave, self.kt)
        mi, ni = divmod(q, self.nt)
        return self.class_for_tiles(mi, ni, ki)

    def stage_fact(self, stage: int | str, wave: int) -> ActivityFact:
        name = STAGES[stage] if isinstance(stage, int) else stage
        cls = self.class_for_wave(wave)
        return self.class_stages[int(cls["index"])][name]

    def is_final(self, wave: int) -> bool:
        return bool(self.class_for_wave(wave)["final_k"])

    def class_signature(self, wave: int) -> tuple[Any, ...]:
        cls = self.class_for_wave(wave)
        return (int(cls["actual_m"]), int(cls["actual_n"]), int(cls["actual_k"]),
                bool(cls["m_tail"]), bool(cls["n_tail"]), bool(cls["k_tail"]),
                bool(cls["first_k"]), bool(cls["final_k"]),
                tuple(self.class_stages[int(cls["index"])][s] for s in STAGES))

    def output_shape(self, q: int) -> tuple[int, int]:
        if not 0 <= q < self.outputs:
            raise FailClosed("output tile index outside lazy geometry")
        mi, ni = divmod(q, self.nt)
        cls = self.class_for_tiles(mi, ni, 0)
        return int(cls["actual_m"]), int(cls["actual_n"])

    def same_shape_run_end(self, q: int) -> int:
        """Exclusive end of a contiguous equal-(actual M,N) output run."""
        m, n = self.output_shape(q)
        mi, ni = divmod(q, self.nt)
        end = self.outputs
        if self.N % self.tn and self.nt > 1:
            if n == self.tn:
                end = min(end, mi * self.nt + self.nt - 1)
            else:
                end = min(end, (mi + 1) * self.nt)
        if self.M % self.tm and self.mt > 1:
            m_tail_start = (self.mt - 1) * self.nt
            if m == self.tm:
                end = min(end, m_tail_start)
        return max(q + 1, end)

    def whole_row_classword(self, mi: int) -> tuple[Any, ...]:
        """Compact row word as class-derived intervals; O(classes), not O(N tiles)."""
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for cls in self.classes:
            if not (int(cls["m_tile_begin"]) <= mi <
                    int(cls["m_tile_begin"]) + int(cls["m_tile_count"])):
                continue
            key = (int(cls["n_tile_begin"]), int(cls["n_tile_count"]))
            grouped.setdefault(key, []).append(cls)
        segments = []
        cursor = 0
        for (n_begin, n_count), records in sorted(grouped.items()):
            if n_begin != cursor:
                raise FailClosed("whole-row classword has a gap or overlap")
            records.sort(key=lambda c: int(c["k_tile_begin"]))
            k_cursor = 0
            k_word = []
            for cls in records:
                k_begin, k_count = int(cls["k_tile_begin"]), int(cls["k_tile_count"])
                if k_begin != k_cursor:
                    raise FailClosed("classword K partition has a gap or overlap")
                cid = int(cls["index"])
                class_signature = (
                    int(cls["actual_m"]), int(cls["actual_n"]), int(cls["actual_k"]),
                    bool(cls["m_tail"]), bool(cls["n_tail"]), bool(cls["k_tail"]),
                    bool(cls["first_k"]), bool(cls["final_k"]),
                    tuple(self.class_stages[cid][s] for s in STAGES),
                )
                k_word.append((k_begin, k_count, class_signature))
                k_cursor += k_count
            if k_cursor != self.kt:
                raise FailClosed("classword K intervals do not span the tile K range")
            segments.append((n_begin, n_count, tuple(k_word)))
            cursor += n_count
        if cursor != self.nt:
            raise FailClosed("whole-row classword does not span N tiles")
        return tuple(segments)

    def _aggregate_class_facts(self) -> dict[str, Any]:
        service = [F(0) for _ in range(13)]
        payload = [[F(0), F(0)] for _ in range(2)]
        int4 = vec = F(0)
        stages = {s: 0 for s in STAGES}
        for cls in self.classes:
            repetitions = int(cls["repeat_count"])
            for stage in STAGES:
                fact = self.class_stages[int(cls["index"])][stage]
                reps = repetitions
                if stage == "store":
                    reps = repetitions if bool(cls["final_k"]) else 0
                stages[stage] += reps
                for r, work in enumerate(fact.work):
                    service[r] += reps * work
                for row in range(2):
                    for direction in range(2):
                        payload[row][direction] += reps * fact.hbm_payload[row][direction]
                int4 += reps * fact.int4_ops
                vec += reps * fact.vec_eqops
        return {"resource_service": service, "hbm_payload": payload,
                "int4_ops": int4, "vec_eqops": vec, "stage_counts": stages}

    def _validate_source_ledger(self) -> None:
        source = self.report["source_counters_full_single_gemm"]
        payload = self.expected["hbm_payload"]
        if sum(row[0] for row in payload) != int(source["hbm_read_bytes"]):
            raise FailClosed("class-fact HBM read payload differs from the source ledger")
        if sum(row[1] for row in payload) != int(source["hbm_write_bytes"]):
            raise FailClosed("class-fact HBM write payload differs from the source ledger")
        if self.expected["resource_service"][3] != int(source["local_bytes"]):
            raise FailClosed("class-fact LOCAL service differs from the source ledger")
        if self.expected["int4_ops"] != int(source["matrix_int4_ops"]):
            raise FailClosed("class-fact INT4 operations differ from the source ledger")
        source_vec = (int(source["vec_equivalent_ops_whole"]) +
                      fraction_from_hex(source["vec_equivalent_ops_fraction_hex"]))
        if self.expected["vec_eqops"] != source_vec:
            raise FailClosed("class-fact vector-equivalent operations differ from the source ledger")


def make_ledger() -> dict[str, F]:
    ledger = {
        "elapsed_s": F(0),
        "int4_ops": F(0),
        "vec_eqops": F(0),
        "compute_busy_s": F(0),
        "hbm0_busy_s": F(0),
        "hbm1_busy_s": F(0),
        "hbm_busy_s": F(0),
        "hbm_shared_s": F(0),
    }
    for r in range(13):
        ledger[f"resource_service_{r}"] = F(0)
        ledger[f"resource_busy_s_{r}"] = F(0)
    for row in range(2):
        for direction in range(2):
            ledger[f"hbm_payload_{row}_{direction}"] = F(0)
    return ledger


@dataclass
class OwnerState:
    next_load: int = 0
    retired: int = 0
    active: dict[int, tuple[int, F]] = field(default_factory=dict)
    loaded: list[int] = field(default_factory=list)
    computed: list[int] = field(default_factory=list)
    finish: F | None = None


class Solo13:
    def __init__(self, geometry: FactsGeometry, *, event_cap: int = 256,
                 accelerate: bool = True):
        if event_cap <= 0:
            raise FailClosed("global expanded-event cap must be positive")
        self.g = geometry
        self.event_cap = event_cap
        self.accelerate = accelerate
        self.o = OwnerState()
        self.now = F(0)
        self.ledger = make_ledger()
        self.events = 0
        self.deadline_trace: list[dict[str, Any]] = []
        self.skipped_waves = 0
        self.skipped_outputs = 0
        self.descriptors: list[dict[str, Any]] = []
        self.seen: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        self.peak_input = self.peak_output = self.peak_active = 0
        self.nonfinal_compute_at_full_output = 0

    def occupancy(self, owner: OwnerState | None = None) -> tuple[int, int]:
        o = self.o if owner is None else owner
        input_occupancy = (len(o.loaded) + int(0 in o.active) +
                           int(1 in o.active))
        output_occupancy = (len(o.computed) + int(2 in o.active) +
                            int(1 in o.active and self.g.is_final(o.active[1][0])))
        return input_occupancy, output_occupancy

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
                raise FailClosed("two-slot readiness queue capacity violated")
            if 2 not in o.active and o.computed:
                o.active[2] = (o.computed.pop(0), F(0))
                changed = True
            if 1 not in o.active and o.loaded:
                wave = o.loaded[0]
                final = self.g.is_final(wave)
                _, out = self.occupancy()
                if not final or out < 2:
                    o.loaded.pop(0)
                    o.active[1] = (wave, F(0))
                    if not final and out == 2:
                        self.nonfinal_compute_at_full_output += 1
                    changed = True
            inp, _ = self.occupancy()
            if 0 not in o.active and o.next_load < self.g.waves and inp < 2:
                wave = o.next_load
                o.active[0] = (wave, F(0))
                o.next_load += 1
                changed = True
            if (o.retired == self.g.outputs and not o.active and
                    not o.loaded and not o.computed):
                o.finish = self.now
                return
            if not changed:
                inp, out = self.occupancy()
                self.peak_input = max(self.peak_input, inp)
                self.peak_output = max(self.peak_output, out)
                if inp > 2 or out > 2:
                    raise FailClosed("two-slot readiness queue capacity violated")
                return

    def refs(self, owner: OwnerState | None = None, *,
             include_next: bool = True) -> list[int]:
        o = self.o if owner is None else owner
        refs = [wave for wave, _ in o.active.values()]
        refs += list(o.loaded) + list(o.computed)
        if include_next and o.next_load < self.g.waves:
            refs.append(o.next_load)
        return refs

    def anchor(self, owner: OwnerState | None = None) -> int:
        o = self.o if owner is None else owner
        return min(self.refs(o), default=o.next_load)

    def key(self) -> tuple[Any, ...] | None:
        o = self.o
        if o.finish is not None:
            return None
        refs = self.refs(o)
        base = min(refs, default=o.next_load)

        def tag(wave: int) -> tuple[Any, ...]:
            return wave - base, self.g.class_signature(wave)

        return (
            tuple((stage, *tag(wave)) for stage, (wave, _) in sorted(o.active.items())),
            tuple(tag(wave) for wave in o.loaded),
            tuple(tag(wave) for wave in o.computed),
            o.next_load - base,
            o.retired - base // self.g.kt,
            self.g.class_signature(o.next_load)
            if o.next_load < self.g.waves else None,
        )

    def snapshot(self) -> dict[str, Any]:
        o = self.o
        return {
            "active": dict(o.active), "loaded": list(o.loaded),
            "computed": list(o.computed), "next_load": o.next_load,
            "retired": o.retired, "anchor": self.anchor(o),
            "now": self.now, "ledger": dict(self.ledger),
            "events": self.events, "trace_start": len(self.deadline_trace),
        }

    def output_copy_bound(self, old: dict[str, Any], d: int) -> int:
        g, o = self.g, self.o
        if d % g.kt:
            return 0
        period_outputs = d // g.kt
        if period_outputs <= 0:
            return 0
        if o.next_load - old["next_load"] != d:
            return 0
        if o.retired - old["retired"] != period_outputs:
            return 0
        q0 = old["next_load"] // g.kt
        qcur = o.next_load // g.kt
        if qcur != q0 + period_outputs:
            return 0
        live_refs = self.refs(o)
        wave_bounds = [(g.waves - 1 - wave) // d for wave in live_refs
                       if wave < g.waves]
        wave_bounds.append((g.waves - o.next_load) // d)
        wave_bound = max(0, min(wave_bounds, default=0))
        if wave_bound == 0:
            return 0

        if period_outputs % g.nt == 0:
            rows_per_cycle = period_outputs // g.nt
            qlast = q0 + period_outputs - 1
            source_mtiles = {g.output_shape(q)[0] for q in (q0, qlast)}
            source_m = next(iter(source_mtiles)) if len(source_mtiles) == 1 else None
            if source_m is None:
                return 0
            source_row = g.whole_row_classword(q0 // g.nt)
            if g.whole_row_classword(qlast // g.nt) != source_row:
                return 0
            current_rows = {
                (wave // g.kt) // g.nt
                for wave in live_refs if wave < g.waves
            }
            if any(g.whole_row_classword(row) != source_row for row in current_rows):
                return 0
            row_bound = wave_bound
            if source_m != g.tm:
                return 0
            if g.M % g.tm:
                tail_row = g.mt - 1
                for wave in live_refs:
                    if wave >= g.waves:
                        continue
                    row = (wave // g.kt) // g.nt
                    row_bound = min(
                        row_bound,
                        max(0, (tail_row - 1 - row) // rows_per_cycle),
                    )
            return min(wave_bound, row_bound)

        if not (q0 < g.outputs and qcur < g.outputs):
            return 0
        shape = g.output_shape(q0)
        if g.same_shape_run_end(q0) < q0 + period_outputs:
            return 0
        if (g.output_shape(qcur) != shape or
                g.same_shape_run_end(qcur) < qcur + period_outputs):
            return 0
        class_bound = wave_bound
        for wave in live_refs:
            if wave >= g.waves:
                continue
            q = wave // g.kt
            run_end = g.same_shape_run_end(q)
            class_bound = min(class_bound,
                              max(0, (run_end - 1 - q) // period_outputs))
        return min(wave_bound, class_bound)

    def try_skip(self, old: dict[str, Any]) -> bool:
        o, g = self.o, self.g
        old_owner = OwnerState(
            next_load=old["next_load"], retired=old["retired"],
            active=dict(old["active"]), loaded=list(old["loaded"]),
            computed=list(old["computed"]),
        )
        old_refs = self.refs(old_owner)
        new_refs = self.refs(o)
        d = self.anchor(o) - old["anchor"]
        if d <= 0 or not old_refs or not new_refs:
            return False
        if len(old["active"]) != len(o.active) or set(old["active"]) != set(o.active):
            return False
        for stage, (wave0, progress0) in old["active"].items():
            wave1, progress1 = o.active[stage]
            if (wave1 - wave0 != d or progress1 != progress0 or
                    g.class_signature(wave1) != g.class_signature(wave0)):
                return False
        if (len(old["loaded"]) != len(o.loaded) or
                len(old["computed"]) != len(o.computed)):
            return False
        if any(b - a != d for a, b in zip(old["loaded"], o.loaded)):
            return False
        if any(b - a != d for a, b in zip(old["computed"], o.computed)):
            return False
        if o.next_load - old["next_load"] != d:
            return False

        if d < g.kt:
            if any(wave // g.kt != old["anchor"] // g.kt or
                   wave % g.kt == 0 or wave % g.kt >= g.kt - 1
                   for wave in old_refs + new_refs):
                return False
            max_k = max(wave % g.kt for wave in old_refs + new_refs)
            count = (g.kt - 2 - max_k) // d
            if count <= 0:
                return False
            kind = "Kmiddle_identity"
            outputs = 0
            category_word = "same output; full-K interior only; first/final K remain explicit"
        else:
            count = self.output_copy_bound(old, d)
            if count <= 0:
                return False
            kind = "output_classword_identity"
            outputs = (d // g.kt) * count
            period_outputs = d // g.kt
            q0 = old["next_load"] // g.kt
            if period_outputs % g.nt == 0:
                row_word = g.whole_row_classword(q0 // g.nt)
                category_word = {
                    "kind": "whole-row-classword",
                    "rows_per_period": period_outputs // g.nt,
                    "segments": len(row_word),
                    "first_row": q0 // g.nt,
                    "actual_m": g.output_shape(q0)[0],
                }
            else:
                category_word = {
                    "kind": "same-shape-contiguous-run",
                    "start_output": q0,
                    "exclusive_end": g.same_shape_run_end(q0),
                    "actual_shape": g.output_shape(q0),
                }

        before = old["ledger"]
        unit = {name: self.ledger[name] - before[name]
                for name in self.ledger}
        deadline_template = self.deadline_trace[old["trace_start"]:]
        for name, value in unit.items():
            self.ledger[name] += value * count
        self.now = self.ledger["elapsed_s"]
        wave_shift = d * count
        o.active = {stage: (wave + wave_shift, progress)
                    for stage, (wave, progress) in o.active.items()}
        o.loaded = [wave + wave_shift for wave in o.loaded]
        o.computed = [wave + wave_shift for wave in o.computed]
        o.next_load += wave_shift
        if outputs:
            o.retired += outputs
        else:
            o.retired += wave_shift // g.kt
        self.skipped_waves += wave_shift
        self.skipped_outputs += outputs
        self.descriptors.append({
            "kind": kind,
            "step_waves": d,
            "repeat_count": count,
            "step_outputs": d // g.kt if d % g.kt == 0 else 0,
            "category_word": category_word,
            "template_events": self.events - old["events"],
            "return_map": "exact identity on every live progress coordinate",
            "guard": "same live stage/queue topology, exact translated wave and class facts, exact progress equality, classword invariant; fixed solo capacities",
            "owner_q_alpha": "one owner: alpha_i=Q_i/sum(Q); exact Q-weighted normalization cancels to the shared progress-rate tier",
            "ledger_per_step": {key: str(value) for key, value in unit.items()},
            "deadline_template": deadline_template,
            "skip_waves": wave_shift,
            "skip_outputs": outputs,
        })
        return True

    def _rates(self, active: dict[tuple[int, int], F]) -> dict[tuple[int, int], F]:
        """Exact one-owner Q/alpha progressive filling over 13 physical rows."""
        facts = {key: self.g.stage_fact(key[0], key[1]) for key in active}
        owner_q = sum((fact.canonical_q for fact in facts.values()), F(0))
        if not facts or owner_q <= 0:
            raise FailClosed("active stages have no positive owner canonical Q")
        alpha = {key: fact.canonical_q / owner_q for key, fact in facts.items()}
        rates = {key: F(0) for key in active}
        free = set(active)
        for _ in range(len(active)):
            if not free:
                break
            bounds: dict[tuple[str, Any], F] = {}
            for row, capacity in enumerate(self.g.capacities):
                used = sum(facts[key].work[row] * rates[key] for key in active)
                slope = sum(facts[key].work[row] / owner_q for key in free)
                if slope:
                    bounds[("resource", row)] = (capacity - used) / slope
            for key in free:
                private = facts[key].private_time_s
                if private:
                    bounds[("private", key)] = (1 / private - rates[key]) * owner_q
            if not bounds:
                raise FailClosed("unbounded 13-domain progressive-fill tier")
            step = min(bounds.values())
            if step < 0:
                raise FailClosed("infeasible 13-domain allocator state")
            for key in free:
                rates[key] += step / owner_q
            frozen = set()
            for (kind, index), value in bounds.items():
                if value != step:
                    continue
                if kind == "resource":
                    frozen.update(key for key in free
                                  if facts[key].work[index] > 0)
                else:
                    frozen.add(index)
            if not frozen:
                raise FailClosed("13-domain progressive fill failed to freeze a limiter")
            free.difference_update(frozen)
        if free:
            raise FailClosed("13-domain progressive-fill tier bound exhausted")
        for row, capacity in enumerate(self.g.capacities):
            used = sum(facts[key].work[row] * rates[key] for key in active)
            if used > capacity:
                raise FailClosed("13-domain allocation exceeds capacity")
        self.last_allocation = {
            "owner_q": owner_q,
            "alpha": alpha,
            "rates": rates,
        }
        return rates

    def event(self) -> None:
        if self.events >= self.event_cap:
            raise FailClosed("global expanded-event cap reached")
        self.settle()
        if self.o.finish is not None:
            return
        o, g = self.o, self.g
        active = {(stage, wave): progress
                  for stage, (wave, progress) in o.active.items()}
        rates = self._rates(active)
        self.peak_active = max(self.peak_active, len(active))
        facts = {key: g.stage_fact(key[0], key[1]) for key in active}
        deadlines = {
            key: (1 - progress) / rates[key]
            for key, progress in active.items() if rates[key] > 0
        }
        if not deadlines:
            raise FailClosed("deadlock: no active positive-rate stage")
        delta = min(deadlines.values())
        selected = {key for key, deadline in deadlines.items()
                    if deadline == delta}
        base = self.anchor()
        self.deadline_trace.append({
            "delta_s": str(delta),
            "owner_q": str(self.last_allocation["owner_q"]),
            "active": [
                {
                    "stage": stage, "wave_offset": wave - base,
                    "class": self.g.class_for_wave(wave)["index"],
                    "progress": str(progress), "rate": str(rates[(stage, wave)]),
                    "canonical_q": str(facts[(stage, wave)].canonical_q),
                    "alpha": str(self.last_allocation["alpha"][(stage, wave)]),
                    "private_time_s": str(facts[(stage, wave)].private_time_s),
                    "resource_work_13": [str(x) for x in facts[(stage, wave)].work],
                }
                for (stage, wave), progress in sorted(active.items())
            ],
            "deadline_margin_s": [
                {"stage": stage, "wave_offset": wave - base,
                 "margin": str(deadlines[(stage, wave)] - delta)}
                for stage, wave in sorted(deadlines)
            ],
            "selected": [
                {"stage": stage, "wave_offset": wave - base}
                for stage, wave in sorted(selected)
            ],
        })

        self.ledger["elapsed_s"] += delta
        busy = [False] * 13
        for key, progress in active.items():
            stage, wave = key
            fact = facts[key]
            rate = rates[key]
            increment = delta * rate
            o.active[stage] = (wave, progress + increment)
            for row in range(13):
                self.ledger[f"resource_service_{row}"] += increment * fact.work[row]
                if rate > 0 and fact.work[row] > 0:
                    busy[row] = True
            for row in range(2):
                for direction in range(2):
                    self.ledger[f"hbm_payload_{row}_{direction}"] += (
                        increment * fact.hbm_payload[row][direction]
                    )
            self.ledger["int4_ops"] += increment * fact.int4_ops
            self.ledger["vec_eqops"] += increment * fact.vec_eqops

        for row in range(13):
            if busy[row]:
                self.ledger[f"resource_busy_s_{row}"] += delta
        if busy[0]:
            self.ledger["compute_busy_s"] += delta
        if busy[4]:
            self.ledger["hbm0_busy_s"] += delta
        if busy[5]:
            self.ledger["hbm1_busy_s"] += delta
        if busy[4] or busy[5]:
            self.ledger["hbm_busy_s"] += delta
        self.now += delta

        for stage, wave in sorted(selected):
            del o.active[stage]
            if stage == 0:
                o.loaded.append(wave)
            elif stage == 1:
                if g.is_final(wave):
                    o.computed.append(wave)
            else:
                o.retired += 1
        self.events += 1
        self.settle()

    def ledger_json(self) -> dict[str, Any]:
        return {
            "elapsed_s": str(self.ledger["elapsed_s"]),
            "resource_service_13": [str(self.ledger[f"resource_service_{r}"])
                                    for r in range(13)],
            "resource_busy_s_13": [str(self.ledger[f"resource_busy_s_{r}"])
                                   for r in range(13)],
            "local_bytes": str(self.ledger["resource_service_3"]),
            "compute_service": str(self.ledger["resource_service_0"]),
            "hbm_row_payload_bytes": [
                [str(self.ledger[f"hbm_payload_{r}_{d}"]) for d in range(2)]
                for r in range(2)
            ],
            "int4_ops": str(self.ledger["int4_ops"]),
            "vec_eqops": str(self.ledger["vec_eqops"]),
            "compute_busy_s": str(self.ledger["compute_busy_s"]),
            "hbm0_busy_s": str(self.ledger["hbm0_busy_s"]),
            "hbm1_busy_s": str(self.ledger["hbm1_busy_s"]),
            "hbm_busy_s": str(self.ledger["hbm_busy_s"]),
            "hbm_shared_s": str(self.ledger["hbm_shared_s"]),
        }

    def _validate_completed_ledger(self) -> None:
        expected = self.g.expected
        if self.ledger["int4_ops"] != expected["int4_ops"]:
            raise FailClosed("completed INT4 ledger differs from cached class facts")
        if self.ledger["vec_eqops"] != expected["vec_eqops"]:
            raise FailClosed("completed vector ledger differs from cached class facts")
        if self.ledger["resource_service_3"] != expected["resource_service"][3]:
            raise FailClosed("completed LOCAL ledger differs from cached class facts")
        for row in range(2):
            for direction in range(2):
                if self.ledger[f"hbm_payload_{row}_{direction}"] != \
                        expected["hbm_payload"][row][direction]:
                    raise FailClosed("completed HBM payload ledger differs from class facts")
        for row in range(13):
            if self.ledger[f"resource_service_{row}"] != \
                    expected["resource_service"][row]:
                raise FailClosed(f"completed resource service row {row} differs from class facts")
        if self.o.retired != self.g.outputs:
            raise FailClosed("completed output retirement count differs from geometry")

    def report(self, status: str, reason: str | None = None) -> dict[str, Any]:
        inp, out = self.occupancy()
        expected = {
            "resource_service_13": [str(x) for x in self.g.expected["resource_service"]],
            "hbm_payload_row_direction": [
                [str(x) for x in row] for row in self.g.expected["hbm_payload"]
            ],
            "int4_ops": str(self.g.expected["int4_ops"]),
            "vec_eqops": str(self.g.expected["vec_eqops"]),
            "stage_counts": self.g.expected["stage_counts"],
        }
        report = {
            "schema_version": 1,
            "status": status,
            "reason": reason,
            "scope": "offline single-owner continuous internal-time model; compiler facts; exact Fraction; no silicon or production descriptor certificate",
            "facts_source": {
                "synthetic_hardware_fingerprint":
                    self.g.report["synthetic_hardware"]["hardware_fingerprint"],
                "facts_report_fixture": self.g.report.get("fixture_identity"),
                "facts_report_stage_submission_status":
                    self.g.report["live_state"]["status"],
                "geometry_class_count": len(self.g.classes),
                "class_fact_limit": 32,
                "no_tile_core_arrays": True,
            },
            "geometry": {
                "M": self.g.M, "N": self.g.N, "K": self.g.K,
                "tm": self.g.tm, "tn": self.g.tn, "tk": self.g.tk,
                "m_tiles": self.g.mt, "n_tiles": self.g.nt, "k_tiles": self.g.kt,
                "outputs": str(self.g.outputs), "waves": str(self.g.waves),
                "path": self.g.path, "dtypes": self.g.dtypes,
                "kappa": [str(x) for x in self.g.kappa],
            },
            "capacities_13": [str(x) for x in self.g.capacities],
            "global_expanded_event_cap": self.event_cap,
            "expanded_events": self.events,
            "skipped_waves": str(self.skipped_waves),
            "skipped_outputs": str(self.skipped_outputs),
            "descriptor_count": len(self.descriptors),
            "descriptors": self.descriptors,
            "peak_active_stages": self.peak_active,
            "peak_input_slots": self.peak_input,
            "peak_output_slots": self.peak_output,
            "nonfinal_compute_started_while_output_full":
                self.nonfinal_compute_at_full_output,
            "live_state": {
                "next_load_wave": str(self.o.next_load),
                "next_compute_wave": str(self.o.retired + len(self.o.computed) +
                                         (1 if 1 in self.o.active else 0)),
                "retired_outputs": str(self.o.retired),
                "loaded_waves": [str(x) for x in self.o.loaded],
                "computed_final_waves": [str(x) for x in self.o.computed],
                "active": [
                    {"stage": STAGES[s], "wave": str(w), "progress": str(p)}
                    for s, (w, p) in sorted(self.o.active.items())
                ],
                "input_occupancy": inp, "output_occupancy": out,
                "finish_s": str(self.o.finish) if self.o.finish is not None else None,
            },
            "ledger": self.ledger_json(),
            "class_fact_ledger": expected,
            "busy_union_semantics": {
                "compute_busy_s": "one union of active COMPUTE_GATE users",
                "hbm_busy_s": "one union of active HBM0 or HBM1 users",
                "hbm_shared_s": "zero for one owner; no two-owner overlap charge",
            },
            "certificate_boundary": [
                "exact skip identity requires unchanged active stage/queue topology and exact equality of every live progress fraction",
                "classword skip requires full facts-derived row word equality or a contiguous same-shape run; arithmetic bounds stop at M/N/K class changes",
                "solo owner only, fixed capacities and facts; external stops/capacity changes/cancellation are outside this prototype",
                "global expanded-event cap never resets; exceeding it fails closed without per-wave fallback",
                "C++ Runtime integer-ns wake results are checked separately and are not claimed equal to this continuous exact internal-time model",
            ],
        }
        if reason:
            report["production"] = "not_implemented"
        return report

    def run(self) -> dict[str, Any]:
        try:
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
            self._validate_completed_ledger()
            return self.report("passed")
        except (FailClosed, AssertionError, ZeroDivisionError) as error:
            return self.report("failed_closed", str(error))


def independent_continuous_oracle(report: dict[str, Any]) -> dict[str, Any]:
    """Use the independent 13-row readiness allocator with no public-ns wake."""
    import importlib.util
    import sys
    root = Path(__file__).resolve().parents[2]
    module_path = root / "die" / "validation" / "os_lifecycle_runtime_oracle.py"
    spec = importlib.util.spec_from_file_location("independent_os_lifecycle_oracle",
                                                   module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the independent readiness oracle")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    class ContinuousOracle(module.Oracle):
        def run_continuous(self):
            while self.retired < self.outputs:
                self.settle()
                if not self.active:
                    raise AssertionError("continuous independent oracle deadlocked")
                allocation = module.rates(self.active, self.capacities)
                delta = min(item["remaining"] / allocation[stage]
                            for stage, item in self.active.items()
                            if allocation[stage] > 0)
                self.advance(self.now + delta)
            if self.active or self.inputs or self.outputs_live:
                raise AssertionError("continuous independent queues did not drain")
            return self

    oracle = ContinuousOracle(report).run_continuous()
    return {
        "finish_s": str(oracle.now),
        "hbm_payload_row_direction": [
            [str(x) for x in row] for row in oracle.hbm
        ],
        "int4_ops": str(oracle.int4),
        "vec_eqops": str(oracle.vec),
        "compute_busy_s": str(oracle.compute_busy),
        "hbm0_busy_s": str(oracle.row_busy[0]),
        "hbm1_busy_s": str(oracle.row_busy[1]),
        "hbm_busy_s": str(oracle.hbm_busy),
        "events": oracle.events,
    }


def compare_ledgers(lhs: dict[str, Any], rhs: dict[str, Any]) -> None:
    for field in ("elapsed_s", "resource_service_13", "resource_busy_s_13",
                  "local_bytes", "compute_service", "hbm_row_payload_bytes",
                  "int4_ops", "vec_eqops", "compute_busy_s", "hbm0_busy_s",
                  "hbm1_busy_s", "hbm_busy_s", "hbm_shared_s"):
        if lhs["ledger"][field] != rhs["ledger"][field]:
            raise AssertionError(f"ledger mismatch in {field}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True,
                        help="JSON output from os_lifecycle_runtime_driver")
    parser.add_argument("--cap", type=int, default=256,
                        help="global expanded exact-event cap (default 256)")
    parser.add_argument("--no-accelerate", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.facts.read_text())
    result = Solo13(FactsGeometry(data), event_cap=args.cap,
                    accelerate=not args.no_accelerate).run()
    result["facts_json"] = str(args.facts.resolve())
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
