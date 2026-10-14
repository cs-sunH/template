#!/usr/bin/env python3
"""Measure exact rational widths in the frozen 13-row Solo facts solver.

This imports the already-existing solver and feeds it captured Runtime facts.
The solver itself is not edited.  AuditFraction records every normalized
Fraction result produced by its arithmetic operators, with a conservative
unreduced cross-product width for each binary operation.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/os-single-owner-13-20261014/solo_facts_13.py"
FACT_DIR = ROOT / "work/os-single-owner-13-20261014/current"
STOP_STUDY = ROOT / "work/os-solo-stop-20261014/solo_stop_13.py"
STOP_CHECK = ROOT / "work/os-solo-stop-20261014/check_stops.py"
STOP_VALIDATION = ROOT / "work/os-solo-stop-20261014/stop_validation.json"
OUT_DIR = Path(__file__).resolve().parent


class WidthAudit:
    def __init__(self) -> None:
        self.phase_name = "setup"
        self.rationals: dict[str, dict] = defaultdict(self._new_record)
        self.integers: dict[str, dict] = defaultdict(self._new_int_record)

    @staticmethod
    def _new_record() -> dict:
        return {"count": 0, "max_numerator_bits": 0,
                "max_denominator_bits": 0, "max_total_bits": 0,
                "max_scratch_numerator_bits": 0,
                "max_scratch_denominator_bits": 0,
                "numerator_witness": None, "denominator_witness": None,
                "scratch_witness": None}

    @staticmethod
    def _new_int_record() -> dict:
        return {"count": 0, "max_bits": 0, "witness": None}

    @contextmanager
    def phase(self, name: str):
        old = self.phase_name
        self.phase_name = name
        try:
            yield
        finally:
            self.phase_name = old

    @staticmethod
    def bits(value: int) -> int:
        return max(1, abs(value).bit_length())

    def rational(self, value: Fraction, *, op: str = "observed",
                 scratch_num: int = 0, scratch_den: int = 0,
                 operands: tuple[Fraction, ...] = ()) -> None:
        key = f"{self.phase_name}/{op}"
        row = self.rationals[key]
        nbits, dbits = self.bits(value.numerator), self.bits(value.denominator)
        row["count"] += 1
        if nbits > row["max_numerator_bits"]:
            row["max_numerator_bits"] = nbits
            row["numerator_witness"] = str(value)
        if dbits > row["max_denominator_bits"]:
            row["max_denominator_bits"] = dbits
            row["denominator_witness"] = str(value)
        row["max_total_bits"] = max(row["max_total_bits"], nbits + dbits)
        if scratch_num > row["max_scratch_numerator_bits"] or \
                scratch_den > row["max_scratch_denominator_bits"]:
            row["scratch_witness"] = {
                "value": str(value), "numerator_bits": scratch_num,
                "denominator_bits": scratch_den,
                "operands": [str(x) for x in operands],
            }
        row["max_scratch_numerator_bits"] = max(
            row["max_scratch_numerator_bits"], scratch_num)
        row["max_scratch_denominator_bits"] = max(
            row["max_scratch_denominator_bits"], scratch_den)

    def integer(self, name: str, value: int) -> None:
        key = f"{self.phase_name}/{name}"
        row = self.integers[key]
        width = self.bits(value)
        row["count"] += 1
        if width > row["max_bits"]:
            row["max_bits"] = width
            row["witness"] = str(value)


AUDIT = WidthAudit()


class AuditFraction(Fraction):
    """Fraction subclass that observes exact results and scratch bounds."""
    __hash__ = Fraction.__hash__

    def __new__(cls, numerator=0, denominator=None, *, _record=True):
        if isinstance(numerator, Fraction) and denominator is None:
            numerator, denominator = numerator.numerator, numerator.denominator
        obj = super().__new__(cls, numerator, denominator)
        if _record:
            AUDIT.rational(obj, op="constructed")
        return obj

    @staticmethod
    def _fraction(value) -> Fraction:
        if isinstance(value, Fraction):
            return Fraction(value.numerator, value.denominator)
        return Fraction(value)

    @staticmethod
    def _widths(value: Fraction) -> tuple[int, int]:
        return AUDIT.bits(value.numerator), AUDIT.bits(value.denominator)

    def _binary(self, other, method: str, op: str):
        result = getattr(Fraction, method)(self, other)
        if result is NotImplemented:
            return NotImplemented
        if isinstance(result, Fraction):
            a, b = self._fraction(self), self._fraction(other)
            an, ad = self._widths(a)
            bn, bd = self._widths(b)
            if op in ("add", "sub"):
                sn = max(an + bd, bn + ad) + 1
                sd = ad + bd
            elif op == "mul":
                sn, sd = an + bn, ad + bd
            elif op == "div":
                sn, sd = an + bd, ad + bn
            elif op == "rdiv":
                sn, sd = bn + ad, bd + an
            else:
                sn, sd = 0, 0
            AUDIT.rational(result, op=op,
                           scratch_num=sn, scratch_den=sd, operands=(a, b))
            return AuditFraction(result, _record=False)
        return result

    def __add__(self, other): return self._binary(other, "__add__", "add")
    def __radd__(self, other): return self._binary(other, "__radd__", "add")
    def __sub__(self, other): return self._binary(other, "__sub__", "sub")
    def __rsub__(self, other): return self._binary(other, "__rsub__", "sub")
    def __mul__(self, other): return self._binary(other, "__mul__", "mul")
    def __rmul__(self, other): return self._binary(other, "__rmul__", "mul")
    def __truediv__(self, other): return self._binary(other, "__truediv__", "div")
    def __rtruediv__(self, other): return self._binary(other, "__rtruediv__", "rdiv")

    def _compare(self, other, method: str, op: str):
        result = getattr(Fraction, method)(self, other)
        if result is not NotImplemented:
            a, b = self._fraction(self), self._fraction(other)
            an, ad = self._widths(a)
            bn, bd = self._widths(b)
            if op == "eq":
                sn, sd = max(an, bn), max(ad, bd)
            else:
                sn, sd = max(an + bd, bn + ad), max(ad, bd)
            AUDIT.rational(self, op=f"compare_{op}",
                           scratch_num=sn, scratch_den=sd, operands=(a, b))
        return result

    def __eq__(self, other): return self._compare(other, "__eq__", "eq")
    def __lt__(self, other): return self._compare(other, "__lt__", "lt")
    def __le__(self, other): return self._compare(other, "__le__", "le")
    def __gt__(self, other): return self._compare(other, "__gt__", "gt")
    def __ge__(self, other): return self._compare(other, "__ge__", "ge")

    def __neg__(self):
        result = Fraction.__neg__(self)
        AUDIT.rational(result, op="neg", scratch_num=AUDIT.bits(result.numerator),
                       scratch_den=AUDIT.bits(result.denominator))
        return AuditFraction(result, _record=False)

    def __abs__(self):
        result = Fraction.__abs__(self)
        AUDIT.rational(result, op="abs", scratch_num=AUDIT.bits(result.numerator),
                       scratch_den=AUDIT.bits(result.denominator))
        return AuditFraction(result, _record=False)


def load_solver(path: Path = STUDY, module_name: str = "solo_facts_13_audited"):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen solver {STUDY}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.F = AuditFraction

    def wrap_method(cls, name: str, phase: str, after=None):
        original = getattr(cls, name)
        def wrapped(self, *args, **kwargs):
            with AUDIT.phase(phase):
                result = original(self, *args, **kwargs)
            if after is not None and (name != "try_skip" or result):
                with AUDIT.phase(f"{phase}_state"):
                    after(self, result)
            return result
        setattr(cls, name, wrapped)

    def sample_state(solver, _result=None):
        for name, value in solver.ledger.items():
            AUDIT.rational(value, op=f"ledger_{name}")
        for stage, (_wave, progress) in solver.o.active.items():
            AUDIT.rational(progress, op=f"active_progress_stage_{stage}")
        AUDIT.rational(solver.now, op="elapsed_time")
        for descriptor in solver.descriptors[-1:]:
            AUDIT.integer("skip_repeat_count", int(descriptor["repeat_count"]))
            AUDIT.integer("skip_step_waves", int(descriptor["step_waves"]))
            AUDIT.integer("skip_wave_shift", int(descriptor["skip_waves"]))
        AUDIT.integer("expanded_event_count", solver.events)
        AUDIT.integer("skipped_wave_count", solver.skipped_waves)
        AUDIT.integer("skipped_output_count", solver.skipped_outputs)

    wrap_method(module.FactsGeometry, "_aggregate_class_facts", "class_fact_aggregation")
    wrap_method(module.Solo13, "_rates", "allocation")
    wrap_method(module.Solo13, "event", "event", sample_state)
    wrap_method(module.Solo13, "key", "identity_key")
    wrap_method(module.Solo13, "output_copy_bound", "skip_guard")
    wrap_method(module.Solo13, "try_skip", "skip", sample_state)
    wrap_method(module.Solo13, "run", "lifecycle")
    if hasattr(module.Solo13, "resume_until"):
        wrap_method(module.Solo13, "resume_until", "resume")
    return module


def facts_values(geometry) -> None:
    with AUDIT.phase("fact_inputs"):
        for value in geometry.capacities:
            AUDIT.rational(value)
        for value in geometry.kappa:
            AUDIT.rational(value)
        for class_id, stages in geometry.class_stages.items():
            AUDIT.integer("class_id", class_id)
            for stage, fact in stages.items():
                for value in fact.work:
                    AUDIT.rational(value)
                for value in (fact.canonical_q, fact.private_time_s,
                              fact.int4_ops, fact.vec_eqops):
                    AUDIT.rational(value)
                for row in fact.hbm_payload:
                    for value in row:
                        AUDIT.integer("hbm_payload_bytes", value)


def compare_frozen_result(module, result: dict, fixture: str,
                          mode: str) -> dict:
    reference_path = FACT_DIR / f"{mode}_{fixture}.json"
    if not reference_path.is_file():
        raise AssertionError(f"missing frozen reference report: {reference_path}")
    reference = json.loads(reference_path.read_text())
    module.compare_ledgers(result, reference)
    for field in ("status", "expanded_events", "skipped_waves", "skipped_outputs"):
        if result[field] != reference[field]:
            raise AssertionError(f"{fixture}/{mode} differs from frozen {field}")
    return {"path": str(reference_path), "sha256": sha256(reference_path)}


def geometry_integers(report: dict) -> None:
    g = report["geometry"]
    with AUDIT.phase("geometry_integer_inputs"):
        for name in ("M", "N", "K", "tm", "tn", "tk", "core_count",
                     "m_tiles", "n_tiles", "k_tiles", "output_tile_count",
                     "consumption_wave_count"):
            AUDIT.integer(name, int(g[name]))
        for cls in g["classes"]:
            for name in ("repeat_count", "m_tile_begin", "m_tile_count",
                         "n_tile_begin", "n_tile_count", "k_tile_begin",
                         "k_tile_count"):
                AUDIT.integer(f"class_{name}", int(cls[name]))


def run_fixture(module, facts_path: Path, *, accelerate=True, cap=256) -> dict:
    # Isolate width maxima for a per-run record while retaining aggregate runs.
    saved_rationals, saved_integers = AUDIT.rationals, AUDIT.integers
    AUDIT.rationals = defaultdict(AUDIT._new_record)
    AUDIT.integers = defaultdict(AUDIT._new_int_record)
    AUDIT.phase_name = "facts_load"
    data = json.loads(facts_path.read_text())
    with AUDIT.phase("facts_load"):
        geometry = module.FactsGeometry(data)
    facts_values(geometry)
    geometry_integers(data)
    with AUDIT.phase("solver_init"):
        solver = module.Solo13(geometry, event_cap=cap, accelerate=accelerate)
    result = solver.run()
    frozen_reference = compare_frozen_result(
        module, result, facts_path.stem.removeprefix("facts_"),
        "powered" if accelerate else "expanded")
    report = {
        "fixture": facts_path.stem.removeprefix("facts_"),
        "mode": "powered" if accelerate else "expanded",
        "status": result["status"], "reason": result["reason"],
        "events": result["expanded_events"],
        "descriptors": result["descriptor_count"],
        "skipped_waves": result["skipped_waves"],
        "skipped_outputs": result["skipped_outputs"],
        "geometry": result["geometry"],
        "frozen_reference_match": frozen_reference,
        "rational_widths": {k: v for k, v in sorted(AUDIT.rationals.items())},
        "integer_widths": {k: v for k, v in sorted(AUDIT.integers.items())},
        "max_normalized_numerator_bits": max(
            (v["max_numerator_bits"] for v in AUDIT.rationals.values()), default=0),
        "max_normalized_denominator_bits": max(
            (v["max_denominator_bits"] for v in AUDIT.rationals.values()), default=0),
        "max_conservative_scratch_numerator_bits": max(
            (v["max_scratch_numerator_bits"] for v in AUDIT.rationals.values()), default=0),
        "max_conservative_scratch_denominator_bits": max(
            (v["max_scratch_denominator_bits"] for v in AUDIT.rationals.values()), default=0),
        "max_integer_bits": max((v["max_bits"] for v in AUDIT.integers.values()), default=0),
    }
    AUDIT.rationals, AUDIT.integers = saved_rationals, saved_integers
    return report


def measured_widths() -> dict:
    return {
        "rational_widths": {k: v for k, v in sorted(AUDIT.rationals.items())},
        "integer_widths": {k: v for k, v in sorted(AUDIT.integers.items())},
        "max_normalized_numerator_bits": max(
            (v["max_numerator_bits"] for v in AUDIT.rationals.values()), default=0),
        "max_normalized_denominator_bits": max(
            (v["max_denominator_bits"] for v in AUDIT.rationals.values()), default=0),
        "max_conservative_scratch_numerator_bits": max(
            (v["max_scratch_numerator_bits"] for v in AUDIT.rationals.values()), default=0),
        "max_conservative_scratch_denominator_bits": max(
            (v["max_scratch_denominator_bits"] for v in AUDIT.rationals.values()), default=0),
        "max_integer_bits": max((v["max_bits"] for v in AUDIT.integers.values()), default=0),
    }


def reset_widths() -> tuple[dict, dict]:
    saved = AUDIT.rationals, AUDIT.integers
    AUDIT.rationals = defaultdict(AUDIT._new_record)
    AUDIT.integers = defaultdict(AUDIT._new_int_record)
    return saved


def restore_widths(saved: tuple[dict, dict]) -> None:
    AUDIT.rationals, AUDIT.integers = saved


def run_stop_fixture(module, fixture: str, *, huge=False) -> dict:
    saved = reset_widths()
    facts_path = FACT_DIR / f"facts_{fixture}.json"
    data = json.loads(facts_path.read_text())
    with AUDIT.phase("facts_load"):
        geometry = module.FactsGeometry(data)
    facts_values(geometry)
    geometry_integers(data)
    with AUDIT.phase("solver_init"):
        baseline = module.Solo13(geometry, event_cap=256, accelerate=True)
    base_report = baseline.run()
    if base_report["status"] != "passed":
        raise AssertionError(f"baseline failed for {fixture}: {base_report['reason']}")
    frozen_reference = compare_frozen_result(module, base_report, fixture, "powered")
    finish = baseline.now

    # Rebuild facts/solver so the stop execution starts with an empty lifecycle.
    with AUDIT.phase("facts_load"):
        stopped_geometry = module.FactsGeometry(data)
    if huge:
        first_stop, second_stop = finish / 3, finish * module.F(2, 3)
        with AUDIT.phase("external_stop_inputs"):
            AUDIT.rational(first_stop, op="one_third_exact_stop")
            AUDIT.rational(second_stop, op="two_thirds_exact_stop")
        with AUDIT.phase("solver_init"):
            candidate = module.Solo13(stopped_geometry, event_cap=256,
                                      accelerate=True, stop_at=first_stop)
        first = candidate.run()
        if first["status"] != "paused":
            raise AssertionError(f"first huge stop failed for {fixture}: {first['reason']}")
        events_after_first = candidate.events
        second = candidate.resume_until(second_stop)
        if second["status"] != "paused":
            raise AssertionError(f"second huge stop failed for {fixture}: {second['reason']}")
        events_after_second = candidate.events
        final = candidate.resume_until()
        if final["status"] != "passed" or candidate.ledger != baseline.ledger:
            raise AssertionError(f"stop/resume ledger mismatch for {fixture}")
        if candidate.events > 256:
            raise AssertionError("huge stop/resume exceeded the execution-wide event cap")
        stop_metrics = {
            "stop_kind": "exact fractions at finish/3 and 2*finish/3",
            "finish_s": str(finish), "first_stop_s": str(first_stop),
            "second_stop_s": str(second_stop),
            "status_sequence": [first["status"], second["status"], final["status"]],
            "events_after_first_stop": events_after_first,
            "events_after_second_stop": events_after_second,
            "final_events": candidate.events,
            "skipped_waves": str(candidate.skipped_waves),
            "final_ledger_equals_unstopped": True,
            "unstopped_frozen_reference_match": frozen_reference,
        }
    else:
        # An integer nanosecond stop, followed by exact stops and a capacity
        # change.  The capacity update must invalidate old identity returns.
        stop_ns = 17
        first_stop = module.F(stop_ns, 10**9)
        third_stop, half_stop = finish / 3, finish / 2
        with AUDIT.phase("external_stop_inputs"):
            AUDIT.integer("absolute_stop_ns", stop_ns)
            AUDIT.integer("ns_denominator", 10**9)
            AUDIT.rational(first_stop, op="integer_ns_stop")
            AUDIT.rational(third_stop, op="one_third_exact_stop")
            AUDIT.rational(half_stop, op="one_half_exact_stop")
        with AUDIT.phase("solver_init"):
            candidate = module.Solo13(stopped_geometry, event_cap=256,
                                      accelerate=True, stop_at=first_stop)
        first = candidate.run()
        if first["status"] != "paused":
            raise AssertionError(f"integer ns stop failed for {fixture}: {first['reason']}")
        third = candidate.resume_until(third_stop)
        if third["status"] != "paused":
            raise AssertionError(f"first exact resume stop failed for {fixture}: {third['reason']}")
        new_capacities = tuple(value * (module.F(7, 8) if row in (4, 5) else 1)
                               for row, value in enumerate(candidate.g.capacities))
        cap_stop = candidate.resume_until(half_stop, capacities=new_capacities)
        if cap_stop["status"] != "paused":
            raise AssertionError(f"capacity-changed resume failed for {fixture}: {cap_stop['reason']}")
        after_capacity_change = candidate.events
        backwards = candidate.resume_until(third_stop)
        if backwards["status"] != "failed_closed":
            raise AssertionError("backwards stop was not rejected")
        if candidate.now != half_stop:
            raise AssertionError("backwards stop modified the lifecycle clock")
        final = candidate.resume_until()
        if final["status"] != "passed" or candidate.events > 256:
            raise AssertionError(f"capacity-change resume failed for {fixture}")
        stop_metrics = {
            "stop_kind": "17 integer nanoseconds; exact finish/3 and finish/2; then 7/8 HBM capacity change",
            "finish_s": str(finish), "integer_stop_ns": stop_ns,
            "integer_stop_s": str(first_stop), "exact_stops_s": [str(third_stop), str(half_stop)],
            "status_sequence": [first["status"], third["status"], cap_stop["status"],
                                backwards["status"], final["status"]],
            "events_after_capacity_change": after_capacity_change,
            "final_events": candidate.events,
            "capacity_change_invalidated_seen_returns": True,
            "backwards_stop_preserved_now": True,
            "final_status": final["status"],
            "unstopped_frozen_reference_match": frozen_reference,
        }
    stop_metrics.update(measured_widths())
    restore_widths(saved)
    return {"fixture": fixture, **stop_metrics}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def budget_analysis(runs: list[dict], stop_runs: list[dict]) -> dict:
    all_runs = [(f"{row['fixture']}/{row['mode']}", row) for row in runs]
    all_runs += [(f"{row['fixture']}/stop_resume", row) for row in stop_runs]
    state_limit = 256
    scratch_candidates = (256, 544, 768)
    state_over = []
    scratch_over: dict[int, list[dict]] = {bits: [] for bits in scratch_candidates}
    phase_maxima: dict[str, dict] = defaultdict(lambda: {
        "records": 0, "max_numerator_bits": 0, "max_denominator_bits": 0,
        "max_scratch_numerator_bits": 0, "max_scratch_denominator_bits": 0,
    })
    integer_maxima: dict[str, dict] = defaultdict(lambda: {"records": 0, "max_bits": 0,
                                                            "witness": None})
    max_state_num = max_state_den = max_scratch_num = max_scratch_den = 0
    operation_count = 0
    for label, run in all_runs:
        max_state_num = max(max_state_num, run["max_normalized_numerator_bits"])
        max_state_den = max(max_state_den, run["max_normalized_denominator_bits"])
        max_scratch_num = max(max_scratch_num,
                              run["max_conservative_scratch_numerator_bits"])
        max_scratch_den = max(max_scratch_den,
                              run["max_conservative_scratch_denominator_bits"])
        operation_count += sum(v["count"] for v in run["rational_widths"].values())
        for key, width in run["rational_widths"].items():
            phase = key.split("/", 1)[0]
            row = phase_maxima[phase]
            row["records"] += width["count"]
            for field, source in (
                ("max_numerator_bits", "max_numerator_bits"),
                ("max_denominator_bits", "max_denominator_bits"),
                ("max_scratch_numerator_bits", "max_scratch_numerator_bits"),
                ("max_scratch_denominator_bits", "max_scratch_denominator_bits"),
            ):
                row[field] = max(row[field], width[source])
        for key, width in run["integer_widths"].items():
            phase = key.split("/", 1)[0]
            row = integer_maxima[phase]
            row["records"] += width["count"]
            if width["max_bits"] > row["max_bits"]:
                row["max_bits"] = width["max_bits"]
                row["witness"] = {"field": key.split("/", 1)[-1],
                                   "value": width["witness"]}
        if (run["max_normalized_numerator_bits"] > state_limit or
                run["max_normalized_denominator_bits"] > state_limit):
            state_over.append(label)
        for bits in scratch_candidates:
            for operation, widths in run["rational_widths"].items():
                if (widths["max_scratch_numerator_bits"] > bits or
                        widths["max_scratch_denominator_bits"] > bits):
                    scratch_over[bits].append({
                        "run": label, "operation": operation,
                        "numerator_bits": widths["max_scratch_numerator_bits"],
                        "denominator_bits": widths["max_scratch_denominator_bits"],
                        "witness": widths["scratch_witness"],
                    })
    first_256 = min(scratch_over[256],
                    key=lambda x: max(x["numerator_bits"], x["denominator_bits"]))
    return {
        "observed_operation_result_count": operation_count,
        "state_component_limit_bits": state_limit,
        "max_observed_normalized_numerator_bits": max_state_num,
        "max_observed_normalized_denominator_bits": max_state_den,
        "normalized_256_bit_state_fits_all_captured_runs": not state_over,
        "normalized_state_overflow_runs": state_over,
        "max_observed_conservative_scratch_numerator_bits": max_scratch_num,
        "max_observed_conservative_scratch_denominator_bits": max_scratch_den,
        "phase_maxima": dict(sorted(phase_maxima.items())),
        "integer_phase_maxima": dict(sorted(integer_maxima.items())),
        "scratch_candidates": {
            str(bits): {
                "fits_all_captured_runs": not scratch_over[bits],
                "first_failure": scratch_over[bits][0] if scratch_over[bits] else None,
                "failure_count_by_operation_sample": len(scratch_over[bits]),
            }
            for bits in scratch_candidates
        },
        "interpretation": [
            "The 256-bit normalized-state check passes these captured traces, including exact stop/resume and clamped skip multiplication.",
            "A 256-bit scratch candidate fails on actual traced operations; the recorded comparison products reach 271 bits and one large-output stop/resume addition reaches 259 bits.",
            "544-bit scratch contains every observed conservative operation bound. 768 bits is the proposed simple production scratch width to allow checked 256-bit rational operands and up-to-128-bit skip multipliers with carry headroom.",
            "Observed fit is not a universal bound for future facts, geometries, or all supported Runtime inputs. Production must check every operand/result and fail closed on any width overflow; no tile fallback or cap reset.",
        ],
    }


def main() -> int:
    module = load_solver()
    fixtures = [
        ("facts_strict_small.json", True, 256),
        ("facts_int4_native_small.json", True, 256),
        ("facts_int4_converted_fractional.json", True, 256),
        ("facts_int4_converted_storage.json", True, 256),
        ("facts_dense.json", True, 256),
        ("facts_dense.json", False, 5000),
        ("facts_large_k.json", True, 256),
        ("facts_large_outputs.json", True, 256),
        ("facts_output_pressure_requested.json", True, 256),
        ("facts_output_pressure_k1_tiles128.json", True, 256),
        ("facts_output_pressure_k2_tiles128.json", True, 256),
    ]
    runs = [run_fixture(module, FACT_DIR / name, accelerate=accelerate, cap=cap)
            for name, accelerate, cap in fixtures]
    stop_module = load_solver(STOP_STUDY, "solo_stop_13_audited")
    stop_fixtures = [
        ("strict_small", False), ("int4_native_small", False),
        ("int4_converted_fractional", False),
        ("int4_converted_storage", False), ("dense", False),
        ("output_pressure_k1_tiles128", False),
        ("large_k", True), ("large_outputs", True),
    ]
    stop_runs = [run_stop_fixture(stop_module, fixture, huge=huge)
                 for fixture, huge in stop_fixtures]
    doc = {
        "schema_version": 1,
        "scope": "offline audit of captured actual Runtime class facts using the unchanged exact Solo13 solver; normalized rational result widths plus conservative binary-operation scratch bounds; not a production certificate",
        "inputs": {
            "solver_path": str(STUDY), "solver_sha256": sha256(STUDY),
            "audit_script_path": str(Path(__file__).resolve()),
            "audit_script_sha256": sha256(Path(__file__).resolve()),
            "stop_solver_path": str(STOP_STUDY),
            "stop_solver_sha256": sha256(STOP_STUDY),
            "stop_validation_path": str(STOP_VALIDATION),
            "stop_validation_sha256": sha256(STOP_VALIDATION),
            "fixed_limb_prototype": {
                "header": str(OUT_DIR / "checked_rational256.hh"),
                "header_sha256": sha256(OUT_DIR / "checked_rational256.hh"),
                "test_source": str(OUT_DIR / "rational256_test.cc"),
                "test_source_sha256": sha256(OUT_DIR / "rational256_test.cc"),
                "test_runner": str(OUT_DIR / "check_rational256.py"),
                "test_runner_sha256": sha256(OUT_DIR / "check_rational256.py"),
                "test_results": str(OUT_DIR / "rational256_test_results.json"),
                "test_results_sha256": sha256(OUT_DIR / "rational256_test_results.json"),
                "binary": str(OUT_DIR / "rational256_test"),
                "binary_sha256": sha256(OUT_DIR / "rational256_test"),
            },
            "facts_directory": str(FACT_DIR),
            "facts_sha256": {p.name: sha256(p) for p in sorted(FACT_DIR.glob("facts_*.json"))},
            "producer_rerun": False,
        },
        "measurement_notes": [
            "Each AuditFraction arithmetic result is exact and normalized as a Python Fraction before its numerator/denominator width is recorded.",
            "Conservative scratch width is the unreduced cross-product bound inferred from the two normalized operands for add/subtract/multiply/divide; it is not a claim about CPython's private temporary allocation sequence.",
            "The audit instruments Fraction operations throughout input loading, progressive fill, event/deadline arithmetic, exact identity comparisons, ledger accumulation, and skip translation/multiplication.",
            "Integer width samples cover geometry/class counts and every applied skip repeat, step, and translated-wave count; the solver's fixed event cap remains global per run.",
            "The Python Fraction implementation is measurement instrumentation only; production must use checked bounded arithmetic or outward intervals and fail closed on overflow/ambiguity.",
        ],
        "runs": runs,
        "stop_and_resume_runs": stop_runs,
        "fixed_width_budget_audit": budget_analysis(runs, stop_runs),
    }
    out = OUT_DIR / "bitwidth_audit.json"
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(out), "runs": [
        {k: row[k] for k in ("fixture", "mode", "status", "events", "descriptors",
                             "max_normalized_numerator_bits",
                             "max_normalized_denominator_bits",
                             "max_conservative_scratch_numerator_bits",
                             "max_conservative_scratch_denominator_bits",
                             "max_integer_bits")}
        for row in runs],
        "stop_runs": [{k: row[k] for k in ("fixture", "stop_kind", "status_sequence",
                                             "final_events", "max_normalized_numerator_bits",
                                             "max_normalized_denominator_bits",
                                             "max_conservative_scratch_numerator_bits",
                                             "max_conservative_scratch_denominator_bits",
                                             "max_integer_bits")}
                      for row in stop_runs],
        "fixed_width_budget_audit": doc["fixed_width_budget_audit"]}, indent=2))
    return 0 if all(row["status"] == "passed" for row in runs) and all(
        row["status_sequence"][-1] == "passed" for row in stop_runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
