#!/usr/bin/env python3
"""Offline discrete consumption-wave oracle for the fixed-buffer pipeline.

This is deliberately a small, independent event simulator. It consumes only
explicit wave work from this file/JSON input and does not call or import the
coarse_die compiler, estimator, or runtime. It is a model-to-model granularity
check, not evidence about physical hardware.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EPS = 1.0e-12


@dataclass(frozen=True)
class Wave:
    load_bytes: float
    compute_ops: float
    store_bytes: float


@dataclass
class Request:
    name: str
    arrival_s: float
    waves: list[Wave]
    next_load: int = 0
    loaded: list[int] = field(default_factory=list)
    computed: list[int] = field(default_factory=list)
    load_active: tuple[int, float] | None = None
    compute_active: tuple[int, float] | None = None
    store_active: tuple[int, float] | None = None
    finish_s: float | None = None
    started: bool = False
    completions: list[tuple[int, str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class Active:
    request: int
    stage: str
    wave: int
    resource: str


def partition_waves(waves: list[Wave], epoch_count: int) -> list[Wave]:
    if epoch_count < 1 or epoch_count > len(waves):
        raise ValueError("epoch_count must be in [1, number of waves]")
    n = len(waves)
    boundaries = [(i * n) // epoch_count for i in range(epoch_count + 1)]
    grouped: list[Wave] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        chunk = waves[start:stop]
        if not chunk:
            raise ValueError("granularity partition created an empty service epoch")
        grouped.append(
            Wave(
                sum(w.load_bytes for w in chunk),
                sum(w.compute_ops for w in chunk),
                sum(w.store_bytes for w in chunk),
            )
        )
    return grouped


def simulate(request_specs: list[dict[str, Any]], load_slots: int,
             output_slots: int, hbm_bytes_per_s: float,
             compute_ops_per_s: float) -> dict[str, Any]:
    if load_slots < 1 or output_slots < 1:
        raise ValueError("buffer capacities must be positive")
    if hbm_bytes_per_s <= 0 or compute_ops_per_s <= 0:
        raise ValueError("oracle resource rates must be positive")
    requests = [
        Request(
            str(spec["name"]),
            float(spec.get("arrival_s", 0.0)),
            [Wave(float(row["load_bytes"]), float(row["compute_ops"]),
                  float(row["store_bytes"])) for row in spec["waves"]],
        )
        for spec in request_specs
    ]
    for request in requests:
        if request.arrival_s < 0 or not request.waves:
            raise ValueError("requests need a nonnegative arrival and at least one wave")
        for wave in request.waves:
            if min(wave.load_bytes, wave.compute_ops, wave.store_bytes) < 0:
                raise ValueError("wave work must be nonnegative")

    rates = {"hbm": hbm_bytes_per_s, "compute": compute_ops_per_s}
    now = 0.0
    events = 0
    busy = {"hbm": 0.0, "compute": 0.0}
    shared_hbm = 0.0
    trace: list[dict[str, Any]] = []

    def active_activities() -> list[Active]:
        active: list[Active] = []
        for ri, r in enumerate(requests):
            if not r.started or r.finish_s is not None:
                continue
            if r.load_active is not None:
                active.append(Active(ri, "load", r.load_active[0], "hbm"))
            if r.compute_active is not None:
                active.append(Active(ri, "compute", r.compute_active[0], "compute"))
            if r.store_active is not None:
                active.append(Active(ri, "store", r.store_active[0], "hbm"))
        return active

    def start_eligible() -> None:
        changed = True
        while changed:
            changed = False
            for r in requests:
                if not r.started or r.finish_s is not None:
                    continue
                input_occupancy = len(r.loaded) + int(r.load_active is not None) + int(r.compute_active is not None)
                output_occupancy = len(r.computed) + int(r.compute_active is not None) + int(r.store_active is not None)
                if (r.load_active is None and r.next_load < len(r.waves)
                        and input_occupancy < load_slots):
                    wi = r.next_load
                    r.next_load += 1
                    r.load_active = (wi, r.waves[wi].load_bytes)
                    changed = True
                if r.compute_active is None and r.loaded and output_occupancy < output_slots:
                    wi = r.loaded.pop(0)
                    r.compute_active = (wi, r.waves[wi].compute_ops)
                    changed = True
                if r.store_active is None and r.computed:
                    wi = r.computed.pop(0)
                    r.store_active = (wi, r.waves[wi].store_bytes)
                    changed = True

    def stage_remaining(request: Request, stage: str) -> float:
        state = getattr(request, f"{stage}_active")
        return float(state[1]) if state is not None else 0.0

    def finish_stage(ri: int, stage: str) -> None:
        nonlocal events
        r = requests[ri]
        state = getattr(r, f"{stage}_active")
        if state is None:
            return
        wi = state[0]
        setattr(r, f"{stage}_active", None)
        if stage == "load":
            r.loaded.append(wi)
        elif stage == "compute":
            r.computed.append(wi)
        else:
            r.completions.append((wi, stage, now))
            if wi == len(r.waves) - 1:
                r.finish_s = now
        events += 1
        trace.append({"tick_s": now, "request": r.name, "stage": stage, "wave": wi})

    while True:
        for r in requests:
            if not r.started and r.arrival_s <= now + EPS:
                r.started = True
                events += 1
        start_eligible()
        if all(r.finish_s is not None for r in requests):
            break

        active = active_activities()
        next_arrival = min((r.arrival_s for r in requests if not r.started), default=math.inf)
        if not active:
            if math.isinf(next_arrival):
                raise RuntimeError("oracle deadlocked with unfinished requests")
            now = next_arrival
            continue

        per_activity_rate: dict[tuple[int, str], float] = {}
        for resource, capacity in rates.items():
            uses = [a for a in active if a.resource == resource]
            if not uses:
                continue
            by_request: dict[int, list[Active]] = {}
            for activity in uses:
                by_request.setdefault(activity.request, []).append(activity)
            owners = len(by_request)
            for ri, activities in by_request.items():
                share = capacity / owners / len(activities)
                for activity in activities:
                    per_activity_rate[(ri, activity.stage)] = share

        completion_deltas: list[float] = []
        for activity in active:
            rate = per_activity_rate[(activity.request, activity.stage)]
            rem = stage_remaining(requests[activity.request], activity.stage)
            if rem <= EPS:
                completion_deltas.append(0.0)
            else:
                completion_deltas.append(rem / rate)
        delta = min(min(completion_deltas), next_arrival - now)
        if delta < -EPS or math.isinf(delta):
            raise RuntimeError("oracle produced an invalid next event")

        active_by_resource: dict[str, list[Active]] = {"hbm": [], "compute": []}
        for activity in active:
            active_by_resource[activity.resource].append(activity)
        for resource, uses in active_by_resource.items():
            if uses:
                busy[resource] += delta
                if resource == "hbm" and len({a.request for a in uses}) >= 2:
                    shared_hbm += delta
        for activity in active:
            ri, stage = activity.request, activity.stage
            request = requests[ri]
            wi, remaining = getattr(request, f"{stage}_active")
            rate = per_activity_rate[(ri, stage)]
            updated = max(0.0, remaining - rate * delta)
            setattr(request, f"{stage}_active", (wi, updated))
        now += delta

        completed: list[tuple[int, str]] = []
        for activity in active:
            if stage_remaining(requests[activity.request], activity.stage) <= EPS:
                completed.append((activity.request, activity.stage))
        for ri, stage in completed:
            finish_stage(ri, stage)
        if delta <= EPS and not completed and next_arrival > now + EPS:
            raise RuntimeError("oracle made no progress")

    return {
        "elapsed_s": now,
        "finish_by_request_s": {r.name: r.finish_s for r in requests},
        "busy_s": busy,
        "shared_hbm_s": shared_hbm,
        "event_count": events,
        "trace": trace,
    }


def fixture_requests(offset_s: float = 0.0,
                     shape: str = "balanced") -> list[dict[str, Any]]:
    fixture_path = (Path(__file__).resolve().parents[1] /
                    "tests/fixtures/discrete_wave_scenarios.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    for scenario in fixture["scenarios"]:
        if scenario["shape"] == shape and math.isclose(
                scenario["phase_offset_s"], offset_s,
                rel_tol=0.0, abs_tol=1e-12):
            return copy.deepcopy(scenario["requests"])
    raise KeyError(f"missing discrete-wave fixture: {shape} phase_offset={offset_s}")


def relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), EPS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "oracle_granularity_report.json")
    parser.add_argument("--hbm-bytes-per-s", type=float, default=1000.0)
    parser.add_argument("--compute-ops-per-s", type=float, default=8.0)
    parser.add_argument("--input-slots", type=int, default=2)
    parser.add_argument("--output-slots", type=int, default=2)
    parser.add_argument("--runtime-driver", type=Path,
                        help="compiled oracle_granularity_test executable for real C++ Runtime comparison")
    args = parser.parse_args()
    if args.hbm_bytes_per_s <= 0 or args.compute_ops_per_s <= 0:
        parser.error("resource rates must be positive")

    scenarios = []
    passing_max = 0
    scenario_specs = [("single", 0.0)]
    for shape in ("balanced", "compute_long_memory_short", "memory_long_compute_short", "restore_burst"):
        scenario_specs.extend((shape, offset) for offset in (0.0, 0.37, 1.20))
    for shape, offset in scenario_specs:
            req = fixture_requests(offset, shape)
            exact = simulate(req, args.input_slots, args.output_slots,
                             args.hbm_bytes_per_s, args.compute_ops_per_s)
            levels = {}
            for epoch_count in (5, 8, 16):
                grouped = [
                    {**request,
                     "waves": [w.__dict__ for w in partition_waves(
                         [Wave(**wave) for wave in request["waves"]], epoch_count
                     )]}
                    for request in req
                ]
                result = simulate(grouped, args.input_slots, args.output_slots,
                                  args.hbm_bytes_per_s, args.compute_ops_per_s)
                completion_errors = {
                    name: relative_error(result["finish_by_request_s"][name], exact["finish_by_request_s"][name])
                    for name in exact["finish_by_request_s"]
                }
                busy_error = relative_error(result["busy_s"]["hbm"], exact["busy_s"]["hbm"])
                shared_error = relative_error(result["shared_hbm_s"], exact["shared_hbm_s"])
                is_concurrent = len(req) > 1
                completion_limit = 0.05 if is_concurrent else 0.03
                passed = (max(completion_errors.values(), default=0.0) <= completion_limit
                          and busy_error <= 0.05
                          and (shared_error <= 0.05 or exact["shared_hbm_s"] == 0))
                levels[str(epoch_count)] = {
                    "completion_relative_error_by_request": completion_errors,
                    "max_completion_relative_error": max(completion_errors.values(), default=0.0),
                    "completion_error_limit": completion_limit,
                    "hbm_busy_relative_error": busy_error,
                    "shared_hbm_relative_error": shared_error,
                    "event_count": result["event_count"],
                    "elapsed_s": result["elapsed_s"],
                    "finish_by_request_s": result["finish_by_request_s"],
                    "passed": passed,
                }
                if passed and (passing_max == 0 or epoch_count < passing_max):
                    passing_max = epoch_count
            is_concurrent = "B" in exact["finish_by_request_s"]
            exact_order = (
                exact["finish_by_request_s"]["A"] <
                exact["finish_by_request_s"]["B"]
            ) if is_concurrent else None
            reversed_at: list[int] = []
            undetermined_at: list[int] = []
            if is_concurrent:
                for level in (5, 8, 16):
                    result = levels[str(level)]
                    fa = result["finish_by_request_s"]["A"]
                    fb = result["finish_by_request_s"]["B"]
                    if abs(fa - fb) <= 0.05 * max(fa, fb, EPS):
                        undetermined_at.append(level)
                    elif (fa < fb) != exact_order:
                        reversed_at.append(level)
            scenarios.append({
                "shape": shape,
                "phase_offset_s": offset,
                "oracle_event_count": exact["event_count"],
                "oracle_elapsed_s": exact["elapsed_s"],
                "oracle_finish_by_request_s": exact["finish_by_request_s"],
                "oracle_hbm_busy_s": exact["busy_s"]["hbm"],
                "oracle_shared_hbm_s": exact["shared_hbm_s"],
                "exact_finish_order": (
                    "single_request" if not is_concurrent else
                    ("A_before_B" if exact_order else "B_before_or_tied_A")
                ),
                "ranking_reversal_levels": reversed_at,
                "undetermined_levels": undetermined_at,
                "levels": levels,
            })

    # The oracle's E=16 partition is one group per reference wave. This only
    # checks the Python oracle's grouping path; it is not a library certificate.
    passing_levels = []
    for level in (5, 8, 16):
        if all(s["levels"][str(level)]["passed"] for s in scenarios):
            passing_levels.append(level)
    selected = min(passing_levels) if passing_levels else None
    report: dict[str, Any] = {
        "schema_version": 1,
        "oracle": "independent_discrete_consumption_wave_v1",
        "evidence_scope": "synthetic_grouped_runtime",
        "status": "oracle_self_check" if selected is not None else "failed",
        "oracle_self_check": "passed" if selected is not None else "failed",
        "coarse_library_granularity": "not_run",
        "granularity_validation": "not_run",
        "oracle_self_check_minimum_E": selected,
        "selected_minimum_E": None,
        "tested_E": [5, 8, 16],
        "standalone_completion_error_limit": 0.03,
        "concurrent_completion_error_limit": 0.05,
        "shared_hbm_busy_error_limit": 0.05,
        "buffer_slots": {"input": args.input_slots, "output": args.output_slots},
        "resource_capacities": {"HBM_Bps": args.hbm_bytes_per_s,
                                 "COMPUTE_GATE_ops_per_s": args.compute_ops_per_s},
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "notes": [
            "E=16 retains one record per discrete consumption wave and is the Python oracle reference partition",
            "the oracle uses event-driven wave readiness and finite input/output slots",
            "this is model-to-model evidence only and does not validate silicon rates",
            "oracle self-check does not certify or claim the public C++ library granularity",
            "the 3%/5% budgets are frozen from design documentation",
        ],
    }

    if args.runtime_driver:
        driver = args.runtime_driver.expanduser().resolve()
        report["runtime_driver"] = str(driver)
        if not driver.is_file():
            report["coarse_library_reason"] = "compiled C++ Runtime driver is unavailable"
            report["status"] = "not_run"
        else:
            started = time.perf_counter_ns()
            try:
                completed = subprocess.run(
                    [str(driver), "--hbm-bytes-per-s", str(args.hbm_bytes_per_s),
                     "--compute-ops-per-s", str(args.compute_ops_per_s),
                     "--input-slots", str(args.input_slots),
                     "--output-slots", str(args.output_slots)],
                    capture_output=True, text=True, check=False)
            except OSError as exc:
                report["coarse_library_reason"] = str(exc)
                report["status"] = "failed"
            else:
                report["runtime_driver_elapsed_ns"] = time.perf_counter_ns() - started
                report["runtime_driver_return_code"] = completed.returncode
                report["runtime_driver_stderr"] = completed.stderr
                if completed.returncode != 0:
                    report["coarse_library_reason"] = (
                        f"C++ Runtime driver exited {completed.returncode}")
                    report["runtime_driver_stdout"] = completed.stdout
                    report["status"] = "failed"
                else:
                    try:
                        runtime_payload = json.loads(completed.stdout)
                    except json.JSONDecodeError as exc:
                        report["coarse_library_reason"] = (
                            f"C++ Runtime driver output is not JSON: {exc}")
                        report["runtime_driver_stdout"] = completed.stdout
                        report["status"] = "failed"
                    else:
                        actual_rows = runtime_payload.get("measurements", [])
                        actual_map = {
                            (row.get("shape"), float(row.get("phase_offset_s", 0.0)),
                             int(row.get("E", 0))): row
                            for row in actual_rows
                        }
                        expected_keys = {
                            (scenario["shape"], float(scenario["phase_offset_s"]), E)
                            for scenario in scenarios for E in (5, 8, 16)
                        }
                        missing = sorted(expected_keys - set(actual_map))
                        level_results: dict[str, dict[str, Any]] = {}
                        for E in (5, 8, 16):
                            max_completion_errors: list[float] = []
                            busy_errors: list[float] = []
                            shared_errors: list[float] = []
                            passed_all = not missing
                            compared = 0
                            wall_ns = cpu_ns = event_total = 0
                            for scenario in scenarios:
                                key = (scenario["shape"],
                                       float(scenario["phase_offset_s"]), E)
                                row = actual_map.get(key)
                                if row is None:
                                    passed_all = False
                                    continue
                                expected_finish = scenario["oracle_finish_by_request_s"]
                                actual_finish = row.get("finish_by_request_s", {})
                                if set(expected_finish) != set(actual_finish):
                                    passed_all = False
                                    continue
                                finish_errors = {
                                    name: relative_error(float(actual_finish[name]),
                                                         float(expected_finish[name]))
                                    for name in expected_finish
                                }
                                max_error = max(finish_errors.values(), default=0.0)
                                completion_limit = 0.05 if len(finish_errors) > 1 else 0.03
                                busy_error = relative_error(
                                    float(row["hbm_busy_s"]),
                                    float(scenario["oracle_hbm_busy_s"]))
                                shared_error = relative_error(
                                    float(row["hbm_shared_s"]),
                                    float(scenario["oracle_shared_hbm_s"]))
                                scenario_passed = (
                                    max_error <= completion_limit and busy_error <= 0.05 and
                                    (shared_error <= 0.05 or
                                     scenario["oracle_shared_hbm_s"] == 0)
                                )
                                passed_all = passed_all and scenario_passed
                                compared += 1
                                max_completion_errors.append(max_error)
                                busy_errors.append(busy_error)
                                shared_errors.append(shared_error)
                                wall_ns += int(row.get("host_wall_ns", 0))
                                cpu_ns += int(row.get("host_cpu_ns", 0))
                                event_total += int(row.get("internal_events", 0))
                                scenario["levels"][str(E)]["coarse_library"] = {
                                    "completion_relative_error_by_request": finish_errors,
                                    "max_completion_relative_error": max_error,
                                    "completion_error_limit": completion_limit,
                                    "hbm_busy_relative_error": busy_error,
                                    "shared_hbm_relative_error": shared_error,
                                    "runtime_finish_by_request_s": actual_finish,
                                    "runtime_hbm_busy_s": row["hbm_busy_s"],
                                    "runtime_hbm_shared_s": row["hbm_shared_s"],
                                    "runtime_internal_events": row.get("internal_events"),
                                    "runtime_allocation_updates": row.get("allocation_updates"),
                                    "runtime_host_wall_ns": row.get("host_wall_ns"),
                                    "runtime_host_cpu_ns": row.get("host_cpu_ns"),
                                    "passed": scenario_passed,
                                }
                            level_results[str(E)] = {
                                "passed_every_scenario": passed_all,
                                "compared_scenarios": compared,
                                "max_completion_relative_error": max(max_completion_errors,
                                                                      default=None),
                                "max_hbm_busy_relative_error": max(busy_errors, default=None),
                                "max_shared_hbm_relative_error": max(shared_errors,
                                                                      default=None),
                                "runtime_internal_events_total": event_total,
                                "runtime_host_wall_ns_total": wall_ns,
                                "runtime_host_cpu_ns_total": cpu_ns,
                            }
                        passing_library = [E for E in (5, 8, 16)
                                           if level_results[str(E)]["passed_every_scenario"]]
                        chosen_library = min(passing_library) if passing_library else None
                        granularity_status = "passed" if chosen_library is not None else "failed"
                        report["status"] = granularity_status
                        report["runtime_grouped_granularity"] = granularity_status
                        report["runtime_grouped_minimum_E"] = chosen_library
                        report["runtime_grouped_levels"] = level_results
                        report["runtime_grouped_missing_scenarios"] = missing
                        report["runtime_grouped_measurement_status"] = runtime_payload.get("status")
                        report["runtime_grouped_measurement_count"] = len(actual_rows)
                        report["notes"].append(
                            "public C++ Runtime measurements use manually assembled grouped-wave Plans; compiled operator templates remain unvalidated")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"oracle status={report['status']} library_granularity={report['coarse_library_granularity']} "
          f"oracle_self_check_E={selected} selected_E={report['selected_minimum_E']} "
          f"scenarios={len(scenarios)} output={args.output}")
    if report["status"] == "passed":
        return 0
    if report["status"] == "failed":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
