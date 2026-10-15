#!/usr/bin/env python3
"""Compare a compiler-selected GEMM plan with an independent K-wave oracle.

The driver calls the public C++ compile() path and measures its selected Plan
through the actual Runtime. This Python file independently expands only the
small test fixture into output-tile/K-tile consumptions. It is offline evidence
for model-to-model granularity error, not a production execution path or a
claim about measured hardware.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

RESOURCE_IDS = {
    "COMPUTE_GATE": 0,
    "SRAM_READ": 1,
    "SRAM_WRITE": 2,
    "LOCAL": 3,
    "HBM0": 4,
    "HBM1": 5,
    "INJECT0": 6,
    "INJECT1": 7,
    "XPOS": 8,
    "XNEG": 9,
    "YPOS": 10,
    "YNEG": 11,
    "HOP": 12,
}
RESOURCE_NAMES = list(RESOURCE_IDS)
NS_PER_SECOND = 1_000_000_000
EPS = 1.0e-12
NONBINDING_CAPACITY = 1.0e30
READINESS_SEMANTICS_REVISION = "final_k_output_reservation_v2"


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), EPS)


def role_resource_work(work: list[float], role: str, index: int) -> float:
    return float(work[index])


@dataclass(frozen=True)
class Wave:
    output_tile: int
    k_tile: int
    final_k: bool
    read_row_bytes: tuple[int, int]
    write_row_bytes: tuple[int, int]
    vec_ops: int
    compute_work_s: float
    eta_hbm: float
    # Canonical compute quantity remains LOCAL payload even when that physical
    # domain is nonbinding. The legacy equal-progress allocator ignores Q.
    compute_local_bytes: int = 0

    def resource_work(self) -> dict[str, list[float]]:
        load = [0.0] * len(RESOURCE_NAMES)
        compute = [0.0] * len(RESOURCE_NAMES)
        store = [0.0] * len(RESOURCE_NAMES)
        load[RESOURCE_IDS["HBM0"]] = self.read_row_bytes[0] / self.eta_hbm
        load[RESOURCE_IDS["HBM1"]] = self.read_row_bytes[1] / self.eta_hbm
        compute[RESOURCE_IDS["COMPUTE_GATE"]] = self.compute_work_s
        store[RESOURCE_IDS["HBM0"]] = self.write_row_bytes[0] / self.eta_hbm
        store[RESOURCE_IDS["HBM1"]] = self.write_row_bytes[1] / self.eta_hbm
        return {"load": load, "compute": compute, "store": store}


@dataclass
class Active:
    request: int
    stage: str
    wave: int
    work: list[float]
    private_time_s: float = 0.0
    progress: float = 0.0


@dataclass
class Request:
    name: str
    arrival_s: float
    waves: list[Wave]
    output_count: int
    started: bool = False
    next_load: int = 0
    loaded: list[int] = field(default_factory=list)
    computed: list[int] = field(default_factory=list)
    active_load: Active | None = None
    active_compute: Active | None = None
    active_store: Active | None = None
    retired_outputs: int = 0
    finish_s: float | None = None


def row0_partition(total: int, prefix: int) -> int:
    """Partition a byte stream exactly like ceil(total/2), floor(total/2)."""
    return (prefix + total + 1) // 2 - (prefix + 1) // 2


def make_waves(raw: dict[str, Any], *, require_five_epoch_fixture: bool = True) -> tuple[list[Wave], dict[str, Any]]:
    plan = raw["compiled_plan"]
    summary = plan["summary"]
    task = plan["task"]
    shape = task["tensor_shapes"]
    m, n, k = (int(shape[name]) for name in ("m", "n", "k"))
    tm, tn, tk = (int(summary[name]) for name in ("tm", "tn", "tk"))
    hw = raw["synthetic_hardware"]
    if int(hw["core_count"]) != 1 or int(hw["core_columns_nx"]) != 1 or int(hw["core_rows_ny"]) != 1:
        raise ValueError("fixture precondition failed: expected exactly one Core")
    if summary.get("dataflow_id") != "output_stationary_v2":
        raise ValueError("fixture precondition failed: selected dataflow is not OS")
    if task.get("execution_profile") != "fp32_vec_matrix_assumed_v1":
        raise ValueError("fixture precondition failed: expected strict FP32 Vec profile")
    if any(task["storage_dtype"].get(name) != "fp32"
           for name in ("weights", "activations", "output")):
        raise ValueError("fixture precondition failed: all storage dtypes must be FP32")
    if require_five_epoch_fixture and plan["E"] != 5:
        raise ValueError(f"fixture expected the available five macro epochs, got E={plan['E']}")

    tm_count, tn_count, tk_count = ceildiv(m, tm), ceildiv(n, tn), ceildiv(k, tk)
    output_rounds = ceildiv(tm_count * tn_count, 1)
    expected_wave_count = output_rounds * tk_count
    if int(summary["output_round_count"]) != output_rounds:
        raise ValueError("compiled output_round_count disagrees with selected tile geometry")
    if int(summary["consumption_wave_count"]) != expected_wave_count:
        raise ValueError("compiled consumption_wave_count is not R*Tk")

    p_vec = float(hw["p_vec_per_core_ops_per_s"])
    eta_vec = float(hw["eta_vec"])
    eta_hbm = float(hw["eta_hbm_matrix"])
    hbm_rows = [float(hw["resource_capacities"][i]["capacity"])
                for i in (RESOURCE_IDS["HBM0"], RESOURCE_IDS["HBM1"])]
    if p_vec <= 0 or eta_vec <= 0 or eta_hbm <= 0 or min(hbm_rows) <= 0:
        raise ValueError("fixture has a nonpositive compute or HBM capacity")

    waves: list[Wave] = []
    read_prefix = 0
    write_prefix = 0
    total_read = total_write = total_ops = 0
    accum_read = accum_write = 0
    output_tile_classes: dict[str, int] = {}
    k_tile_extent_classes: dict[str, int] = {}
    for q in range(tm_count * tn_count):
        m_tile, n_tile = divmod(q, tn_count)
        m_elems = min(tm, m - m_tile * tm)
        n_elems = min(tn, n - n_tile * tn)
        output_tile_class = f"{m_elems}x{n_elems}"
        output_tile_classes[output_tile_class] = (
            output_tile_classes.get(output_tile_class, 0) + 1)
        for k_tile in range(tk_count):
            k_elems = min(tk, k - k_tile * tk)
            k_tile_extent_class = str(k_elems)
            k_tile_extent_classes[k_tile_extent_class] = (
                k_tile_extent_classes.get(k_tile_extent_class, 0) + 1)
            read = 4 * k_elems * (m_elems + n_elems)
            ops = 2 * m_elems * n_elems * k_elems
            final_k = k_tile + 1 == tk_count
            write = 4 * m_elems * n_elems if final_k else 0
            row0_read = row0_partition(read, read_prefix)
            row1_read = read - row0_read
            if read:
                read_prefix += read
            row0_write = row0_partition(write, write_prefix)
            row1_write = write - row0_write
            if write:
                write_prefix += write
            waves.append(Wave(q, k_tile, final_k,
                              (row0_read, row1_read),
                              (row0_write, row1_write), ops,
                              ops / (p_vec * eta_vec), eta_hbm,
                              read + 4 * m_elems * n_elems * (2 if k_tile else 1)))
            total_read += read
            total_write += write
            total_ops += ops
            accum_write += 4 * m_elems * n_elems
            if k_tile:
                accum_read += 4 * m_elems * n_elems

    expected_sram_read = total_read + accum_read + total_write
    expected_sram_write = total_read + accum_write
    expected_local = total_read + accum_read + accum_write
    max_m, max_n, max_k = min(m, tm), min(n, tn), min(k, tk)
    input_tile_bytes = 4 * max_m * max_k + 4 * max_k * max_n
    accumulator_tile_bytes = 4 * max_m * max_n
    pending_output_slots = 2 * 4 * max_m * max_n
    sram_peak = 2 * input_tile_bytes + accumulator_tile_bytes + pending_output_slots
    expected = {
        "shape": {"m": m, "n": n, "k": k},
        "selected_tiles": {"tm": tm, "tn": tn, "tk": tk,
                           "Tm": tm_count, "Tn": tn_count, "Tk": tk_count},
        "output_round_count": output_rounds,
        "consumption_wave_count": expected_wave_count,
        "output_tile_classes_m_by_n": output_tile_classes,
        "k_tile_extent_classes_by_wave": k_tile_extent_classes,
        "has_k_tail": k % tk != 0,
        "waves_with_store": sum(1 for wave in waves if wave.final_k),
        "true_output_tiles": tm_count * tn_count,
        "hbm_read_bytes": total_read,
        "hbm_write_bytes": total_write,
        "vec_equivalent_ops": total_ops,
        "conversion_elements": 0,
        "int4_input_bytes": 0,
        "executed_int4_ops": 0,
        "accumulator_write_bytes": accum_write,
        "accumulator_read_bytes": accum_read,
        "sram_read_bytes": expected_sram_read,
        "sram_write_bytes": expected_sram_write,
        "local_bus_bytes": expected_local,
        "sram_peak_bytes_per_core": sram_peak,
        "sram_capacity_bytes_per_core": int(hw["core_sram_bytes_per_core"]),
        "rho_sram_capacity_bytes_per_core": (int(hw["core_sram_bytes_per_core"])
                                               * float(hw["rho_sram"])),
        "logical_tensor_bytes": 4 * (m * k + k * n + m * n),
        "hbm_read_by_row": [read_prefix // 2 + read_prefix % 2,
                            read_prefix // 2],
        "hbm_write_by_row": [write_prefix // 2 + write_prefix % 2,
                             write_prefix // 2],
        "compute_work_s": total_ops / (p_vec * eta_vec),
        "hbm_eta": eta_hbm,
    }
    return waves, expected


def aggregate_plan_ledger(plan: dict[str, Any]) -> tuple[list[float], list[list[float]], list[list[int]], int, int]:
    total_work = [0.0] * len(RESOURCE_NAMES)
    role_work = {name: [0.0] * len(RESOURCE_NAMES)
                 for name in ("load", "compute", "store")}
    hbm_bytes = [[0, 0], [0, 0]]
    vec_ops = int4_ops = 0
    for epoch in plan["epochs"]:
        if list(epoch["credit_capacity"]) != [2.0, 2.0]:
            raise ValueError(f"compiled epoch {epoch['name']} does not declare two input/output slots")
        for activity in epoch["activities"]:
            role = activity["role"]
            for index, work in enumerate(activity["work_by_resource"]):
                total_work[index] += float(work)
                if role in role_work:
                    role_work[role][index] += float(work)
            for row in range(2):
                for direction in range(2):
                    hbm_bytes[row][direction] += int(
                        activity["hbm_bytes_by_row_direction"][row][direction])
            vec_ops += int(activity["vec_eqops"])
            int4_ops += int(activity["int4_ops"])
    return (total_work, [role_work[k] for k in ("load", "compute", "store")],
            hbm_bytes, vec_ops, int4_ops)


def check_ledger(raw: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    plan = raw["compiled_plan"]
    summary = plan["summary"]
    work, role_work, hbm, activity_vec_ops, activity_int4_ops = aggregate_plan_ledger(plan)
    eta_hbm = expected["hbm_eta"]
    independently_derived = [None] * len(RESOURCE_NAMES)
    independently_derived[RESOURCE_IDS["COMPUTE_GATE"]] = expected["compute_work_s"]
    independently_derived[RESOURCE_IDS["SRAM_READ"]] = expected["sram_read_bytes"]
    independently_derived[RESOURCE_IDS["SRAM_WRITE"]] = expected["sram_write_bytes"]
    independently_derived[RESOURCE_IDS["LOCAL"]] = expected["local_bus_bytes"]
    independently_derived[RESOURCE_IDS["HBM0"]] = (
        expected["hbm_read_by_row"][0] + expected["hbm_write_by_row"][0]) / eta_hbm
    independently_derived[RESOURCE_IDS["HBM1"]] = (
        expected["hbm_read_by_row"][1] + expected["hbm_write_by_row"][1]) / eta_hbm

    checks = {
        "summary_hbm_read_bytes": int(summary["hbm_read_bytes_total"]) == expected["hbm_read_bytes"],
        "summary_hbm_write_bytes": int(summary["hbm_write_bytes_total"]) == expected["hbm_write_bytes"],
        "summary_vec_equivalent_ops": int(summary["vec_equivalent_ops"]) == expected["vec_equivalent_ops"],
        "summary_conversion_elements": int(summary["conversion_elements"]) == 0,
        "summary_int4_input_bytes": int(summary["int4_input_bytes"]) == 0,
        "summary_executed_int4_ops": int(summary["executed_int4_ops"]) == 0,
        "summary_sram_read_bytes": int(summary["sram_read_bytes"]) == expected["sram_read_bytes"],
        "summary_sram_write_bytes": int(summary["sram_write_bytes"]) == expected["sram_write_bytes"],
        "summary_local_bus_bytes": int(summary["local_bus_bytes"]) == expected["local_bus_bytes"],
        "summary_logical_tensor_bytes": int(summary["logical_tensor_bytes"]) == expected["logical_tensor_bytes"],
        "summary_sram_peak": int(summary["sram_peak_bytes_per_core"]) == expected["sram_peak_bytes_per_core"],
        "sram_capacity_feasible": expected["sram_peak_bytes_per_core"] <= expected["rho_sram_capacity_bytes_per_core"],
        "summary_source_matrix_ops": int(summary["source_matrix_ops"]) == expected["vec_equivalent_ops"],
        "summary_executed_matrix_ops": int(summary["executed_matrix_ops"]) == expected["vec_equivalent_ops"],
        "plan_activity_vec_equivalent_ops": activity_vec_ops == expected["vec_equivalent_ops"],
        "plan_activity_int4_ops": activity_int4_ops == 0,
        "compiled_activity_hbm_bytes": hbm == [
            [expected["hbm_read_by_row"][0], expected["hbm_write_by_row"][0]],
            [expected["hbm_read_by_row"][1], expected["hbm_write_by_row"][1]]],
    }
    numeric_errors: dict[str, float] = {}
    for name, resource in RESOURCE_IDS.items():
        derived = independently_derived[resource]
        if derived is not None:
            numeric_errors[name] = relative_error(work[resource], float(derived))
            checks[f"plan_resource_{name}"] = numeric_errors[name] <= 1.0e-9
    capacity_rows = raw["synthetic_hardware"]["resource_capacities"]
    ledger = []
    nonbinding_ok = True
    finish_reference = float(raw["standalone"]["finish_s"])
    for index, name in enumerate(RESOURCE_NAMES):
        desc = capacity_rows[index]
        capacity = float(desc["capacity"])
        enabled = bool(desc["enabled"])
        service = work[index] / capacity if capacity > 0 else None
        target = name in {"COMPUTE_GATE", "HBM0", "HBM1"}
        if enabled and not target:
            nonbinding = service is not None and service <= max(finish_reference, 1.0e-9) * 1.0e-6
            nonbinding_ok = nonbinding_ok and nonbinding
            classification = "explicit_nonbinding_capacity" if nonbinding else "unexpected_capacity_demand"
        elif not enabled:
            nonbinding = work[index] == 0.0
            nonbinding_ok = nonbinding_ok and nonbinding
            classification = "disabled_and_zero_work" if nonbinding else "disabled_resource_has_work"
        else:
            nonbinding = False
            classification = "oracle_modeled"
        ledger.append({
            "resource_id": index,
            "resource": name,
            "enabled": enabled,
            "capacity_per_s": capacity,
            "compiled_activity_total_work": work[index],
            "independently_derived_work": independently_derived[index],
            "minimum_service_s": service,
            "classification": classification,
            "nonbinding_proved_by_fixture_capacity": nonbinding,
        })
    checks["nonbinding_resources"] = nonbinding_ok
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "plan_resource_relative_errors": numeric_errors,
        "resources": ledger,
        "compiled_activity_role_work": {
            role: dict(zip(RESOURCE_NAMES, values))
            for role, values in zip(("load", "compute", "store"), role_work)
        },
        "compiled_hbm_payload_by_row_direction": hbm,
        "compiled_activity_vec_equivalent_ops": activity_vec_ops,
        "compiled_activity_int4_ops": activity_int4_ops,
        "repeat_count_treatment": "summed each activity once; repeat_count is audit metadata",
    }


def fill_rates(active: list[Active], capacities: list[float]) -> list[float]:
    """Progressive filling for equal-priority discrete activities.

    Activities start at equal normalized progress. A saturated physical domain
    freezes the activities using it; activities on independent domains can
    continue. This small fixture has two HBM rows and one compute domain.
    """
    count = len(active)
    rates = [0.0] * count
    free = set(range(count))
    for _ in range(count + 1):
        if not free:
            break
        delta = math.inf
        limiting_resources: set[int] = set()
        for resource in (RESOURCE_IDS["COMPUTE_GATE"],
                         RESOURCE_IDS["HBM0"], RESOURCE_IDS["HBM1"]):
            used = sum(active[i].work[resource] * rates[i]
                       for i in range(count))
            slope = sum(active[i].work[resource] for i in free)
            if slope <= 0.0:
                continue
            available = capacities[resource] - used
            candidate = max(0.0, available) / slope
            if candidate < delta - 1.0e-15:
                delta = candidate
                limiting_resources = {resource}
            elif abs(candidate - delta) <= 1.0e-15:
                limiting_resources.add(resource)
        private_limiter: set[int] = set()
        for i in free:
            if active[i].private_time_s > 0.0:
                bound = 1.0 / active[i].private_time_s
                candidate = max(0.0, bound - rates[i])
                if candidate < delta - 1.0e-15:
                    delta = candidate
                    limiting_resources = set()
                    private_limiter = {i}
                elif abs(candidate - delta) <= 1.0e-15:
                    private_limiter.add(i)
        if not math.isfinite(delta):
            raise RuntimeError("oracle active set has no finite resource bound")
        if delta < -EPS:
            raise RuntimeError("oracle progressive fill found a negative rate increment")
        for i in free:
            rates[i] += max(0.0, delta)
        frozen: set[int] = set(private_limiter)
        for resource in limiting_resources:
            frozen.update(i for i in free if active[i].work[resource] > 0.0)
        if not frozen:
            raise RuntimeError("oracle progressive fill did not freeze a limiting activity")
        free.difference_update(frozen)
    if free:
        raise RuntimeError("oracle progressive fill exceeded its fixed activity bound")
    for resource in (RESOURCE_IDS["COMPUTE_GATE"],
                     RESOURCE_IDS["HBM0"], RESOURCE_IDS["HBM1"]):
        used = sum(active[i].work[resource] * rates[i] for i in range(count))
        if used > capacities[resource] * (1.0 + 1.0e-9):
            raise RuntimeError(
                f"oracle progressive fill exceeded {RESOURCE_NAMES[resource]} capacity: "
                f"used={used} capacity={capacities[resource]}")
    return rates


def discrete_oracle(waves: list[Wave], output_count: int,
                    arrival_times: list[float], capacities: list[float], *,
                    rate_allocator=None,
                    capture_lifecycle_trace: bool = False) -> dict[str, Any]:
    names = ["comp0"] if len(arrival_times) == 1 else ["comp0", "comp1"]
    requests = [Request(name, arrival, waves, output_count)
                for name, arrival in zip(names, arrival_times)]
    now = 0.0
    hbm_busy = compute_busy = shared_hbm = shared_compute = 0.0
    row_busy = [0.0, 0.0]
    row_shared = [0.0, 0.0]
    served_work = [0.0] * len(RESOURCE_NAMES)
    stage_completions = {"load": 0, "compute": 0, "store": 0}
    event_count = 0
    max_events = 1000000
    lifecycle_trace: list[dict[str, Any]] = []
    wave_ledger = {
        "wave_count_per_request": len(waves),
        "final_k_wave_count_per_request": sum(wave.final_k for wave in waves),
        "hbm_read_bytes_by_row_per_request": [
            sum(wave.read_row_bytes[row] for wave in waves) for row in (0, 1)],
        "hbm_write_bytes_by_row_per_request": [
            sum(wave.write_row_bytes[row] for wave in waves) for row in (0, 1)],
        "vec_equivalent_ops_per_request": sum(wave.vec_ops for wave in waves),
    }

    def request_done(request: Request) -> bool:
        return request.retired_outputs >= request.output_count

    def output_occupancy(request: Request) -> int:
        return (len(request.computed) + int(request.active_store is not None) +
                int(request.active_compute is not None and
                    waves[request.active_compute.wave].final_k))

    def record_lifecycle(event: str, request: Request, **details: Any) -> None:
        if not capture_lifecycle_trace:
            return
        lifecycle_trace.append({"time_s": now, "event": event,
                                "request": request.name,
                                "output_occupancy": output_occupancy(request),
                                **details})

    def start_eligible() -> None:
        changed = True
        while changed:
            changed = False
            for ri, request in enumerate(requests):
                if not request.started or request.finish_s is not None:
                    continue
                input_occupancy = (len(request.loaded) +
                                   int(request.active_load is not None) +
                                   int(request.active_compute is not None))
                if (request.active_load is None and request.next_load < len(waves)
                        and input_occupancy < 2):
                    wi = request.next_load
                    request.next_load += 1
                    request.active_load = Active(ri, "load", wi,
                                                 waves[wi].resource_work()["load"])
                    record_lifecycle("stage_start", request, stage="load",
                                     wave=wi, final_k=waves[wi].final_k)
                    changed = True
                if request.active_compute is None and request.loaded:
                    wi = request.loaded[0]
                    output_before = output_occupancy(request)
                    final_k = waves[wi].final_k
                    if not final_k or output_before < 2:
                        request.loaded.pop(0)
                        request.active_compute = Active(
                            ri, "compute", wi,
                            waves[wi].resource_work()["compute"],
                            waves[wi].compute_work_s)
                        record_lifecycle(
                            "stage_start", request, stage="compute", wave=wi,
                            final_k=final_k,
                            output_occupancy_before=output_before,
                            output_occupancy_after_reservation=(
                                output_before + int(final_k)))
                        changed = True
                    else:
                        record_lifecycle(
                            "compute_waiting_for_output_slot", request, wave=wi,
                            final_k=final_k,
                            output_occupancy_before=output_before)
                if request.active_store is None and request.computed:
                    wi = request.computed.pop(0)
                    request.active_store = Active(ri, "store", wi,
                                                  waves[wi].resource_work()["store"])
                    record_lifecycle("stage_start", request, stage="store",
                                     wave=wi, final_k=waves[wi].final_k)
                    changed = True

    while not all(request_done(request) for request in requests):
        for request in requests:
            if not request.started and request.arrival_s <= now + EPS:
                request.started = True
                event_count += 1
        start_eligible()
        active: list[Active] = []
        active_refs: list[tuple[Request, str]] = []
        for request in requests:
            if request.finish_s is not None:
                continue
            for stage in ("load", "compute", "store"):
                item = getattr(request, f"active_{stage}")
                if item is not None:
                    active.append(item)
                    active_refs.append((request, stage))
        next_arrival = min((request.arrival_s for request in requests
                            if not request.started), default=math.inf)
        if not active:
            if not math.isfinite(next_arrival):
                raise RuntimeError("discrete oracle deadlocked with unfinished outputs")
            now = next_arrival
            continue

        rates = (fill_rates(active, capacities) if rate_allocator is None
                 else rate_allocator(active, capacities))
        completions: list[tuple[Request, str, int]] = []
        deltas = []
        for item, rate in zip(active, rates):
            if rate <= 0.0:
                deltas.append(math.inf)
            else:
                deltas.append(max(0.0, 1.0 - item.progress) / rate)
        delta = min(min(deltas), next_arrival - now)
        if not math.isfinite(delta) or delta < -EPS:
            raise RuntimeError("discrete oracle produced an invalid event interval")

        active_hbm_by_row: list[set[int]] = [set(), set()]
        active_compute_owners: set[int] = set()
        for item, rate in zip(active, rates):
            if rate <= 0.0:
                continue
            if item.work[RESOURCE_IDS["HBM0"]] > 0.0:
                active_hbm_by_row[0].add(item.request)
            if item.work[RESOURCE_IDS["HBM1"]] > 0.0:
                active_hbm_by_row[1].add(item.request)
            if item.work[RESOURCE_IDS["COMPUTE_GATE"]] > 0.0:
                active_compute_owners.add(item.request)
        if active_hbm_by_row[0] or active_hbm_by_row[1]:
            hbm_busy += delta
        if active_compute_owners:
            compute_busy += delta
        any_shared_hbm = False
        for row in (0, 1):
            if active_hbm_by_row[row]:
                row_busy[row] += delta
            if len(active_hbm_by_row[row]) >= 2:
                row_shared[row] += delta
                any_shared_hbm = True
        if any_shared_hbm:
            shared_hbm += delta
        if len(active_compute_owners) >= 2:
            shared_compute += delta

        for item, rate in zip(active, rates):
            item.progress = min(1.0, item.progress + rate * delta)
            for resource, amount in enumerate(item.work):
                served_work[resource] += amount * rate * delta
        now += delta
        for request, stage in active_refs:
            item = getattr(request, f"active_{stage}")
            if item is None or item.progress < 1.0 - 1.0e-10:
                continue
            wi = item.wave
            output_before = output_occupancy(request)
            setattr(request, f"active_{stage}", None)
            if stage == "load":
                request.loaded.append(wi)
            elif stage == "compute":
                if waves[wi].final_k:
                    request.computed.append(wi)
            elif stage == "store":
                request.retired_outputs += 1
            stage_completions[stage] += 1
            event_count += 1
            record_lifecycle("stage_complete", request, stage=stage, wave=wi,
                             final_k=waves[wi].final_k,
                             output_occupancy_before=output_before)
            if event_count > max_events:
                raise RuntimeError("discrete oracle exceeded the fixed event budget")
        for request in requests:
            if request.finish_s is None and request_done(request):
                request.finish_s = now
        if delta <= EPS and next_arrival > now + EPS:
            raise RuntimeError("discrete oracle made no progress")

    result = {
        "readiness_semantics_revision": READINESS_SEMANTICS_REVISION,
        "finish_s_by_request": {request.name: request.finish_s for request in requests},
        "hbm_busy_s": hbm_busy,
        "hbm_shared_s": shared_hbm,
        "compute_busy_s": compute_busy,
        "compute_shared_s": shared_compute,
        "row_busy_s": row_busy,
        "row_shared_s": row_shared,
        "served_work_by_resource": dict(zip(RESOURCE_NAMES, served_work)),
        "discrete_wave_work_ledger": wave_ledger,
        "stage_completions": stage_completions,
        "event_count": event_count,
        "buffer_slots": {"input": 2, "output": 2},
        "dependency_rules": {
            "compute_requires_completed_load": True,
            "nonfinal_k_store_bytes": 0,
            "store_only_after_each_output_true_final_k": True,
            "output_slot_lifetime": "reserved at final-K compute start; freed only after copyout completes",
        },
    }
    if capture_lifecycle_trace:
        result["lifecycle_trace"] = lifecycle_trace
    return result


def make_capacities(raw: dict[str, Any]) -> list[float]:
    capacities = [float(item["capacity"])
                  for item in raw["synthetic_hardware"]["resource_capacities"]]
    for name in RESOURCE_NAMES:
        if name not in {"COMPUTE_GATE", "HBM0", "HBM1"}:
            if capacities[RESOURCE_IDS[name]] > 0.0:
                capacities[RESOURCE_IDS[name]] = NONBINDING_CAPACITY
    return capacities


def compare_measurement(actual: dict[str, Any], oracle: dict[str, Any],
                       phase_offset_s: float, concurrent: bool) -> dict[str, Any]:
    actual_counters = actual["counters"]
    oracle_finishes = oracle["finish_s_by_request"]
    finish_errors: dict[str, float] = {}
    if concurrent:
        actual_finishes = actual["finish_s_by_request"]
        for name in ("comp0", "comp1"):
            finish_errors[name] = relative_error(float(actual_finishes[name]),
                                                 float(oracle_finishes[name]))
        finish_limit = 0.05
    else:
        actual_finish = float(actual["finish_s"])
        finish_errors["solo"] = relative_error(actual_finish,
                                                float(oracle_finishes["comp0"]))
        finish_limit = 0.03
    actual_hbm_busy = float(actual_counters["hbm_busy_ns"]) / NS_PER_SECOND
    hbm_error = relative_error(actual_hbm_busy, float(oracle["hbm_busy_s"]))
    actual_shared = float(actual_counters["hbm_shared_ns"]) / NS_PER_SECOND
    shared_error = relative_error(actual_shared, float(oracle["hbm_shared_s"]))
    shared_required = concurrent and float(oracle["hbm_shared_s"]) > 0.0
    requests = 2 if concurrent else 1
    ledger = oracle["discrete_wave_work_ledger"]
    exact_hbm = actual_counters.get("hbm_served_bytes_by_row_direction_exact")
    exact_categories = actual_counters.get("served_by_category")
    runtime_work_reconciliation: dict[str, Any]
    required_counter_fields = []
    if exact_hbm is None:
        required_counter_fields.append("hbm_served_bytes_by_row_direction_exact")
    if exact_categories is None or any(
            "vec_eqops_exact_decimal" not in item
            for item in exact_categories.values()):
        required_counter_fields.append("served_by_category.vec_eqops_exact_decimal")
    if required_counter_fields:
        runtime_work_reconciliation = {
            "status": "not_available",
            "available": False,
            "reason": "Runtime report lacks serialized long-double work counters",
            "missing_counter_fields": required_counter_fields,
        }
    else:
        expected_hbm = [
            [ledger["hbm_read_bytes_by_row_per_request"][row] * requests,
             ledger["hbm_write_bytes_by_row_per_request"][row] * requests]
            for row in (0, 1)]
        actual_hbm_decimal = [[Decimal(str(value)) for value in row]
                               for row in exact_hbm]
        actual_vec_ops = sum((Decimal(item["vec_eqops_exact_decimal"])
                              for item in exact_categories.values()), Decimal(0))
        expected_vec_ops = ledger["vec_equivalent_ops_per_request"] * requests
        hbm_deltas = []
        strict_hbm_equal = True
        within_hbm_tolerance = True
        tolerance = Decimal("1e-9")
        for row in (0, 1):
            row_deltas = []
            for direction in (0, 1):
                wanted = Decimal(expected_hbm[row][direction])
                actual_value = actual_hbm_decimal[row][direction]
                delta = actual_value - wanted
                relative_delta = abs(delta) / max(Decimal(1), abs(wanted))
                strict_hbm_equal = strict_hbm_equal and delta == 0
                within_hbm_tolerance = within_hbm_tolerance and relative_delta <= tolerance
                row_deltas.append({
                    "actual_serialized_decimal": str(actual_value),
                    "expected_integer": expected_hbm[row][direction],
                    "signed_delta": str(delta),
                    "absolute_delta": str(abs(delta)),
                    "relative_delta": str(relative_delta),
                    "strict_integer_equality": delta == 0,
                    "within_relative_tolerance": relative_delta <= tolerance,
                })
            hbm_deltas.append(row_deltas)
        vec_delta = actual_vec_ops - Decimal(expected_vec_ops)
        vec_relative_delta = abs(vec_delta) / max(Decimal(1), Decimal(expected_vec_ops))
        strict_vec_equal = vec_delta == 0
        within_vec_tolerance = vec_relative_delta <= tolerance
        history_complete = actual_counters.get("accounting_history_complete")
        strict_integer_equality = strict_hbm_equal and strict_vec_equal
        within_relative_tolerance = within_hbm_tolerance and within_vec_tolerance
        if history_complete is False:
            reconciliation_status = "incomplete_history"
        elif within_relative_tolerance:
            reconciliation_status = "within_relative_tolerance"
        else:
            reconciliation_status = "outside_relative_tolerance"
        runtime_work_reconciliation = {
            "status": reconciliation_status,
            "available": True,
            "counter_arithmetic": "serialized_cumulative_long_double_decimal",
            "planned_expected_arithmetic": "independent_integer_wave_ledger",
            "strict_integer_equality": strict_integer_equality,
            "within_relative_tolerance": within_relative_tolerance,
            "relative_tolerance": str(tolerance),
            "accounting_history_complete": history_complete,
            "hbm_bytes_by_row_direction": hbm_deltas,
            "vec_equivalent_ops": {
                "actual_serialized_decimal": str(actual_vec_ops),
                "expected_integer": expected_vec_ops,
                "signed_delta": str(vec_delta),
                "absolute_delta": str(abs(vec_delta)),
                "relative_delta": str(vec_relative_delta),
                "strict_integer_equality": strict_vec_equal,
                "within_relative_tolerance": within_vec_tolerance,
            },
        }
    row_busy_actual = [float(value) / NS_PER_SECOND
                       for value in actual_counters.get("row_busy_ns", [0, 0])]
    row_shared_actual = [float(value) / NS_PER_SECOND
                         for value in actual_counters.get("row_shared_ns", [0, 0])]
    row_busy_errors = [relative_error(row_busy_actual[row],
                                      float(oracle["row_busy_s"][row]))
                       for row in (0, 1)]
    row_shared_errors = [relative_error(row_shared_actual[row],
                                        float(oracle["row_shared_s"][row]))
                         for row in (0, 1)]
    statuses = {
        "completion_time": max(finish_errors.values(), default=math.inf) <= finish_limit,
        "hbm_busy_time": hbm_error <= 0.05,
        "shared_hbm_busy_time": (shared_error <= 0.05 if shared_required
                                 else not shared_required),
    }
    return {
        "phase_offset_s": phase_offset_s,
        "actual_runtime": {
            "finish_s_by_request": ({k: float(v) for k, v in actual["finish_s_by_request"].items()}
                                     if concurrent else {"solo": float(actual["finish_s"])}),
            "hbm_busy_s": actual_hbm_busy,
            "hbm_shared_s": actual_shared,
            "compute_busy_s": float(actual_counters["compute_busy_ns"]) / NS_PER_SECOND,
            "internal_events": int(actual_counters["internal_events"]),
            "allocation_updates": int(actual_counters["allocation_updates"]),
            "accounting_history_complete": actual_counters.get(
                "accounting_history_complete"),
            "serialized_hbm_served_bytes_counter_decimal": exact_hbm,
            "served_by_category_serialized_decimal_counters": exact_categories,
            "row_busy_s": row_busy_actual,
            "row_shared_s": row_shared_actual,
        },
        "independent_discrete_oracle": oracle,
        "runtime_cumulative_serialization_reconciliation": runtime_work_reconciliation,
        "relative_errors": {
            "finish_by_request": finish_errors,
            "hbm_busy": hbm_error,
            "shared_hbm_busy": shared_error,
            "row_busy_by_hbm_row": row_busy_errors,
            "row_shared_by_hbm_row": row_shared_errors,
        },
        "limits": {"completion": finish_limit, "hbm_busy": 0.05,
                   "shared_hbm_busy": 0.05 if shared_required else None},
        "checks": statuses,
        "status": "passed" if all(statuses.values()) else "failed",
    }


def build_report(raw: dict[str, Any]) -> dict[str, Any]:
    waves, expected = make_waves(raw)
    ledger = check_ledger(raw, expected)
    integer_check_names = (
        "summary_hbm_read_bytes", "summary_hbm_write_bytes",
        "summary_vec_equivalent_ops", "summary_conversion_elements",
        "summary_int4_input_bytes", "summary_executed_int4_ops",
        "summary_sram_read_bytes", "summary_sram_write_bytes",
        "summary_local_bus_bytes", "summary_logical_tensor_bytes",
        "summary_sram_peak", "summary_source_matrix_ops",
        "summary_executed_matrix_ops", "plan_activity_vec_equivalent_ops",
        "plan_activity_int4_ops", "compiled_activity_hbm_bytes",
    )
    integer_checks = {name: ledger["checks"][name]
                      for name in integer_check_names}
    planned_integer_conservation = {
        "status": "passed" if all(integer_checks.values()) else "failed",
        "arithmetic": "exact_integer_equality",
        "checks": integer_checks,
        "expected_integer_work": {
            "hbm_read_bytes": expected["hbm_read_bytes"],
            "hbm_write_bytes": expected["hbm_write_bytes"],
            "vec_equivalent_ops": expected["vec_equivalent_ops"],
            "hbm_bytes_by_row_direction": [
                [expected["hbm_read_by_row"][0], expected["hbm_write_by_row"][0]],
                [expected["hbm_read_by_row"][1], expected["hbm_write_by_row"][1]],
            ],
        },
        "compiled_plan_integer_work": {
            "hbm_bytes_by_row_direction": ledger[
                "compiled_hbm_payload_by_row_direction"],
            "vec_equivalent_ops": ledger["compiled_activity_vec_equivalent_ops"],
            "int4_ops": ledger["compiled_activity_int4_ops"],
        },
    }
    capacities = make_capacities(raw)
    solo_oracle = discrete_oracle(waves, expected["true_output_tiles"], [0.0], capacities)
    solo_reference = {
        **solo_oracle,
        "finish_s_by_request": {"comp0": solo_oracle["finish_s_by_request"]["comp0"]},
    }
    solo = compare_measurement(raw["standalone"], solo_reference, 0.0, False)

    concurrent_results = []
    for actual in raw["two_comp"]:
        offset = float(actual["phase_offset_s"])
        oracle = discrete_oracle(waves, expected["true_output_tiles"],
                                 [0.0, offset], capacities)
        result = compare_measurement(actual, oracle, offset, True)
        concurrent_results.append(result)

    plan = raw["compiled_plan"]
    summary = plan["summary"]
    tile_info = expected["selected_tiles"]
    frame_credit_units = int(summary["output_round_count"])
    k_waves_per_output = int(tile_info["Tk"])
    append_epoch_units = frame_credit_units
    root_cause = {
        "compiled_macro_E": plan["E"],
        "available_E": [5],
        "E8": "unavailable_production_has_no_E8_template",
        "E16": "unavailable_production_has_no_E16_template",
        "append_epoch_frame_units_inferred_from_output_round_credit": append_epoch_units,
        "actual_output_round_count_R": frame_credit_units,
        "k_consumptions_per_output_round_Tk": k_waves_per_output,
        "independent_expected_R_times_Tk": frame_credit_units * k_waves_per_output,
        "compiled_plan_summary_consumption_wave_count": int(summary["consumption_wave_count"]),
        "factor_between_consumer_waves_and_frame_units": k_waves_per_output,
        "k_tile_extents": expected["k_tile_extent_classes_by_wave"],
        "m_n_output_tile_classes": expected["output_tile_classes_m_by_n"],
        "finding": (
            "The compiler summary reports R*Tk consumption waves, while append_epoch receives R output rounds. "
            "Its five macro sections aggregate K work at output-round granularity; the independent oracle releases "
            "compute only after each discrete K-wave load and produces/stores an output only after that output "
            "tile's final K wave. Shape and K extent classes are reported separately for this probe."
        ),
    }

    all_fixture_checks = [ledger["status"] == "passed", solo["status"] == "passed"]
    all_fixture_checks.extend(item["status"] == "passed" for item in concurrent_results)
    fixture_status = "passed" if all(all_fixture_checks) else "failed"
    return {
        "schema_version": 1,
        "oracle": "independent_compiled_gemm_discrete_consumption_wave_v2",
        "readiness_semantics_revision": READINESS_SEMANTICS_REVISION,
        "fixture_kind": "synthetic_model_to_model",
        "measurement_source": raw["measurement_source"],
        "compiled_plan": {
            "compile_path": plan["compiler_path"],
            "fingerprint": plan["fingerprint"],
            "hardware_fingerprint": plan["hardware_fingerprint"],
            "E": plan["E"],
            "candidate_count": summary["candidate_count"],
            "selected_template": summary["selected_template"],
            "dataflow_id": summary["dataflow_id"],
            "tiles": {key: summary[key] for key in ("tm", "tn", "tk")},
            "output_round_count": summary["output_round_count"],
            "consumption_wave_count": summary["consumption_wave_count"],
            "summary": summary,
            "epoch_names_repeat_and_credit": [
                {"name": epoch["name"], "repeat_count_audit_only": epoch["repeat_count"],
                 "credit_initial": epoch["credit_initial"],
                 "credit_capacity": epoch["credit_capacity"],
                 "activity_roles": [activity["role"] for activity in epoch["activities"]]}
                for epoch in plan["epochs"]],
        },
        "synthetic_hardware": raw["synthetic_hardware"],
        "fixture_scope": {
            "shape": expected["shape"],
            "core_count": raw["synthetic_hardware"]["core_count"],
            "dataflow": "single-core output-stationary",
            "profile": "fp32_vec_matrix_assumed_v1 with FP32 storage",
            "buffer_slots": {"input": 2, "output": 2},
            "copyout_rule": "output tile occupies a slot from final-K production until store completion",
            "compute_ready_rule": "a K-wave compute starts only after that exact K-wave load is complete",
            "nonfinal_k_store_rule": "zero store activity for every nonfinal K tile; one store after each output tile final K",
            "resource_dominance": "HBM0/HBM1 and COMPUTE_GATE are finite; all other enabled capacities are 1e30 and disabled paths carry zero work",
            "hardware_evidence": "model_to_model_fixture_only; no silicon or external measured data",
        },
        "independent_expected_work": expected,
        "planned_integer_conservation": planned_integer_conservation,
        "compiled_plan_ledger": ledger,
        "comparisons": {"standalone": solo, "two_comp": concurrent_results},
        "acceptance_limits": {
            "standalone_finish_relative_error": 0.03,
            "concurrent_finish_relative_error": 0.05,
            "hbm_busy_relative_error": 0.05,
            "shared_hbm_busy_relative_error": 0.05,
        },
        "available_granularity_levels": {
            "available_E": [5],
            "E5_fixture_comparison": fixture_status,
            "E8": {"status": "not_run", "reason": "production compiler/runtime provides no E8 expansion"},
            "E16": {"status": "not_run", "reason": "production compiler/runtime provides no E16 expansion"},
            "selected_minimum_E": None,
            "granularity_validation": ("failed" if fixture_status == "failed" else "not_run"),
            "scope_note": "One fixture cannot certify broad granularity coverage, even if this fixture meets its error limits.",
        },
        "root_cause_clue": root_cause,
        "status": fixture_status,
        "notes": [
            "The C++ side used coarse_die::compile() and then the actual public Runtime; no Plan was hand-constructed.",
            "The Python oracle alone enumerates this small fixture's output tiles and K tiles.",
            "Activity totals were summed once; repeat_count was never multiplied into work.",
            "A failed threshold remains failed. E8/E16 remain not_run because no such production templates exist.",
        ],
    }


def driver_shape_args(m: int | None, n: int | None,
                      k: int | None) -> list[str]:
    supplied = (m is not None, n is not None, k is not None)
    if any(supplied) and not all(supplied):
        raise ValueError("M, N, and K must be supplied together")
    if not any(supplied):
        return []
    if min(int(m), int(n), int(k)) <= 0:
        raise ValueError("M, N, and K must be positive")
    return ["--m", str(m), "--n", str(n), "--k", str(k)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path,
                        default=Path("work/build-die/coarse_compiled_gemm_oracle_driver"),
                        help="compiled C++ driver that uses compile() and actual Runtime")
    parser.add_argument("--m", type=int, help="GEMM M dimension; supply M/N/K together")
    parser.add_argument("--n", type=int, help="GEMM N dimension; supply M/N/K together")
    parser.add_argument("--k", type=int, help="GEMM K dimension; supply M/N/K together")
    parser.add_argument("--axis", choices=("k_depth", "k_tail", "mn_shape", "baseline"),
                        help="single-factor axis label stored in the report")
    parser.add_argument("--probe-name", help="human-readable probe label stored in the report")
    parser.add_argument("--output", type=Path,
                        help="write the JSON report here; stdout is used when omitted")
    args = parser.parse_args()
    try:
        shape_args = driver_shape_args(args.m, args.n, args.k)
    except ValueError as error:
        parser.error(str(error))
    driver = args.driver.expanduser().resolve()
    if not driver.is_file():
        payload = json.dumps({
            "status": "not_run",
            "granularity_validation": "not_run",
            "reason": f"compiled-plan Runtime driver is unavailable: {driver}",
            "available_granularity_levels": {
                "available_E": [],
                "E5": "not_run",
                "E8": "not_run_unavailable",
                "E16": "not_run_unavailable",
            },
        }, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
        return 2
    completed = subprocess.run([str(driver), *shape_args], capture_output=True, text=True,
                               check=False)
    if completed.returncode != 0:
        payload = {
            "status": "failed",
            "granularity_validation": "failed",
            "driver": str(driver),
            "driver_return_code": completed.returncode,
            "driver_stderr": completed.stderr,
            "available_granularity_levels": {
                "available_E": [],
                "E5": "failed_before_comparison",
                "E8": "not_run_unavailable",
                "E16": "not_run_unavailable",
            },
        }
        report = json.dumps(payload, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report + "\n", encoding="utf-8")
        else:
            print(report)
        return 1
    try:
        raw = json.loads(completed.stdout)
        report_obj = build_report(raw)
        shape = raw["compiled_plan"]["task"]["tensor_shapes"]
        summary = raw["compiled_plan"]["summary"]
        report_obj["probe"] = {
            "name": args.probe_name,
            "axis": args.axis,
            "requested_shape": {name: int(shape[name]) for name in ("m", "n", "k")},
            "selected_tiles": {name: int(summary[name]) for name in ("tm", "tn", "tk")},
            "selected_tile_signature": "x".join(
                str(int(summary[name])) for name in ("tm", "tn", "tk")),
        }
    except Exception as error:  # Keep failed evidence machine-readable.
        report_obj = {
            "status": "failed",
            "granularity_validation": "failed",
            "driver": str(driver),
            "driver_stderr": completed.stderr,
            "driver_output_prefix": completed.stdout[:2000],
            "failure": f"{type(error).__name__}: {error}",
            "available_granularity_levels": {
                "available_E": [],
                "E5": "failed_during_oracle_or_ledger",
                "E8": "not_run_unavailable",
                "E16": "not_run_unavailable",
            },
        }
    report = json.dumps(report_obj, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)
    return 0 if report_obj.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
