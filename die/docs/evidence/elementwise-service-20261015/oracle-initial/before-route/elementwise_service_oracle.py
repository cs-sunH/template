#!/usr/bin/env python3
"""Offline, single-Core discrete elementwise reference, never production code.

Derives real 256-element blocks from task facts, not production credit curves.
Explicitly compares invocation materialization barriers with a second reference
that permits independent invocations to pipeline. All 13 shared resource rows,
owner Q/alpha and PRIVATE bounds participate; routing rows are nonbinding in the
supported synthetic fixture. No calibration parameters are fitted.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

NAMES = ("COMPUTE_GATE", "SRAM_READ", "SRAM_WRITE", "LOCAL", "HBM0", "HBM1",
         "INJECT0", "INJECT1", "XPOS", "XNEG", "YPOS", "YNEG", "HOP")
TARGET, FALLBACK = 0.10, 0.20
NS = 1_000_000_000


def storage(n: int, dtype: str) -> int:
    if dtype == "int4":
        return (n + 1) // 2
    return n * {"bf16": 2, "fp16": 2, "fp32": 4}[dtype]


@dataclass
class Block:
    invocation: int
    elements: int
    read_bytes: int
    write_bytes: int
    eqops: int
    rows_read: tuple[int, int]
    rows_write: tuple[int, int]


@dataclass
class Activity:
    request: int
    block: int
    role: str
    work: list[float]
    private_time_s: float
    quantity: float
    progress: float = 0.0


def grouped_rates(active: list[Activity], capacities: list[float]) -> list[float]:
    """Nonnegative progressive filling, same owner definition as grouped oracle.

    Each eligible activity has alpha=Q/sum_owner(Q), hence v increases by
    floor/sum_owner(Q). This discrete reference has no signed fluid constraints.
    This generalizes grouped_gemm_oracle's three-domain reference to all 13.
    """
    totals: dict[int, float] = {}
    for a in active:
        if a.quantity <= 0 or not math.isfinite(a.quantity):
            raise ValueError("invalid canonical quantity")
        if any(w > 0 and c <= 0 for w, c in zip(a.work, capacities)):
            raise ValueError("fixture demanded a disabled resource")
        totals[a.request] = totals.get(a.request, 0.0) + a.quantity
    weights = [1 / totals[a.request] for a in active]
    rates = [0.0] * len(active)
    free = set(range(len(active)))
    for _ in active:
        if not free:
            break
        rows: list[tuple[float, set[int]]] = []
        for r, cap in enumerate(capacities):
            slope = math.fsum(active[i].work[r] * weights[i] for i in free)
            if slope:
                used = math.fsum(a.work[r] * rates[i] for i, a in enumerate(active))
                residual = cap - used
                if residual < -1e-9 * max(1.0, cap):
                    raise ArithmeticError("resource budget exceeded")
                rows.append((max(0.0, residual) / slope,
                             {i for i in free if active[i].work[r] > 0}))
        for i in free:
            private = active[i].private_time_s
            if private:
                rows.append((max(0.0, 1 / private - rates[i]) / weights[i], {i}))
        if not rows:
            raise ArithmeticError("unbounded oracle activity")
        step = min(bound for bound, _ in rows)
        for i in free:
            rates[i] += weights[i] * step
        frozen: set[int] = set()
        for bound, members in rows:
            if abs(bound - step) <= 1e-11 * max(abs(step), abs(bound), 1e-30):
                frozen.update(members)
        if not frozen:
            raise ArithmeticError("no fairness bottleneck frozen")
        free.difference_update(frozen)
    if free:
        raise ArithmeticError("fairness tier bound exceeded")
    for r, cap in enumerate(capacities):
        if math.fsum(a.work[r] * rates[i] for i, a in enumerate(active)) > cap * (1 + 1e-9):
            raise ArithmeticError("final oracle resource budget exceeded")
    return rates


def make_blocks(raw: dict[str, Any]) -> tuple[list[Block], dict[str, Any]]:
    f, h = raw["fixture"], raw["hardware"]
    if int(h["derived"]["die"]["core_count"]) != 1:
        raise ValueError("this oracle certifies only the declared one-Core synthetic scope")
    n, repeats = int(f["elements_per_invocation"]), int(f["executions"])
    if not 0 < ((n + 255) // 256) * repeats <= 4096:
        raise ValueError("offline reference limited to 4096 blocks")
    streams = 1 if f["op"] == "gelu" else 2
    ops = {"residual": 1, "gelu": 15, "swiglu": 16}[f["op"]]
    blocks: list[Block] = []
    prefix = [0, 0]
    # Byte-striped two-row fixture. Prefix parity makes row payloads integral
    # and preserves ceil(total/2), floor(total/2) over repeated odd tails.
    def split(payload: int, direction: int) -> tuple[int, int]:
        old = prefix[direction]
        prefix[direction] += payload
        row0 = (prefix[direction] + 1) // 2 - (old + 1) // 2
        return row0, payload - row0
    for invocation in range(repeats):
        for begin in range(0, n, 256):
            size = min(256, n - begin)
            read = streams * storage(size, f["input_dtype"])
            write = storage(size, f["output_dtype"])
            blocks.append(Block(invocation, size, read, write, size * ops,
                                split(read, 0), split(write, 1)))
    reads, writes = sum(b.read_bytes for b in blocks), sum(b.write_bytes for b in blocks)
    maximum = min(n, 256)
    slots = 1 if len(blocks) == 1 else 2
    expected = {"hbm_read_bytes_total": reads, "hbm_write_bytes_total": writes,
                "logical_tensor_bytes": reads + writes,
                "vec_equivalent_ops": sum(b.eqops for b in blocks),
                "source_vec_ops": sum(b.eqops for b in blocks),
                "sram_read_bytes": reads + writes, "sram_write_bytes": reads + writes,
                "local_bus_bytes": reads + writes, "output_round_count": len(blocks),
                "consumption_wave_count": len(blocks),
                "sram_peak_bytes_per_core": slots * (streams * storage(maximum, f["input_dtype"])
                                                    + storage(maximum, f["output_dtype"])) + 4 * maximum,
                "slots_per_input_output": slots,
                "blocks_per_invocation": (n + 255) // 256}
    return blocks, expected


def factory(raw: dict[str, Any], blocks: list[Block]):
    hw = raw["hardware"]
    caps = [float(r["capacity"]) for r in hw["resources"]]
    peak = float(hw["derived"]["core"]["P_vec_per_core_ops_per_s"])
    eta_v = float(hw["input"]["performance_profile"]["eta_vec"])
    eta_h = float(hw["input"]["performance_profile"]["eta_hbm_vector"])

    def activity(owner: int, block: int, role: str) -> Activity:
        b = blocks[block]
        work = [0.0] * 13
        if role == "load":
            work[2] = b.read_bytes
            work[4], work[5] = (n / eta_h for n in b.rows_read)
            quantity = b.read_bytes
        elif role == "store":
            work[1] = b.write_bytes
            work[4], work[5] = (n / eta_h for n in b.rows_write)
            quantity = b.write_bytes
        else:
            work[0] = b.eqops / (peak * eta_v)
            work[1], work[2] = b.read_bytes, b.write_bytes
            work[3] = b.read_bytes + b.write_bytes
            quantity = b.read_bytes + b.write_bytes
        private = max([work[0]] + [work[r] / caps[r] for r in (1, 2, 3) if work[r]])
        return Activity(owner, block, role, work, private, quantity)
    return caps, activity


def simulate(blocks: list[Block], arrivals: list[float], slots: int,
             capacities: list[float], activity_factory, barrier: bool) -> dict[str, Any]:
    count = len(arrivals)
    # Per block states: not loaded / loading / buffered / computing / produced /
    # storing / stored. Slots remain reserved through their physical lifetime.
    states = [["pending"] * len(blocks) for _ in arrivals]
    live: list[Activity] = []
    now = 0.0
    finish: list[float | None] = [None] * count
    first: list[float | None] = [None] * count
    busy = {"hbm_busy_s": 0.0, "hbm_shared_s": 0.0, "compute_busy_s": 0.0,
            "row_busy_s": [0.0, 0.0], "row_shared_s": [0.0, 0.0]}
    event_count = 0
    while any(value is None for value in finish):
        for owner in range(count):
            if arrivals[owner] > now + 1e-13 or finish[owner] is not None:
                continue
            state = states[owner]
            if all(s == "stored" for s in state):
                finish[owner] = now
                continue
            if not any(a.request == owner and a.role == "store" for a in live):
                candidates = [i for i, s in enumerate(state) if s == "produced"]
                if candidates:
                    index = candidates[0]; state[index] = "storing"
                    live.append(activity_factory(owner, index, "store"))
            if not any(a.request == owner and a.role == "compute" for a in live):
                output_used = sum(s in ("computing", "produced", "storing") for s in state)
                candidates = [i for i, s in enumerate(state) if s == "buffered"]
                if candidates and output_used < slots:
                    index = candidates[0]; state[index] = "computing"
                    live.append(activity_factory(owner, index, "compute"))
            if not any(a.request == owner and a.role == "load" for a in live):
                input_used = sum(s in ("loading", "buffered", "computing") for s in state)
                candidates = [i for i, s in enumerate(state) if s == "pending"]
                if candidates and input_used < slots:
                    index = candidates[0]
                    ready = not barrier or all(s == "stored" for i, s in enumerate(state)
                                               if blocks[i].invocation < blocks[index].invocation)
                    if ready:
                        state[index] = "loading"
                        live.append(activity_factory(owner, index, "load"))
        if all(value is not None for value in finish):
            break
        rates = grouped_rates(live, capacities) if live else []
        future = [a - now for a in arrivals if a > now + 1e-13]
        future += [(1 - a.progress) / rate for a, rate in zip(live, rates) if rate > 0]
        if not future:
            raise ArithmeticError("discrete oracle deadlocked")
        delta = min(future)
        if delta <= 0 or not math.isfinite(delta):
            raise ArithmeticError("nonadvancing oracle event")
        hbm_owners: set[int] = set()
        compute_owners: set[int] = set()
        row_owners = [set(), set()]
        for a, rate in zip(live, rates):
            if rate <= 0:
                continue
            a.progress += rate * delta
            if a.role == "compute":
                compute_owners.add(a.request)
            for row in range(2):
                if a.work[4 + row] > 0:
                    hbm_owners.add(a.request); row_owners[row].add(a.request)
        if hbm_owners:
            busy["hbm_busy_s"] += delta
        if len(hbm_owners) >= 2:
            busy["hbm_shared_s"] += delta
        if compute_owners:
            busy["compute_busy_s"] += delta
        for row in range(2):
            if row_owners[row]:
                busy["row_busy_s"][row] += delta
            if len(row_owners[row]) >= 2:
                busy["row_shared_s"][row] += delta
        now += delta
        for a in live:
            if a.progress >= 1 - 1e-10:
                states[a.request][a.block] = {"load": "buffered", "compute": "produced", "store": "stored"}[a.role]
                if a.role == "compute" and first[a.request] is None:
                    first[a.request] = now
        live = [a for a in live if a.progress < 1 - 1e-10]
        event_count += 1
        if event_count > 100000:
            raise ArithmeticError("offline event guard exceeded")
    return {"finish_s": finish, "first_output_ready_s": first, "counters": busy,
            "discrete_events": event_count,
            "invocation_materialization_barrier": barrier,
            "buffer_semantics": "input_reserved_load_start_through_compute_end; output_reserved_compute_start_through_store_end"}


def error(actual: float, expected: float) -> dict[str, Any]:
    absolute = abs(actual - expected)
    relative = absolute / abs(expected) if expected else (0.0 if absolute < 1e-9 else None)
    return {"actual_s": actual, "reference_s": expected, "absolute_error_s": absolute,
            "relative_error": relative,
            "target_10_percent": relative is not None and relative <= TARGET,
            "within_20_percent": relative is not None and relative <= FALLBACK}


def compare(actual: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    arrivals = [float(n) / NS for n in actual["arrivals_ns"]]
    completion = [error(float(n) / NS - arrival, t - arrival)
                  for n, t, arrival in zip(actual["finish_ns"], ref["finish_s"], arrivals)]
    ready = [error(float(n) / NS - arrival, t - arrival)
             for n, t, arrival in zip(actual["first_output_ready_ns"], ref["first_output_ready_s"], arrivals)]
    counters = actual["counters"]
    busy = {}
    for name in ("hbm_busy_s", "hbm_shared_s", "compute_busy_s"):
        counter = name[:-2] + "_ns"
        busy[name] = error(float(counters.get(counter + "_exact", counters[counter])) / NS,
                           ref["counters"][name])
    all_metrics = completion + list(busy.values())
    fallback = all(m["within_20_percent"] for m in all_metrics)
    target = all(m["target_10_percent"] for m in all_metrics)
    return {"status": "passed_target_10_percent" if target else "within_20_percent_unaccepted_without_recorded_attempts" if fallback else "failed",
            "completion": completion, "first_output_readiness": ready, "busy_original_definition": busy,
            "completion_target_10_percent": all(m["target_10_percent"] for m in completion),
            "completion_within_20_percent": all(m["within_20_percent"] for m in completion),
            "reference_shared_hbm_s": ref["counters"]["hbm_shared_s"],
            "note": "20% is diagnostic coverage, not automatic acceptance; no calibration attempts were made"}


def report(raw: dict[str, Any]) -> dict[str, Any]:
    if raw.get("measurement_source") != "actual_compiled_mapping_and_cpp_runtime":
        raise ValueError("requires actual C++ driver evidence")
    blocks, expected = make_blocks(raw)
    capacities, activity = factory(raw, blocks)
    summary = raw["compiled_plan"]["summary"]
    checks = {key: key in summary and int(summary[key]) == value for key, value in expected.items()
              if key not in ("slots_per_input_output", "blocks_per_invocation")}
    independent_ops = sum(b.eqops for b in blocks)
    activities = [a for e in raw["compiled_plan"]["epochs"] for a in e["activities"]]
    checks["activity_vec_eqops_conserved"] = sum(int(a["vec_eqops"]) for a in activities) == independent_ops
    for direction, label in ((0, "read"), (1, "write")):
        measured = sum(int(a["hbm_bytes"][row][direction]) for a in activities for row in range(2))
        checks[f"activity_hbm_{label}_bytes_conserved"] = measured == expected[f"hbm_{label}_bytes_total"]
    comparisons = {}
    for name, actual in raw["runs"].items():
        arrivals = [float(n) / NS for n in actual["arrivals_ns"]]
        comparisons[name] = {}
        for barrier in (True, False):
            reference = simulate(blocks, arrivals, expected["slots_per_input_output"], capacities, activity, barrier)
            comparisons[name]["invocation_barrier" if barrier else "independent_invocations_pipeline"] = {
                "reference": reference, "comparison": compare(actual, reference)}
    return {"schema_version": 1, "oracle": "independent_discrete_elementwise_single_core_v1",
            "scope": "synthetic model-to-model, not TX8/reference or target hardware calibration",
            "fixture": raw["fixture"], "ledger": {"status": "passed" if all(checks.values()) else "failed",
                "checks": checks, "independently_derived": expected},
            "comparisons": comparisons,
            "fairness": "all 13 nonnegative resource rows; owner Q/alpha; stage PRIVATE; at most one L/C/S per owner",
            "busy_definition": "union of time with positive physical HBM service; shared counts distinct owners, never utilization weighted",
            "routing_scope": "single Core; shared NoC/inject rows nonbinding; integral alternating-byte HBM stripe",
            "limits": {"target_relative_error": TARGET, "fallback_relative_error": FALLBACK,
                       "fallback_accepted": False},
            "production_granularity_validation": "not_run",
            "independent_target_hardware_calibration": "not_available",
            "limitations": ["Does not certify multi-Core routing or per-Core fairness.",
                "NoC/inject per-stage service is not independently reconstructed; those rows are nonbinding in this fixture.",
                "Discrete and macro activities have different Q normalization units.",
                "Both repeat barrier semantics are reported; agreement with either is not proof of external dependency correctness.",
                "Conversion and SFU hardware costs remain outside the existing fixed arithmetic eq-op template."]}


def self_test() -> dict[str, Any]:
    blocks = [Block(i, 1, 1, 1, 1, (1, 0), (0, 1)) for i in range(4)]
    caps = [1.0] * 13
    def activity(owner, block, role):
        work = [0.0] * 13
        work[{"load": 4, "compute": 0, "store": 5}[role]] = 1.0
        return Activity(owner, block, role, work, 0.0, 1.0)
    serial = simulate(blocks, [0.0], 2, caps, activity, True)
    pipeline = simulate(blocks, [0.0], 2, caps, activity, False)
    one = simulate(blocks[:1], [0.0], 1, caps, activity, True)
    sync = simulate(blocks[:1], [0.0, 0.0], 1, caps, activity, True)
    work = [0.0] * 13
    work[4] = 1.0
    weighted = grouped_rates([Activity(0, 0, "load", work, 0, 10),
                              Activity(0, 1, "compute", work, 0, 20),
                              Activity(1, 0, "load", work, 0, 5)], caps)
    checks = {"single_block_L_C_S_equals_3": abs(one["finish_s"][0] - 3) < 1e-12,
              "four_serial_invocations_equal_12": abs(serial["finish_s"][0] - 12) < 1e-12,
              "four_pipelined_invocations_equal_6": abs(pipeline["finish_s"][0] - 6) < 1e-12,
              "first_output_gate_equals_2": abs(pipeline["first_output_ready_s"][0] - 2) < 1e-12,
              "two_sync_owners_equal_6": all(abs(t - 6) < 1e-12 for t in sync["finish_s"]),
              "owner_Q_alpha_floor": all(abs(a - b) < 1e-12 for a, b in zip(weighted, (0.125, 0.125, 0.75)))}
    if not all(checks.values()):
        raise AssertionError(checks)
    return {"status": "passed", "checks": checks,
            "scope": "hand-calculated unit-stage discrete oracle checks, not production validation"}


def suite(driver: Path, directory: Path) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    cases = []
    for op in ("residual", "gelu", "swiglu"):
        for n, repeats in ((257, 1), (256, 4), (513, 2)):
            cases.append((op, n, repeats, "balanced", "fp32" if op == "gelu" else "bf16", "fp32"))
        cases.append((op, 257, 1, "local", "bf16", "fp32"))
        cases.append((op, 1, 3, "balanced", "int4", "int4"))
    cases += [("residual", 257, 1, profile, "bf16", "fp32") for profile in ("memory", "compute")]
    results = []
    for op, n, repeats, profile, ad, od in cases:
        name = f"{op}-e{n}-i{repeats}-{profile}-{ad}-{od}"
        command = [str(driver.resolve()), "--op", op, "--elements", str(n), "--executions", str(repeats),
                   "--profile", profile, "--input-dtype", ad, "--output-dtype", od]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        (directory / f"{name}.stderr.txt").write_text(completed.stderr)
        (directory / f"{name}.raw.json").write_text(completed.stdout)
        entry = {"case": name, "command": command, "returncode": completed.returncode}
        if completed.returncode:
            entry["status"] = "driver_failed"
        else:
            try:
                result = report(json.loads(completed.stdout))
                (directory / f"{name}.report.json").write_text(json.dumps(result, indent=2) + "\n")
                entry["ledger"] = result["ledger"]["status"]
                entry["comparisons"] = {mode: {semantics: item["comparison"]["status"]
                    for semantics, item in comparisons.items()} for mode, comparisons in result["comparisons"].items()}
            except Exception as exc:
                entry.update(status="oracle_failed", error=f"{type(exc).__name__}: {exc}")
        results.append(entry)
    result = {"scope": "17 synthetic single-Core cases; solo/sync/offset; both repeat semantics",
              "oracle_self_test": self_test(), "driver_sha256": hashlib.sha256(driver.read_bytes()).hexdigest(),
              "oracle_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cases": results}
    (directory / "suite.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--raw", type=Path)
    modes.add_argument("--driver", type=Path)
    parser.add_argument("--output", type=Path, help="report JSON, or suite directory with --driver")
    args = parser.parse_args()
    if args.driver:
        if args.output is None:
            parser.error("--driver requires --output directory")
        result = suite(args.driver, args.output)
        print(json.dumps({"suite": str(args.output / "suite.json"), "cases": len(result["cases"])}))
        return
    result = self_test() if args.self_test else report(json.loads(args.raw.read_text()))
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
