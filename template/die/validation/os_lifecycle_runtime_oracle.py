#!/usr/bin/env python3
"""Offline exact readiness/allocation check of the single-Core Runtime driver.

The C++ compiler supplies immutable per-class work facts. This checker independently
decodes the small geometry, owns discrete two-input/two-output queues, and uses
Fraction progressive filling over all 13 physical rows and each PRIVATE bound.
It reproduces the public integer-ns wake boundary: existing activities may finish
and reallocate inside advance_to(), while new stages start only at that wake.
This is a driver/model comparison, not a production granularity or skip certificate.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess

NS = 10**9
STAGES = ("load", "compute", "store")


def ceil_ns(seconds):
    value = seconds * NS
    return (value.numerator + value.denominator - 1) // value.denominator


def dyadic(value):
    if value is None:
        return F(0)
    significand, exponent = int(value["significand"]), int(value["exponent"])
    return F(significand * 2**exponent) if exponent >= 0 else F(significand, 2**-exponent)


def rates(active, capacities):
    # For one owner alpha_i=Q_i/sum(Q), hence y_i/alpha_i=sum(Q)*v_i.
    # Exact max-min allocation therefore raises progress rates equally within
    # each tier; the owner Q multiplier cancels without changing physical work.
    result = {stage: F(0) for stage in active}
    free = set(active)
    while free:
        limits = []
        for row in range(13):
            slope = sum(active[s]["work"][row] for s in free)
            if slope:
                used = sum(active[s]["work"][row] * result[s] for s in active)
                limits.append(((capacities[row] - used) / slope, "row", row))
        for stage in free:
            private = active[stage]["private"]
            if private:
                limits.append((1 / private - result[stage], "private", stage))
        if not limits:
            raise AssertionError("unbounded activity")
        step = min(item[0] for item in limits)
        assert step >= 0, "infeasible allocation"
        for stage in free:
            result[stage] += step
        frozen = set()
        for bound, kind, key in limits:
            if bound == step:
                if kind == "row":
                    frozen.update(s for s in free if active[s]["work"][key] > 0)
                else:
                    frozen.add(key)
        assert frozen, "fill tier made no progress"
        free -= frozen
    for row in range(13):
        assert sum(active[s]["work"][row] * result[s] for s in active) <= capacities[row]
    return result


class Oracle:
    def __init__(self, data):
        self.g = data["geometry"]
        assert self.g["core_count"] == 1
        self.classes = self.g["classes"]
        self.capacities = tuple(map(F, data["synthetic_hardware"]["hardware_capacities_13"]))
        self.kt = self.g["k_tiles"]
        self.nt = self.g["n_tiles"]
        self.outputs = int(self.g["output_tile_count"])
        self.waves = self.outputs * self.kt
        if self.waves > 8192:
            raise ValueError("offline oracle accepts only small fixtures (<=8192 consumptions)")
        self.now = F(0)
        self.active = {}
        self.inputs, self.outputs_live = {}, {}
        self.next_load = self.next_compute = self.retired = 0
        self.starts, self.finishes = {}, {}
        self.hbm = [[F(0), F(0)], [F(0), F(0)]]
        self.int4 = self.vec = self.hbm_busy = self.compute_busy = F(0)
        self.row_busy = [F(0), F(0)]
        self.events = 0

    def fact(self, ordinal, stage):
        q, k = divmod(ordinal, self.kt)
        mi, ni = divmod(q, self.nt)
        matches = [c for c in self.classes
                   if c["m_tile_begin"] <= mi < c["m_tile_begin"] + c["m_tile_count"]
                   and c["n_tile_begin"] <= ni < c["n_tile_begin"] + c["n_tile_count"]
                   and c["k_tile_begin"] <= k < c["k_tile_begin"] + c["k_tile_count"]]
        assert len(matches) == 1
        fact = matches[0][stage + "_fact"]
        typed = fact["typed_ops"]
        return {"ordinal": ordinal, "remaining": F(1),
                "work": tuple(map(F, fact["resource_work_13"])),
                "private": F(fact["private_time_s"]),
                "hbm": fact["hbm_row_payload_bytes"],
                "int4": F(typed["int4_ops"]),
                "vec": F(typed["vec_eqops_whole"]) + dyadic(typed["vec_eqops_fraction"])}

    def start(self, stage, ordinal):
        assert stage not in self.active
        self.active[stage] = self.fact(ordinal, stage)
        self.starts[(stage, ordinal)] = ceil_ns(self.now)

    def settle(self):
        while True:
            ready_outputs = [q for q, state in self.outputs_live.items() if state == "ready"]
            if "store" not in self.active and ready_outputs:
                q = min(ready_outputs)
                self.outputs_live[q] = "storing"
                self.start("store", (q + 1) * self.kt - 1)
                continue
            ordinal = self.next_compute
            if "compute" not in self.active and self.inputs.get(ordinal) == "ready":
                final = ordinal % self.kt == self.kt - 1
                if not final or len(self.outputs_live) < 2:
                    self.inputs[ordinal] = "computing"
                    if final:
                        self.outputs_live[ordinal // self.kt] = "reserved"
                    self.start("compute", ordinal)
                    continue
            if "load" not in self.active and self.next_load < self.waves and len(self.inputs) < 2:
                ordinal = self.next_load
                self.next_load += 1
                self.inputs[ordinal] = "loading"
                self.start("load", ordinal)
                continue
            assert len(self.inputs) <= 2 and len(self.outputs_live) <= 2
            assert len(self.active) <= 3
            return

    def complete(self, stage):
        item = self.active.pop(stage)
        ordinal = item["ordinal"]
        self.finishes[(stage, ordinal)] = ceil_ns(self.now)
        if stage == "load":
            self.inputs[ordinal] = "ready"
        elif stage == "compute":
            del self.inputs[ordinal]
            self.next_compute += 1
            if ordinal % self.kt == self.kt - 1:
                self.outputs_live[ordinal // self.kt] = "ready"
        else:
            del self.outputs_live[ordinal // self.kt]
            self.retired += 1

    def advance(self, goal):
        while self.active and self.now < goal:
            allocation = rates(self.active, self.capacities)
            deadlines = {s: item["remaining"] / allocation[s]
                         for s, item in self.active.items() if allocation[s] > 0}
            assert deadlines, "readiness oracle deadlocked"
            delta = min(min(deadlines.values()), goal - self.now)
            busy = [False] * 13
            for stage, item in self.active.items():
                portion = delta * allocation[stage]
                item["remaining"] -= portion
                self.int4 += portion * item["int4"]
                self.vec += portion * item["vec"]
                for row in range(2):
                    for direction in range(2):
                        self.hbm[row][direction] += portion * item["hbm"][row][direction]
                for row in range(13):
                    busy[row] |= allocation[stage] > 0 and item["work"][row] > 0
            if busy[0]:
                self.compute_busy += delta
            if busy[4] or busy[5]:
                self.hbm_busy += delta
            for row in range(2):
                if busy[row + 4]:
                    self.row_busy[row] += delta
            self.now += delta
            for stage in STAGES:
                if stage in self.active and self.active[stage]["remaining"] == 0:
                    self.complete(stage)
                    self.events += 1
        self.now = goal

    def run(self):
        while self.retired < self.outputs:
            self.settle()
            assert self.active, "no eligible activity"
            allocation = rates(self.active, self.capacities)
            delta = min(item["remaining"] / allocation[s]
                        for s, item in self.active.items() if allocation[s] > 0)
            self.advance(F(ceil_ns(self.now + delta), NS))
            if self.events > 3 * self.waves:
                raise AssertionError("offline completion budget exceeded")
        assert not self.active and not self.inputs and not self.outputs_live
        return self


def check(data):
    assert data["live_state"]["status"] == "complete"
    expected = Oracle(data).run()
    actual_starts, actual_finishes = {}, {}
    for event in data["events"]:
        key = event["stage"], int(event["ordinal"])
        if event["phase"] == "start":
            actual_starts[key] = event["time_ns"]
        elif event["phase"] == "terminal":
            actual_finishes[key] = event["extra"]["finish_ns"]
    assert actual_starts.keys() == expected.starts.keys()
    assert actual_finishes.keys() == expected.finishes.keys()
    start_error = max(abs(actual_starts[k] - expected.starts[k]) for k in actual_starts)
    finish_error = max(abs(actual_finishes[k] - expected.finishes[k]) for k in actual_finishes)
    # Binary64 Runtime allocation can move a ceil boundary by one tick. Keep
    # this numerical bound separate from the original production 3%/5% gates.
    assert start_error <= 2 and finish_error <= 2, (start_error, finish_error)
    observed = data["runtime_counters"]
    # Every terminal activity has exact completed work. Lifetime counters are
    # long-double integrals and can retain a few ulps after many prefixes; keep
    # that measured discrepancy visible instead of rounding/clamping the data.
    terminal_hbm = [[F(0), F(0)], [F(0), F(0)]]
    terminal_int4 = terminal_vec = F(0)
    for event in data["events"]:
        if event["phase"] != "terminal":
            continue
        served = event["extra"]["served"]
        for row in range(2):
            for direction in range(2):
                terminal_hbm[row][direction] += F(served["hbm_bytes_by_row_direction_exact"][row][direction])
        terminal_int4 += F(served["int4_ops_exact_decimal"])
        terminal_vec += F(served["vec_eqops_exact_decimal"])
    assert terminal_hbm == expected.hbm
    assert terminal_int4 == expected.int4 and terminal_vec == expected.vec
    absolute_work_bound = max(F(1), expected.int4, expected.vec,
                              sum(v for row in expected.hbm for v in row))
    integral_rounding_bound = 4 * len(actual_finishes) * absolute_work_bound / 2**63
    hbm_error = max(abs(F(observed["hbm_served_bytes_by_row_direction_exact"][r][direction]) - expected.hbm[r][direction])
                    for r in range(2) for direction in range(2))
    assert hbm_error <= integral_rounding_bound, (hbm_error, integral_rounding_bound)
    comp = observed["served_by_category"]["comp"]
    assert abs(F(comp["int4_ops_exact_decimal"]) - expected.int4) <= integral_rounding_bound
    assert abs(F(comp["vec_eqops_exact_decimal"]) - expected.vec) <= integral_rounding_bound
    busy_error_ns = abs(F(observed["hbm_busy_ns_exact"]) - expected.hbm_busy * NS)
    assert busy_error_ns <= 2 * len(actual_finishes), busy_error_ns
    assert observed["hbm_shared_ns"] == 0, "one fair owner is not two sharing owners"
    assert observed["active_activities"] == 0 and data["live_state"]["runtime_drained"]
    return {"status": "passed", "stages": len(actual_finishes),
            "max_start_error_ns": start_error, "max_finish_error_ns": finish_error,
            "hbm_busy_error_ns": float(busy_error_ns),
            "lifetime_hbm_rounding_error_bytes": str(hbm_error),
            "terminal_work_conservation": "exact",
            "exact_finish_ns": ceil_ns(expected.now),
            "scope": "offline single-owner readiness and exact 13-row allocation; C++ supplied work facts; no production granularity/skip certificate"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = {"strict": [], "int4_native": ["--path", "int4", "--a-dtype", "int4", "--w-dtype", "int4", "--output-dtype", "int4"],
             "int4_converted": ["--path", "int4", "--a-dtype", "bf16", "--w-dtype", "bf16", "--output-dtype", "bf16"],
             "fractional_conversion": ["--path", "int4", "--a-dtype", "bf16", "--kappa-quant", "0.3125", "--kappa-acc", "0.375"],
             "single_k": ["--m", "3", "--n", "5", "--k", "1"],
             "short_tiles": ["--m", "1", "--n", "129", "--k", "65"]}
    reports = {}
    for name, options in cases.items():
        result = subprocess.run([str(args.driver.resolve()), *options], check=True, capture_output=True, text=True)
        reports[name] = check(json.loads(result.stdout))
    payload = json.dumps(reports, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
