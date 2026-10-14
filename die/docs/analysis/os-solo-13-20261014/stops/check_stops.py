#!/usr/bin/env python3
"""Exact external-boundary study; no production Runtime timing claim."""
from fractions import Fraction as F
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import solo_stop_13 as solo

spec = importlib.util.spec_from_file_location(
    "independent_readiness", ROOT / "die/validation/os_lifecycle_runtime_oracle.py")
oracle_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle_module)


class PrefixOracle(oracle_module.Oracle):
    def __init__(self, data):
        super().__init__(data)
        self.service = [F(0)] * 13
        self.busy = [F(0)] * 13

    def until(self, stop):
        assert stop >= self.now
        while True:
            self.settle()
            if self.now == stop or self.retired == self.outputs:
                return self
            assert self.active
            allocation = oracle_module.rates(self.active, self.capacities)
            delta = min(stop - self.now, min(
                item["remaining"] / allocation[s]
                for s, item in self.active.items() if allocation[s] > 0))
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
                    self.service[row] += portion * item["work"][row]
                    busy[row] |= allocation[stage] > 0 and item["work"][row] > 0
            for row in range(13):
                if busy[row]:
                    self.busy[row] += delta
            if busy[0]:
                self.compute_busy += delta
            if busy[4] or busy[5]:
                self.hbm_busy += delta
            for row in range(2):
                if busy[row + 4]:
                    self.row_busy[row] += delta
            self.now += delta
            for stage in oracle_module.STAGES:
                if stage in self.active and self.active[stage]["remaining"] == 0:
                    self.complete(stage)
                    self.events += 1


def check(candidate, reference):
    assert candidate.now == reference.now
    owner = candidate.o
    assert owner.next_load == reference.next_load
    assert owner.retired == reference.retired
    active = {oracle_module.STAGES[k]: (wave, 1 - progress)
              for k, (wave, progress) in owner.active.items()}
    assert active == {s: (x["ordinal"], x["remaining"])
                      for s, x in reference.active.items()}
    assert owner.loaded == sorted(q for q, state in reference.inputs.items()
                                  if state == "ready")
    assert owner.computed == sorted((q + 1) * reference.kt - 1
        for q, state in reference.outputs_live.items() if state == "ready")
    ledger = candidate.ledger
    assert ledger["elapsed_s"] == reference.now
    assert [ledger[f"resource_service_{r}"] for r in range(13)] == reference.service
    assert [ledger[f"resource_busy_s_{r}"] for r in range(13)] == reference.busy
    assert [[ledger[f"hbm_payload_{r}_{d}"] for d in range(2)]
            for r in range(2)] == reference.hbm
    assert ledger["int4_ops"] == reference.int4
    assert ledger["vec_eqops"] == reference.vec
    assert ledger["compute_busy_s"] == reference.compute_busy
    assert ledger["hbm_busy_s"] == reference.hbm_busy
    assert [ledger[f"hbm{r}_busy_s"] for r in range(2)] == reference.row_busy


def exercise(name):
    data = json.loads((ROOT / "work/os-single-owner-13-20261014/current" /
                       f"facts_{name}.json").read_text())
    finished = solo.Solo13(solo.FactsGeometry(data))
    assert finished.run()["status"] == "passed"
    finish = finished.now
    # Integer-ns stops plus exact simultaneous/individual completion boundaries.
    dense_reference = PrefixOracle(data).until(finish)
    boundaries = [F(ns, 10**9) for ns in (0, 1, 17, 101, 1000, 10_000)
                  if F(ns, 10**9) < finish]
    boundaries += [finish / 3, finish / 2, finish - F(1, 10**9), finish]
    event_ns = sorted(set(dense_reference.finishes.values()))
    for ns in event_ns[::max(1, len(event_ns) // 8)]:
        boundaries += [F(ns + offset, 10**9) for offset in (-1, 0, 1)
                       if 0 <= F(ns + offset, 10**9) <= finish]
    boundaries = sorted(set(boundaries))
    max_events = 0
    for boundary in boundaries:
        model = solo.Solo13(solo.FactsGeometry(data), stop_at=boundary)
        report = model.run()
        assert report["status"] == ("passed" if boundary == finish else "paused"), report
        check(model, PrefixOracle(data).until(boundary))
        events_before = model.events
        assert model.resume_until()["status"] == "passed"
        assert model.events >= events_before and model.events <= 256
        assert model.ledger == finished.ledger
        max_events = max(max_events, model.events)
    # Sequential stops and a real capacity change invalidate old identity maps.
    boundary = finish / 3
    model = solo.Solo13(solo.FactsGeometry(data), stop_at=boundary)
    assert model.run()["status"] == "paused"
    reference = PrefixOracle(data).until(boundary)
    new_capacity = tuple(value * (F(7, 8) if row in (4, 5) else 1)
                         for row, value in enumerate(model.g.capacities))
    reference.capacities = new_capacity
    events_before = model.events
    assert model.resume_until(finish / 2, capacities=new_capacity)["status"] == "paused"
    check(model, reference.until(finish / 2))
    assert model.events >= events_before and model.events <= 256
    backwards = model.resume_until(boundary)
    assert backwards["status"] == "failed_closed"
    assert model.now == finish / 2
    assert model.resume_until()["status"] == "passed"
    check(model, reference.until(model.now))
    return {"fixture": name, "boundaries": len(boundaries),
            "max_events_pause_resume": max_events,
            "capacity_change_events": model.events, "status": "passed"}


def huge(name):
    data = json.loads((ROOT / "work/os-single-owner-13-20261014/current" /
                       f"facts_{name}.json").read_text())
    reference = solo.Solo13(solo.FactsGeometry(data))
    assert reference.run()["status"] == "passed"
    finish = reference.now
    candidate = solo.Solo13(solo.FactsGeometry(data), stop_at=finish / 3)
    assert candidate.run()["status"] == "paused"
    events_before = candidate.events
    # Huge shapes are compared with exact no-stop powering, never expanded.
    assert candidate.resume_until(finish * F(2, 3))["status"] == "paused"
    assert candidate.events >= events_before
    assert candidate.resume_until()["status"] == "passed"
    assert candidate.ledger == reference.ledger and candidate.events <= 256
    return {"fixture": name, "status": "passed", "total_events": candidate.events,
            "skipped_waves": str(candidate.skipped_waves),
            "comparison": "same exact study without stops; no independent huge expansion"}


def main():
    result = {"status": "passed", "scope": "exact Fraction offline solo; production not changed",
              "small": [exercise(name) for name in (
                  "strict_small", "int4_native_small", "int4_converted_fractional",
                  "int4_converted_storage", "dense", "output_pressure_k1_tiles128")],
              "huge": [huge(name) for name in ("large_k", "large_outputs")]}
    result["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (HERE / "solo_stop_13.py", HERE / "check_stops.py")}
    (HERE / "stop_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
