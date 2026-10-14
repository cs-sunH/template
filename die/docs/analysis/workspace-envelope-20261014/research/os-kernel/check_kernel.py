#!/usr/bin/env python3
"""Differential checks for the scratch fixed-width C++ Solo13 kernel."""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FACTS = ROOT / "work/os-single-owner-13-20261014/current"
BIN = HERE / "solo13_kernel"

sys.path.insert(0, str(ROOT / "work/os-solo-stop-20261014"))
import solo_stop_13 as solo

spec = importlib.util.spec_from_file_location(
    "solo13_stop_checker", ROOT / "work/os-solo-stop-20261014/check_stops.py")
stop_checker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = stop_checker
assert spec.loader is not None
spec.loader.exec_module(stop_checker)


def fraction(value: object) -> F:
    return value if isinstance(value, F) else F(str(value))


def flatten(value):
    if isinstance(value, list):
        return [item for child in value for item in flatten(child)]
    return [value]


def cpp_run(fixture: str, scenario: str) -> tuple[dict, float]:
    facts_path = FACTS / f"facts_{fixture}.json"
    start = time.perf_counter()
    result = subprocess.run(
        [str(BIN), "--facts", str(facts_path), "--scenario", scenario],
        check=False, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise AssertionError(
            f"C++ kernel exited {result.returncode} for {fixture}/{scenario}: "
            f"{result.stderr.strip()}")
    return json.loads(result.stdout), elapsed


def compare_ledger(cpp: dict, model: dict) -> None:
    actual = cpp["ledger"]
    expected = model["ledger"] if "ledger" in model else model
    pairs = (
        ("elapsed_s", "elapsed_s"),
        ("resource_service_13", "resource_service_13"),
        ("resource_busy_s_13", "resource_busy_s_13"),
        ("hbm_row_payload_bytes", "hbm_payload_row_direction"),
        ("int4_ops", "int4_ops"),
        ("vec_eqops", "vec_eqops"),
        ("compute_busy_s", "compute_busy_s"),
        ("hbm0_busy_s", "hbm0_busy_s"),
        ("hbm1_busy_s", "hbm1_busy_s"),
        ("hbm_busy_s", "hbm_busy_s"),
        ("hbm_shared_s", "hbm_shared_s"),
    )
    for expected_key, actual_key in pairs:
        got = [fraction(x) for x in flatten(actual[actual_key])]
        want = [fraction(x) for x in flatten(expected[expected_key])]
        assert got == want, (expected_key, got, want)


def compare_live(cpp: dict, model: dict) -> None:
    got, want = cpp["live_state"], model["live_state"]
    assert int(got["next_load_wave"]) == int(want["next_load_wave"])
    assert int(got["retired_outputs"]) == int(want["retired_outputs"])
    assert [int(x) for x in got["loaded_waves"]] == [
        int(x) for x in want["loaded_waves"]]
    assert [int(x) for x in got["computed_final_waves"]] == [
        int(x) for x in want["computed_final_waves"]]
    actual_active = {
        (entry["stage"], int(entry["wave"])): fraction(entry["progress"])
        for entry in got["active"]}
    expected_active = {
        (entry["stage"], int(entry["wave"])): fraction(entry["progress"])
        for entry in want["active"]}
    assert actual_active == expected_active, (actual_active, expected_active)
    assert bool(got["finish"]) == (want["finish_s"] is not None)


def compare_fraction_model(cpp_state: dict, report: dict, model_report: dict) -> None:
    assert cpp_state["status"] == model_report["status"]
    assert cpp_state["expanded_events"] == model_report["expanded_events"]
    assert fraction(cpp_state["skipped_waves"]) == fraction(model_report["skipped_waves"])
    assert fraction(cpp_state["skipped_outputs"]) == fraction(model_report["skipped_outputs"])
    compare_ledger(cpp_state, model_report)
    compare_live(cpp_state, model_report)
    assert cpp_state["peak_active_stages"] <= 3
    assert cpp_state["peak_input_slots"] <= 2
    assert cpp_state["peak_output_slots"] <= 2
    assert cpp_state["bounded_kernel"]["global_expanded_event_cap"] == 256
    assert cpp_state["bounded_kernel"]["tile_or_core_arrays"] is False
    assert cpp_state["bounded_kernel"]["public_integer_ns_runtime_clock_used"] is False


def compare_independent_prefix(cpp_state: dict, data: dict, oracle=None) -> None:
    end = fraction(cpp_state["ledger"]["elapsed_s"])
    if oracle is None:
        oracle = stop_checker.PrefixOracle(data).until(end)
    else:
        assert oracle.now == end
    ledger = cpp_state["ledger"]
    for key, expected in (
        ("resource_service_13", oracle.service),
        ("resource_busy_s_13", oracle.busy),
        ("hbm_payload_row_direction", oracle.hbm),
    ):
        assert [fraction(x) for x in flatten(ledger[key])] == [
            fraction(x) for x in flatten(expected)], key
    for key, expected in (
        ("elapsed_s", oracle.now), ("int4_ops", oracle.int4),
        ("vec_eqops", oracle.vec), ("compute_busy_s", oracle.compute_busy),
        ("hbm0_busy_s", oracle.row_busy[0]),
        ("hbm1_busy_s", oracle.row_busy[1]), ("hbm_busy_s", oracle.hbm_busy),
    ):
        assert fraction(ledger[key]) == expected, key

    owner = cpp_state["live_state"]
    actual_active = {
        entry["stage"]: (int(entry["wave"]), fraction(entry["progress"]))
        for entry in owner["active"]}
    independent_active = {
        stage: (item["ordinal"], 1 - item["remaining"])
        for stage, item in oracle.active.items()}
    assert actual_active == independent_active
    assert int(owner["next_load_wave"]) == oracle.next_load
    assert int(owner["retired_outputs"]) == oracle.retired
    assert [int(x) for x in owner["loaded_waves"]] == sorted(
        ordinal for ordinal, state in oracle.inputs.items() if state == "ready")
    assert [int(x) for x in owner["computed_final_waves"]] == sorted(
        (q + 1) * oracle.kt - 1
        for q, state in oracle.outputs_live.items() if state == "ready")


def load_facts(fixture: str) -> dict:
    return json.loads((FACTS / f"facts_{fixture}.json").read_text())


def test_full_fixtures(run_meta: dict) -> None:
    small = ("strict_small", "int4_native_small", "int4_converted_fractional",
             "int4_converted_storage", "dense", "output_pressure_requested",
             "output_pressure_k2_tiles128", "output_pressure_k1_tiles128")
    for fixture in small:
        data = load_facts(fixture)
        output, elapsed = cpp_run(fixture, "full")
        state = output["result"]
        reference = solo.Solo13(solo.FactsGeometry(data)).run()
        compare_fraction_model(state, data, reference)
        compare_independent_prefix(state, data)
        run_meta["runs"].append({"fixture": fixture, "scenario": "full",
                                  "status": state["status"],
                                  "events": state["expanded_events"],
                                  "skipped_waves": state["skipped_waves"],
                                  "wall_seconds": elapsed,
                                  "peak_output_slots": state["peak_output_slots"],
                                  "output_slots_reached_two":
                                      state["peak_output_slots"] == 2,
                                  "final_compute_waits_for_output_slot":
                                      state["final_compute_waited_for_full_output_slot"],
                                  "nonfinal_compute_at_full_output":
                                      state["nonfinal_compute_started_while_output_full"],
                                  "independent_fraction_prefix": "passed"})

    for fixture in ("large_k", "large_outputs"):
        data = load_facts(fixture)
        output, elapsed = cpp_run(fixture, "full")
        state = output["result"]
        reference = solo.Solo13(solo.FactsGeometry(data)).run()
        compare_fraction_model(state, data, reference)
        assert state["expanded_events"] <= 256
        assert int(state["skipped_waves"]) > 0
        run_meta["runs"].append({"fixture": fixture, "scenario": "full",
                                  "status": state["status"],
                                  "events": state["expanded_events"],
                                  "skipped_waves": state["skipped_waves"],
                                  "skipped_outputs": state["skipped_outputs"],
                                  "wall_seconds": elapsed,
                                  "large_shape_oracle": "same exact Fraction model; no per-wave expansion"})


def test_integer_stop(run_meta: dict) -> None:
    fixture = "strict_small"
    data = load_facts(fixture)
    output, elapsed = cpp_run(fixture, "integer-stop")
    model = solo.Solo13(solo.FactsGeometry(data), stop_at=F(17, 10**9))
    pause = model.run()
    assert pause["status"] == "paused"
    compare_fraction_model(output["pause"], data, pause)
    compare_independent_prefix(output["pause"], data)
    assert model.resume_until()["status"] == "passed"
    compare_fraction_model(output["result"], data, model.report("passed"))
    run_meta["runs"].append({"fixture": fixture, "scenario": "integer-stop-resume",
                              "status": output["result"]["status"],
                              "pause_events": output["pause"]["expanded_events"],
                              "final_events": output["result"]["expanded_events"],
                              "wall_seconds": elapsed,
                              "independent_fraction_prefix": "passed"})


def test_pause_resume(fixture: str, run_meta: dict, *, independent: bool = True) -> None:
    data = load_facts(fixture)
    full = solo.Solo13(solo.FactsGeometry(data))
    assert full.run()["status"] == "passed"
    finish = full.now
    output, elapsed = cpp_run(fixture, "pause-resume")
    model = solo.Solo13(solo.FactsGeometry(data), stop_at=finish / 3)
    p1 = model.run()
    p2 = model.resume_until(finish * F(2, 3))
    final = model.resume_until()
    assert p1["status"] == p2["status"] == "paused"
    assert final["status"] == "passed"
    compare_fraction_model(output["pause1"], data, p1)
    compare_fraction_model(output["pause2"], data, p2)
    compare_fraction_model(output["result"], data, final)
    if independent:
        compare_independent_prefix(output["pause1"], data)
        compare_independent_prefix(output["pause2"], data)
    assert output["events_before_resume1"] == p1["expanded_events"]
    assert output["events_before_resume2"] == p2["expanded_events"]
    assert output["events_before_resume1"] <= output["events_before_resume2"] <= 256
    assert output["result"]["expanded_events"] <= 256
    run_meta["runs"].append({"fixture": fixture, "scenario": "pause-resume",
                              "status": final["status"],
                              "events_at_stops": [p1["expanded_events"], p2["expanded_events"]],
                              "total_events": final["expanded_events"],
                              "wall_seconds": elapsed,
                              "global_event_budget_preserved": True,
                              "independent_prefix": "passed" if independent else
                                  "not expanded; exact Fraction resume comparison only"})


def test_capacity_invalidation(run_meta: dict) -> None:
    fixture = "strict_small"
    data = load_facts(fixture)
    full = solo.Solo13(solo.FactsGeometry(data))
    assert full.run()["status"] == "passed"
    finish = full.now
    first_stop, second_stop = finish / 3, finish / 2
    output, elapsed = cpp_run(fixture, "capacity-stop")
    model = solo.Solo13(solo.FactsGeometry(data), stop_at=first_stop)
    p1 = model.run()
    new_caps = tuple(value * F(7, 8) if i in (4, 5) else value
                     for i, value in enumerate(model.g.capacities))
    before = model.events
    p2 = model.resume_until(second_stop, capacities=new_caps)
    assert p1["status"] == p2["status"] == "paused"
    compare_fraction_model(output["pause1"], data, p1)
    compare_fraction_model(output["pause2"], data, p2)
    oracle = stop_checker.PrefixOracle(data).until(first_stop)
    compare_independent_prefix(output["pause1"], data, oracle)
    oracle.capacities = new_caps
    oracle.until(second_stop)
    compare_independent_prefix(output["pause2"], data, oracle)
    backwards_clock = model.now
    reversed_report = model.resume_until(first_stop)
    assert reversed_report["status"] == "failed_closed" and model.now == backwards_clock
    assert output["backwards"]["status"] == "failed_closed"
    assert output["clock_preserved_after_backwards"] is True
    assert output["events_before_capacity_change"] <= output["events_after_capacity_change"] <= 256
    assert output["backwards"]["expanded_events"] == output["events_after_capacity_change"]
    assert output["capacity_invalidations_after_resume"] == 1
    final = model.resume_until()
    assert final["status"] == "passed"
    compare_fraction_model(output["result"], data, final)
    oracle.until(fraction(output["pause2"]["ledger"]["elapsed_s"]))
    oracle.until(fraction(output["result"]["ledger"]["elapsed_s"]))
    compare_independent_prefix(output["result"], data, oracle)
    run_meta["runs"].append({"fixture": fixture, "scenario": "capacity-change-stop-resume",
                              "status": final["status"],
                              "events_before_change": output["events_before_capacity_change"],
                              "events_after_change": output["events_after_capacity_change"],
                              "capacity_invalidations": output["capacity_invalidations_after_resume"],
                              "global_event_budget_preserved":
                                  output["events_after_capacity_change"] <= 256,
                              "reverse_stop_preserved_clock": output["clock_preserved_after_backwards"],
                              "wall_seconds": elapsed})


def test_cap_exhaustion(run_meta: dict) -> None:
    output, elapsed = cpp_run("dense", "unpowered-cap")
    state = output["result"]
    assert state["status"] == "failed_closed"
    assert state["expanded_events"] == 256
    assert state["reason"] == "global expanded-event cap reached"
    assert state["identity_skips_enabled"] is False
    assert state["history_entries"] == 0
    run_meta["runs"].append({"fixture": "dense", "scenario": "unpowered-cap-exhaustion",
                              "status": state["status"], "events": state["expanded_events"],
                              "wall_seconds": elapsed,
                              "result": "fail-closed exactly at global 256-event cap"})


def main() -> int:
    if not BIN.exists():
        raise SystemExit("compile solo13_kernel first; see build_and_check.sh")
    run_meta = {
        "schema": "solo13_cpp_kernel_check_v1",
        "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                check=True, capture_output=True, text=True).stdout.strip(),
        "kernel_source_sha256": hashlib.sha256((HERE / "solo13_kernel.cc").read_bytes()).hexdigest(),
        "kernel_header_sha256": hashlib.sha256((HERE / "solo13_kernel.hh").read_bytes()).hexdigest(),
        "fixed_rational_source_sha256": hashlib.sha256(
            (ROOT / "work/os-fixedwidth-13-20261014/checked_rational256.hh").read_bytes()).hexdigest(),
        "geometry_source_sha256": hashlib.sha256(
            (ROOT / "die/src/GemmOSGeometry.hh").read_bytes()).hexdigest(),
        "access_counter_source_sha256": hashlib.sha256(
            (ROOT / "die/src/GemmOSAccessCounter.hh").read_bytes()).hexdigest(),
        "resource_facts_declaration_sha256": hashlib.sha256(
            (ROOT / "die/src/GemmOSResourceFacts.hh").read_bytes()).hexdigest(),
        "runtime_facts_driver_source_sha256": hashlib.sha256(
            (ROOT / "die/tests/os_lifecycle_runtime_driver.cc").read_bytes()).hexdigest(),
        "runtime_facts_driver_binary_sha256": hashlib.sha256(
            (FACTS / "os_lifecycle_runtime_driver").read_bytes()).hexdigest(),
        "runtime_facts_producer_identity_sha256": hashlib.sha256(
            (FACTS / "producer_identity.json").read_bytes()).hexdigest(),
        "compile_command": [
            "g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-O2",
            "-I astra-sim-wscllm-LRU/extern/helper", "-I die/include",
            "work/os-solo-cpp-13-20261014/solo13_kernel.cc",
            "-o work/os-solo-cpp-13-20261014/solo13_kernel"],
        "kernel_binary_sha256": hashlib.sha256(BIN.read_bytes()).hexdigest(),
        "compiler": subprocess.run(["g++", "--version"], cwd=ROOT,
                                    check=True, capture_output=True,
                                    text=True).stdout.splitlines()[0],
        "facts_sha256": {},
        "runs": [],
        "scope": "scratch-only single-owner/core continuous exact-time kernel; no Runtime/C ABI/production certificate",
    }
    fixtures = ("strict_small", "int4_native_small", "int4_converted_fractional",
                "int4_converted_storage", "dense", "output_pressure_requested",
                "output_pressure_k2_tiles128", "output_pressure_k1_tiles128",
                "large_k", "large_outputs")
    for name in fixtures:
        path = FACTS / f"facts_{name}.json"
        run_meta["facts_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    test_full_fixtures(run_meta)
    test_integer_stop(run_meta)
    test_pause_resume("strict_small", run_meta)
    test_pause_resume("int4_converted_fractional", run_meta)
    test_capacity_invalidation(run_meta)
    test_pause_resume("large_k", run_meta, independent=False)
    test_pause_resume("large_outputs", run_meta, independent=False)
    test_cap_exhaustion(run_meta)
    run_meta["status"] = "passed"
    target = HERE / "kernel_validation.json"
    target.write_text(json.dumps(run_meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(run_meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, ValueError, ZeroDivisionError) as error:
        print(f"FAILED: {error}", file=sys.stderr)
        raise
