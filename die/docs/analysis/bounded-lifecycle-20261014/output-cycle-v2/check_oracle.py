#!/usr/bin/env python3
"""Compare scratch C++ outputs with the independent per-wave GEMM oracle.

The equal-progress policy is compared against compiled_gemm_oracle.py. The
Q/alpha policy has a distinct allocator, so its event-by-event and accelerated
paths are compared with each other by the C++ program, not mislabeled as an
independent allocator cross-check here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shlex
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents
            if (p / "die" / "validation" / "compiled_gemm_oracle.py").is_file())
ORACLE_PATH = REPO / "die" / "validation" / "compiled_gemm_oracle.py"
DEFAULT_RAW = HERE / "raw-fixture.json"
TOL = 2.0e-10


def load_oracle():
    spec = importlib.util.spec_from_file_location("compiled_gemm_oracle", ORACLE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import independent oracle: {ORACLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_records(path: Path) -> tuple[dict[tuple[str, str, str], dict[str, str]],
                                        dict[tuple[str, str], dict[str, str]]]:
    records: dict[tuple[str, str, str], dict[str, str]] = {}
    checks: dict[tuple[str, str], dict[str, str]] = {}
    for raw_line in path.read_text().splitlines():
        if not raw_line.startswith(("record ", "check ")):
            continue
        fields = dict(field.split("=", 1) for field in shlex.split(raw_line)[1:])
        if raw_line.startswith("record "):
            records[(fields["policy"], fields["scenario"], fields["kind"])] = fields
        else:
            checks[(fields["policy"], fields["scenario"])] = fields
    return records, checks


def as_float(value: str) -> float:
    return float(Fraction(value))


def require_close(label: str, actual: str, expected: float) -> None:
    got = as_float(actual)
    if not math.isclose(got, expected, rel_tol=TOL, abs_tol=TOL):
        raise AssertionError(f"{label}: C++={got:.17g}, independent oracle={expected:.17g}")


def compare_ledger(record: dict[str, str], expected: dict[str, Any], owners: int) -> None:
    for i in range(owners):
        for field, key in (("read", "hbm_read_bytes"),
                           ("write", "hbm_write_bytes"),
                           ("ops", "vec_equivalent_ops"),
                           ("local", "local_bus_bytes")):
            actual = int(record[f"owner{i}_{field}"])
            if actual != int(expected[key]):
                raise AssertionError(f"owner{i} {field}: C++={actual}, wave oracle={expected[key]}")
        expected_counts = {
            "load_done": expected["consumption_wave_count"],
            "compute_done": expected["consumption_wave_count"],
            "store_done": expected["true_output_tiles"],
            "retired": expected["true_output_tiles"],
        }
        for field, value in expected_counts.items():
            actual = int(record[f"owner{i}_{field}"])
            if actual != int(value):
                raise AssertionError(f"owner{i} {field}: C++={actual}, expected={value}")


def compare_scenario(oracle_module: Any, raw: dict[str, Any], expected: dict[str, Any],
                     records: dict[tuple[str, str, str], dict[str, str]],
                     checks: dict[tuple[str, str], dict[str, str]],
                     scenario: str, arrivals: list[float]) -> dict[str, Any]:
    waves, rebuilt = oracle_module.make_waves(raw)
    if rebuilt != expected:
        raise AssertionError("fixture summary changed between oracle calls")
    capacities = oracle_module.make_capacities(raw)
    discrete = oracle_module.discrete_oracle(
        waves, int(expected["true_output_tiles"]), arrivals, capacities)
    owners = len(arrivals)
    cycle_status: dict[str, str] = {}
    policies = ("equal_progress", "grouped_q_alpha_q1", "grouped_q_alpha_qlocal")
    for policy in policies:
        event = records.get((policy, scenario, "event"))
        cycle = records.get((policy, scenario, "cycle"))
        if event is None or event.get("status") != "completed":
            raise AssertionError(f"missing/incomplete event path: {policy}/{scenario}")
        compare_ledger(event, expected, owners)
        if cycle is None:
            raise AssertionError(f"missing cycle record: {policy}/{scenario}")
        if cycle.get("status") == "completed":
            check = checks.get((policy, scenario))
            if check is None or check.get("cycle_equals_event") != "true":
                raise AssertionError(f"completed cycle differs from event path: {policy}/{scenario}")
            compare_ledger(cycle, expected, owners)
            cycle_status[policy] = "completed_and_equal"
        elif cycle.get("status") == "fail_closed" and cycle.get("fail_reason") == \
                "no_exact_joint_repeat_within_probe_limit":
            cycle_status[policy] = "expected_fail_closed_at_probe_cap"
        else:
            raise AssertionError(f"unexpected cycle status: {policy}/{scenario}/{cycle.get('status')}")
        
    # This oracle deliberately uses equal activity progress. Compare its
    # model-facing busy masks and finish times only with the matching policy.
    for kind in (("event", "cycle") if cycle_status["equal_progress"] ==
                 "completed_and_equal" else ("event",)):
        record = records[("equal_progress", scenario, kind)]
        for i in range(owners):
            name = f"comp{i}"
            require_close(f"{scenario}/{kind}/{name} finish", record[f"finish{i}"],
                          float(discrete["finish_s_by_request"][name]))
        for key, oracle_key in (("hbm_busy", "hbm_busy_s"),
                                ("hbm_shared", "hbm_shared_s"),
                                ("compute_busy", "compute_busy_s"),
                                ("compute_shared", "compute_shared_s")):
            require_close(f"{scenario}/{kind}/{key}", record[key], float(discrete[oracle_key]))
        for row in range(2):
            require_close(f"{scenario}/{kind}/row_busy{row}", record[f"row_busy{row}"],
                          float(discrete["row_busy_s"][row]))
            require_close(f"{scenario}/{kind}/row_shared{row}", record[f"row_shared{row}"],
                          float(discrete["row_shared_s"][row]))
    return {
        "scenario": scenario,
        "arrivals_s": arrivals,
        "equal_progress_oracle_finish_s": discrete["finish_s_by_request"],
        "equal_progress_oracle_hbm_busy_s": discrete["hbm_busy_s"],
        "equal_progress_oracle_hbm_shared_s": discrete["hbm_shared_s"],
        "cycle_status": cycle_status,
        "grouped_q_alpha_q1": "pure-compute toy policy; event ledger is checked against wave oracle and cycle against event; LP probe is separate",
        "grouped_q_alpha_qlocal": "GEMM-local canonical-Q policy; event ledger is checked against wave oracle and cycle against event; LP probe is separate",
        "status": "passed_with_fail_closed" if "expected_fail_closed_at_probe_cap" in
                  cycle_status.values() else "passed",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cpp_output", type=Path,
                        help="stdout captured from bounded_lifecycle")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--offset", action="append", default=[],
                        help="also check exact custom rational arrival offsets, e.g. 1/97")
    args = parser.parse_args()

    raw = json.loads(args.raw.read_text())
    oracle_module = load_oracle()
    waves, expected = oracle_module.make_waves(raw)
    records, checks = parse_records(args.cpp_output)
    results = [
        compare_scenario(oracle_module, raw, expected, records, checks, "solo", [0.0]),
        compare_scenario(oracle_module, raw, expected, records, checks,
                         "two_offset_0", [0.0, 0.0]),
        compare_scenario(oracle_module, raw, expected, records, checks,
                         "two_offset_005", [0.0, 0.05]),
    ]
    for offset in args.offset:
        name = "two_offset_r_" + offset.replace("/", "_")
        results.append(compare_scenario(
            oracle_module, raw, expected, records, checks, name,
            [0.0, float(Fraction(offset))]))
    status = "event_oracle_passed_with_expected_fail_closed" if any(
        row["status"] == "passed_with_fail_closed" for row in results) else "passed"
    print(json.dumps({
        "status": status,
        "fixture": expected["selected_tiles"],
        "waves": len(waves),
        "expected_per_owner": {
            "read_bytes": expected["hbm_read_bytes"],
            "write_bytes": expected["hbm_write_bytes"],
            "local_bytes": expected["local_bus_bytes"],
            "vec_ops": expected["vec_equivalent_ops"],
        },
        "results": results,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
