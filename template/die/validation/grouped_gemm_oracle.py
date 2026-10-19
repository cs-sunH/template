#!/usr/bin/env python3
"""Independent discrete GEMM reference using the specified owner Q/alpha policy.

The equal-progress reference is retained alongside the grouped comparison.
Only this offline oracle enumerates waves. Production precision thresholds,
source constants, work ledgers and the Runtime are not changed here.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import compiled_gemm_oracle as reference

DOMAINS = tuple(reference.RESOURCE_IDS[name]
                for name in ("COMPUTE_GATE", "HBM0", "HBM1"))
K_READINESS_DIAGNOSTIC_SOURCE = (
    "diagnostic_cpp_k_ready_facts_and_actual_public_runtime")


def grouped_rates(active: list[Any], capacities: list[float],
                  quantities: list[float]) -> list[float]:
    """Solve monotone grouped max-min tiers for eligible discrete activities.

    y_a=Q_a*v_a and alpha_a=Q_a/sum_owner(Q), so one unit of the
    fairness floor increases v_a by 1/sum_owner(Q). Owner sums stay fixed
    throughout the tiers; they change only when activity membership changes.
    Signed continuous-credit constraints are absent from this discrete fixture.
    """
    count = len(active)
    if len(quantities) != count:
        raise ValueError("one canonical quantity is required per activity")
    if any(not math.isfinite(capacities[r]) or capacities[r] < 0 for r in DOMAINS):
        raise ValueError("domain capacities must be finite and nonnegative")
    # A disabled domain makes any positive-demand activity infeasible. It must
    # not keep a share of its owner's Q; signed-credit coupling is out of scope.
    eligible = {i for i, item in enumerate(active)
                if not any(item.work[r] > 0 and capacities[r] == 0 for r in DOMAINS)}
    totals: dict[int, float] = {}
    for i, (item, quantity) in enumerate(zip(active, quantities)):
        if not math.isfinite(quantity) or quantity <= 0:
            raise ValueError("canonical quantity must be positive and finite")
        if i not in eligible:
            continue
        totals[item.request] = totals.get(item.request, 0.0) + quantity
    if any(not math.isfinite(total) for total in totals.values()):
        raise OverflowError("owner canonical quantity overflow")
    weights = [1.0 / totals[item.request] if i in eligible else 0.0
               for i, item in enumerate(active)]
    rates = [0.0] * count
    free = eligible
    for _ in range(count):
        if not free:
            break
        # Each candidate is a primal floor upper bound. A binding row supplies
        # its positive reciprocal slope as a one-row dual certificate.
        rows: list[tuple[str, int, float]] = []
        for resource in DOMAINS:
            used = math.fsum(item.work[resource] * rates[i]
                             for i, item in enumerate(active))
            slope = math.fsum(active[i].work[resource] * weights[i] for i in free)
            if slope <= 0:
                continue
            residual = capacities[resource] - used
            tolerance = 1e-10 * max(1.0, abs(capacities[resource]), abs(used))
            if residual < -tolerance:
                raise ArithmeticError("grouped oracle exceeded resource capacity")
            rows.append(("resource", resource, max(0.0, residual) / slope))
        for i in free:
            if active[i].private_time_s > 0:
                residual = 1.0 / active[i].private_time_s - rates[i]
                if residual < -1e-10 * max(1.0, abs(rates[i])):
                    raise ArithmeticError("grouped oracle exceeded private bound")
                rows.append(("private", i, max(0.0, residual) / weights[i]))
        if not rows:
            raise ArithmeticError("grouped oracle activity has no finite bound")
        delta = min(row[2] for row in rows)
        if not math.isfinite(delta) or delta < 0:
            raise ArithmeticError("invalid grouped oracle fairness floor")
        for i in free:
            rates[i] += weights[i] * delta
        frozen: set[int] = set()
        for kind, index, bound in rows:
            if abs(bound - delta) > 1e-12 * max(1.0, abs(bound), abs(delta)):
                continue
            if kind == "private":
                frozen.add(index)
            else:
                frozen.update(i for i in free if active[i].work[index] > 0)
        if not frozen:
            raise ArithmeticError("grouped oracle failed to freeze a bottleneck")
        free.difference_update(frozen)
    if free:
        raise ArithmeticError("grouped oracle exceeded its activity-count tier bound")
    for resource in DOMAINS:
        used = math.fsum(item.work[resource] * rates[i]
                         for i, item in enumerate(active))
        if used > capacities[resource] * (1.0 + 1e-9):
            raise ArithmeticError("grouped oracle final resource budget exceeded")
    return rates


def discrete_grouped(waves: list[Any], output_count: int,
                     arrivals: list[float], capacities: list[float]) -> dict[str, Any]:
    def allocate(active, rates):
        quantities = []
        for item in active:
            wave = waves[item.wave]
            quantities.append(float(wave.compute_local_bytes or 1)
                              if item.stage == "compute" else float(sum(
                                  wave.read_row_bytes if item.stage == "load"
                                  else wave.write_row_bytes)))
        return grouped_rates(active, rates, quantities)

    result = reference.discrete_oracle(waves, output_count, arrivals, capacities,
                                       rate_allocator=allocate)
    result["allocation_policy"] = "owner_grouped_q_alpha"
    result["canonical_quantity"] = (
        "load/store=physical HBM payload; compute=LOCAL payload, "
        "even though LOCAL capacity is nonbinding")
    return result


def build_report(raw: dict[str, Any]) -> dict[str, Any]:
    report = reference.build_report(raw)
    legacy_comparisons = report["comparisons"]
    waves, expected = reference.make_waves(raw)
    if sum(wave.compute_local_bytes for wave in waves) != expected["local_bus_bytes"]:
        raise AssertionError("independent per-wave LOCAL quantities do not conserve work")
    capacities = reference.make_capacities(raw)
    solo = discrete_grouped(waves, expected["true_output_tiles"], [0.0], capacities)
    comparisons = {"standalone": reference.compare_measurement(
        raw["standalone"], solo, 0.0, False), "two_comp": []}
    for actual in raw["two_comp"]:
        offset = float(actual["phase_offset_s"])
        oracle = discrete_grouped(waves, expected["true_output_tiles"],
                                  [0.0, offset], capacities)
        comparisons["two_comp"].append(reference.compare_measurement(
            actual, oracle, offset, True))
    passed = (report["compiled_plan_ledger"]["status"] == "passed" and
              comparisons["standalone"]["status"] == "passed" and
              all(item["status"] == "passed" for item in comparisons["two_comp"]))
    status = "passed" if passed else "failed"
    report.update(schema_version=2,
                  oracle="independent_compiled_gemm_grouped_discrete_consumption_wave_v3",
                  readiness_semantics_revision=(
                      reference.READINESS_SEMANTICS_REVISION),
                  allocation_policy="owner_grouped_q_alpha",
                  legacy_equal_progress_comparisons=legacy_comparisons,
                  comparisons=comparisons, status=status)
    report["available_granularity_levels"]["E5_fixture_comparison"] = status
    report["available_granularity_levels"]["granularity_validation"] = (
        "failed" if not passed else "not_run")
    report["root_cause_clue"]["fairness_reference"] = (
        "Legacy equal-progress differs from the specified owner Q/alpha policy "
        "when owners have different eligible stage sets. Both comparisons are "
        "retained; grouped policy is the design-aligned reference. Macro and "
        "discrete canonical quantities also span different K/frame units.")
    report["notes"].append(
        "This oracle corrects the reference allocation policy; it does not alter "
        "the 3%/5% error limits or certify a production K lifecycle.")
    if (raw.get("measurement_source") == K_READINESS_DIAGNOSTIC_SOURCE and
            raw.get("baseline_compiled_plan_executed") is False):
        diagnostic_only = raw.get("diagnostic_only")
        if diagnostic_only is None:
            diagnostic_only = raw.get("standalone", {}).get(
                "readiness_diagnostic", {}).get("diagnostic_only")
        report["fixture_kind"] = "diagnostic_cpp_k_readiness_facts_and_actual_public_runtime"
        report["status_scope"] = "diagnostic_only"
        report["diagnostic_scope"] = {
            "diagnostic_only": diagnostic_only is True,
            "baseline_compiled_plan_executed": False,
            "compiled_plan_role": "geometry_mapping_and_work_facts_only",
            "diagnostic_stage_plans_hand_constructed": True,
            "acceptance_limits_are_production_certification": False,
        }
        report["compiled_plan"]["execution_role"] = (
            "geometry_mapping_and_work_facts_only; not executed by this diagnostic")
        report["fixture_scope"]["hardware_evidence"] = (
            "actual public C++ Runtime with synthetic fixture hardware capacities; "
            "diagnostic only")
        report["available_granularity_levels"] = {
            "diagnostic_k_readiness_comparison": status,
            "production_granularity_validation": "not_run",
        }
        report["root_cause_clue"] = {
            "applicability": (
                "compiled_plan_metadata_only; compiled_plan_not_executed"),
            "baseline_compiled_plan_executed": False,
            "finding": (
                "This comparison checks discrete per-K readiness against the actual "
                "public Runtime using diagnostic stage Plans. It does not measure "
                "compiled E5 execution or select a production granularity."),
        }
        report["notes"] = [
            note for note in report["notes"]
            if "no Plan was hand-constructed" not in note]
        report["notes"].append(
            "The compiled plan supplies geometry, mapping, and work facts only. "
            "The C++ diagnostic executes hand-constructed per-stage Plans through "
            "the public Runtime; its comparison does not certify E5 or minimum E.")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path,
                        default=Path("work/build-die/coarse_compiled_gemm_oracle_driver"))
    parser.add_argument("--raw", type=Path, help="previous real C++ driver JSON")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.raw:
        raw = json.loads(args.raw.read_text())
    else:
        driver = args.driver.resolve()
        if not driver.is_file():
            report = {"status": "not_run", "granularity_validation": "not_run",
                      "reason": f"real C++ driver unavailable: {driver}"}
            text = json.dumps(report, indent=2) + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text)
            else:
                print(text, end="")
            return 2
        raw = json.loads(subprocess.run([str(driver)], check=True,
                                        capture_output=True, text=True).stdout)
    report = build_report(raw)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
