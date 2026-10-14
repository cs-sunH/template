#!/usr/bin/env python3
"""Verify measured width claims and fail-closed stop/resume records."""
import json
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = json.loads((HERE / "bitwidth_audit.json").read_text())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main() -> int:
    runs = REPORT["runs"]
    stops = REPORT["stop_and_resume_runs"]
    budget = REPORT["fixed_width_budget_audit"]
    require(len(runs) == 11, "expected all powered/expanded geometry fixtures")
    require(all(row["status"] == "passed" for row in runs),
            "one or more captured-fact solves did not pass")
    require(len(stops) == 8 and all(row["status_sequence"][-1] == "passed"
                                    for row in stops),
            "one or more stop/resume traces did not complete")
    require(budget["normalized_256_bit_state_fits_all_captured_runs"],
            "captured normalized rational state exceeds the 256-bit candidate")
    require(not budget["scratch_candidates"]["256"]["fits_all_captured_runs"],
            "the expected real 256-bit scratch counterexample was not observed")
    require(budget["scratch_candidates"]["544"]["fits_all_captured_runs"],
            "measured operation bounds exceed 544-bit scratch")
    require(budget["scratch_candidates"]["768"]["fits_all_captured_runs"],
            "measured operation bounds exceed 768-bit scratch")

    first = budget["scratch_candidates"]["256"]["first_failure"]
    require(first is not None and first["numerator_bits"] > 256,
            "256-bit scratch failure has no numerator-width witness")
    value = Fraction(first["witness"]["value"])
    require(value.numerator.bit_length() <= 256 and
            value.denominator.bit_length() <= 256,
            "scratch-overflow witness should still have a bounded normalized result")
    require(first["witness"]["numerator_bits"] > 256,
            "first scratch failure witness does not exceed 256 bits")
    operands = [Fraction(text) for text in first["witness"]["operands"]]
    require(len(operands) == 2 and "compare_" in first["operation"],
            "first scratch witness lacks an exact comparison operand pair")
    left, right = operands
    expected_cross_bits = max(
        left.numerator.bit_length() + right.denominator.bit_length(),
        right.numerator.bit_length() + left.denominator.bit_length())
    expected_den_bits = max(left.denominator.bit_length(),
                            right.denominator.bit_length())
    require(first["witness"]["numerator_bits"] == expected_cross_bits and
            first["witness"]["denominator_bits"] == expected_den_bits,
            "recorded comparison scratch does not match its exact operands")

    small = [row for row in stops if row["fixture"] not in ("large_k", "large_outputs")]
    require(all(row["status_sequence"] ==
                ["paused", "paused", "paused", "failed_closed", "passed"]
                for row in small),
            "integer/exact pause, capacity invalidation, backward-stop, resume sequence changed")
    huge = [row for row in stops if row["fixture"] in ("large_k", "large_outputs")]
    require(all(row["status_sequence"] == ["paused", "paused", "passed"] and
                row["final_events"] <= 256 for row in huge),
            "large stop/resume exceeded its retained global event budget")
    require(all(row.get("integer_stop_ns") == 17 for row in small),
            "integer nanosecond clamp input was not recorded")
    require(all(row["capacity_change_invalidated_seen_returns"] and
                row["backwards_stop_preserved_now"] for row in small),
            "capacity-change/backwards-stop fail-closed semantics changed")
    print("fixed_width_audit_check=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
