#!/usr/bin/env python3
"""Compare exact C++ scratch lifecycle results with independent grouped waves."""
import argparse
import json
import math
import shlex
import sys
from fractions import Fraction
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "die/validation/grouped_gemm_oracle.py").is_file())
sys.path.insert(0, str(ROOT / "die/validation"))
import compiled_gemm_oracle as reference
import grouped_gemm_oracle as grouped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--cpp-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.raw.read_text())
    waves, expected = reference.make_waves(raw)
    capacities = reference.make_capacities(raw)
    records = {}
    for line in args.cpp_output.read_text().splitlines():
        if not line.startswith("record "):
            continue
        row = dict(token.split("=", 1) for token in shlex.split(line)[1:])
        if row["policy"] == "grouped_q_alpha_qlocal":
            records[(row["scenario"], row["kind"])] = row
    scenarios = {"solo": [0.], "two_offset_0": [0., 0.],
                 "two_offset_005": [0., .05]}
    for offset in ("1/7", "1/97", "14/101", "91/97"):
        scenarios["two_offset_r_" + offset.replace("/", "_")] = [0., float(Fraction(offset))]
    results = []
    for name, arrivals in scenarios.items():
        independent = grouped.discrete_grouped(waves, expected["true_output_tiles"], arrivals, capacities)
        for kind in ("event", "cycle"):
            actual = records[(name, kind)]
            if actual["status"] != "completed":
                raise AssertionError((name, kind, actual["status"]))
            comparisons = [(f"finish{i}", independent["finish_s_by_request"][f"comp{i}"])
                           for i in range(len(arrivals))]
            comparisons += [(k, independent[v]) for k, v in (
                ("hbm_busy", "hbm_busy_s"), ("hbm_shared", "hbm_shared_s"),
                ("compute_busy", "compute_busy_s"), ("compute_shared", "compute_shared_s"))]
            for row in range(2):
                comparisons += [(f"row_busy{row}", independent["row_busy_s"][row]),
                                (f"row_shared{row}", independent["row_shared_s"][row])]
            for key, wanted in comparisons:
                got = float(Fraction(actual[key]))
                if not math.isclose(got, wanted, rel_tol=2e-10, abs_tol=2e-12):
                    raise AssertionError((name, kind, key, got, wanted))
        results.append({"scenario": name, "event_and_cycle": "passed",
            "finish_s": independent["finish_s_by_request"],
            "hbm_shared_s": independent["hbm_shared_s"]})
    args.output.write_text(json.dumps({"status": "passed", "scenarios": 7,
        "wave_count": len(waves), "canonical_compute_quantity": "LOCAL bytes",
        "scope": "1-Core FP32 scratch lifecycle, fixed three finite domains; not production certificate",
        "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
