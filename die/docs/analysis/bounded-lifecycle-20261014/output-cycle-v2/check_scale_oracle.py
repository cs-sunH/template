#!/usr/bin/env python3
"""Independent fixture-wave expansion for small R; closed forms for large R."""

from __future__ import annotations

import importlib.util
import json
import shlex
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents
            if (p / "die" / "validation" / "compiled_gemm_oracle.py").is_file())
ORACLE_PATH = ROOT / "die" / "validation" / "compiled_gemm_oracle.py"
BASE_RAW = HERE / "raw-fixture.json"


def load_oracle():
    spec = importlib.util.spec_from_file_location("compiled_gemm_oracle", ORACLE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import independent wave oracle: {ORACLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_output(path: Path):
    configs: dict[str, dict[str, str]] = {}
    records: dict[tuple[str, str], dict[str, str]] = {}
    checks: dict[str, dict[str, str]] = {}
    for line in path.read_text().splitlines():
        if line.startswith("config "):
            row = dict(field.split("=", 1) for field in shlex.split(line)[1:])
            configs[row["scenario"]] = row
        elif line.startswith("record "):
            row = dict(field.split("=", 1) for field in shlex.split(line)[1:])
            records[(row["kind"], row["scenario"])] = row
        elif line.startswith("check "):
            row = dict(field.split("=", 1) for field in shlex.split(line)[1:])
            checks[row["scenario"]] = row
    if not configs:
        raise AssertionError("missing scale config")
    return configs, records, checks


def dynamic_raw(m: int, n: int, k: int) -> dict:
    raw = json.loads(BASE_RAW.read_text())
    plan = raw["compiled_plan"]
    plan["task"]["tensor_shapes"] = {"m": m, "n": n, "k": k}
    summary = plan["summary"]
    tm, tn, tk = (int(summary[name]) for name in ("tm", "tn", "tk"))
    tm_count = (m + tm - 1) // tm
    tn_count = (n + tn - 1) // tn
    tk_count = (k + tk - 1) // tk
    output_count = tm_count * tn_count
    summary["output_round_count"] = output_count
    summary["consumption_wave_count"] = output_count * tk_count
    return raw


def expected_closed(m: int, n: int, k: int, tm: int, tn: int, tk: int) -> dict[str, int]:
    tm_count, tn_count, tk_count = ((x + t - 1) // t
                                    for x, t in ((m, tm), (n, tn), (k, tk)))
    read = 4 * k * (tn_count * m + tm_count * n)
    write = 4 * m * n
    ops = 2 * m * n * k
    local = read + (2 * tk_count - 1) * 4 * m * n
    return {"read": read, "write": write, "ops": ops, "local": local,
            "outputs": tm_count * tn_count,
            "waves": tm_count * tn_count * tk_count}


def check(path: Path, expanded: bool) -> dict:
    configs, records, checks = parse_output(path)
    config = next(iter(configs.values()))
    m, n, k = (int(config[name]) for name in ("m", "n", "k"))
    tm, tn, tk = (int(config[name]) for name in ("tm", "tn", "tk"))
    expected = expected_closed(m, n, k, tm, tn, tk)
    if int(config["outputs"]) != expected["outputs"] or int(config["waves"]) != expected["waves"]:
        raise AssertionError("C++ wave geometry differs from closed-form tile counts")
    if k % tk == 0:
        raise AssertionError("scale fixture lost its K-tail")

    expansion: dict[str, object] = {"performed": False}
    if expanded:
        oracle = load_oracle()
        raw = dynamic_raw(m, n, k)
        waves, wave_summary = oracle.make_waves(raw)
        if len(waves) != expected["waves"]:
            raise AssertionError("independent expanded wave count mismatch")
        for key, field in (("read", "hbm_read_bytes"), ("write", "hbm_write_bytes"),
                           ("ops", "vec_equivalent_ops"), ("local", "local_bus_bytes"),
                           ("outputs", "true_output_tiles"),
                           ("waves", "consumption_wave_count")):
            if int(wave_summary[field]) != expected[key]:
                raise AssertionError(f"expanded oracle {key} differs from closed form")
        expansion = {"performed": True, "wave_count": len(waves),
                     "ledger": {key: wave_summary[field] for key, field in (
                         ("read", "hbm_read_bytes"), ("write", "hbm_write_bytes"),
                         ("ops", "vec_equivalent_ops"), ("local", "local_bus_bytes"))}}

    result_records = {}
    for scenario in configs:
        cycle = records.get(("cycle", scenario))
        if cycle is None:
            raise AssertionError(f"missing accelerated record for {scenario}")
        if cycle["status"] == "completed":
            for owner in range(int(configs[scenario]["owners"])):
                for key, suffix in (("read", "read"), ("write", "write"),
                                    ("ops", "ops"), ("local", "local")):
                    got = int(cycle[f"owner{owner}_{suffix}"])
                    if got != expected[key]:
                        raise AssertionError(f"{scenario} owner{owner} {key}: {got} != {expected[key]}")
                if int(cycle[f"owner{owner}_retired"]) != expected["outputs"]:
                    raise AssertionError(f"{scenario} owner{owner} retirement mismatch")
            check_row = checks.get(scenario)
            if expanded and check_row and check_row.get("cycle_equals_event") != "true":
                raise AssertionError(f"{scenario}: event/cycle check failed")
        elif cycle["status"] != "fail_closed":
            raise AssertionError(f"unexpected accelerated status {cycle['status']}")

        event = records.get(("event", scenario))
        event_count = None
        if expanded:
            if event is None or event["status"] != "completed":
                raise AssertionError(f"{scenario}: small-R event oracle did not complete")
            if checks.get(scenario, {}).get("cycle_equals_event") != "true":
                raise AssertionError(f"{scenario}: small-R cycle differs from event path")
            event_count = int(event["executed_events"])
            for owner in range(int(configs[scenario]["owners"])):
                for key, suffix in (("read", "read"), ("write", "write"),
                                    ("ops", "ops"), ("local", "local")):
                    if int(event[f"owner{owner}_{suffix}"]) != expected[key]:
                        raise AssertionError(f"{scenario} event owner{owner} {key} ledger mismatch")
        elif event is not None:
            raise AssertionError("large-R run unexpectedly included event-by-event expansion")

        result_records[scenario] = {
            "status": cycle["status"],
            "event_executed_events": event_count,
            "executed_events": int(cycle["executed_events"]),
            "inner_skipped_cycles": int(cycle["inner_skipped_cycles"]),
            "outer_skipped_cycles": int(cycle["outer_skipped_cycles"]),
            "outer_skipped_outputs": int(cycle["outer_skipped_outputs"]),
            "outer_exact_cycles": int(cycle["outer_exact_cycles"]),
            "outer_cached_cycle_hits": int(cycle["outer_cached_cycle_hits"]),
            "outer_unskippable_cycle_hits": int(cycle["outer_unskippable_cycle_hits"]),
            "outer_rejected_key_repeats": int(cycle["outer_rejected_key_repeats"]),
            "outer_cycle_output_step": int(cycle["outer_cycle_output_step"]),
            "max_inner_probes": int(cycle["max_inner_probes"]),
            "max_outer_probes": int(cycle["max_outer_probes"]),
            "max_progress_num_bits": int(cycle["max_progress_num_bits"]),
            "max_progress_den_bits": int(cycle["max_progress_den_bits"]),
            "max_rate_num_bits": int(cycle["max_rate_num_bits"]),
            "max_rate_den_bits": int(cycle["max_rate_den_bits"]),
            "owner0_next_load": int(cycle["owner0_next_load"]),
            "owner0_retired": int(cycle["owner0_retired"]),
        }
    return {"fixture": {"m": m, "n": n, "k": k, "tm": tm, "tn": tn, "tk": tk,
                        **expected},
            "expanded_wave_oracle": expansion,
            "cycles": result_records,
            "interpretation": ("expanded-oracle and event comparison completed" if expanded else
                               "no wave/event expansion; completed cycle totals checked by closed form, fail-closed prefixes remain partial")}


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[2] not in ("expanded", "closed"):
        raise SystemExit("usage: check_scale_oracle.py CPP_OUTPUT expanded|closed")
    result = check(Path(sys.argv[1]), sys.argv[2] == "expanded")
    result["completion_status"] = (
        "completed" if all(row["status"] == "completed" for row in result["cycles"].values())
        else "fail_closed_on_probe_cap")
    result["checker_status"] = "passed"  # ledger/prefix assertions justify only this label
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
