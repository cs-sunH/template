#!/usr/bin/env python3
"""Independent checks for the solo shared-HBM descriptor prototype."""
from __future__ import annotations

from fractions import Fraction as F
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents
            if (p / "die" / "validation" / "compiled_gemm_oracle.py").is_file())
ORACLE_PATH = ROOT / "die" / "validation" / "compiled_gemm_oracle.py"
DRIVER = ROOT / "work" / "build-die" / "coarse_compiled_gemm_oracle_driver"
sys.path.insert(0, str(HERE))
from solo_shared_hbm import GemmFP32, Solo  # noqa: E402


def load_oracle():
    spec = importlib.util.spec_from_file_location("compiled_gemm_oracle", ORACLE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load independent oracle: {ORACLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dynamic_raw(raw: dict, m: int, n: int, k: int) -> dict:
    raw = json.loads(json.dumps(raw))
    plan = raw["compiled_plan"]
    plan["task"]["tensor_shapes"] = {"m": m, "n": n, "k": k}
    s = plan["summary"]
    tm, tn, tk = (int(s[name]) for name in ("tm", "tn", "tk"))
    outputs = ((m + tm - 1) // tm) * ((n + tn - 1) // tn)
    waves = (k + tk - 1) // tk
    s["output_round_count"] = outputs
    s["consumption_wave_count"] = outputs * waves
    return raw


class OracleGeometry:
    """Lazy exact work reconstruction from the compiler driver's selected plan."""
    def __init__(self, raw: dict, oracle: object):
        self.raw = raw
        self._wave_table, self.summary = oracle.make_waves(raw)
        self.M, self.N, self.K = (int(self.summary["shape"][x]) for x in ("m", "n", "k"))
        self.tm, self.tn, self.tk = (int(self.summary["selected_tiles"][x]) for x in ("tm", "tn", "tk"))
        self.mt, self.nt, self.kt = (int(self.summary["selected_tiles"][x]) for x in ("Tm", "Tn", "Tk"))
        self.outputs = int(self.summary["true_output_tiles"])
        self.wave_count = len(self._wave_table)
        self.waves = self.wave_count
        hw = raw["synthetic_hardware"]
        self.eta_hbm = F(str(hw["eta_hbm_matrix"]))
        self.compute_rate = F(str(hw["p_vec_per_core_ops_per_s"])) * F(str(hw["eta_vec"]))
        raw_caps = oracle.make_capacities(raw)
        self.capacities = tuple(F(str(raw_caps[i])) for i in (0, 4, 5))

    @property
    def waves_count(self):
        return self.wave_count

    def facts(self, w):
        if not 0 <= w < self.wave_count:
            raise ValueError("wave index out of range")
        x = self._wave_table[w]
        q, ki = divmod(w, self.kt)
        mi, ni = divmod(q, self.nt)
        m = min(self.tm, self.M - mi * self.tm)
        n = min(self.tn, self.N - ni * self.tn)
        k = min(self.tk, self.K - ki * self.tk)
        read = sum(x.read_row_bytes)
        write = sum(x.write_row_bytes)
        return (m, n, k, x.final_k, ki == 0, read, write,
                x.vec_ops, x.compute_local_bytes)

    def work(self, stage, w):
        x = self._wave_table[w]
        m, n, k, final, first, read, write, ops, local = self.facts(w)
        if stage == 0:
            row0, row1 = x.read_row_bytes
            return (F(0), F(row0) / self.eta_hbm, F(row1) / self.eta_hbm), F(read) / self.eta_hbm, F(0)
        if stage == 1:
            c = F(ops) / self.compute_rate
            return (c, F(0), F(0)), F(local), c
        if stage == 2 and final:
            row0, row1 = x.write_row_bytes
            return (F(0), F(row0) / self.eta_hbm, F(row1) / self.eta_hbm), F(write) / self.eta_hbm, F(0)
        raise ValueError("invalid oracle stage")

    def output_shape(self, q):
        mi, ni = divmod(q, self.nt)
        return min(self.tm, self.M - mi * self.tm), min(self.tn, self.N - ni * self.tn)

    def same_shape_run_end(self, q):
        shape = self.output_shape(q)
        end = self.outputs
        m, n = shape
        mi, ni = divmod(q, self.nt)
        if self.N % self.tn and self.nt > 1:
            end = min(end, mi * self.nt + (self.nt - 1 if n == self.tn else self.nt))
        if self.M % self.tm and self.mt > 1:
            mstart = (self.mt - 1) * self.nt
            end = min(end, mstart if m == self.tm else self.outputs)
        return max(q + 1, end)

    def expected(self):
        return {"read": int(self.summary["hbm_read_bytes"]),
                "write": int(self.summary["hbm_write_bytes"]),
                "ops": int(self.summary["vec_equivalent_ops"]),
                "local": int(self.summary["local_bus_bytes"]),
                "outputs": self.outputs, "waves": self.wave_count}


def close(label: str, actual: float, expected: float, tol: float = 2.0e-10):
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{label}: {actual:.17g} != {expected:.17g}")


def main() -> int:
    oracle = load_oracle()
    raw = json.loads(subprocess.check_output([str(DRIVER)], text=True))
    small_raw = dynamic_raw(raw, 17, 65, 129)
    geom = OracleGeometry(small_raw, oracle)
    waves, summary = oracle.make_waves(small_raw)
    capacities = oracle.make_capacities(small_raw)

    # This matches the independent oracle's equal-activity-progress policy.
    equal = Solo(geom, cap=10_000, accelerate=False, policy="equal_progress").run()
    discrete = oracle.discrete_oracle(waves, geom.outputs, [0.0], capacities)
    close("compiled oracle solo finish", float(F(equal["finish_s"])),
          float(discrete["finish_s_by_request"]["comp0"]))
    for ours, theirs in (("hbm_busy", "hbm_busy_s"),
                         ("hbm0_busy", None), ("hbm1_busy", None),
                         ("compute_busy", "compute_busy_s")):
        expected = (float(discrete[theirs]) if theirs else
                    float(discrete["row_busy_s"][0 if ours == "hbm0_busy" else 1]))
        close(f"compiled oracle {ours}", float(F(equal["ledger"][ours])), expected)
    for i, field in enumerate(("read", "write", "ops", "local")):
        expected_key = {"read": "hbm_read_bytes", "write": "hbm_write_bytes",
                        "ops": "vec_equivalent_ops", "local": "local_bus_bytes"}[field]
        if int(equal["expected_work"][field]) != int(summary[expected_key]):
            raise AssertionError(f"compiled wave {field} ledger mismatch")

    # Grouped-Q return powering is checked against its own exact expanded event
    # schedule on a small geometry, then shared-HBM ledgers are checked closed.
    grouped_event = Solo(geom, cap=10_000, accelerate=False).run()
    grouped_powered = Solo(geom, cap=256, accelerate=True).run()
    for field in ("finish_s",):
        if grouped_event[field] != grouped_powered[field]:
            raise AssertionError(f"grouped accelerated {field} mismatch")
    if grouped_event["ledger"] != grouped_powered["ledger"]:
        raise AssertionError("grouped accelerated ledger mismatch")

    dense_g = GemmFP32(65, 65, 4097)
    dense_event = Solo(dense_g, cap=10_000, accelerate=False).run()
    dense_powered = Solo(dense_g, cap=256, accelerate=True).run()
    if (dense_event["finish_s"] != dense_powered["finish_s"] or
            dense_event["ledger"] != dense_powered["ledger"]):
        raise AssertionError("full/tail-K and output-row powering differs from exact event path")

    # Exercise different active-set bottlenecks. Each accelerated run must
    # reproduce its unaccelerated exact Fraction event path.
    bottlenecks = []
    for name, caps in (("load_store_limited", (F(1), F(1000), F(1000))),
                       ("compute_limited", (F(1), F(10**30), F(10**30)))):
        g = GemmFP32(65, 65, 129, capacities=caps)
        exact = Solo(g, cap=10_000, accelerate=False).run()
        powered = Solo(g, cap=256, accelerate=True).run()
        if exact["finish_s"] != powered["finish_s"] or exact["ledger"] != powered["ledger"]:
            raise AssertionError(f"{name}: accelerated schedule differs")
        bottlenecks.append({"name": name, "events_expanded": exact["events"],
                            "events_powered": powered["events"],
                            "finish_s": powered["finish_s"],
                            "hbm_busy": powered["ledger"]["hbm_busy"],
                            "compute_busy": powered["ledger"]["compute_busy"]})

    # Large K and output counts: a global 256 discovery/event budget remains
    # constant; each repeat count is obtained by arithmetic class boundaries.
    large_k = Solo(GemmFP32(65, 65, 64 * 1_000_000 + 1), cap=256).run()
    large_outputs = Solo(GemmFP32(16 * 1_000_004 + 1, 65, 4097), cap=256).run()
    store_pressure_exact = Solo(GemmFP32(65, 65, 1), cap=10_000,
                                accelerate=False).run()
    store_pressure_powered = Solo(GemmFP32(65, 65, 1), cap=256,
                                  accelerate=True).run()
    if (store_pressure_exact["finish_s"] != store_pressure_powered["finish_s"] or
            store_pressure_exact["ledger"] != store_pressure_powered["ledger"] or
            store_pressure_powered["peak_output_slots"] != 2):
        raise AssertionError("final-K/store/retire output backpressure mismatch")

    # Exercise arithmetic tile-class boundaries across small full/tail shapes.
    edge_count = 0
    for tm, tn in ((4, 3), (2, 4)):
        for m in (1, 3, 4, 5, 9, 17):
            for n in (1, 2, 3, 5, 9):
                for k in (1, 63, 65, 129):
                    g = GemmFP32(m, n, k, tm, tn, 64 if k > 1 else 1)
                    exact = Solo(g, cap=10_000, accelerate=False).run()
                    powered = Solo(g, cap=256, accelerate=True).run()
                    if (exact["finish_s"] != powered["finish_s"] or
                            exact["ledger"] != powered["ledger"]):
                        raise AssertionError(f"edge geometry differs: {(m, n, k, tm, tn)}")
                    edge_count += 1
    for label, result in (("large_k", large_k), ("large_outputs", large_outputs)):
        if result["events"] > result["global_event_cap"]:
            raise AssertionError(f"{label}: discovery exceeded global cap")
        if result["status"] != "passed":
            raise AssertionError(f"{label}: did not complete")

    result = {
        "status": "passed_offline_independent_solo_checks",
        "scope": "FP32 strict output-stationary formula; exact Fraction shared HBM0/HBM1 + COMPUTE_GATE; no production certificate",
        "compiled_cpp_driver": str(DRIVER),
        "compiled_wave_oracle": {"shape": [17, 65, 129],
            "waves": len(waves), "outputs": geom.outputs,
            "equal_progress_finish_s": equal["finish_s"],
            "independent_finish_s": discrete["finish_s_by_request"]["comp0"],
            "equal_progress_busy_s": {"hbm": equal["ledger"]["hbm_busy"],
                                       "compute": equal["ledger"]["compute_busy"]},
            "independent_busy_s": {"hbm": discrete["hbm_busy_s"],
                                   "row": discrete["row_busy_s"],
                                   "compute": discrete["compute_busy_s"]}},
        "grouped_q_exact_expansion": {"events": grouped_event["events"],
                                      "accelerated_events": grouped_powered["events"],
                                      "descriptors": grouped_powered["descriptors"],
                                      "ledger": grouped_powered["ledger"]},
        "full_k_and_output_expansion": {"geometry": dense_powered["geometry"],
                                        "events_expanded": dense_event["events"],
                                        "events_powered": dense_powered["events"],
                                        "descriptors": dense_powered["descriptors"],
                                        "ledger": dense_powered["ledger"]},
        "bottleneck_checks": bottlenecks,
        "output_credit_check": {"geometry": store_pressure_powered["geometry"],
                                 "events_expanded": store_pressure_exact["events"],
                                 "events_powered": store_pressure_powered["events"],
                                 "peak_output_slots": store_pressure_powered["peak_output_slots"],
                                 "retired_outputs": store_pressure_powered["expected_work"]["outputs"],
                                 "finish_s": store_pressure_powered["finish_s"]},
        "full_tail_edge_sweep": {"exact_geometry_count": edge_count,
                                 "status": "all accelerated ledgers and finish times equal expanded paths"},
        "large_k": {"geometry": large_k["geometry"], "events": large_k["events"],
                    "cap": large_k["global_event_cap"], "skipped_waves": large_k["skipped_waves"],
                    "skipped_outputs": large_k["skipped_outputs"],
                    "descriptor_count": large_k["descriptor_count"]},
        "large_outputs": {"geometry": large_outputs["geometry"],
                          "events": large_outputs["events"],
                          "cap": large_outputs["global_event_cap"],
                          "skipped_waves": large_outputs["skipped_waves"],
                          "skipped_outputs": large_outputs["skipped_outputs"],
                          "descriptor_count": large_outputs["descriptor_count"]},
        "limitations": ["single owner only; no arrivals, external capacity changes, cancellation, or runtime stop events",
                         "strict FP32 bytes only; INT4 packed-tail work needs its own exact fact producer",
                         "nonidentity solo return maps fail closed", "model-to-model only; no silicon claim"]}
    (HERE / "selfcheck.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
