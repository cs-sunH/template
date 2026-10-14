#!/usr/bin/env python3
"""Run bounded exact checks over real public-Runtime GEMM class facts."""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(STUDY))
import solo_facts_13 as solo


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNTIME_ORACLE = load_module(
    "os_lifecycle_runtime_public_ns_oracle",
    ROOT / "die/validation/os_lifecycle_runtime_oracle.py",
)


def as_json(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {str(k): as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(x) for x in value]
    return value


def exact_continuous_oracle(data: dict) -> dict:
    """Independent 13-row queue/allocator run without integer-ns wake rounding."""
    base = RUNTIME_ORACLE

    class ContinuousOracle(base.Oracle):
        def __init__(self, report):
            super().__init__(report)
            self.resource_service = [F(0) for _ in range(13)]
            self.resource_busy = [F(0) for _ in range(13)]

        def advance_continuous(self, goal):
            while self.active and self.now < goal:
                allocation = base.rates(self.active, self.capacities)
                deadlines = {
                    stage: item["remaining"] / allocation[stage]
                    for stage, item in self.active.items()
                    if allocation[stage] > 0
                }
                if not deadlines:
                    raise AssertionError("independent continuous allocator deadlocked")
                delta = min(min(deadlines.values()), goal - self.now)
                busy = [False] * 13
                for stage, item in self.active.items():
                    rate = allocation[stage]
                    portion = delta * rate
                    item["remaining"] -= portion
                    self.int4 += portion * item["int4"]
                    self.vec += portion * item["vec"]
                    for row in range(2):
                        for direction in range(2):
                            self.hbm[row][direction] += (
                                portion * item["hbm"][row][direction]
                            )
                    for row, work in enumerate(item["work"]):
                        self.resource_service[row] += portion * work
                        busy[row] |= rate > 0 and work > 0
                for row, in_use in enumerate(busy):
                    if in_use:
                        self.resource_busy[row] += delta
                if busy[0]:
                    self.compute_busy += delta
                if busy[4] or busy[5]:
                    self.hbm_busy += delta
                for row in range(2):
                    if busy[row + 4]:
                        self.row_busy[row] += delta
                self.now += delta
                for stage in base.STAGES:
                    if (stage in self.active and
                            self.active[stage]["remaining"] == 0):
                        self.complete(stage)
                        self.events += 1
            self.now = goal

        def run_continuous(self):
            while self.retired < self.outputs:
                self.settle()
                if not self.active:
                    raise AssertionError("independent continuous queues deadlocked")
                allocation = base.rates(self.active, self.capacities)
                delta = min(
                    item["remaining"] / allocation[stage]
                    for stage, item in self.active.items()
                    if allocation[stage] > 0
                )
                self.advance_continuous(self.now + delta)
            if self.active or self.inputs or self.outputs_live:
                raise AssertionError("independent continuous queues did not drain")
            return self

    oracle = ContinuousOracle(data).run_continuous()
    return {
        "elapsed_s": oracle.now,
        "resource_service_13": oracle.resource_service,
        "resource_busy_s_13": oracle.resource_busy,
        "hbm_payload_row_direction": oracle.hbm,
        "int4_ops": oracle.int4,
        "vec_eqops": oracle.vec,
        "compute_busy_s": oracle.compute_busy,
        "hbm0_busy_s": oracle.row_busy[0],
        "hbm1_busy_s": oracle.row_busy[1],
        "hbm_busy_s": oracle.hbm_busy,
        "events": oracle.events,
    }


def check_continuous(candidate: dict, reference: dict) -> None:
    mapping = {
        "elapsed_s": "elapsed_s",
        "resource_service_13": "resource_service_13",
        "resource_busy_s_13": "resource_busy_s_13",
        "hbm_row_payload_bytes": "hbm_payload_row_direction",
        "int4_ops": "int4_ops",
        "vec_eqops": "vec_eqops",
        "compute_busy_s": "compute_busy_s",
        "hbm0_busy_s": "hbm0_busy_s",
        "hbm1_busy_s": "hbm1_busy_s",
        "hbm_busy_s": "hbm_busy_s",
    }
    for candidate_field, reference_field in mapping.items():
        candidate_value = candidate["ledger"][candidate_field]
        reference_value = as_json(reference[reference_field])
        if candidate_value != reference_value:
            raise AssertionError(
                f"continuous oracle mismatch in {candidate_field}: "
                f"{candidate_value!r} != {reference_value!r}"
            )


def check_shape_runs(g: solo.FactsGeometry, *, full: bool) -> dict:
    if full:
        starts = range(g.outputs)
    else:
        starts = sorted({
            0, min(g.nt - 1, g.outputs - 1), min(g.nt, g.outputs - 1),
            max(0, g.outputs - g.nt - 1), max(0, g.outputs - g.nt),
            g.outputs - 1,
        })
    checked = 0
    for q in starts:
        end = g.same_shape_run_end(q)
        if not q < end <= g.outputs:
            raise AssertionError(f"invalid same-shape run end at q={q}: {end}")
        shape = g.output_shape(q)
        if g.output_shape(end - 1) != shape:
            raise AssertionError(f"same-shape run changed before exclusive end at q={q}")
        if end < g.outputs and g.output_shape(end) == shape:
            raise AssertionError(f"same-shape run ended before a matching tile at q={q}")
        if full:
            expected_end = q + 1
            while (expected_end < g.outputs and
                   g.output_shape(expected_end) == shape):
                expected_end += 1
            if end != expected_end:
                raise AssertionError(
                    f"same-shape run mismatch at q={q}: {end} != {expected_end}"
                )
        checked += 1

    row_probes = sorted({0, max(0, g.mt - 2), g.mt - 1})
    words = {row: g.whole_row_classword(row) for row in row_probes}
    for row, word in words.items():
        if not word or sum(segment[1] for segment in word) != g.nt:
            raise AssertionError(f"row {row} classword is incomplete")
    if g.mt > 1:
        if g.M % g.tm:
            full_row = g.mt - 2
            if (g.whole_row_classword(0) != g.whole_row_classword(full_row) or
                    g.whole_row_classword(g.mt - 1) ==
                    g.whole_row_classword(full_row)):
                raise AssertionError("full-row/tail-M classword boundary is wrong")
        elif g.whole_row_classword(0) != g.whole_row_classword(g.mt - 1):
            raise AssertionError("uniform M rows produced different classwords")
    if g.N % g.tn and g.nt > 1:
        if len(words[0]) < 2:
            raise AssertionError("whole-row word omitted the N-tail segment")
        if words[0][-1][0] + words[0][-1][1] != g.nt:
            raise AssertionError("N-tail segment did not end at the row boundary")
    return {
        "same_shape_queries": checked,
        "same_shape_mode": "all outputs" if full else "O(1) boundary probes",
        "whole_row_probed_rows": row_probes,
        "whole_row_segment_counts": {str(row): len(word) for row, word in words.items()},
    }


def check_class_facts(g: solo.FactsGeometry) -> dict:
    classes = g.classes
    if not 0 < len(classes) <= 32:
        raise AssertionError("class cache exceeded the fixed <=32 bound")
    if len(g.capacities) != 13 or g.outputs * g.kt != g.waves:
        raise AssertionError("geometry/capacity facts are incomplete")
    if any(hasattr(g, name) for name in ("tiles", "cores", "tile_array", "core_array")):
        raise AssertionError("facts geometry retained a tile/core array")
    first = any(bool(c["first_k"]) for c in classes)
    final = any(bool(c["final_k"]) for c in classes)
    middle = any(not c["first_k"] and not c["final_k"] and
                 int(c["k_tile_begin"]) < g.kt - 1 for c in classes)
    if not (first and final and middle):
        raise AssertionError("fixture lacks distinct first/middle/final K facts")
    return {
        "class_count": len(classes),
        "class_bound": 32,
        "waves": str(g.waves),
        "outputs": str(g.outputs),
        "k_first_middle_final_present": True,
        "aggregate_source_ledger": "exact class repetition counts x per-class facts",
        "resource_rows": len(g.capacities),
    }


def check_int4_tail(g: solo.FactsGeometry) -> dict:
    if not g.path.startswith("int4"):
        return {"applicable": False}
    tails = [c for c in g.classes if bool(c["k_tail"])]
    if not tails or not any(int(c["actual_k"]) % 2 for c in tails):
        raise AssertionError("INT4 fixture lacks an odd-K tail class")
    observed = []
    for cls in tails:
        fact = cls["load_fact"]
        payload = fact.get("hbm_row_payload_bytes",
                           fact.get("hbm_bytes_by_row_direction"))
        if not any(int(v) for row in payload for v in row):
            raise AssertionError("odd-K INT4 tail class has no emitted HBM payload")
        observed.append({"class": cls["index"], "actual_k": cls["actual_k"],
                         "load_hbm_payload": payload})
    fractional = any(
        c["compute_fact"]["typed_ops"].get("vec_eqops_fraction") is not None
        for c in g.classes
    )
    return {
        "applicable": True,
        "odd_k_tail_classes": observed,
        "fractional_typed_ops_present": fractional,
        "payload_origin": "per-class C++ Runtime facts; no byte formula",
    }


def persist_model_report(kind: str, name: str, report: dict, facts_path: Path) -> None:
    payload = dict(report)
    payload["facts_json"] = str(facts_path.relative_to(ROOT))
    (HERE / f"{kind}_{name}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def queue_semantics_test(g: solo.FactsGeometry) -> dict:
    if g.kt < 2 or g.outputs < 3:
        raise AssertionError("queue transition test requires three K tiles/outputs")
    nonfinal_wave = 2 * g.kt
    final_wave = 3 * g.kt - 1
    if g.is_final(nonfinal_wave) or not g.is_final(final_wave):
        raise AssertionError("queue transition test selected wrong K classes")

    nonfinal = solo.Solo13(g, event_cap=256)
    nonfinal.o.active = {2: (g.kt - 1, F(0))}
    nonfinal.o.computed = [2 * g.kt - 1]
    nonfinal.o.loaded = [nonfinal_wave]
    nonfinal.o.next_load = nonfinal_wave + 1
    if nonfinal.occupancy()[1] != 2:
        raise AssertionError("nonfinal transition fixture is not output-full")
    nonfinal.settle()
    if nonfinal.o.active.get(1, (None,))[0] != nonfinal_wave:
        raise AssertionError("nonfinal Compute was blocked by full output queue")
    if nonfinal.nonfinal_compute_at_full_output != 1:
        raise AssertionError("nonfinal-at-full-output transition was not counted")

    final = solo.Solo13(g, event_cap=256)
    final.o.active = {2: (g.kt - 1, F(0))}
    final.o.computed = [2 * g.kt - 1]
    final.o.loaded = [final_wave]
    final.o.next_load = final_wave + 1
    if final.occupancy()[1] != 2:
        raise AssertionError("final transition fixture is not output-full")
    final.settle()
    if 1 in final.o.active or final.o.loaded[0] != final_wave:
        raise AssertionError("final Compute failed to reserve output capacity")
    return {
        "status": "passed",
        "nonfinal_compute_starts_at_output_occupancy": 2,
        "final_compute_remains_queued_at_output_occupancy": 2,
        "scope": "queue transition semantics with real class facts; observed state is injected and is not a full-schedule reachability claim",
    }


def run_case(name: str, *, expanded: bool, event_cap: int = 256) -> dict:
    facts_path = HERE / f"facts_{name}.json"
    data = json.loads(facts_path.read_text())
    g = solo.FactsGeometry(data)
    powered = solo.Solo13(g, event_cap=event_cap, accelerate=True).run()
    if powered["status"] != "passed":
        raise AssertionError(f"{name}: powered candidate failed closed: {powered['reason']}")
    if powered["expanded_events"] > event_cap:
        raise AssertionError(f"{name}: expanded-event cap exceeded")
    persist_model_report("powered", name, powered, facts_path)

    result = {
        "geometry_facts": check_class_facts(g),
        "shape_facts": check_shape_runs(g, full=g.outputs <= 8192),
        "int4_tail_facts": check_int4_tail(g),
        "powered": {
            "status": powered["status"],
            "expanded_events": powered["expanded_events"],
            "global_expanded_event_cap": powered["global_expanded_event_cap"],
            "skipped_waves": powered["skipped_waves"],
            "skipped_outputs": powered["skipped_outputs"],
            "descriptors": powered["descriptors"],
            "peak_active_stages": powered["peak_active_stages"],
            "peak_input_slots": powered["peak_input_slots"],
            "peak_output_slots": powered["peak_output_slots"],
            "nonfinal_compute_started_while_output_full":
                powered["nonfinal_compute_started_while_output_full"],
            "ledger": powered["ledger"],
        },
        "facts_source_status": data["live_state"]["status"],
        "facts_submission_budget": data["stage_submission_cap"],
        "source_hardware_fingerprint": data["synthetic_hardware"]["hardware_fingerprint"],
    }

    if expanded:
        no_skip = solo.Solo13(g, event_cap=4096, accelerate=False).run()
        if no_skip["status"] != "passed":
            raise AssertionError(f"{name}: no-skip expansion failed: {no_skip['reason']}")
        persist_model_report("expanded", name, no_skip, facts_path)
        solo.compare_ledgers(powered, no_skip)
        result["no_skip_expansion"] = {
            "status": "passed",
            "expanded_events": no_skip["expanded_events"],
            "exact_all_ledger_fields_match": True,
        }
        reference = exact_continuous_oracle(data)
        check_continuous(powered, reference)
        result["independent_continuous_oracle"] = {
            "status": "passed",
            "continuous_clock": True,
            "integer_ns_wake_rounding": False,
            "exact_service_busy_payload_typed_and_finish_match": True,
            "expanded_terminal_events": reference["events"],
        }
        if data["live_state"]["status"] == "complete":
            try:
                public = RUNTIME_ORACLE.check(data)
                result["public_runtime_integer_ns_oracle"] = public
            except AssertionError as error:
                result["public_runtime_integer_ns_oracle"] = {
                    "status": "deviation",
                    "detail": str(error),
                    "scope": "public integer-ns wake comparison; kept separate from exact continuous oracle",
                }
        else:
            result["public_runtime_integer_ns_oracle"] = {
                "status": "not_applicable",
                "detail": "C++ facts probe stopped at its stage-submission cap before a complete lifecycle",
            }

    return result


def run_output_pressure_case(name: str) -> dict:
    data = json.loads((HERE / f"facts_{name}.json").read_text())
    g = solo.FactsGeometry(data)
    powered = solo.Solo13(g, event_cap=256, accelerate=True).run()
    expanded = solo.Solo13(g, event_cap=4096, accelerate=False).run()
    if powered["status"] != "passed" or expanded["status"] != "passed":
        raise AssertionError(f"{name}: pressure fixture did not complete")
    solo.compare_ledgers(powered, expanded)
    facts_path = HERE / f"facts_{name}.json"
    persist_model_report("powered", name, powered, facts_path)
    persist_model_report("expanded", name, expanded, facts_path)
    reference = exact_continuous_oracle(data)
    check_continuous(powered, reference)

    life = data["lifetime_counters"]
    if powered["peak_output_slots"] != life["max_output_queue_occupancy"]:
        raise AssertionError(f"{name}: model/runtime output slot peaks disagree")
    if powered["peak_input_slots"] != life["max_input_queue_occupancy"]:
        raise AssertionError(f"{name}: model/runtime input slot peaks disagree")
    if life["max_runtime_active_activities"] > 3:
        raise AssertionError(f"{name}: more than three stage activities were active")

    compute_starts = [
        event for event in data["events"]
        if event["stage"] == "compute" and event["phase"] == "start"
    ]
    k_tiles = int(g.kt)
    for event in compute_starts:
        final = int(event["k_tile"]) + 1 == k_tiles
        reserved = int(event["extra"]["reserved_output_slot"]) >= 0
        if final != reserved:
            raise AssertionError(f"{name}: actual final-K reservation facts disagree")
        if int(event["extra"]["output_queue_occupancy"]) > 2:
            raise AssertionError(f"{name}: Runtime start exceeded output queue capacity")

    try:
        public = RUNTIME_ORACLE.check(data)
        public_result = {"status": "passed", **public}
    except AssertionError as error:
        public_result = {"status": "deviation", "detail": str(error)}

    return {
        "geometry": {
            key: data["geometry"][key]
            for key in ("M", "N", "K", "tm", "tn", "tk", "m_tiles",
                        "n_tiles", "k_tiles", "output_tile_count",
                        "consumption_wave_count", "path")
        },
        "facts_class_count": len(g.classes),
        "facts_class_bound": 32,
        "facts_aggregate_source_ledger": "passed",
        "producer_status": data["live_state"]["status"],
        "producer_stages": life["submitted_stages"],
        "runtime_slot_peaks": {
            "input": life["max_input_queue_occupancy"],
            "output": life["max_output_queue_occupancy"],
            "active_activities": life["max_runtime_active_activities"],
        },
        "solo_slot_peaks": {
            "input": powered["peak_input_slots"],
            "output": powered["peak_output_slots"],
        },
        "actual_compute_starts": [
            {
                "ordinal": event["ordinal"],
                "output_q": event["output_q"],
                "k_tile": event["k_tile"],
                "time_ns": event["time_ns"],
                "post_start_output_occupancy":
                    event["extra"]["output_queue_occupancy"],
                "reserved_output_slot": event["extra"]["reserved_output_slot"],
            }
            for event in compute_starts
        ],
        "powered_expanded_events": powered["expanded_events"],
        "expanded_vs_powered_all_ledgers": "exact match",
        "independent_continuous_oracle": "exact match for all 13 service/busy rows, HBM payload, typed work, and finish",
        "public_runtime_integer_ns_oracle": public_result,
        "final_compute_reservations_match_k_final_facts": True,
        "actual_queue_pressure": (
            "two live output slots observed"
            if life["max_output_queue_occupancy"] == 2
            else "two-slot output capacity not reached"
        ),
        "final_ready_blocked_witness": False,
        "nonfinal_compute_at_full_output_witness": False,
        "witness_boundary": "driver emits starts/terminals, not rejected readiness attempts; coincident input/store terminals do not prove that output fullness alone blocked a final Compute",
    }


def source_identity() -> dict:
    source_files = [
        "die/tests/os_lifecycle_runtime_driver.cc",
        "die/src/GemmOSResourceFacts.hh",
        "die/src/GemmOSAccessCounter.hh",
        "die/src/GemmOSGeometry.hh",
        "die/src/GemmTailBudget.hh",
        "die/src/EquivalentWork.hh",
        "die/src/Mapping.cc",
        "die/src/Execution.cc",
        "die/src/Hardware.cc",
        "die/include/coarse_die/Execution.hh",
        "die/include/coarse_die/Hardware.hh",
        "die/include/coarse_die/Plan.hh",
        "die/CMakeLists.txt",
        "work/build-die/CMakeCache.txt",
    ]
    hashes = {}
    for relative in source_files:
        path = ROOT / relative
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    binary = HERE / "os_lifecycle_runtime_driver"
    archive = ROOT / "work/build-die/libcoarse_die.a"
    run = lambda *args: subprocess.check_output(args, cwd=ROOT, text=True).strip()
    compiler = run("g++", "--version").splitlines()[0]
    head = run("git", "rev-parse", "HEAD")
    branch = run("git", "branch", "--show-current")
    status = run("git", "status", "--short", "die")
    facts = {}
    for path in sorted(HERE.glob("facts_*.json")):
        raw = path.read_bytes()
        data = json.loads(raw)
        facts[path.name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "geometry": data["geometry"],
            "hardware_fingerprint": data["synthetic_hardware"]["hardware_fingerprint"],
            "fixture_identity": data.get("fixture_identity"),
            "producer_lifecycle_status": data["live_state"]["status"],
        }
    return {
        "identity_semantics": "hashes name the actual source files, library archive, and executable used for these facts; commit is contextual only",
        "git_branch_context": branch,
        "git_head_context": head,
        "git_status_for_die": status,
        "compiler": compiler,
        "compile_flags": ["-std=c++17", "-Wall", "-Wextra", "-Werror", "-O2"],
        "compile_command_shape": [
            "g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-O2",
            "-Idie/include", "-Iastra-sim-wscllm/extern/helper",
            "-DCOARSE_DIE_SOURCE_DIR=\"<repo>/die\"",
            "die/tests/os_lifecycle_runtime_driver.cc",
            "work/build-die/libcoarse_die.a",
        ],
        "driver_source_sha256": hashes["die/tests/os_lifecycle_runtime_driver.cc"],
        "source_sha256": hashes,
        "research_code_sha256": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "work/os-single-owner-13-20261014/solo_facts_13.py",
                "work/os-single-owner-13-20261014/current/study.py",
                "work/os-single-owner-13-20261014/current/run_probes.sh",
            )
        },
        "library_archive": {
            "path": "work/build-die/libcoarse_die.a",
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        },
        "driver_binary": {
            "path": str(binary.relative_to(ROOT)),
            "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        },
        "facts_probes": facts,
    }


def main() -> int:
    identity = source_identity()
    (HERE / "producer_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n"
    )

    case_names = [
        "strict_small", "int4_native_small", "int4_converted_fractional",
        "int4_converted_storage", "dense", "large_k", "large_outputs",
    ]
    cases = {}
    for name in case_names:
        cases[name] = run_case(
            name,
            expanded=name in {
                "strict_small", "int4_native_small",
                "int4_converted_fractional", "int4_converted_storage", "dense",
            },
        )

    pressure_names = [
        "output_pressure_requested", "output_pressure_k2_tiles128",
        "output_pressure_k1_tiles128",
    ]
    output_pressure = {
        name: run_output_pressure_case(name) for name in pressure_names
    }

    transition_data = json.loads((HERE / "facts_strict_small.json").read_text())
    transition_geometry = solo.FactsGeometry(transition_data)
    queue_test = queue_semantics_test(transition_geometry)

    cap_test = solo.Solo13(transition_geometry, event_cap=1,
                           accelerate=False).run()
    if (cap_test["status"] != "failed_closed" or
            "global expanded-event cap reached" not in cap_test["reason"]):
        raise AssertionError("global expanded-event cap did not fail closed")

    for name in ("large_k", "large_outputs"):
        powered = cases[name]["powered"]
        if powered["expanded_events"] > 256:
            raise AssertionError(f"{name}: exceeded global 256 exact-event cap")
        if not powered["descriptors"]:
            raise AssertionError(f"{name}: large shape completed without skip descriptors")
    row_periods = [
        descriptor["category_word"]
        for descriptor in cases["large_outputs"]["powered"]["descriptors"]
        if isinstance(descriptor["category_word"], dict) and
        descriptor["category_word"].get("kind") == "whole-row-classword"
    ]
    if not row_periods or row_periods[0]["segments"] < 2:
        raise AssertionError("large output skip did not include full and N-tail row segments")

    result = {
        "schema_version": 1,
        "status": "passed",
        "identity_file": "producer_identity.json",
        "scope": "offline exact one-owner continuous 13-domain bounded study; no production/descriptor/silicon certificate",
        "global_expanded_event_cap": 256,
        "queue_transition_test": queue_test,
        "actual_output_pressure_fixtures": output_pressure,
        "cap_test": {
            "status": cap_test["status"],
            "reason": cap_test["reason"],
            "no_per_tile_fallback": True,
        },
        "large_output_row_word": row_periods[0],
        "cases": cases,
        "certificate_boundary": [
            "Compiler facts are actual public Runtime driver outputs, cached by <=32 access classes; the Python geometry retains no tile/core arrays.",
            "All facts use this run's SHA-identified source, library archive, and driver; git HEAD is recorded only as context.",
            "Exact skip requires identical live progress, stage and queue translation, class facts, and an equal facts-derived K/run/whole-row classword; skips stop at shape/tail boundaries.",
            "The event cap is global across the complete Solo run, never reset per output; cap exhaustion fails closed with no per-wave fallback.",
            "The independent continuous oracle uses the same compiler-supplied class facts and independently expands queues/allocation without integer-ns wake rounding.",
            "Public Runtime integer-ns checks are separate; the converted int4-storage fixture records a 3 ns finish discrepancy against a 2 ns check tolerance.",
            "Large facts probes stopped at the driver's stage_submission_cap and supply class facts only; they are not full public-runtime schedule comparisons.",
            "The nonfinal-at-full-output case is a manually injected queue transition check. Complete multi-K Runtime fixtures observed output occupancy peak 1; the separate K=1 pressure fixture reached 2 live output slots.",
            "No fixture produced an event-level witness that an already-loaded final Compute was blocked solely by a full output queue or that nonfinal Compute started at that state; coincident input/store terminals leave that cause unresolved.",
            "Solo one-owner fixed-capacity research only; this does not certify production scheduling, multi-owner contention, external cancellation, or hardware performance.",
        ],
    }
    output = HERE / "study_validation.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"PASS: 13-domain facts study; summary={output}")
    for name, case in cases.items():
        p = case["powered"]
        print(f"{name}: status={p['status']} events={p['expanded_events']} "
              f"skip_waves={p['skipped_waves']} skip_outputs={p['skipped_outputs']}")
        public = case.get("public_runtime_integer_ns_oracle")
        if public:
            print(f"  public-ns={public.get('status', 'passed')} "
                  f"detail={public.get('detail', 'within measured tolerance')}")
    print(f"queue transitions: {queue_test['status']}; cap: {cap_test['status']}")
    for name, pressure in output_pressure.items():
        print(f"{name}: runtime output peak={pressure['runtime_slot_peaks']['output']} "
              f"public-ns={pressure['public_runtime_integer_ns_oracle']['status']} "
              f"final-ready-block-witness={pressure['final_ready_blocked_witness']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
