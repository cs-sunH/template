#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
build_dir="$script_dir/build-die"
driver="$script_dir/os_lifecycle_runtime_driver"
log="$script_dir/os_lifecycle_runtime_validation.log"

exec > >(tee "$log") 2>&1
cd "$repo_dir"

if [[ -f /workspace/cloud-setup/activate.sh ]]; then
  # shellcheck disable=SC1091
  source /workspace/cloud-setup/activate.sh
fi

echo "[build] cmake --build $build_dir --target coarse_die -j2"
cmake --build "$build_dir" --target coarse_die -j2

echo "[compile] standalone public-Runtime readiness driver"
"${CXX:-g++}" -std=c++17 -Wall -Wextra -Werror -O2 \
  -I"$repo_dir/die/include" \
  -I"$repo_dir/astra-sim-wscllm/extern/helper" \
  -DCOARSE_DIE_SOURCE_DIR="\"$repo_dir/die\"" \
  "$repo_dir/die/tests/os_lifecycle_runtime_driver.cc" \
  "$build_dir/libcoarse_die.a" -o "$driver"

echo "[run] default strict FP32"
"$driver" > "$script_dir/os_lifecycle_runtime_default.json"
echo "[run] native INT4"
"$driver" --path int4 --a-dtype int4 --w-dtype int4 --output-dtype int4 \
  --diagnostic-cap 1024 > "$script_dir/os_lifecycle_runtime_int4_native.json"
echo "[run] bf16 input conversion to INT4"
"$driver" --path int4 --a-dtype bf16 --w-dtype int4 --output-dtype int4 \
  --kappa-quant 0.3125 --kappa-acc 0.375 --diagnostic-cap 1024 \
  > "$script_dir/os_lifecycle_runtime_int4_converted.json"
echo "[run] fixture-wide stage submission cap"
"$driver" --cap 5 > "$script_dir/os_lifecycle_runtime_capped.json"

echo "[verify] reports and Runtime lifecycle counters"
python3 - "$script_dir" <<'PY'
import json
import sys
from decimal import Decimal
from pathlib import Path

root = Path(sys.argv[1])
reports = {
    name: json.loads((root / f"os_lifecycle_runtime_{name}.json").read_text())
    for name in ("default", "int4_native", "int4_converted", "capped")
}

def require(condition, message):
    if not condition:
        raise SystemExit(f"FAIL: {message}")

for name in ("default", "int4_native", "int4_converted"):
    report = reports[name]
    life = report["lifetime_counters"]
    live = report["live_state"]
    require(live["status"] == "complete", f"{name}: expected complete status")
    require(live["runtime_drained"], f"{name}: Runtime not drained")
    require(live["public_runtime_live_executions"] == 0,
            f"{name}: live executions remain")
    require(life["submitted_stages"] == life["destroyed_executions"],
            f"{name}: not every submitted stage was destroyed")
    require(len(report["events"]) == 2 * life["submitted_stages"],
            f"{name}: stage trace storage is not bounded to start/terminal records")
    owner = report["runtime_owner"]
    require(owner == {"request_id": 1, "node_id": 901,
                     "execution_id": 901, "generation": 1},
            f"{name}: fixture owner identity changed between stages")
    require(life["max_input_queue_occupancy"] <= 2 and
            life["max_output_queue_occupancy"] <= 2,
            f"{name}: a fixed queue exceeded capacity")
    require(report["stage_submission_cap"]["per_output_reset"] is False and
            report["stage_submission_cap"]["production_event_cap_certificate"] is False and
            report["production_bounded_descriptor_claim"] is False,
            f"{name}: offline budget was presented as a production event bound")
    require(report["runtime_internal_events"] ==
            report["runtime_counters"]["internal_events"],
            f"{name}: Runtime internal-event count was not reported verbatim")
    require(life["max_runtime_active_activities"] <= 3,
            f"{name}: more than three Runtime activities were active")

    active_by_stage = {"load": 0, "compute": 0, "store": 0}
    for event in report["events"]:
        stage = event["stage"]
        if event["phase"] == "start":
            require(active_by_stage[stage] == 0,
                    f"{name}: overlapping {stage} activity starts")
            active_by_stage[stage] += 1
        elif event["phase"] == "terminal":
            require(active_by_stage[stage] == 1,
                    f"{name}: {stage} terminal has no single active stage")
            active_by_stage[stage] -= 1
    require(all(count == 0 for count in active_by_stage.values()),
            f"{name}: stage activity timeline did not drain")

default = reports["default"]
life = default["lifetime_counters"]
require(life["submitted_by_stage"] == {"load": 12, "compute": 12, "store": 4},
        "default: unexpected stage submission counts")
require(life["outputs_reserved_at_final_compute_start"] == "4" and
        life["outputs_produced_at_final_compute_end"] == "4" and
        life["outputs_retired_after_store"] == "4",
        "default: final-K output lifecycle counts disagree")
require(life["max_input_queue_occupancy"] == 2,
        "default: did not exercise the two-slot input queue")

for name in ("default", "int4_native", "int4_converted"):
    report = reports[name]
    source = report["source_counters_full_single_gemm"]
    runtime = report["runtime_counters"]
    served = runtime["served_by_category"]["comp"]
    k_tiles = report["geometry"]["k_tiles"]
    require(len(report["geometry"]["classes"]) <= 32,
            f"{name}: class cache exceeded its fixed bound")
    require(report["geometry"]["cached_resource_fact_count"] ==
            len(report["geometry"]["classes"]),
            f"{name}: resource facts are not cached by access class")
    require("kappa_quant" in report["resource_facts_input"] and
            "kappa_acc" in report["resource_facts_input"],
            f"{name}: resource-facts input omitted kappa")
    require(len(report["synthetic_hardware"]["hardware_capacities_13"]) == 13,
            f"{name}: Runtime hardware capacity snapshot is incomplete")
    for cls in report["geometry"]["classes"]:
        require(len(cls["hardware_capacities_13"]) == 13 and
                len(cls["resource_names"]) == 13,
                f"{name}: class hardware capacity snapshot is incomplete")
        for stage_key in ("load_fact", "compute_fact", "store_fact"):
            fact = cls[stage_key]
            require(len(fact["resource_work_13"]) == 13 and
                    len(fact["hbm_row_payload_bytes"]) == 2,
                    f"{name}: {stage_key} omitted resource/HBM payload facts")
            require("canonical_q" in fact and "private_time_s" in fact and
                    "typed_ops" in fact and
                    "vec_eqops_fraction" in fact["typed_ops"],
                    f"{name}: {stage_key} omitted typed activity facts")
    require(abs(Decimal(source["matrix_int4_ops"]) -
                Decimal(served["int4_ops_exact_decimal"])) < Decimal("0.000001"),
            f"{name}: source/runtime INT4 operations differ")
    require(abs(Decimal(source["vec_equivalent_ops_whole"]) +
                Decimal.from_float(float.fromhex(
                    source["vec_equivalent_ops_fraction_hex"])) -
                Decimal(served["vec_eqops_exact_decimal"])) < Decimal("0.000001"),
            f"{name}: source/runtime vector-equivalent operations differ")
    hbm = runtime["hbm_served_bytes_by_row_direction_exact"]
    for direction, ledger_key in ((0, "hbm_read_bytes"), (1, "hbm_write_bytes")):
        runtime_total = sum(Decimal(hbm[row][direction]) for row in range(2))
        require(abs(runtime_total - Decimal(source[ledger_key])) < Decimal("0.000001"),
                f"{name}: source/runtime {ledger_key} differ")

    load_terminal = set()
    compute_started = set()
    compute_terminal = set()
    output_ready = set()
    store_started = set()
    store_terminal = set()
    for event in report["events"]:
        phase, stage = event["phase"], event["stage"]
        ordinal, q, k_tile = event["ordinal"], event["output_q"], event["k_tile"]
        if stage == "load" and phase == "terminal":
            load_terminal.add(ordinal)
        elif stage == "compute" and phase == "start":
            require(ordinal in load_terminal,
                    f"{name}: Compute began before its Load terminal")
            compute_started.add(ordinal)
            final_k = k_tile + 1 == k_tiles
            reserved = event["extra"]["reserved_output_slot"]
            require((reserved >= 0) == final_k,
                    f"{name}: output reservation did not match final-K Compute start")
        elif stage == "compute" and phase == "terminal":
            require(ordinal in compute_started,
                    f"{name}: Compute terminal has no start")
            compute_terminal.add(ordinal)
            if k_tile + 1 == k_tiles:
                output_ready.add(q)
        elif stage == "store" and phase == "start":
            require(k_tile + 1 == k_tiles,
                    f"{name}: Store was attributed to a non-final K tile")
            require(q in output_ready,
                    f"{name}: Store began before final Compute terminal")
            store_started.add(q)
        elif stage == "store" and phase == "terminal":
            require(q in store_started,
                    f"{name}: Store terminal has no start")
            store_terminal.add(q)
    if report["live_state"]["status"] == "complete":
        require(len(compute_terminal) == int(report["geometry"]["total_k_consumptions"]),
                f"{name}: completed report is missing Compute terminals")
        require(len(store_terminal) == int(report["geometry"]["output_tile_count"]),
                f"{name}: completed report is missing Store terminals")

capped = reports["capped"]
life = capped["lifetime_counters"]
live = capped["live_state"]
require(live["status"] == "stage_submission_cap_reached" and
        life["submitted_stages"] == 5,
        "cap: did not stop at the fixture-wide five-stage submission limit")
require(live["runtime_drained"] and
        live["public_runtime_live_executions"] == 0,
        "cap: a submitted stage remained live after terminal collection")
require(any(slot["state"] == "ready" for slot in live["input_slots"]),
        "cap: expected queued readiness state was not reported")

print("PASS: default, native INT4, converted INT4, and cap lifecycle reports")
for name, report in reports.items():
    life = report["lifetime_counters"]
    print(f"{name}: status={report['live_state']['status']} "
          f"submitted={life['submitted_stages']} destroyed={life['destroyed_executions']} "
          f"runtime_advances={life['runtime_advances']} "
          f"max_queues={life['max_input_queue_occupancy']}/"
          f"{life['max_output_queue_occupancy']}")
PY
