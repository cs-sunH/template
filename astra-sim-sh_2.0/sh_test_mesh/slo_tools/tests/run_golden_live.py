#!/usr/bin/env python3
"""B3-5 T2 golden live runner: hand-crafted queues through the real simulator.

Usage (from anywhere; repo = the one containing this slo_tools copy):
  python3 run_golden_live.py g1
  python3 run_golden_live.py g2
  python3 run_golden_live.py g3
  python3 run_golden_live.py g2r <full_detail_run_dir>  # queue semantics on
                                                        # real 60s products
  python3 run_golden_live.py g4 <full_detail_run_dir>   # restore cross-check
  python3 run_golden_live.py all <full_detail_run_dir>

G1/G2/G3 write a hand-crafted queue + canonical sidecar into traces/, point
trace_config.csv at them, run plan_materializer + run_online_strategy.sh
(SH_METRICS_DETAIL=full, SH_FIRST_TOKEN_SPLIT=1), and assert hand-derived
expectations against request_metrics.csv and slo_tools outputs.  G4 runs
restore_decomposition on an existing full-detail run and independently
recomputes the three-segment decomposition from the raw [METRIC] records
(anchor min/max + request prefill window), diffing field by field.

Queue/sidecar column formats auto-adapt to the repo (sh_1.0 8-column queue +
15-column canonical_sidecar.csv; sh_2.0 9-column queue with the
next_trigger_type column + 10-column canonical_digest.csv).

The original trace_config.csv pointer is saved and restored; golden queue
products stay in traces/ (clean-script enumerated root).  Exit 0 = all
assertions passed.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SLO_TOOLS_DIR = TESTS_DIR.parent
SH_TEST_MESH = SLO_TOOLS_DIR.parent
REPO = SH_TEST_MESH.parent
WL = SH_TEST_MESH / "workload" / "llama2_7b_inference"
RUNNER = SH_TEST_MESH / "run_scripts" / "run_online_strategy.sh"
TRACES = WL / "traces"
GENERATED = SH_TEST_MESH / "generated"
WORKROOT = Path("/tmp/slo_wps/b3")

REPO_TAG = "S2" if REPO.name.endswith("sh_2.0") else "S1"

S1_QUEUE_HEADER = [
    "session_id", "turn_index", "request_id", "prefill_length",
    "decode_length", "session_arrival_time_ns", "inter_request_interval_ns",
    "description",
]
S1_SIDECAR_HEADER = [
    "request_id", "turn_index", "session_id", "raw_prefix_tokens",
    "raw_new_prefill_tokens", "effective_prompt_tokens", "prefill_length",
    "decode_length", "session_arrival_time_ns", "inter_request_interval_ns",
    "prefix_mode", "digest", "human_time_ns", "tool_time_ns", "request_type",
]
S2_QUEUE_HEADER = [
    "session_id", "turn_index", "request_id", "prefill_length",
    "decode_length", "session_arrival_time_ns", "inter_request_interval_ns",
    "next_trigger_type", "description",
]
S2_SIDECAR_HEADER = [
    "request_id", "raw_prefix_tokens", "raw_new_prefill_tokens",
    "effective_prompt_tokens", "digest", "prefix_mode", "human_time_ns",
    "tool_time_ns", "request_type", "decode_length",
]

FAILURES: list[str] = []
NOTES: list[str] = []


def check(cond, message):
    if cond:
        print(f"  PASS: {message}")
    else:
        print(f"  FAIL: {message}")
        FAILURES.append(message)


def note(message):
    NOTES.append(message)
    print(f"  note: {message}")


def read_pointer() -> str:
    for line in (WL / "trace_config.csv").read_text().splitlines():
        if line.startswith("config,request_queue_csv,"):
            return line.split(",")[2]
    raise SystemExit("trace_config.csv has no request_queue_csv row")


def set_pointer(rel: str) -> None:
    subprocess.run(
        [sys.executable, "/tmp/slo_wps/set_trace_pointer.py",
         str(WL / "trace_config.csv"), rel],
        check=True, stdout=subprocess.DEVNULL)


def write_golden_inputs(name: str, rows: list[dict]) -> Path:
    """rows: dicts with keys session_id, turn_index, request_id, prefill,
    decode, arrival (turn 0), interval (turn>0), trigger, human, tool,
    rtype.  Returns the queue path; sidecar written next to it."""
    queue = TRACES / f"golden_{name}_request_queue{'_recompute' if REPO_TAG == 'S1' else ''}.csv"
    sidecar_name = queue.name.replace(
        "_request_queue_recompute.csv", "_canonical_sidecar.csv").replace(
        "_request_queue.csv", "_canonical_digest.csv")
    sidecar = TRACES / sidecar_name
    qrows, srows = [], []
    for row in rows:
        qrows.append({
            "session_id": row["session_id"],
            "turn_index": row["turn_index"],
            "request_id": row["request_id"],
            "prefill_length": row["prefill"],
            "decode_length": row["decode"],
            "session_arrival_time_ns":
                row["arrival"] if row["turn_index"] == 0 else "",
            "inter_request_interval_ns":
                row["interval"] if row["turn_index"] > 0 else "",
            **({"next_trigger_type": row["trigger"]} if REPO_TAG == "S2"
               else {}),
            "description": (
                "golden live case (SLO B3-5); turn-0 prefix folded into "
                "prefill_length (recompute caliber)"),
        })
        srow_common = {
            "request_id": row["request_id"],
            "raw_prefix_tokens": 0,
            "raw_new_prefill_tokens": row["prefill"],
            "effective_prompt_tokens": row["prefill"],
            "digest": "golden",
            "prefix_mode": "recompute",
            "human_time_ns": row.get("human", ""),
            "tool_time_ns": row.get("tool", ""),
            "request_type": row["rtype"],
        }
        if REPO_TAG == "S1":
            srows.append({
                **srow_common,
                "turn_index": row["turn_index"],
                "session_id": row["session_id"],
                "prefill_length": row["prefill"],
                "decode_length": row["decode"],
                "session_arrival_time_ns":
                    row["arrival"] if row["turn_index"] == 0 else "",
                "inter_request_interval_ns":
                    row["interval"] if row["turn_index"] > 0 else "",
            })
        else:
            srows.append({**srow_common, "decode_length": row["decode"]})
    header = S2_QUEUE_HEADER if REPO_TAG == "S2" else S1_QUEUE_HEADER
    with queue.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(qrows)
    sheader = (S2_SIDECAR_HEADER if REPO_TAG == "S2"
               else S1_SIDECAR_HEADER)
    with sidecar.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sheader)
        writer.writeheader()
        writer.writerows(srows)
    return queue


def run_simulation(name: str, queue: Path) -> Path:
    run_dir = WORKROOT / REPO.name / "golden" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    for entry in GENERATED.glob("llama2_7b_inference_54npus_*"):
        subprocess.run(["rm", "-rf", str(entry)], check=True)
    proc = subprocess.run(
        [sys.executable, "plan_materializer.py"], cwd=WL,
        capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr)
        raise SystemExit("plan_materializer failed")
    env = dict(os.environ)
    env["SH_METRICS_DETAIL"] = "full"
    env["SH_FIRST_TOKEN_SPLIT"] = "1"
    time_log = run_dir.parent / f"{name}.time.log"
    with time_log.open("w") as tlog:
        proc = subprocess.run(
            ["/usr/bin/time", "-v", "bash", str(RUNNER), str(run_dir),
             str(queue)],
            stdout=subprocess.DEVNULL, stderr=tlog, env=env, cwd=REPO)
    if proc.returncode != 0:
        tail = subprocess.run(["tail", "-5", str(run_dir / "cpp.log")],
                              capture_output=True, text=True)
        print(tail.stdout)
        raise SystemExit(f"runner failed rc={proc.returncode}")
    return run_dir


def load_request_metrics(run_dir: Path) -> list[dict]:
    with (run_dir / "request_metrics.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def metric_records(run_dir: Path, kind: str) -> list[dict]:
    pattern = re.compile(r"\[METRIC\]\s+(\{.*\})")
    out = []
    for line in (run_dir / "cpp.log").read_text().splitlines():
        match = pattern.search(line)
        if not match:
            continue
        record = json.loads(match.group(1))
        if record.get("type") == kind:
            out.append(record)
    return out


def as_int(value) -> int:
    return None if value in ("", "NA", None) else int(value)


def request_instance_by_id(run_dir: Path) -> dict[str, int]:
    """Per-request prefill instance from the decision log.

    The [METRIC] request records carry the synthetic manifest's placeholder
    prefill_instance (always 0 in online mode); the scheduler's actual
    selection is only in online_decision_log.jsonl prefill decisions."""
    out: dict[str, int] = {}
    path = run_dir / "results" / "online_decision_log.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("kind") == "prefill":
                out[record["request_id"]] = (
                    record.get("decision") or {}).get(
                        "prefill_instance_index")
    return out


# --------------------------------------------------------------------- G1 --

def scenario_g1() -> None:
    print("[G1] single request, no contention")
    rows = [{
        "session_id": "session_g1s", "turn_index": 0,
        "request_id": "session_g1s_request_0", "prefill": 512, "decode": 8,
        "arrival": 1_000_000, "interval": "", "trigger": "human",
        "rtype": "human",
    }]
    queue = write_golden_inputs("g1", rows)
    pointer = read_pointer()
    try:
        set_pointer(f"llama2_7b_inference/traces/{queue.name}")
        run_dir = run_simulation("g1", queue)
    finally:
        set_pointer(pointer)
    metrics = load_request_metrics(run_dir)
    check(len(metrics) == 1, "G1 emits exactly 1 request row")
    row = metrics[0]
    check(row["terminal_status"] == "completed", "G1 terminal completed")
    arrival = as_int(row["arrival_ns"])
    ps, pe = as_int(row["prefill_start_ns"]), as_int(row["prefill_end_ns"])
    ds, comp = as_int(row["decode_start_ns"]), as_int(row["completion_ns"])
    queue_ns, prefill_ns = as_int(row["queue_ns"]), as_int(row["prefill_ns"])
    gap, decode_ns = (as_int(row["prefill_decode_gap_ns"]),
                      as_int(row["decode_ns"]))
    e2e = as_int(row["e2e_ns"])
    check(queue_ns == 0, "G1 queue_ns == 0 (idle fleet)")
    check(ps == arrival, f"G1 prefill_start == arrival ({ps} == {arrival})")
    check(e2e == queue_ns + prefill_ns + gap + decode_ns,
          "G1 E2E == queue + prefill + gap + decode (decomposition sum)")
    check(e2e == comp - arrival, "G1 E2E == completion - arrival")
    ft, src = row["first_token_ns"], row["first_token_source"]
    check(ft not in ("", "NA") and src == "exact",
          f"G1 first_token present, source=exact (ft={ft})")
    ft = as_int(ft)
    check(pe <= ft <= comp, "G1 arrival-chain: prefill_end <= ft <= completion")
    check(ds <= ft, "G1 ft >= decode_start")
    first_step = ft - ds
    dl = as_int(row["decode_length"])
    note(f"first_token-arrival={ft - arrival} ns = queue+prefill+gap"
         f"{queue_ns + prefill_ns + gap} + first_decode_step{first_step};"
         f" mean step={decode_ns / dl:.0f} ns over dl={dl}")


# --------------------------------------------------------------------- G2 --

def scenario_g2() -> None:
    print("[G2] contention queueing (30-request burst saturating the fleet)")
    # Design note (B3): the fleet runs rolling 8-iteration trains; a single
    # latecomer folds into the next train without waiting (measured: 9
    # occupiers + 1 latecomer at +59ms -> queue_ns=0).  Real queueing needs a
    # saturated fleet (the 60s window shows 1102/1454 queued at ~24 req/s),
    # so G2 uses a 30-request simultaneous burst: ~9 start immediately (one
    # per instance, queue_ns=0) and the rest queue behind the instance qp
    # backlogs; per-instance service stays FCFS in queue order.
    rows = []
    for index in range(30):
        rows.append({
            "session_id": f"session_g2s{index:02d}", "turn_index": 0,
            "request_id": f"session_g2s{index:02d}_request_0",
            "prefill": 8000, "decode": 60, "arrival": 1_000_000,
            "interval": "", "trigger": "human", "rtype": "human",
        })
    queue = write_golden_inputs("g2", rows)
    pointer = read_pointer()
    try:
        set_pointer(f"llama2_7b_inference/traces/{queue.name}")
        run_dir = run_simulation("g2", queue)
    finally:
        set_pointer(pointer)
    metrics = load_request_metrics(run_dir)
    check(len(metrics) == 30, "G2 emits 30 request rows")
    check(all(r["terminal_status"] == "completed" for r in metrics),
          "G2 all completed")
    # In this fleet's rolling-train design every qp member's first prefill
    # chunk enters the first train round, so a light burst shows queue_ns==0
    # and contention manifests as prefill STRETCHING (measured on the real
    # 60s window: 1102/1454 queued, i.e. queue_ns>0 needs deep bursts; the
    # g2r step below cross-checks the queue semantics on the real products).
    early = [as_int(r["prefill_ns"]) for r in metrics[:9]]
    late = [as_int(r["prefill_ns"]) for r in metrics[9:]]
    check(min(late) > max(early),
          f"G2 contention stretches shared-instance prefill "
          f"(members 9-29 min {min(late)} > members 0-8 max {max(early)})")
    check(all(as_int(r["queue_ns"]) == 0 for r in metrics),
          "G2 light-burst regime: queue_ns == 0 (no admission deferral)")
    ok_identity = all(
        as_int(r["prefill_start_ns"])
        == as_int(r["arrival_ns"]) + as_int(r["queue_ns"])
        for r in metrics)
    check(ok_identity, "G2 prefill_start == arrival + queue_ns (all rows)")
    # per-instance grouping from the decision log (real scheduler choice).
    inst_by_id = request_instance_by_id(run_dir)
    per_instance: dict[int, list[dict]] = {}
    for row in metrics:
        per_instance.setdefault(inst_by_id.get(row["request_id"]),
                                []).append(row)
    check(len(per_instance) >= 9,
          f"G2 burst spread across the fleet (instances={len(per_instance)})")
    fcfs_ok = True
    for instance, members in per_instance.items():
        members.sort(key=lambda r: as_int(r["prefill_end_ns"]))
        order = [int(r["queue_index"]) for r in members]
        if order != sorted(order):
            fcfs_ok = False
    check(fcfs_ok,
          "G2 per-instance prefill completion order is FCFS in queue_index")
    for row in metrics:
        total = (as_int(row["queue_ns"]) + as_int(row["prefill_ns"])
                 + as_int(row["prefill_decode_gap_ns"])
                 + as_int(row["decode_ns"]))
        check(as_int(row["e2e_ns"]) == total,
              f"G2 decomposition sum identity {row['request_id']}")
    note(f"instances={len(per_instance)}; prefill_ns early max={max(early)} "
         f"late min={min(late)} (contention stretch); queue_ns all 0 -- "
         f"queue_ns>0 regime cross-checked by g2r on the 60s products")


# --------------------------------------------------------------------- G3 --

def scenario_g3() -> None:
    print("[G3] one session, two turns (human-triggered return)")
    rows = [
        {"session_id": "session_g3s", "turn_index": 0,
         "request_id": "session_g3s_request_0", "prefill": 4000,
         "decode": 50, "arrival": 1_000_000, "interval": "",
         "trigger": "human", "human": 2_000_000, "rtype": "human"},
        # turn 1 carries a trailing tool gap (5 ms) to a successor outside
        # the golden window: the sidecar passthrough fields are then present
        # on every row, so slo_stats T_session is evaluable (rows with BOTH
        # human/tool empty -- e.g. a true last row -- are counted missing by
        # the tool's NA semantics; same behavior as real in-window cuts).
        {"session_id": "session_g3s", "turn_index": 1,
         "request_id": "session_g3s_request_1", "prefill": 500,
         "decode": 30, "arrival": "", "interval": 2_000_000,
         "trigger": "tool", "tool": 5_000_000, "rtype": "tool"},
    ]
    queue = write_golden_inputs("g3", rows)
    pointer = read_pointer()
    try:
        set_pointer(f"llama2_7b_inference/traces/{queue.name}")
        run_dir = run_simulation("g3", queue)
    finally:
        set_pointer(pointer)
    metrics = load_request_metrics(run_dir)
    check(len(metrics) == 2, "G3 emits 2 request rows")
    by_turn = {r["turn_index"]: r for r in metrics}
    t0, t1 = by_turn["0"], by_turn["1"]
    check(t0["request_type"] == "human" and t1["request_type"] == "tool",
          f"G3 request_type passthrough ({t0['request_type']}, "
          f"{t1['request_type']})")
    arr0, comp0 = as_int(t0["arrival_ns"]), as_int(t0["completion_ns"])
    arr1, comp1 = as_int(t1["arrival_ns"]), as_int(t1["completion_ns"])
    check(arr1 >= comp0,
          f"G3 turn-1 arrival is completion+interval closed loop "
          f"({arr1} >= {comp0}, gap={arr1 - comp0})")
    t_session = comp1 - arr0 - 7_000_000
    note(f"T_session hand calc = last_completion({comp1}) - first_arrival"
         f"({arr0}) - sum(human+tool)(2_000_000+5_000_000) = {t_session} ns")
    # slo_tools session command cross-check (tools must run while this
    # scenario's generated/ plan dir still exists).  human/tool passthrough
    # fields live in the plan token manifest (manifest.json), so pass it
    # explicitly: the default per-request resolution (metrics_manifest.json)
    # does not carry them (integration gap recorded in the B3 gate file).
    init = metric_records(run_dir, "init")
    token_manifest = None
    if init:
        manifest_path = init[0].get("manifest_path")
        if manifest_path:
            token_manifest = Path(str(manifest_path)).parent / "manifest.json"
    with tempfile.TemporaryDirectory() as tmp:
        sess_csv = Path(tmp) / "sess.csv"
        cmd = [sys.executable, str(SLO_TOOLS_DIR / "slo_stats.py"), "session",
               str(run_dir), "-o", str(sess_csv)]
        if token_manifest and token_manifest.is_file():
            cmd += ["--request-manifest", str(token_manifest)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        check(proc.returncode == 0, "slo_stats session runs on G3 products")
        if proc.returncode == 0 and sess_csv.exists():
            with sess_csv.open(newline="") as handle:
                srows = list(csv.DictReader(handle))
            got = as_int(srows[0]["t_session_ns"])
            check(got == t_session,
                  f"G3 slo_stats T_session == hand calc ({got} == {t_session})")
        # kv_hit_state: turn0 no_history; turn1 full (S1 local_hit / S2
        # three-state local_hbm -> full -- remote restore is not recompute).
        hits_csv = Path(tmp) / "hits.csv"
        proc = subprocess.run(
            [sys.executable, str(SLO_TOOLS_DIR / "kv_cache_adapter.py"),
             str(run_dir), "-o", str(Path(tmp) / "cache.csv"),
             "--hit-states", str(hits_csv)],
            capture_output=True, text=True)
        check(proc.returncode == 0, "kv_cache_adapter runs on G3 products")
        if proc.returncode == 0 and hits_csv.exists():
            with hits_csv.open(newline="") as handle:
                hrows = list(csv.DictReader(handle))
            by_id = {r["request_id"]: r for r in hrows}
            st0 = by_id.get("session_g3s_request_0", {}).get("kv_hit_state")
            st1 = by_id.get("session_g3s_request_1", {}).get("kv_hit_state")
            check(st0 == "no_history", f"G3 turn0 kv_hit_state == no_history "
                  f"(got {st0})")
            check(st1 == "full",
                  f"G3 turn1 kv_hit_state == full (S1 local_hit / S2 "
                  f"three-state local_hbm->full; got {st1})")


# ------------------------------------------------------------------ G2r --

def scenario_g2r(run_dir: Path) -> None:
    print(f"[G2r] queue-semantics cross-check on the real 60s products "
          f"({run_dir})")
    metrics = load_request_metrics(run_dir)
    queued = [r for r in metrics if as_int(r["queue_ns"]) > 0]
    check(len(queued) > len(metrics) // 2,
          f"G2r the saturated real window queues most requests "
          f"({len(queued)}/{len(metrics)})")
    ok_identity = all(
        as_int(r["prefill_start_ns"])
        == as_int(r["arrival_ns"]) + as_int(r["queue_ns"])
        for r in queued)
    check(ok_identity,
          "G2r queued rows satisfy prefill_start == arrival + queue_ns")
    ok_sum = all(
        as_int(r["e2e_ns"]) == (as_int(r["queue_ns"]) + as_int(r["prefill_ns"])
                                + as_int(r["prefill_decode_gap_ns"])
                                + as_int(r["decode_ns"]))
        for r in metrics)
    check(ok_sum, "G2r decomposition sum identity holds on all rows")
    inst_by_id = request_instance_by_id(run_dir)
    per_instance: dict[int, list[dict]] = {}
    for row in metrics:
        per_instance.setdefault(inst_by_id.get(row["request_id"]),
                                []).append(row)
    fcfs_bad = 0
    for instance, members in per_instance.items():
        members.sort(key=lambda r: as_int(r["prefill_start_ns"]))
        order = [int(r["queue_index"]) for r in members]
        if order != sorted(order):
            fcfs_bad += 1
    # Strict queue_index FCFS by prefill_start is NOT a law of this
    # scheduler: the KV-epoch admit gate may defer an earlier request while
    # a later one proceeds.  Recorded informationally, not asserted.
    note(f"per-instance queue_index-vs-prefill_start FCFS violations: "
         f"{fcfs_bad}/{len(per_instance)} instances (admit-gate deferral "
         f"is expected scheduler behavior, recorded not asserted)")
    top = max(queued, key=lambda r: as_int(r["queue_ns"]))
    note(f"max queue_ns={as_int(top['queue_ns'])} ({top['request_id']}); "
         f"queued={len(queued)}/{len(metrics)}")


# --------------------------------------------------------------------- G4 --

def scenario_g4(run_dir: Path) -> None:
    print(f"[G4] restore three-segment cross-check on {run_dir}")
    import argparse
    sys.path.insert(0, str(SLO_TOOLS_DIR))
    import restore_decomposition
    out = run_dir / "golden_g4_restore.csv"
    ns = argparse.Namespace(run_dir=run_dir, output=str(out), json="")
    check(restore_decomposition.cmd_restore(ns) == 0,
          "restore_decomposition runs on the full-detail products")
    with out.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    requests = {str(r["queue_index"]): r
                for r in metric_records(run_dir, "request")}
    anchors: dict[str, list[int]] = {}
    for record in metric_records(run_dir, "memory_anchor"):
        anchors.setdefault(str(record.get("subject_id")), []).append(
            int(record["tick_ns"]))
    anchored = [r for r in rows if r["restore_start_ns"] != "NA"]
    check(len(anchored) > 0,
          f"G4 run has anchored requests to verify (n={len(anchored)})")
    mismatches = 0
    sum_identity_checked = 0
    for row in anchored:
        qi = row["queue_index"]
        ticks = anchors.get(qi, [])
        rs_h, rc_h = min(ticks), max(ticks)
        ps = as_int(requests[qi]["prefill_start_ns"])
        pe = as_int(requests[qi]["prefill_end_ns"])
        # spec sec.3.2 fixed formulas (clamped), mirrored independently:
        pre_h = max(0, min(rc_h, ps) - rs_h)
        hidden_h = max(0, min(rc_h, pe) - max(rs_h, ps))
        exposed_h = max(0, rc_h - pe)
        duration_h = rc_h - rs_h
        ratio_h = (hidden_h / duration_h) if duration_h > 0 else None
        expected = {
            "restore_start_ns": str(rs_h), "restore_complete_ns": str(rc_h),
            "pre_prefill_restore_ns": str(pre_h),
            "hidden_restore_ns": str(hidden_h),
            "exposed_restore_stall_ns": str(exposed_h),
        }
        for field, want in expected.items():
            if row[field] != want:
                print(f"    mismatch {qi}.{field}: tool={row[field]} "
                      f"hand={want}")
                mismatches += 1
        if row["hidden_ratio"] != "NA" and ratio_h is not None:
            # tool CSV prints 6 decimals; compare at print precision
            if abs(float(row["hidden_ratio"]) - ratio_h) > 5e-7:
                print(f"    mismatch {qi}.hidden_ratio: "
                      f"{row['hidden_ratio']} vs {ratio_h}")
                mismatches += 1
        # clamped-formula sum identity: pre+hidden+exposed == rc-rs holds
        # whenever the restore starts before prefill end (rs <= pe)
        if rs_h <= pe:
            total = pre_h + hidden_h + exposed_h
            if total != rc_h - rs_h:
                mismatches += 1
                print(f"    mismatch {qi} sum identity: {total} vs "
                      f"{rc_h - rs_h}")
            sum_identity_checked += 1
    check(mismatches == 0,
          f"G4 all anchored rows match the independent hand recomputation "
          f"(mismatches={mismatches})")
    no_anchor = [r for r in rows if r["restore_start_ns"] == "NA"]
    check(all(r["pre_prefill_restore_ns"] == "NA"
              and r["hidden_restore_ns"] == "NA"
              and r["exposed_restore_stall_ns"] == "NA"
              and r["hidden_ratio"] == "NA" for r in no_anchor),
          f"G4 unanchored rows carry NA in all four derived fields "
          f"(n={len(no_anchor)})")
    note(f"G4 verified {len(anchored)} anchored rows "
         f"({sum_identity_checked} with rs<=pe clamped sum identity "
         f"pre+hidden+exposed == rc-rs), {len(no_anchor)} NA rows")


def main() -> int:
    scenario = sys.argv[1] if len(sys.argv) > 1 else "all"
    full_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if scenario in ("g4", "all"):
        if full_dir is None:
            print("g4 needs a full-detail run_dir argument")
            return 2
        scenario_g4(full_dir)
    if scenario in ("g2r", "all"):
        if full_dir is None:
            print("g2r needs a full-detail run_dir argument")
            return 2
        scenario_g2r(full_dir)
    if scenario in ("g1", "all"):
        scenario_g1()
    if scenario in ("g2", "all"):
        scenario_g2()
    if scenario in ("g3", "all"):
        scenario_g3()
    print()
    print(json.dumps({"repo": REPO.name, "tag": REPO_TAG,
                      "failures": FAILURES, "notes": NOTES}, indent=1))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
