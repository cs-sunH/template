#!/usr/bin/env python3
"""B3-5 T2 golden case LIVE runner (sh_3.0 仓库适配版).

Runs real mini simulations for G1/G2/G3 (and G4 via the 60s full run's
anchored request, the spec's sanctioned fallback for repos where a
2-request queue cannot trigger history restore) and asserts the
hand-derived expectations.

S3 队列适配：原生 9 列（含 next_trigger_type）+ 原生命名模式
``<stem>_request_queue.csv``（generate_trace.load_request_queue 严格列检，
timing 必须 1000ns 整除，trigger ∈ {human,tool}）。

用法（在仓根或任意目录）:
  python3 sh_test_mesh/slo_tools/tests/run_golden_live.py [--g4-run DIR]

输出：逐场景断言结果（JSON 摘要打到 stdout；全过 exit 0）。
临时产物落 /tmp/slo_wps/b3/<repo>/golden/；仓内只动 trace_config 指针
（finally 恢复占位）与 generated/（计划目录，跑完由调用方清理）。
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
WL = REPO / "sh_test_mesh" / "workload" / "llama2_7b_inference"
SLO_TOOLS = REPO / "sh_test_mesh" / "slo_tools"
RUNNER = REPO / "sh_test_mesh" / "run_scripts" / "run_online_strategy.sh"
GEN = REPO / "sh_test_mesh" / "generated"
TRACE_CONFIG = WL / "trace_config.csv"
PLACEHOLDER = "llama2_7b_inference/request_queue_placeholder.csv"
QUEUE_HEADER = ("session_id,turn_index,request_id,prefill_length,"
                "decode_length,session_arrival_time_ns,"
                "inter_request_interval_ns,next_trigger_type,description")
GAP_HUMAN_NS = 200_000_000  # G3 turn gap (200 ms, human reply)

RESULTS: dict = {"scenarios": {}, "ok": True}


def note(scenario, name, ok, detail=""):
    RESULTS["scenarios"].setdefault(scenario, {})[name] = {
        "ok": bool(ok), "detail": str(detail)[:400]}
    if not ok:
        RESULTS["ok"] = False
    print(("PASS" if ok else "FAIL"), scenario, name, detail[:200], flush=True)


def set_pointer(rel_or_abs: str):
    subprocess.run([sys.executable, "/tmp/slo_wps/set_trace_pointer.py",
                    str(TRACE_CONFIG), rel_or_abs], check=True,
                   capture_output=True)


def materialize_plan(log: Path) -> Path:
    for d in GEN.glob("llama2_7b_inference_54npus_*"):
        shutil.rmtree(d, ignore_errors=True)
    rc = subprocess.run([sys.executable, "plan_materializer.py"],
                        cwd=WL, capture_output=True, text=True)
    log.write_text(rc.stdout + rc.stderr, encoding="utf-8")
    if rc.returncode != 0:
        raise RuntimeError(f"plan materializer failed: {rc.stderr[-400:]}")
    info = json.loads(rc.stdout)
    return Path(info["plan_dir"])


def run_sim(queue: Path, run_dir: Path, env_extra: dict | None = None):
    env = {"SH_METRICS_DETAIL": "full", "SH_FIRST_TOKEN_SPLIT": "1"}
    env.update(env_extra or {})
    cmd = ["bash", str(RUNNER), str(run_dir), str(queue)]
    rc = subprocess.run(cmd, cwd=REPO, env={**__import__("os").environ, **env},
                        capture_output=True, text=True)
    if rc.returncode != 0:
        raise RuntimeError(f"runner failed rc={rc.returncode}: "
                           f"{(rc.stdout + rc.stderr)[-500:]}")
    return run_dir


def write_queue(path: Path, rows: list[list[str]]):
    """Native 9-column queue, native naming (timing on the 1000 ns grid)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(QUEUE_HEADER + "\n")
        for r in rows:
            assert len(r) == 9, r
            f.write(",".join(str(x) for x in r) + "\n")


def read_request_metrics(run_dir: Path):
    with open(run_dir / "request_metrics.csv", newline="",
              encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_decision_log(run_dir: Path):
    out = []
    with open(run_dir / "results" / "online_decision_log.jsonl",
              encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def slo_tool(args, cwd=SLO_TOOLS):
    return subprocess.run([sys.executable] + args, cwd=cwd,
                          capture_output=True, text=True)


# ---------------------------------------------------------------------------
# G1 单请求无竞争
# ---------------------------------------------------------------------------

def build_g1(base: Path) -> Path:
    queue = base / "golden_g1_request_queue.csv"
    write_queue(queue, [[
        "session_g1", 0, "session_g1_request_0", 2048, 64, 0, "", "human",
        "B3-5 G1 single request no contention"]])
    return queue


def scenario_g1(base: Path):
    run = run_sim(build_g1(base), base / "g1_run")
    rows = read_request_metrics(run)
    note("G1", "single_row", len(rows) == 1, f"rows={len(rows)}")
    r = rows[0]
    a, ps, pe = int(r["arrival_ns"]), int(r["prefill_start_ns"]), int(r["prefill_end_ns"])
    ds, comp = int(r["decode_start_ns"]), int(r["completion_ns"])
    dl = int(r["decode_length"])
    # S3 admission lands on the next 1 ns scheduler tick after arrival
    # (observed queue_ns=1 at arrival 0); a true queue would be >> 1 us.
    note("G1", "queue_ns_within_one_grid_tick",
         int(r["queue_ns"]) <= 1000, f"queue_ns={r['queue_ns']}")
    # E2E = queue + prefill + gap + decode（fail-closed 已保证，此处独立复算）
    e2e = comp - a
    note("G1", "e2e_decomposition",
         e2e == int(r["queue_ns"]) + int(r["prefill_ns"])
         + int(r["prefill_decode_gap_ns"]) + int(r["decode_ns"]),
         f"e2e={e2e}")
    # SPLIT=1：first_token − arrival = queue + prefill + gap + 首步
    ft = int(r["first_token_ns"])
    first_step = int(r["decode_ns"]) // dl
    lhs, rhs = ft - a, int(r["queue_ns"]) + int(r["prefill_ns"]) + int(r["prefill_decode_gap_ns"]) + first_step
    note("G1", "first_token_equals_prefill_plus_first_step",
         abs(lhs - rhs) <= max(1000, first_step // 100),
         f"ft-arrival={lhs} expected={rhs} first_step={first_step} decode_ns={r['decode_ns']} dl={dl}")
    note("G1", "first_token_exact_source", r["first_token_source"] == "exact",
         f"source={r['first_token_source']}")
    # informational: solo decode steps are near-uniform but not exactly
    # (98050661/64 leaves remainder 37 -> last step shorter).
    RESULTS["scenarios"].setdefault("G1", {})["solo_step_remainder_info"] = {
        "decode_ns": r["decode_ns"], "dl": dl,
        "remainder": int(r["decode_ns"]) % dl}


# ---------------------------------------------------------------------------
# G2 双请求同 instance 排队
# ---------------------------------------------------------------------------

def _g2_internals(run: Path):
    """prefill/decode instance + admission/drain ticks from the artifacts.

    Mirrors load_imbalance.collect_intervals: admission = decision.
    admission_time_ns (S3 explicit field) else prefill tick; instance =
    decode decision decode_instance_index; drain = first train_ledger row
    draining the request.
    """
    dec = read_decision_log(run)
    prefill_instance, admission_tick, decode_instance = {}, {}, {}
    for rec in dec:
        rid = rec.get("request_id")
        d = rec.get("decision") or {}
        if rec.get("kind") == "prefill":
            prefill_instance[rid] = d.get("prefill_instance_index")
            admission_tick[rid] = d.get("admission_time_ns", rec["tick"])
        elif rec.get("kind") == "decode":
            decode_instance[rid] = d.get("decode_instance_index")
    drains = {}
    with open(run / "results" / "train_ledger.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("first_step"):
                # split first-step rows duplicate the membership listing
                # (strip-list rule); drains are attributed to the remainder
                # row's tick.
                continue
            for rid in d.get("drains", []) or []:
                drains.setdefault(rid, d["tick"])
    return prefill_instance, admission_tick, decode_instance, drains


def _hand_timeavg_backlog(intervals, bucket_ns):
    """Independent re-implementation of load_imbalance's bucketed timeavg:
    span = min admission .. max drain; bucket counts any overlap
    ([admission, drain) vs [b*bucket, +bucket)); covered = n*bucket."""
    span_start = min(i[0] for i in intervals)
    span_end = max(i[1] for i in intervals)
    n_buckets = max(1, -(-(span_end - span_start) // bucket_ns))
    series = [0] * n_buckets
    for adm, drn in intervals:
        first = (adm - span_start) // bucket_ns
        last = (drn - span_start) // bucket_ns
        for b in range(max(0, first), min(n_buckets, last + 1)):
            bstart = span_start + b * bucket_ns
            if drn > bstart and adm < bstart + bucket_ns:
                series[b] += 1
    covered = sum(series) * bucket_ns
    return covered / (span_end - span_start)


def build_g2(base: Path) -> Path:
    queue2 = base / "golden_g2_request_queue.csv"
    write_queue(queue2, [[
        "session_g2a", 0, "session_g2a_request_0", 4096, 64, 0, "", "human",
        "B3-5 G2 two sessions same tick"],
        ["session_g2b", 0, "session_g2b_request_0", 4096, 64, 0, "", "human",
         "B3-5 G2 two sessions same tick"]])
    return queue2


def scenario_g2(base: Path):
    run = run_sim(build_g2(base), base / "g2_run")
    rows = read_request_metrics(run)
    pi, adm, dec_inst, drn = _g2_internals(run)
    note("G2", "two_rows", len(rows) == 2, f"rows={len(rows)}")
    insts = [pi.get(r["request_id"]) for r in rows]
    note("G2", "same_instance", len(set(insts)) == 1 and None not in insts,
         f"prefill instances={insts}")
    if len(set(insts)) != 1 or None in insts:
        return
    r0, r1 = sorted(rows, key=lambda r: int(r["prefill_start_ns"]))
    # S3 unified-instance co-batching: the second admission follows the
    # first within a small offset (not a FIFO head-of-line queue); both
    # prefills then interleave inside the shared prefill train.
    note("G2", "second_admission_offset_small",
         0 < int(r1["queue_ns"]) <= 100_000,
         f"queue_ns0={r0['queue_ns']} queue_ns1={r1['queue_ns']}")
    note("G2", "prefill_windows_overlap",
         int(r1["prefill_start_ns"]) < int(r0["prefill_end_ns"]),
         f"[{r1['prefill_start_ns']},{r1['prefill_end_ns']}) vs "
         f"[{r0['prefill_start_ns']},{r0['prefill_end_ns']})")
    # same prefill_length (4096) each: sharing the instance dilates the
    # later request's prefill strictly beyond the earlier one's.
    note("G2", "contention_dilation_second_longer",
         int(r1["prefill_length"]) == int(r0["prefill_length"])
         and int(r1["prefill_ns"]) > int(r0["prefill_ns"]),
         f"p0={r0['prefill_ns']} p1={r1['prefill_ns']} (same "
         f"{r0['prefill_length']} tokens)")
    for r in (r0, r1):
        e2e = int(r["completion_ns"]) - int(r["arrival_ns"])
        note("G2", f"decomposition_sums_{r['request_id']}",
             e2e == int(r["queue_ns"]) + int(r["prefill_ns"])
             + int(r["prefill_decode_gap_ns"]) + int(r["decode_ns"]),
             f"e2e={e2e}")
    # load_imbalance rebuild vs 手算（独立复算同一分桶算法，桶 1ms）。
    rids = [r["request_id"] for r in rows]
    if all(x in adm and x in drn and x in dec_inst for x in rids):
        bucket_ns = 1_000_000
        by_inst = {}
        for x in rids:
            by_inst.setdefault(dec_inst[x], []).append((adm[x], drn[x]))
        hand = {i: _hand_timeavg_backlog(ivs, bucket_ns)
                for i, ivs in by_inst.items()}
        drill = base / "g2_drill_manifest.json"
        repo_manifest = json.loads((SLO_TOOLS / "slo_params_manifest.json")
                                   .read_text(encoding="utf-8"))
        repo_manifest["params"]["imbalance_bucket_ns"] = {
            "value": bucket_ns, "unit": "ns",
            "derivation_program": ("主规格 §1.5-B：不均衡统计桶桶长推导同链路"
                                   "时间桶（从孤立基线实测取数并记录比值）"),
            "evidence": "DRILL-ONLY golden-live 锚点（1ms），非 B4 推导值",
            "rationale": None}
        drill.write_text(json.dumps(repo_manifest), encoding="utf-8")
        # SPLIT=1 的 train_ledger first_step 行复用整行发射器、重复列
        # drains/joiners（B3-2 剥离清单同款规则）→ 影子 run_dir 过滤后喂给
        # load_imbalance（five-repo 同文工具不改动）。
        shadow = base / "g2_li_rundir" / "results"
        shadow.mkdir(parents=True, exist_ok=True)
        shutil.copy(run / "results" / "online_decision_log.jsonl",
                    shadow / "online_decision_log.jsonl")
        shutil.copy(run / "cpp.log", shadow.parent / "cpp.log")
        with open(shadow / "train_ledger.jsonl", "w", encoding="utf-8") as out:
            for line in open(run / "results" / "train_ledger.jsonl",
                             encoding="utf-8"):
                if not json.loads(line).get("first_step"):
                    out.write(line)
        rc = slo_tool(["load_imbalance.py", str(shadow.parent),
                       "--manifest", str(drill),
                       "-o", str(base / "g2_li.csv")])
        ok = rc.returncode == 0
        detail = f"rc={rc.returncode} hand={ {i: round(v, 6) for i, v in hand.items()} }"
        if ok:
            li = list(csv.DictReader(open(base / "g2_li.csv")))
            tool = {int(r["instance_index"]): float(r["time_avg_backlog"])
                    for r in li}
            detail += f" tool={tool}"
            ok = set(tool) == set(hand) and all(
                abs(tool[i] - hand[i]) < 1e-6 for i in hand)
        note("G2", "load_imbalance_matches_hand", ok, detail)
    else:
        note("G2", "load_imbalance_matches_hand", False,
             "missing admission/drain/decode-interval inputs")


# ---------------------------------------------------------------------------
# G3 单 session 两轮（human gap）
# ---------------------------------------------------------------------------

def build_g3(base: Path) -> Path:
    queue = base / "golden_g3_request_queue.csv"
    write_queue(queue, [[
        "session_g3", 0, "session_g3_request_0", 1536, 48, 0, "", "human",
        "B3-5 G3 session two turns; row0 trigger=human"],
        ["session_g3", 1, "session_g3_request_1", 512, 32, "",
         GAP_HUMAN_NS, "human",
         "B3-5 G3 turn1 arrives at completion0+gap (closed loop)"]])
    return queue


def scenario_g3(base: Path):
    run = run_sim(build_g3(base), base / "g3_run")
    rows = read_request_metrics(run)
    note("G3", "two_rows", len(rows) == 2, f"rows={len(rows)}")
    r0, r1 = rows
    # request_type 透传：turn0=human（无前置 gap 规则）；turn1=human（row0
    # next_trigger_type=human × 本行 interval）。
    note("G3", "request_type_passthrough",
         r0["request_type"] == "human" and r1["request_type"] == "human",
         f"types={r0['request_type']},{r1['request_type']}")
    # 闭环到达：arrival1 = completion0 + gap。
    note("G3", "closed_loop_arrival",
         int(r1["arrival_ns"]) == int(r0["completion_ns"]) + GAP_HUMAN_NS,
         f"arrival1={r1['arrival_ns']} completion0={r0['completion_ns']}")
    # T_session 手算 = completion1 − arrival0 − Σ(human+tool)=gap。
    # S3 的 metrics_manifest.json（合成口径）不带 human/tool 透传（只有
    # 决策侧 manifest.json 带）——用决策侧字段补丁出 --request-manifest。
    expected_ts = int(r1["completion_ns"]) - int(r0["arrival_ns"]) - GAP_HUMAN_NS
    patched = base / "g3_request_manifest.json"
    init = None
    for line in open(run / "cpp.log", encoding="utf-8"):
        if '"type":"init"' in line:
            init = json.loads(line.split("[METRIC] ", 1)[1])
            break
    mm = json.loads(open(init["manifest_path"], encoding="utf-8").read())
    dm = json.loads((Path(init["manifest_path"]).parent
                     / "manifest.json").read_text(encoding="utf-8"))
    gaps = {r["request_id"]: (r.get("human_time_ns"),
                              r.get("tool_time_ns"))
            for r in dm.get("requests", [])}
    for rr in mm["requests"]:
        h, tl = gaps.get(rr["request_id"], (None, None))
        # turn-0 无前置 gap：按 0 填充（总和不变；None 会被 session 工具的
        # 缺字段保守判定计缺 → NA）。
        rr["human_time_ns"] = h if h is not None else (0 if rr.get("turn_index") == 0 else None)
        rr["tool_time_ns"] = tl
    patched.write_text(json.dumps(mm), encoding="utf-8")
    rc = slo_tool(["slo_stats.py", "session", str(run),
                   "--request-manifest", str(patched), "-o",
                   str(base / "g3_session.csv")])
    tool_ts = None
    if rc.returncode == 0:
        srows = list(csv.DictReader(open(base / "g3_session.csv")))
        tool_ts = int(srows[0]["t_session_ns"]) if srows else None
    note("G3", "t_session_hand_value", tool_ts == expected_ts,
         f"hand={expected_ts} tool={tool_ts}")
    # 已知差距记录：S3 metrics_manifest 无 gap 透传（session 工具默认源）
    # → 不补丁时 t_session=NA（B4 session 统计在 S3 需主控统一裁决）。
    rc_na = slo_tool(["slo_stats.py", "session", str(run), "-o",
                      str(base / "g3_session_default.csv")])
    RESULTS["scenarios"].setdefault("G3", {})[
        "metrics_manifest_gap_passthrough"] = {
            "detail": "default source yields t_session=NA on S3 "
                      "(synthetic metrics_manifest lacks human/tool; "
                      "decision-side manifest.json has them)"}
    # kv_hit_state：S3 partial 语义——turn1 历史 sticky 驻留本地 HBM
    #（kv_reserve 1M tokens >> 上下文，无逐出）→ affinity=resident_local_hbm
    # → kv_hit_state=full；turn0 first_request_non_edge → no_history。
    rc = slo_tool(["kv_cache_adapter.py", str(run)])
    hits = list(csv.DictReader(open(run / "kv_hit_states.csv"))) \
        if (run / "kv_hit_states.csv").exists() else []
    by_rid = {h["request_id"]: h for h in hits}
    note("G3", "kv_hit_state_hand",
         by_rid.get(r0["request_id"], {}).get("kv_hit_state") == "no_history"
         and by_rid.get(r1["request_id"], {}).get("kv_hit_state") == "full",
         f"turn0={by_rid.get(r0['request_id'], {}).get('kv_hit_state')} "
         f"turn1={by_rid.get(r1['request_id'], {}).get('kv_hit_state')} "
         f"evidence1={by_rid.get(r1['request_id'], {}).get('evidence', '')[:120]}")


# ---------------------------------------------------------------------------
# G4 restore 三段（S3 fallback：60s full run 有锚点请求）
# ---------------------------------------------------------------------------

def scenario_g4(base: Path, g4_run: Path):
    if g4_run is None or not (g4_run / "cpp.log").exists():
        note("G4", "run_available", False, f"no 60s full run at {g4_run}")
        return
    # 独立手算：从 cpp.log 原始锚点重算三段，与 restore_decomposition 输出 diff。
    anchors: dict[int, list[int]] = {}
    prefill_window: dict[str, dict] = {}
    with open(g4_run / "cpp.log", encoding="utf-8") as f:
        for line in f:
            if "[METRIC]" not in line:
                continue
            d = json.loads(line.split("[METRIC] ", 1)[1])
            if d.get("type") == "memory_anchor":
                anchors.setdefault(int(d["subject_id"]), []).append(int(d["tick_ns"]))
            elif d.get("type") == "request":
                prefill_window[str(d["queue_index"])] = {
                    "request_id": d.get("request_id"),
                    "prefill_start_ns": d.get("prefill_start_ns"),
                    "prefill_end_ns": d.get("prefill_end_ns")}
    out_csv = base / "g4_restore_tool.csv"
    rc = slo_tool(["restore_decomposition.py", str(g4_run), "-o", str(out_csv)])
    note("G4", "tool_exit", rc.returncode == 0, f"rc={rc.returncode}")
    if rc.returncode != 0:
        return
    tool_rows = {r["queue_index"]: r for r in csv.DictReader(open(out_csv))}
    checked = mismatches = 0
    sample = None
    for qi, ticks in anchors.items():
        row = tool_rows.get(str(qi))
        if row is None or row["restore_start_ns"] == "NA":
            continue
        pw = prefill_window[str(qi)]
        rs, rcpt = min(ticks), max(ticks)
        ps, pe = int(pw["prefill_start_ns"]), int(pw["prefill_end_ns"])
        pre = max(0, min(rcpt, ps) - rs)
        hidden = max(0, min(rcpt, pe) - max(rs, ps))
        exposed = max(0, rcpt - pe)
        dur = rcpt - rs
        ratio = hidden / dur if dur else None
        checked += 1
        ok = (int(row["restore_start_ns"]) == rs
              and int(row["restore_complete_ns"]) == rcpt
              and int(row["pre_prefill_restore_ns"]) == pre
              and int(row["hidden_restore_ns"]) == hidden
              and int(row["exposed_restore_stall_ns"]) == exposed
              and (ratio is None and row["hidden_ratio"] == "NA"
                   or abs(float(row["hidden_ratio"]) - ratio) <= 5e-7))
        if not ok:
            mismatches += 1
        if sample is None:
            sample = {"queue_index": qi, "request_id": pw["request_id"],
                      "anchors": f"[{rs},{rcpt}]",
                      "prefill": f"[{ps},{pe}]",
                      "segments": f"pre={pre} hidden={hidden} exposed={exposed}",
                      "match": ok}
    note("G4", "hand_recomputed_segments_match",
         checked > 0 and mismatches == 0,
         f"anchored_requests_checked={checked} mismatches={mismatches} "
         f"sample={json.dumps(sample, ensure_ascii=False)}")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g4-run", default=None,
                    help="60s full run dir for the G4 anchored-request check")
    ap.add_argument("--out", default="/tmp/slo_wps/b3/S3/golden")
    args = ap.parse_args()
    base = Path(args.out)
    base.mkdir(parents=True, exist_ok=True)
    original = None
    for line in TRACE_CONFIG.read_text(encoding="utf-8").splitlines():
        if line.startswith("config,request_queue_csv,"):
            original = line.split(",")[2]
    try:
        for fn, builder in (("G1", build_g1), ("G2", build_g2),
                            ("G3", build_g3)):
            q = builder(base)
            set_pointer(str(q))
            materialize_plan(base / f"{Path(q).stem}_planmat.log")
            t0 = time.time()
            if fn == "G1":
                scenario_g1(base)
            elif fn == "G2":
                scenario_g2(base)
            else:
                scenario_g3(base)
            print(f"[{fn}] took {time.time()-t0:.1f}s", flush=True)
        scenario_g4(base, Path(args.g4_run) if args.g4_run else None)
    finally:
        if original is not None:
            set_pointer(original)
    (base / "golden_verdicts.json").write_text(
        json.dumps(RESULTS, indent=1, ensure_ascii=False), encoding="utf-8")
    print("OVERALL:", "PASS" if RESULTS["ok"] else "FAIL")
    return 0 if RESULTS["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
