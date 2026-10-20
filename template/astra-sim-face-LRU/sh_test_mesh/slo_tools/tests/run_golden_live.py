#!/usr/bin/env python3
"""B3-5 T2 golden case 运行版（真仿真断言，G1-G4）。

用 hand-craft 队列（8 列 + canonical sidecar 15 列，落在仓内
workload/llama2_7b_inference/traces/ 下，命名以
``_request_queue_recompute.csv`` 结尾以便 sidecar 兄弟 join）逐场景跑
run_online_strategy.sh（detail=full、SH_FIRST_TOKEN_SPLIT=1），对
request_metrics.csv / online_decision_log / train_ledger / kv_cache_adapter
输出做手算断言：

  G1 单请求无竞争：queue_ns = prefill_start−arrival，恒含准入图开销
     （P1-1 锚点修复起断言 0 ≤ queue_ns < prefill/20；有竞争时
     queue ≈ 占位者 prefill，量级分离）；e2e=queue+prefill+gap+decode
     （整数恒等）；first_token−arrival = queue+prefill+gap+首步迭代
     时长，其中首步迭代时长 = decode_ns/decode_length（单请求批恒定 →
     迭代均匀）。
  G2 同刻突发排队（请求数 > 实例数）：溢出请求 queue_ns>0 且
     queue_ns≈占位者 prefill；分解和恒等；load_imbalance.py 输出与
     独立手算重建（admission=prefill tick、instance=decode_instance、
     drain=train_ledger drains tick、桶长=manifest）一致。
  G3 单 session 两轮 human/tool：T_session = turn0.e2e + interval +
     turn1.e2e（到达=父完成+interval 闭环恒等）；request_type 逐行透传；
     第二轮 kv_hit_state 由 decision history_action 手推并与
     kv_cache_adapter --reconcile 输出一致。
  G4 已知 restore：迷你 3 轮 session 若产生 history NOC_MIGRATE 锚点
     （memory_anchor code7），对锚点请求核对三段和 = 总 restore 时长
     ——仅锚点窗与 prefill 窗重叠或更早（rs≤pe）的形态；窗整体落在
     prefill 结束后的 rs>pe 形态（行内判别 pre=hidden=0，如 turn-0
     prefill→decode 迁移腿）属规格登记的非恒等形态（各指标测试建议.md
     §B），豁免并记 note，全部豁免时同 FALLBACK_G4；无锚点则打印
     FALLBACK_G4（改用 t3_full_off_split 真实锚点核对，由调用方执行）。

Usage: python3 run_golden_live.py <repo_root> [--work-dir DIR]
退出码 0 = 全过；非 0 = 有断言失败（stderr 列明细）。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

QUEUE_COLUMNS = [
    "session_id", "turn_index", "request_id", "prefill_length",
    "decode_length", "session_arrival_time_ns", "inter_request_interval_ns",
    "description",
]
SIDECAR_COLUMNS = [
    "request_id", "turn_index", "session_id", "raw_prefix_tokens",
    "raw_new_prefill_tokens", "effective_prompt_tokens", "prefill_length",
    "decode_length", "session_arrival_time_ns", "inter_request_interval_ns",
    "prefix_mode", "digest", "human_time_ns", "tool_time_ns",
    "request_type",
]
REQUEST_METRICS_RELPATH = "request_metrics.csv"
PLACEHOLDER_REL = "llama2_7b_inference/request_queue_placeholder.csv"


class GoldenCheckError(AssertionError):
    pass


def check(cond: bool, message: str) -> None:
    if not cond:
        raise GoldenCheckError(message)


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def write_provenance_sidecar(repo: Path, queue_path: Path) -> None:
    """为 golden 队列写 <queue>.provenance.json（M-16 门对齐，移植自
    wscllm 仓 run_golden_live 同款修复，2026-10-07）。

    C++ WindowedTraceReader 的 provenance 门（2026-10-07 起本仓缺席即
    fail-closed，对齐 wscllm-LRU M-16 2026-09-30）对每个队列 CSV
    fail-closed 地要求同名 .provenance.json 边车，缺失即视同截断输入、
    在首个 Submit 前拒绝提交。golden live 此前只写 8 列 CSV，四个场景
    自本门落地起启动即 reader_fatal。统计算法零复制：直接复用权威物化
    器 traces/derive_20_first_30_seconds.py 的 queue_provenance（与
    C++ 索引遍逐字节同口径，含 CSV writer 的 CRLF 字节与 '#' 注释行
    跳过）。"""
    derive_path = (
        repo / "sh_test_mesh/workload/llama2_7b_inference/traces"
        / "derive_20_first_30_seconds.py")
    spec = importlib.util.spec_from_file_location(
        "derive_20_first_30_seconds", derive_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    provenance = module.queue_provenance(
        str(queue_path), "run_golden_live.py hand-crafted golden queue")
    sidecar = Path(str(queue_path) + ".provenance.json")
    with sidecar.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")


def set_pointer(trace_config: Path, rel_queue: str) -> None:
    lines = trace_config.read_text(encoding="utf-8").splitlines(keepends=True)
    out = []
    for line in lines:
        if line.startswith("config,request_queue_csv,"):
            parts = line.rstrip("\n").split(",")
            parts[2] = rel_queue
            line = ",".join(parts) + ("\n" if line.endswith("\n") else "")
        out.append(line)
    trace_config.write_text("".join(out), encoding="utf-8")


def instance_count(repo_root: Path) -> tuple[int, int]:
    """(unified_or_prefill_instances, total_instances) from trace_config."""
    trace_config = (repo_root / "sh_test_mesh/workload/llama2_7b_inference"
                    / "trace_config.csv")
    total = 0
    prefill_like = 0
    with trace_config.open(encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if row and row[0] == "inference_group":
                total += 1
                roles = {cell.strip() for cell in row[6:] if cell.strip()}
                if not roles or "decode" not in roles:
                    prefill_like += 1
    return prefill_like, total


def run_scenario(repo_root: Path, work_dir: Path, name: str,
                 queue_rel: str) -> Path:
    wl = repo_root / "sh_test_mesh/workload/llama2_7b_inference"
    gen = repo_root / "sh_test_mesh/generated"
    run_dir = work_dir / name
    set_pointer(wl / "trace_config.csv", queue_rel)
    for stale in list(gen.glob("llama2_7b_inference_54npus_*")) + \
            list(gen.glob("llama2_7b_wsc_llm_inference_54npus_*")):
        shutil.rmtree(stale)
    subprocess.run(
        [sys.executable, "plan_materializer.py"], cwd=wl, check=True,
        stdout=subprocess.DEVNULL)
    env = dict(os.environ)
    env["SH_METRICS_DETAIL"] = "full"
    env["SH_FIRST_TOKEN_SPLIT"] = "1"
    subprocess.run(
        ["bash",
         str(repo_root / "sh_test_mesh/run_scripts/run_online_strategy.sh"),
         str(run_dir), str(repo_root / "sh_test_mesh/workload" / queue_rel)],
        env=env, check=True, stdout=subprocess.DEVNULL)
    check((run_dir / REQUEST_METRICS_RELPATH).is_file(),
          f"{name}: request_metrics.csv missing")
    return run_dir


def read_request_rows(run_dir: Path) -> list[dict]:
    with (run_dir / REQUEST_METRICS_RELPATH).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_int(value) -> int:
    return int(value)


# ---------------------------------------------------------------------------
# Scenario queue definitions (hand-crafted)
# ---------------------------------------------------------------------------

def g1_rows() -> tuple[list[dict], list[dict]]:
    queue = [{
        "session_id": "g1_s0", "turn_index": 0,
        "request_id": "g1_s0_r0", "prefill_length": 5000,
        "decode_length": 64, "session_arrival_time_ns": 1_000_000,
        "inter_request_interval_ns": "", "description": "golden G1",
    }]
    sidecar = [{
        "request_id": "g1_s0_r0", "turn_index": 0, "session_id": "g1_s0",
        "raw_prefix_tokens": 0, "raw_new_prefill_tokens": 5000,
        "effective_prompt_tokens": 5000, "prefill_length": 5000,
        "decode_length": 64, "session_arrival_time_ns": 1_000_000,
        "inter_request_interval_ns": "", "prefix_mode": "recompute",
        "digest": "golden", "human_time_ns": "", "tool_time_ns": "",
        "request_type": "tool",
    }]
    return queue, sidecar


def g2_rows(prefill_instances: int) -> tuple[list[dict], list[dict]]:
    count = prefill_instances + 3
    queue, sidecar = [], []
    for i in range(count):
        queue.append({
            "session_id": f"g2_s{i}", "turn_index": 0,
            "request_id": f"g2_s{i}_r0",
            "prefill_length": 4000 + 100 * i, "decode_length": 48,
            "session_arrival_time_ns": 2_000_000,
            "inter_request_interval_ns": "",
            "description": "golden G2 same-tick burst",
        })
        sidecar.append({
            "request_id": f"g2_s{i}_r0", "turn_index": 0,
            "session_id": f"g2_s{i}", "raw_prefix_tokens": 0,
            "raw_new_prefill_tokens": 4000 + 100 * i,
            "effective_prompt_tokens": 4000 + 100 * i,
            "prefill_length": 4000 + 100 * i, "decode_length": 48,
            "session_arrival_time_ns": 2_000_000,
            "inter_request_interval_ns": "", "prefix_mode": "recompute",
            "digest": "golden", "human_time_ns": "", "tool_time_ns": "",
            "request_type": "human",
        })
    return queue, sidecar


def g3_rows() -> tuple[list[dict], list[dict]]:
    queue = [
        {"session_id": "g3_s0", "turn_index": 0, "request_id": "g3_s0_r0",
         "prefill_length": 4000, "decode_length": 32,
         "session_arrival_time_ns": 1_000_000,
         "inter_request_interval_ns": "",
         "description": "golden G3 turn0"},
        {"session_id": "g3_s0", "turn_index": 1, "request_id": "g3_s0_r1",
         "prefill_length": 800, "decode_length": 48,
         "session_arrival_time_ns": "",
         "inter_request_interval_ns": 50_000_000,
         "description": "golden G3 turn1"},
    ]
    sidecar = [
        {"request_id": "g3_s0_r0", "turn_index": 0, "session_id": "g3_s0",
         "raw_prefix_tokens": 0, "raw_new_prefill_tokens": 4000,
         "effective_prompt_tokens": 4000, "prefill_length": 4000,
         "decode_length": 32, "session_arrival_time_ns": 1_000_000,
         "inter_request_interval_ns": "",
         "prefix_mode": "recompute", "digest": "golden",
         "human_time_ns": "", "tool_time_ns": "",
         "request_type": "tool"},
        {"request_id": "g3_s0_r1", "turn_index": 1, "session_id": "g3_s0",
         "raw_prefix_tokens": 4000, "raw_new_prefill_tokens": 800,
         "effective_prompt_tokens": 4800, "prefill_length": 800,
         "decode_length": 48, "session_arrival_time_ns": "",
         "inter_request_interval_ns": 50_000_000,
         "prefix_mode": "recompute", "digest": "golden",
         "human_time_ns": "50000000", "tool_time_ns": "",
         "request_type": "human"},
    ]
    return queue, sidecar


def g4_rows() -> tuple[list[dict], list[dict]]:
    queue = [
        {"session_id": "g4_s0", "turn_index": 0, "request_id": "g4_s0_r0",
         "prefill_length": 6000, "decode_length": 32,
         "session_arrival_time_ns": 1_000_000,
         "inter_request_interval_ns": "",
         "description": "golden G4 turn0"},
        {"session_id": "g4_s0", "turn_index": 1, "request_id": "g4_s0_r1",
         "prefill_length": 900, "decode_length": 32,
         "session_arrival_time_ns": "",
         "inter_request_interval_ns": 40_000_000,
         "description": "golden G4 turn1"},
        {"session_id": "g4_s0", "turn_index": 2, "request_id": "g4_s0_r2",
         "prefill_length": 700, "decode_length": 32,
         "session_arrival_time_ns": "",
         "inter_request_interval_ns": 20_000_000,
         "description": "golden G4 turn2"},
    ]
    sidecar = [
        {"request_id": "g4_s0_r0", "turn_index": 0, "session_id": "g4_s0",
         "raw_prefix_tokens": 0, "raw_new_prefill_tokens": 6000,
         "effective_prompt_tokens": 6000, "prefill_length": 6000,
         "decode_length": 32, "session_arrival_time_ns": 1_000_000,
         "inter_request_interval_ns": "",
         "prefix_mode": "recompute", "digest": "golden",
         "human_time_ns": "", "tool_time_ns": "",
         "request_type": "tool"},
        {"request_id": "g4_s0_r1", "turn_index": 1, "session_id": "g4_s0",
         "raw_prefix_tokens": 6000, "raw_new_prefill_tokens": 900,
         "effective_prompt_tokens": 6900, "prefill_length": 900,
         "decode_length": 32, "session_arrival_time_ns": "",
         "inter_request_interval_ns": 40_000_000,
         "prefix_mode": "recompute", "digest": "golden",
         "human_time_ns": "40000000", "tool_time_ns": "",
         "request_type": "human"},
        {"request_id": "g4_s0_r2", "turn_index": 2, "session_id": "g4_s0",
         "raw_prefix_tokens": 6900, "raw_new_prefill_tokens": 700,
         "effective_prompt_tokens": 7600, "prefill_length": 700,
         "decode_length": 32, "session_arrival_time_ns": "",
         "inter_request_interval_ns": 20_000_000,
         "prefix_mode": "recompute", "digest": "golden",
         "human_time_ns": "", "tool_time_ns": "20000000",
         "request_type": "tool"},
    ]
    return queue, sidecar


# ---------------------------------------------------------------------------
# Scenario assertions
# ---------------------------------------------------------------------------

def assert_row_identity(row: dict, label: str, notes: list[str]) -> None:
    parts = ["queue_ns", "prefill_ns", "prefill_decode_gap_ns", "decode_ns"]
    values = [to_int(row[column]) for column in parts]
    e2e = to_int(row["e2e_ns"])
    check(sum(values) == e2e,
          f"{label}: decomposition {values} sums {sum(values)} != e2e {e2e}")
    arrival = to_int(row["arrival_ns"])
    completion = to_int(row["completion_ns"])
    check(completion - arrival == e2e,
          f"{label}: completion-arrival {completion - arrival} != e2e {e2e}")
    notes.append(f"{label}: q/p/g/d={values} e2e={e2e} (identity OK)")


def assert_first_token(row: dict, label: str, notes: list[str]) -> None:
    first_token = row["first_token_ns"]
    check(first_token != "NA", f"{label}: first_token NA under SPLIT=1")
    check(row["first_token_source"] == "exact",
          f"{label}: first_token_source {row['first_token_source']} != exact")
    arrival = to_int(row["arrival_ns"])
    completion = to_int(row["completion_ns"])
    value = to_int(first_token)
    check(arrival <= value <= completion,
          f"{label}: first_token {value} outside [{arrival},{completion}]")
    queue = to_int(row["queue_ns"])
    prefill = to_int(row["prefill_ns"])
    gap = to_int(row["prefill_decode_gap_ns"])
    decode = to_int(row["decode_ns"])
    decode_length = to_int(row["decode_length"])
    first_step = decode // decode_length
    expected = queue + prefill + gap + first_step
    drift = (value - arrival) - expected
    rel = abs(drift) / max(expected, 1)
    check(rel <= 0.02,
          f"{label}: first_token-arrival {value - arrival} != "
          f"q+p+g+first_step {expected} (rel {rel:.3e}; tolerance 2% "
          f"covers first-step batch marker/wakeup graph ops on top of the "
          f"uniform-iteration model)")
    notes.append(
        f"{label}: ft-arrival={value - arrival} expected={expected} "
        f"first_step=decode/{decode_length}={first_step} rel={rel:.2e}")


def scenario_g1(run_dir: Path, notes: list[str]) -> None:
    rows = read_request_rows(run_dir)
    check(len(rows) == 1, f"G1: {len(rows)} rows != 1")
    row = rows[0]
    check(row["terminal_status"] == "completed",
          f"G1: terminal_status {row['terminal_status']}")
    # G1 口径统筹(2026-09-29,P1-1 锚点修复):queue_ns = prefill_start
    # (pstart 标记 issue)− arrival,恒含准入图开销(TP 就绪屏障 + 起始
    # 标记链,纳秒~微秒级)——旧 ==0 断言恰依赖"prefill_start 被
    # dep-free 准入门抢先"的错误锚点。无竞争属性改为:queue_ns 严格
    # 小于 prefill 的 5%(有竞争时 queue ≈ 占位者 prefill,量级分离)。
    queue = to_int(row["queue_ns"])
    prefill = to_int(row["prefill_ns"])
    check(0 <= queue < max(prefill // 20, 1),
          f"G1: queue_ns {queue} not in [0, prefill/20={prefill // 20}) "
          "(no contention; admission-graph overhead allowed)")
    notes.append(f"G1: queue_ns={queue} (admission overhead, prefill={prefill})")
    assert_row_identity(row, "G1", notes)
    assert_first_token(row, "G1", notes)


def scenario_g2(run_dir: Path, notes: list[str], expected_overflow: int,
                manifest_path: Path) -> None:
    rows = read_request_rows(run_dir)
    check(all(r["terminal_status"] == "completed" for r in rows),
          "G2: incomplete requests")
    for row in rows:
        assert_row_identity(row, f"G2.{row['queue_index']}", notes)
    decisions = read_jsonl(run_dir / "results/online_decision_log.jsonl")
    prefill_inst = {}
    for entry in decisions:
        if entry.get("kind") == "prefill":
            prefill_inst[entry["request_id"]] = entry["decision"].get(
                "prefill_instance_index")
    arrival = to_int(rows[0]["arrival_ns"])
    # X3 + P1-1 口径(2026-09-29):queue_ns 现含全体行共有的准入图开销
    # 下界,排队行 = 明显高于该下界者(下界取全部行的最小 queue_ns)。
    min_queue = min(to_int(r["queue_ns"]) for r in rows)
    queued = [
        r for r in rows
        if to_int(r["queue_ns"]) > 2 * max(min_queue, 1)]
    notes.append(
        f"G2: {len(rows)} requests, overflow expected {expected_overflow}, "
        f"queued observed {len(queued)} (min_queue floor {min_queue})")
    # X3:expected_overflow 升级为硬断言(旧版只进日志,仅断言 >=1)。
    check(len(queued) == expected_overflow,
          f"G2: expected {expected_overflow} overflow (queued) requests, "
          f"observed {len(queued)}")
    # For each queued request, the occupying earlier request on the same
    # instance must have finished prefill before this one starts, and the
    # queue time must approximate one occupying prefill span.
    ratios = []
    by_inst: dict[int, list[dict]] = {}
    for row in rows:
        inst = prefill_inst.get(row["request_id"])
        by_inst.setdefault(inst, []).append(row)
    for row in queued:
        inst = prefill_inst.get(row["request_id"])
        start = to_int(row["arrival_ns"]) + to_int(row["queue_ns"])
        earlier = [r for r in by_inst.get(inst, [])
                   if to_int(r["prefill_start_ns"]) < start and r is not row]
        check(earlier, f"G2: queued {row['request_id']} has no earlier "
                       f"same-instance request")
        occupier = max(earlier, key=lambda r: to_int(r["prefill_start_ns"]))
        occ_prefill_end = to_int(occupier["prefill_end_ns"])
        check(occ_prefill_end <= start,
              f"G2: {row['request_id']} started {start} before occupier "
              f"{occupier['request_id']} prefill end {occ_prefill_end}")
        ratio = to_int(row["queue_ns"]) / max(
            occ_prefill_end - to_int(occupier["arrival_ns"]), 1)
        ratios.append(ratio)
        notes.append(
            f"G2: {row['request_id']} (inst {inst}) queue_ns="
            f"{row['queue_ns']} vs occupier {occupier['request_id']} "
            f"prefill_end-arrival={occ_prefill_end - to_int(occupier['arrival_ns'])} "
            f"ratio={ratio:.3f}")
    check(all(0.5 <= ratio <= 2.0 for ratio in ratios),
          f"G2: queue_ns vs occupier prefill ratio out of [0.5,2.0]: {ratios}")
    load_imbalance_handcheck(run_dir, notes, manifest_path)


def load_imbalance_handcheck(run_dir: Path, notes: list[str],
                             manifest_path: Path) -> None:
    """Independent rebuild of the WP7 integral vs slo_tools/load_imbalance."""
    slo_tools = Path(__file__).resolve().parents[1]
    json_out = run_dir / "golden_li.json"
    subprocess.run(
        [sys.executable, str(slo_tools / "load_imbalance.py"),
         str(run_dir), "--manifest", str(manifest_path),
         "--json", str(json_out)],
        check=True, stdout=subprocess.DEVNULL)
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    bucket_ns = payload.get("bucket_ns")
    check(bucket_ns, "G2: no imbalance bucket length in tool output")
    decisions = read_jsonl(run_dir / "results/online_decision_log.jsonl")
    ledger = read_jsonl(run_dir / "results/train_ledger.jsonl")
    admission, decode_inst = {}, {}
    for entry in decisions:
        if entry.get("kind") == "prefill":
            admission[entry["request_id"]] = entry["tick"]
        elif entry.get("kind") == "decode":
            decode_inst[entry["request_id"]] = entry["decision"].get(
                "decode_instance_index")
    drain = {}
    for entry in ledger:
        if entry.get("first_step"):
            continue  # WP9 emission-boundary rows are not terminal drains
        if str(entry.get("train_id", "")).startswith("prefill_train"):
            continue  # P-side degenerate rows drain at admission, not terminal
        for req in entry.get("exits", []) or []:
            drain.setdefault(req, entry["tick"])
    spans = []
    for rid, adm in admission.items():
        if rid in decode_inst and rid in drain:
            spans.append((decode_inst[rid], adm, drain[rid]))
    t0 = min(span[1] for span in spans)
    t1 = max(span[2] for span in spans)
    horizon = t1 - t0
    # L19（2026-09-29）精确积分口径：积压 = 在途请求对时间的积分，
    # 每实例 b̄ = Σ(drain−admission)/horizon——与 load_imbalance 的
    # _interval_covered_ns 同式（旧"重叠桶计数×桶长"镜像已随工具侧
    # 修正退役：部分重叠边缘桶整桶计入会虚增积压）。bucket_ns 仍读取
    # 并核对在输出（登记/对账参数，不进入取值）。
    covered: dict[int, int] = {}
    for inst, adm, dr in spans:
        covered[inst] = covered.get(inst, 0) + max(0, dr - adm)
    means = [covered[inst] / horizon for inst in covered]
    mean = sum(means) / len(means)
    var = sum((m - mean) ** 2 for m in means) / len(means)
    cv = math.sqrt(var) / mean
    tool_cv = payload.get("cv_time_avg")
    note = (f"G2: load_imbalance hand CV={cv:.6f} tool CV={tool_cv} "
            f"(instances={len(means)} horizon={horizon})")
    notes.append(note)
    if tool_cv is not None:
        check(abs(cv - float(tool_cv)) <= 1e-9,
              f"G2: load_imbalance CV hand {cv} != tool {tool_cv}")


def scenario_g3(run_dir: Path, notes: list[str]) -> None:
    rows = read_request_rows(run_dir)
    check(len(rows) == 2, f"G3: {len(rows)} rows != 2")
    turn0, turn1 = rows
    for row in rows:
        assert_row_identity(row, f"G3.{row['turn_index']}", notes)
    check(turn0["request_type"] == "tool",
          f"G3: turn0 request_type {turn0['request_type']} != tool")
    check(turn1["request_type"] == "human",
          f"G3: turn1 request_type {turn1['request_type']} != human")
    # Closed-loop arrival: turn1.arrival = turn0.completion + interval.
    interval = 50_000_000
    closed = to_int(turn0["completion_ns"]) + interval
    check(to_int(turn1["arrival_ns"]) == closed,
          f"G3: turn1 arrival {turn1['arrival_ns']} != turn0 completion "
          f"+ interval {closed}")
    t_session = (to_int(turn0["e2e_ns"]) + interval
                 + to_int(turn1["e2e_ns"]))
    span = to_int(turn1["completion_ns"]) - to_int(turn0["arrival_ns"])
    check(span == t_session,
          f"G3: session span {span} != e2e0+interval+e2e1 {t_session}")
    notes.append(f"G3: T_session={span} = {to_int(turn0['e2e_ns'])} + "
                 f"{interval} + {to_int(turn1['e2e_ns'])} (hand OK)")
    assert_first_token(turn0, "G3.turn0", notes)
    assert_first_token(turn1, "G3.turn1", notes)
    # kv_hit_state hand derivation：双级（-LRU 方案 A，与
    # kv_cache_adapter._hit_state_face_wscllm 同口径）——优先 decision
    # history_location_before 三态映射，字段缺失回退 history_action 四值表。
    decisions = read_jsonl(run_dir / "results/online_decision_log.jsonl")
    actions = {}
    locations = {}
    for entry in decisions:
        if entry.get("kind") == "prefill":
            actions[entry["request_id"]] = entry["decision"].get(
                "history_action")
            locations[entry["request_id"]] = entry["decision"].get(
                "history_location_before")
    location_mapping = {"local_hbm": "full",
                        "partial_hbm_remote": "partial",
                        "remote_memory": "full"}
    mapping = {"NO_HISTORY": "no_history", "LOCAL_HIT": "full",
               "NOC_MIGRATE": "full", "RECOMPUTE": "miss"}
    slo_tools = Path(__file__).resolve().parents[1]
    hs_path = run_dir / "golden_kv_hit_states.csv"
    subprocess.run(
        [sys.executable, str(slo_tools / "kv_cache_adapter.py"),
         str(run_dir), "--hit-states", str(hs_path), "--reconcile"],
        check=True, stdout=subprocess.DEVNULL)
    with hs_path.open(encoding="utf-8") as handle:
        adapter_states = {r["request_id"]: r["kv_hit_state"]
                          for r in csv.DictReader(handle)}
    for rid, action in actions.items():
        location = locations.get(rid)
        if location is not None:
            expected = location_mapping.get(location, "not_supported")
        else:
            expected = mapping.get(action, "not_supported")
        got = adapter_states.get(rid)
        check(got == expected,
              f"G3: kv_hit_state {rid}: adapter {got} != hand {expected} "
              f"(location={location}, history_action={action})")
        notes.append(f"G3: {rid} location={location} "
                     f"history_action={action} -> {got} (hand OK)")


def scenario_g4(run_dir: Path, notes: list[str]) -> bool:
    """Returns True if anchored restore requests exist (checked), else False."""
    slo_tools = Path(__file__).resolve().parents[1]
    out_csv = run_dir / "golden_restore.csv"
    subprocess.run(
        [sys.executable, str(slo_tools / "restore_decomposition.py"),
         str(run_dir), "-o", str(out_csv)],
        check=True, stdout=subprocess.DEVNULL)
    with out_csv.open(encoding="utf-8") as handle:
        restore_rows = list(csv.DictReader(handle))
    anchored = [r for r in restore_rows if r["restore_start_ns"] != "NA"]
    if not anchored:
        notes.append("G4: no anchored restore request in mini scenario -> "
                     "FALLBACK_G4 (use t3_full_off_split anchors)")
        return False
    checked = 0
    for row in anchored:
        total = to_int(row["restore_complete_ns"]) - to_int(
            row["restore_start_ns"])
        seg_sum = (to_int(row["pre_prefill_restore_ns"])
                   + to_int(row["hidden_restore_ns"])
                   + to_int(row["exposed_restore_stall_ns"]))
        # 规格登记的非恒等形态（experiment/各指标测试建议.md §B）：A 类
        # 公式的 exposed 段不夹锚点窗，锚点窗整体落在 prefill 结束之后
        # （rs>pe）时三段和 = rc−pe ≠ rc−rs——行内判别为 pre 与 hidden
        # 同 0（turn-0 的 prefill→decode 迁移腿即此形）。公式按 §2.2
        # 红线冻结不改，本层对此形态豁免恒等断言、只记 note。
        if (to_int(row["pre_prefill_restore_ns"]) == 0
                and to_int(row["hidden_restore_ns"]) == 0):
            notes.append(
                f"G4: {row['request_id']} anchor window entirely after "
                f"prefill (pre=hidden=0, spec-documented rs>pe shape): "
                f"identity exempt (seg_sum={seg_sum}, window={total})")
            continue
        check(seg_sum == total,
              f"G4: {row['request_id']} segments {seg_sum} != restore total "
              f"{total}")
        notes.append(
            f"G4: {row['request_id']} three segments {seg_sum} == restore "
            f"total {total} (hand OK)")
        checked += 1
    if checked == 0:
        notes.append("G4: every anchored row is the spec-documented rs>pe "
                     "shape; no identity asserted -> FALLBACK_G4")
        return False
    return True


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_root", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="slo_params_manifest override (B4 pending -> provisional file)")
    args = parser.parse_args()
    manifest_path = (args.manifest or Path(__file__).resolve().parents[1]
                     / "slo_params_manifest.json")
    repo = args.repo_root.resolve()
    wl = repo / "sh_test_mesh/workload/llama2_7b_inference"
    traces = wl / "traces"
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    prefill_instances, _total = instance_count(repo)
    scenarios = {
        "golden_g1": g1_rows(),
        "golden_g2": g2_rows(prefill_instances),
        "golden_g3": g3_rows(),
        "golden_g4": g4_rows(),
    }
    notes: list[str] = []
    failures: list[str] = []
    status: dict[str, str] = {}
    try:
        for name, (queue_rows, sidecar_rows) in scenarios.items():
            queue_path = traces / f"{name}_request_queue_recompute.csv"
            sidecar_path = traces / f"{name}_canonical_sidecar.csv"
            write_csv(queue_path, QUEUE_COLUMNS, queue_rows)
            write_provenance_sidecar(repo, queue_path)
            write_csv(sidecar_path, SIDECAR_COLUMNS, sidecar_rows)
            rel = (f"llama2_7b_inference/traces/{queue_path.name}")
            try:
                run_dir = run_scenario(repo, work, name, rel)
                if name == "golden_g1":
                    scenario_g1(run_dir, notes)
                elif name == "golden_g2":
                    scenario_g2(run_dir, notes, expected_overflow=3,
                                manifest_path=manifest_path)
                elif name == "golden_g3":
                    scenario_g3(run_dir, notes)
                else:
                    scenario_g4(run_dir, notes)
                status[name] = "PASS"
                print(f"[golden] {name}: PASS")
            except (GoldenCheckError, subprocess.CalledProcessError) as error:
                status[name] = f"FAIL: {error}"
                failures.append(f"{name}: {error}")
                print(f"[golden] {name}: FAIL: {error}", file=sys.stderr)
    finally:
        # 指针恢复必须先于任何逃逸路径：except 只捕两类已知异常，未预期
        # 异常（OSError/KeyError/…）会穿出循环——恢复放 finally，仓内
        # trace_config.csv 不得遗留指向 golden 队列（SerDes 改造方案阶段0
        # golden 锚授权补齐；本仓深挖清单 §5.13 亦登记该清理逻辑冗余）。
        set_pointer(wl / "trace_config.csv", PLACEHOLDER_REL)
        # R4-4b 同款清理（移植自 wscllm 仓，2026-10-07）：golden_* 草稿
        # 三件套（队列 / canonical sidecar / provenance）写在仓内 traces/
        # ——遗留队列带着 VALID provenance 边车，会被误当成真实物化输入。
        # 随指针还原一并删除。
        for stale_golden in traces.glob("golden_g*_request_queue_recompute.csv"):
            stale_golden.unlink(missing_ok=True)
            Path(str(stale_golden) + ".provenance.json").unlink(missing_ok=True)
            stale_golden.with_name(
                stale_golden.name.replace(
                    "_request_queue_recompute.csv", "_canonical_sidecar.csv")
            ).unlink(missing_ok=True)
        for stale in list((repo / "sh_test_mesh/generated").glob(
                "llama2_7b_inference_54npus_*")) + \
                list((repo / "sh_test_mesh/generated").glob(
                    "llama2_7b_wsc_llm_inference_54npus_*")):
            shutil.rmtree(stale)
    report = {"status": status, "notes": notes}
    (work / "golden_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for note in notes:
        print(f"[golden-note] {note}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
