#!/usr/bin/env python3
"""B3-5 T2 golden case 运行版（真仿真断言，G1-G4）。

用 hand-craft 队列（8 列 + canonical sidecar 15 列，落在仓内
workload/llama2_7b_inference/traces/ 下，命名以
``_request_queue_recompute.csv`` 结尾以便 sidecar 兄弟 join）逐场景跑
run_online_strategy.sh（detail=full、SH_FIRST_TOKEN_SPLIT=1），对
request_metrics.csv / online_decision_log / train_ledger / kv_cache_adapter
输出做手算断言：

  G1 单请求无竞争：queue_ns=0；e2e=queue+prefill+gap+decode（整数恒等）；
     first_token−arrival = queue+prefill+gap+首步迭代时长，其中
     首步迭代时长 = decode_ns/decode_length（单请求批恒定 → 迭代均匀）。
  G2 同刻突发排队（请求数 > 实例数）：溢出请求 queue_ns>0 且
     queue_ns≈占位者 prefill；分解和恒等；load_imbalance.py 输出与
     独立手算重建（admission=prefill tick、instance=decode_instance、
     drain=train_ledger drains tick、桶长=manifest）一致。
  G3 单 session 两轮 human/tool：T_session = turn0.e2e + interval +
     turn1.e2e（到达=父完成+interval 闭环恒等）；request_type 逐行透传；
     第二轮 kv_hit_state 由 decision history_action 手推并与
     kv_cache_adapter --reconcile 输出一致。
  G4 已知 restore：迷你 3 轮 session 若产生 history NOC_MIGRATE 锚点
     （memory_anchor code7），对锚点请求核对三段和 = 总 restore 时长；
     无锚点则打印 FALLBACK_G4（改用 t3_full_off_split 真实锚点核对，
     由调用方执行）。
  R1 relevant_distributed（B4a/2026-09-02 编写，场景执行放 B4b）：同刻
     突发 8 请求使 r0/r6 共享同一 P→D 对；占位者 A 全段落 D，溢出请求
     B 的 prefill 段吃满 D 剩余后溢出 P-piece——对决策日志逐条核对
     kv_placement 位置表（区间划分/decode 段钉 D/prefill 段 token 序
     遵循全序）、3100 逐 route 字节与 Σ3100+P-piece 守恒、3300 逐 route
     bytes = p_m×R_{m,s} 与 participation 合计 = decode_length。断言函数
     的单元级自测 = --selftest（算例 B 形状 fixture，不跑仿真）。

Usage: python3 run_golden_live.py <repo_root> --work-dir DIR
       python3 run_golden_live.py --selftest
退出码 0 = 全过；非 0 = 有断言失败（stderr 列明细）。
"""

from __future__ import annotations

import argparse
import csv
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
                 queue_rel: str, *, runner: str = "run_online_strategy.sh",
                 env_extra: dict | None = None,
                 first_token_split: bool = True) -> Path:
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
    env["SH_FIRST_TOKEN_SPLIT"] = "1" if first_token_split else "0"
    if env_extra:
        env.update(env_extra)
    subprocess.run(
        ["bash", str(repo_root / "sh_test_mesh/run_scripts" / runner),
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


RELEVANT_QUEUE_SPEC = {
    # request_id: (prefill_length, decode_length)——含义见 g5_rows docstring
    "g5_s0_r0": (1_700_000, 16),  # A 占位者（D 容量填充）
    "g5_s1_r0": (200, 8),
    "g5_s2_r0": (200, 8),
    "g5_s3_r0": (200, 8),
    "g5_s4_r0": (200, 8),
    "g5_s5_r0": (200, 8),
    "g5_s6_r0": (200_000, 16),  # B 溢出到 P 的请求
    "g5_s7_r0": (200, 8),
}


def g5_rows() -> tuple[list[dict], list[dict]]:
    """relevant_distributed 变体小场景（R1：一个溢出到 P 的请求）。

    同刻突发 8 请求（6 个 prefill 实例 → 第 1 波 r0..r5 铺满全部 P、
    第 2 波 r6/r7 落回最小排队深度的 P0/P1——select_prefill_instance 的
    ordering_key=(request_count, instance_index)，故 r0 与 r6 共享同一
    P→D 对，静态映射 P1→d0/P4→d0）。

    容量口径（B4b/2026-09-02 按真 run 的 journal 权重行校准，替换 B4a
    原写的 ≈149k tok 误估——实测每路容量 171,798,691,840 B、rank0/1
    权重 2,335,890,091 B（5 头路 2,201,655,979 B），绑定约束 = rank0/1
    整头 shard 98,304 B/tok → 每实例空域 = 1,723,921 tok）：
      - r0 = 占位者 A（prefill 1,700,000 tok）：几乎吃满 d0 空域
        （1,723,921 tok，②整段暂存 1.7M ≤ P1 空域裕量 ≈23.9k tok）→
        位置表全 D（scatter_remote 单段 + decode_local）、3100 整段散
        布、零 3300；
      - r6 = 溢出请求 B（prefill 200,000 tok）：d0 在 A 驻留 + r1 填充
        （208 tok）后剩余 ≈23.7k tok < 200k → D piece 吃满剩余、其余
        ≈176k tok 溢出为 P-piece（prefill_stay）——目标断言对象；200k
        ≤ P1 空域（条件②整段暂存裕量充足）。注意 B 的②当前档在 A 的
        staging 释放前必失败（P1 同刻已暂存 A 整段 1.7M，剩余 < 200k
        staging 需求——静态 P 共享下结构性必然），故 B 先 FCFS 阻塞、
        A drain 释放 staging 后重准入：本场景顺带活体验证背压→释放→
        重准入链路；
      - r1..r5/r7 = 填充小请求（200 tok）：铺满其余 P 实例使其对 r6 无
        吸引力，自身全落各自 D；r7（P4→d0）在 B 之后 d0 已满 → 阻塞至
        B 终轮释放后重准入完成。
    """
    spec = RELEVANT_QUEUE_SPEC
    queue, sidecar = [], []
    for index, (request_id, (prefill, decode)) in enumerate(spec.items()):
        queue.append({
            "session_id": f"g5_s{index}", "turn_index": 0,
            "request_id": request_id, "prefill_length": prefill,
            "decode_length": decode,
            "session_arrival_time_ns": 2_000_000,
            "inter_request_interval_ns": "",
            "description": "golden R1 relevant_distributed overflow",
        })
        sidecar.append({
            "request_id": request_id, "turn_index": 0,
            "session_id": f"g5_s{index}", "raw_prefix_tokens": 0,
            "raw_new_prefill_tokens": prefill,
            "effective_prompt_tokens": prefill, "prefill_length": prefill,
            "decode_length": decode,
            "session_arrival_time_ns": 2_000_000,
            "inter_request_interval_ns": "", "prefix_mode": "recompute",
            "digest": "golden", "human_time_ns": "", "tool_time_ns": "",
            "request_type": "tool",
        })
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
    check(to_int(row["queue_ns"]) == 0,
          f"G1: queue_ns {row['queue_ns']} != 0 (no contention)")
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
    queued = [r for r in rows if to_int(r["queue_ns"]) > 0]
    notes.append(
        f"G2: {len(rows)} requests, overflow expected {expected_overflow}, "
        f"queued observed {len(queued)}")
    check(len(queued) >= 1, "G2: no request queued (burst insufficient)")
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
        for req in entry.get("drains", []) or []:
            drain.setdefault(req, entry["tick"])
    spans = []
    for rid, adm in admission.items():
        if rid in decode_inst and rid in drain:
            spans.append((decode_inst[rid], adm, drain[rid]))
    t0 = min(span[1] for span in spans)
    t1 = max(span[2] for span in spans)
    horizon = t1 - t0
    n_buckets = max(1, math.ceil(horizon / bucket_ns))
    # Count-based bucketing mirroring the tool's rounding convention.
    covered: dict[int, int] = {}
    for inst, adm, dr in spans:
        counts = covered.setdefault(inst, 0)
        first = max(0, int((adm - t0) // bucket_ns))
        last = min(n_buckets, int((dr - t0) // bucket_ns) + 1)
        for b in range(first, last):
            bucket_start = t0 + b * bucket_ns
            bucket_end = bucket_start + bucket_ns
            if dr > bucket_start and adm < bucket_end:
                counts += 1
        covered[inst] = counts
    means = [covered[inst] * bucket_ns / horizon for inst in covered]
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
    # kv_hit_state hand derivation from decision history_action.
    decisions = read_jsonl(run_dir / "results/online_decision_log.jsonl")
    actions = {}
    for entry in decisions:
        if entry.get("kind") == "prefill":
            actions[entry["request_id"]] = entry["decision"].get(
                "history_action")
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
        expected = mapping.get(action, "not_supported")
        got = adapter_states.get(rid)
        check(got == expected,
              f"G3: kv_hit_state {rid}: adapter {got} != hand {expected} "
              f"(history_action={action})")
        notes.append(f"G3: {rid} history_action={action} -> {got} (hand OK)")


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
    for row in anchored:
        total = to_int(row["restore_complete_ns"]) - to_int(
            row["restore_start_ns"])
        seg_sum = (to_int(row["pre_prefill_restore_ns"])
                   + to_int(row["hidden_restore_ns"])
                   + to_int(row["exposed_restore_stall_ns"]))
        check(seg_sum == total,
              f"G4: {row['request_id']} segments {seg_sum} != restore total "
              f"{total}")
        notes.append(
            f"G4: {row['request_id']} three segments {seg_sum} == restore "
            f"total {total} (hand OK)")
    return True


# ---------------------------------------------------------------------------
# relevant_distributed 变体场景（B4a/2026-09-02 编写；场景执行放 B4b）
# ---------------------------------------------------------------------------

RELEVANT_RUNNER = "run_online_strategy_relevant.sh"
# llama2-7b KV 字节换算权威（与 wsc_llm_scheduler.kv_cache_bytes_for_tokens
# 同式、test_wsc_llm_scheduler.py 冻结值；README §WP9 同值）：
# 2×layers×hidden×bpe = 2×32×4096×2 = 524288 B/token/实例；整头分片
# [6,6,5,5,5,5]/32 heads（partition_values_exact，余数在低 rank）。
KV_BYTES_PER_TOKEN = 524288
TP_HEADS = (6, 6, 5, 5, 5, 5)
TP_DEGREE = 6
KV_BYTES_PER_HEAD_PER_TOKEN = KV_BYTES_PER_TOKEN // sum(TP_HEADS)  # 16384


def kv_instance_bytes(tokens: int) -> int:
    """实例级 KV 字节（K+V、全层、全部 32 头）。"""
    return KV_BYTES_PER_TOKEN * tokens


def kv_shard_bytes(tokens: int, relative_rank: int) -> int:
    """整头 shard 逐 rank 字节（= kv_cache_shard_bytes_for_tokens 同式）。"""
    return KV_BYTES_PER_HEAD_PER_TOKEN * TP_HEADS[relative_rank] * tokens


def relevant_parse_decisions(decisions: list[dict]):
    """从决策日志收集 relevant_distributed 的位置表与三族路由行。

    holder 字段名以 B1 KVPlacement 契约（pieces/instance_index/token_start/
    token_end/tier）与 B2 发射函数返回的 routes 结构（category ∈
    {1000,3100,3300}，3300 行自带 request_id）为准；B4b/2026-09-02 已按
    B3 实际决策行对齐：kv_placement/kv_scatter/kv_remote_reads 的 routes
    均在 decision 顶层，1000 族在 prefill 行 decision.history_pull_routes
    （B3 序列化实际键，含 noc_hops）。
    返回 (placements: rid -> holder dict, routes: category -> [(rid, row)])。
    """
    placements: dict[str, dict] = {}
    routes: dict[int, list] = {1000: [], 3100: [], 3300: []}
    for entry in decisions:
        decision = entry.get("decision") or {}
        rid = entry.get("request_id") or "NA"
        for holder in (decision, decision.get("kv_placement")):
            if (isinstance(holder, dict)
                    and isinstance(holder.get("pieces"), list)
                    and holder["pieces"]):
                placements.setdefault(rid, holder)
        route_lists = []
        top = decision.get("routes")
        if isinstance(top, list):
            route_lists.append(top)
        for key in ("history_pull_routes", "kv_scatter", "kv_remote_reads",
                    "kv_history_pull", "history_pull"):
            nested = decision.get(key)
            if isinstance(nested, list):
                rows = nested
            else:
                rows = (nested.get("routes")
                        if isinstance(nested, dict) else None)
            if isinstance(rows, list):
                route_lists.append(rows)
        for rows in route_lists:
            for row in rows:
                if (isinstance(row, dict)
                        and row.get("category") in routes
                        and isinstance(row.get("bytes"), int)):
                    routes[row["category"]].append(
                        (row.get("request_id") or rid, row))
    return placements, routes


def assert_placement_invariants(pieces: list, *, prefill_context_tokens: int,
                                decode_length: int, label: str,
                                notes: list[str]) -> tuple[list[dict], int]:
    """位置表不变量（总文档 §2.2/§3.1，裁决 #7/#8）：

    - 区间划分恰覆盖 [0, prefill_ctx+decode)（排序后无缺口/重叠）；
    - decode 段 [prefill_ctx, final) 单块、tier=decode_local（钉 D）；
    - prefill 段 token 序遵循全序 tier D=0 / P=1 / 中间 die≥2 非降
      （贪心吸收序 = token 序；line 拓扑算例 B 的字面序 (2,0,1) 即本律
      的三实例实例化，见 scenario_relevant_selftest）。
    返回 (prefill 段 pieces, decode 实例号)。
    """
    final_tokens = prefill_context_tokens + decode_length
    ordered = sorted(pieces, key=lambda p: int(p["token_start"]))
    cursor = 0
    for piece in ordered:
        start, end = int(piece["token_start"]), int(piece["token_end"])
        check(start == cursor,
              f"{label}: piece interval [{start},{end}) breaks coverage at "
              f"{cursor} (gap/overlap)")
        cursor = end
    check(cursor == final_tokens,
          f"{label}: pieces cover [0,{cursor}) != final {final_tokens}")
    decode_pieces = [p for p in ordered
                     if int(p["token_start"]) >= prefill_context_tokens]
    check(len(decode_pieces) == 1,
          f"{label}: decode segment split across {len(decode_pieces)} pieces")
    decode_piece = decode_pieces[0]
    check(decode_piece.get("tier") == "decode_local",
          f"{label}: decode piece tier {decode_piece.get('tier')!r} != "
          "decode_local (decode 段硬钉 D)")
    decode_instance = int(decode_piece["instance_index"])
    prefill_pieces = [p for p in ordered
                      if int(p["token_end"]) <= prefill_context_tokens]
    check(prefill_pieces and int(prefill_pieces[0]["token_start"]) == 0,
          f"{label}: prefill segment does not start at token 0")

    def tier_rank(piece) -> int:
        if int(piece["instance_index"]) == decode_instance:
            return 0
        if piece.get("tier") == "prefill_stay":
            return 1
        return 2  # scatter_remote on intermediate die

    tiers = [tier_rank(p) for p in prefill_pieces]
    check(tiers == sorted(tiers),
          f"{label}: prefill-segment token order violates the placement "
          f"total order (tiers {tiers}; expected non-decreasing D<P<mid)")
    sequence = [int(p["instance_index"]) for p in prefill_pieces]
    notes.append(
        f"{label}: placement pieces token-order instances={sequence} "
        f"tiers={tiers} decode pinned on instance {decode_instance} "
        f"(order law OK)")
    return prefill_pieces, decode_instance


def assert_scatter_conservation(routes_3100: list, prefill_pieces: list, *,
                                prefill_context_tokens: int,
                                prefill_instance_index: int | None,
                                label: str, notes: list[str]) -> int:
    """3100 逐条核对 + 守恒（总文档 §2.1/裁决 #9，golden 门）：

    - 实例级守恒：Σ3100 字节 + P-piece(prefill_stay) 字节 =
      kv_cache_bytes_for_tokens(prefill_ctx)；
    - 逐 route 行：bytes = owner piece token 数 × 整头 shard（按
      relative_shard 精确，与 B2 emit_piece_scatter 的 routes 同式）；
      per-owner 聚合 = owner piece 实例级字节；source = 本请求 P。
    返回 prefill_stay token 数。
    """
    stay_tokens = sum(
        int(p["token_end"]) - int(p["token_start"])
        for p in prefill_pieces if p.get("tier") == "prefill_stay")
    owner_tokens: dict[int, int] = {}
    for piece in prefill_pieces:
        if piece.get("tier") == "prefill_stay":
            continue
        owner = int(piece["instance_index"])
        owner_tokens[owner] = (owner_tokens.get(owner, 0)
                               + int(piece["token_end"])
                               - int(piece["token_start"]))
    scatter_bytes = sum(int(row["bytes"]) for _rid, row in routes_3100)
    expected_total = kv_instance_bytes(prefill_context_tokens)
    check(scatter_bytes + kv_instance_bytes(stay_tokens) == expected_total,
          f"{label}: scatter conservation {scatter_bytes} + "
          f"{kv_instance_bytes(stay_tokens)} (P-piece) != kv(prefill_ctx) "
          f"{expected_total}")
    per_owner: dict[int, int] = {}
    for rid, row in routes_3100:
        owner = int(row["target_instance_index"])
        shard = int(row["relative_shard"])
        tokens = owner_tokens.get(owner)
        check(tokens is not None,
              f"{label}: 3100 route to instance {owner} without a matching "
              "non-stay prefill piece")
        check(int(row["bytes"]) == kv_shard_bytes(tokens, shard),
              f"{label}: 3100 route bytes {row['bytes']} != shard formula "
              f"{kv_shard_bytes(tokens, shard)} (owner {owner}, {tokens} "
              f"tokens, shard {shard})")
        if prefill_instance_index is not None:
            check(int(row["source_instance_index"]) == prefill_instance_index,
                  f"{label}: 3100 source {row['source_instance_index']} != "
                  f"prefill instance {prefill_instance_index}")
        per_owner[owner] = per_owner.get(owner, 0) + int(row["bytes"])
    for owner, nbytes in sorted(per_owner.items()):
        check(nbytes == kv_instance_bytes(owner_tokens[owner]),
              f"{label}: 3100 per-owner bytes {nbytes} != instance bytes "
              f"{kv_instance_bytes(owner_tokens[owner])} (owner {owner})")
    notes.append(
        f"{label}: scatter conservation {scatter_bytes} + stay "
        f"{kv_instance_bytes(stay_tokens)} == {expected_total}; owners "
        f"{sorted(per_owner)}; stay_tokens={stay_tokens} (hand OK)")
    return stay_tokens


def assert_remote_reads(routes_3300: list, prefill_pieces: list, *,
                        decode_length: int, decode_instance_index: int,
                        label: str, notes: list[str]) -> None:
    """3300 逐条核对：bytes = p_m × R_{m,s} 精确式（总文档 §2.1/裁决 #10）。

    - 源集合 = prefill 段非 D 实例 piece 集合（钉 D 推论：decode 段恒本地，
      每迭代远程字节为常量）；逐 route 行 bytes = participation × 整头
      shard(R_{m,s})；per (源, shard) 的 participation 合计 = decode_length
      （列车覆盖全部迭代，T_max=8 截断下逐列车求和）；
    - 总量 Σ3300 = decode_length × Σ_s R_{m,s}（实例级）。
    """
    source_tokens: dict[int, int] = {}
    for piece in prefill_pieces:
        owner = int(piece["instance_index"])
        if owner == decode_instance_index:
            continue
        source_tokens[owner] = (source_tokens.get(owner, 0)
                                + int(piece["token_end"])
                                - int(piece["token_start"]))
    read_bytes = sum(int(row["bytes"]) for _rid, row in routes_3300)
    remote_total = sum(source_tokens.values())
    expected_total = decode_length * kv_instance_bytes(remote_total)
    check(read_bytes == expected_total,
          f"{label}: remote reads total {read_bytes} != decode_length "
          f"({decode_length}) x R {kv_instance_bytes(remote_total)} = "
          f"{expected_total}")
    rows_by_source: dict[int, list] = {}
    for rid, row in routes_3300:
        source = int(row["source_instance_index"])
        check(source in source_tokens,
              f"{label}: 3300 route from {source} not in the placement's "
              "non-D source set")
        shard = int(row["relative_shard"])
        participation = int(row["participation"])
        check(int(row["bytes"])
              == participation * kv_shard_bytes(source_tokens[source], shard),
              f"{label}: 3300 bytes {row['bytes']} != p x R "
              f"{participation} x "
              f"{kv_shard_bytes(source_tokens[source], shard)} "
              f"(source {source}, shard {shard})")
        rows_by_source.setdefault(source, []).append((shard, participation))
    for source, rows in sorted(rows_by_source.items()):
        for shard in range(TP_DEGREE):
            total_p = sum(p for s, p in rows if s == shard)
            check(total_p == decode_length,
                  f"{label}: source {source} shard {shard} participation "
                  f"sum {total_p} != decode_length {decode_length}")
    notes.append(
        f"{label}: remote reads total {read_bytes} == {decode_length} x "
        f"{kv_instance_bytes(remote_total)} over sources "
        f"{sorted(source_tokens)} (exact p_m x R_{{m,s}} OK)")


def scenario_relevant(run_dir: Path, notes: list[str]) -> None:
    """R1：relevant_distributed 溢出到 P 的请求逐条手算断言（B4b 执行）。"""
    rows = read_request_rows(run_dir)
    check(len(rows) == len(RELEVANT_QUEUE_SPEC),
          f"R1: {len(rows)} rows != {len(RELEVANT_QUEUE_SPEC)}")
    for row in rows:
        check(row["terminal_status"] == "completed",
              f"R1: {row['request_id']} terminal_status "
              f"{row['terminal_status']}")
        assert_row_identity(row, f"R1.{row['queue_index']}", notes)
    # run 头/配置一致（轻校验：run 头须登记本变体——决策日志里出现
    # relevant_distributed 字样；字段位置/名待 B4b 与 B3 run 头对齐）。
    decision_path = run_dir / "results/online_decision_log.jsonl"
    check(decision_path.is_file(), "R1: online_decision_log.jsonl missing")
    log_text = decision_path.read_text(encoding="utf-8")
    check("relevant_distributed" in log_text,
          "R1: run header/config does not register relevant_distributed")
    placements, routes = relevant_parse_decisions(
        read_jsonl(decision_path))
    stay_by_request: dict[str, int] = {}
    for request_id, (prefill_ctx, decode_len) in RELEVANT_QUEUE_SPEC.items():
        holder = placements.get(request_id)
        check(holder is not None,
              f"R1: no kv_placement pieces row for {request_id}")
        pieces = holder["pieces"]
        check(all(isinstance(p, dict) for p in pieces),
              f"R1: {request_id} pieces are not dicts (schema drift)")
        prefill_pieces, decode_instance = assert_placement_invariants(
            pieces, prefill_context_tokens=prefill_ctx,
            decode_length=decode_len, label=f"R1.{request_id}", notes=notes)
        prefill_instance = holder.get("prefill_instance_index")
        prefill_instance = (int(prefill_instance)
                            if prefill_instance is not None else None)
        stay_by_request[request_id] = assert_scatter_conservation(
            [pair for pair in routes[3100] if pair[0] == request_id],
            prefill_pieces, prefill_context_tokens=prefill_ctx,
            prefill_instance_index=prefill_instance,
            label=f"R1.{request_id}", notes=notes)
        assert_remote_reads(
            [pair for pair in routes[3300] if pair[0] == request_id],
            prefill_pieces, decode_length=decode_len,
            decode_instance_index=decode_instance,
            label=f"R1.{request_id}", notes=notes)
    # 溢出请求唯一性：仅 B（g5_s6_r0）应有 P-piece（A 全段落 D、填充小请求
    # 全落各自 D——量级依据见 g5_rows docstring）。
    overflow = [rid for rid, stay in stay_by_request.items() if stay > 0]
    check(overflow == ["g5_s6_r0"],
          f"R1: requests with P-piece {overflow} != ['g5_s6_r0'] "
          "(D-capacity drift? see g5_rows docstring for the sizing math)")
    # 本场景全部 turn-0：无历史拉回（1000 族零边）。
    check(not routes[1000],
          f"R1: unexpected history-pull routes on a turn-0-only scenario "
          f"({len(routes[1000])} rows)")
    notes.append(f"R1: overflow request g5_s6_r0 stay_tokens="
                 f"{stay_by_request['g5_s6_r0']} (unique, hand OK)")


def scenario_relevant_selftest(notes: list[str]) -> None:
    """断言函数单元级自测（B4a 编写期验证；不跑仿真，B4b 前即可执行）。

    fixture 按总文档 §7 算例 B 的结构构造：line 拓扑 D=2 / P=0 / 中间
    die d1=1，prefill 200 tok / decode 20 tok，prefill 段 piece
    80/60/60 tok 落 (2,0,1)（token 量级取算例 B 原值；字节按本仓
    llama2-7b 公式换算，逐 rank 整头 shard）。驱动与 live 场景完全相同
    的三个断言函数，并含两个 fail-closed 负例（篡改守恒 / 打乱全序必须
    报错）。
    """
    prefill_ctx, decode_len = 200, 20
    pieces = [
        {"instance_index": 2, "token_start": 0, "token_end": 80,
         "tier": "scatter_remote", "distance_to_decode": 0},
        {"instance_index": 0, "token_start": 80, "token_end": 140,
         "tier": "prefill_stay", "distance_to_decode": 1},
        {"instance_index": 1, "token_start": 140, "token_end": 200,
         "tier": "scatter_remote", "distance_to_decode": 2},
        {"instance_index": 2, "token_start": 200, "token_end": 220,
         "tier": "decode_local", "distance_to_decode": 0},
    ]
    # 3100：P0 → owner{2: 80 tok, 1: 60 tok}，逐 rank 整头 shard。
    routes_3100 = []
    for owner, tokens in ((2, 80), (1, 60)):
        for shard in range(TP_DEGREE):
            routes_3100.append(("b_req", {
                "category": 3100, "source_instance_index": 0,
                "target_instance_index": owner, "relative_shard": shard,
                "bytes": kv_shard_bytes(tokens, shard), "noc_hops": 1,
            }))
    # 3300：源 {0: 60 tok(stay), 1: 60 tok(中间 die)}；列车 participation
    # 8+8+4 = 20 = decode_len（T_max=8 截断的列车拆分）。
    routes_3300 = []
    for source, tokens in ((0, 60), (1, 60)):
        for participation in (8, 8, 4):
            for shard in range(TP_DEGREE):
                routes_3300.append(("b_req", {
                    "category": 3300, "request_id": "b_req",
                    "source_instance_index": source,
                    "relative_shard": shard, "participation": participation,
                    "bytes": participation * kv_shard_bytes(tokens, shard),
                    "noc_hops": 1,
                }))
    prefill_pieces, decode_instance = assert_placement_invariants(
        pieces, prefill_context_tokens=prefill_ctx, decode_length=decode_len,
        label="SELFTEST", notes=notes)
    check(decode_instance == 2, "SELFTEST: decode instance != 2")
    sequence = [int(p["instance_index"]) for p in prefill_pieces]
    check(sequence == [2, 0, 1],
          f"SELFTEST: prefill piece instance order {sequence} != (2,0,1) "
          "(总文档 §7 算例 B 的全序字面序)")
    stay = assert_scatter_conservation(
        routes_3100, prefill_pieces, prefill_context_tokens=prefill_ctx,
        prefill_instance_index=0, label="SELFTEST", notes=notes)
    check(stay == 60, f"SELFTEST: stay tokens {stay} != 60")
    assert_remote_reads(routes_3300, prefill_pieces,
                        decode_length=decode_len, decode_instance_index=2,
                        label="SELFTEST", notes=notes)
    # 负例 1：篡改一条 3100 字节 → 守恒断言必须报错。
    tampered = [pair for pair in routes_3100]
    rid, row = tampered[0]
    tampered[0] = (rid, {**row, "bytes": int(row["bytes"]) + 1})
    try:
        assert_scatter_conservation(
            tampered, prefill_pieces, prefill_context_tokens=prefill_ctx,
            prefill_instance_index=0, label="SELFTEST.neg1", notes=notes)
        raise GoldenCheckError("SELFTEST.neg1: tampered scatter accepted")
    except GoldenCheckError as error:
        check("conservation" in str(error) or "shard formula" in str(error),
              f"SELFTEST.neg1: unexpected failure {error}")
    # 负例 2：违反全序（低 token 段给 stay、高 token 段给 D——token 序
    # 与贪心吸收序相反）→ 位置表序断言必须报错。
    bad_order = [
        {"instance_index": 0, "token_start": 0, "token_end": 80,
         "tier": "prefill_stay", "distance_to_decode": 1},
        {"instance_index": 2, "token_start": 80, "token_end": 140,
         "tier": "scatter_remote", "distance_to_decode": 0},
        pieces[2], pieces[3],
    ]
    try:
        assert_placement_invariants(
            bad_order, prefill_context_tokens=prefill_ctx,
            decode_length=decode_len, label="SELFTEST.neg2", notes=notes)
        raise GoldenCheckError("SELFTEST.neg2: order-violating placement "
                               "accepted")
    except GoldenCheckError as error:
        check("total order" in str(error),
              f"SELFTEST.neg2: unexpected failure {error}")
    notes.append("SELFTEST: relevant assertion helpers verified on the "
                 "算例-B-shaped fixture (order (2,0,1), conservation, "
                 "exact p_m x R, two fail-closed negatives)")


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "repo_root", type=Path, nargs="?", default=None,
        help="wscllm repo root (not required with --selftest)")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="slo_params_manifest override (B4 pending -> provisional file)")
    parser.add_argument(
        "--selftest", action="store_true",
        help="run the relevant_distributed assertion selftest only "
        "(no simulation; B4a authoring-time verification)")
    args = parser.parse_args()
    if args.selftest:
        notes: list[str] = []
        try:
            scenario_relevant_selftest(notes)
            for note in notes:
                print(f"[golden-note] {note}")
            print("[golden] relevant selftest: PASS")
            return 0
        except GoldenCheckError as error:
            print(f"[golden] relevant selftest: FAIL: {error}", file=sys.stderr)
            return 1
    if args.repo_root is None or args.work_dir is None:
        parser.error("repo_root and --work-dir are required unless "
                     "--selftest is given")
    repo = args.repo_root.resolve()
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
        "golden_relevant": g5_rows(),
    }
    notes: list[str] = []
    failures: list[str] = []
    status: dict[str, str] = {}
    for name, (queue_rows, sidecar_rows) in scenarios.items():
        queue_path = traces / f"{name}_request_queue_recompute.csv"
        sidecar_path = traces / f"{name}_canonical_sidecar.csv"
        write_csv(queue_path, QUEUE_COLUMNS, queue_rows)
        write_csv(sidecar_path, SIDECAR_COLUMNS, sidecar_rows)
        rel = (f"llama2_7b_inference/traces/{queue_path.name}")
        try:
            if name == "golden_relevant":
                # relevant_distributed 变体：走专用 runner（KV_REMOTE_READ
                # 缺省 physical，环境可覆盖构成 A/B）；拆分保持交付缺省关。
                run_dir = run_scenario(
                    repo, work, name, rel, runner=RELEVANT_RUNNER,
                    env_extra={"KV_REMOTE_READ":
                               os.environ.get("KV_REMOTE_READ", "physical")},
                    first_token_split=False)
                scenario_relevant(run_dir, notes)
            else:
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
    # Restore the neutral pointer whatever happened.
    set_pointer(wl / "trace_config.csv", PLACEHOLDER_REL)
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
