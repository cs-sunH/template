#!/usr/bin/env python3
"""train_a1_eviction_fixture.py -- 拼 batch 改造 A1 合成微 trace 夹具
(2026-08-22;§7.1 A1 层 / §7.7 最小测试集的 eviction 合成覆盖装置)。

trace 小窗不保证触发 KV 压力(5s 窗实测零逐出)——本装置用小容量
硬件(4 GiB/NPU:权重分片 ~2.34 GiB + ~1.9 GiB KV ≈ 22K token)+ 12
个手算可核对的合成请求(3 会话多 turn、异长 decode、交错到达)驱动
完整在线链路,断言:

  1. §7.3 不变量:decode 成员-迭代总数 == Σ decode_length(3,588);
     chunk 总数 == Σ ceil(prefill/512)(61);
  2. 逐出实际发生(准入族 history/prefill/decode evictions 非空;主动
     驱逐退役后完成边界恒零逐出)且运行 PASS;
  3. 全部 12 请求完成(交付 reasons 四类各 12)。

KV 逐出并行化(2026-09-13)追加场景 2"逐出后立刻再到达":V 被 U 的
准入整会话外迁(remote_store 旁路支链)后 200ms 回来,断言 REMOTE 全量
回迁(remote_load,跨实例→store→restore 前递补边路径)真实发生且运行
PASS——专项压 store→restore 排序(主方案 §3.3)。

编排:合成输入落 /tmp → 备份 trace_config.csv → 临时指向合成队列/
小容量硬件(备份→改→还原→diff 验证)→ plan_materializer → 官方
runner → 断言 → 还原 + 清理生成目录。幂等可重跑;二进制缺失即
fail-closed。

用法:bash sh_test_mesh/workload/llama2_7b_inference/online/verify/\
train_a1_eviction_fixture.py   (或 python3 同路径)
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.realpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../.."))
WL = os.path.join(REPO, "sh_test_mesh/workload/llama2_7b_inference")
TRACE_CONFIG = os.path.join(WL, "trace_config.csv")
RUNNER = os.path.join(REPO, "sh_test_mesh/run_scripts/run_online_strategy.sh")
BINARY = os.path.join(
    REPO, "build/astra_analytical/build_congestion_aware/bin/"
          "AstraSim_Analytical_Congestion_Aware_Online")
HARDWARE_SRC = os.path.join(
    REPO, "sh_test_mesh/hardware/face_case5_config_c.json")

# 12 请求手算期望(§7.3 不变量断言用):Σ decode=3588,Σ chunk=61。
REQUESTS = [
    ("session_A", 0, "A_r0", 4000, 300, 0, None),
    ("session_A", 1, "A_r1", 1500, 200, None, 2000000),
    ("session_A", 2, "A_r2", 1200, 150, None, 1500000),
    ("session_B", 0, "B_r0", 6000, 900, 500000, None),
    ("session_B", 1, "B_r1", 2000, 60, None, 3000000),
    ("session_C", 0, "C_r0", 600, 5, 1000000, None),
    ("session_C", 1, "C_r1", 600, 3, None, 1000000),
    ("session_D", 0, "D_r0", 3000, 400, 1500000, None),
    ("session_E", 0, "E_r0", 2500, 250, 2000000, None),
    ("session_F", 0, "F_r0", 2000, 500, 2500000, None),
    ("session_G", 0, "G_r0", 5000, 120, 3000000, None),
    ("session_H", 0, "H_r0", 1000, 700, 3500000, None),
]
EXPECT_DECODE = sum(r[4] for r in REQUESTS)
EXPECT_CHUNKS = sum(-(-r[3] // 512) for r in REQUESTS)

# 场景 2(KV 逐出并行化 2026-09-13):V(12.1K token 终态)先驻留;
# U(18.5K token 终态,turn-0 只能落在唯一非边缘实例,与 V 同址;半逐 V
# 后余量 8.49 GiB 仍不足 U 的 9.03 GiB)到达后把闲置的 V 整会话外迁
# (remote_store 两段式:suffix + full fallback);V 200ms 后回来,经
# REMOTE 全量回迁(remote_load,跨实例→store→restore 前递补边路径)
# 恢复。队列行按会话连续排列(V 两 turn 相邻,U 随后——reader 对非连续
# 会话块 fail-closed)。Σ decode=250,Σ chunk=61。
RETURN_REQUESTS = [
    ("session_V", 0, "V_r0", 12000, 100, 0, None),
    ("session_V", 1, "V_r1", 500, 50, None, 200000000),
    ("session_U", 0, "U_r0", 18400, 100, 100000000, None),
]
RETURN_EXPECT_DECODE = sum(r[4] for r in RETURN_REQUESTS)
RETURN_EXPECT_CHUNKS = sum(-(-r[3] // 512) for r in RETURN_REQUESTS)

def fail(message: str) -> None:
    print(f"[a1-eviction-fixture] FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def _run_scenario(work, queue, requests, expect_decode, expect_chunks,
                  *, require_remote_restore_for=None):
    """跑一个合成场景并断言(PASS / 不变量 / 逐出覆盖 / 可选回迁)。"""
    plan_and_run(queue, work)
    run_dir = os.path.join(work, "run")
    ledger = [json.loads(line) for line in open(
        os.path.join(run_dir, "results/train_ledger.jsonl"))]
    member_iters = sum(r["member_iterations"] for r in ledger)
    chunks = sum(r["prefill_chunks"] for r in ledger)
    if member_iters != expect_decode:
        fail(f"decode member-iterations {member_iters} != {expect_decode}")
    if chunks != expect_chunks:
        fail(f"chunk count {chunks} != {expect_chunks}")
    evictions = 0
    restored_sessions = set()
    store_sessions = set()
    for line in open(os.path.join(
            run_dir, "results/online_decision_log.jsonl")):
        decision = json.loads(line).get("decision", {})
        # 主动驱逐退役后 completion_evictions 恒空；此处断言的是
        # 准入族逐出（history/prefill/decode）。
        for key in ("history_evictions", "prefill_evictions",
                    "decode_evictions", "completion_evictions"):
            evictions += len(decision.get(key) or ())
        for key in ("history_evictions", "prefill_evictions",
                    "decode_evictions"):
            for transfer in decision.get(key) or ():
                if transfer.get("kind") == "remote_store":
                    store_sessions.add(transfer.get("session_id"))
        transfer = decision.get("history_transfer")
        if isinstance(transfer, dict) and transfer.get(
                "kind") == "remote_load":
            restored_sessions.add(transfer.get("session_id"))
    if evictions == 0:
        fail("eviction path not exercised (§7.7 requires synthetic "
             "coverage)")
    if require_remote_restore_for is not None:
        for session_id in require_remote_restore_for:
            if session_id not in store_sessions:
                fail(f"scenario-2 expected session {session_id} to be "
                     "evicted via remote_store")
            if session_id not in restored_sessions:
                fail(f"scenario-2 expected session {session_id} to be "
                     "restored via remote_load (store->restore ordering "
                     "coverage)")
    print(f"[a1-eviction-fixture] PASS: trains={len(ledger)} "
          f"member_iters={member_iters} chunks={chunks} "
          f"evictions={evictions} "
          f"restored_sessions={sorted(restored_sessions) or '-'}")
    return 0


def plan_and_run(queue, work) -> None:
    """重生成 plan 目录并跑官方 runner(run_dir = work/run)。"""
    gen_root = os.path.join(REPO, "sh_test_mesh/generated")
    for name in os.listdir(gen_root):
        if name.startswith("llama2_7b_inference_54npus_plan_"):
            shutil.rmtree(os.path.join(gen_root, name))
    subprocess.run(
        [sys.executable, "plan_materializer.py"], cwd=WL,
        check=True, capture_output=True)
    run_dir = os.path.join(work, "run")
    result = subprocess.run(
        ["bash", RUNNER, run_dir, queue],
        capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        fail(f"runner failed: {result.stdout[-2000:]} {result.stderr}")


def write_queue(path, requests) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "session_id", "turn_index", "request_id", "prefill_length",
            "decode_length", "session_arrival_time_ns",
            "inter_request_interval_ns", "next_trigger_type",
            "description"])  # sh_3.0 队列 schema 含 next_trigger_type 列
        for row in requests:
            writer.writerow(
                list(row)[:7] + ["tool", "A1 synthetic micro-trace"])


def main() -> int:
    if not os.path.exists(BINARY):
        fail(f"online binary missing: {BINARY} (build first)")
    work = tempfile.mkdtemp(prefix="/tmp/wsc_a1_fixture.")
    hardware = json.load(open(HARDWARE_SRC, encoding="utf-8"))
    hardware["local-hbm"]["capacity-profiles"]["a1-eviction-4gib"] = {
        "bytes": 4294967296,
        "label": "A1 synthetic small capacity (eviction fixture)",
        "note": "Synthetic 4 GiB/NPU capacity (weights ~2.34 GiB/rank + "
                "~1.9 GiB KV) to force KV eviction in the A1 fixture.",
    }
    hardware_path = os.path.join(work, "hw_small.json")
    json.dump(hardware, open(hardware_path, "w", encoding="utf-8"),
              indent=2)

    backup = open(TRACE_CONFIG, encoding="utf-8").read()
    config = backup
    config = config.replace(
        "config,hardware_config,hardware/face_case5_config_c.json",
        f"config,hardware_config,{hardware_path}")
    config = config.replace(
        "config,local_hbm_capacity_profile,validation-160gib",
        "config,local_hbm_capacity_profile,a1-eviction-4gib")
    # D-clear (2026-09-05)：kv_reserve_context_tokens 配置链保留（审计
    # 口径）但运行期零消费者，按 CSV 原值使用即可，无需替换（R8）。
    lines = [line for line in config.splitlines()
             if line.startswith("config,request_queue_csv,")]
    if len(lines) != 1:
        fail("trace_config must carry exactly one request_queue_csv line")
    original_pointer = lines[0]
    try:
        queue = os.path.join(work, "queue.csv")
        write_queue(queue, REQUESTS)
        open(TRACE_CONFIG, "w", encoding="utf-8").write(
            config.replace(
                original_pointer,
                f"{original_pointer.split(',', 2)[0]},"
                f"{original_pointer.split(',', 2)[1]},{queue},,,"
                "A1 fixture temporary pointer"))
        _run_scenario(
            work, queue, REQUESTS, EXPECT_DECODE, EXPECT_CHUNKS)

        # 场景 2：逐出后立刻再到达（KV 逐出并行化 store→restore 排序）。
        queue2 = os.path.join(work, "queue_return.csv")
        write_queue(queue2, RETURN_REQUESTS)
        open(TRACE_CONFIG, "w", encoding="utf-8").write(
            config.replace(
                original_pointer,
                f"{original_pointer.split(',', 2)[0]},"
                f"{original_pointer.split(',', 2)[1]},{queue2},,,"
                "A1 fixture temporary pointer (scenario 2)"))
        _run_scenario(
            work, queue2, RETURN_REQUESTS, RETURN_EXPECT_DECODE,
            RETURN_EXPECT_CHUNKS,
            require_remote_restore_for=["session_V"])
        return 0
    finally:
        open(TRACE_CONFIG, "w", encoding="utf-8").write(backup)
        gen_root = os.path.join(REPO, "sh_test_mesh/generated")
        for name in os.listdir(gen_root):
            if name.startswith("llama2_7b_inference_54npus_plan_"):
                shutil.rmtree(os.path.join(gen_root, name))
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
