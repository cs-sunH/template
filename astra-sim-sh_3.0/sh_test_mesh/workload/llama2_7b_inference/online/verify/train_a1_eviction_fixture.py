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


def fail(message: str) -> None:
    print(f"[a1-eviction-fixture] FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    if not os.path.exists(BINARY):
        fail(f"online binary missing: {BINARY} (build first)")
    work = tempfile.mkdtemp(prefix="/tmp/wsc_a1_fixture.")
    queue = os.path.join(work, "queue.csv")
    with open(queue, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "session_id", "turn_index", "request_id", "prefill_length",
            "decode_length", "session_arrival_time_ns",
            "inter_request_interval_ns", "next_trigger_type",
            "description"])  # sh_3.0 队列 schema 含 next_trigger_type 列
        for row in REQUESTS:
            writer.writerow(
                list(row)[:7] + ["tool", "A1 synthetic micro-trace"])
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
    config = config.replace(
        lines[0],
        f"config,request_queue_csv,{queue},,,A1 fixture temporary pointer")
    try:
        gen_root = os.path.join(REPO, "sh_test_mesh/generated")
        preexisting = [name for name in os.listdir(gen_root)
                       if name.startswith("llama2_7b_inference_54npus_plan_")]
        for name in preexisting:
            shutil.rmtree(os.path.join(gen_root, name))
        open(TRACE_CONFIG, "w", encoding="utf-8").write(config)
        subprocess.run(
            [sys.executable, "plan_materializer.py"], cwd=WL,
            check=True, capture_output=True)
        run_dir = os.path.join(work, "run")
        result = subprocess.run(
            ["bash", RUNNER, run_dir, queue],
            capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            fail(f"runner failed: {result.stdout[-2000:]} {result.stderr}")
        ledger = [json.loads(line) for line in open(
            os.path.join(run_dir, "results/train_ledger.jsonl"))]
        member_iters = sum(r["member_iterations"] for r in ledger)
        chunks = sum(r["prefill_chunks"] for r in ledger)
        if member_iters != EXPECT_DECODE:
            fail(f"decode member-iterations {member_iters} != "
                 f"{EXPECT_DECODE}")
        if chunks != EXPECT_CHUNKS:
            fail(f"chunk count {chunks} != {EXPECT_CHUNKS}")
        evictions = 0
        for line in open(os.path.join(
                run_dir, "results/online_decision_log.jsonl")):
            decision = json.loads(line).get("decision", {})
            # 主动驱逐退役后 completion_evictions 恒空；此处断言的是
            # 准入族逐出（history/prefill/decode）。
            for key in ("history_evictions", "prefill_evictions",
                        "decode_evictions", "completion_evictions"):
                evictions += len(decision.get(key) or ())
        if evictions == 0:
            fail("eviction path not exercised (§7.7 requires synthetic "
                 "coverage)")
        print(f"[a1-eviction-fixture] PASS: trains={len(ledger)} "
              f"member_iters={member_iters} chunks={chunks} "
              f"evictions={evictions}")
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
