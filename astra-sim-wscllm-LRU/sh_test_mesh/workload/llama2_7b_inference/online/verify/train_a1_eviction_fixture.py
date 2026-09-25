#!/usr/bin/env python3
"""train_a1_eviction_fixture.py -- wscllm-LRU 拼 batch 改造 A1 合成微 trace
夹具(2026-09-06,B3;face/sh_2.0 母本 train_a1_eviction_fixture.py 同构
编排,wscllm 三态决策日志键适配)。

trace 小窗不保证触发 KV 压力——本装置用小容量硬件(3.5 GiB/NPU:权重
分片 ~2.24 GiB + 实例 KV 预算 ~14.2K token)+ 12 个手算可核对的合成请求
(3 会话多 turn、异长 decode、交错到达)驱动完整在线链路,断言:

  1. §7.3 不变量:decode 成员-迭代总数 == Σ decode_length;
     chunk 总数 == Σ ceil(prefill/512)(RECOMPUTE 已删,无重算加项);
  2. 逐出实际发生且形状正确:决策日志的逐出条目(admission/history/
     prefill/decode 侧,B3 契约 §3 行结构)非空,remote_store 条目带
     layer_start/layer_end 层段与 stage 可归因;
  3. 全部 12 请求完成(run 正常退出 + train_ledger 覆盖)。

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
PLAN_PREFIX = "llama2_7b_wsc_llm_inference_54npus_plan_"

# 12 请求手算期望(§7.3 不变量断言用)。行结构(request CSV 8 列):
# (session, turn, request_id, prefill, decode, arrival_ns, interval_ns)。
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

# wscllm 决策日志逐出键(B3 契约 §3;新旧键并存为超集,本装置两类都核)。
EVICTION_KEYS = (
    "admission_evictions", "decode_target_evictions",
    "completion_evictions",
    "history_evictions", "prefill_evictions", "decode_evictions",
)


def fail(message: str) -> None:
    print(f"[a1-eviction-fixture] FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    if not os.path.exists(BINARY):
        fail(f"online binary missing: {BINARY} (build first)")
    work = tempfile.mkdtemp(prefix="/tmp/wsc_llm_a1_fixture.")
    queue = os.path.join(work, "queue.csv")
    with open(queue, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "session_id", "turn_index", "request_id", "prefill_length",
            "decode_length", "session_arrival_time_ns",
            "inter_request_interval_ns", "description"])
        for row in REQUESTS:
            writer.writerow(list(row) + ["A1 synthetic micro-trace"])
    with open(HARDWARE_SRC, encoding="utf-8") as source:
        hardware = json.load(source)
    hardware["local-hbm"]["capacity-profiles"]["a1-eviction-4gib"] = {
        "bytes": 3758096384,
        "label": "A1 synthetic small capacity (eviction fixture)",
        "note": "Synthetic 3.5 GiB/NPU capacity: fits the largest single "
                "request of the synthetic queue on an empty instance "
                "(~0.77 GiB/rank final KV) while the aggregate session "
                "KV (~50K tokens over 3 Decode instances vs ~14.2K-token "
                "instance budget) forces whole-session tiered LRU eviction.",
    }
    hardware_path = os.path.join(work, "hw_small.json")
    with open(hardware_path, "w", encoding="utf-8") as handle:
        json.dump(hardware, handle, indent=2)

    with open(TRACE_CONFIG, encoding="utf-8") as source:
        backup = source.read()
    config = backup
    config = config.replace(
        "config,hardware_config,hardware/face_case5_config_c.json",
        f"config,hardware_config,{hardware_path}")
    config = config.replace(
        "config,local_hbm_capacity_profile,validation-160gib",
        "config,local_hbm_capacity_profile,a1-eviction-4gib")
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
                       if name.startswith(f"{PLAN_PREFIX}")]
        for name in preexisting:
            shutil.rmtree(os.path.join(gen_root, name))
        with open(TRACE_CONFIG, "w", encoding="utf-8") as handle:
            handle.write(config)
        subprocess.run(
            [sys.executable, "plan_materializer.py"], cwd=WL,
            check=True, capture_output=True)
        run_dir = os.path.join(work, "run")
        result = subprocess.run(
            ["bash", RUNNER, run_dir, queue],
            capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            fail(f"runner failed: {result.stdout[-2000:]} {result.stderr}")
        with open(os.path.join(run_dir, "results/train_ledger.jsonl")) as f:
            ledger = [json.loads(line) for line in f]
        member_iters = sum(r["member_iterations"] for r in ledger)
        chunks = sum(r["prefill_chunks"] for r in ledger)
        if member_iters != EXPECT_DECODE:
            fail(f"decode member-iterations {member_iters} != "
                 f"{EXPECT_DECODE}")
        if chunks != EXPECT_CHUNKS:
            fail(f"chunk count {chunks} != expected {EXPECT_CHUNKS} "
                 f"(RECOMPUTE removed: no recompute surcharge)")
        evictions = 0
        remote_store_entries = 0
        bad_layer_domains = 0
        with open(os.path.join(
                run_dir, "results/online_decision_log.jsonl")) as f:
            for line in f:
                decision = json.loads(line).get("decision", {})
                for key in EVICTION_KEYS:
                    entries = decision.get(key) or ()
                    for entry in entries:
                        evictions += 1
                        if entry.get("kind") == "remote_store":
                            remote_store_entries += 1
                            layer_start = entry.get("layer_start")
                            layer_end = entry.get("layer_end")
                            if (not isinstance(layer_start, int)
                                    or not isinstance(layer_end, int)
                                    or not 0 <= layer_start < layer_end):
                                bad_layer_domains += 1
        if evictions == 0:
            fail("eviction path not exercised (§7.7 requires synthetic "
                 "coverage)")
        if remote_store_entries == 0:
            fail("no remote_store eviction entries in the decision log")
        if bad_layer_domains:
            fail(f"{bad_layer_domains} remote_store entries lack a valid "
                 "layer domain (contract §3 row structure)")
        print(f"[a1-eviction-fixture] PASS: trains={len(ledger)} "
              f"member_iters={member_iters} chunks={chunks} "
              f"evictions={evictions} remote_store={remote_store_entries}")
        return 0
    finally:
        with open(TRACE_CONFIG, "w", encoding="utf-8") as handle:
            handle.write(backup)
        gen_root = os.path.join(REPO, "sh_test_mesh/generated")
        for name in os.listdir(gen_root):
            if name.startswith(PLAN_PREFIX):
                shutil.rmtree(os.path.join(gen_root, name))
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
