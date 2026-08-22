#!/usr/bin/env python3
"""train_a1_eviction_fixture.py -- sh_2.0 拼 batch 改造 A1 合成微 trace
夹具（2026-08-22；§7.1 A1 层 / §7.7 最小测试集的 eviction + partial
前缀两段式迁移合成覆盖装置；母本 sh_1.0 train_a1_eviction_fixture.py
同构编排，断言集按 sh_2.0 特性扩展）。

trace 小窗不保证触发 KV 压力——本装置用小容量硬件（4 GiB/NPU：权重
分片 ~2.34 GiB + ~1.9 GiB KV ≈ 22K token）+ 11 个手算可核对的合成请求
（3 会话多 turn、异长 decode、交错到达）驱动完整在线链路，断言：

  1. §7.3 不变量：decode 成员-迭代总数 == Σ decode_length（3,588）；
     chunk 总数 == Σ ceil(prefill/512)（61）；
  2. 逐出实际发生（typed eviction：human 会话先减半——三态 KV 的
     partial_hbm_remote 状态可达）且运行 PASS；
  3. sh_2.0 特性保留专项：partial 前缀两段式迁移至少发生一次
     （train_ledger 存在 partial=True 的列车 = 首 chunk 按层段拆分
     发射的 admission→列车两段式流水；typed eviction 的“先减半”
     保证小容量下 partial 状态必然先于全逐出出现）；
  4. 全部 11 请求完成（交付 reasons 四类各 11）。

编排：合成输入落 /tmp → 备份 trace_config.csv → 临时指向合成队列/
小容量硬件（备份→改→还原→diff 验证）→ plan_materializer → 官方
runner → 断言 → 还原 + 清理生成目录。幂等可重跑；二进制缺失即
fail-closed。

用法：bash sh_test_mesh/workload/llama2_7b_inference/online/verify/\
train_a1_eviction_fixture.py   （或 python3 同路径）
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

# 14 请求（§7.3 不变量断言用，Σ decode/chunks 由行数据求和）。
# 容量压力设计（4 GiB/NPU ≈ 20.4K token/实例 KV 预算；reserve 3K
# token 水位）：12 个 turn-0 会话（各 ~9.5K token final）在 ~3ms 内
# 到达，queue-depth 均衡使 3 个实例各承载 2 个会话（成对实例剩余
# ~1.1K < reserve 3K → 完成边界 enforce_reserve 必触发 typed 两阶段
# 逐出）；成对实例上先完成者（A/B，短 decode）成为后完成者
# （J/K，长 decode）完成时的逐出候选——half-suffix 逐出后剩余
# ~5.8K ≥ reserve 3K，恰好停在 partial_hbm_remote（不全逐出）。
# A/B 的 turn-1（interval 2s，晚于配对会话完成/逐出点）到达时
# history_location_before == partial_hbm_remote → admission 走 partial
# 前缀两段式迁移分支 → 首 chunk 列车 partial=True。
# 行结构：(session, turn, request_id, prefill, decode, arrival_ns,
#         interval_ns, next_trigger_type)
REQUESTS = [
    ("session_A", 0, "A_r0", 9000, 400, 0, None, "human"),
    ("session_A", 1, "A_r1", 1500, 200, None, 2000000000, "human"),
    ("session_B", 0, "B_r0", 9000, 350, 250000, None, "human"),
    ("session_B", 1, "B_r1", 1500, 60, None, 2000000000, "human"),
    ("session_C", 0, "C_r0", 9000, 300, 500000, None, "human"),
    ("session_D", 0, "D_r0", 9000, 600, 750000, None, "human"),
    ("session_E", 0, "E_r0", 9000, 250, 1000000, None, "human"),
    ("session_F", 0, "F_r0", 9000, 500, 1250000, None, "human"),
    ("session_G", 0, "G_r0", 9000, 550, 1500000, None, "human"),
    ("session_H", 0, "H_r0", 9000, 450, 1750000, None, "human"),
    ("session_I", 0, "I_r0", 9000, 300, 2000000, None, "human"),
    ("session_J", 0, "J_r0", 9000, 900, 2250000, None, "human"),
    ("session_K", 0, "K_r0", 9000, 900, 2500000, None, "human"),
    ("session_L", 0, "L_r0", 9000, 900, 2750000, None, "human"),
]
EXPECT_DECODE = sum(r[4] for r in REQUESTS)
EXPECT_CHUNKS = sum(-(-r[3] // 512) for r in REQUESTS)


def fail(message: str) -> None:
    print(f"[a1-eviction-fixture] FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    if not os.path.exists(BINARY):
        fail(f"online binary missing: {BINARY} (build first)")
    work = tempfile.mkdtemp(prefix="/tmp/wsc_sh20_a1_fixture.")
    queue = os.path.join(work, "queue.csv")
    with open(queue, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "session_id", "turn_index", "request_id", "prefill_length",
            "decode_length", "session_arrival_time_ns",
            "inter_request_interval_ns", "next_trigger_type", "description"])
        for row in REQUESTS:
            writer.writerow(list(row) + ["A1 synthetic micro-trace"])
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
    config = config.replace(
        "config,kv_reserve_context_tokens,1000000",
        "config,kv_reserve_context_tokens,3000")
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
            # sh_2.0 决策日志口径：逐出以 *_eviction_count 计数字段记录
            # （母本 sh_1.0 为列表字段；此处按本仓 schema 计数）。
            for key in ("history_eviction_count", "prefill_eviction_count",
                        "decode_eviction_count",
                        "completion_eviction_count"):
                evictions += decision.get(key, 0) or 0
        if evictions == 0:
            fail("eviction path not exercised (§7.7 requires synthetic "
                 "coverage)")
        # sh_2.0 特性保留专项：partial 前缀两段式迁移（typed eviction
        # 减半 → partial_hbm_remote → 后续 turn 的 admission 走 prefix
        # 迁移 + suffix 恢复分支 → 首 chunk 列车 partial=True）。
        partial_trains = [r for r in ledger if r.get("partial")]
        if not partial_trains:
            fail("partial prefix two-stage migration was not exercised "
                 "(expected at least one partial=True train under the "
                 "4 GiB fixture)")
        print(f"[a1-eviction-fixture] PASS: trains={len(ledger)} "
              f"member_iters={member_iters} chunks={chunks} "
              f"evictions={evictions} partial_trains={len(partial_trains)}")
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
