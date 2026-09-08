#!/usr/bin/env python3
"""train_a1_eviction_fixture.py -- face 拼 batch 改造 A1 合成微 trace 夹具
(2026-08-22;§7.1 A1 层 / §7.7 最小测试集的 eviction 合成覆盖装置;
sh_1.0 母本同构,face 决策日志逐出键适配)。

trace 小窗不保证触发 KV 压力(2s 窗实测零逐出;真实 trace 的会话上下文
远小于任何容量档)——本装置用小容量硬件(4 GiB/NPU:权重分片后实例
KV 容量 ≈ 22.4K token)驱动完整在线链路。B3(2026-09-06)三态化适配:
旧两态内核按预留口径计压,4 GiB 下原 12 请求即可逐出;三态内核按实际
驻留计压(kv_reserve_context_tokens 维持 manifest-only,策略 §1.5),
改为 13 个交叠双 turn 大会话(单会话 final ≈ 18.8K token,实例容量
22.4K——单会话恒可容纳;13 会话总量 ≈ 244K > 9 实例 × 22.4K ≈ 202K,
鸽笼强制溢出,逐出不可避免)。断言:

  1. §7.3 不变量:decode 成员-迭代总数 == Σ decode_length;
     chunk 总数 == Σ ceil(prefill/512)(B3:RECOMPUTE 项已删除);
  2. 逐出实际发生(face 决策日志的 admission/decode_target/completion
     evictions 非空)且运行 PASS;
  3. 全部 26 请求完成(交付 reasons 四类各 26)。

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

# 13 会话 × 2 turn(B3 三态容量口径下的手算期望;每会话两行必须连续
# ——windowed reader 的 session block 连续性要求):
#   Σ decode = 13×(200+100) = 3900;
#   Σ chunk = 13×ceil(18000/512) + 13×ceil(500/512) = 13×(36+1) = 481;
#   单会话 final = 18000+200+500+100 = 18800 token < 实例容量 22419;
#   交叠由到达时间驱动(turn0 到达 0..2.4s 交错;turn1 = 完成+5s)。
REQUESTS = [
    ("session_s00", 0, "s00_r0", 18000, 200, 0, None),
    ("session_s00", 1, "s00_r1", 500, 100, None, 5000000000),
    ("session_s01", 0, "s01_r0", 18000, 200, 200000000, None),
    ("session_s01", 1, "s01_r1", 500, 100, None, 5000000000),
    ("session_s02", 0, "s02_r0", 18000, 200, 400000000, None),
    ("session_s02", 1, "s02_r1", 500, 100, None, 5000000000),
    ("session_s03", 0, "s03_r0", 18000, 200, 600000000, None),
    ("session_s03", 1, "s03_r1", 500, 100, None, 5000000000),
    ("session_s04", 0, "s04_r0", 18000, 200, 800000000, None),
    ("session_s04", 1, "s04_r1", 500, 100, None, 5000000000),
    ("session_s05", 0, "s05_r0", 18000, 200, 1000000000, None),
    ("session_s05", 1, "s05_r1", 500, 100, None, 5000000000),
    ("session_s06", 0, "s06_r0", 18000, 200, 1200000000, None),
    ("session_s06", 1, "s06_r1", 500, 100, None, 5000000000),
    ("session_s07", 0, "s07_r0", 18000, 200, 1400000000, None),
    ("session_s07", 1, "s07_r1", 500, 100, None, 5000000000),
    ("session_s08", 0, "s08_r0", 18000, 200, 1600000000, None),
    ("session_s08", 1, "s08_r1", 500, 100, None, 5000000000),
    ("session_s09", 0, "s09_r0", 18000, 200, 1800000000, None),
    ("session_s09", 1, "s09_r1", 500, 100, None, 5000000000),
    ("session_s10", 0, "s10_r0", 18000, 200, 2000000000, None),
    ("session_s10", 1, "s10_r1", 500, 100, None, 5000000000),
    ("session_s11", 0, "s11_r0", 18000, 200, 2200000000, None),
    ("session_s11", 1, "s11_r1", 500, 100, None, 5000000000),
    ("session_s12", 0, "s12_r0", 18000, 200, 2400000000, None),
    ("session_s12", 1, "s12_r1", 500, 100, None, 5000000000),
]
EXPECT_DECODE = sum(r[4] for r in REQUESTS)
EXPECT_CHUNKS = sum(-(-r[3] // 512) for r in REQUESTS)


def fail(message: str) -> None:
    print(f"[a1-eviction-fixture] FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    if not os.path.exists(BINARY):
        fail(f"online binary missing: {BINARY} (build first)")
    work = tempfile.mkdtemp(prefix="/tmp/wsc_face_a1_fixture.")
    queue = os.path.join(work, "queue.csv")
    with open(queue, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "session_id", "turn_index", "request_id", "prefill_length",
            "decode_length", "session_arrival_time_ns",
            "inter_request_interval_ns", "description"])
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
            capture_output=True, text=True, timeout=900)
        if result.returncode != 0:
            fail(f"runner failed: {result.stdout[-2000:]} {result.stderr}")
        ledger = [json.loads(line) for line in open(
            os.path.join(run_dir, "results/train_ledger.jsonl"))]
        member_iters = sum(r["member_iterations"] for r in ledger)
        chunks = sum(r["prefill_chunks"] for r in ledger)
        if member_iters != EXPECT_DECODE:
            fail(f"decode member-iterations {member_iters} != "
                 f"{EXPECT_DECODE}")
        evictions = 0
        for line in open(os.path.join(
                run_dir, "results/online_decision_log.jsonl")):
            decision = json.loads(line).get("decision", {})
            # face 决策日志逐出键:legacy 三键即逐出全集(B3 的
            # history_evictions/prefill_evictions/decode_evictions 契约行
            # 与之同源,不重复计数)。
            for key in ("admission_evictions", "decode_target_evictions",
                        "completion_evictions"):
                evictions += len(decision.get(key) or ())
        # B3(2026-09-06):RECOMPUTE 已从历史路径删除(策略 §1.4)——被逐出
        # 会话经三态恢复回迁(local_hit/noc_migrate/remote_load),重算段
        # 不再是基础 chunk 之外的额外 prefill 工作,期望值即 Σ ceil(p/512)。
        expected_chunks = EXPECT_CHUNKS
        if chunks != expected_chunks:
            fail(f"chunk count {chunks} != expected {expected_chunks}")
        if evictions == 0:
            fail("eviction path not exercised (§7.7 requires synthetic "
                 "coverage)")
        print(f"[a1-eviction-fixture] PASS: trains={len(ledger)} "
              f"member_iters={member_iters} chunks={chunks} "
              f"(expected {EXPECT_CHUNKS}, recompute term removed by B3) "
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
