#!/usr/bin/env python3
"""train_a1_eviction_fixture.py -- sh_2.0 拼 batch 改造 A1 合成微 trace
夹具（2026-08-22；§7.1 A1 层 / §7.7 最小测试集的 eviction + partial
前缀两段式迁移合成覆盖装置；母本 sh_1.0 train_a1_eviction_fixture.py
同构编排；2026-09-13 压力重设计 v2 + 裸仓可跑性修复，见下）。

trace 小窗不保证触发 KV 压力——本装置用小容量硬件档（a1-eviction-4gib
合成 profile）+ 三波 11 个手算可核对的合成请求（10 个会话：session_BIG
两个 turn + 9 个单轮 session_T0..T8）驱动完整在线链路，断言：

  1. §7.3 不变量：decode 成员-迭代总数 == Σ decode_length（6,200）；
     chunk 总数 == Σ ceil(prefill/512)（137）；
  2. 逐出实际发生且恰为设计的那一笔（硬断言）：准入期 typed stage-1
     半后缀逐出恰 1 次、victim 集合唯一为 session_BIG、reason 为
     request_admission_capacity_suffix_half；运行 PASS；
  3. sh_2.0 特性保留专项：partial 前缀两段式迁移（硬断言，确定性）：
     波3 session_BIG turn-1 在逐出后的 partial_hbm_remote 历史上再
     准入 → admission 走 prefix 迁移 + suffix 恢复分支 → 首 chunk
     列车按层段拆分发射 partial=True；
  4. 全部 11 请求完成。

容量压力设计（v2，2026-09-13 立案排查后重设计；立案报告
/tmp/evpar/s2fix/REPORT.md：v1 的"12 会话 × ~9.5K 对 ~20.4K 严重超订"
在 least-loaded 放置下不成立——均衡后每实例至多 2×~9.5K < 预算，
逐出从未可能触发）。

实例 KV 预算推导（手算可核对；数字由 face_scheduler.py 的
model_weight_shard_bytes_by_tp_rank / kv_cache_shard_bytes_for_tokens
口径得出）：binding rank = 持有 6/32 注意力头的 rank0/1——
  权重分片 2,335,890,091 B；每 token KV 分片 98,304 B
    （= 2(K/V) × 32 层 × 128 head_dim × 6 头 × 2 B）；
  预算 = floor((4 GiB − 2,335,890,091) / 98,304) = 19,928 token/实例。

三波编排（到达与完成时刻解耦）：
  波1 t=0：session_BIG turn-0（prefill 13,400 + decode 600 → final
    14,000 token，独跑；turn-1 interval 16s 挂起 → 完成后 KV 驻留，
    是全系统唯一"已完成 + inactive"可逐 victim）；
  波2 t=10e9 绝对到达：9 个单轮 session_T*（prefill 6,000 + decode
    600 → final 6,600）同刻到达。least-loaded 放置（平局按
    last_arrival/None 先、实例号）使 9 个 T 恰一实例一个——落在
    session_BIG 驻留实例上的那个 T 触发准入期逐出：
      14,000 + 6,600 = 20,600 > 19,928（缺 672 token
      ≈ 65,985,195 B/rank）→ stage-1 半后缀逐出 session_BIG
      （16/32 层 offload，688,128,000 B/rank）→ 7,000 + 6,600
      = 13,600 ≤ 19,928 恰好够即停 → session_BIG 转
      partial_hbm_remote（不全逐出；D4-I3 过逐出复核由原始缺口
      65,985,195 B > 0 保证通过）。T 间任意共驻 ≤ 3×6,600
      = 19,800 ≤ 预算，无 deep-gap 崩溃路径；
  波3 ≈ t≥16s：session_BIG turn-1（prefill 600 + decode 200 →
    final 14,800；interval 16s 为完成相对——BIG_r0 实测 ~2s 完成，
    波2 单轮退役 ~11.7s，裕量 ≥ 4s）在波2 全部退役后到达，独占
    实例 14,800 ≤ 19,928，历史 = partial_hbm_remote → partial
    前缀两段式迁移 + 后缀恢复 → 首 chunk 列车 partial=True。
（v1 的 EVPAR_A1_IMMEDIATE 变体随重设计移除：新压力不再依赖晚期
再准入时序，其 store→restore 时序意图由本波次的确定性
"波2 逐出 store → 波3 partial 恢复"承接；图级精确断言见
online/test_store_restore_ordering.py。）

裸仓可跑性（2026-09-13）：run_online_strategy.sh 硬编码
validation-160gib runtime_config 路径；本夹具在物化 a1-eviction-4gib
runtime_config 后，若该硬编码路径缺失则镜像一份并把 system.json 的
local-mem-capacity-bytes 拨回 160 GiB 历史传输层口径（C++ 侧不参与
KV 逐出预算——逐出算术在 Python 决策层的 4 GiB 档），运行后清理
本次自建部分。

编排：合成输入落 /tmp → 备份 trace_config.csv → 临时指向合成队列/
小容量硬件（备份→改→还原→diff 验证）→ plan_materializer → 官方
runner → 断言 → 还原 + 清理生成目录。幂等可重跑；二进制缺失即
fail-closed。

用法：python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/
train_a1_eviction_fixture.py（shebang 已设，直接执行亦可）
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
# run_online_strategy.sh 硬编码的 runtime_config 目录名（传输层 160 GiB
# 历史口径，见 docstring"裸仓可跑性"段）。
RUNNER_RC_NAME = "face_case5_config_c__validation-160gib__edge_remote_memory_pool"
TRANSPORT_CAPACITY_BYTES = 171798691840  # 160 GiB

# 11 请求 = session_BIG × 2 turn + 9 个单轮 session_T*（§7.3 不变量
# Σ decode/chunks 由行数据求和）。压力算术见 docstring。
# 行结构：(session, turn, request_id, prefill, decode, arrival_ns,
#         interval_ns, next_trigger_type)
REQUESTS = [
    ("session_BIG", 0, "BIG_r0", 13400, 600, 0, None, "human"),
    ("session_BIG", 1, "BIG_r1", 600, 200, None, 16000000000, "human"),
] + [
    (f"session_T{index}", 0, f"T{index}_r0",
     6000, 600, 10000000000, None, "human")
    for index in range(9)
]
EXPECT_DECODE = sum(r[4] for r in REQUESTS)
EXPECT_CHUNKS = sum(-(-r[3] // 512) for r in REQUESTS)
# 设计逐出指纹：恰 1 笔准入期半后缀逐出，victim 唯一 session_BIG。
EXPECT_EVICTIONS = 1
EXPECT_VICTIMS = {"session_BIG"}
EXPECT_EVICTION_REASON = "request_admission_capacity_suffix_half"


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
        "note": "Synthetic 4 GiB/NPU capacity (binding-rank KV budget "
                "19,928 tokens/instance after the 2,335,890,091 B weight "
                "shard) to force deterministic KV eviction in the A1 "
                "fixture.",
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
    gen_root = os.path.join(REPO, "sh_test_mesh/generated")
    rc_root = os.path.join(gen_root, "runtime_config")
    rc_preexisting = set()
    try:
        os.makedirs(gen_root, exist_ok=True)
        preexisting = [name for name in os.listdir(gen_root)
                       if name.startswith("llama2_7b_inference_54npus_plan_")]
        for name in preexisting:
            shutil.rmtree(os.path.join(gen_root, name))
        rc_preexisting = (os.path.isdir(rc_root)
                          and set(os.listdir(rc_root)) or set())
        open(TRACE_CONFIG, "w", encoding="utf-8").write(config)
        subprocess.run(
            [sys.executable, "plan_materializer.py"], cwd=WL,
            check=True, capture_output=True)
        # 裸仓可跑性：runner 硬编码 validation-160gib runtime_config
        # 路径；缺失则从本次物化的 a1-eviction-4gib 档镜像一份，仅把
        # system.json 的 local-mem-capacity-bytes 拨回 160 GiB 传输层
        # 口径（其余三件与档位无关）。自建部分在 finally 清理。
        a1_names = [name for name in os.listdir(rc_root)
                    if name.startswith("face_case5_config_c__a1-eviction-4gib__")]
        if len(a1_names) != 1:
            fail(f"expected exactly one a1-eviction-4gib runtime_config "
                 f"dir, found: {a1_names}")
        runner_rc = os.path.join(rc_root, RUNNER_RC_NAME)
        if not os.path.exists(runner_rc):
            shutil.copytree(
                os.path.join(rc_root, a1_names[0]), runner_rc)
            system_path = os.path.join(runner_rc, "system.json")
            system = json.load(open(system_path, encoding="utf-8"))
            system["local-mem-capacity-bytes"] = TRANSPORT_CAPACITY_BYTES
            json.dump(system, open(system_path, "w", encoding="utf-8"),
                      indent=2)
        run_dir = os.path.join(work, "run")
        result = subprocess.run(
            ["bash", RUNNER, run_dir, queue],
            capture_output=True, text=True, timeout=420)
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
        victims = set()
        reasons = set()
        completions = 0
        for line in open(os.path.join(
                run_dir, "results/online_decision_log.jsonl")):
            record = json.loads(line)
            decision = record.get("decision", {})
            if record.get("kind") == "completion":
                completions += 1
            # sh_2.0 决策日志口径：逐出以 *_eviction_count 计数字段记录
            # （母本 sh_1.0 为列表字段）；主动驱逐退役后 completion 段
            # 恒 0，此处断言的是准入族逐出（history/prefill/decode）。
            for key in ("history_eviction_count", "prefill_eviction_count",
                        "decode_eviction_count",
                        "completion_eviction_count"):
                evictions += decision.get(key, 0) or 0
            for entry in decision.get("history_evictions") or []:
                victims.add(entry.get("session_id"))
                reasons.add(entry.get("reason"))
            for entry in decision.get("decode_evictions") or []:
                victims.add(entry.get("session_id"))
                reasons.add(entry.get("reason"))
        if completions != len(REQUESTS):
            fail(f"completed requests {completions} != {len(REQUESTS)}")
        # 硬断言：逐出指纹必须与 docstring 手算算术完全一致（确定性
        # 压力设计；victim 集合逐 run 恒定，可作对拍锚点）。
        if evictions != EXPECT_EVICTIONS:
            fail(f"eviction count {evictions} != {EXPECT_EVICTIONS} "
                 f"(victims={sorted(victims)}, reasons={sorted(reasons)})")
        if victims != EXPECT_VICTIMS:
            fail(f"eviction victims {sorted(victims)} != "
                 f"{sorted(EXPECT_VICTIMS)}")
        if EXPECT_EVICTION_REASON not in reasons:
            fail(f"expected suffix-half eviction reason "
                 f"{EXPECT_EVICTION_REASON!r} missing from {sorted(reasons)}")
        # sh_2.0 特性保留专项：partial 前缀两段式迁移（准入逐出减半 →
        # partial_hbm_remote → 波3 turn-1 的 admission 走 prefix 迁移 +
        # suffix 恢复分支 → 首 chunk 列车 partial=True）。波3 在波2 全部
        # 退役后到达，恢复为确定性硬断言（v1 的 R8 尽力降级随 v2 压力
        # 重设计废除）。
        partial_trains = [r for r in ledger if r.get("partial")]
        if not partial_trains:
            fail("partial prefix two-stage migration was not exercised "
                 "(expected session_BIG turn-1 first-chunk partial train)")
        print(f"[a1-eviction-fixture] PASS: trains={len(ledger)} "
              f"member_iters={member_iters} chunks={chunks} "
              f"evictions={evictions} victims={sorted(victims)} "
              f"partial_trains={len(partial_trains)}")
        return 0
    finally:
        open(TRACE_CONFIG, "w", encoding="utf-8").write(backup)
        if os.path.isdir(rc_root):
            for name in os.listdir(rc_root):
                # 只清理本次运行自建的 runtime_config 目录（裸仓还原）；
                # 预存目录（历史 stock 物化产物）保持原样。
                if name not in rc_preexisting:
                    shutil.rmtree(os.path.join(rc_root, name),
                                  ignore_errors=True)
            if not os.listdir(rc_root):
                os.rmdir(rc_root)
        gen_root = os.path.join(REPO, "sh_test_mesh/generated")
        for name in os.listdir(gen_root):
            if name.startswith("llama2_7b_inference_54npus_plan_"):
                shutil.rmtree(os.path.join(gen_root, name))
        if not os.listdir(gen_root):
            os.rmdir(gen_root)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
