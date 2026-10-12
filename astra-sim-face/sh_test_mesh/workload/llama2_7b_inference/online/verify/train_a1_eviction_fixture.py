#!/usr/bin/env python3
"""train_a1_eviction_fixture.py -- face 拼 batch 改造 A1 合成微 trace 夹具
(2026-08-22;§7.1 A1 层 / §7.7 最小测试集的 eviction 合成覆盖装置;
sh_1.0 母本同构,face 决策日志逐出键适配)。

trace 小窗不保证触发 KV 压力(5s 窗实测零逐出)——本装置用小容量
硬件 + 手算可核对的合成请求驱动完整在线链路。2026-09-30 重设计负载:
原 12 请求 / 8 会话被 face 加权映射 1:1 摊到 9 个实例,任一实例常驻
KV 远低于 4 GiB 档可用空间,逐出条件结构性不可达(实测三种逐出计数
恒 0)。新负载按逐出算术构造:

  - 容量档 a1-eviction-4gib(4 GiB/NPU):关键 rank(0/1,权重分片
    2,335,890,091 B、KV 98,304 B/token)可用 ~1.96 GiB;
  - 单会话 12,000 token 常驻 ~1.10 GiB(空实例可容纳);
  - 双会话同实例 ~2.20 GiB > 可用(被动逐出必触发);
  - 10 个首 turn 大会话 > 9 个实例(鸽笼必共位,后到者准入逼迫逐出
    已完成 inactive 会话);全部会话 200ms 后回访第 2 turn——被逐
    会话历史 ABSENT,顺带覆盖 RECOMPUTE 恢复路径。

断言:

  1. §7.3 不变量:decode 成员-迭代总数 == Σ decode_length;
     chunk 总数 == Σ ceil(prefill/512) + 动态折算的 RECOMPUTE 重算段;
  2. 逐出实际发生(face 决策日志的 admission/decode_target/completion
     evictions 非空)且运行 PASS;
  3. 全部请求完成(由断言 1 的 member-iteration 总数间接蕴含:
     decode 未交付完则总数必缺;交付 reasons 无直接断言)。

编排:合成输入落 /tmp → 备份 trace_config.csv → 既有 generated plan
目录移入 /tmp 备份(官方 runner 要求恰好一个 54npus 目录;结束后原样
迁回,不破坏仓内状态)→ 临时指向合成队列/小容量硬件(备份→改→还原)
→ plan_materializer → 官方 runner → 断言 → 迁回既有 plan + 清理本装置
生成的目录。幂等可重跑;二进制缺失即 fail-closed。

用法:python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/\
train_a1_eviction_fixture.py
"""
import csv
import json
import os
import shutil
import signal
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

# 合成请求手算期望(§7.3 不变量断言用;逐出算术见模块 docstring)。
# 首 turn 10 个 12000-token 会话交错到达(鸽笼必共位),第 2 turn 全体
# 200ms 后回访:被逐会话 RECOMPUTE、其余 LOCAL_HIT/NOC_MIGRATE。
# 行序按会话成对排列(reader 要求同会话行在 CSV 中连续)。
_BIG_PREFILL = 12000
_SESSIONS = [chr(ord("A") + i) for i in range(10)]
REQUESTS = []
for _i, _name in enumerate(_SESSIONS):
    REQUESTS.append((
        f"session_{_name}", 0, f"{_name}_r0", _BIG_PREFILL, 5,
        _i * 200000, None))
    REQUESTS.append((
        f"session_{_name}", 1, f"{_name}_r1", 1000, 5, None, 200000000))
EXPECT_DECODE = sum(r[4] for r in REQUESTS)
EXPECT_CHUNKS = sum(-(-r[3] // 512) for r in REQUESTS)


def fail(message: str) -> None:
    print(f"[a1-eviction-fixture] FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def run_subprocess(cmd, *, cwd, timeout):
    """R3-K6(2026-09-29):带超时的子进程调用;超时杀整个进程组后再
    fail——旧写法 runner 用裸 timeout=(超时只杀直接子进程,bash runner
    的孙进程沦为孤儿,与 finally 的 rmtree 竞态)、plan_materializer 干脆
    无超时(挂起即整装置挂死)。start_new_session 使进程组可整体 SIGKILL,
    清理目录时不再有存活写入者。"""
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.communicate()
        fail(f"subprocess timed out after {timeout}s: {cmd}")
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def main() -> int:
    if not os.path.exists(BINARY):
        fail(f"online binary missing: {BINARY} (build first)")
    work = tempfile.mkdtemp(prefix="/tmp/wsc_face_a1_fixture.")
    # 既有 generated plan 目录的 /tmp 备份点(见 try 内迁移与 finally 迁回)。
    plan_backup = os.path.join(work, "plan_backup")
    os.makedirs(plan_backup)
    queue = os.path.join(work, "queue.csv")
    with open(queue, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "session_id", "turn_index", "request_id", "prefill_length",
            "decode_length", "session_arrival_time_ns",
            "inter_request_interval_ns", "description"])
        for row in REQUESTS:
            writer.writerow(list(row) + ["A1 synthetic micro-trace"])
    with open(HARDWARE_SRC, encoding="utf-8") as handle:
        hardware = json.load(handle)
    hardware["local-hbm"]["capacity-profiles"]["a1-eviction-4gib"] = {
        "bytes": 4294967296,
        "label": "A1 synthetic small capacity (eviction fixture)",
        "note": "Synthetic 4 GiB/NPU capacity (weights ~2.34 GiB/rank + "
                "~1.9 GiB KV) to force KV eviction in the A1 fixture.",
    }
    hardware_path = os.path.join(work, "hw_small.json")
    with open(hardware_path, "w", encoding="utf-8") as handle:
        json.dump(hardware, handle, indent=2)

    with open(TRACE_CONFIG, encoding="utf-8") as handle:
        backup = handle.read()
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
        # 既有 plan 目录迁入 /tmp 备份(官方 runner 的 GEN_MATCH 门要求
        # 恰好一个 54npus 目录);finally 原样迁回——旧写法直接 rmtree
        # 且不还原,摧毁仓内既有物化状态。
        preexisting = (
            [name for name in os.listdir(gen_root)
             if name.startswith("llama2_7b_inference_54npus_plan_")]
            if os.path.isdir(gen_root) else [])
        for name in preexisting:
            shutil.move(os.path.join(gen_root, name),
                        os.path.join(plan_backup, name))
        with open(TRACE_CONFIG, "w", encoding="utf-8") as handle:
            handle.write(config)
        materialize = run_subprocess(
            [sys.executable, "plan_materializer.py"], cwd=WL, timeout=120)
        if materialize.returncode != 0:
            fail(f"plan_materializer failed: {materialize.stdout[-2000:]} "
                 f"{materialize.stderr}")
        run_dir = os.path.join(work, "run")
        # cwd 沿用调用方(与改前一致);cmd/输入均绝对路径。
        result = run_subprocess(
            ["bash", RUNNER, run_dir, queue], cwd=None, timeout=300)
        if result.returncode != 0:
            fail(f"runner failed: {result.stdout[-2000:]} {result.stderr}")
        with open(os.path.join(run_dir, "results/train_ledger.jsonl"),
                  encoding="utf-8") as handle:
            ledger = [json.loads(line) for line in handle]
        member_iters = sum(r["member_iterations"] for r in ledger)
        chunks = sum(r["prefill_chunks"] for r in ledger)
        if member_iters != EXPECT_DECODE:
            fail(f"decode member-iterations {member_iters} != "
                 f"{EXPECT_DECODE}")
        evictions = 0
        recompute_chunks = 0
        with open(os.path.join(run_dir,
                               "results/online_decision_log.jsonl"),
                  encoding="utf-8") as handle:
            for line in handle:
                decision = json.loads(line).get("decision", {})
                # face 决策日志逐出键(face 语义:准入/decode 目标/完成三处)。
                for key in ("admission_evictions", "decode_target_evictions",
                            "completion_evictions"):
                    evictions += len(decision.get(key) or ())
                # face 语义:被逐出会话的历史走 RECOMPUTE 恢复——重算段是
                # 基础 chunk 之外的额外 prefill 工作(两段式 chunk 计划的
                # 第一段;sh_1.0 母本无此口径,期望值按决策日志动态折算)。
                if decision.get("history_action") == "RECOMPUTE":
                    recompute_chunks += -(-int(
                        decision.get("history_recompute_tokens") or 0) // 512)
        expected_chunks = EXPECT_CHUNKS + recompute_chunks
        if chunks != expected_chunks:
            fail(f"chunk count {chunks} != base {EXPECT_CHUNKS} + "
                 f"recompute {recompute_chunks} = {expected_chunks}")
        if evictions == 0:
            fail("eviction path not exercised (§7.7 requires synthetic "
                 "coverage)")
        print(f"[a1-eviction-fixture] PASS: trains={len(ledger)} "
              f"member_iters={member_iters} chunks={chunks} "
              f"(base {EXPECT_CHUNKS} + recompute {recompute_chunks}) "
              f"evictions={evictions}")
        return 0
    finally:
        with open(TRACE_CONFIG, "w", encoding="utf-8") as handle:
            handle.write(backup)
        gen_root = os.path.join(REPO, "sh_test_mesh/generated")
        if os.path.isdir(gen_root):
            # 只清理本装置运行期间生成的 plan 目录。
            for name in os.listdir(gen_root):
                if name.startswith("llama2_7b_inference_54npus_plan_"):
                    shutil.rmtree(os.path.join(gen_root, name))
        # 既有 plan 目录原样迁回(仓内状态零残留)。
        for name in os.listdir(plan_backup):
            shutil.move(os.path.join(plan_backup, name),
                        os.path.join(gen_root, name))
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
