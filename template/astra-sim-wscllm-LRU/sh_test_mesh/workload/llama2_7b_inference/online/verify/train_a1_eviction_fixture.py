#!/usr/bin/env python3
"""train_a1_eviction_fixture.py -- wscllm-LRU 拼 batch 改造 A1 合成微 trace
夹具(2026-09-06,B3;face/sh_2.0 母本 train_a1_eviction_fixture.py 同构
编排,wscllm 二态决策日志键适配;2026-10-07 die 时代重设计:撤小容量
硬件档,放大合成请求制造 KV 压力强制逐出)。

登记(py-verify-D2 2026-09-27):手动 CLI 夹具工具——全仓无外部引用、
不被 pytest 收集,仅 __main__ 手动可达,属有意保留的手动装置(非死文件)。

die 时代背景(2026-10-04 五仓接入 die 内建模):config_resolver 契约
§3.4 硬校验——hardware die 节在场时所选 capacity-profile 字节必须
等于 die 派生容量(本仓 64,000,000,000 B,经 libdie_model_cabi.so
同源派生实测),小容量档在 die 派生粒度(= hbm-sides × hbm-per-side
× 16 GiB)下无法表达;故撤除 3.5 GiB 小容量硬件手术,改用放大合成
请求触发逐出。

每实例 KV 预算算术(session_kv_manager 记账口径):模型 12.551 GiB
按 TP=6 整头切分(partition_values_exact,余量低 rank 优先 →
[6,6,5,5,5,5] 头/rank),每 rank 预算 = 64e9 − 权重分片;KV 按
kv_cache_bytes_for_tokens 口径 2×32 层×本 rank 头数×128 维×2 B
分摊,绑定 rank = 6 头 rank(98,304 B/token)→ 627,280 token/实例。
3 个 Decode 实例经 wsc_llm_scheduler 静态 P-D 映射与最少排队 prefill
选择均分,聚合预算 ≈ 1.88M token。

设计:84 个双轮会话——turn-0 prefill 32,000、decode 5,到达
k×400M ns(6 个 prefill 实例流水线近饱和:每实例 2.6G ns 服务 vs
2.4G ns 到达间隔,并发常驻 ≈7-9,规避高并发预填拥挤导致的仿真步速
退化);turn-1(完成闭环 +interval 语义,见 run_golden_live.py
G3 恒等)prefill 512、decode 5,interval 逐会话反推把再准入浪精确
放到回填池满之后(T_wave 42G ns 基底 + k×15M 错开,按 6 实例
×3G ns 轮转的完成估计回推)。首轮回填聚合 2.69M token(每实例
≈1.43×预算);turn-1 再准入触发整会话 LRU 逐出(session_kv_
manager:completed 非活跃全本地会话为候选,mark_complete 完成
保留 KV;终端单轮会话完成即清账,故必须双轮)。墙钟标定(探针
与真跑实测):decode 成员迭代 ~50µs/iter、prefill ~81µs/token,
本设计仿真地平线 ≈80-90G ns(逐出/恢复波相占比可观),实测墙钟
约 95-105 分钟(runner 超时上限同步放宽,见 main)。断言:

  1. §7.3 不变量:decode 成员-迭代总数 == Σ decode_length;
     chunk 总数 == Σ ceil(prefill/512)(RECOMPUTE 已删,无重算加项);
  2. 逐出实际发生且形状正确:决策日志的逐出条目(admission/history/
     prefill/decode 侧,B3 契约 §3 行结构)非空,remote_store 条目带
     layer_start/layer_end 层段与 stage 可归因;
  3. 全部 168 请求完成(run 正常退出 + train_ledger 覆盖)。

编排:合成输入落 /tmp(queue.csv 伴生 .provenance.json 边车,M-16 门)
→ 备份 trace_config.csv → 仅改指 request_queue_csv 到合成队列(其余
保持 die 默认;替换一律 fail-closed,目标串不命中即报错退出)
→ plan_materializer → 官方 runner → 断言 → 还原 + 清理生成目录。
幂等可重跑;二进制缺失即 fail-closed。

用法:bash sh_test_mesh/workload/llama2_7b_inference/online/verify/\
train_a1_eviction_fixture.py   (或 python3 同路径)
"""
import csv
import importlib.util
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
PLAN_PREFIX = "llama2_7b_wsc_llm_inference_54npus_plan_"

# 84 个双轮会话(die 时代重设计,预算算术与编排标定见模块 docstring)。
# 行结构(request CSV 8 列):(session, turn, request_id, prefill, decode,
# arrival_ns, interval_ns);None 落 CSV 空串。
SESSIONS = 84
TURN0_PREFILL, TURN0_DECODE = 32000, 5
TURN1_PREFILL, TURN1_DECODE = 512, 5
# turn-0 到达步进:6 个 prefill 实例流水线利用率 ~0.85(服务 2.6G ns vs
# 实例内到达间隔 6×510M=3.06G ns),并发常驻 ~6、队列有界——步进 400M 的
# 前版利用率 >1、队列无限增长、步速崩溃(探针实测),故收紧。
ARRIVAL_STEP_NS = 510_000_000
# turn-1 再准入浪编排:把每条会话的 turn-1 精确放到回填池满之后——
# interval_k = T_wave + k×stagger − 按 6 实例轮转估计的 turn-0 完成
# 时刻(每实例 3G ns 估计,留量),钳底 1M ns 防负。
T_WAVE_NS = 42_000_000_000
INTERVAL_STAGGER_NS = 15_000_000
PREFILL_INSTANCES = 6
PER_INSTANCE_PREFILL_EST_NS = 3_000_000_000

REQUESTS = []
for _k in range(SESSIONS):
    _completion_est = (
        -(-(_k + 1) // PREFILL_INSTANCES) * PER_INSTANCE_PREFILL_EST_NS)
    _interval = max(1_000_000,
                    T_WAVE_NS + _k * INTERVAL_STAGGER_NS - _completion_est)
    REQUESTS.append(
        (f"session_{_k:02d}", 0, f"S{_k:02d}_r0",
         TURN0_PREFILL, TURN0_DECODE, _k * ARRIVAL_STEP_NS, None))
    REQUESTS.append(
        (f"session_{_k:02d}", 1, f"S{_k:02d}_r1",
         TURN1_PREFILL, TURN1_DECODE, None, _interval))
EXPECT_DECODE = sum(r[4] for r in REQUESTS)
EXPECT_CHUNKS = sum(-(-r[3] // 512) for r in REQUESTS)

# wscllm 决策日志逐出键(B3 契约 §3)。计数只取分账键,不取超集/别名键:
# decode_target_evictions 与 decode_evictions 为同内容别名键(生产侧决策行
# 以同一 runtime 序列序列化两次,见 wsc_llm_online_scheduler 组装处注释
# "防双计"),本装置只核其一(py-verify-C3 修复);admission_evictions 是
# history_evictions ∪ prefill_evictions 的 union 账本(调度器 B3 分账注释
# 自证,两个组成段已各自入键)——遍历它会把每条 history/prefill 逐出再
# 计一次,报告数字约 2× 失真,故摘除(Minor 核实修复)。
EVICTION_KEYS = (
    "decode_target_evictions",
    "completion_evictions",
    "history_evictions", "prefill_evictions",
)


def fail(message: str) -> None:
    print(f"[a1-eviction-fixture] FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def write_provenance_sidecar(queue_path: str) -> None:
    """为合成队列写 <queue>.provenance.json（M-16 provenance 门对齐，
    与 sh_test_mesh/slo_tools/tests/run_golden_live.py 的
    write_provenance_sidecar 同款修复，2026-10-07）。

    C++ WindowedTraceReader 的 provenance 门（M-16，2026-09-30 起缺席
    即 fail-closed）对每个队列 CSV fail-closed 地要求同名
    .provenance.json 边车，缺失即视同截断输入、在首个 Submit 前
    reader_fatal；plan_materializer 的 F2_PY 对账（plan_materializer.py）
    亦会校 sidecar 记录的 csv_sha256 与队列字节一致。统计算法零复制：
    直接复用权威物化器 traces/derive_20_first_30_seconds.py 的
    queue_provenance（与 C++ 索引遍逐字节同口径，含 CSV writer 的 CRLF
    字节与 '#' 注释行跳过）。边车落在队列同目录（/tmp 工作目录），随
    finally 的 work 目录清理一并删除。"""
    derive_path = os.path.join(
        WL, "traces", "derive_20_first_30_seconds.py")
    spec = importlib.util.spec_from_file_location(
        "derive_20_first_30_seconds", derive_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    provenance = module.queue_provenance(
        queue_path,
        "train_a1_eviction_fixture.py hand-crafted A1 synthetic queue")
    sidecar = queue_path + ".provenance.json"
    with open(sidecar, "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")


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
    # M-16 provenance 门（WindowedTraceReader，2026-09-30 起缺席即
    # fail-closed）：合成队列必须带同名 .provenance.json 边车，否则装载
    # 阶段 reader_fatal、首个 Submit 前中止。边车必须与队列字节对账
    # （plan_materializer 的 F2_PY 校 csv_sha256），故在写完 queue.csv
    # 之后立即伴生，先于下方 plan_materializer 调用。
    write_provenance_sidecar(queue)

    with open(TRACE_CONFIG, encoding="utf-8") as source:
        backup = source.read()
    config = backup
    lines = [line for line in config.splitlines()
             if line.startswith("config,request_queue_csv,")]
    if len(lines) != 1:
        fail("trace_config must carry exactly one request_queue_csv line")
    config = config.replace(
        lines[0],
        f"config,request_queue_csv,{queue},,,A1 fixture temporary pointer")
    if config == backup:
        fail("trace_config request_queue_csv surgery was a no-op "
             "(die-era config drift? expected exactly one "
             "config,request_queue_csv line to rewrite)")
    # R4 所有权边界：记录进入夹具前已存在的 plan 目录——finally 只删
    # 本夹具运行期间新出现的目录（plan_materializer 发布的合成 plan），
    # 不再盲删全部前缀目录（失败路径下用户的官方物化产物得以保留）。
    gen_root_entry = os.path.join(REPO, "sh_test_mesh/generated")
    preexisting_plans = frozenset(
        name for name in os.listdir(gen_root_entry)
        if name.startswith(PLAN_PREFIX)) if os.path.isdir(gen_root_entry) else frozenset()
    try:
        # （R3 复核修正）不再预先删除既有 plan 目录：plan_materializer 的
        # D3 对齐发布（_publish_plan_dir，2026-10-01）在成功发布本夹具的
        # plan 时自行清扫其它 cfg8 世代并维持"恰一"不变量——旧预删在夹具
        # 中途失败时会把用户的既有官方物化产物一并删掉且不可恢复。本夹具
        # 的 plan 产生于下方 plan_materializer 调用之后，GEN_MATCH 恰一闸
        # 在该点之后才被 runner 触碰。
        with open(TRACE_CONFIG, "w", encoding="utf-8") as handle:
            handle.write(config)
        subprocess.run(
            [sys.executable, "plan_materializer.py"], cwd=WL,
            check=True, capture_output=True)
        run_dir = os.path.join(work, "run")
        result = subprocess.run(
            ["bash", RUNNER, run_dir, queue],
            capture_output=True, text=True, timeout=7200)
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
                 f"layer domain (contract §3 row structure)")
        print(f"[a1-eviction-fixture] PASS: requests={len(REQUESTS)} "
              f"trains={len(ledger)} "
              f"member_iters={member_iters} chunks={chunks} "
              f"evictions={evictions} remote_store={remote_store_entries}")
        return 0
    finally:
        with open(TRACE_CONFIG, "w", encoding="utf-8") as handle:
            handle.write(backup)
        gen_root = os.path.join(REPO, "sh_test_mesh/generated")
        if os.path.isdir(gen_root):
            for name in os.listdir(gen_root):
                if name.startswith(PLAN_PREFIX) and name not in preexisting_plans:
                    shutil.rmtree(os.path.join(gen_root, name))
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
