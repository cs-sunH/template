# 命令速查（sh_3.0，阶段 0 步骤 0-6 固化）

来源：sh_3.0仓库改造详细执行方案.md 附录 B。本仓库为裸仓库
（request-neutral）：不物化任何默认 request 队列；仿真输入由调用方按
方案文档 §3 步骤 0-1 从 `agent-traces/tracelab/astra_compute_20.csv`
前 30 秒（唯一允许输入，用户指示 2026-08-15）物化 sidecar_restore
三件套后在 trace_config.csv（或 runner 的 --request-queue-csv）显式
指定；正式入口缺失输入 fail-closed（随机 stub 仅限显式 fixture）。
物化规则与 30s 实测（1177 请求/112 session）见方案文档与
sh_3.0改造执行实录.md。

```bash
REPO=/home/sunhao/wsc-simulator/template/astra-sim-sh_3.0
cd $REPO

# 全管线（clean + build + generate + run + postprocess；约 1 分钟）
bash sh_test_mesh/run_scripts/runall.sh

# 仅构建（含在线目标，阶段 1 起）
bash sh_test_mesh/run_scripts/build_analytical_aware.sh

# 仅 trace 生成（读 trace_config.csv；request_queue_csv 须先指向物化输入）
cd sh_test_mesh/workload/llama2_7b_inference && python3 generate_trace.py
# 只看解析结果（不生成）
python3 generate_trace.py --print-shell-config
# 决策日志（replay 源；产物字节与生产一致）
python3 generate_trace.py --replay-record --metrics-detail=full

# 静态仿真二进制（完整参数以 run_sh_test_aware.sh 为准）
$REPO/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware \
    --workload-configuration=... --comm-group-configuration=... --system-configuration=... \
    --remote-memory-configuration=... --network-configuration=... \
    --logging-folder=off --metrics-configuration=... --metrics-detail=full

# 在线（阶段 1 起）
python3 sh_test_mesh/workload/llama2_7b_inference/online/online_service.py \
    --bridge-dir <run_dir>/bridge --mode replay|strategy \
    --decision-log <path> --plan-dir <离线 plan 目录> &
$REPO/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware_Online \
    --online-mode replay|strategy --bridge-dir <run_dir>/bridge \
    --request-queue-csv ... --close-input --metrics-detail=off
# 或一键：bash sh_test_mesh/run_scripts/run_online_replay.sh <run_dir> <csv> <decision_log>
#          bash sh_test_mesh/run_scripts/run_online_strategy.sh <run_dir> <csv>
#          bash sh_test_mesh/run_scripts/run_online_idle_fixture.sh
#          bash sh_test_mesh/run_scripts/run_online_same_tick_milestone.sh

# 单元测试
cd sh_test_mesh/workload/llama2_7b_inference && python3 -m pytest test_face_scheduler.py -q
cd $REPO && python3 -m pytest sh_test_mesh/tests/ -q

# EventQueue map 版单元测试（阶段 1）
g++ -std=c++17 -I extern/network_backend/analytical/include \
    astra-sim/workload/execution_driven/tests/event_queue_deferred_test.cc \
    extern/network_backend/analytical/common/event-queue/EventQueue.cpp \
    -o /tmp/eq_test && /tmp/eq_test

# 等价验收（阶段 2）
python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/tier_b_compare.py \
    --baseline sh_test_mesh/baseline/20_30s --online <online_dir>

# 对账（阶段 3）
python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/ledger_reconcile.py \
    --online <online_dir>

# 红线检查（每阶段）
git diff <phase-start>..<phase-end> -- \
  template/astra-sim-sh_3.0/sh_test_mesh/workload/llama2_7b_inference/face_scheduler.py \
  template/astra-sim-sh_3.0/sh_test_mesh/workload/llama2_7b_inference/generate_face_trace.py
# 字节等价门（generated 逐文件比对，禁止 diff -r）
for f in sh_test_mesh/baseline/20_30s/generated/*; do \
  cmp "$f" "sh_test_mesh/generated/<label>/${f##*/}"; done
```

决策日志：`--replay-record` 生成 `decision_log.jsonl`（293,801 行；
md5 见 baseline/20_30s/PROVENANCE.md），归档于
`sh_test_mesh/baseline/20_30s/decision_log.jsonl`。
