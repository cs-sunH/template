# 命令速查（sh_3.0，阶段 0 步骤 0-6 固化；路径③④口径 2026-08-18）

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

# 构建（全部 CMake 目标：_Online + 机制 fixtures）
cmake --build build/astra_analytical/build_congestion_aware -j

# 0. 物化输入（sidecar_restore 三件套；规则见 traces/PROVENANCE.md）
cd sh_test_mesh/workload/llama2_7b_inference
python3 traces/materialize_20_30s.py \
  /home/sunhao/wsc-simulator/agent-traces/tracelab/astra_compute_20.csv traces/
cd $REPO

# 1. 物化 plan 目录（runtime_config 四小件 + manifest + metrics_manifest）
cd sh_test_mesh/workload/llama2_7b_inference && python3 plan_materializer.py && cd $REPO

# 2. ③④ 在线 runner（GEN_MATCH：generated/ 下恰一 *_54npus_* 目录）
bash sh_test_mesh/run_scripts/run_online_strategy.sh <run_dir> <绝对路径 request_csv>
bash sh_test_mesh/run_scripts/run_online_strategy_sensing.sh <run_dir> <绝对路径 request_csv>

# 3. 机制 fixtures + EventQueue C++ 单测
bash sh_test_mesh/run_scripts/run_online_idle_fixture.sh [run_root]
bash sh_test_mesh/run_scripts/run_online_wakeup_guard_fixture.sh <run_root>
bash sh_test_mesh/run_scripts/run_online_same_tick_milestone.sh [run_root]
g++ -std=c++17 -I extern/network_backend/analytical/include \
    astra-sim/workload/execution_driven/tests/event_queue_deferred_test.cc \
    extern/network_backend/analytical/common/event-queue/EventQueue.cpp \
    -o /tmp/eq_test && /tmp/eq_test

# 4. 后处理 + ④ 分层账本对账
bash sh_test_mesh/run_scripts/run_metrics_postprocess.sh <run_dir>/cpp.log
python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/sh30_ledger_reconcile.py \
    --run-dir <run_dir> --manifest <run_dir>/results/online_decision_log.jsonl

# 5. 单元测试（双根）
cd sh_test_mesh/workload/llama2_7b_inference && python3 -m pytest test_face_scheduler.py -q
cd $REPO/sh_test_mesh && python3 -m pytest tests/ -q

# 红线检查（每阶段）
git diff <phase-start>..<phase-end> -- \
  template/astra-sim-sh_3.0/sh_test_mesh/workload/llama2_7b_inference/face_scheduler.py \
  template/astra-sim-sh_3.0/sh_test_mesh/workload/llama2_7b_inference/generate_face_trace.py
```

# 一键清空测试记录（删除 generated/results/online_runs/log/缓存与 traces 数据件，
# git 恢复 trace_config；保留 build 与 tracked 文件；--full 连 build 一起清）
bash sh_test_mesh/run_scripts/clean_test_records.sh [--full]
