# 命令速查（sh_2.0，仿真输入仅限 20.csv 前30s；路径③④口径 2026-08-18）

输入（用户指示 2026-08-15，仅此一档）：`workload/llama2_7b_inference/traces/
astra_compute_20_first_30_seconds_request_queue.csv`（+ context sidecar +
canonical digest，见 traces/PROVENANCE.md）。禁止其他 csv 与更长窗口。

```bash
REPO=/home/sunhao/wsc-simulator/template/astra-sim-sh_2.0
cd $REPO

# 构建（全部 CMake 目标：_Online + 机制 fixtures）
cmake --build build/astra_analytical/build_congestion_aware -j

# 物化输入（sidecar_restore 双件；规则与 md5 冻结值见 traces/PROVENANCE.md）
cd sh_test_mesh/workload/llama2_7b_inference
python3 traces/materialize_first_30s.py \
  /home/sunhao/wsc-simulator/agent-traces/tracelab/astra_compute_20.csv traces/
cd $REPO

# 物化 plan 目录（runtime_config 四小件 + manifest + metrics_manifest）
cd sh_test_mesh/workload/llama2_7b_inference && python3 plan_materializer.py && cd $REPO

# ③④ 在线 runner（GEN_MATCH：generated/ 下恰一 *_54npus_* 目录）
bash sh_test_mesh/run_scripts/run_online_strategy.sh <run_dir> <绝对路径 request_csv>
bash sh_test_mesh/run_scripts/run_online_strategy_sensing.sh <run_dir> <绝对路径 request_csv>

# 两段式直接驱动（等价于一键 runner）
python3 sh_test_mesh/workload/llama2_7b_inference/online/online_service.py \
    --bridge-dir <run_dir>/bridge --mode strategy --plan-dir <plan 目录> &
$REPO/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware_Online \
    --online-mode strategy --bridge-dir <run_dir>/bridge \
    --request-queue-csv <绝对路径 request_csv> --close-input --metrics-detail=off

# 机制 fixtures + EventQueue C++ 单测
bash sh_test_mesh/run_scripts/run_online_idle_fixture.sh [run_root]
bash sh_test_mesh/run_scripts/run_online_wakeup_guard_fixture.sh <run_root>
bash sh_test_mesh/run_scripts/run_online_same_tick_milestone.sh [run_root]
g++ -std=c++17 -I extern/network_backend/analytical/include \
    astra-sim/workload/execution_driven/tests/event_queue_deferred_test.cc \
    extern/network_backend/analytical/common/event-queue/EventQueue.cpp \
    extern/network_backend/analytical/common/event-queue/EventList.cpp \
    -o /tmp/eq_test && /tmp/eq_test

# 后处理 + ④ 分层账本对账
bash sh_test_mesh/run_scripts/run_metrics_postprocess.sh <run_dir>/cpp.log
python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/ledger_reconcile.py --online <run_dir>
python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/ledger_reconcile_sh20.py --run-dir <sensing_run_dir>

# Python 单元测试（双根）
cd sh_test_mesh/workload/llama2_7b_inference && python3 -m pytest test_face_scheduler.py online/test_turn0_eviction_patch.py -q
cd $REPO/sh_test_mesh && python3 -m pytest tests/ -q

# 红线检查（每阶段）
git diff <phase-start>..<phase-end> -- \
    template/astra-sim-sh_2.0/sh_test_mesh/workload/llama2_7b_inference/face_scheduler.py \
    template/astra-sim-sh_2.0/sh_test_mesh/workload/llama2_7b_inference/generate_face_trace.py
```

# 一键清空测试记录（删除 generated/results/online_runs/log/缓存与 traces 数据件，
# git 恢复 trace_config；保留 build 与 tracked 文件；--full 连 build 一起清）
bash sh_test_mesh/run_scripts/clean_test_records.sh [--full]
