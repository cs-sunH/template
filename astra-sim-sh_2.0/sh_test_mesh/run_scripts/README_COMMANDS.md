# 命令速查（sh_2.0，仿真输入仅限 20.csv 前30s）

输入（用户指示 2026-08-15，仅此一档）：`workload/llama2_7b_inference/traces/
astra_compute_20_first_30_seconds_request_queue.csv`（+ context sidecar +
canonical digest，见 traces/PROVENANCE.md）。禁止其他 csv 与更长窗口。

```bash
REPO=/home/sunhao/wsc-simulator/template/astra-sim-sh_2.0
cd $REPO

# 全管线（clean + build + generate + run + postprocess）
bash sh_test_mesh/run_scripts/runall.sh

# 仅构建（阶段 1 起含在线目标）
bash sh_test_mesh/run_scripts/build_analytical_aware.sh

# trace 生成 / 只看解析
cd sh_test_mesh/workload/llama2_7b_inference && python3 generate_trace.py
python3 generate_trace.py --print-shell-config
# 决策日志物化（replay 源；生产产物字节不变）
python3 generate_trace.py --replay-record

# 可复现 run_id
RUN_OUTPUT_LOG_TIMESTAMP=20260816_000000 bash sh_test_mesh/run_scripts/run_sh_test_aware.sh

# 在线（阶段 1 起）
python3 sh_test_mesh/workload/llama2_7b_inference/online/online_service.py \
    --bridge-dir <run_dir>/bridge --mode replay|strategy \
    --decision-log <path> --plan-dir <离线 plan 目录> &
$REPO/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware_Online \
    --online-mode replay|strategy --bridge-dir <run_dir>/bridge \
    --request-queue-csv ... --close-input --metrics-detail=off
# 或一键：bash sh_test_mesh/run_scripts/run_online_replay.sh（阶段 1-10 提供）

# Python 单元测试
cd sh_test_mesh/workload/llama2_7b_inference && python3 -m pytest test_face_scheduler.py -q

# C++ 机制单测（阶段 1）
g++ -std=c++17 -I extern/network_backend/analytical/include \
    astra-sim/workload/execution_driven/tests/event_queue_deferred_test.cc \
    extern/network_backend/analytical/common/event-queue/EventQueue.cpp \
    extern/network_backend/analytical/common/event-queue/EventList.cpp \
    -o /tmp/eq_test && /tmp/eq_test

# 等价验收（阶段 2）/ 对账（阶段 3）
python3 online/verify/tier_b_compare.py --baseline <dir> --online <dir> --layers B0,B1,B2,B3,B4
python3 online/verify/ledger_reconcile.py --online <dir>

# 字节等价门（逐文件 cmp，不用 diff -r）
for f in sh_test_mesh/baseline/20_30s/generated/*; do
    cmp "$f" "<当前generated>/${f##*/}" || echo "DIFF: $f"; done

# 红线检查（每阶段）
git diff <phase-start>..<phase-end> -- \
    template/astra-sim-sh_2.0/sh_test_mesh/workload/llama2_7b_inference/face_scheduler.py \
    template/astra-sim-sh_2.0/sh_test_mesh/workload/llama2_7b_inference/generate_face_trace.py
```
