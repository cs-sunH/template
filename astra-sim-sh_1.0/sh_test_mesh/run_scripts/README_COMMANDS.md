# sh_1.0 命令速查（阶段 0 固化；阶段 7 裸仓库态修订 2026-08-16；路径③④口径 2026-08-18）

仿真输入固定为 20.csv 前30s（用户指示 2026-08-15，见方案 §0.3）；裸仓库
态不预置输入，由调用方按 `traces/PROVENANCE.md` 物化（禁止 50/80/120 档、
trunc1M 及其他 csv、禁止更长窗口）。相对路径解析基准 =
sh_test_mesh/workload/。

```bash
REPO=/home/sunhao/wsc-simulator/template/astra-sim-sh_1.0
cd $REPO

# 0. 物化输入（裸仓库态必做；产物 md5 冻结值见 traces/PROVENANCE.md）
cd sh_test_mesh/workload/llama2_7b_inference
python3 traces/derive_20_first_30_seconds.py \
  /home/sunhao/wsc-simulator/agent-traces/tracelab/astra_compute_20.csv traces/
# 然后把 trace_config.csv 的 request_queue_csv 行指向
# traces/astra_compute_20_first_30_seconds_request_queue_recompute.csv
cd $REPO

# 1. 构建（全部 CMake 目标：_Online + 机制 fixtures）
cmake --build build/astra_analytical/build_congestion_aware -j

# 2. 物化 plan 目录（runtime_config 四小件 + manifest + metrics_manifest + face_lut.csv）
cd sh_test_mesh/workload/llama2_7b_inference && python3 plan_materializer.py && cd $REPO

# 3. ③④ 在线 runner（GEN_MATCH：generated/ 下恰一 llama2_7b_inference_54npus_* 目录）
bash sh_test_mesh/run_scripts/run_online_strategy.sh <run_dir> <绝对路径 request_csv>
bash sh_test_mesh/run_scripts/run_online_strategy_sensing.sh <run_dir> <绝对路径 request_csv>

# 4. 在线机制 fixtures
bash sh_test_mesh/run_scripts/run_online_idle_fixture.sh [run_root]
bash sh_test_mesh/run_scripts/run_online_wakeup_guard_fixture.sh <run_root>
bash sh_test_mesh/run_scripts/run_online_same_tick_milestone.sh [run_root]

# 5. 后处理 + ④ 分层账本对账
bash sh_test_mesh/run_scripts/run_metrics_postprocess.sh <run_dir>/cpp.log
python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/ledger_reconcile_sh10.py \
    --run-dir <sensing_run_dir> [--expected <物化请求数>]

# 6. 单元测试（双根；裸仓库态 24+7skip+33，物化输入后全量）
cd sh_test_mesh/workload/llama2_7b_inference && python3 -m pytest test_face_scheduler.py online/test_turn0_eviction_probe_sh10.py -q
cd $REPO/sh_test_mesh && python3 -m pytest tests/ -q

# 7. 验收/对账工具（online/verify/）
python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/graph_batch_audit.py <run_dir>...
python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/ledger_reconcile_sh10.py --run-dir <sensing_run_dir>
```

注：阶段 0-6 的基线归档（baseline/20_30s，含决策日志与基线产物 58 件）
已在阶段 7 裸仓库还原中删除；其字节等价门的证据与冻结 md5 见
sh_1.0改造执行实录.md（阶段 0/2），重建即上述 0→2 步。

# 一键清空测试记录（删除 generated/results/online_runs/log/缓存与 traces 数据件，
# git 恢复 trace_config；保留 build 与 tracked 文件；--full 连 build 一起清）
bash sh_test_mesh/run_scripts/clean_test_records.sh [--full]
