# sh_1.0 命令速查（阶段 0 固化；阶段 7 裸仓库态修订 2026-08-16）

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

# 1. 全管线（clean + build + generate + run + postprocess）
bash sh_test_mesh/run_scripts/runall.sh

# 2. 仅构建（含在线目标，阶段 1 起）
bash sh_test_mesh/run_scripts/build_analytical_aware.sh

# 3. 仅 trace 生成（读 trace_config.csv）；决策日志侧通道：--replay-record
cd sh_test_mesh/workload/llama2_7b_inference && python3 generate_trace.py
python3 generate_trace.py --print-shell-config   # 只看解析结果

# 决策日志（阶段 0 步骤 0-5 观测通道；生产产物字节不变）
python3 generate_trace.py --replay-record

# 4. 静态仿真二进制（必须传 --remote-memory-configuration；完整参数见
# run_sh_test_aware.sh；RUN_OUTPUT_LOG_TIMESTAMP 环境变量可固定 run_id）

# 5. 在线 runner（阶段 1-4；ET 目录动态解析 sh_test_mesh/generated/ 下
#    唯一 llama2_7b_inference_54npus_* 目录——先跑第 1/3 步生成）
bash sh_test_mesh/run_scripts/run_online_replay.sh <run_dir> <绝对路径 request_csv> <绝对路径 decision_log.jsonl>
bash sh_test_mesh/run_scripts/run_online_strategy.sh <run_dir> <绝对路径 request_csv>
bash sh_test_mesh/run_scripts/run_online_strategy_sensing.sh <run_dir> <绝对路径 request_csv>

# 6. 单元测试（裸仓库态：双根 20+57 passed / 7 skipped；物化输入后
#    全量 26+63 passed）
cd sh_test_mesh && python3 -m pytest workload/llama2_7b_inference/test_face_scheduler.py tests/ -q

# 7. 验收/对账工具（online/verify/）
python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/graph_batch_audit.py <run_dir>...
python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/ledger_reconcile_sh10.py --run-dir <sensing_run_dir>
```

注：阶段 0-6 的基线归档（baseline/20_30s，含 decision_log 与静态 ET 58
件）已在阶段 7 裸仓库还原中删除；其字节等价门的证据与冻结 md5 见
sh_1.0改造执行实录.md（阶段 0/2），重建即上述 0→1 步。
