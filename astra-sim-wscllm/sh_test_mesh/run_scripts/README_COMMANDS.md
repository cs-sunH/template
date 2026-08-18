# 命令速查(路径③④口径 2026-08-18)

来源: wscllm仓库改造详细执行方案.md 附录 B(2026-08-15 原样复制后按
request-neutral 收尾更新)。本仓库为裸仓库:不物化任何 request 队列,
正式入口缺失输入 fail-closed(见方案文档 §0.3 与 §3 步骤 0-1 物化规则)。

```bash
REPO=/home/sunhao/wsc-simulator/template/astra-sim-wscllm
cd $REPO

# request-neutral:仿真输入由调用方按方案文档 §3 步骤 0-1 物化规则自行
# 物化后,在 trace_config.csv 的 request_queue_csv(或在线 runner 的
# --request-queue-csv 参数)中显式指定;仓库不绑定默认队列,缺失输入时
# 正式入口 exit 非 0 并打印 fail-closed 信息(不落任何 stub 队列)。
# (相对路径解析基准 = sh_test_mesh/workload/,见 generate_wsc_llm_trace.py 装载段)

# 构建（全部 CMake 目标：_Online + 机制 fixtures）
cmake --build build/astra_analytical/build_congestion_aware -j

# 物化 plan 目录（runtime_config 四小件 + manifest + metrics_manifest）
cd sh_test_mesh/workload/llama2_7b_inference && python3 plan_materializer.py && cd $REPO

# ③④ 在线 runner（GEN_MATCH：generated/ 下恰一 llama2_7b_wsc_llm_inference_54npus_* 目录）
bash sh_test_mesh/run_scripts/run_online_strategy.sh <run_dir> <绝对路径 request_csv>
bash sh_test_mesh/run_scripts/run_online_strategy_sensing.sh <run_dir> <绝对路径 request_csv>
# legacy 第二变体（strategy 模式 + trace_config_legacy.csv）
bash sh_test_mesh/run_scripts/run_online_strategy_legacy.sh <run_dir> <request_csv> <legacy_gen>

# 两段式直接驱动（等价于一键 runner）
python3 sh_test_mesh/workload/llama2_7b_inference/online/online_service.py \
    --bridge-dir <run_dir>/bridge --mode strategy --plan-dir <plan 目录> &
$REPO/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware_Online \
    --online-mode strategy --bridge-dir <run_dir>/bridge \
    --request-queue-csv <绝对路径 request_csv> --metrics-detail=full

# 机制 fixtures
bash sh_test_mesh/run_scripts/run_online_idle_fixture.sh [run_root]
bash sh_test_mesh/run_scripts/run_online_wakeup_guard_fixture.sh <run_root>
bash sh_test_mesh/run_scripts/run_online_same_tick_milestone.sh [run_root]

# 后处理 + ④ 分层账本对账
bash sh_test_mesh/run_scripts/run_metrics_postprocess.sh <run_dir>/cpp.log
python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/ledger_reconcile.py \
    --bridge-dir <run_dir>/bridge \
    --manifest <run_dir>/results/online_decision_log.jsonl \
    --cpp-log <run_dir>/cpp.log

# 单元测试（双根）
cd sh_test_mesh/workload/llama2_7b_inference && python3 -m pytest test_wsc_llm_scheduler.py test_wsc_llm_legacy_online_scheduler.py test_checkpointing.py -q
cd $REPO/sh_test_mesh && python3 -m pytest tests/ -q
```

# 一键清空测试记录（删除 generated/results/online_runs/log/缓存与 traces 数据件，
# git 恢复 trace_config；保留 build 与 tracked 文件；--full 连 build 一起清）
bash sh_test_mesh/run_scripts/clean_test_records.sh [--full]
