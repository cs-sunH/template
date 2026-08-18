# 命令速查（face 仓，request-neutral 收尾形态）

> 本仓库为裸仓库：不物化任何 request 队列，正式入口缺失输入 fail-closed
> （已实测 exit=1）。仿真输入由调用方按方案文档 §3 步骤 0-1 物化规则
> （唯一允许源 = agent-traces/tracelab/astra_compute_20.csv 前 30 秒，
> 用户指示 2026-08-15）自行物化后，在 trace_config.csv 的
> request_queue_csv（或在线 runner 的 --request-queue-csv）显式指定。
> generate_trace.py / generate_face_trace.py 的 main 为 fail-closed 拒绝桩
> （离线生成入口已删除；③④ 输入物化入口 = plan_materializer.py）。

```bash
REPO=/home/sunhao/wsc-simulator/template/astra-sim-face
cd $REPO

# 构建（全部 CMake 目标：_Online + 机制 fixtures）
cmake --build build/astra_analytical/build_congestion_aware -j

# 物化 plan 目录（runtime_config 四小件 + manifest.json + metrics_manifest.json；
# 输入队列需先按上方物化规则落地）
cd sh_test_mesh/workload/llama2_7b_inference && python3 plan_materializer.py && cd $REPO

# ③④ 在线 runner（GEN_MATCH：generated/ 下恰一 llama2_7b_inference_54npus_* 目录）
bash sh_test_mesh/run_scripts/run_online_strategy.sh <run_dir> <绝对路径 request_csv>
bash sh_test_mesh/run_scripts/run_online_strategy_sensing.sh <run_dir> <绝对路径 request_csv>
# legacy 第二变体（strategy 模式 + trace_config_legacy.csv；<legacy_gen> = 调用方物化的 legacy plan 目录）
bash sh_test_mesh/run_scripts/run_online_strategy_legacy.sh <run_dir> <request_csv> <legacy_gen>

# 在线机制 fixtures（③④ 机制回归）
bash sh_test_mesh/run_scripts/run_online_idle_fixture.sh [run_root]
bash sh_test_mesh/run_scripts/run_online_wakeup_guard_fixture.sh <run_root>
bash sh_test_mesh/run_scripts/run_online_same_tick_milestone.sh [run_root]
bash sh_test_mesh/run_scripts/bridge_race_stress_repro.sh

# 后处理（[METRIC] 行已在 <run_dir>/cpp.log）
bash sh_test_mesh/run_scripts/run_metrics_postprocess.sh <run_dir>/cpp.log

# ④ 分层账本对账
python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/ledger_reconcile.py \
    --bridge-dir <run_dir>/bridge \
    --manifest <run_dir>/results/online_decision_log.jsonl \
    --cpp-log <run_dir>/cpp.log

# 单元测试（双根）
cd sh_test_mesh/workload/llama2_7b_inference && python3 -m pytest test_face_scheduler.py test_checkpointing.py -q
cd $REPO/sh_test_mesh && python3 -m pytest tests/ -q
```

## 基线归档（收尾形态已删除）

- 阶段 0-7 的基线归档（baseline/20_30s、20_30s_legacy）与物化输入
  （traces/）已按裸仓库收尾删除；验证证据见 face改造执行实录.md 与
  方案文档 §15。重新物化输入后可按上述命令重建基线。

# 一键清空测试记录（删除 generated/results/online_runs/log/缓存与 traces 数据件，
# git 恢复 trace_config；保留 build 与 tracked 文件；--full 连 build 一起清）
bash sh_test_mesh/run_scripts/clean_test_records.sh [--full]
