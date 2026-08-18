# 命令速查（face 仓，request-neutral 收尾形态）

> 本仓库为裸仓库：不物化任何 request 队列，正式入口缺失输入 fail-closed
> （已实测 exit=1）。仿真输入由调用方按方案文档 §3 步骤 0-1 物化规则
> （唯一允许源 = agent-traces/tracelab/astra_compute_20.csv 前 30 秒，
> 用户指示 2026-08-15）自行物化后，在 trace_config.csv 的
> request_queue_csv（或在线 runner 的 --request-queue-csv）显式指定。

```bash
REPO=/home/sunhao/wsc-simulator/template/astra-sim-face
cd $REPO

# 全管线（clean + build + generate + run + postprocess）
bash sh_test_mesh/run_scripts/runall.sh

# 仅构建（含在线目标，阶段 1 起）
bash sh_test_mesh/run_scripts/build_analytical_aware.sh

# 仅 trace 生成（读 trace_config.csv）
cd sh_test_mesh/workload/llama2_7b_inference && python3 generate_trace.py
# 只看解析结果（不生成）
python3 generate_trace.py --print-shell-config
# replay 决策日志物化（阶段 0 步骤 0-5 新增开关；生产路径字节不变）
python3 generate_trace.py --replay-record  # 需先物化输入（request-neutral）

# 静态仿真
bash sh_test_mesh/run_scripts/run_sh_test_aware.sh
# （run 日志文件名可用 RUN_OUTPUT_LOG_TIMESTAMP 环境变量固定，run_id 可复现）
RUN_OUTPUT_LOG_TIMESTAMP=20260816_fixed bash sh_test_mesh/run_scripts/run_sh_test_aware.sh
# 后处理
bash sh_test_mesh/run_scripts/run_metrics_postprocess.sh <run.log>

# 单元测试
cd sh_test_mesh/workload/llama2_7b_inference && python3 -m pytest test_face_scheduler.py -q
# （或 python3 -m unittest test_face_scheduler）

# 字节等价门（generated 平铺归档 vs 嵌套现行：逐文件 cmp，禁用 diff -r）
for f in sh_test_mesh/baseline/20_30s/generated/*; do \
    cmp "$f" "sh_test_mesh/generated/<label>/${f##*/}"; done

# 在线（阶段 1 起，按阶段补齐）
# python3 sh_test_mesh/workload/llama2_7b_inference/online/online_service.py ...
# $REPO/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware_Online ...
# bash sh_test_mesh/run_scripts/run_online_replay.sh
```

## 基线归档（收尾形态已删除）

- 阶段 0-7 的基线归档（baseline/20_30s、20_30s_legacy）与物化输入
  （traces/）已按裸仓库收尾删除；验证证据见 face改造执行实录.md 与
  方案文档 §15。重新物化输入后可按上述命令重建基线。
