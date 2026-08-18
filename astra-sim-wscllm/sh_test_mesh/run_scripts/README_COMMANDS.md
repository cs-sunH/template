# 命令速查(阶段 0 步骤 0-6 固化)

来源: wscllm仓库改造详细执行方案.md 附录 B(2026-08-15 原样复制后按
request-neutral 收尾更新)。本仓库为裸仓库:不物化任何 request 队列,
正式入口缺失输入 fail-closed(见方案文档 §0.3 与 §3 步骤 0-1 物化规则)。

```bash
REPO=/home/sunhao/wsc-simulator/template/astra-sim-wscllm
cd $REPO

# request-neutral:仿真输入由调用方按方案文档 §3 步骤 0-1 物化规则自行
# 物化后,在 trace_config.csv 的 request_queue_csv(或 runner 的
# --request-queue-csv 参数)中显式指定;仓库不绑定默认队列,缺失输入时
# 正式入口 exit 非 0 并打印 fail-closed 信息(不落任何 stub 队列)。
# (相对路径解析基准 = sh_test_mesh/workload/,见 generate_wsc_llm_trace.py:289-292)

# 全管线（clean + build + generate + run + postprocess，单档约 1.5-4 分钟）
bash sh_test_mesh/run_scripts/runall.sh

# 仅构建（含在线目标）
bash sh_test_mesh/run_scripts/build_analytical_aware.sh

# 仅 trace 生成（读 trace_config.csv）
cd sh_test_mesh/workload/llama2_7b_inference && python3 generate_trace.py
# 只看解析结果（不生成）
python3 generate_trace.py --print-shell-config

# 静态仿真二进制
$REPO/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware \
    --workload-configuration=... --comm-group-configuration=... --system-configuration=... \
    --remote-memory-configuration=... --network-configuration=... \
    --logging-folder=off --metrics-configuration=... --metrics-detail=full
# （完整参数以 sh_test_mesh/run_scripts/run_sh_test_aware.sh 为准）

# 在线（阶段 1 起）
python3 sh_test_mesh/workload/llama2_7b_inference/online/online_service.py \
    --bridge-dir <run_dir>/bridge --mode replay|strategy \
    --decision-log <path> --plan-dir <离线 plan 目录> &
$REPO/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware_Online \
    --online-mode replay|strategy --bridge-dir <run_dir>/bridge \
    --request-queue-csv ... --metrics-detail=full
# 或一键：bash sh_test_mesh/run_scripts/run_online_replay.sh（阶段 1-10 提供）

# 单元测试
cd sh_test_mesh/workload/llama2_7b_inference && python3 -m pytest test_wsc_llm_scheduler.py -q

# 等价验收（阶段 2）
python3 online/verify/tier_b_compare.py --baseline <baseline_dir> --online <online_dir> \
    --layers B0,B1,B2,B3,B4

# 对账（阶段 3）
python3 online/verify/ledger_reconcile.py --online <online_dir>
```
