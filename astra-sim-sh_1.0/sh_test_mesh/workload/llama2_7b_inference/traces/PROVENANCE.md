# 仿真输入 provenance（20.csv 前 30 秒）

> **裸仓库态（阶段 7 还原，2026-08-16）**：物化产物（request_queue +
> canonical sidecar 两 csv）已删除，本目录只保留物化器
> `derive_20_first_30_seconds.py` 与本文件。重建（调用方执行）：
> ```bash
> cd sh_test_mesh/workload/llama2_7b_inference
> python3 traces/derive_20_first_30_seconds.py \
>   /home/sunhao/wsc-simulator/agent-traces/tracelab/astra_compute_20.csv \
>   traces/astra_compute_20_first_30_seconds_request_queue_recompute.csv \
>   traces/astra_compute_20_first_30_seconds_canonical_sidecar.csv
> # 产物 md5 冻结值见下（与 face 仓逐字节一致）；然后在
> # trace_config.csv 的 request_queue_csv 行指定物化队列路径。
> ```
> 基线归档 `sh_test_mesh/baseline/20_30s/`（含 decision_log.jsonl 与
> 静态 ET 58 件）同轮删除；重建 = 物化输入 → trace_config 指定 →
> `runall.sh`（决策日志经 `--replay-record` 重录，md5 冻结值见
> sh_1.0改造执行实录.md 阶段 0）。

- 物化日期：2026-08-16（sh_1.0 阶段 0 步骤 0-1）
- 源文件：`/home/sunhao/wsc-simulator/agent-traces/tracelab/astra_compute_20.csv`
  - md5：`fc74a48e874cf7798794d7bb17c426b7`（与 wscllm 蓝本/face 仓登记一致）
  - 表头：`arrival_time,session_id,request_id,prefix_len,prefill_length,decode_length,human_time,tool_time`（arrival_time 单位 ns）
- 物化脚本：`derive_20_first_30_seconds.py`（单次遍历同源产出执行队列 + canonical sidecar；与 face 仓同脚本逐字节一致）
- 产物：
  - `astra_compute_20_first_30_seconds_request_queue_recompute.csv`
    （md5 `ee7af9d0bc3e9c2e211d1e30c325bd45`，与 face 仓物化逐字节一致）
  - `astra_compute_20_first_30_seconds_canonical_sidecar.csv`
    （md5 `8525bf190950a33b6e6bd66c0c36c2e7`，同上）
- 实测数量：**1,177 请求 / 112 session**（与蓝本 wscllm 实测一致）；prefill 66-169395，
  decode 1-13812，窗口内最大到达 29.988879 s，timing 全部为 1000 ns 整数倍。
- 物化规则（冻结）：
  1. 窗口：session 首行 arrival_time < 30,000,000,000 ns 才纳入；turn 派生到达 =
     上一到达 + 上一 gap，≥ 窗口上限即截断该 session 后续；
  2. gap：turn j 之后等待 = turn j 行的 `human_time || tool_time`（空→0）；
  3. recompute（prefix_mode=recompute）：turn-0 prefill = prefix_len +
     prefill_length（prefix 折入，仅留 provenance）；其余 turn prefill = prefill_length；
  4. `session_arrival_time_ns` 用源绝对时间，不做窗口归一化。
  interval 口径 = human_time||tool_time 逐行 gap。
- **约束声明（用户指示 2026-08-15，最高原则）**：仿真输入仅限 20.csv 前 30 秒；
  50/80/120 档、`astra_compute_120_trunc1M.csv` 及其他一切 csv、超过 30 秒的时间窗口
  禁止仿真，直至用户另行授权。
- 血统警示：`TraceLab_*_v1/derived/` 下的 empirical 包（description 含
  "auto-generated queued session request"）是另一血统合成数据，不得作为物化参考；
  本仓 trace_config.csv 原指向的该血统路径已在阶段 0 整体替换。
- fail-closed：正式入口（`load_face_trace_config`）在 `load_request_queue` 之前检查
  输入存在性，缺失即 `sys.exit(1)`；随机 `create_default_request_queue` stub 不进正式路径。
