# 20.csv 前 30 秒物化输入 PROVENANCE（sh_3.0，重建版）

- 重建日期：2026-08-16（回灌修复轮；阶段 7 裸仓库还原曾删除本目录的
  物化器与 PROVENANCE，本次按 sh_2.0 仓同款"保留重建入口"约定重建）
- 源文件：`/home/sunhao/wsc-simulator/agent-traces/tracelab/astra_compute_20.csv`
  - md5：`fc74a48e874cf7798794d7bb17c426b7`
  - 列：`arrival_time,session_id,request_id,prefix_len,prefill_length,decode_length,human_time,tool_time`（arrival_time 单位 ns）
- **仿真输入约束（用户指示 2026-08-15）**：本目录输入仅允许
  `astra_compute_20.csv` 前 30 秒（`arrival_time < 30,000,000,000 ns`）。
  禁止 50/80/120 档、`trunc1M` 及其他 csv，禁止更长窗口。
- 物化脚本：`materialize_20_30s.py`（重建版；规则①-④与阶段 0 原版
  同源——单遍历产出三件套，queue description 文案为冻结 md5 反推确认的
  原版措辞）。
- 产物（重建后与阶段 0 冻结产物 **逐字节一致**，md5 为证）：
  1. `astra_compute_20_first_30_seconds_request_queue.csv`
     - md5 `8b615b5a12b15143a10bbbf13f31a6bb`（实录步骤 0-1 冻结值）
  2. `astra_compute_20_first_30_seconds_request_context.csv`（context
     sidecar；sh_3.0 **sidecar_restore 加载**——trace_config :12/:13 两行
     都指向本目录）
     - md5 `25b7da357cac249c14bf4a3ae26ddfd4`（实录步骤 0-1 冻结值）
  3. `astra_compute_20_first_30_seconds_canonical_digest.csv`
     （`sha256(request_id|raw_prefix_tokens|raw_new_prefill_tokens|
     effective_prompt_tokens)`，prefix_mode=sidecar_restore；与 sh_2.0
     同构产物 md5 `476a02d0b276bc66abda32663b74a173` 同源）
- 物化规则（沿用 3 分钟 frozen 血统，template_back/history/20_3mins）：
  1. 窗口：session 首行 arrival_time < 30e9 才纳入；turn 派生到达 =
     上一到达 + 上一 gap（gap = human_time 非空取之，否则 tool_time 非空
     取之，否则 0）；派生到达 >= 30e9 即截断该 session 后续 turn。
  2. queue 保持源 `prefill_length`（仅新增 token，不折入 prefix）；
     sidecar 交付 `prefix_tokens` = 源 `prefix_len`、
     `input_tokens_total = prefix_tokens + prefill_length`（逐行，含
     turn>0）；禁止双计 prefix。sidecar_restore 生效时 turn-0 prefill
     workload = input_tokens_total。
  3. `session_arrival_time_ns` 用源绝对时间（仅 turn-0 填写），不归一化。
  4. interval 口径 = human_time||tool_time 逐行 gap（turn>0 填写）。
- 物化实测（与阶段 0 登记一致）：**112 session / 1177 请求**；
  turn-0 prefix>0 行 79；max input_tokens_total = 169,395（generated 目录
  名 `p80-169395` 即此口径）；timing 字段全部 `% 1000 == 0`。
- 标定常数（合同⑨）：`average_decode_length = 459.4944774851317`、
  `max_d_token = 53924`、`request_count = 1177`。
- 重建入口：`python3 traces/materialize_20_30s.py`（源路径可作 argv[1]
  覆盖）；产出后 `trace_config.csv` :12 指向 queue、:13 指向 context
  sidecar。裸仓库态两行均为占位（request-neutral），物化输入不入库。
