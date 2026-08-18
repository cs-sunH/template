# 20.csv 前 30 秒物化输入 PROVENANCE（sh_2.0）

- 物化日期：2026-08-16
- 源文件：`/home/sunhao/wsc-simulator/agent-traces/tracelab/astra_compute_20.csv`
  - md5：`fc74a48e874cf7798794d7bb17c426b7`
  - 列：`arrival_time,session_id,request_id,prefix_len,prefill_length,decode_length,human_time,tool_time`（arrival_time 单位 ns）
- **仿真输入约束（用户指示 2026-08-15）**：本目录输入仅允许
  `astra_compute_20.csv` 前 30 秒（`arrival_time < 30,000,000,000 ns`）。
  禁止 50/80/120 档、`trunc1M` 及其他 csv，禁止更长窗口。
- 物化脚本：`materialize_first_30s.py`（单次遍历同源产出三份文件）
- 产物：
  1. `astra_compute_20_first_30_seconds_request_queue.csv`（8 列执行队列；
     `prefill_length` = 源新增 token 数，**不折入 prefix**——sidecar_restore 口径）
  2. `astra_compute_20_first_30_seconds_request_context.csv`（5 列 context
     sidecar：`prefix_tokens` = 源 `prefix_len`，
     `input_tokens_total = prefix_tokens + prefill_length`，逐行，含 turn>0；
     列集与 `load_request_prefix_tokens` 校验对齐）
  3. `astra_compute_20_first_30_seconds_canonical_digest.csv`（canonical
     digest：`sha256(request_id|raw_prefix_tokens|raw_new_prefill_tokens|
     effective_prompt_tokens)`，prefix_mode=sidecar_restore）
- 物化规则（沿用 3 分钟 frozen 血统，template_back/history/20_3mins）：
  1. 窗口：session 首行 arrival_time < 30e9 才纳入；turn 派生到达 =
     上一到达 + 上一 gap（gap = human_time 非空取之，否则 tool_time 非空
     取之，否则 0）；派生到达 >= 30e9 即截断该 session 后续 turn。
  2. `session_arrival_time_ns` 用源绝对时间（仅 turn-0 填写），不做归一化。
  3. interval 口径 = human_time||tool_time 逐行 gap（turn>0 填写）。
  4. queue/sidecar 双文件同一次遍历产出，防止口径漂移；禁止双计 prefix。
- 物化实测：**112 session / 1177 请求**（与蓝本同规则 30s 物化值一致）；
  timing 字段全部满足 `% 1000 == 0`（loader 校验通过）；
  最大派生到达 25,959,142,000 ns（< 30e9）。
- 标定常数（合同⑨，物化期从冻结输入按离线同一推导导出；离线/在线共用；
  禁止在线运行时增量统计）：
  - `average_decode_length = 459.4944774851317`（sum/len 全量 mean）
  - `max_d_token = 53924`（PREFILL_RANGE 上界 3-53924，p3-53924 即
    prefill_length 全距；max prefill_length = 53924）
  - `request_count = 1177`
- sidecar on/off：`trace_config.csv:13` 的 `request_queue_context_csv` 默认
  空即 off（当前 checked-in 语义：窗口内 session 首请求无历史 KV）。sidecar
  文件无论 on/off 都物化并存档（canonical 输入的一部分）。开启 sidecar 属
  实验配置变化，须经用户裁决并在合同⑧登记。
- planner pickle 缓存提示：缓存键含 configuration_digest（随输入变化自动
  失效），旧缓存无需手工清理；metrics-on 运行绕过缓存强制重建。
- 复验登记（2026-08-16 回灌修复轮）：重跑物化器，产物确定性复核——
  queue md5 `6aaf28365c783c01630278e2e7ce2c89`、context md5
  `25b7da357cac249c14bf4a3ae26ddfd4`、digest md5
  `476a02d0b276bc66abda32663b74a173`；112/1177 与标定常数不变。
  注意 configuration_digest（trace_config 字节级）随 :12 描述文案变化，
  generated 目录名因此不同（runner 已改 glob 动态解析，不受影响）。
