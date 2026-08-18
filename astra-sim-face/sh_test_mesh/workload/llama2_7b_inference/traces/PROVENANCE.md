# PROVENANCE — 20.csv 前 30 秒物化输入（face 仓）

物化日期: 2026-08-16
物化脚本: `traces/derive_20_first_30_seconds.py`（本目录，单次遍历同时产出
执行队列与 canonical sidecar；与 wscllm 蓝本同一脚本同一规则——face 与
wscllm 共享同一源文件与同一派生规则，物化产物逐字节一致）。

## 源文件

- 路径: `/home/sunhao/wsc-simulator/agent-traces/tracelab/astra_compute_20.csv`
- md5: `fc74a48e874cf7798794d7bb17c426b7`（2026-08-16 实测，与蓝本 PROVENANCE 一致）
- 规模: 13,839,570 行（含表头），236,190 个 session
- 表头: `arrival_time,session_id,request_id,prefix_len,prefill_length,
  decode_length,human_time,tool_time`
- arrival_time 单位: 纳秒（ns）

## 物化规则（①-④，与 3min frozen 血统一致）

1. 窗口: session 首行 `arrival_time < 30,000,000,000 ns`（前 30 秒）才纳入；
   turn 派生到达 = 上一到达 + 上一 gap，派生到达 `>= 30e9` 即截断该 session
   后续 turn。
2. gap（interval 口径）: turn j 之后的等待 = turn j 行的
   `human_time || tool_time`（两者皆空 → 0）。interval 口径 =
   "human_time||tool_time 逐行 gap"（总体方案 §5.7 合同⑨ 二选一之一）。
3. recompute: turn-0 `prefill_length = prefix_len + prefill_length`
   （turn-0 prefix 折入 prefill）；其余 turn 不变。
4. `session_arrival_time_ns` 用源绝对时间，不做窗口归一化。

## 规则保真验证（2026-08-16，face 仓复跑）

- face 仓内脚本对源 csv 重派生 30s 窗口，与仓内
  `astra_compute_20_first_30_seconds_request_queue_recompute.csv`
  **逐字节一致**（cmp 通过）。
- 实测: **1177 请求 / 112 session**；30s 窗口内最大到达 29.988879 s；
  prefill 范围 66–169395；decode 范围 1–13812。
- 与蓝本（wscllm）物化值 1177/112 一致（同一源同一规则）。

## 产物文件（本目录）

| 文件 | 说明 |
|---|---|
| `astra_compute_20_first_30_seconds_request_queue_recompute.csv` | 8 列执行队列（session_id,turn_index,request_id,prefill_length,decode_length,session_arrival_time_ns,inter_request_interval_ns,description） |
| `astra_compute_20_first_30_seconds_canonical_sidecar.csv` | canonical sidecar，按 request_id 关联: raw_prefix_tokens / raw_new_prefill_tokens / effective_prompt_tokens(=recompute 队列 prefill_length) / prefill_length / decode_length / session_arrival_time_ns / inter_request_interval_ns / prefix_mode=recompute / digest |
| `derive_20_first_30_seconds.py` | 物化脚本（可复跑） |

digest 口径: sha256(8 列队列数据行，UTF-8，逗号连接)——B0 校验依据。
effective_prompt_tokens 口径: turn-0 = prefix_len + prefill_length，
其余 turn = prefill_length（与执行队列 prefill_length 一致）。

## 约束声明（硬约束，用户指示 2026-08-15，见 face 执行方案 §0.3）

仿真输入仅限 `agent-traces/tracelab/astra_compute_20.csv` 前 30 秒
（arrival_time < 30,000,000,000 ns，即本目录物化输入）；
**50/80/120 档与更长时间窗口禁止仿真，直至用户另行授权**。

## 附:Python 环境记录（2026-08-16）

- 解释器: `/usr/bin/python3` = 3.14.4
- 物化使用标准库 csv/hashlib，无第三方依赖
