# slo_tools —— SLO 离线后处理工具集（WP3/WP4/WP5/WP7 + Hop-Bytes）

五仓同名同构（本目录五个仓逐字节相同；逐仓语义差异全部收在脚本内的
REPO_VARIANTS / REPO_HOP_SOURCES 表，按 run_dir 自动识别 repo_variant）。
纯离线、纯标准库、只读输入；不注册仿真事件、不反向参与调度。

## 与 slo_params_manifest.json 的关系

`slo_params_manifest.json` 是 B 类参数的唯一来源（schema_version=1，五仓
逐字节相同）。当前所有 `value=null`（B4 批次按 derivation_program 引用的
主规格条款推导后填充）。**所有依赖参数的命令 fail-closed**：遇 null/缺失
即退出码 2 并指明参数名与推导条款，绝不内置示例值。`bucket_percentiles`
填充后 value 结构为 `{"percentiles":[...], "prefill_edges_tokens":[...],
"decode_edges_tokens":[...]}`（含首尾哨兵；桶 i 覆盖
`edges[i] <= x < edges[i+1]`，末桶闭合）。

## 各脚本用途 / 输入 / 输出

| 脚本 | WP | 输入（run_dir 内） | 输出（`-o`，缺省写 run_dir 下同名文件；`-` = stdout） |
|---|---|---|---|
| `slo_stats.py e2e-stats` | WP3 | request_metrics.csv | slo_e2e_stats.csv：P50/P99 主档（`--extra-pct 90,95` 附录档）；整数纳秒、先分位后转单位（nearest-rank，与 C++ MetricCollector 同法） |
| `slo_stats.py violation` | WP3 | request_metrics.csv + `--t-isolated` CSV + manifest α | slo_violation.json：`Deadline(r)=α×T_isolated(bucket(r))`；violation_rate=分子/分母（分母=全部终态 completed+rejected+dropped+timed_out+failed）；**proxy 字段显式断言拒绝**（输入带 first_token_ns/first_token_source 即 fail-closed） |
| `slo_stats.py bucket-stats` | WP3 | 同上 | slo_bucket_stats.csv：长度分桶 + slowdown=E2E/T_isolated(bucket) |
| `slo_stats.py session` | WP3 | request_metrics.csv + per-request manifest | slo_session.csv：T_session=末轮 completion−首轮 arrival−Σ(human+tool)，P50/P95 打 stderr；human/tool 均未透传 → NA 行+计数（互斥缺侧按 0） |
| `slo_stats.py backlog` | WP3 | request_metrics.csv | slo_backlog.csv：arrival(+1)/completion(−1) 事件重建在途请求数时序（同刻先减后加；`--bucket-ns` 可分桶取 max） |
| `slo_stats.py warmup` | WP3 | request_metrics.csv + manifest | slo_warmup.json：剔除前缀前后 P99 E2E 对比（窗口/门限=manifest warmup_window/warmup_change_threshold） |
| `slo_stats.py normalized` | WP3 | 多个 run_dir | slo_normalized.csv：Normalized Time/TPS（0-1，对齐比较组内 max；对齐 FACE/WSC-LLM 展示口径） |
| `slo_stats.py scan-export` | WP3 | `--point RUN_DIR=λ`（可重复）+ 可选 `--t-isolated` | slo_scan_export.csv：λ、violation_rate、tput、goodput=tput×(1−violation_rate)；**drain 完备断言 input==completed 不过则拒绝出表** |
| `load_imbalance.py` | WP7 | results/online_decision_log.jsonl + results/train_ledger.jsonl + manifest(imbalance_bucket_ns) | slo_load_imbalance.csv + stderr JSON：逐请求 (instance, admission→drain) → 各 instance 积压时序 → 时间平均 CV（总体标准差）与 Max/Mean；不做账本 dump |
| `restore_decomposition.py` | WP5 | cpp.log（**full 档**：type=memory_anchor + type=request） | slo_restore_decomposition.csv：restore_start=min(锚点)/restore_complete=max(锚点)（按 subject_id=queue_index 逐请求归属）→ §3.2 固定公式三段+hidden_ratio；无锚点请求四推导字段全 NA+计数 |
| `kv_cache_adapter.py` | WP4 | results/online_decision_log.jsonl + per-request manifest | cache_events.csv（action_id,request_id,start_ns,end_ns,bytes,source,target,cause）+ kv_hit_states.csv（full/partial/miss/no_history/not_supported + 证据列）+ 命中率双分母；`--reconcile` 与 native 逐项对账（以 native 为准，不一致 exit≠0） |
| `hopbytes.py` | — | results/online_decision_log.jsonl | slo_hopbytes_total.csv + slo_hopbytes_per_request.csv：Hop-Bytes=Σ bytes×noc_hops（只聚合产物中真实携带 hops 字段的记录；无覆盖仓输出 coverage=0 并标注 TODO_*） |

`slo_common.py` 为共享框架（manifest 装载、request_metrics.csv 冻结列序
校验、NA 语义、nearest-rank 分位、repo_variant 自动识别），不含逐仓语义。

## 逐仓映射要点（证据见脚本内注释与 REPO_VARIANTS）

* **face/wscllm**：prefill.decision.history_action 四值 → full/miss/
  no_history（LOCAL_HIT/NOC_MIGRATE→full，RECOMPUTE→miss；基线核对
  RECOMPUTE 全量重算，无 partial 语义）。
* **sh_1.0**：history_transfer.kind ∈ {local_hit, noc_migrate,
  remote_load} → full；null 且 turn=0 → no_history。shards 带 noc_path
  （hopbytes 唯一 shard 级数据源）。
* **sh_2.0**：B2wp9py（2026-08-26）起准入决策序列化
  history_location_before{,_instance_index}/history_resident_prefix_
  layers → 三态映射 local_hbm→full / partial_hbm_remote→partial /
  remote_memory→full（remote 整体恢复非重算；无 recompute，miss 不出现
  属预期）；旧产物（字段缺失）回退 not_supported。
* **sh_3.0**：prefill_affinity_reason 五值（sh30_online_scheduler.py:
  1150-1221）；resident_prefix_layers（PARTIAL_HBM_REMOTE）→ partial。
* **hopbytes 覆盖**：sh_1.0=shard 级 noc_path；sh_2.0=决策级
  transfer_hop_bytes（B2wp9py 起，coverage=1）；wscllm=实例级
  static_route.hop_count（仅 PD 迁移）；face/sh_3.0 产物无 hops
  字段 → coverage=0（TODO_FACE/TODO_S3，不臆造）。

## fail-closed 纪律

* 退出码：0=成功；2=FAIL-CLOSED（输入缺列/格式错/参数 null/对账不一致）。
* request_metrics.csv 列序与 EXECUTION_PLAN §2 冻结值逐一相等，否则报错。
* T_isolated 表缺桶、分桶边界越界、terminal_status 非法等一律非零退出。
* kv 对账（--reconcile）与 Σshards==total_bytes 不变量以 native 为准。

## 测试运行法

```bash
# 仓根目录执行（两法等价，57 例当前全过；仅标准库）
python3 sh_test_mesh/slo_tools/tests/test_slo_contract.py      # T0 契约
python3 sh_test_mesh/slo_tools/tests/test_golden_g1g4.py       # T2 golden 骨架
python3 -m unittest discover -s sh_test_mesh/slo_tools/tests -v
```

T2 golden（G1 单请求无竞争 / G2 双请求排队 / G3 session 两轮 / G4 restore
三段）目前用合成 request_metrics/manifest/anchor 样本断言手算值；**仿真侧
fixture 留待 B3 批次**接入真实 2s full 档产物后以同一断言口径复跑。

## hbm_watermark.py —— WP8 补充主数据源：离线 HBM KV 水位线重建（B2 线5）

在线模式 C++ 内存账本为空属既有现状（C++ hbm_watermark 全零为预期），真实
KV 占用时序在 python 侧产物。经五仓 B0 基线实测核对：`bridge/ledger.jsonl`
仅 sensing 档落盘且逐 request 记层、**不含 bytes**，不是占用数据源；本脚本
重放 **results/online_decision_log.jsonl**（KV 动作：恢复/迁移 bytes、逐出
条目 bytes+victim、P→D 迁移）+ **物化 plan manifest**（manifest.json，含
prefill_context_tokens/final_context_tokens/history_tokens_before）重建每
实例占用时间序列。逐仓字段差异收在脚本内 REPO_VARIANTS（先实测登记，不猜测）。

* **算法**：按文件顺序（=seq）重放，记录内固定次序「逐出 → 恢复/迁移 →
  增长」；会话所在实例与本地 bytes 由脚本跟踪，恢复方向由跟踪态判定
  （本地跨实例=搬移、远端/同实例=只增）；已落盘 bytes 与
  f(tokens)=2·layers·hidden_size·bytes_per_elem·tokens 逐条对账（五仓
  基线 restore 比值全 1.0，S3 另有 0.5 半层）。增长按 manager 语义
  "长到 f(目标 tokens)"。
* **输出**：① `slo_hbm_watermark_series.csv`（逐实例每活动桶：桶末占用/
  桶内峰值/逐出事件数/逐出 bytes——appendix 时序图数据）；②
  `slo_hbm_watermark_instances.csv`（逐实例 coverage/容量/时长/峰值/均值/
  残留/逐出/违规）；③ `--json` 汇总（动作计数、对账直方图、异常计数、
  容量链证据）。桶长 = manifest `watermark_sample_period_ns`，**null →
  5,000,000 ns 临时锚点**，全输出（JSON 字段/CSV 列/stderr）标
  `bucket_ns_provisional=true`（B4 填充后自动生效）。
* **容量链**（逐仓登记，不编造）：trace_config.csv config 行
  `local_hbm_capacity_profile` → 仓内 `sh_test_mesh/hardware/*.json` 的
  `local-hbm.capacity-profiles[<profile>].bytes`（每 NPU）×
  npus_per_instance（request manifest requests[].prefill_ranks 长度）。
  五仓 B0 均为 validation-160gib=171,798,691,840 B/NPU × 6 =
  1,030,792,151,040 B/实例。任一环缺失 → capacity=NA，违规检查降级为
  "峰值记录"并注明。
* **违规判定**：coverage=full/full_reconciled 且容量已知时，
  occupancy>capacity 的事件点计数**必须为 0**；>0 → stderr 标红 + 退出码
  3（与 fail-closed 的 2 区分）——可能是重建口径错误，也可能是真实超卖，
  宁可报错不可静默。
* **逐仓覆盖度**（B0 基线 60s_summary 实测）：
  * FACE/S1：`full`——逐出条目全量落盘（bytes+victim+实例），重放闭合，
    0 异常 0 违规（FACE 另有 9 例 decode 准入静默逐出经 RECOMPUTE 断言对账）。
  * W：`full_reconciled`——账本缺口：decode 准入期逐出
    （decode_target_evictions 在 prefill 记录落盘后才累积，decode 记录不含
    逐出列表）不落盘；重放在 `history_cache_state_before=EVICTED` 的恢复点
    按账本断言对账扣减（silent_evictions_reconciled；静默逐出真实时刻不可
    见，该窗口内占用为上界）。
  * S3：`full_reconciled`——账本缺口：completion
    `kv_location_after_completion=partial_hbm_remote` 的 suffix 半层释放不
    落盘；重放在下次恢复点按「恢复前本地 = f(h) − 恢复 bytes」对账
    （restore_prefix_reconciled）。
  * S2：B3-6（2026-08-27）起新产物逐条序列化 *_evictions（victim/bytes，
    字段名同 S1）+ history_transfers（partial 两段式恢复逐段对象）→
    字段在场即 `full`（逐出可归因、恢复逐段对账、completion 自身释放已含
    于 completion_evictions 故关闭 kv_location_after_completion 归因）；
    旧产物（B3 前基线）回退 `count_only`——只有 *_eviction_count，他人
    逐出不可归因：occupancy 为上界（未归因逐出不扣减），violation 检查
    降级，occupancy_valid=false，自身会话去向经
    kv_location_after_completion 归因（remote=全量扣减；partial=比例
    未知，保留+计数，不臆造）。
* **fail-closed**：缺 token manifest/trace_config、tick 回退、请求集两源
  不一致、重复决策、负占用、逐出对象不在跟踪态、逐出 bytes 超跟踪值、
  增长为负 → 退出码 2。软异常（source/kind/bytes 对账不符）计数入 JSON 不
  中断。
* **测试**：`tests/test_hbm_watermark.py`（16 例：手算桶时序/峰值/均值/
  逐出/违规计数、五仓语义各一例、fail-closed 六例、列序冻结）；运行法同上
  （`python3 sh_test_mesh/slo_tools/tests/test_hbm_watermark.py`）。

