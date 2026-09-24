# slo_tools —— SLO 离线后处理工具集（WP3/WP4/WP5/WP7 + Hop-Bytes）

共享工具保留逐仓变体识别；joint 仓的 `load_imbalance.py`、`domain_metrics.py`
及其测试已有 joint 专属修订，不能再按四仓逐字节相同处理。其余逐仓语义
由脚本内的 REPO_VARIANTS / REPO_HOP_SOURCES 表按 run_dir 识别。
原 sanctioned 例外 tests/run_golden_live.py 已
删除——2026-09-23 P11 死依赖清除：其硬依赖仓外 /tmp/slo_wps/
set_trace_pointer.py，golden live 流程不可运行，golden 语义由
tests/test_golden_g1g4.py 离线承担；见"测试运行法"节）。
纯离线、纯标准库、只读输入；不注册仿真事件、不反向参与调度。

## cpp.log 回退顺序（归档兼容，P1/2026-08-28）

所有读 cpp.log 的入口（`slo_common.read_init_record`——各工具的
repo_variant 识别与 manifest 定位都经它；以及 `restore_decomposition.py`）
统一经 `resolve_cpp_metric_log(run_dir)` 取路径，按序探测：

1. `cpp.log`（未归档 / `SH_ARCHIVE_RUN=0` 的 run_dir）；
2. `metrics.log`（归档后常驻——`archive_run_outputs.sh` 对 cpp.log
   `[METRIC]` 行的无损抽取）；
3. `cpp.log.gz`（归档后全量原文，gzip 文本模式读）。

三条路径读到的 `[METRIC]` 记录集合同一——同一 run_dir 归档前后跑
slo_tools 输出逐字节一致（2s 冒烟全量对拍验证）。runner（P2）还会把
per-request manifest（`metrics_manifest.json`/`manifest.json`）拷入
run_dir 根：`load_request_manifest` 现有优先级 `--request-manifest` >
`run_dir/metrics_manifest.json` > cpp.log init 行指向的仓内 generated/
副本，故拷入后 run_dir 自包含、与仓还原（裸仓删 generated/）状态解耦。
`run_scripts/run_slo_postprocess.sh`（P3）在 runner 尾部自动调用本目录
工具产 9 步最细粒度产物（清单/失败语义见其头注；不传任何分桶/聚合参数；
hbm_watermark 一步产三件：intervals/plot_series/instances，P1/2026-08-30）。

## 单遍合并 driver（A4，2026-08-29）

`slo_postprocess_driver.py` 把上述 9 步合并为**单进程单遍**执行：
RunContext 一次装载共享输入（request_metrics.csv/decision log/
manifest 族/[METRIC] 流各恰读一次，解释器启动 9→1），decision log 单遍
流按固定次序（kv → load → hop → watermark）喂给四个消费者；各步 stderr
整块捕获后按步序回放，`slo_postprocess.log` 与逐工具串行**逐字节一致**
（含 run:/ok:/FAIL: 行与 `slo_postprocess.FAIL` 条目语义、G1-G5 门控与
跳过说明行——门控在 shell 探测后经 `SH_SLO_HAVE_RM`/`SH_SLO_HAVE_TL`/
`SH_SLO_SKIP_HBM` 传入，裸调用 driver 时自探测并复刻同款说明行）。
工具 CLI 零破坏：七个脚本仍可独立调用，driver 只 import 复用其
prepare/consume/emit 拆分面。需排序分位/事件的排序经
`slo_common.BoundedSorter` 有界外部排序（内存合同：chunk 容量 =
env `SH_SLO_SORT_CHUNK`，缺省 65536 元素；满则排序后 pickle spill 临时
文件 + heapq.merge 多路归并；只用于整数/整数元组全序，输出序列 ≡
sorted()；需稳定序的调用方自带单调序号列），nearest-rank 公式不动。
等价性验证：`tests/test_driver_parity.py`（BoundedSorter 单测 + 合成
run_dir 上"旧链复刻 vs 单遍 driver"13 产物+日志逐字节对拍，含 G3/失败
路径变体）。

## 与 slo_params_manifest.json 的关系

`slo_params_manifest.json` 是 B 类参数的唯一来源（schema_version=1；2026-09-23
P11 死键清除起本仓删除 10 个零消费者 campaign 档案键——五仓逐字节相同冻结
待主控统一同步后恢复，删除键清单见 manifest `_note`）。当前所有 `value=null`（B4 批次按 derivation_program 引用的
主规格条款推导后填充）。**所有依赖参数的命令 fail-closed**：遇 null/缺失
即退出码 2 并指明参数名与推导条款，绝不内置示例值。`bucket_percentiles`
填充后 value 结构为 `{"percentiles":[...], "prefill_edges_tokens":[...],
"decode_edges_tokens":[...]}`（含首尾哨兵；interior edges 为各左桶
闭上界——如 decode=91 归 d1、prefill=415 归 p1，末桶无上限、
吸收 x > edges[-1]；2026-09-05 口径裁决，与 campaign_common.BucketGrid
语义统一，edges 数组值未变）。

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
| `load_imbalance.py` | WP7 | results/online_decision_log.jsonl + results/train_ledger.jsonl + manifest(imbalance_bucket_ns) | slo_load_imbalance.csv + stderr JSON：joint 以 `completion.tick` 作实际完成端点，train_ledger exits 仅核对成员与实例；缺 completion 拒绝输出。历史非 joint run 的终点仍为列车发射时刻代理，标注 `legacy_train_ledger_emit`。逐请求区间生成各实例时间平均 CV（总体标准差）与 Max/Mean；不做账本 dump |
| `restore_decomposition.py` | WP5 | cpp.log（**full 档**：type=memory_anchor + type=request） | slo_restore_decomposition.csv：restore_start=min(锚点)/restore_complete=max(锚点)（按 subject_id=queue_index 逐请求归属）→ §3.2 固定公式三段+hidden_ratio；无锚点请求四推导字段全 NA+计数 |
| `kv_cache_adapter.py` | WP4 | results/online_decision_log.jsonl + per-request manifest | cache_events.csv（action_id,request_id,start_ns,end_ns,bytes,source,target,cause）+ kv_hit_states.csv（full/partial/miss/no_history/not_supported + 证据列）+ 命中率双分母；`--reconcile` 与 native 逐项对账（以 native 为准，不一致 exit≠0） |
| `hopbytes.py` | — | results/online_decision_log.jsonl | slo_hopbytes_total.csv + slo_hopbytes_per_request.csv：Hop-Bytes=Σ bytes×noc_hops（只聚合产物中真实携带 hops 字段的记录；无覆盖仓输出 coverage=0 并标注 TODO_*） |
| `domain_metrics.py`（C16，joint 专属条件步） | WP6b | results/online_decision_log.jsonl（joint_admission/completion 行，C5 schema）+ per-request/plan manifest + trace_config.csv + hardware json + manifest(domain_delta_adm_ns/domain_delta_sensitivity_multipliers) | slo_domain_requests.csv（三口径逐请求：D_feed/D_econ(δ=0)/实际选择 + 边界对拍 + H/I/F 静态参考组件）+ slo_domain_instances.csv（逐 (请求,实例)：双域同图数据/内外等值线差值/方向/瓶颈段/配额域派生 0/1 列与 port_* port_snapshot 实测值）+ slo_domain_summary.json（四动作计数 recompute elected/forced 分列、|D|/hop 分位/方向半径方差/约束不满足原因、预测/实测差、δ∈{σ̂,2σ̂,4σ̂} 离线敏感性重定价（A3'：δ_adm=0 空转对拍臂已撤销）、home 迁移轨迹（completion 披露 + kv_delta_journal 结算权威层——sidecar 第四键已序列化并由本工具四层分级消费，F3 2026-09-22）、ε/δ 口径登记与叙事纪律注释）；**无 joint_admission 行的 run 完全静默跳过**（四仓共链零扰动） |

`slo_common.py` 为共享框架（manifest 装载、request_metrics.csv 冻结列序
校验、NA 语义、nearest-rank 分位、repo_variant 自动识别），不含逐仓语义。

joint run 的 `trace_config.csv.snapshot` 与 `hardware_config.json.snapshot`
由运行入口归档。`domain_metrics.py` 推导 ρ 时按显式 CLI 参数、run 快照、
run 内旧命名副本的次序读取；缺 run 本地证据则 ρ 为 NA 并告警，不回退当前
checkout 的配置，以免历史结果随仓内配置变动而漂移。

## domain_metrics.py —— 局部统一内存域三口径（C16/WP6b，2026-09-22；joint）

设计文档《joint机制改造方案_局部统一内存域》§5.1-§5.3 的测量工具化：
三类观测**必须分开**（F13）——D_feed（预测供给，remote-read 候选
applicable 的实例集，逐成员标注本地参考与性能容忍度）/ D_econ（预测经济，
C_alt* = min 遍历**全部实例的适用非 remote 动作（必含 copy）**，
D_econ = {e: C_remote(e) ≤ C_alt*+δ}；同位置动作比较另报、口径单列）/
实际选择与 measured（在线只有最终 (e,a)；四动作计数 recompute 仅 elected、
forced(no_history/quota_deferred/evicted_permanent 三成因，K7-① 扩枚举)
单列，§19.2 两口径不得混报；独立同
状态 measured 集合不在本产物=C4b 受控测量，如实标 NA 不与预测混合）。
派生：|D|、hop 分位、方向分布（分方向半径方差）、瓶颈资源（breakdown
六段 argmax）、约束不满足原因、预测/实测差异；双域同图数据（D_econ
等值线 vs 配额准入域——**两域之差即配额作用量**；N6 口径消歧：
quota_admissible/quota_admissible_remote 列 = **配额前结构域**（候选
applicable ∨ quota_ 前缀不可行理由派生，"移除配额后结构可行"），
实际配额准入域 = remote_applicable/applicable 列，配额前成本等值线
输入 = remote_cost_pre_quota_ns 列，run 级配额作用量 =
quota_counterfactual_flips（K7/P2-11 2026-09-23：C11 后
inapplicable_reason 可导出，恒 NA 的 WP3 前注记废止；port_snapshot/
flow_snapshot 实测值 C11 已接通——quota-on 臂有数值、off 态 NA 保持，
"仍为 C5 冻结 NA 占位"系过时表述 N13 订正）；内等值线（C_remote=C_stay_ref）/外等值线（C_remote=C_alt*）
逐 (请求,实例) 导出。边界对拍纪律：d≤ρ 锚点**仅历史扫描项平价**；
remote/copy 分界**不设单变量理论线**（V_copy≈[H+I+b_c−F]+/V_remote≈
[I+b_r−F]+ 仅容量静态参考，b_c/b_r 未随 schema 披露 → 锚点 NA、组件
H/I/F 分列）；**无 n\* = d′/(d−ρ) 式**；对拍是诊断不是验收（偏差本身
是模型诊断结果）。δ 主臂 = manifest `domain_delta_adm_ns`（0，与在线
δ_adm 终冻同值同域 ns）；敏感性臂 `domain_delta_sensitivity_multipliers`
（{1,2,4}×σ̂）为 A3' 强制修订——撤销 δ=0 空转对拍臂，σ̂ 三级链（观测
|pred−measured| p50 → 冷启动 merge 腿量级锚点）见 estimate_sigma_hat。
预测 `joint_cost_ns` 的终点是 merge_done：有 merge 流时与决策日志的
`merge_done.tick` 配对；无 merge 流时用 `completion.tick`，服务响应时延
另列。completion 披露了 merge 字节而缺 merge_done 行时，实测误差为 NA
并告警，避免把提前的 service_done 当作结算终点压低 σ̂。
home 迁移轨迹：completion 行（决策时刻）+ kv_delta_journal（结算时刻，
两源不混）——journal 已随 run 末 ledger 侧车序列化（`kv_delta_journal`
第四键，F3 2026-09-22 修复批），本工具按四层可信度分级消费：
settlement_full_join > settlement_partial_join > settlement_empty >
decision_log_only（certified 层的 checksum 证书本仓不产出，tier_note
如实降级不冒认；seq 链断裂 fail-closed——生产端逐键独立导出，单键
失败落 `kv_delta_journal_export_error` 哨兵、其余键不受连坐，消费端
识别哨兵/顶层非对象/键值非列表（值级 schema 损坏，H7 值级贯彻）均
按 decision_log_only 层 + 明示注记处理，A12'
2026-09-22 复审批）。叙事纪律内置：饱和下域由
配额关闭而非 argmin 涌现，"涌现域"表述限定配额界内；域是解释与可选
准入分析对象，不裁剪任何动作/实例候选。非 joint run 完全静默跳过
（字节对拍契约零扰动）。

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
* **hopbytes 覆盖**（四仓同步收口后，各变体采集器已合并进同一
  REPO_HOP_SOURCES 表）：sh_1.0=shard 级 noc_path/noc_hops；
  wscllm=实例级 static_route.hop_count + history 迁移 noc_hops
  （B2wp9py 起）；face=per-TP-shard hops 列表（B2wp9py 起）；
  sh_2.0=决策级 transfer_hop_bytes（WP9-线5 起）；sh_3.0=completion_
  evictions shards[].noc_hops（B4 起）。四仓新产物均可覆盖；旧产物按
  字段缺席回退 bytes_without_hops/coverage=0（不臆造 hop 数）。

## fail-closed 纪律

* 退出码：0=成功；2=FAIL-CLOSED（输入缺列/格式错/参数 null/对账不一致/
  journal 损坏）；3=容量违规——仅 hbm_watermark 的 per_rank_total_hbm_
  certified 层（journal+checksum 权威重放的逐 rank physical 认证；
  upper_bound_only 上界层超限只诊断、exit 0）。
* request_metrics.csv 列序与 EXECUTION_PLAN §2 冻结值逐一相等，否则报错。
* T_isolated 表缺桶、分桶边界越界、terminal_status 非法等一律非零退出。
* kv 对账（--reconcile）与 Σshards==total_bytes 不变量以 native 为准。

## 测试运行法

```bash
# 仓根目录执行（2026-09-24 实测 discover 171 例、1 skipped；仅标准库）
# 单文件命令用于定位，discover 汇总整个 tests/ 目录。
python3 sh_test_mesh/slo_tools/tests/test_slo_contract.py      # T0 契约
python3 sh_test_mesh/slo_tools/tests/test_golden_g1g4.py       # T2 golden 骨架
python3 -m unittest discover -s sh_test_mesh/slo_tools/tests -v
python3 sh_test_mesh/slo_tools/tests/test_hbm_watermark.py     # WP8 手算
python3 sh_test_mesh/slo_tools/tests/test_driver_parity.py     # A4 driver 对拍
```

T2 golden（G1 单请求无竞争 / G2 双请求排队 / G3 session 两轮 / G4 restore
三段）用合成 request_metrics/manifest/anchor 样本断言手算值
（`tests/test_golden_g1g4.py` 离线承担全部 golden 语义）。

**golden live runner 退役（2026-09-23 P11 死依赖清除）**：原
`tests/run_golden_live.py` 硬依赖仓外 `/tmp/slo_wps/set_trace_pointer.py`
（不存在），live 流程不可运行——文件已删除、其 2026-08-28 sanctioned
例外裁决随之失效（登记开关清单）。原仓族适配边界中**仍然有效**的约束：

* **必须四仓一致**：G1-G4 各场景的断言语义与期望值推导口径——如 G1 的
  e2e=queue+prefill+gap+decode 整数恒等、G2 的 queue_ns≈占位者 prefill、
  G3 的外生等待（human/tool interval）扣除方式与 request_type 透传、
  G4 的 restore 三段和=总时长与锚点 min/max 归属（现约束对象为
  test_golden_g1g4.py 的合成样本断言）。改这些断言时四仓
  必须同步评审，防止某仓的期望值推导悄然漂移。

## hbm_watermark.py —— WP8 补充主数据源：离线 HBM KV 水位线重建（B2 线5；P1 四层可信度改造 2026-08-30）

在线模式 C++ 内存账本为空属既有现状（C++ hbm_watermark 全零为预期），真实
KV 占用时序在 python 侧产物。经五仓 B0 基线实测核对：`bridge/ledger.jsonl`
仅 sensing 档落盘且逐 request 记层、**不含 bytes**，不是占用数据源；占用重放
的输入 = **results/online_decision_log.jsonl**（KV 动作：恢复/迁移 bytes、
逐出条目 bytes+victim、P→D 迁移）+ **物化 plan manifest**（manifest.json，含
prefill_context_tokens/final_context_tokens/history_tokens_before），以及
（阶段2 起）**results/kv_delta_journal.jsonl** 权威逐 rank delta 账本。
逐仓字段差异收在脚本内 REPO_VARIANTS（先实测登记，不猜测）。

* **四层可信度**（tier 由 run_dir 内容自动判定，log 一行说明依据，无新
  CLI 参数）：
  * `per_rank_total_hbm_certified`——journal + checksum json 都在场且四道
    门全过（sha256 相符/行级链自洽/重放终态==证书终态/守恒四项 checks 全
    true）：流式重放 journal 得逐 rank physical=weight+resident+reserved
    时序，对行内 capacity_bytes 逐 rank 认证——**正式容量判决只在本层给
    出**，违规 → stderr 标红 + 退出码 3（fail-loud）。
  * `resident_kv_exact`——journal 在场、链自洽，checksum 缺失：逐 rank
    时序精确但无 run 末守恒证书，容量检查如实报告、不作正式判决。
  * `lifecycle_replay_exact`——journal+checksum 在场、链与终态一致，但守
    恒 checks 有 false（run 末守恒未过）：生命周期精确、无证书，仅报告。
  * `upper_bound_only`——journal 缺失（阶段2 前全部旧 run）：decision-log
    重放对真实占用是**上界**（terminal 退休/decode 准入逐出两缺口不落盘
    → 17.24TB 级幻影）。**不得输出物理违规认证**：occupancy_valid 恒
    false、退出码 3 废除（超限只作诊断计数并标注 tier、退出码 0）——旧
    run 重跑得本层+诊断是预期语义，不是回归。
  * journal sha256 不符/行级链断裂/终态与证书矛盾 → fail-closed 退出码 2
    （账本损坏不得静默降级）。journal 在场时 decision-log 重放照常执行，
    instances CSV 的 `upper_bound_peak_occupancy_bytes` 列与 summary 的
    `decision_log_upper_bound_comparison` 块输出对照值（差异 = 账本缺口
    的直接可视化）。
* **算法**：decision-log 按（文件顺序=seq）重放、journal 按 sequence 流式
  重放；记录内固定次序「逐出 → 恢复/迁移 → 增长」；会话所在实例与本地
  bytes 由脚本跟踪，恢复方向由跟踪态判定（本地跨实例=搬移、远端/同实例
  =只增）；已落盘 bytes 与 f(tokens)=2·layers·hidden_size·bytes_per_elem
  ·tokens 逐条对账（五仓基线 restore 比值全 1.0，S3 另有 0.5 半层）。
  增长按 manager 语义"长到 f(目标 tokens)"。
* **事件流 stats（P1-②）**：peak/mean/residual/violation 在变点（RLE）
  归并时 O(动作数) 内存计算，与 span/桶长彻底解耦——同事件流换任意桶长
  stats 逐字段不变（单测断言）。
* **输出**：① `slo_hbm_intervals.csv`（权威 RLE 变点区间：per instance
  区间起止 tick、起止占用、区间内峰值、逐出叠加；无损、O(动作数) 行，
  可从它恢复任意桶长序列——单测断言无损恢复）；②
  `slo_hbm_plot_series.csv`（绘图产物，**取代旧 slo_hbm_watermark_
  series.csv**）：全局行预算 R=manifest `watermark_series_row_budget`
  （5,000,000）约束，B_eff = max(B_requested, ceil(S·N/(R−N)))（S=全局
  span=末 KV 事件−首 KV 事件、N=有事件实例数）；R≤N 拒绝稠密输出只给
  RLE（正常完成+log 说明）；流式边算边写（桶行随游标推进逐行落盘，内存
  O(实例数+活跃游标)，series 行字典全量物化已消灭）；③
  `slo_hbm_watermark_instances.csv`（逐实例 tier/coverage/容量/桶长元
  数据/时长/峰值/均值/残留/逐出/违规 + 上界对照列）；④ `--json` 汇总。
  桶长 = manifest `watermark_sample_period_ns`，null → 5,000,000 ns 临时
  锚点并全输出标 `bucket_ns_provisional=true`（该字段语义=缺正式锚点，
  与行预算调整元数据 resolution_adjusted/adjustment_reason/row_budget/
  span_ns/bucket_origin_ns 分立，不得混用）。
* **容量三口径（P1-③，summary `capacity_calibers` 分列；逐 rank 剖面函数
  逐字拷贝自 workload/llama2_7b_inference/session_kv_manager.py 的
  model_weight_shard_bytes_by_tp_rank(:185)/kv_cache_shard_bytes_for_
  tokens(:220)，同步义务见脚本内注记）**：
  1. 正式认证：逐 rank physical=weight+resident+reserved ≤ capacity_bytes
     （数据源=journal 行；仅 certified 层构成判决）；
  2. resident 硬上限：reservation=0 时任意时刻成立 =
     min_r ⌊(capacity−weight_r)/kv_r⌋ token × Σkv_r（llama2_7b/swiglu/
     TP6/160GiB 锚定 1,723,864 token = **903,801,208,832 B**；journal 模式
     下逐 rank 超限计数为诊断口径）；
  3. 水位目标：kv_reserve_context_tokens(1M/rank) 扣减后同式（锚定
     723,864 token = **379,513,208,832 B**）——非任意时刻上限，**仅报告
     不作判决**。
* **容量链**（逐仓登记，不编造）：trace_config.csv config 行
  `local_hbm_capacity_profile` → 仓内 `sh_test_mesh/hardware/*.json` 的
  `local-hbm.capacity-profiles[<profile>].bytes`（每 NPU）×
  npus_per_instance（request manifest requests[].prefill_ranks 长度）。
  五仓 B0 均为 validation-160gib=171,798,691,840 B/NPU × 6 =
  1,030,792,151,040 B/实例。任一环缺失 → capacity=NA，违规检查降级为
  "峰值记录"并注明。
* **退出码**：0 正常（含上界/无证书层的超限诊断）；2 fail-closed（缺文
  件/缺列/结构错/重放不一致/journal 损坏）；3 容量违规——**仅
  per_rank_total_hbm_certified 层**的逐 rank physical > capacity_bytes。
* **逐仓覆盖度**（decision-log 重放路径的口径；B0 基线 60s_summary 实测）：
  * FACE/S1：`full`——逐出条目全量落盘（bytes+victim+实例），重放闭合，
    0 异常 0 违规（FACE 另有 9 例 decode 准入静默逐出经 RECOMPUTE 断言对账）。
  * W：`full_reconciled`——账本缺口：decode 准入期逐出
    （decode_target_evictions 在 prefill 记录落盘后才累积，decode 记录不含
    逐出列表）不落盘，full_tracelab 本 run 实测 7,923 例
    RECOMPUTE(state_before=EVICTED) 隐含静默逐出（802 为 FACE 基线旧数）；
    重放在 `history_cache_state_before=EVICTED` 的恢复点按账本断言对账扣减
    （silent_evictions_reconciled；静默逐出真实时刻不可见，该窗口内占用
    为上界）。evict_bytes 列在 full/full_reconciled 均输出真实值（P1-⑤；
    NA 仅保留 count_only 档）。
  * S3：`full_reconciled`——账本缺口：completion
    `kv_location_after_completion=partial_hbm_remote` 的 suffix 半层释放不
    落盘；重放在下次恢复点按「恢复前本地 = f(h) − 恢复 bytes」对账
    （restore_prefix_reconciled）。
  * S2：B3-6（2026-08-27）起新产物逐条序列化 *_evictions（victim/bytes，
    字段名同 S1）+ history_transfers（partial 两段式恢复逐段对象）→
    字段在场即 `full`；旧产物（B3 前基线）回退 `count_only`——只有
    *_eviction_count，他人逐出不可归因：occupancy 为上界（未归因逐出不
    扣减），自身会话去向经 kv_location_after_completion 归因（remote=全量
    扣减；partial=比例未知，保留+计数，不臆造）。
* **fail-closed**：缺 token manifest/trace_config、tick 回退、请求集两源
  不一致、重复决策、负占用、逐出对象不在跟踪态、逐出 bytes 超跟踪值、
  增长为负 → 退出码 2。软异常（source/kind/bytes 对账不符）计数入 JSON 不
  中断。
* **测试**：`tests/test_hbm_watermark.py`（39 例：手算桶时序/峰值/均值/
  逐出/违规计数、四仓语义各一例、fail-closed 六例、列序冻结、四层 tier
  判定（含"聚合过单 rank 超"锚定用例）、三口径锚定数值、stats 多桶长
  不变、RLE 无损恢复、行预算/B_eff、真 journal 对拍）；运行法同上
  （`python3 sh_test_mesh/slo_tools/tests/test_hbm_watermark.py`）。
