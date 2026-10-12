# slo_tools —— SLO 离线后处理工具集（WP3/WP4/WP5/WP7 + Hop-Bytes）

同名同构工具族（**2026-09-26 起各脚本头注已声明按仓分叉**：slo_common /
slo_stats / kv_cache_adapter / hopbytes / load_imbalance / hbm_watermark /
slo_postprocess_driver 等与 face/wscllm 家族同名脚本**不再跨仓逐字节相同**、
各在头注指认实际同步基线与跨仓同步义务范围（restore_decomposition 曾为唯一
逐字节一致例外，2026-09-30 起各仓亦各自演化——本仓 C-3 锚点描述 + R6-U8
孤儿计数、joint 仓同款分支，以函数语义为对拍基准）；本仓自 2026-09-05 A.5
起的预期差异见下；逐仓语义差异全部收在脚本内的
REPO_VARIANTS / REPO_HOP_SOURCES 表，按 run_dir 自动识别 repo_variant）。
唯一 sanctioned 例外：tests/run_golden_live.py（见"测试运行法"节的
仓族适配边界声明，2026-08-28 主控裁决）。
A.5（2026-09-05）wscllm 工作树附加说明：relevant_distributed 变体已随
《A.5-legacy变体调度器清除修改方案.md》清除——本仓 hopbytes.py 与
kv_cache_adapter.py 的 relevant 专属分支已删（其余三仓的同名副本不带这些
分支，同步回各仓 = 各仓删自身副本中的对应段；本仓不再逐字节相同，属本
方案预期差异）。
2026-09-24 死代码清除（本仓深挖文档 §2）附加差异：slo_common.py 删
percentile_from_sorted/ns_to_ms、slo_stats.py 删 APPENDIX_PCTS_DEFAULT
与 cmd_violation 死局部 deadlines、hopbytes.py 删 prefill 行 history_
transfer_shards[] 恒死分支（写侧零生产）——均为全仓零引用符号，删除后
本仓与三姊妹仓同名副本再添预期差异（同步回各仓 = 各仓按自身深挖清单
处置）。
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

`slo_params_manifest.json` 是 B 类参数的唯一来源（schema_version=1）。B4b
批次已按 derivation_program 引用的主规格条款推导后**填充全部
value+evidence+rationale**（当前无 null 值；本仓 manifest 另含 2026-09-25
审计注记与逐条目 consumers 读者登记，暂领先姊妹仓冻结基线、跨仓已非逐字节
相同）。**所有依赖参数的命令 fail-closed**：遇 null/缺失
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
| `slo_stats.py violation` | WP3 | request_metrics.csv + `--t-isolated` CSV + manifest α | slo_violation.json：`Deadline(r)=α×T_isolated(bucket(r))`；violation_rate=分子/分母（分母=全部终态 completed+rejected+dropped+timed_out+failed；分子显式排除非 completed 行——非 completed 终态计分母不进分子，D2 修复收口 scan-export 同款漏洞）；**proxy 字段显式断言拒绝**（输入带 first_token_ns/first_token_source 即 fail-closed） |
| `slo_stats.py bucket-stats` | WP3 | 同上 | slo_bucket_stats.csv：长度分桶 + slowdown=E2E/T_isolated(bucket) |
| `slo_stats.py session` | WP3 | request_metrics.csv + per-request manifest | slo_session.csv：T_session=末轮 completion−首轮 arrival−Σ(human+tool)（H5① initiator 口径：Σ 只含前 n−1 轮——末轮行上的等待物理落在 last_completion 之后、窗口外；2026-09-30 本仓同步），P50/P95 打 stderr；human/tool 均未透传 → NA 行+计数（互斥缺侧按 0；末轮行不进缺数据核对） |
| `slo_stats.py backlog` | WP3 | request_metrics.csv | slo_backlog.csv：arrival(+1)/completion(−1) 事件重建在途请求数时序（同刻先减后加；`--bucket-ns` 分桶取 max——桶值=桶内事件峰值与桶起点携入值较大者，零事件桶按上一事件值延续补行（[首事件桶, 末事件桶] 内不丢桶，2026-09-30 C-4）；`--bucket-ns` 非整数/非正亦 fail-closed）；**末态 in_flight≠0 或时序出现负在途 → fail-closed**（与 violation/scan 同口径，M6-05） |
| `slo_stats.py warmup` | WP3 | request_metrics.csv + manifest | slo_warmup.json：剔除前缀前后 P99 E2E 对比（窗口/门限=manifest warmup_window/warmup_change_threshold；p99_full_ns==0 物理不可达、fail-closed，Report 4.3 口径） |
| `slo_stats.py normalized` | WP3 | 多个 run_dir | slo_normalized.csv：Normalized Time/TPS（0-1，对齐比较组内 max；对齐 FACE/WSC-LLM 展示口径） |
| `slo_stats.py scan-export` | WP3 | `--point RUN_DIR=λ`（可重复）+ 可选 `--t-isolated` | slo_scan_export.csv：λ、violation_rate、tput、goodput=tput×(1−violation_rate)；**λ 有限性门**（inf/nan/1e999 等非有限值 fail-closed，INPUT-01 2026-09-30——NaN 比较恒假、Inf 使 λ 缩放列失义）；**drain 完备断言 input==completed 不过则拒绝出表** |
| `load_imbalance.py` | WP7 | results/online_decision_log.jsonl + results/train_ledger.jsonl + manifest(imbalance_bucket_ns) + trace_config.csv(inference_group 组行数→配置实例总数，与 hbm_watermark.count_configured_instances 同源) | slo_load_imbalance.csv + stderr JSON：逐请求 (instance, admission→drain) → 各 instance 积压时间积分（时间比例精确积分，METRIC-01）→ **全部配置实例**（含全程空闲 b̄=0）上的时间平均 CV（总体标准差）与 Max/Mean；配置实例数不可得即 fail-closed；不做账本 dump |
| `restore_decomposition.py` | WP5 | cpp.log（**full 档**：type=memory_anchor + type=request） | slo_restore_decomposition.csv：restore_start=min(锚点)/restore_complete=max(锚点)（按 subject_id=queue_index 逐请求归属）→ §3.2 固定公式三段+hidden_ratio；无锚点请求四推导字段全 NA+计数 |
| `kv_cache_adapter.py` | WP4 | results/online_decision_log.jsonl + per-request manifest | cache_events.csv（action_id,request_id,start_ns,end_ns,bytes,source,target,cause）+ kv_hit_states.csv（full/partial/miss/no_history/not_supported + 证据列）+ 命中率双分母；`--reconcile` 与 native 逐项对账（以 native 为准，不一致 exit≠0） |
| `hopbytes.py` | — | results/online_decision_log.jsonl | slo_hopbytes_total.csv + slo_hopbytes_per_request.csv：Hop-Bytes=Σ bytes×noc_hops（只聚合产物中真实携带 hops 字段的记录；覆盖率为 0 时输出 hop_bytes=0 并在 notes 显式标注 coverage=0，不臆造 hop 数）。**wscllm 变体口径警示：`hop_bytes_total` 混合两种 hop 单位——decode 行 shard 级为 mesh 跳、prefill 行迁移/回退为实例图跳；单一总和跨动作类不可比（拆列会破坏冻结列结构，以 granularity+notes 与脚本内口径警示为准）** |

`slo_common.py` 为共享框架（manifest 装载、request_metrics.csv 冻结列序
校验、NA 语义、nearest-rank 分位、repo_variant 自动识别），不含逐仓语义。

## 逐仓映射要点（证据见脚本内注释与 REPO_VARIANTS）

* **face/wscllm**：双级（-LRU 方案 A，2026-09-07）——新产物优先
  prefill.decision.history_location_before 三态映射（local_hbm→full /
  partial_hbm_remote→partial / remote_memory→full，照 sh_2.0；full_local/
  full_remote 由 evidence 列区分；五值域不扩、hit_n=full+partial 分子不
  动），字段缺失（旧产物 / 新产物 NO_HISTORY 恒 None）回退 history_action
  四值 → full/miss/no_history（LOCAL_HIT/NOC_MIGRATE→full，RECOMPUTE→
  miss；旧基线 RECOMPUTE 全量重算，无 partial 语义）。事件提取同步双级：
  history_transfers 在场走契约逐段/逐出行，legacy admission/decode_target/
  completion_evictions 镜像行不再消费（防 B3 双序列化双计）。
* **sh_1.0**：history_transfer.kind ∈ {local_hit, noc_migrate,
  remote_load} → full；null 且 turn=0 → no_history。shards 带 noc_path
  （hopbytes 唯一 shard 级数据源）。
* **sh_2.0**：B2wp9py（2026-08-26）起准入决策序列化
  history_location_before{,_instance_index}/history_resident_prefix_
  layers → 三态映射 local_hbm→full / partial_hbm_remote→partial /
  remote_memory→full（remote 整体恢复非重算；无 recompute，miss 不出现
  属预期）；旧产物（字段缺失）回退 not_supported。
* **sh_3.0**：prefill_affinity_reason 五值（sh_3.0 仓 sh30_online_
  scheduler.py 的 affinity 三分支；去行号化 2026-09-30）；
  resident_prefix_layers（PARTIAL_HBM_REMOTE）→ partial。
* **hopbytes 覆盖**（四仓同步收口后，各变体采集器已合并进同一
  REPO_HOP_SOURCES 表）：sh_1.0=shard 级 noc_path/noc_hops；
  wscllm=decode 行 prefill_decode_transfer.shards[] 逐 shard
  bytes×noc_hops（2026-09-05 起）+ 实例级聚合回退（decode=
  static_route.hop_count、prefill history 迁移=noc_hops，B2wp9py 起）；
  face=per-TP-shard hops 列表（B2wp9py 起）；sh_2.0=决策级
  transfer_hop_bytes（WP9-线5 起）；sh_3.0=completion_
  evictions shards[].noc_hops（B4 起）。wscllm 逐出/回迁契约行
  （history_evictions/prefill_evictions/decode_evictions/completion_
  evictions；旧产物 admission_evictions+decode_target_evictions，
  单键取用防 B3 双序列化双计）自 2026-09-24（M29）起计入
  bytes_without_hops——契约行无 shard 级路由，wscllm coverage 如实
  低于 1（此前逐出字节整类不进账，notes 误称满覆盖）。
  2026-09-26（G7-15）起顶层 noc_hops 缺席的 prefill 行（REMOTE_RESTORE
  远端回迁）逐对象消费 transfer_hop_bytes 行的 shard 级 hops 计入覆盖分子
  （行字节合计与 history_transfer_bytes 相符才启用；旧产物回退标量
  bytes_without_hops）。旧产物按
  字段缺席回退 bytes_without_hops/coverage=0（不臆造 hop 数）。

## fail-closed 纪律

* 退出码：0=成功；2=FAIL-CLOSED（输入缺列/格式错/参数 null/对账不一致/
  journal 损坏）；3=容量违规——仅 hbm_watermark 的 per_rank_total_hbm_
  certified 层（journal+checksum 权威重放的逐 rank physical 认证；
  upper_bound_only 上界层超限只诊断、exit 0）。exit 3 **全链透传**
  （A5-1/infra-4，2026-09-30）：单遍 driver 任一步以 3 退出即整体 3
  （不再折叠为 1），`run_slo_postprocess.sh` 尾部同样原样透传。
* request_metrics.csv 列序与 EXECUTION_PLAN §2 冻结值逐一相等，否则报错。
* T_isolated 表缺桶、分桶边界越界、terminal_status 非法等一律非零退出。
* kv 对账（--reconcile）与 Σshards==total_bytes 不变量以 native 为准。
* 2026-09-30 批次工具面扩充（语义以脚本内注记为准，此处撮要）：
  kv_cache_adapter——sh_1.0 `history_transfer` 非 dict 真值、decision log
  文件在场但零记录、同请求重复 prefill 决策（不再 last-wins）、reconcile
  双侧提取式对称（total_bytes 非整数 → Σshards 回退双侧同规）一律
  fail-closed；load_imbalance——积分区间=admission ∪ decode 两源并集
  （只遍历单源会丢区间）、配置实例数缺失即 fail-closed、分母候选链不回退
  仓内默认 trace_config、区间实例超出配置域即 fail-closed；
  restore_decomposition——memory_anchor 的 subject_id/tick_ns 严格整数
  类型（bool 显式排除）且非负，残缺/负锚点 fail-closed。

## 测试运行法

```bash
# 仓根目录执行（两法等价；仅标准库。旧"2026-09-05 实测 118 例：114 过 +
# 2 失败 + 1 错误 + 1 跳过"计数已作废——用例集其后持续演进，passed/skip
# 基线以 sh_test_mesh/run_scripts/run_python_unit_tests.sh 逐根
# 实测输出为准（2026-09-30 实测本根 194 passed 1 skipped，唯一 skip 为
# test_hbm_watermark 的 /var/tmp phase2 journal 真件对拍节）；仓根 README §5
# 同口径登记）
python3 sh_test_mesh/slo_tools/tests/test_slo_contract.py      # T0 契约
python3 sh_test_mesh/slo_tools/tests/test_golden_g1g4.py       # T2 golden 骨架
python3 -m unittest discover -s sh_test_mesh/slo_tools/tests -v
python3 sh_test_mesh/slo_tools/tests/test_hbm_watermark.py     # WP8 手算
python3 sh_test_mesh/slo_tools/tests/test_driver_parity.py     # A4 driver 对拍
```

T2 golden（G1 单请求无竞争 / G2 双请求排队 / G3 session 两轮 / G4 restore
三段）用合成 request_metrics/manifest/anchor 样本断言手算值；仿真侧 live
版为 `tests/run_golden_live.py`，手动触发跑真仿真断言。

**run_golden_live.py 仓族适配边界（sanctioned 例外，2026-08-28 主控裁决；
不删除——T2 是《各指标测试建议.md》验收清单必备层；暂不合并——脚手架层
差异，合并的假绿/假红风险不抵收益）**。三版各自适配的队列格式：
face+wscllm = 8 列 recompute 队列 + canonical sidecar；sh_1.0+sh_2.0 =
仓自适应（sh_1.0 8 列 + 15 列 sidecar / sh_2.0 9 列 + 10 列 digest）；
sh_3.0 = 原生 9 列（含 next_trigger_type）。边界：

* **允许仓族差异**：队列格式构造、sidecar 命名/join、场景驱动脚手架
  （build/run/CLI 形态）；
* **必须四仓一致**：G1-G4 各场景的断言语义与期望值推导口径——如 G1 的
  e2e=queue+prefill+gap+decode 整数恒等、G2 的 queue_ns≈占位者 prefill、
  G3 的外生等待（human/tool interval）扣除方式（H5① initiator 口径：
  Σ 只含前 n−1 轮，末轮行上的等待物理落在窗口外；2026-09-30 本仓同步）
  与 request_type 透传、
  G4 的 restore 三段和=总时长与锚点 min/max 归属。改这些断言时四仓
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
    门全过（sha256 相符/行级链自洽/重放终态==证书终态/守恒 checks 全
    true）：流式重放 journal 得逐 rank physical=weight+resident+reserved
    时序，对行内 capacity_bytes 逐 rank 认证——**正式容量判决只在本层给
    出**，违规 → stderr 标红 + 退出码 3（fail-loud）。第三道门（重放终态
    ==证书终态）的证书分量自 M-32（2026-09-30）起全分量严格类型校验：
    weight/resident/reserved（v2 另加 remote）缺失、str/float、bool 冒充
    整数、physical 字段残缺一律 exit 2——证书残缺不得 certified。
  * `resident_kv_exact`——journal 在场、链自洽，checksum 缺失：逐 rank
    时序精确但无 run 末守恒证书，容量检查如实报告、不作正式判决。
  * `lifecycle_replay_exact`——journal+checksum 在场、链与终态一致，但守
    恒 checks 有 false（run 末守恒未过），或证书缺必需项（sha256/
    line_count/ranks 缺失或类型不符——完整性证明不完整，HBM-01：不得
    certified）：生命周期精确、无证书，仅报告并显式列出缺陷。
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
  ·tokens 逐条对账（五仓基线 restore 比值全 1.0，S3 另有 0.5 半层；
  G7-5 2026-09-27 起 legacy 旧产物的 0.5 半层分段恢复比值在 watermark
  诊断账中计 `restore_bytes_mismatch`——重放占用数学不变）。
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
  model_weight_shard_bytes_by_tp_rank / kv_cache_bytes_for_tokens /
  kv_cache_shard_bytes_for_tokens / kv_cache_shard_bytes_for_layer_range
  四个同名符号（定位以文件+符号名为准），同步义务见脚本内注记）**：
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
  "峰值记录"并注明（`degraded_peak_recorded`——M-32 2026-09-30 起**仅
  journal 缺席层适用**：journal 在场的逐 rank 认证用行内 capacity_bytes、
  不依赖配置容量，capacity 缺失只降级展示列，不再给 certified 判决同框
  打降级标注）。
* **退出码**：0 正常（含上界/无证书层的超限诊断）；2 fail-closed（缺文
  件/缺列/结构错/重放不一致/journal 损坏）；3 容量违规——**仅
  per_rank_total_hbm_certified 层**的逐 rank physical > capacity_bytes
  （该 3 经单遍 driver 与 `run_slo_postprocess.sh` 全链透传，见
  "fail-closed 纪律"节）。journal 缺失而 checksum 在场的孤儿证书
  （M-33，2026-09-30）标注 `orphan_checksum_upper_bound_only`、与
  "journal 机制存在前"旧 run 的 `diagnostic_upper_bound_only` 分立——
  两者同为 exit 0 上界口径，不再混同。
* **逐仓覆盖度**（decision-log 重放路径的口径；B0 基线 60s_summary 实测）：
  * FACE/S1：`full`——逐出条目全量落盘（bytes+victim+实例），重放闭合，
    0 异常 0 违规（FACE 另有 9 例 decode 准入静默逐出经 RECOMPUTE 断言对账）。
  * W：`full_reconciled`——旧产物账本缺口：decode 准入期逐出
    （decode_target_evictions 在 prefill 记录落盘后才累积，decode 记录不含
    逐出列表）不落盘，full_tracelab 本 run 实测 7,923 例
    RECOMPUTE(state_before=EVICTED) 隐含静默逐出（802 为 FACE 基线旧数）；
    重放在 `history_cache_state_before=EVICTED` 的恢复点按账本断言对账扣减
    （silent_evictions_reconciled；静默逐出真实时刻不可见，该窗口内占用
    为上界）。-LRU tiered 新产物（B4，2026-09-07）沿用 full_reconciled
    口径换新缺口：净额准入重试把多次尝试的逐出累积到最终获准请求的行上，
    条目到时序晚于 victim 其间的恢复/迁移事件——逐出条目 report-only
    （evict_events/evict_bytes 照记，不动占用/跟踪态），旧代驻留在恢复点按
    admission 时点 `history_location_before` 对账核销（remote_memory→0；
    partial_hbm_remote→noc 段前缀 bytes）。journal 行 schema v2（B2 remote
    列）自此接受：v1/v2 同文件不得混版；v2 的 remote 链参与行级校验但不计
    入逐 rank physical（远端池不占本地 HBM）；certified 层守恒 checks 五项
    （v1 四项 + remote_account_zero）。journal 模式的 evict_events/evict_
    bytes 单一取自 journal 重放侧：evict_events 自 M-31（2026-09-30）起为
    **会话去重口径**（journal 行逐 rank 提交，同会话一次逐出按 TP 落多行，
    逐行计数会 ×TP 虚增——按事务内会话集去重后与 decision-log 路径同口径，
    不再 ×TP；bytes 逐行求和本就不虚增）。
    evict_bytes 列在 full/
    full_reconciled 均输出真实值（P1-⑤；NA 仅保留 count_only 档）。
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
* **测试**：`tests/test_hbm_watermark.py`（57 例：手算桶时序/峰值/均值/
  逐出/违规计数、多仓语义各一例、fail-closed 系列、列序冻结、四层 tier
  判定（含"聚合过单 rank 超"锚定用例）、三口径锚定数值、stats 多桶长
  不变、RLE 无损恢复、行预算/B_eff、journal 行级门（sha/链/终态/证书
  缺陷降级）与真 journal 对拍）；运行法同上
  （`python3 sh_test_mesh/slo_tools/tests/test_hbm_watermark.py`）。

