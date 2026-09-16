# astra-sim-joint - 三机制联合策略仓（T＋J＋E 消融统一实现）

> 本仓为 Execution-Driven 改造后的**裸仓库终态**（沿用基底两条在线仿真
> 路线；仓库 request-neutral，正式入口缺失输入 fail-closed）。由
> `template/astra-sim-sh_3.0` 工作树复制建立（基底快照：commit `4e9e2ca`
> + 工作树实际状态，含 2026-09-13"KV 逐出与推理计算并行执行"改造）。设计
> 规格：`paper_write/本论文撰写内部材料/三机制联合策略_template仓库设计
> 方案.md`（下称《设计方案》）；实验总纲：`paper_write/本论文撰写内部
> 材料/chatgpt建议实验思路.md`。
>
> 本仓是 ASTRA-sim 2.0 的晶圆级芯片（Wafer-Scale Chip, WSC）推理仿真
> 改造仓。七个同源仓共享同一套硬件架构建模与术语，仅在"请求→实例映射
> 与 KV 管理策略"上分化；本仓的分化点 = **T/J/E 三机制联合策略与八组合
> 消融能力**。硬件参数唯一权威来源仍是
> `sh_test_mesh/hardware/face_case5_config_c.json`，运行时配置由
> `sh_test_mesh/config_resolver.py` 派生，禁止手改。

## 1. 本仓定位：同一套共同实现内的 T/J/E 独立开关

本仓承载三机制（《设计方案》§1）：

| 机制 | 开关 | on 语义 | off 共同替代（§7.1） |
| --- | --- | --- | --- |
| **T** 工具/人类区分 | `category_mode=typed` | SH 严格类别逐出：合法 victim 先 human 类后 tool 类，类内 `(last_completion_ns, session_id)` FIFO；human 层未耗尽且缺口未满足不得动 tool | `lru`：同一合法 victim 集合上的类型无关 LRU（单遍），其余流程不变 |
| **J** 联合决策 | `scheduler_mode=joint` | 全 instance 候选（无容量/边缘/距离掩码），发射点联合选择 instance × stay/recompute/copy/remote-read | `load-first`：先按在线负载（ns 服务台账）在全部 instance 选点，再在该位置选 KV 动作；`affinity-first`（home 优先、缺省回退 load-first）为 J 边际比较第二参照，不进八组合 |
| **E** 自适应层数逐出 | `layer_policy=adaptive` | 按层消费期限公式计算最小热前缀 `k_hide` 软目标（§5.2），释放时目标外后缀→目标内后缀两次扫描 | `minimal_layer_groups`：同一合法对象排序下按逐 rank 缺口释放最少完整层组；`legacy_half` 为原半层逻辑回归对照，不是 E-off |

**remote on/off**（`remote_actions=on|off`）是与三机制正交的能力消融
开关：off 仅从候选动作集中移除 remote-read，全部 instance、其余动作、
驱逐、合并与生命周期不变；单独记录，不与 T/J/E 合并。

### 1.1 八组合固定映射（§7.1 硬性要求，2026-09-13 裁定）

`JOINT_ABLATION_COMBO` 一键预设（无 J 组合一律 `load-first`）：

| 组合 | category_mode | scheduler_mode | layer_policy |
| --- | --- | --- | --- |
| 无 T/J/E（"三无"基础组） | lru | load-first | minimal_layer_groups |
| T | typed | load-first | minimal_layer_groups |
| J | lru | joint | minimal_layer_groups |
| E | lru | load-first | adaptive |
| T＋J | typed | joint | minimal_layer_groups |
| T＋E | typed | load-first | adaptive |
| J＋E | lru | joint | adaptive |
| T＋J＋E（完整方法） | typed | joint | adaptive |

八组合共用同一套生命周期、KV 账本、资源计费、自动合并与事件模型；每个
开关只改变其指定分支（KV 物理布局、新 KV 写回/合并语义、硬件配置、初始
状态、在线估计与更新规则、动作计费在八组合间严格一致）。开关经环境变量
一次读取、fail-closed；预设与显式开关互斥（同值重复指定同样拒绝）：

```bash
# 八组合预设（任选其一）
JOINT_ABLATION_COMBO=TJE   # 或 none/T/J/E/TJ/TE/JE
# 或显式开关（与预设互斥）
JOINT_CATEGORY_MODE=typed|lru
JOINT_SCHEDULER_MODE=joint|load-first|affinity-first
JOINT_LAYER_POLICY=adaptive|legacy_half|minimal_layer_groups
JOINT_REMOTE_ACTIONS=on|off          # 缺省 on
```

开关入口裁定（2026-09-14 用户裁定，经 kimi 审查上呈）：**接受 JOINT_*
环境变量族作为策略开关唯一入口**（原执行方案 D7 的 joint_config.json
路径不实施；偏差登记见仓根 `PROVENANCE.md`）；**默认 = 完整三机制
TJE**（原执行方案 D9 的 affinity-first/legacy_half 回归对拍档经显式
env 获得，即 §8.3 #9 哨兵配置）。推荐经 `joint_runner.py` 起跑（D14）：

```bash
python3 sh_test_mesh/run_scripts/joint_runner.py <run_dir> <request_csv> \
    --combo TJE            # 或 --category/--scheduler/--layer/--remote 显式开关
```

runner 叠加：仓内仿真锁（`sh_test_mesh/runs/.single_simulation.lock`，
flock 覆盖子进程）、二进制 sha256 前置校验、`<run_dir>/invocation.json`
（argv/开关/哈希/exit/UTC）、陈旧策略 env 清洗（SH30_*/SH3_CAUSAL_*/
SH_JOINT_* 全弹出；白名单运维变量取值披露）。

运行侧 provenance：每次运行把 `joint_config.manifest_dict()`（四开关 +
combo + off 替代语义披露 + 来源 env）写入 `<bridge_dir>/joint_mechanism_
manifest.json`；每次准入决策在 decision log 落 `joint_mode/joint_action/
joint_cost_ns/origin_home_instance/horizon_source` 与全部候选的成本表；
completion 行落 `merge_transfers`（增量归并回 home 的真实传输摘要）。
基底的 `SH30_ABLATION`（no_lb/no_affinity）已退役：**任何显式值（含
`none`）即启动失败**（fail-closed，防陈旧脚本静默假装旧消融生效）。

## 2. 策略架构（《设计方案》§7 模块边界）

```
sh_test_mesh/workload/llama2_7b_inference/
├── joint/                        # 三机制策略包（本仓新增）
│   ├── joint_config.py           #   开关解析/八组合预设/manifest
│   ├── eviction_priority.py      #   T：typed/lru victim 类别序（纯函数）
│   ├── layer_eviction_policy.py  #   E：k_hide 期限公式 + 三模式释放计划
│   │                             #      + 在线估计器（输入长度/EWMA）
│   ├── joint_cost_model.py       #   J：无 oracle (instance×action) 代价
│   │                             #      + 链路流登记表 + 因果时域估计
│   ├── joint_scheduler.py        #   J：joint/load-first/affinity-first 选择
│   └── test_joint_mechanisms.py  #   定向验收单测（§8 验收问题）
├── face_scheduler.py             # KVCacheManager：T/E 注入 + home/工作
│                                 #   副本账本 + merge_back 合并事务
└── online/
    ├── sh30_online_scheduler.py  # joint 准入（替换三段式）+ 因果 decode
    │                             #   增长 + 完成合并 + remote 读流
    ├── graph_batch_builder.py    # 多传输准入（copy 前缀+后缀）/merge 发射
    └── online_service.py         # 服务入口 + joint manifest 落盘
```

共同生命周期（§2/§3）：轮次 `QUEUED → PREPARING → EXECUTING →
COMPUTE_DONE → MERGE_WAIT/MERGING → COMMITTED → SERVICE_DONE` 的账本
映射——

* **home（§2.1）**：每 session 持 `home_instance`（首轮发射建立）；
  异地执行（copy/recompute/remote-read）与逐出**都不改变 home**。
* **执行与增长（§3.1）**：新 KV（prefill 输入 + 实际 decode）在执行
  instance 产生；准入预约**动作感知足迹**（R1'：stay/copy/recompute
  按 history+input 整份、remote-read 仅 input 增量——单源
  `joint_reservation_context_tokens` 贯穿可行性/预约/物理可行性三处；
  无 PARTIAL 驻留钉扎：容量只影响所需字节与驱逐等待计价，不做候选
  掩码），decode 按列车核销的实际消费**因果增长**（每次增长经同一
  T+E 释放机制准备空间；容量不足时进入**停滞/唤醒**（R14）：会话
  暂缓后续列车参与，容量释放（KV 纪元 ⊕ 实例纪元重试键）后唤醒，
  等待如实计入 E2E，全停滞死锁由守卫显式 fail-closed；不提前按真实
  最终 decode 长度预约——`final_context_tokens` 不进任何决策输入）。
* **合并（§2.2/§3.2）**：`merge_back` 在完成处理时把新增量归并回
  origin_home——基础 LOCAL 的增量前缀经 NoC 回传（home 侧空间经 T+E
  真实准备；容量不足时本会话基础前缀逐层**自降级**（R4：受限 victim
  视图经 layer_policy 的自我释放独立事务——不经 victim 池、不触碰
  其他会话；k=0 兜底落入 REMOTE 归并，永不失败；降级事件入
  `merge_degrade_events` 台账））；基础 PARTIAL 的前缀部分回传 +
  后缀部分写回池 backing；基础 REMOTE 的整份增量写回池。执行端工作
  副本（含 copy 的历史复份）释放，不自动形成永久第二份历史；恰好
  归并一次由 `last_merged_request_id` 版本键断言保证（`mark_complete`
  拒绝未合并的工作副本——service_done 位于 merge_done 之后，**到达
  时序同样重锚 merge_done**（R11）：下一轮 alarm = merge 尾标记完成
  + interval，下一轮 interval gate 前递依赖 merge 尾标记节点）。
* **动作语义（§2.2 表）**：stay=本地命中/部分恢复；copy=基础历史复制
  为工作副本（前缀 NoC + 缺失后缀池恢复）——**实现/测试锚点（R16，
  2026-09-15）**：物化 = `face_scheduler.py` `_noc_transfer`（层区间化，
  `[0, base_prefix)`）+ `_remote_load_transfer`（`[base_prefix, L)`）、
  计价 = `joint_cost_model.py` `estimate_action` ACTION_COPY 两腿
  （`noc_prefix+pool_suffix_restore`）；单测 =
  `joint/test_joint_review3_fixes.py`（复合两笔物化/计价闭式/图发射/
  水印重放/merge 回归 22 用例，含 8b 图发射快路径）+ 容量压力夹具（PARTIAL×copy 端到端
  命中）；recompute=**只重算缺失区间**（R13：@驻留目标复用权威前缀、
  仅物化缺失后缀层（span 基 = H，recompute@home 与 stay 同构本地提交、
  merge 零流量）；@异地/REMOTE 基础整份重算（span 基 = 0））；
  remote-read=基础历史留
  home，执行端仅驻留新增量 KV——**适用性边界（N1(a) 裁定）**：要求
  基础历史全层驻留（LOCAL）；PARTIAL 的池后缀无"前缀 home + 后缀
  池"混合读流原语（后端能力边界，非 joint 理论排除），PARTIAL 会话
  跨实例服务走 copy。
* **逐出（§4/§5.6）**：T 定类别/对象序（`eviction_class_order`），E
  定层数（`LayerEvictionPolicy.plan_release`，只读计划、提交前复核）；
  逐 rank 缺口检查（不用实例汇总掩盖单 rank 超容）；deep-gap 台账
  fail-closed 沿用基底 D4（准入/merge/decode 语境按 N3' 类型化
  `KVCapacityError` 分别转延迟/自降级/停滞，合同类异常原样上抛）。

## 3. 状态披露（《设计方案》§5.6/§8：已设计/已实现/已验证分开陈述）

| 项 | 状态 | 说明 |
| --- | --- | --- |
| T 开关（typed/lru） | 已实现＋单测验证 | 类别严格序、满足即停、未知类别 fallback 披露 |
| E：minimal_layer_groups / legacy_half | 已实现＋单测验证 | legacy 与基底逐字节等价（既有测试回归通过） |
| E：adaptive（k_hide） | **公式＋计划已实现、已单测验证（解析例 1/17/25）；运行期事件递推预测器未实现** | 当前 r_j/c_j 为均匀层解析模型（r_j 按池端口仲裁份额后的有效速率（P1，R15-3）+ roofline 派生 + 在线输入均值）；正式预测器的共享资源逐事件递推（§5.2 末段）为后续项 |
| J：选择与代价模型 | 已实现＋单测验证 | 解析近似代价（关键路径合成、因果时域估计）；**在线反馈通道已接线（R15，2026-09-14）**：链路流登记表按逐 shard 全路径登记/完成事件注销（F-B 并集瓶颈除数）、池端口仲裁份额（含 E 内核 r_j，P1）、ServiceFactors EWMA（P3 α=1−exp(−Δt/τ) 时间衰减；transfer 因子保留接口位——节点级传输完成遥测未交付，传输争用在线修正由除数通道承担；样本纯度排除 joiner 迁移/partial 恢复/copy 门控传输/remote-read 读流列车——复审 M5）；decode 负载标定在线化（N12：session 均值 → run 均值 → 冷启动 1，全 trace 均值常数不再进决策输入；估计器版本入快照缓存失效键——复审 M3）；merge 段增量计价（复审 K4：input + 因果 decode 增长，不随基础历史膨胀；REMOTE 基走池端口口径——自查 C）；准入重试键 = KV 纪元 ⊕ 失败候选集纪元（复审 M2，R2.5 口径）；无 oracle（`RequestView` 结构性不接受 decode_length/final_context_tokens） |
| home/merge 账本事务 | 已实现＋单测验证 | 增量恰好归并一次（版本键断言）、无双份驻留、home 保持；R4 自降级 + R11 service_done 重锚 merge_done；容量缺口台账落账边界 = 确认终态（复审 K6：可恢复失败随 `KVCapacityError.deep_gap_records` 携带，死锁守卫/降级耗尽才提交；run 末导 `joint_kv_ledgers.json` 侧车含 merge_degrade/deep_gap 两台账）；decode 停滞死锁守卫逃逸条件覆盖全部事件源（pending_decode_ready、在途 merge watch、到达堆、未到达请求——EOF 终态全停滞才触发，复审 K1 + 自查 A）；逐列车 decode 增长逐出三路进图（旁路支链 + rid#decode 流登记，自查 D——池写不再免费） |
| remote-read 执行流 | **v1 已实现（保守口径）；逐迭代 credit 交错流未实现** | v1 = drain 边界合成读流（home→exec 真实 NoC 字节流，总量=剩余步数×全上下文 KV，列车 readiness barrier 门控列车体）——读不与计算重叠，保守方向；计价基数含 input + 在线 decode 增长（N11，因果可见；执行侧终态上下文为后端真值不进决策）；适用性要求基础 LOCAL（N1(a)，见上"动作语义"） |
| merge 物理流 | 已实现 | 完成批发射真实回传/池写回流 + merge 尾标记节点（watch 送达后重锚下一轮到达）；下一轮数据准备经 store-tail 前递补偿等待（合并成本进入下一轮数据等待；全部动作/八组合一致处理） |
| 逐层恢复与 prefill 重叠 | 未实现（列车级恢复门保守化，沿基底口径） | §5.1 的逐层交错恢复为后续项；k_hide 目标在保守执行下仍约束保留层数 |
| SLO 水印对 joint 的口径 | 已登记（joint 词表已扩，R12） | slo_tools 三工具已登记 astra-sim-joint；水印重放词表已扩 joint 专属事件（merge 增量回传/池写回/home_merge_base_degrade 自降级/跨实例工作副本释放——SessionState base/working 双驻留记账；决策行携带 joint_action/history_transfers/重算口径字段；完成行显式披露 `joint_working_copy` 真值——复审 K3；home 侧结算只并入增量、base 不双计——复审 K2），"home 低估/执行端高估"口径缺口消除；决策日志重放仍为上界口径（逐 rank 判决需 kv_delta_journal 权威层，本仓暂不产出，同 S3） |
| 容量逐出的决策日志披露（R17，2026-09-17） | 已实现＋单测验证 | 旁路逐出咽喉点（`_emit_eviction_only_nodes`）在图发射后同 tick 落 `kind=kv_eviction` 决策行（decision.evictions 携带 `_transfer_summary` 条目）——覆盖 decode 增长逐出（成功/停滞/唤醒三路）与准入失败已提交逐出（潜伏位点：当批零触发——reserve 的 feasible 预检拦截在先 + prepare 受 R1' 预约不变量覆盖，R17-7 探针实证）；`prefill_evictions` 为结构性恒空死通道（准入预约覆盖全动作足迹，drain expand gap≡0），注释+单测钉死不新增空字段载运；转移摘要补序列化 `resident_prefix_layers_before/after` 与 `source_instance_index`（连锁不变量免重建 + victim 归位，R17-1d）；tick 守护 = 断言 `_batch` 在场（N4）。工具侧配套：hbm_watermark 主循环 kind 门前拦截 kv_eviction（自担 tick 单调 + 逐条 `apply_evict` + 非空率哨兵）+ `_joint_prefill` fail-early 对称校验（prefill_shrink/discarded>0 fail-closed）+ (primary, base) 二元组前缀连锁不变量（R17-2/3/4）；hopbytes `collect_joint` 消费 kv_eviction 池写流（此前通道 2/3 系统性漏计）并移除 prefill_evictions 死读。历史 run 决策日志定格缺失、离线不可重建——水印三件套与可信 hopbytes 需带本修复重仿真（第三次错误处置，方案 v4） |
| 三机制消融/外部对照实验 | 未运行 | 本次交付为仓库构建＋单元级验证；仿真规模实验按另行登记的有限计划进行（§8 第 6 步） |

## 4. 快速开始（构建与运行入口沿用基底）

```bash
cd <本仓根>
# ① 构建（与 sh_3.0 同款；裸仓无 build/：先放拼装 CMakeLists）
#    mkdir -p build/astra_analytical && cat > build/astra_analytical/CMakeLists.txt <<'EOF'
#    cmake_minimum_required(VERSION 3.22)
#    project(AstraSim_Analytical_Build)
#    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../ astra_sim_build)
#    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../extern/network_backend/analytical backend_build)
#    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../astra-sim/network_frontend/analytical frontend_build)
#    EOF
cmake -S build/astra_analytical -B build/astra_analytical/build_congestion_aware -DBUILDTARGET=congestion_aware -DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON
cmake --build build/astra_analytical/build_congestion_aware -j
# ② 物化输入（唯一允许源 = agent-traces/tracelab/astra_compute_20.csv
#    前 2 秒）+ ③ plan 物化 + ④ 运行（与基底同款）
cd sh_test_mesh/workload/llama2_7b_inference && python3 plan_materializer.py && cd <仓根>
python3 sh_test_mesh/run_scripts/joint_runner.py <run_dir> <绝对路径 request_csv> --combo TJE
# 等价直跑（不经 runner 的锁/哈希/清洗层）：
# JOINT_ABLATION_COMBO=TJE bash sh_test_mesh/run_scripts/run_online_strategy.sh <run_dir> <绝对路径 request_csv>
# ⑤ 指标后处理/对账/SLO 提取/⑥ 裸仓还原：与基底同款
bash sh_test_mesh/run_scripts/run_metrics_postprocess.sh <run_dir>/cpp.log
bash sh_test_mesh/run_scripts/clean_test_records.sh && bash sh_test_mesh/run_scripts/clean_build_artifacts.sh
```

## 5. 与基底 sh_3.0 的语义差异（迁移清单）

1. **准入**：三段式准入（首请求避边缘/LOCAL·PARTIAL sticky/REMOTE 边缘
   负载均衡）与 PARTIAL 钉扎**整体移除**，替换为 joint 框架三模式
   （§7.1）；HBM 容量不再作为准入过滤（只影响动作计价中的驱逐等待）。
2. **decode 选点**：恒同执行 instance（§1.1）；基底 no_affinity 的
   drain 均衡迁移随 SH30_ABLATION 退役。
3. **预约口径**：准入预约只覆盖已知 prefill 上下文；decode 逐列车因果
   增长（§3.1；基底在 drain 一次按真实最终长度扩容的 oracle 用法移除）。
4. **跨实例历史**：noc 迁移（move 语义）→ copy 工作副本 + 完成时
   merge_back 增量归并回 home（§2.2；基底 copy/recompute 顺便改写
   session 位置的行为已修正）。
5. **逐出**：类别序经 T 模块、层数经 E 策略；`legacy_half` 缺省保持与
   基底逐字节等价（供回归对照）。
6. **完成路径**：完成批新增 merge 回传流发射；`mark_complete` 强制
   merge-先-完成次序。

## 6. 测试

```bash
cd sh_test_mesh
python3 -m pytest tests/ slo_tools/tests/ workload/llama2_7b_inference \
    --ignore=slo_tools/tests/test_driver_parity.py \
    --ignore=slo_tools/tests/test_golden_g1g4.py \
    --ignore=slo_tools/tests/test_slo_contract.py -q
# 交付基线：321 passed + 1 skipped + 9 subtests（R17 批起 = 311 基线 +
# test_joint_review3_fixes.py 27 用例（R16 批 22 + R17-5a 4 用例：通道 2
# kv_eviction 行落盘一致性/通道 3 潜伏位点接线/通道 1 恒空谱系显式化/
# tick 守护；kimi 终审 P3 迁入 hopbytes kv_eviction 采集用例 1）+
# test_hbm_watermark.py JointSession411GoldenTests 5 用例
#（session_000411 生命周期切片重放闭合/精确水位/挖空敏感性 + 前缀不符/
# tick 回退/discarded 拒收 fail 路径）；hopbytes kv_eviction 采集用例经 kimi 终审 P3
# 自 slo_tools/tests/test_slo_contract.py（--ignore 面）迁入上处基线车道。
# 口径订正（外部审查处置批 2026-09-15；三轮深审再订正分解式）：此前
# 此栏把全树计数配给了 workload/llama2_7b_inference 四目标窄命令
# （该命令实收 229 = 本仓目标面 joint/+根两 test 文件+online/）；
# 全树实收 = 目标面 + slo_tools 可收集 + sh_test_mesh/
# tests → R17 批 320 passed + 1 skipped（skip = slo_tools test_hbm_
# watermark.py phase2 journal fixture 缺失）。slo_tools 三文件
# （driver_parity/golden_g1g4/slo_contract）从 sh_test_mesh 根收集
# 即 ImportError（伴生模块 synthetic.py 在其目录内、不在 sys.path；
# 目录内可收集 80 项）——基底固有的调用目录依赖，与本仓改动无关，
# --ignore 排除（目录内直跑时 driver_parity 2 失败 + slo_contract 1
# 失败为 HEAD 先在，git stash 还原基线复跑坐实，与 R17 无关）。
```

定向验收覆盖（`joint/test_joint_mechanisms.py` 对应《设计方案》§8 表）：
八组合固定映射与 fail-closed；T 严格序/满足即停/未知类别披露；E k_hide
解析例（L=32, c=1ms, q=0, r=0.5/2/4ms ⇒ 1/17/25）与三模式计划；J 三模式
选择与 remote-off 仅移除 remote 候选；home/merge（增量恰好一次、无双份
驻留、recompute/remote-read 工作副本口径）；八组合同实现配置切换。

端到端冒烟（astra_compute_20.csv 前 2 秒，21 请求）：执行方案 §8.3 的
**13 配置矩阵全部 PASS**（八组合 #1–#8 + 辅助臂 #9 typed/affinity-first/
legacy_half、#10 typed/affinity-first/adaptive、#11 typed/affinity-first/
minimal_layer_groups、#12/#13 remote-off 正交臂）；行为矩阵：joint 组
选点 10/4/6/1 vs load-first 组 12/6/2/1（J 联合选择真实改变选点），
load-first/none 系真实触发 7 次跨实例 copy + 7 次增量 merge 回传流
（R15/K4 计价修复后决策序列快照，修复前为 6/6）；
affinity-first 恒 stay（home 优先与联合选择的最优重合）。**冒烟矩阵已
脚本化**：`sh_test_mesh/run_scripts/joint_smoke_matrix.sh`（2s 窗 10
配置；产物落仓外持久目录 `/home/sunhao/joint_smoke_evidence/<combo>/`
全套留存——run.log/决策日志/env 快照/退出码，复审可独立复验；
PROVENANCE §7-14/§9）。

**容量压力夹具（R16-4-7，2026-09-15）**：
`sh_test_mesh/run_scripts/joint_capacity_stress_fixture.sh [window_ns]
[capacity_profile] [evidence_root] [combo]`（缺省 10s × stress-28gib ×
TJE，缺省档即可全判据 PASS）——小 HBM 压力档
（`hardware/face_case5_config_c_stress.json` 孪生源，备份→切
trace_config→还原）使 E 逐出真实发生、PARTIAL 形成、跨实例
复合 copy 真实命中；判据 = 退出码 0 + PARTIAL×copy 计数 > 0（noc 腿
layer_end < L）+ deep_gap 台账空（硬门禁，台账侧车缺失 = fail-closed）；merge_degrade 计数落盘
披露（run 目录 `judge_summary.json`）不门禁——R4 设计内计价行为，
10s 窗内与 PARTIAL 形成通道同链不可分（外部审查处置批 2026-09-15
改造；此前"两台账空"硬门禁使缺省档必然 FAIL）。中量程门禁档 =
150s × stress-128gib（两台账皆 0）/stress-96gib（动态工况最全，见
PROVENANCE §11）。runner 经 `SH_RUNTIME_RC_DIR` 覆盖指向
stress 档 runtime_config（正式跑保持缺省 160gib）。证据落
`/home/sunhao/joint_r16_stress_evidence/`；PROVENANCE §11。

## 7. 硬件概念（沿用 sh_3.0 口径）

**A. 晶圆级芯片（wafer-scale chip）＝整个计算系统**：一片二维 mesh 晶圆
（行列数由 `mesh.rows/columns` 给出），芯粒间由片上网络互连，每颗芯粒自
带本地 HBM；NPU 数恒等于行×列，row-major 编号。

**B. NoC（片上网络）**：非环绕二维 mesh 有向链路（die-to-die），XY 确
定性路由、逐跳 store-and-forward、共享链路 FIFO 排队（拥塞感知）；分析
级模型（C++ 侧 `extern/network_backend/analytical/congestion_aware/`，
Python 侧同语义确定性 XY 路由函数）。

**C. 芯粒（die/NPU/rank）**：计算单元（Roofline 计时）+ 本地 HBM（KV
热层）+ 路由端口 +（边缘芯粒）远端内存端口。

**D. 边缘芯粒与片外统一内存池**：`unified-kv-cache-pool` 为 KV 冷层；
仅 mesh 物理边界芯粒直连（远端内存端口），非边缘芯粒借道最近边缘芯粒；
统一逻辑地址空间（跨缘写读）；端口严格 FIFO（`耗时 = 端口时延 +
字节/端口带宽`），池容量不设限。本仓启用远端池（全部边缘端口生效），
KV 冷热分层为三态（LOCAL/PARTIAL/REMOTE）。

**E. 实例（instance）**：紧密相邻芯粒的连续实心矩形，TP whole-head 分
片；全部实例铺满晶圆；布局唯一来源 `trace_config.csv` 的 `inference_
group` 行。本仓为统一实例（P+D 同实例）；一 session 的权威 KV 基础历史
驻留单一 home 实例（执行副本另计）。

**F. 七仓对照表中本仓行**：三机制联合（T 逐出序 + J 联合选择 + E 自适
应层数）；三态 KV；home 保持 + 完成合并；八组合消融。其余六仓见
`template/astra-sim-sh_3.0/README.md` 对照表。

**G. 本地 HBM 带宽模型**：N 用户严格均分流体模型（读写同价、事件驱动
重分配、端点计费与直通规则）——沿用基底 `LocalHbmBandwidthModel`。

**H. KV 逐出与推理计算并行执行**（2026-09-13，基底改造随工作树继承）：
逐出物理传输旁路支链化 + store→restore 前递依赖补偿 + 瞬态双占用窗口
口径——本仓 merge 回传流同样经 `_register_store_tails` 登记，下一轮池读
经前递补边等待写入完成。

执行驱动机制层（RequestIngress/DecisionMailbox/GraphBatchCommitter 等）、
typed 响应解析（C1）、在线二进制旗标（`--online-*` 家族）、运行开销与
日志瘦身开关（SH_ARCHIVE_RUN/SH_SLO_POSTPROCESS/SH_GRAPH_DIGESTS 等）、
机制回归 fixtures、SLO 后处理链（slo_tools 九项产物）、冒烟输入纪律
（astra_compute_20.csv 前 2 秒）——均沿用基底实现与口径（详见基底
README 同名章节；本仓未改动其语义）。

## 8. 边界与纪律

- 仿真输入唯一允许源与窗口遵循 Agents.md 冒烟规范。
- 缺失输入 fail-closed；deep-gap 逐无可逐仍不满足按 rank 记账后
  fail-closed（沿用 D4）。
- 决策输入无 oracle：CSV 未来行/未来输出长度/真实返回时间不进任何
  决策；完成时已观测长度仅用于实际结算与在线估计器更新。
- E 的 adaptive 公式与因果规则已冻结（§5）；不做逐 workload 拟合。
- 本仓不运行三机制消融/外部对照实验（§8 第 6 步另行登记）。

[ASTRA-sim](https://astra-sim.github.io/) is a distributed AI system simulator.
It models the end-to-end software and hardware stack of modern AI systems.
For our website and documentation, see [astra-sim.github.io](https://astra-sim.github.io/).
