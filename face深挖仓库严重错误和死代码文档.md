# face 仓深挖：严重错误与死代码（供姊妹仓套用）

> **审查对象**：`template/astra-sim-face`（FACE 动态全池化混合调度基准仓）。
> **方法**：14 个模块、181 个源文件分桶全覆盖深挖（每文件逐行读，snippet 预验证无遗漏）；每条"严重错误/死代码候选"由独立复核员全新上下文裁定（只认亲眼读到的代码）；`cmake` Release 构建 + `ctest` 确定性基线（**均 exit 0**，在 /tmp 仓外构建，仓库保持裸仓态）。
> **结果**：154 条发现 → **确认 86 条**（high 3 / medium 12 / low 71）、**驳回 13 条**误报、其余为未送复核的可疑点。
> **用途**：§1/§2 是本仓实锤结论；§3 是提炼的**错误模式清单**——其余 5 个同源仓（face-LRU / wscllm / wscllm-LRU / joint 等，与 face 已分化 70+ 处）深挖时按模式优先排查。逐条完整证据见工作流产物"深挖审查全量证据表"与发现看板（154 项）。
> **口径**：路径均相对各仓根（`astra-sim/...`），姊妹仓同构可直接对位。各仓映射/KV 特性代码是核心资产，本清单只找"错"与"死"，不评架构。

---

## 1. 确认的严重错误（重点）

### 1.1 high（3 条）

| # | 位置 | 问题与触发 |
|---|---|---|
| H1 | `astra-sim/system/CommunicatorGroup.cc:130-148` | 子集群通信域分支把 `CollectiveImplLookup` 注册表**共享**的 `CollectiveImpl*` 交给 `implementations_should_be_removed=true` 的 plan：同 ComType 两个子集群组退出 → **double free**；任一组运行期销毁后 `get_collective_impl` 返回悬垂指针 → **UAF**。本地 `config_resolver` 生成的 comm_group 恒带 dimensions 走安全分支；数组形式/无 dimensions 的 comm_group.json 或 torch pg 子集 ranks 组即触发。 |
| H2 | `extern/network_backend/analytical/congestion_aware/fluid/tests/FluidSchedulerLinkObserverTest.cpp:256-257` | legacy 参照速率按**旧二进制换算**（2^30/1e9）取值，而同提交 4e9e2ca 已把 `bw_GBps_to_Bpns` 改为 SI 恒等（1 GB/s=1 B/ns），该回归 fixture 在 HEAD **必然失败**（复核实测 exit 1；`StaleEventCancellationTest.cpp:127` 同款 medium）。**凡改过带宽单位换算的姊妹仓必查同类测试参照。** |
| H3 | `examples/run_scripts/analytical/congestion_aware/`（HGX-H100-validated.sh / Ring_allgather_16npus.sh / run_analytical_with_custom_collective.sh） | 引用的 `build/astra_analytical/build.sh` 与无 `_Online` 后缀的二进制随 Path-1 移除**已不存在**，脚本必失败；`--remote-memory-configuration` 参数被 `allow_unrecognised_options` **静默吞掉**。 |

### 1.2 medium（12 条）

| # | 位置 | 问题（一句话） |
|---|---|---|
| M1 | `astra-sim/system/Sys.cc:1165` | 配置缺 `preferred-dataset-splits` 键时构造值 0，`size/0` → **SIGFPE 崩溃**（462 份现存配置均含键，属新配置漏写即踩的雷）。 |
| M2 | `astra-sim/system/Sys.cc:1177-1181` | `scheduling_policy` 无构造初始化，缺 `scheduling-policy` 键时读未初始化枚举（**UB**），调度分支随机。 |
| M3 | `astra-sim/system/Sys.cc:923-924` | `collectiveOptimization` 同款缺键 UB，All_Reduce 单趟/RS+AG 双趟**随机选定**，通信量失真。 |
| M4 | `astra-sim/system/SimSendCaller.hh:18-19` | SimSend/SimRecvCaller 用 **int 存消息大小**，`uint64_t count` 静默截断，>2GiB 单条消息（rendezvous 路径可达）按错误字节数算延迟。 |
| M5 | `astra-sim/system/Sys.cc:285-289` | `inter_dimension_scheduling` 构造钉死 Ascending 且**无任何配置解析**：RoundRobin/OfflineGreedy 全部死分支，用户手写该键被**静默忽略**。 |
| M6 | `astra-sim/system/SharedBusStat.hh:136-146` | 请求计数器为 0 时 `double /= 0` → **NaN 静默污染统计**（custom collective 流从不递增计数器，阶段完成必现）；姊妹点 `Sys.cc:1331` 同款。 |
| M7 | `astra-sim/system/astraccl/custom_collectives/CustomAlgorithm.hh:36-66` | 无析构函数，每次 custom 集合通信**泄漏 1 个 ETFeeder**（含打开的 fd），fd 耗尽后 `std::exit(1)` 中止仿真（条件：启用 custom 实现）。 |
| M8 | `astra-sim/system/astraccl/native_collectives/logical_topology/BinaryTree.cc:64-70` | 仅支持 2 的幂节点数；非 2 幂维度经 `std::map::operator[]` 取出 nullptr 直接解引用 → **段错误**（复核实机复现 exit 139；仅 DBT 算法，Ring/HalvingDoubling 不受影响）。 |
| M9 | `astra-sim/workload/Workload.cc:461-463` | roofline 模式 `num_ops==0`（或配置缺 local-mem-bw → perf=0）时 `0/0` → NaN/Inf，`static_cast<uint64_t>` 为 **UB**；在线 compact 路径遇非有限值直接 **exit(1)** 中止整个仿真。 |
| M10 | `astra-sim/workload/MetricCollector.cc:512-517` | node event_code **先按 uint8_t 窄化再校验范围**：≥256 的非法码回绕成合法码（257→1=PREFILL_START_ISSUE），本应 fatal 的清单被静默接受、指标错挂。 |
| M11 | `astra-sim/network_frontend/analytical/congestion_aware/main_online.cc:755-761` | link id 估算器按 Ring 维 `2*npus` 计，但宽度==2 的 Ring 实际退化为 mesh 只耗 `npus` 个 id → 该维之后**所有维 link id 整体偏移**，edge 归因/link_bucket 全错位（需多维权且含 Ring-2 维）。 |
| M12 | 测试域 3 条 | ① `tests/calendar_reader_oracle_test.cc:96-99` legacy 对照臂注释称 window 0 无界、实际默认 high_water=128 有界——真实队列模式**必误报失败**（复核实测 legacy=43 vs calendar=200）；② `tests/cli_online_test.cc`（R1–R14 契约测试）**未入任何构建**，仅头注 g++ 手工可达；③ `ServiceCoordinator` fence API 零调用 → `pending_fence_count()` 诊断恒 0（失真）。 |

### 1.3 未送复核但值得人工看（可疑点摘录，非实锤）

- `Sys.cc:1169`：All_Gather 且 size < splits 时 chunk 整除得 0；
- `Sys.cc:219`：`collective_impl_lookup` 构造 new、析构不 delete（每 Sys 实例泄漏 1 个）；
- `LogGP.cc:72`：size==0 传输使 `(size-1)` 回绕为负延迟加到无符号 Tick；
- `MetricCollector.cc:2053-2064`：WP8 直排积分不夹负、与水位线积分口径不一致；
- `tests/windowed_trace_reader_test.cc:1065`：Part Q 疑似固化错误行为的断言。

## 2. 确认的死代码（71+ 条，按族归类）

| 族 | 代表实例（均已独立复核零引用/恒假） |
|---|---|
| 整类/整文件零引用 | `AstraSimDataAPI/LayerData`（system 与 common **双副本同 guard**）；`Torus3D`；`LocalRingGlobalBinaryTree`；`LocalRingNodeA2AGlobalDBT`（其 dim==2 分支接回即返回 nullptr，**死代码里还藏着错**）；`AstraComputeAPI/ComputeKernel`+4 个 create_llm_* 工厂（get_static_runtime 还返回未初始化聚合）；`MetricCollector` Phase-0 PerformanceCounters 整套框架；FluidScheduler Phase-7 拥塞快照集群（注：joint 仓 `link_count()` 有真实消费，跨仓删除需登记豁免） |
| 钉死枚举→死分支 | `intra_dimension_scheduling` 钉死 FIFO → RG/SmallestFirst/LessRemainingPhaseFirst 三分支永假（连同 `inter_dimension_scheduling` 见 M5）；`HardwareResource.cc:123-125` else 内恒假 `==0`；`Workload.cc:404-408` `if(true)`；`GraphBatchCommitter.cc:459-465`/`1016-1028` comm.tag（uint32_t）值域检查恒假；`OnlineCli.cc:233-237` uint64 上限检查恒假 |
| 只写不读字段 | `network_bandwidth`（唯一"读者"是 Statistics.cc:834-862 **注释掉的报告块**）；`consumed_idx_`；`OnlineStatisticsState::operation_intensity/is_memory_bound`；`message_end`；`MyPacket::call`+notifier 回调链；`MemMovRequest::latency`（恒传 0）；`rank_instance_conflicts`（累加后 `(void)` 丢弃，注释却宣称 "conflicts are counted"）；`OnlineDriverContext` 三字段（`expected_requests` 快照恒 0） |
| 死函数/死访问器 | `was_json_id_committed`、`FileDecisionBridge::stats()`、`slo_watermark_period_ns()`、`OnlineStatsCounters::reset()`、`CmdLineParser::get_options()`、`get_average_latency_per_dimension`、`on_local_hbm_restore_issue`（跨仓 SH2 parity 占位）、`flush_pending_starts_deferred`（头注释"must go through this entry"失实）、`Statistics::retire_online_operator`（hh:254 注释指认的填充路径也已失实） |
| 孤儿测试 | `cli_online_test.cc`、`event_queue_deferred_test.cc`（头注还写着 wscllm 仓根）、`ingress_idle_fixture.cc`——均未入 CMake，仅注释 g++ 可达，19 个兄弟测试都在构建里 |
| 死配置 | `remote-mem-bw`（config_resolver.py:366 写入、C++ 零读取，后端已移除）；`boost-mode`（模板透传无人读）；`pipeline-tile-fraction`；`slo_params_manifest.json` 21 个冻结参数中 11 键无脚本消费者 |
| 遗留残片 | `sh_test_mesh/slo_tools/` 下 5 个 `.bak_caliberfix_20260905`；`inputs/` 整目录 10 个上游样例 json 零引用（与"裸仓两条在线路线"声明矛盾） |

## 3. 姊妹仓深挖套用清单（错误模式 → 排查动作）

| # | 模式 | face 实例 | 排查动作 |
|---|---|---|---|
| P1 | 配置键缺省缺失 → UB/除零 | M1/M2/M3 | 逐键比对 `Sys::initialize_sys` 解析键 vs 构造函数默认值：无默认值的键（枚举、除数）即雷 |
| P2 | 钉死枚举 → 死分支 + 静默吞配置 | M5、intra 维 | grep 成员全部写点；写点唯一且为常量 ⇒ 全部分支死、手写键被忽略 |
| P3 | 共享指针所有权别名 → double free/UAF | H1 | 注册表/缓存返回裸指针处，查接收方是否带删除语义（plan 析构、重载赋值） |
| P4 | 计数器 0 除 → NaN 污染统计 | M6 | 所有 `平均时间 /= counter` 处问一句：存在 counter 恒 0 的流类型吗 |
| P5 | 整型窄化/截断 | M4、M10 | uint64→int 形参；`get<uint8_t>()` 先窄后校验；`stoi` 尾随垃圾；uint64 比较 uint64::max 恒假 |
| P6 | 单位换算改动未同步测试参照 | H2 | 改过 GB(2^30)/SI 换算的仓，所有测试常量按新口径重推（LRU/joint 重点） |
| P7 | 估算 id 与实际建链失配 | M11 | 拓扑 id 估算函数 vs 后端真实 connect 次数，逐维类型核对（含退化分支） |
| P8 | 非常规输入形状崩溃 | M1/M8/M9 | 非 2 幂拓扑、size==0、num_ops==0、缺键 json——每个外部输入的形状假设都要有守卫 |
| P9 | 资源泄漏中止仿真 | M7 | feeder/句柄类：构造开资源、析构必须关；`new` 成员查 delete |
| P10 | 孤儿测试/恒真断言/注释构建 | M12 | 测试文件是否在 CMake 目标里；`expect(true)`、被改 fixture 掩蔽的对照臂 |
| P11 | resolver 透传死键 | remote-mem-bw 族 | python 写入的每个 system json 键逐键 grep C++ 读取点 |
| P12 | CMake GLOB 编入死文件 | Torus3D 等 | 零引用 ≠ 不可见：GLOB 模式的库，删源文件与"是否编译"无关，需单独审计 |
| P13 | 跨仓 parity 占位零调用 | on_local_hbm_restore_issue 等 | 头注释自认 "no caller/parity" 的接口单独列表，删除前查姊妹仓是否消费 |
| P14 | 死代码判定必须复核 | 13 条驳回 | 本轮 `BasicEventHandlerData` 默认构造、`ask_for_schedule`、`pending_events` 均被误判死代码（实有测试/生产引用）——**零引用结论必须由第二人独立 grep 全形式（宏/字符串/测试/CMake）** |

## 4. 覆盖与可信度

- **覆盖**：`astra-sim/` 181 个源文件分桶全覆盖（glob 预验证无遗漏无空桶）；14 模块并行；extern 仅定制甄别（helper/ 三方库未审、上游文件抽样）；运行期行为/长跑稳定性未实测。
- **可信度**：86/154 经独立复核确认；13 条驳回已留档；`cmake` Release 构建 exit 0、`ctest` 全绿 exit 0（注意：cli_online_test 等 3 个孤儿测试与默认 OFF 的 fluid fixture **不在** ctest 范围内，故其失败不反映在基线上）。
- **完整证据**：工作流"深挖审查全量证据表"（逐条 path:line + 复核意见）与发现看板（154 项含未送复核可疑点）；本文档仅录重点。

---

## 5. 修复状态（2026-09-23，两轮修复工作流后）

**验收基线**：修复后 `cmake` Release 构建 exit 0；`ctest` 全部通过（含新接入的 3 个孤儿测试）；fluid 两个 opt-in 回归测试专项验证通过；冒烟 `agent-traces/tracelab/astra_compute_20.csv` 前 2 秒官方 runner 通路通过；裸仓四要素还原核验通过；未做任何 git commit。

### 5.1 已修复（严重错误全量）
- **high 3**：H1 所有权改为保型克隆（未用基类拷贝——`CollectiveImpl` 非多态且消费点有 C 风格下转，拷贝会切片，改为按 type 分派克隆）；H2/H3 已修/已删。
- **medium 12**：M1–M12 全部修复（M5 补齐 `inter/intra-dimension-scheduling` 配置解析并登记开关清单；M8 非 2 幂 fail-closed；M10 宽值校验后再窄化等）。
- **补漏同族 3 条**（修复过程中逐条登记、逐条补修）：`NetworkStat.hh:40` 零计数器除法守卫（M6 同款）；`CommunicatorGroup.cc` 子集群分支空实现向量 fail-closed；`CommunicatorGroup.cc` 整集群分支（主路径）同型 fail-closed（均防 `Sys.cc:839` 越界 UB，坏配置由 UB 变为清晰报错）。
- 另修：`~Sys` 补 `delete collective_impl_lookup`；`LogGP.cc:72` size==0 守卫。

### 5.2 已清除（死代码）
- 删除整文件：`Torus3D`、`LocalRingGlobalBinaryTree`、`LocalRingNodeA2AGlobalDBT`（.cc/.hh）、`common/AstraComputeAPI.hh/.cc`、`common/AstraSimDataAPI.hh`、`system/AstraSimDataAPI.hh`、`inputs/` 目录、3 个失效 example 脚本、5 个 `.bak_caliberfix` 残片。
- 删除死分支/死字段/死函数/死访问器：钉死枚举分支、恒假检查、只写不读字段（`network_bandwidth` 全链、`consumed_idx_`、`message_end`、MyPacket 回调链等）、`retire_online_operator`、PerformanceCounters 框架、`on_local_hbm_restore_issue`、`get_options()`、`was_json_id_committed`、fence API、`reset()`、`stats()` 等；`DecisionEvent::seq` 等同批评估后按口径保留（见 5.3）。
- 死配置：`remote-mem-bw`（resolver+断言同步清理）、`boost-mode`、`pipeline-tile-fraction`。
- 测试接入：`cli_online_test` / `event_queue_deferred_test` / `ingress_idle_fixture` 已入 CMake；`calendar_reader_oracle_test` legacy 臂改真无界（window 0）。

### 5.3 评估后保留（非死代码或留证，姊妹仓套用时同口径）
- **仅测试引用的观测 API**：`NodeStore::pending_count`、`DecisionEvent::seq`、`WindowedTraceReader`/`RequestIngress` 一批观测访问器、`Workload::fire()`、`Sys::pending_events`、`ask_for_schedule`、`BasicEventHandlerData` 默认构造——有真实消费者（测试/生产），复核已驳回"死代码"误判。
- **跨仓输出 schema 稳定**：`local_hbm_restore_bytes` 发射行保留（恒 0，注释已注明 face 无该模型）。
- **冻结留证数据**：`slo_params_manifest.json` 11 个无消费者键保留（留证文件而非代码）。
- **已知残留风险（登记待议，未修）**：`MetricCollector` 静态事件在在线生产路径会被 `clear_static_node_events()` 清除（错挂影响仅限直接消费方测试）；`scheduling-policy` 等新键写入非字符串 JSON 时沿用 nlohmann 异常路径（与既有键一致）。

### 5.4 文档同步
face `README.md` 已按删改同步；`experiment/仿真各功能开关清单.md` 已登记两个新配置键并退役死键条目。
