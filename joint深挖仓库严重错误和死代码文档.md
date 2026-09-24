# joint 仓深挖：严重错误与死代码（对照 face 模式）

> **审查对象**：`template/astra-sim-joint`（三机制联合仓：astra-sim 内核 + extern 网络后端/远端内存 + sh_test_mesh 脚本与配置域）。
> **方法**：14 个模块桶、345 个源文件分桶全覆盖深挖（每文件逐行读）；每条"严重错误/死代码候选"由独立复核员全新上下文裁定（只认亲眼读到的代码与 grep/find 输出）；`cmake` Release 构建 + `ctest` + `tests/run_all.sh` 确定性基线（**均 exit 0**，在 /tmp 仓外构建，仓库保持裸仓态）。
> **结果**：确认 **105 条**（**high 5 / medium 20 / low 80**）、**驳回 2 条**误报；同位点重复登记 9 组（对账规则见 §4，按清单登记条数计入总数）。
> **用途**：§1/§2 是本仓实锤结论；§3 是 face 14 个 P 模式在本仓的迁移命中对照；§4 留覆盖、基线与驳回存档，供修复工作流逐条落地。逐条完整证据见工作流产物"深挖审查全量证据表"（各条含候选原文 evidence 与独立复核意见 confirmNote）。
> **口径**：审查对象为**工作树当前状态（含未提交修改）**，git HEAD `9a95e06a` 与脏文件清单（43 M / 5 D / 33 ??）见 §4 基线取证；路径均相对仓根（`astra-sim/...`）。映射/KV 特性代码是核心资产，本清单只找"错"与"死"，不评架构。本文各条的压缩表述（触发前提、配置可达性、解析器一等配置值、实配键值等佐证细节）均出自该条独立复核意见（confirmNote）；终检另对关键断言实读核验并在文中标注行号（如 CollectiveImplLookup.cc:26/:40、Sys.cc:1210、llama2_7b_roofline_template.json:5、generated system.json:6-21、comm_group.json 9 组全带 dimensions、OnlineCli.hh:175 与 cc:324-329、graph_batch_builder.py:220-221/:306-307 等）。注意：全部 high 崩溃类发现均为**配置门控的潜伏缺陷**——官方实配（generated system.json:6-21 四个 implementation 键全 `["ring","ring"]`、三配置键齐备、无 custom 键；comm_group.json 全带 dimensions）当下不触发，属"配置一变即踩"的雷。

---

## 1. 确认的严重错误（重点）

### 1.1 high（5 条）

| # | 位置 | 问题与触发 |
|---|---|---|
| H1 | `astra-sim/system/astraccl/custom_collectives/CustomAlgorithm.cc:24` | 每次 custom 集合通信**泄漏 1 个 ETFeeder**（et_feeder 裸 new、类无析构、经基类指针 delete），长跑内存耗尽中止仿真；触发：system json 启用 `*-implementation-custom`（CollectiveImplLookup.cc:85-150 一等解析路径，仓内现配零命中——复核意见核记）。face P9 同位点，较 face（M7 medium）**升 high**。 |
| H2 | `astra-sim/system/astraccl/native_collectives/logical_topology/BinaryTree.cc:65` | 非 2 幂节点数时四个访问器经 `std::map::operator[]` 取出插入的 **nullptr 直接解引用 → 段错误**（构造只给 2^⌊log2N⌋ 个节点分配 id、无 2 幂守卫——复核意见核记）；`doubleBinaryTree` 是解析器一等配置值（CollectiveImplLookup.cc:26，终检实读），选中非 2 幂维度即确定性崩溃（现配全 ring）。face P8 同源实锤。 |
| H3 | `astra-sim/system/CommunicatorGroup.cc:146` | 子集群分支把 `CollectiveImplLookup` 注册表**共享**的 `CollectiveImpl*` 连同 `implementations_should_be_removed=true` 交给 plan，析构 delete 后注册表留**悬垂指针**：同 ComType 两组退出 **double free**、运行中查表 **UAF**（custom impl 单元素向量 + 无 dimensions 组才触发；官方 comm_group.json 恒带 dimensions 走安全分支——终检实读 9 组全带 dimensions）。face P3 同源。 |
| H4 | `astra-sim/system/Sys.cc:1189` | `determine_chunk_size` 用 `preferred_dataset_splits` 做除数，该键构造缺省 0、仅 JSON 含键才覆盖（Sys.cc:215/414-416 无 else 无零值校验），**缺键或显式 0 即除零 SIGFPE**——任何集合通信必经此路径；官方模板与生成配置供 6（llama2_7b_roofline_template.json:5，终检实读），属新配置漏写即踩。face P1 同源。 |
| H5 | `astra-sim/system/Sys.cc:1201` | `scheduling_policy`/`collectiveOptimization` 两枚举既无类内初始化（Sys.hh:348/381）也不在构造函数赋值，仅 JSON 含对应键才赋值；**缺键时读未初始化枚举（UB）**，垃圾优先级/调度分支随机，三值不中时 `assert(false)+exit(-1)`（Sys.cc:1210，终检实读）。face P1 同源。 |

### 1.2 medium（20 条）

| # | 位置 | 问题（一句话） |
|---|---|---|
| M1 | `astra-sim/common/AstraComputeAPI.hh/.cc` | 整对文件死代码：AstraComputeAPI/ComputeKernel/4 个 create_* 工厂全仓零引用，仍被根 CMake GLOB 编入 AstraSim 库（P12）。 |
| M2 | `astra-sim/common/AstraSimDataAPI.hh:33` | LayerData/AstraSimDataAPI 两类零引用死代码：唯一 include 是 system/ 下兼容 shim，而 shim 自身零包含者（P12）。 |
| M3 | `astra-sim/network_frontend/analytical/congestion_aware/main_online.cc:389-394` | `compute_mesh_edge_links` 对宽度 2 的 Ring 无条件按 2·npus 估算 link id，后端 radix==2 退化 mesh 实耗 npus 个 → 其后**所有维估算 id 整体偏移**、Ring-2 维边界链路漏出 edge 集，link_bucket/edge 归因失真（现配无 Ring-2，需用户自备配置触发——复核意见核记）。face P7。 |
| M4 | `astra-sim/system/astraccl/custom_collectives/CustomAlgorithm.cc:21` | 构造从不初始化基类 `data_size/final_data_size/comType`，CollectivePhase 构造立即读取这三个**未初始化成员（UB）**，垃圾值流入流优先级排序（复核意见：唯一排序消费分支当前被钉死 FIFO 排除，实锤为构造即 UB 读；需启用 custom 实现）。 |
| M5 | `astra-sim/system/astraccl/native_collectives/collective_algorithm/HalvingDoubling.cc:42` | 非 2 幂节点数无任何守卫：log2 截断 + `rank_offset` 倍增序列对非 2 幂 N 永不匹配 → **配对错乱**（收发不配对流挂死/静默失真；纯静态推演，`halvingDoubling` 为解析器一等配置值——CollectiveImplLookup.cc:40，终检实读）。face P8。 |
| M6 | `astra-sim/system/CommunicatorGroup.cc:146` | H3 同位点 medium 口径：单元素 impl 向量（1 维 native 或任一 `*-implementation-custom`——复核意见核记的触发形状）不经 `size>1` 替换直接入带删除语义 plan → 悬垂/双免；官方路径不触发，**纯数组组定义即合法输入形状踩雷**（与 H3 跨桶同登记）。face P3。 |
| M7 | `astra-sim/system/MyPacket.hh:26` | `stream_id` 全仓零赋值，Ring.cc:239 与 HalvingDoubling.cc:264 却读取传入 RecvPacketEventHandlerData——**未初始化读（UB）**（下游恰好无人消费，纯 UB 风险）。 |
| M8 | `astra-sim/system/scheduling/OfflineGreedy.cc:152` | `inter_dimension_scheduling` 钉死 Ascending 且无解析键 → offline_greedy 永不构造、`get_chunk_scheduling` 生产不可达；复核修正：原候选称 Journal 测试消费本函数不实（终检 grep 全仓仅 OfflineGreedy.hh:77、OfflineGreedy.cc:152/176 与死分支 Sys.cc:919 四处，Journal 测试只测 publish/consume）。face P2。 |
| M9 | `astra-sim/system/SimSendCaller.hh:18` | 消息大小整条 **int 截断链**（SimSend/SimRecvCaller→MemBus→MemMovRequest 均 int），uint64 count 静默窄化，>2^31 字节（≈2GiB）单条消息按错误字节数计时（rendezvous/延迟路径可达）。face P5。 |
| M10 | `astra-sim/system/Sys.cc:1189` | H4/H5 同位点跨桶登记（system 核心 B 桶口径：三配置键 contains-only、无有效缺省的成族登记）。face P1。 |
| M11 | `astra-sim/system/Sys.cc:209` | `intra_dimension_scheduling` 构造钉死 FIFO 且**无任何配置解析**（face 仓有 `intra-dimension-scheduling` 键，本仓没有）：insert_stream 中 RG/SmallestFirst/LessRemainingPhaseFirst 三分支死代码。face P2。 |
| M12 | `astra-sim/system/Sys.cc:210` | `inter_dimension_scheduling` 同款钉死 Ascending 无解析键：RoundRobin/OnlineGreedy/OfflineGreedy/Flex 全死分支，**offline_greedy 永不构造**，dim_to_break/logical_broken_dims 传递性死数据；八组合开关（README §1.1"八组合固定映射"，经环境变量注入脚本层——终检实读 README.md:42/:59）均在 workload/脚本层、不触及该枚举。face P2。 |
| M13 | `astra-sim/workload/execution_driven/ServiceCoordinator.cc:125` | `checked_wait_deadline` 护栏**单位错位**（秒 vs steady duration 的纳秒 count），(≈9.2e9, 9.2e18] 秒区间穿透护栏做超 int64 的 double→int64 转换（UB），注释宣称的 "bound-check BEFORE" 顺序实际未兑现；生产入口到不了 UB 区间——`--idle-watchdog-s` 解析对 >1.0e9 直接 reject（kMaxIdleWatchdogSeconds，OnlineCli.hh:175、OnlineCli.cc:324-329），main_online.cc:1830 只传该已校验值（终检实读调用链）；自测用例恰在 UB 区间靠 x86 饱和碰巧通过。 |
| M14 | `astra-sim/workload/execution_driven/tests/calendar_reader_oracle_test.cc:98` | legacy 对照臂注释称 "UNBOUNDED arm: window 0"，实际默认 high_water=128 有界且从不 notify_consumed——真实队列（文件自证 full 22,816-row TraceLab queue，旧有界臂在该队列实测 491 个 clamp 后提交；test :20/:30，终检实读）下两臂序列必不等、**文档化的真实输入验证模式必败**（合成 8 行 fixture 恰好单窗口可过）。face P10。 |
| M15 | `astra-sim/workload/execution_driven/WindowedTraceReader.cc:200` | CSV 数值列裸 `std::stoi/stoull` 无词法校验：负数经 strtoull **回绕**成 ~1.8e19、尾部垃圾截断，畸形值静默进仿真（结构性畸形均 fail-closed，唯数值列 fail-open；OnlineCli.cc:238 已修同款隐患而 CSV 侧未同步——终检实读）。face P8。 |
| M16 | `astra-sim/workload/MetricCollector.cc:512` | manifest event_code **先 `get<uint8_t>` 窄化后做 1..8 校验**：256 偏移非法码（264→8）被静默放行并误路由为对应事件，fail-closed 校验被窄化击穿。face M10/P5 同位点，未分化修复。 |
| M17 | `astra-sim/workload/RemoteFifoLedger.cc:18` | sensing 台账**生产链路整体死**：头注释指名 main_online.cc 在 --sensing-enabled 时 set_enabled，实际 main_online 零引用、enabled_ 恒 false，record_* 恒 no-op、sidecar/查询 API 全无生产消费者——台账遥测静默缺失且无任何报错（与 §2.1 workload 族 hh:82 同子系统跨桶登记）。 |
| M18 | `extern/network_backend/analytical/congestion_aware/fluid/tests/StaleEventCancellationTest.cpp:127` | 带宽常量按**旧二进制换算**（1e9/2^30 = "恰 1 B/ns"）取值，而 NetworkFunction.cpp 已改 SI 恒等（0.93 B/ns）：全部整数完成 tick 断言按现实现**必然失败**（算术推证，未运行）；opt-in OFF 排除在构建树外，从不报错。face H2 的 sibling 同款（与 M19 跨桶同登记）。face P6。 |
| M19 | 同 M18（extern 公共后端+远端内存桶独立复核登记） | 同一 fixture 双桶各自确认：SI 补丁作废了回归契约、又被构建排除遮蔽——测试资产失真，启用即必红。face P6。 |
| M20 | `sh_test_mesh/slo_tools/domain_metrics.py:554` | `_min_cost` 配额守卫与 M2/N6 docstring **语义相反**（布尔反演）：缺省 True 时配额拒绝候选被跳过（违 M2）、False 时反被纳入（违 N6），配额 run 的 alt_star/alt_star_actual 与 replay_mismatch 失真（复核意见核记 ：1491 消费链）；现有配额测试夹具全部只拒 remote-read、该分支零覆盖（复核意见核记 test_domain_metrics.py 三处夹具）。 |

### 1.3 未送复核的可疑点摘录（各桶存疑未立案，非实锤）

- `OfflineGreedy.cc:239-240` Flex 早退分支对已填满的 result 再次 push 非参与维（尾部重复项今日被消费方截断，且路径生产不可达）；
- `Sys.cc:872-879/1366` custom collective 用 queue_id 编码 pos_in_comm，`active_Streams.at(...)` 在 pos≥队列数时抛 out_of_range（仓内零配置启用，登记观察）；
- `LogGP.cc:243-252` Consider_Retire 取 `retirements.front()` 统计却 erase `talking_it`（多请求在途统计错位；仅 model-shared-bus=1 可达，官方模板无此键）；
- `GeneralComplexTopology.cc:26` 空 impl 向量时 `size()-1` 无符号回绕 + `get_basic_topology_at_dimension` 无越界检查（face P1 同族风险，触发链在 Sys/Workload 侧，shipped 配置四键齐全）；
- `main_online.cc:1313` detach FIFO 线程持有栈对象 ingress/svc 引用，run 末若被唤醒存在销毁后访问窗口（注释明示 fixture-only、随进程消亡）；
- `LocalMemUsageTracker.cc:199` recordWrites 对重复 tensor 写 assert(false)（release 下静默保留首写窗口 → trace 失真风险，未证实当前 trace 可触发）。

## 2. 确认的死代码与低危缺陷

low 共 80 条：69 条死代码按 13 族归类（§2.1），11 条低危严重错误单列（§2.2）。

### 2.1 死代码（low 69 条，按族归类）

| 族 | 代表实例（均经独立复核零引用/恒假/只写不读） |
|---|---|
| 整类/整文件零引用（GLOB 仍编入，P12） | `Torus3D`（构造内另有 `total_nodes/(vertical_dim*local_dim)` 潜在除零，死不可达——终检实读相符）；`LocalRingGlobalBinaryTree`；`LocalRingNodeA2AGlobalDBT`（dim==2 接回即返 nullptr——**死代码里还藏着错**）；`CSVWriter` 整类零实例化（唯一引用是 `UsageTracker::report` 形参，生产不调用）；`BasicLogicalTopology::basic_topology` 字段连同 BasicTopology 枚举无真实消费者 |
| astraccl 死函数/死成员/残片 | `BinaryTree::print(Node*)`、`RingTopology::is_enabled()` 零调用；`Algorithm::name` 只写不读；`CollectiveImplType` 4 个死枚举值（AllToAll/DoubleBinaryTreeLocalAllToAll/LocalRingNodeA2AGlobalDBT/HierarchicalRing）；`BYPASS_PERNODE_CUSTOM` 无生产者（头注释自认 no usecase，CollectiveImplLookup.cc:203 恒真比较）；`CustomAlgorithm::id` 私有遮蔽基类且只写不读；`Ring::dimension`、`HalvingDoubling::dimension/injection_policy/routing` 未初始化且零读写；`Ring.cc:130`/`HalvingDoubling.cc:141` 空条件死语句 `if (id == 0) {}`；`DoubleBinaryTreeAllReduce.cc:169` `snd_req2.dstRank` 误写 left_child（实发 right_child）——错值写进零消费字段 |
| 钉死枚举→死分支（low 条） | `Sys.cc:210`（system 核心 B 桶 low 登记，与 M11/M12 同位点跨桶重复）：RoundRobin/OnlineGreedy/OfflineGreedy* 与 RG/SmallestFirst/LessRemainingPhaseFirst 全部不可达、offline_greedy 永不构造 |
| system 层死函数/死字段 | `QueueLevels` 四参构造、`Roofline` 单参构造+`set_bandwidth`（且不初始化 bandwidth，2 条跨桶登记归并此处）、`SendPacketEventHandlerData` 双参构造（不初始化 wlhd）、`Sys::get_average_latency_per_dimension` 零调用；`Sys::stream_priorities` 只写不读；`BaseStream::test/test2/phase_latencies[10]` 调试残字段；`MyPacket` 死字段群（fm_id 零写零读，cycles_needed/sender/ready_time 只写不读） |
| common 死枚举值 | `PacketRouting` 枚举（限定名零引用，仅剩 HalvingDoubling 一个自身也死的字段）；`req_type_e` 的 BFLOAT16/FP32 成员（全部路径恒 UINT8） |
| 前端死接口/死字段/死键 | `--compute-scale` CLI 键（allow_unrecognised 下被静默吞）；`CmdLineParser::get_options()` 零调用；`OnlineDriverContext` 三字段只写不读（expected_requests 恒 0 快照，注释与实际用途不符） |
| execution_driven 死函数/死字段 | `was_json_id_committed`、`OnlineStatsCounters::reset()`、`RequestIngress::scheduled_future_arrival_count`（2 条跨桶登记归并此处）、`ServiceCoordinator::on_fence_scheduled/resolved` 死函数对（→ pending_fence 诊断恒 0）、`DecisionMailbox::count_no_decision_python_callback`（生产零调用 → main_online.cc:2081 run-end 门恒真）、`WindowedTraceReader::consumed_idx_` 只写不读 |
| execution_driven 恒假检查/注释失实 | `GraphBatchCommitter.cc:474/1041` comm tag（uint32_t）值域/负值检查恒假（P13，根因 GraphSource.hh:100 类型）；`OnlineCli.cc:251` uint64 上界检查恒假（P5，真溢出已由 ERANGE 拦）；注释失实四处：`GraphBatchCommitter.hh:36` [node] 规则漏 completion 第三边界（按文档实现会误拒合法批）、`OnlineCli.hh:26` "Frozen default 30e9"（实际 0 无界）、`OnlineCli.cc:262` "0 = off (frozen default)"（缺省已武装 1s）、`DecisionBridge.hh:227` channel_bytes 口径未随 B1 改造同步 |
| workload 层死代码 | `Workload.cc:410` `if (true)` 死 else + `issue_comp` throw 后不可达 return；`OnlineStatisticsState::operation_intensity/is_memory_bound/network_bandwidth` 只写不读（连带 `OperatorStatistics::network_bandwidth` 唯一读者是注释掉的报告块）；`HardwareResource.cc:148` 静态路径恒假再判、`report()`/`num_npus` 零消费；`MetricCollector` Phase-0 PerformanceCounters 整族死框架、`slo_watermark_period_ns()` 死访问器、`rank_instance_conflicts` 计数后 `(void)` 丢弃（注释却宣称 "counted, never silently resolved"）；`Statistics::retire_online_operator` 零调用且 hh:255 注释指认的填充路径失实；`RemoteFifoLedger` 永久失活死子系统（hh:82 条，P13，与 M17 同子系统跨桶登记） |
| extern 后端/远存死代码 | `Event::get_handler_arg()` 孤儿访问器（EventList 改用 QueuedEvent 后遗留）；`BasicTopology::get_basic_topology_type` 零调用且 Ring/Switch 从不赋值（调用即 assert/Undefined）；FluidScheduler Phase-7 拥塞快照集群零消费（`link_congestion_snapshot`/`LinkCongestionSnapshot`/`link_state_epoch_` 只写，P13 跨仓 parity 占位）；`get_completion_heap_size()` 零调用；`Link::bandwidth` 只写副本（仅 bandwidth_Bpns 被消费）；`AnalyticalRemoteMemory::architecture_name/port_mapping_rule` 零调用（头注释宣称的 main_online 消费不存在，P13） |
| 孤儿测试（P10，清单 5 条登记归并） | `cli_online_test.cc`（424 行）、`event_queue_deferred_test.cc`（486 行）、`ingress_idle_fixture.cc`（538 行）共 1448 行契约测试**不在任何构建目标**（头注 g++ 配方指向姊妹仓 wscllm；OnlineCli.cc:7 与 main_online.cc:11 却声称 "unit-tested in tests/cli_online_test.cc"）；`event_queue_deferred_test.cc:423` 恒真三元 `std::_Exit(reached_end ? 0 : 0)`（行尾注释语义未实现，不掩盖回归，独立位点单列） |
| 死配置键（P11） | `local-mem-capacity-bytes`（resolver 写 system.json、C++ 零读，2 条跨桶登记归并此处）；`boost-mode`（模板/生成配置透传零读者）；`logical-pool`（写 remote_memory.json 无功能性读者）；`slo_params_manifest.json` 10 个 campaign 键（scan_*/rolling_*/wp6_*/wp9_*）零脚本消费者（档案性条目，删前需登记） |
| 脚本域死依赖 | `run_golden_live.py:56` 硬依赖仓外不存在的 `/tmp/slo_wps/set_trace_pointer.py`，golden live 流程当前不可运行（sanctioned 手动例外，golden 语义离线由 test_golden_g1g4.py 承担——终检实读该文件存在） |

### 2.2 低危（low）严重错误（11 条）

| # | 位置 | 问题与触发 |
|---|---|---|
| L1 | `astra-sim/system/SharedBusStat.hh:137` | 计数器为 0 时 `take_bus_stats_average` 将 8 个 delay 累加器除以 int 计数器 → 0/0 得 **NaN**。**口径注明**：清单该条候选原文称"整型除零 SIGFPE"，独立复核意见实证修正——实读 SharedBusStat.hh:138-158，被除数全为 double 成员、除数为 int，`double /= int` 经通常算术转换是 IEEE 浮点除（0/0→NaN），不触发整型 #DE；本报告按复核口径记 NaN，仅污染内部统计字段、聚合后无任何读者，custom 流退休必现。face P4 机制命中、形态降级。 |
| L2 | `astra-sim/system/Sys.cc:1355` | custom 流 RecvPacketEventHandlerData 默认构造 owner=nullptr → `StreamBaseline::consume` 永不执行、net_message_counter 0/0 得 NaN（不出内部统计对象；需 custom 配置）。face P4 同源。 |
| L3 | `astra-sim/system/SimSendCaller.cc:13` | rendezvous 路径 uint64→int 截断，>2^31 字节消息符号扩展成 ~1.8e19 传后端；需 `--rendezvous-protocol` 开启且单条消息超 2^31 字节。face P5。 |
| L4 | `astra-sim/workload/Workload.cc:534` | roofline `num_ops==0` → 0/0=NaN 三路扩散（在线 compact 路径 isfinite 门 **exit 中止仿真**/static 统计污染/static_cast UB）；本仓生成器钳位结构性封死触发（graph_batch_builder.py:220-221 `_uint64=max(1,int(value))`，:306-307 作用于 comp 的 num_ops/tensor_size——终检实读），仅外部直写 JSON 可达（face P8 同位点，high 候选降 low）。 |
| L5 | `astra-sim/workload/MetricCollector.cc:1138` | active_kernel_roofline_* 仅 roofline_enabled 门控，total_comp_time==0 时 0/0 → NaN 被 dump 写成 null 且无 *_note（与 ：1659 同源分支的 `>0` 保护不一致）。face P8。 |
| L6 | `extern/network_backend/analytical/congestion_aware/basic-topology/Ring.cpp:22` | 一维 Ring 宽 1/2 重复/自环建链：Release 下 Device 静默覆盖、directed_links 留孤儿 LinkId 污染链路表与逐链遥测（debug 下 assert 崩溃）；需 Ring+宽 1/2 配置。face P8。 |
| L7 | `extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.cc:124` | remote-mem-bw 以 uint64 接收 json 浮点**静默截断**（与 Sys.cc double 消费精度分叉）；终检实读：全仓含该键的 JSON 仅 2 份（同一 runtime_config 目录的 system.json 与 remote_memory.json），值均 512.0 无损。face P5。 |
| L8 | `extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.cc:170` | PER_NODE 模式缺 num-nodes/num-npus-per-node **静默缺省 0**，构造成功、首次远端访问除零 SIGFPE/空容器越界；受权生成器不产出该形状，手写配置经 CLI 直通可达（复核意见核记）。face P1。 |
| L9 | `astra-sim/workload/execution_driven/tests/windowed_trace_reader_test.cc:1065` | Part Q "正常路径"对照臂把 128 当第三参（P0 重写后语义已变 max_arrival_ns），声明到达 1000/2000 ns 的两行全被拒、零提交，gate 断言**空转**（套件级覆盖仍在 Part K/M——终检核实在文件内，故降 low）。face P6。 |
| L10 | `sh_test_mesh/run_scripts/run_online_strategy.sh:278` | `set -e` 下 `wait CPP_PID` 非零即退出，:279-288 失败诊断尾段（log tail/"bridge retained"）在其目标场景（C++ 失败）**不可达**（最小复现已验证）；fail-closed 保持、日志仍落盘（:270-292 重定向），仅控制台诊断损失。 |
| L11 | `sh_test_mesh/run_scripts/run_online_strategy_sensing.sh:159` | 同 L10 同款（:159 wait + 同构诊断段），C++ 失败分支诊断不可达。 |

## 3. 对照 face 错误模式的命中小结

face 14 个 P 模式在本仓逐一重查，**12 个实锤命中**：P1（缺键除零/未初始化枚举，H4/H5/M10，较 face 升 high）、P2（intra/inter 维调度钉死死分支，M8/M11/M12）、P3（注册表共享指针悬垂/双免，H3/M6）、P5（int 截断链 M9、`get<uint8_t>` 先窄后校验 M16、恒假上界检查、remote-mem-bw 浮点截断 L7）、P6（旧 2^30 测试参照 M18/M19、Part Q 第三参语义漂移 L9）、P7（Ring-2 估算失配 M3）、P8（BinaryTree/HalvingDoubling 非 2 幂、CSV 裸解析 M15、roofline 0/0 L4/L5、Ring-1/2 建链 L6）、P9（ETFeeder 泄漏 H1）、P10（孤儿测试三件、legacy 对照臂 M14、恒真守卫/三元）、P11（boost-mode 等 4 处死键）、P12（GLOB 编入死文件 M1/M2 及 §2.1 首族）、P13（tag 恒假、parity 占位、远存导出助手）；P4 机制命中但**形态修正**——`double /= int` 经通常算术转换是浮点除得 NaN 而非整型 SIGFPE（终检实读 SharedBusStat.hh 核证，见 L1 口径注明），两条候选因此降 low（L1/L2）；P14 为复核纪律本身，本轮拦截 2 条误报（§4.1）。**未按 face 结论迁移处**：face 判死的 `link_count()` 在本仓有真实消费（main_online.cc:1564）、`on_local_hbm_restore_issue` 有真实生产调用（Workload.cc:477）、face H2 的 FluidSchedulerLinkObserverTest 在本仓无恙（:258 起 N11 注释已隔离 legacy 参照）——以上三处本轮终检实读核验；另"生产侧 SI 口径一致、无 2^30 残留消费点"仅系 extern 公共后端+远端内存桶核记（未独立复核，不作确认结论）。跨仓结论必须在本仓重新 grep 定案。

## 4. 覆盖与可信度

- **覆盖**：14 桶 345 文件全覆盖、每文件逐行读（common+astraccl 17 / system A 27 / system B 35 / native_collectives 28 / 前端 16 / workload A 6 / workload B 8 / ed 核心 A 12 / ed 核心 B 13 / ed 测试 A 12 / ed 测试 B 13 / extern 后端 29 / extern 公共+远存 13 / 脚本域 116），合计约 4.74 万行（脚本桶行数未单计）。
- **可信度**：审计发现均来自静态逐行阅读与 grep/find——构建/ctest/run_all 基线与 L10 的 wait/set -e 最小复现系审查流程另行执行（非逐行阅读所得，证据见下方基线条与对应条目）；105 条全部经独立复核员全新上下文逐条裁定（只认亲眼读到的证据）；2 条驳回留档见 §4.1。
- **对账规则**（总数 105 按清单登记条数计）：§1 的 25 条（high 5 + medium 20）与 §2.2 的 11 条与清单**逐条一一对应**；§2.1 的 69 条按 13 族归并，其中跨桶同位点重复登记 9 组——①H3/M6（CommunicatorGroup.cc:146）、②H4/M10（Sys.cc:1189）、③M8/M12 与 §2.1"钉死枚举"low 条（inter/intra 维调度 3 条登记）、④M18/M19（StaleEventCancellationTest.cpp:127）、⑤Roofline.cc:13/:14（2 条 low）、⑥RequestIngress.hh:228（2 条 low）、⑦config_resolver.py:434（2 条 low）、⑧M17 与 §2.1 workload 族 hh:82 条（RemoteFifoLedger medium+low）、⑨孤儿测试 5 条登记（cli_online_test.cc :1/:59/合并条 + event_queue_deferred_test.cc:1 + ingress_idle_fixture.cc:1）归并一格（:423 恒真三元为独立位点单列）——重复条均并入相应族并在族内注明。
- **基线**（/tmp/joint-deep-review-build 仓外构建，仓库保持裸仓态）：`cmake` Release configure **exit 0**（GNU 15.2.0 / Release / spdlog 1.14.1 / Protobuf 3.21.12）；build **exit 0**；`ctest` **exit 0**——但输出 "No tests were found!!!"（唯一 add_test 的 OfflineGreedyScheduleJournalTest 缺省 OFF，ctest 空转：孤儿/opt-in 测试的失败不反映在基线）；`tests/run_all.sh` **exit 0**（Ran 54 tests OK）。
- **版本**：git HEAD `9a95e06a78f004ca37f6fad62791bb65069b005f`；工作树脏文件 81 项（43 M / 5 D / 33 ??），本次审查按工作树现状（含未提交修改）取证。
- **未覆盖/限缩**：运行期行为与长跑稳定性未实测（未跑仿真），M5 卡死、M18 必败等结论系纯静态/算术推演；vendored 三方库（extern/helper 的 json.hpp、spdlog/fmt）未审；NDEBUG 下断言失效、EventQueue 析构顺序等上游继承行为因无具体可达证据未立案。

### 4.1 复核驳回留档（2 条）

| 位置 | 驳回理由（一句话） |
|---|---|
| `astra-sim/system/Usage.cc:10`（候选：Usage 整类零引用） | 驳回：UsageTracker.cc:23/36/49 三处栈上实例化 Usage，increase/decrease_usage 被生产 Sys.cc:89/116 真实调用——原 grep 被路径名含 "UsageTracker" 的过滤误伤。 |
| `astra-sim/system/CSVWriter.cc:22`（候选：CSVWriter 全链死代码、唯一入口 UsageTracker::report 零调用） | 驳回："report 零调用者"不成立——system_history_lifecycle_test.cc:314 `online.report(nullptr, 0)` 是编译内真实调用且该测试在构建目标内（analytical CMakeLists.txt:342）；窄口径残余（类零实例化、initialize_csv/finalize_csv 零调用）已保留在 §2.1。 |
