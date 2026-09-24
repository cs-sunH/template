# wscllm 仓深挖：严重错误与死代码

> **审查对象**：`template/astra-sim-wscllm`（face 同源姊妹仓：astra-sim 内核 + 在线图执行根 + FluidScheduler 网络后端 + sh_test_mesh 负载物化与 SLO 工具链）。
> **方法**：16 个审查桶、321 个文件分桶全覆盖（13 个 C++ 桶 229 文件逐行读，仿真脚本运行链/负载物化脚本/杂项与构建编排 3 桶 92 文件；全桶清单与缩写见 §3 覆盖与口径行）；每条"严重错误/死代码候选"由独立复核员全新上下文裁定（只认亲眼读到的代码）。基线：`cmake` Release 构建 exit 0；`ctest` 退出码 0 但系**零用例空跑**（根 CMakeLists.txt:117 唯一 CTest 注册项默认 OFF，默认基线不含任何 C++ 单测，exit 0 不构成回归证据）——工作流基线记录，本文档撰写未重跑。**本轮全部结论基于静态逐行读码 + 独立复核亲读，无本仓运行期实测**（证据口径见 §3）。chakra（extern/graph_frontend）与 face 仓逐字节一致（`diff -rq` 为空）按 face 轮口径豁免深挖；extern/helper 三方库（spdlog/yaml-cpp/fmt/json/cxxopts）未审。
> **结果**：**确认 123 条**——错误 41（high 4 / medium 30 / low 7）、死代码 82；独立复核"驳回"裁定 2 个（**净驳回不进正文 1 条**，另 1 条双裁并存按确认收录——见 §3）；其余为未送复核可疑点（候选总量与复核覆盖见 §3，摘录见 §1.3）。
> **用途**：§1/§2 是本仓实锤修复清单（修复与保留口径直接套用 face 文档 §5.2/§5.3：删死码、补缺键守卫、接回孤儿测试、冻结留证保留；**并按 §1.4 并查 face §5.1 已修同源雷、按 §2 豁免注防跨仓误删**）；附录为本轮新提炼模式；§3 交代覆盖、计数口径与裁定分歧。逐条完整证据见本轮工作流"深挖审查全量证据表"与发现看板。
> **口径**：路径均相对本仓根 `template/astra-sim-wscllm/`；计数按"桶×条"——同一问题多桶独立登记分别计入总数，正文表格已按问题合并去重呈现（41 条错误去重后 29 个独立问题、82 条死代码去重后 70 个独立问题，行内括号注明条数）；只找"错"与"死"，不评架构。桶名缩写：**A 桶/B 桶**＝system核心A/B，**astraccl**＝astraccl集合通信，**根A/根B**＝在线图执行根A/B，**测试A/B/C**＝在线图测试A/B/C，**网络前端**＝网络前端与公共接口，**后端A/后端B**＝外部网络后端A/B，**脚本链**＝仿真脚本运行链，**物化**＝负载物化脚本，**杂项**＝杂项与构建编排。

---

## 1. 确认的严重错误（重点）

### 1.1 high（4 条，3 个独立问题）

| # | 位置 | 问题与触发 |
|---|---|---|
| H1 | `astra-sim/system/CommunicatorGroup.cc:130-148` | **face H1 原样仍在**：无 dimensions 的子集群通信域分支把 `CollectiveImplLookup` 注册表**共享**的 `CollectiveImpl*` 原封交给 `implementations_should_be_removed=true` 的 CollectivePlan，组析构即 `delete` 注册表指针——同 ComType 两个子集群组先后退出 → **double free**；先退组销毁后再 `get_collective_impl` 返回悬垂指针 → **UAF**；且 `CollectiveImpl` 无虚析构（CollectiveImpl.hh:36-43，CustomCollectiveImpl 含 std::string 成员），经基类指针 delete 亦 UB+泄漏。触发：数组形式/缺 dimensions 键的 comm_group.json（Workload.cc:228-233 令 dimension_sizes 为空）或 torch pg 子集 ranks；生产 `config_resolver.py:391-393` 恒写 dimensions 走安全分支（潜在雷），torch pg 子集路径因在线链路 inputs_values 恒空而休眠。**system核心A、astraccl 两桶独立确认，计 2 条。** |
| H2 | `astra-sim/system/Sys.cc:1176` | 配置缺 `preferred-dataset-splits` 键时 `preferred_dataset_splits` 保持构造值 0（Sys.cc:214；仅键存在才赋值 Sys.cc:413-415，全仓无第三处赋值/校验），`determine_chunk_size` 无守卫 `size / preferred_dataset_splits` → **SIGFPE**；在线链路真实可达（Workload.cc:636→generate_all_reduce→Sys.cc:833）。出厂模板与 config_resolver 派生链均含该键——**手写新 system json 漏写即踩**（face M1 同款，severity 升降见下方校准说明）。A 桶登记 high；B 桶同条登记 medium（计入 §1.2 的 30 条口径，见该节说明）。 |
| H3 | `astra-sim/system/astraccl/native_collectives/logical_topology/BinaryTree.cc:65` | **face M8 同款**：doubleBinaryTree 实际只建 2^floor(log2(T)) 个节点，非 2 幂节点维时最高若干 rank 经 `std::map::operator[]` 取出 nullptr 直接解引用 → **段错误**。触发：system json 配 `doubleBinaryTree` 且参与维节点数非 2 幂——本仓 54=6×8 NPUs，6 的维即触发；出厂模板全 ring 未启用（severity 升降见下方校准说明；仅 DBT 算法，Ring/HalvingDoubling 不受影响）。 |

### 1.2 medium（30 条，20 个独立问题）

> **30 条的构成（本节自含）**：下表 20 行按"桶×条"括注合计 **29 条**；另有 B 桶把 §1.1 H2（`Sys.cc:1176`，A 桶登记 high）同条**另登记 medium 计 1 条**——同一问题由两桶独立裁定、在 high/medium 各计 1 条。表内没有这一行，重构 30 须并入该条。

| # | 位置 | 问题（一句话，含触发） |
|---|---|---|
| M1 | `astra-sim/system/Sys.hh:347`（2 条） | `scheduling_policy` 无类内初始化器、构造函数不赋值（全仓唯一赋值点在 `j.contains("scheduling-policy")` 守卫内、无 else 默认值），system json **缺键**时 `get_priority` 读未初始化枚举（**UB**），调度优先级方向随机；非法值会 panic、缺键完全无守卫（face M2 同款）。 |
| M2 | `astra-sim/system/Sys.hh:380`（1 条） | `collectiveOptimization` 同款缺键 UB：缺 `collective-optimization` 键时 All_Reduce 单趟或 RS+AG 双趟**随机选定**，通信量口径失真（face M3 同款）。A 桶同条以"现存 14 个 system JSON 触发不可达"驳回、B 桶复核确认收录——两裁并存，见 §3。 |
| M3 | `astra-sim/system/Sys.cc:208-209`（2 条） | `intra/inter_dimension_scheduling` 全仓唯一写点**钉死** FIFO/Ascending、无任何配置解析（`dimension-scheduling` 键全仓零命中）：RoundRobin/OnlineGreedy/OfflineGreedy(Flex) 与 RG/SmallestFirst/LessRemainingPhaseFirst 全部死分支，手写该键被**静默忽略**；连带 `offline_greedy` 恒 nullptr（Sys.cc:290-293 恒假），OfflineGreedy 整链生产不可达（仅剩单测引用且其 CMakeLists.txt:105 option 默认 OFF、无脚本打开）（face M5 同款）。 |
| M4 | `astra-sim/system/PacketBundle.cc:57-61`（2 条） | `needs_processing` 路径无守卫除以 `sys->local_mem_bw`（缺 `local-mem-bw` 键构造默认 0）：`size/0=+inf`（size=0 得 NaN）→ `static_cast<uint64_t>(inf)` 为 **UB**（x86 实践得巨 tick），仿真挂死/结果失真；Sys.cc:460-464 的零速率守卫只关 hbm contention 不覆盖此处，**违背 README G 节"local-mem-bw<=0 自动回退旧行为"承诺**；且该路径是 ring RS/AR 数据包的常规路径（Ring.cc:124-127）。 |
| M5 | `astra-sim/system/SharedBusStat.hh:136-146` | 请求计数器为 0 时 `double /= 0` → **NaN 静默污染统计**；custom 集合通信流从不递增计数器（唯一递增点 StreamBaseline.cc:43 不在 custom 回调路径上，custom 事件均路由 CustomAlgorithm::call），阶段完成必现（face M6 同款）。 |
| M6 | `astra-sim/system/SimSendCaller.hh:18`（SimRecvCaller.hh:19 同款） | SimSend/SimRecvCaller 用 **int 存消息大小**，`uint64_t count` 经 Rendezvous/延迟路径（Sys.cc:1534-1537/1555-1558、RendezvousSendData.cc:21-30 等）静默窄化，>2GiB 单条消息按错误字节数算传输延迟（face M4 同款）。 |
| M7 | `astra-sim/system/astraccl/custom_collectives/CustomAlgorithm.hh:36-66`（2 条） | 无析构函数，构造时 `new` 的 ETFeeder（内含常开 ifstream fd）永不释放：Sys.cc:1166 每次 generate_collective_phase 都 new 一个 → 每执行一次 custom 集合通信**泄漏 1 个 ETFeeder+fd**，fd 耗尽后打开抛异常、ctor 捕获后 `std::exit(1)` **中止整个仿真**；触发=配置任意 `*-implementation-custom`（生产模板未启用）（face M7 同款）。 |
| M8 | `astra-sim/workload/Workload.cc:471-473` | roofline 闭式路径对 `num_ops==0` 或 `perf==0`（local-mem-bw/peak-perf 缺键时默认 0，Sys.cc:179/184+416-426 条件解析）无守卫：`0/0` → NaN/Inf，`static_cast<uint64_t>` 为 **UB**；在线 compact 路径遇 NaN 直接 **exit(1)** 中止整个仿真（face M9 同款）。 |
| M9 | `astra-sim/workload/MetricCollector.cc:512-517` | node event_code **先 `get<uint8_t>()` 窄化再校验 1..8**：≥257 的非法码回绕成合法码（257→1=PREFILL_START_ISSUE）被静默接受，本应 fail-closed 的清单校验被绕过（face M10 同款）。 |
| M10 | `astra-sim/workload/execution_driven/WindowedTraceReader.cc:200-205` | CSV 行字段解析用**裸 `std::stoi/stoull`**（:200/202-204/285/304 五个调用点，全文件唯一 try 在 :348-350 的 JSON 门不护行解析）：尾随垃圾被静默截断接受（`'1000x'` 按 1000 入账、统计失真）；空/非法字段抛未捕获异常 → **std::terminate SIGABRT**，绕过本文件自身 `[Error]...exit(EXIT_FAILURE)` 的 fail-closed 约定（两处 pump 调用点 main_online.cc:1159/1488 均不在 try 内）。 |
| M11 | `astra-sim/workload/execution_driven/tests/calendar_reader_oracle_test.cc:98` | oracle 的 legacy 对照臂宣称 window=0 全量无界，实际两参构造落默认 `high_water=128` **有界窗口**（LegacyOracleWindowedTraceReader.hh:52-54 默认形参）——真实队列（argv[1]）模式下该等价性测试**必误报失败**（face M12① 同款；静态推证，未实测）。 |
| M12 | `astra-sim/workload/execution_driven/tests/windowed_trace_reader_test.cc:1065` | Part Q"正常路径"对照臂把 128 传给 `max_arrival_ns`（=128ns 窗口，WindowedTraceReader.hh:136-137 唯一构造重载），CSV 两行 turn-0 到达 1000/2000ns **全部被拒** → 对照臂零提交、两条断言**空真通过**，"arrival gate passes"的正常路径覆盖为虚（静态推证，未实测）。 |
| M13 | `astra-sim/network_frontend/analytical/congestion_aware/main_online.cc:758-764`（2 条） | **face M11 同款**：link id 估算器对 Ring 维一律 `next_id += 2*npus`（注释自称已含 radix==2 mesh fallback，实为假），而后端宽度==2 的 Ring 退化为 mesh 只耗 `npus` 个 id（MultiDimTopology.cpp:195-198 → connect_mesh_dimension :177-188）→ 该维之后**所有维 link id 整体偏移 +npus**，link_bucket/link_total 的 edge 归因静默全错；触发=多维权且含宽度 2 的 Ring 维。 |
| M14 | `extern/network_backend/analytical/congestion_aware/fluid/tests/StaleEventCancellationTest.cpp:129-130`（3 条） | 链路常量按**旧二进制换算**前提构造（`kOneBytePerNsGbps = 1e9/2^30 ≈ 0.9313`，注释宣称"恰为 1 B/ns"），而本仓 `bw_GBps_to_Bpns` 已改 **SI 恒等**（NetworkFunction.cpp:14-18，LOCAL PATCH 注释自证）→ 建链速率 0.9313 B/ns，mouse/elephant 完成 tick 精确断言**必失败**（face H2 同款；severity 校准见下方说明。后端A/后端B/杂项三桶重复登记）。 |
| M15 | `extern/network_backend/analytical/congestion_aware/fluid/tests/FluidSchedulerLinkObserverTest.cpp:256-257`（3 条） | legacy 参照速率 `full_rate = 2^30/1e9 ≈ 1.0737 B/ns`，与同文件 `Link(1.0)/Link(0.5)` 在 SI 恒等换算下的 1.0/0.5 B/ns **不等价** → streaming/legacy 等价性与手算表断言**必失败**（face H2 同源同位；severity 校准见下方说明。三桶重复登记）。 |
| M16 | `extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.cc:170` | PER_NODE_MEMORY_EXPANSION 配置缺 `num-npus-per-node`（构造 :53-61 静默置 0、无正值校验，对比 remote-mem-bw 有 >0 fail-closed）时 `issue()` **整数除零 SIGFPE**；缺 `num-nodes` 时端口容器为空 → `ongoing_transaction[port_index]` **越界 UB**。 |
| M17 | `sh_test_mesh/run_scripts/run_online_strategy_sensing.sh:179` | sensing 版归档列表**落后 strategy 版**（对照 run_online_strategy.sh:184/:192-195，缺 `kv_delta_journal` 与 checksum mv 块）：hbm_watermark 的 journal 权威重放/正式容量判决层（certified）在官方 sensing 路线上**永不触发**，产物滞留 bridge/ 并在下一次同 run_dir 运行时被 `rm -rf` 清掉（脚本归档层断链；与 §1.3 RemoteFifoLedger 的 C++ 记账层断链**相互独立、对象不同**，见 §1.3 末"sensing 断链对照"）。 |
| M18 | `sh_test_mesh/slo_tools/tests/test_slo_contract.py:473` | train_ledger fixture 仍是 2026-09-04 exits 口径修正**前的旧形态**（drains 带内容、exits=[]），而 load_imbalance.py:115-127 仅从 exits 归集 → drains 为空 → collect_intervals 必抛 SloToolError，用例无 assertRaises 包裹**必失败**（静态推证，未实测）。 |
| M19 | `sh_test_mesh/slo_tools/tests/test_driver_parity.py:275` | driver parity 合成 fixture 只写 drains、**完全无 exits 键** → load_imbalance fail-closed（exit 2），新旧链该步骤必失败、`test_happy_path_byte_identical`/`test_g3` 的 rc==0 断言必失败；`test_failure_token_manifest_missing` 的 expect_rc=1 被该无关失败"碰巧"满足，**掩盖真实失败路径**（静态推证，未实测）。 |
| M20 | `astra-sim/workload/execution_driven/tests/cli_online_test.cc:58-61`（1 条 medium 错误登记） | R1–R14 在线 CLI 契约测试（含 B.3/FP1/R13b 回归项）未入任何构建目标或脚本，仅头注 g++ 手工可达；OnlineCli.cc:6-7、main_online.cc:11 注释宣称 "unit-tested in tests/cli_online_test.cc" **失实**——OnlineCli 解析契约的保护网静默失效（另两桶以死代码登记同一问题，见 §2 孤儿测试族）。 |

**low（7 条，6 个独立问题）**

| # | 位置 | 问题（一句话） |
|---|---|---|
| L1 | `astra-sim/workload/execution_driven/ServiceCoordinator.cc:51-61`、`main_online.cc:1563-1564`（2 条） | fence 计数 API（on_fence_scheduled/on_fence_resolved）全仓零调用 → `pending_fence_count_` 恒 0：input-open 死端诊断**永远打印 pending_fence=0**（stall 归因失真），`finished_locked()` 中 `pending_fence_count_==0` 条件恒真（face M12③ 同款；根B/网络前端两桶登记）。 |
| L2 | `astra-sim/workload/execution_driven/tests/node_store_test.cc:534` | README.md:567/:577 把 `..._NodeStoreTest` 列入"无参数直跑"清单，实际无 `--fixture-et` 即打印 `FAIL: --fixture-et is required` 并 return 1；所需 `sh_test_mesh/generated/completion_fixture/fixture.0.et` 是 gitignored 生成物，裸仓必缺——**按 README 指引运行必失败**。 |
| L3 | `astra-sim/workload/execution_driven/tests/windowed_trace_reader_test.cc:879` | Part O 注释称"仅 data_rows 错也 fail-closed"，实际该 sidecar 八个被比字段（fnv1a64/bytes 等）**全部写错**，子进程在首个字段 csv_fnv1a64 即 abort——data_rows 专项校验从未被隔离覆盖。 |
| L4 | `sh_test_mesh/workload/llama2_7b_inference/metrics_schema.py:303` | NodeMetricEvent 对非法 event_code 的报错文本宣称值域 "1-7"，实际 EVENT_EDGE_BY_CODE 已含 `EVENT_FIRST_TOKEN_COMPLETE=8`（合法值域实为 1-8）——判定逻辑正确、拒绝时**诊断文本失实**，误导 fixture 作者。 |
| L5 | `examples/run_scripts/analytical/congestion_aware/HGX-H100-validated.sh:28` | 上游样例脚本引用不存在的 `build/astra_analytical/build.sh` 与已随离线路线删除的旧二进制名 `AstraSim_Analytical_Congestion_Aware`（现役目标带 `_Online` 后缀），`set -e` 下执行**必在编译步失败**（face H3 对位，本仓降 low：仅上游样例脚本、不属官方两条在线路线）。 |
| L6 | `astra-sim/network_frontend/analytical/CMakeLists.txt:354` | 6 个测试可执行目标未设 RUNTIME_OUTPUT_DIRECTORY（二进制落前端默认目录而非 bin/，且全仓无 CMAKE_RUNTIME_OUTPUT_DIRECTORY 兜底），README 明文宣称 MetricOneShotEraseTest 在 bin/ 下"无参数直跑"——**按文档路径执行必找不到文件**。 |

**严重度校准说明（对位 face 的升降级依据）**

- **H2 较 face M1（medium）升 high**：A 桶复核证实缺键路径在**在线主链路真实可达**（Workload.cc:636→generate_all_reduce→Sys.cc:833）且全仓无任何零值校验；face 轮记 medium 的理由是"462 份现存配置均含键、属新配置漏写即踩的雷"——两仓代码面相同，差别只在配置覆盖面与本轮复核对可达性的证词，故按 A 桶 high 收录（B 桶同条记 medium 并存）。
- **H3 较 face M8（medium，face 有 exit 139 实测）升 high**：升级依据是**后果确定性**而非出现频率——一旦手写 doubleBinaryTree 配置且维非 2 幂，`std::map::operator[]` 空指针解引用是确定性段错误（face 轮同源实测 exit 139 佐证该失效模式）；"生产模板全 ring 未启用"仅说明出厂配置不触发，不代表风险低。
- **M14/M15 较 face H2（high，face 复核实测 exit 1）降 medium**：二者是 opt-in 回归 fixture，不在本仓默认构建/ctest 基线内（默认基线本就零用例），失真的是回归保护网而非生产路径；且本仓为静态推证、未复跑实测（见 §3 证据口径），不及 face 轮"实测 exit 1"的证据强度。
- L5 降级理由已在行内注明（仅上游样例脚本，不属官方路线）。

### 1.3 未送复核但值得人工看（可疑点摘录，非实锤）

- `astra-sim/system/Sys.cc:224`：`collective_impl_lookup = new CollectiveImplLookup(id)` 全仓无对应 delete，每 Sys 实例泄漏 lookup 及其注册的全部 CollectiveImpl（A 桶、astraccl 两桶同报）；
- `astra-sim/system/Sys.cc:1175-1182`：All_Gather 且 size < preferred_dataset_splits 时被排除在 total_nodes 兜底外，chunk 整除得 0 且无下限修正（两桶同报）；
- `astra-sim/system/LogGP.cc:72/:107`：size==0 时 `(size-1)` 为负，与无符号 Tick 相加回绕成巨值延迟；
- `astra-sim/system/PacketBundle.cc:43`：uint64_t size 传入 `MemBus::Transmition` 形参 `int bytes`，>2GiB 单条集合通信按截断后字节数计时；
- `astra-sim/system/NetworkStat.hh:40` 与 `Sys.cc:1341-1343`：`/= net_message_counter` 零计数器除法（M5 同款姊妹点；**face §5.1 已修同款，见 §1.4**）；
- `astra-sim/system/CommunicatorGroup.cc:78`：`set_id` 里 `num_streams = id * 1000000`，id≥2148 时 int 溢出（id 来自 torch pgNameInt+1，仅 assert>0）；
- `astra-sim/workload/Workload.cc:260`：`pg_info.substr(2, size-4)` 位于 try 块外，inputs_values 短串（如 "[]"）时无符号下溢 → 未捕获 std::out_of_range；
- `astra-sim/workload/Workload.cc:223-224`：comm_group 文件缺失/非法 JSON 抛未捕获 parse_error → std::terminate 裸崩无诊断（文件名含 "empty" 才被跳过）；
- `astra-sim/system/BasicEventHandlerData.cc:5`：默认构造只初始化 sys_id 不初始化 event，SharedBusStat/StreamStat 经继承携带未初始化成员（当前无人读，潜伏 UB）；
- `astra-sim/system/RecvPacketEventHandlerData.cc:12`：默认构造不初始化 vnet/stream_id/message_end/ready_time，两处裸 `new RecvPacketEventHandlerData` 直接使用；
- `astra-sim/system/QueueLevels.cc:36`：`get_next_queue_at_level(level)` 直接 `levels[level]` 无越界检查，level 来自拓扑维数映射；
- `astra-sim/workload/MetricCollector.cc:505-529`：node_events_by_rank 解析是 load_manifest 唯一无 try/catch 的段，元素类型不符即未捕获异常 terminate；`:511` 负数 node_id 静默回绕为巨大无符号值（与 M9 同段同型缺守卫）；
- `astra-sim/workload/MetricCollector.cc:2053-2064`：WP8 直排积分不夹负、与水位线积分（夹负至 0）口径不一致，仅靠 1% 交叉检查暴露（face 同位置原样）；
- `astra-sim/workload/RemoteFifoLedger.hh:52-56`：头注释宣称 `--sensing-enabled` 接线在本仓不存在——生产零接线（无 set_enabled(true)，仅测试置位），AnalyticalRemoteMemory.cc:192/232 记账点恒被 enabled_ gate 短路（这是 **C++ 记账组件层**的 sensing 断链；脚本归档层的对应断链见 §1.2 M17，两者相互独立，见下方对照）；
- `extern/network_backend/.../NetworkParser.cpp:156` + `Helper.cpp:28`：network.yml 缺 topology 键时 dims_count=0 仍通过全部校验，Release 下 assert 为 NOP 后落入空拓扑；
- `extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.cc:124`：remote_mem_bw 为 uint64_t，小数带宽静默截断（1.9 GB/s→1），>0 校验按截断前 double 判；
- `extern/network_backend/.../fluid/FluidScheduler.cpp:70`：析构只撤 wakeup/observer、不撤仍挂起的 tail 事件，裸 FluidFlow* 上下文悬挂（异常拆卸路径可达）；
- `sh_test_mesh/tests/run_all.sh:9`：头注宣称 "Running all regression tests" 实际只跑 workload 根一个 Python 套件。

> **sensing 断链对照**：正文两处"sensing 没接通"位于**不同层、对象不同**——§1.2 M17 是 sh_test_mesh **脚本归档层**断链（KV delta journal 文件在 sensing 跑照常生成于 bridge/，但 sensing 版归档清单缺它的 mv，hbm_watermark 权威重放层永不触发、产物被清）；本节 RemoteFifoLedger 条是 **C++ 记账组件层**断链（main_online.cc 从未 set_enabled(true)，记账点恒被 gate 短路，记账数据根本不产出）。两者叠加＝本仓 sensing 链在"记账产出"与"归档消费"两端均断，修复时须分别处理、不能互抵。

### 1.4 face 修复轮已证实的同源雷（高优待核，不计入本轮 41/82 统计）

face 文档 §5.1 已当真实缺陷修复（§5.3 登记残留）的同源点，在本仓的现状如下——排修复计划时**不得**因本轮未实锤而跳过：

| 位置 | face 轮处置 | 本仓现状 |
|---|---|---|
| `astra-sim/system/NetworkStat.hh:40`（姊妹点 Sys.cc:1341-1343） | face §5.1 已修"零计数器除法守卫"（M6 补漏同批） | 本轮仅落 §1.3 可疑点、未送复核；代码同源存在 |
| `astra-sim/system/LogGP.cc:72` | face §5.1 已修 size==0 守卫（face 实证过回绕危害） | 同上，§1.3 可疑点 |
| `astra-sim/system/CommunicatorGroup.cc` 整集群主路径 | face §5.1 修复时补"整集群分支空实现向量 fail-closed（防 Sys.cc:839 越界 UB）" | **本仓全文无登记**——修 H1 时应一并核查本仓主路径是否存在同型越界 |
| `astra-sim/workload/MetricCollector`（clear_static_node_events） | face §5.3 登记"已知残留风险（未修）：静态事件在在线生产路径被 clear_static_node_events() 清除" | 本仓同样存在：本轮材料可疑点（MetricCollector.cc:500）证据自述"静态表随后被 clear_static_node_events 清空"，错挂影响仅限直接消费方测试 |

## 2. 确认的死代码（82 条，按族归类）

82 条按"桶×条"计，**去重后 70 个独立问题**（跨桶重复登记：AstraSimDataAPI 三桶、AstraComputeAPI 两桶、OnlineCli 两桶、OnlineStatisticsState 两桶、retire_online_operator / slo_watermark_period_ns / was_json_id_committed / FileDecisionBridge::stats() 各两桶、孤儿测试五次、boost-mode 两桶）；下表按去重后的问题归为 6 族。

| 族（条数） | 代表实例（均已独立复核：零引用/恒假/不可达） |
|---|---|
| 整类/整文件零引用（13 条） | `AstraSimDataAPI`/`LayerData` **双副本**（system 与 common 逐字节相同、同 `__ASTRA_SIM_DATA_API_HH__` guard，三桶登记）；`MemEventHandlerData` 整类零实例化，且 Sys.cc:691-699 分发分支依赖的 CompFinished/MemLoadFinished/MemStoreFinished 三事件值全仓零引用——**分支不可达**；`CSVWriter` 整类（唯一"消费者" UsageTracker::report 以 nullptr 调用且解引用前必 throw）；`Torus3D`、`LocalRingGlobalBinaryTree`、`LocalRingNodeA2AGlobalDBT`（三者经根 CMakeLists.txt:67/70 的 `*.cc` GLOB **仍编入 AstraSim 库**——零引用 ≠ 不编译，face P12 同款；后者 dim==2 分支接回即返回 nullptr，**死代码里藏错**，同 face）；`AstraComputeAPI`/`ComputeKernel`+4 个 create_llm_* 工厂（get_static_runtime 还返回未初始化聚合 timespec_t，接回即 UB，2 条）；MetricCollector Phase-0 `PerformanceCounters` 整套框架（counter 永远全 0）；`metrics_integration.py` legacy 静态 ET 兼容块（约 370 行、13 个符号闭环自消费，终点 ServiceMetrics.write_manifest 已死，`__all__` 同批导出）；FluidScheduler Phase-7 拥塞快照四件套（link_congestion_snapshot/link_state_epoch/link_count/LinkCongestionSnapshot，头注释自认 "the wscllm strategy never consumes it"；**face §2 同条注：joint 仓 `link_count()` 有真实消费——跨仓删除须逐仓核对并登记豁免，机械回扫 face §5.2 删除清单时防误删**） |
| 钉死分支/恒真恒假（14 条） | （M3 对位）`intra/inter_dimension_scheduling` 钉死 FIFO/Ascending → RoundRobin/OnlineGreedy/OfflineGreedy 全死分支 + offline_greedy 恒 nullptr；`HardwareResource.cc:123-125` else 内恒假 `==0`（外层已保证非 0）；`Workload.cc:399-403` `if(true)`（else throw 臂不可达）；`GraphBatchCommitter.cc:456-464` 与 `:1015-1027` comm.tag（uint32_t）值域/负值检查恒假（2 条）；`OnlineCli.cc:233-238` uint64 上限检查恒假（unsigned long long 与自身类型上限比较；ERANGE 已在 :220 拦截真溢出，2 条）；`LocalMemUsageTracker.cc:251-255`（及读回侧 :385）size_t 与 uint64_t 同宽的上限检查恒假；`Sys.cc:226-229` `initialize_sys(...)==false` panic 分支不可达（唯一 return true 在 Sys.cc:498，文件打不开已 exit(1)）；`InputCloseReason::Error` 无任何生产者（Error 命令在 drain 中直接 abort）→ main_online.cc:1773 case 不可达；`event_queue_deferred_test.cc:423` `std::_Exit(reached_end ? 0 : 0)` 死三元式（注释宣称"完成才退 0"与代码不符）；`windowed_trace_reader_test.cc:646` `expect(true, "fixture written")` 恒真断言（silence-warning 的注释理由也不成立，csv 随后被真实消费）；`CollectiveImpl.hh:23` HierarchicalRing/LocalRingNodeA2AGlobalDBT 死枚举值（生成端与分派端均不出现）；`BypassRule::BYPASS_PERNODE_CUSTOM`（头注释自认 "No current usecase"，:203-204 比较分支不可达）；`Common.hh:54` `CollectiveBarrier` 双副本死枚举 |
| 只写不读字段（13 条） | `BaseStream.hh:53-55` `test/test2/phase_latencies[10]` 调试残片（写 0/零引用）；`MemMovRequest::latency`（唯一构造点 LogGP.cc:78 第 5 参恒传 0；且出厂模板无 model-shared-bus 键 → LogGP/MemMovRequest/MemBus 共享总线链在所有出厂配置下**条件性死链**）；`RecvPacketEventHandlerData::message_end`；`Algorithm::name`/基类 `id`（四个算法构造写入后无读者；三个子类自有 id 遮蔽基类副本，CustomAlgorithm 持有两个 id）；`HalvingDoubling::routing`；`OperatorStatistics::network_bandwidth`（唯一"读者"是 Statistics.cc:839-862 **注释掉的报告块**；GraphSource.hh:120 副本同款）；`OnlineStatisticsState::operation_intensity/is_memory_bound`（2 条，紧凑完成路径只消费 utilization 两字段）；`rank_instance_conflicts`（累加后 `(void)` 丢弃，:2338-2340 注释却宣称 "conflicts are counted, never silently resolved"）；`OnlineDriverContext` ingress/systems/expected_requests 三字段（expected_requests 快照点恒 0，循环内才填真值）；`WindowedTraceReader::consumed_idx_` 与 `LegacyOracleWindowedTraceReader::last_file_pos_/consumed_idx_`（2 条，仅自比较推进、无外溢读者）；`BasicTopology::bandwidth/latency` + `Link::bandwidth`（构造赋值后无任何读取，派生类构造形参遮蔽） |
| 死函数/死访问器（31 条） | `Sys.hh:138` `get_collective_implementation`（死声明：无定义无调用）；`SchedulerUnit::get_average_latency_per_dimension`（且 resize(-1) 兜底为死值、total_chunks=0 时 0/0 NaN）；`Roofline` 单参构造（零调用，且不初始化 bandwidth——接回即读未初始化 double）；`Statistics::retire_online_operator`（hh:255 注释指认的填充路径已失实，真实路径是 complete_online_service_operator，2 条）；`on_local_hbm_restore_issue`（hh 注释自认 "no caller...interface parity"，连带 local_hbm_restore_bytes 恒 0，face P13 同款）；`slo_watermark_period_ns()`（同组访问器唯独它无 main_online 消费，2 条）；`was_json_id_committed`（功能已由 resolve_store_id 组合覆盖，2 条）；`FileDecisionBridge::stats()`（main_online.cc:190 注释仍称经 stats() 取数，实际走 stats_report()，2 条）；`OnlineStatsCounters::reset()`（hh:38 "for fixture reuse" 失实）；`RequestIngress::scheduled_future_arrival_count()`（底层数据活跃，仅访问器死）；`CmdLineParser::get_options()`；FluidScheduler 三访问器 `flush_pending_starts_deferred`（头注释 "must go through this entry" 失实）/`get_completion_heap_size`/`link_observer_enabled`（启用门在前端，3 条）；`AnalyticalRemoteMemory::architecture_name/port_mapping_rule`（hh:50-52 注释宣称的消费方不存在）；`Event::get_handler_arg`；`BasicTopology::get_basic_topology_type`（连带 protected basic_topology_type 沦为只写）；`Topology/Device::get_links_count`（整条计数链不可达）；`BinaryTree::print`；`RingTopology::is_enabled`（**藏错**：assert(offset>0) 对向量 ctor 的 offset=-1 接回即炸）；`HardwareResource::report()`（连带 num_cpu_ops/num_gpu_ops/num_gpu_comms 与 tics_* 六字段只写）；Python 侧：`generate_trace.py` `shard_size`+`InferenceGroup`、`KV_CACHE_EVENT_COLUMNS`（16 列常量零引用）、`ledger_reconcile._seq_files`、`graph_batch_builder.node_count_total`、`SessionKVSnapshot.context_tokens`（docstring 指认的消费面不存在）、`_emit_prefill_stage` token_expanded 分支（唯一调用链入口 emit_prefill_batch 已对 granularity 硬 raise，恒不可达） |
| 孤儿测试（5 条，3 个文件） | `cli_online_test.cc`（R1–R14 契约）/`event_queue_deferred_test.cc`（EventQueue deferred 通道唯一回归）/`ingress_idle_fixture.cc`（IDLE 五态+FP1 边界 9 场景）均未入任何 CMake 目标，仅头注 g++ 手工可达——同目录其余约 20 个兄弟测试均登记于 analytical CMakeLists；对位 face M12②（其头注悬空指向的 wscllm 仓根即本仓）。根A/测试A/测试B×2/杂项五次登记；根B 另以 medium 错误登记同一问题（M20） |
| 死配置/死数据（6 条） | `boost-mode`（10 个 inputs json + 2 个 C++ 测试 fixture 写入、全仓 C++ 零读取，静默失效，2 条）；`--compute-scale` CLI 选项定义后全仓零读取（传入被静默忽略）；`slo_params_manifest.json` 21 个冻结参数中 11 键无脚本消费者（face §5.3 口径：冻结留证保留，仅登记）；`inputs/` 整目录 10 个上游样例 system json 零引用（与"裸仓两条在线路线"声明矛盾）；`sh_test_mesh/slo_tools/` 下 5 个 `.bak_caliberfix_20260905` 残片（face 姊妹仓 §5.2 已清、本仓未清） |

---

## 附录 A：本轮新提炼的错误模式（face P1–P14 之外，供修复排期与姊妹仓回扫）

face 轮 13 条模式中 **12 条在本轮再次命中**（P1 缺键 UB/除零、P2 钉死枚举、P3 所有权别名、P4 计数器 0 除、P5 窄化截断、P6 单位换算、P7 估算 id 失配、P8 非常规形状崩溃、P9 泄漏中止、P10 孤儿测试、P12 GLOB 编入死文件——本仓实例见 §2 整类族 Torus3D 等"仍编入库"注、P13 parity 占位）；**P11（resolver 透传死键）未复现**：本仓 config_resolver 派生链未发现"写入即死"的键（模板侧 boost-mode 系出厂模板/上游样例 json 透传、非 resolver 写入，已按死配置登记 §2）。以下为本轮新增：

| # | 模式 | 本仓实例 | 排查动作 |
|---|---|---|---|
| W1 | 测试对照臂自身配错 → 断言空真 | M11（对照臂默认 128 有界、宣称无界）、M12（对照臂 max_arrival=128ns 拒掉全部样本后断言空真通过）、L3（sidecar 全字段写错致专项校验从未被隔离覆盖） | 先独立核对对照臂构造参数确能产出"应通过"的样本，再相信断言通过；对照臂数据逐字段最小化构造 |
| W2 | 成对脚本单边演进 | M17（sensing 版归档列表落后 strategy 版，缺 kv_delta_journal 归档与 checksum mv） | 成对脚本 diff 清单类差异（归档/清理/环境变量）；改一处必查另一处 |
| W3 | 消费端 schema 演进未回扫测试 fixture/文档 | M18/M19（train_ledger 增加 exits 键后旧 fixture 未同步，负载性失败且污染 rc 断言）、L2（README 直跑清单与测试合同矛盾、依赖 gitignored 生成物） | 消费端加字段时全仓 grep fixture 构造点同步更新；README"直跑"清单纳入冒烟验证 |
| W4 | 零调用 API 顶着失实注释（face P13 扩展） | "unit-tested in tests/cli_online_test.cc"（OnlineCli.cc:6-7）、"must go through this entry"（FluidScheduler.h:77-83）、"No current usecase"（CollectiveImplLookup.hh:18）、"conflicts are counted"（MetricCollector.cc:2338-2340）、"single source of truth"（AnalyticalRemoteMemory.hh:50-52）、"for fixture reuse"（OnlineStatsCounters.hh:38）、"record_* are no-ops until enabled"（RemoteFifoLedger.hh:52-56） | 注释指认的消费方/测试逐个点名验证存在性；凡"自述用途"的注释都当作待验命题 |

## 3. 覆盖与可信度

- **覆盖（16 桶清单）**：13 个 C++ 桶——system核心A、system核心B、astraccl集合通信、workload执行与资源、workload统计与指标、在线图执行根A、在线图执行根B、在线图测试A、在线图测试B、在线图测试C、网络前端与公共接口、外部网络后端A、外部网络后端B（229 文件逐行读）；3 个脚本/杂项桶——仿真脚本运行链、负载物化脚本、杂项与构建编排（92 文件）。chakra（extern/graph_frontend）与 face 仓 `diff -rq` 为空（逐字节一致），按 face 轮口径豁免深挖（上游甄别）；`extern/helper` 三方库（spdlog/yaml-cpp/fmt/json/cxxopts）未审；运行期行为/长跑稳定性未实测。**防遗漏预验证缺口**：各桶覆盖声明自述"全部逐行读完、无一跳过"，但本轮材料未见 face 轮式"glob 预验证无遗漏无空桶"的全量对账记录——321 是否等于仓内全部待审文件未经独立预验证，列为口径限制。
- **基线**：`cmake` Release 构建通过（exit 0）；`ctest` 退出码 0 但系**零用例空跑**（根 CMakeLists.txt:117 唯一 CTest 注册项默认 OFF，默认基线不含任何 C++ 单测），exit 0 不构成回归证据——工作流基线记录，本文档撰写未重跑。孤儿测试三件套与 opt-in fluid fixture 同样不在该（空）基线内。
- **复核覆盖**：候选总量 **224 条**＝送独立复核 **125 条**（确认 123＋驳回裁定 2，确认率 123/125）＋未送复核可疑点 **99 条**（实质 98，另 1 条为跨桶对位提示）；§1.3 仅摘录 18 条，全量留档发现看板。候选与可疑点总量系本文档按材料一逐桶清点（材料四仅给定 41/82/2 三个确认/驳回数）。
- **计数口径**：确认 123 条（错误 41＝high 4/medium 30/low 7；死代码 82）按"桶×条"计——同一问题多桶独立登记分别计数；去重后**错误 29 个独立问题**（Sys.cc:1176 一题由 A/B 两桶分记 high/medium 各 1 条）、**死代码 70 个独立问题**（重复登记清单见 §2 引言）。
- **证据口径**：本轮 123 条确认全部基于静态逐行读码＋独立复核亲读，**无本仓运行实测**；文中"必失败/必误报"级断言（M11/M12/M14/M15/M18/M19 等）系静态推证——face 同源项的实测结果（exit 1/exit 139、legacy=43 vs calendar=200）仅在 face 仓成立，修复验收时应在本仓复跑取证。
- **驳回与双裁（复核"驳回"裁定 2 个：净驳回不进正文 1 条，双裁并存 1 条）**：
  1. **净驳回（不进正文）**：`astra-sim/system/MyPacket.hh:24`（A 桶提交）——"四处构造均 3 参（无 msg_size）"的核心断言被亲读**证伪**：HalvingDoubling.cc:204-206/216-218 实为 4 参（含 msg_size）构造，且 msg_size 在 HalvingDoubling.cc:256/266 被 front_end_sim_send/recv 真实读取，整条驳回。其子项（sender 只写、fm_id 零引用、`MyPacket::call`+notifier 回调链等）未单独送复核，**不得实锤化**——注意 face 对位项"MyPacket::call+notifier 死代码"在本仓未获确认，清理前须重查。
  2. **双裁并存（按确认收录 §1.2 M2，非净驳回）**：`astra-sim/system/Sys.hh:380`——A 桶复核认定缺陷属实但触发不可达（现存 14 个 system JSON 中唯一缺 `collective-optimization` 键的 examples/system/custom_collectives/custom_collective.json 走自定义实现路径）；同问题 B 桶复核确认（UB 证据链完整）并按 medium 收录。两裁并存的处理：UB 链为真、现存配置覆盖了触发面，按"缺键即雷"保留，补类内默认值即可两全。
  - 教训同 face P14：零引用/恒假结论必须由第二人独立全形式 grep（宏/字符串/测试/CMake/二进制）后方可实锤。
- **完整证据**：本轮工作流"深挖审查全量证据表"（逐条 path:line + 复核意见）与发现看板（含全部未送复核可疑点）；本文档仅录重点。

---

## 5. 修复状态（2026-09-24，修复工作流后）

> 结构对位 face 文档 §5。十个修复组（Sys核心与调度配置 / 通信域与集合通信算法 / 流与数据包 / workload统计与指标 / 在线执行层 / 网络前端 / 测试与构建 / 外部网络与远存 / 运行脚本与杂项 / 物化脚本）合计**处置 103 条、跳过 17 条**（103 = 14+11+12+13+10+7+11+10+8+7，逐组摘要对账吻合）。**未做任何 git commit**（修复只落工作区）；除本节记录的文档同步三文件外无其它文档改动。

**验收基线**（修复工作流门禁报告，本文档同步批次未重跑构建/测试/仿真）：`cmake` Release 构建、`ctest`、测试直跑（bin/ 下测试可执行）、根测试（仓根 tests/）、Python 套件（sh_test_mesh/ 各 pytest）**全部通过**；**两秒冒烟（物化→plan→官方 runner）实际通过**——物化 22 行队列（21 请求/5 会话，含 canonical sidecar + provenance 三件产出）→ plan_materializer 产 plan 目录 → 官方 runner `[run_online_strategy] cpp_exit=0 python_exit=0` + `PASS: .../fix_smoke_run`，指标后处理与归档全链完成（`/tmp/wscllm-fix-smoke.log` 留档）。工作流曾误报 `NO_CPP_LOG`：系其收尾判据笔误——runner 归档阶段把 `cpp.log` 压缩为 `cpp.log.gz` 并移除原文件，判定脚本在归档后检查原文件存在性所致（2026-09-24 主控复核日志更正记录）。下文标注"亲读"者为本 § 撰写时对现行代码/`git diff` 的逐点核验，其余为修复组摘要口径。

### 5.1 已修复（错误全量：41 条 / 29 个独立问题）

- **high 3**：H1 所有权改为**保型克隆**——`clone_collective_impl`（`CommunicatorGroup.cc:22`，消费点 `:181`），注册表共享的 `CollectiveImpl*` 不再交给带删除语义的 plan（亲读；`CollectiveImpl` 非多态不用基类拷贝，避免切片，对位 face §5.1 口径）；H2 缺 `preferred-dataset-splits` 构造缺省 1（`Sys.cc:198`）+ `determine_chunk_size` 对 ≤0 值回退整段单 chunk（`Sys.cc:1187-1192`，除零 SIGFPE 封死，亲读）；H3 `BinaryTree` 构造对非 2 幂节点数 fail-closed `critical`+`exit(1)`（`BinaryTree.cc` ctor 亲读），连带死函数 `BinaryTree::print` 删除（diff 亲读）。
- **medium 20**：M1 `scheduling_policy` 类内缺省 `FIFO`（`Sys.hh:342`）；M2 `collectiveOptimization` 类内缺省 `Baseline`（`Sys.hh:375`）——两处消缺键 UB（亲读）；M3 `inter/intra-dimension-scheduling` 全量解析补齐（`Sys.cc:355-385`，未知值 `sys_panic`，构造缺省 Ascending/FIFO 于 `Sys.cc:192-193`，亲读），OfflineGreedy 分支随之复活；M4 `PacketBundle.cc:55` 对 `local_mem_bw<=0` 回退 legacy 公式（README G 节"local-mem-bw<=0 自动回退"承诺恢复成立，亲读）；M5 `SharedBusStat.hh:137/:144` 零计数器守卫（亲读）；M6 SimSend/SimRecvCaller 改 `uint64_t` 存消息大小（`SimSendCaller.hh:18` 亲读）；M7 `CustomAlgorithm` 补虚析构释放 ETFeeder（`CustomAlgorithm.hh:39` 亲读）；M8 roofline 非有限值 fail-closed（`Workload.cc:546` 段亲读）；M9 node event 校验改**宽域 int64 先校验后窄化**、负 node_id 同批拦截（`MetricCollector.cc:505-529` 段亲读）；M10 `WindowedTraceReader` CSV 行解析加固（组摘要；文件在本批修改清单）；M11 oracle legacy 对照臂改显式 `high_water=0` 真无界（`calendar_reader_oracle_test.cc` diff 亲读）；M12 Part Q 对照臂改无界 `max_arrival_ns`（`windowed_trace_reader_test.cc` diff 亲读）；M13 link id 估算器对宽度 2 的 Ring 维改按 `npus` 计（`main_online.cc` diff 亲读）；M14/M15 fluid 测试常量改 SI 恒等（`kOneBytePerNsGbps = 1.0`、`full_rate = 1.0L`，两测试文件亲读）；M16 `AnalyticalRemoteMemory` 对 `num-npus-per-node` 补正值 fail-closed（`AnalyticalRemoteMemory.cc:59-71` 亲读）；M17 sensing 版归档补齐 `kv_delta_journal` + checksum 搬运（`run_online_strategy_sensing.sh:179/:185-189` 亲读，与 strategy 版对齐）；M18/M19 train_ledger exits 口径 fixture 修复（`test_slo_contract.py`/`test_driver_parity.py` 本批修改）；M20 三孤儿测试接入前端 CMake（`AstraSim_Analytical_Congestion_Aware_{CliOnlineTest,EventQueueDeferredTest,IngressIdleTest}`，CMakeLists.txt diff 亲读；`OnlineCli.cc:6-7` 注释随之属实）。
- **low 6**：L1 fence 计数 API（`on_fence_scheduled`/`on_fence_resolved`/`pending_fence_count`）整支删除（全仓 grep 零残留，亲读）；L2 README §4 补 NodeStoreTest 例外段（`--fixture-et` 用法与 fail 行为，README diff +9/-2 亲读）；L3 Part O sidecar 对照臂重建、data_rows 专项校验恢复隔离覆盖（组摘要）；L4 event_code 报错文本 1-7→1-8（`metrics_schema.py:303` diff 亲读）；L5 3 个失效 example 脚本删除（`examples/run_scripts/analytical/congestion_aware/`，git status 亲读）；L6 6 个测试可执行目标补 `RUNTIME_OUTPUT_DIRECTORY` 归位 bin/（前端 CMakeLists.txt +111/-0 diff 亲读）。
- **§1.4 face 同源雷 4 条全部落实**（修 H1/M5 时一并核查处置）：`NetworkStat.hh:39` 零计数器除法守卫（亲读）；`LogGP.cc` 两处 size==0 回绕守卫（diff 亲读）；`~Sys` 补 `delete collective_impl_lookup`（`Sys.cc:314` 亲读）；`CommunicatorGroup.cc:145/:159` 空/缺失实现向量 fail-closed（亲读，防 `Sys.cc:839` 型越界 UB）。
- 连带修正：`Sys.cc:224` 可疑点（lookup 泄漏）经 `~Sys` delete 落实；`Sys.cc:1175-1182` All_Gather < splits 兜底同段处置（组摘要）；B 桶对 H2/M2 的同条登记随主修一并闭环。

### 5.2 已清除（死代码 82 条 / 70 个独立问题中的实删部分）

- **删除整文件/整目录**（git status 亲读）：`Torus3D`、`LocalRingGlobalBinaryTree`、`LocalRingNodeA2AGlobalDBT`（.cc/.hh）；`common/AstraComputeAPI.hh/.cc`（任务清单所写 system/ 路径在本仓不存在，`find` 证实仅 common/ 一份，按实删）；`AstraSimDataAPI.hh` system/+common/ 双副本；`inputs/` 整目录 10 个上游样例 system json；congestion_aware 3 个离线示例脚本；5 个 `.bak_caliberfix_20260905` 残片（slo_tools/ 下 README.md/slo_common.py/slo_params_manifest.json/slo_stats.py/test_slo_contract.py）。
- **删除死分支/死枚举值**：system/Common.hh EventType 死值 `CompFinished/MemLoadFinished/MemStoreFinished`（连带 Sys 分发不可达分支）；common/Common.hh 的 `CollectiveBarrier` 副本（diff 亲读）；`CollectiveImpl.hh` 的 `HierarchicalRing`/`LocalRingNodeA2AGlobalDBT` 死枚举值与 `BypassRule::BYPASS_PERNODE_CUSTOM`（全仓 grep 零残留，亲读）；GraphBatchCommitter comm.tag 恒假检查、OnlineCli uint64 恒假上限检查、LocalMemUsageTracker 同宽恒假上限、`Sys::initialize_sys` 不可达 panic 分支、Workload `if(true)`、HardwareResource 恒假 else、main_online 不可达 Error case 等（组摘要口径，文件均在本批修改清单）。
- **删除只写不读字段/死函数/死访问器**（含亲读 diff 证实项）：`MemMovRequest::latency`（LogGP 构造去参）；`LegacyOracleWindowedTraceReader::consumed_idx_/last_file_pos_`；`CmdLineParser::get_options()` 与 `--compute-scale` 死选项（前端全形式 grep 零残留，亲读）；fence 计数 API；`BinaryTree::print`；以及 `get_collective_implementation`、`get_average_latency_per_dimension`、Roofline 单参构造、`retire_online_operator`、`on_local_hbm_restore_issue`、`slo_watermark_period_ns`、`was_json_id_committed`、`FileDecisionBridge::stats()`、`OnlineStatsCounters::reset()`、`RequestIngress::scheduled_future_arrival_count`、FluidScheduler 三访问器与 Phase-7 拥塞快照四件套、`AnalyticalRemoteMemory::architecture_name/port_mapping_rule`、`Event::get_handler_arg`、`BasicTopology::get_basic_topology_type`、`get_links_count`、`RingTopology::is_enabled`、`HardwareResource::report()` 连带六只写字段、`Algorithm::name`/基类 `id`、`HalvingDoubling::routing`、`network_bandwidth` 注释读者链、`rank_instance_conflicts`、OnlineDriverContext 三字段、BasicTopology `bandwidth/latency`+`Link::bandwidth`、BaseStream 调试残片、`RecvPacketEventHandlerData::message_end` 等（组摘要口径）。
- **Python 侧**：`metrics_integration.py` legacy 静态 ET 兼容块整块删除（numstat +10/-402，867→475 行；`METRICS_DETAIL`/`ENABLE_METRICS`/`resolve_metrics_detail` grep 零残留，亲读）；`generate_trace.py` `shard_size`+`InferenceGroup`、`KV_CACHE_EVENT_COLUMNS`、`ledger_reconcile._seq_files`、`graph_batch_builder.node_count_total`、`session_kv_manager` `SessionKVSnapshot.context_tokens`、`_emit_prefill_stage` token_expanded 死分支——物化脚本组 7 文件，删除前对全部待删符号做全工作区全形式 grep（组摘要）。
- **死配置/死选项**：system 模板 `boost-mode` 删除（模板 diff 亲读）；`alarm_cancellation_test.cc`/`remote_fifo_ledger_test.cc` 的 `"boost-mode": 0` fixture 写入行删除（diff 亲读；运行脚本组协调注记原列"范围内不动"，实际已随两测试文件的本批修改一并删除，如实改记）；仓内 `boost-mode` 仅剩上游遗留 `examples/system/native_collectives/` 两处样例 JSON（范围外保留，零读取点）。
- **测试接入**：三孤儿测试入前端 CMake 产出 bin/ 直跑二进制（**未注册 add_test**——M20 的 add_test 子项按组摘要跳过，ctest 默认用例集不含这三个测试）；6 个既有测试目标输出目录归位 bin/。根 `CMakeLists.txt:116-127` 的 OfflineGreedy journal option 保持缺省 OFF 未动（亲读，按协调注记保留）。

### 5.3 评估后保留（各组 17 条跳过的留存项 + 亲读观察，姊妹仓套用同口径）

> 对账：17 条跳过 = 通信域 2 + 流与数据包 4 + workload统计 4 + 在线执行层 2 + 测试与构建 1 + 运行脚本与杂项 4。其中 2 条实际已处置并记入 §5.2（AstraComputeAPI 按实存路径删除；boost-mode 两处 C++ fixture 写入已随测试文件修改删除），1 条系组自注"非跳过、属对齐口径说明"（finish_generic_node event 形参），其余 14 条为真留存；另有修复组注记的保留项（scheduling/ 目录、根 CMake option）与本文档亲读观察（CollectiveBarrier system 副本）一并列入。

1. `CollectiveImpl` 虚析构：**不加**——face 现行实现无虚析构（组亲读）。
2. `CollectivePlan.hh/.cc`：**不动**——与 face 现行文件逐字节一致，H1 修复全部落 `CommunicatorGroup.cc`（组亲读）。
3. `CSVWriter.cc/.hh` + `UsageTracker.hh:13/:28`、`UsageTracker.cc:59`：**保留**——`report(CSVWriter*,int)` 唯一消费者前提在本组范围不成立（组亲读）。
4. `MemEventHandlerData.cc/.hh`：**保留**——"Sys 侧分支由 Sys核心组删"的协调前提经亲读不成立（组亲读）；现状：类文件仍在、`Sys.cc` 已无引用（本文档 grep 亲读）。
5. `MemBus.hh:34-43`+`MemBus.cc:42-88` int bytes 同族截断：**保留**——face 未修、两仓逐字节相同（§1.3 顺带核查项，组亲读）。
6. `HardwareResource.hh:80` `tics_gpu_ops`：**保留**——"只写"前提对该字段不成立（`Workload.cc:1124` 有读者，组亲读）。
7. `Workload.cc finish_generic_node` 的 event 形参 + `HbmCommJoin::completion_event` 链：**保留**——p2p 带宽块删除后该形参 body 不再读取，属对齐口径说明非死码（组亲读）。
8. `RemoteFifoLedger.hh/.cc` + 测试：**保留整链**（ask 第 4 条指示不动；sensing 记账断链系 §1.3 既定口径）。
9. `MetricCollector` `clear_static_node_events`：**保留**（ask 第 4 条指示不动；与 face §5.3"已知残留风险"同款登记）。
10. `OnlineCli.cc:6-7` 注释：**保留**——M20 接线后 "unit-tested in tests/cli_online_test.cc" 已属实（本文档亲读）；`main_online.cc:10-11` 同句"提醒半"属网络前端组范围外。
11. `slo_params_manifest.json` 21 个冻结参数（含 11 个无脚本消费者键）：**冻结留证保留**（face §5.3 口径，与本清单 §10.4 同步）。
12. `tests/run_all.sh:9` 头注：**不修**——按未实锤口径保留（组核实补记：文档所写路径与实际位置有出入，未达实锤不改）。
13. `examples/system/native_collectives/HGX-H100-validated.json:20`、`Ring_4chunks.json:20` 的 `boost-mode` 死键：**范围外保留**（零读取点，上游遗留样例）。
14. `system/scheduling/` 目录 + 根 CMake OfflineGreedy journal option：**保留**——M3 解析补齐使 OfflineGreedy 分支复活，非死码（组亲读 + 本文档亲读）。
15. `CollectiveBarrier` 的 system/Common.hh:54 副本：**已删**（common 副本由『流与数据包』组先行删除；system 副本系两组协调缝隙残留，2026-09-24 主控按 P14 纪律独立全形式 grep 复核全仓唯一命中即声明本身后删除，`g++ -fsyntax-only` 头文件核验通过；该枚举双副本至此清零）。
16. §1.3 未送复核可疑点中未上表者：按"未送复核非实锤"原口径未动（MyPacket 族特别提示见 §3 驳回 1）。
17. **残留注记（已全部清除）**：本 § 撰写时（工作流收尾 clean 脚本执行前）曾登记——fixsmoke 三件物化文件、`trace_config.csv` 指针改指、gitignored 的 `generated/` plan 目录与 `build/` 构建树等冒烟/门禁过程产物。其后工作流收尾段已执行 `clean_test_records.sh` + `clean_build_artifacts.sh`，2026-09-24 主控复核**裸仓四要素全部通过**（`trace_config` 指针=占位 ✓、`traces/` 仅 *.py ✓、无 `generated/` ✓、无 `build/` ✓），本条所列残留均已不存在。全仓 `.bak/.orig/.rej/.tmp/*draft*/*.swp` 扫描（排除 build/）零命中。

### 5.4 文档同步

- **本仓 README.md**：§4 直跑清单补记三孤儿测试接入与 6 目标 bin/ 归位（NodeStoreTest 例外段已由测试与构建组先行落盘）；G 节"local-mem-bw <= 0 自动回退"承诺经 M4 修复恢复成立（`PacketBundle.cc:55-66` 亲读），无矛盾段；README 无 `boost-mode`/`--compute-scale` 表述，无需删改。
- **`experiment/仿真各功能开关清单.md`**：§10.3 `inter/intra-dimension-scheduling` 行登记 W 为可解析仓（与 F 同款取值域/缺省/panic，亲读 `Sys.cc:355-385`）；§10.3 `scheduling-policy`/`collective-optimization`/`preferred-dataset-splits` 三行补 W 缺省兜底口径；§3 `--compute-scale` 行、§3 `inputs/` 示例注、§5.2 遗留指标层注、§12 测试入口行、§13 四条退役行、§13.1 `boost-mode`/`--compute-scale` 两行、§14 差异表 W 列——均按本批 W 仓实际删改同步（详见该清单对应行"2026-09-24 修复批"注记）。
