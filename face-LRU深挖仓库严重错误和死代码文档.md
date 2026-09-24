# face-LRU 仓深挖：严重错误与死代码

> **审查对象**：`template/astra-sim-face-LRU`（face 姊妹仓的 LRU 三态 KV / 两段逐出改造仓，含改造工单 B1 条目 A.2 恢复接线的远端内存后端、HBM 带宽模型与 sh_test_mesh SLO 工具链）。
> **方法**：15 个模块、清单 320 文件分桶全覆盖深挖（另含 1 件任务点名补审的清单外文件 `astra-sim/common/AstraRemoteMemoryAPI.hh`，逐行精读合计 321 件；**未分桶残留 0**）。每条"严重错误/死代码候选"由独立复核员全新上下文裁定（只认亲眼读到的代码）；死代码判定执行 P14 纪律（全形式 grep：含 tests/、CMake、.py/.sh/.json/.md、宏/字符串/取地址形态）。**全程只读纪律：未构建、未跑测试、未跑仿真**——构建基线链异常中止（`WorkflowError: world.run 'bash' timed out after 900000ms`），全部结论为静态读码 + grep，崩溃/必失败均为证据链闭合的静态推演。
> **结果**：**确认 94 条**（high 3 / medium 22 / low 69；按口径 = **严重错误 35 条**（high 3 + medium 22 + low 10）+ **死代码 59 族**）、**驳回 4 条**误报；另有 **154 条可疑点登记**存档于 §5（**146 条未定论 + 7 条已确认结论的跨桶指引 + 1 条已澄清勘误**，均不计入确认数），1 条 P14 排除记录留档于 §4。
> **用途**：§1/§2 是本仓实锤结论；§3 是与孪生基线 face 仓 86 条错误的逐条继承核对——**face 已于 2026-09-23 完成两轮修复（见 face 深挖文档 §5），本仓基线未同步该修复，多处位点呈"face 已修、本仓回归"形态**；§5 留档全部可疑点供后续排查。
> **口径**：路径均相对仓根 `template/astra-sim-face-LRU/`；跨桶重复确认的同一缺陷（9 组，如 H1/H2/H3、face M5、孤儿测试、fence 族、P11 死键、AstraSimDataAPI 双副本等）按唯一缺陷归并为一条并标注"多桶互证"；本清单只找"错"与"死"，不评架构。

**代号与编号索引**（单凭本文可还原指代；P/H/M 系编号源自 face 深挖文档 §3 的错误模式清单与 §1 条目，本仓 §1 的 H/M/L 编号为本仓独立编号，两套编号互不连续、以"face X"前缀区分归属）：

- **B1 / A.2 / B2 / B3**：本仓 -LRU 改造的工单批次与条目代号（B1·A.2=恢复远端内存后端接线；B2/B3=KV 三态化与逐出策略改造，其残留死码见 §2-40/41）。
- **P0**：早于 -LRU 的上一轮改造批次代号（"P0 改造前"指该批之前的接口形态，见 §1-M16）。
- **path-2**：本仓改造中移除 replay 路线的批次代号（见 §5.6）。
- **sh_2.0 谱系**：本仓改造所基于的更早一代仓；-LRU 批次从该谱系加回了若干 face 已删代码（见 §3.2）。
- **S2**：SLO 工具链的历史口径批次代号（`hbm_watermark.py` 的 upgrade_s2_mapping 与 `.bak_caliberfix_20260905` 残片所属）。
- **P 系模式号**（P0-P14）：face 深挖文档 §3 提炼的错误模式编号，本文引用的如 P12=CMake GLOB 把死文件编进构建、P13=跨仓 parity 占位零调用、P14=死代码判定须经第二人独立复核。
- **face H1-H3 / M1-M12 / low 71**：face 深挖文档 §1/§2 条目编号（high 3 + medium 12 + low 71 = 86 条）。
- **V4 gate / Part H / Part Q**：`windowed_trace_reader_test.cc` 内的被测准入门控与其用例分区名（Part Q 为门控自测分区、Part H 为拒收路径分区，见 §1-M16/§5.9）。
- **D4-I3**：两段逐出守卫的内部规则编号（测试注释所称"撤销最后一笔→缺口重开"的逐出量最小性守卫项，见 §1-L8）。
- **R0/R0c/R0c1/R0c3/R0c5/R2b/R5**：`verify/ledger_reconcile.py` 的对账规则编号（见 §1-M21/§5.15）。
- **R1-R16**：`cli_online_test.cc` 定义的在线 CLI 契约条目编号（见 §2-54）。
- **WP6/WP8/WP9**：SLO 工具链工作包子代号（对应 `slo_params_manifest.json` 的 wp* 键）。

---

## 1. 确认的严重错误（35 条）

### 1.1 high（3 条）

| # | 位置 | 问题与触发 |
|---|---|---|
| H1 | `astra-sim/system/CommunicatorGroup.cc:129-149`（源头 `astraccl/CollectiveImplLookup.cc:207/215/222`，删除点 `CollectivePlan.cc:27-31`） | 子集群通信域分支（comm_group 无 dimensions / torch pg 子集 ranks）把注册表**共享**的 `CollectiveImpl*` 原样交给 `implementations_should_be_removed=true` 的 plan：仅 `size()>1` 才换新 Ring（:132-141），`size()==1`（单维 native、单元素 `["ring"]`、custom 实现——仓内主流配置）时共享指针直接进删除语义 plan。同 ComType 两个子集群组退出 → **double free**（`Workload.cc:145 comm_groups.clear()` 统一析构即双删，无需特定时序）；任一组运行期销毁后 `Sys.cc:750/772/794/816` 再取同 ComType 实现 → **UAF**。**双桶互证**。注意：face 原仓已加 `clone_collective_impl` 保型克隆修复（本仓 grep 该符号 0 命中、face 仓 2 命中，本会话亲验）——**face 已修、本仓回归**。 |
| H2 | 实现 `extern/network_backend/analytical/common/NetworkFunction.cpp:11-19`（SI 恒等，"LOCAL PATCH 2026-09"）；失配 fixture 三处：`astra-sim/workload/execution_driven/tests/local_hbm_bandwidth_model_test.cc:44-48,343-349`；`extern/network_backend/analytical/congestion_aware/fluid/tests/FluidSchedulerLinkObserverTest.cpp:256-270`；`extern/.../fluid/tests/StaleEventCancellationTest.cpp:127-130,229` | 本仓主代码 `bw_GBps_to_Bpns` 已改 **SI 恒等**（1 GB/s=1 B/ns），但测试参照仍按旧二进制口径（2^30/1e9）标定：① local_hbm slow 场景硬编码期望 132/244/227，SI 下 60B→140ns，**常规构建目标跑 `--scenario slow` 必失败**；② FluidSchedulerLinkObserver 静态逐段重推 SI 真实 observed（L0={3,5,3,3}/L1={2,3,2,3}/window=20）与 fixture 期望（L0={3,6,3,2}…/window==19）全不合，:268 首个 `require` 失败即 exit 1；③ StaleEventCancellation 的 mouse 完成 tick=1+ceil(1/0.46566)=4≠3。②③为 opt-in（`analytical/CMakeLists.txt:24-30` option 默认 OFF），失败被默认构建掩蔽。**三桶互证**。face 同款定 high（face 已改 SI 自洽），本仓未同步且失配面扩大（新增位点①）。 |
| H3 | `examples/run_scripts/analytical/congestion_aware/` 三脚本：`Ring_allgather_16npus.sh:17,28`、`HGX-H100-validated.sh:17,28`、`run_analytical_with_custom_collective.sh:18,29` | 引用的 `build/astra_analytical/build.sh` 全仓不存在（find 仅命中 extern/helper/fmt 三方测试的 build.sh），二进制名 `AstraSim_Analytical_Congestion_Aware` 缺 `_Online` 后缀且路径错（唯一主目标为 `AstraSim_Analytical_Congestion_Aware_Online`，`astra-sim/network_frontend/analytical/CMakeLists.txt:58`，落 `build_congestion_aware/bin/`）——脚本必失败；实际先死在更早一道闸门：三脚本连 `--online-mode` 都未传（`OnlineCli.cc:324-328` 缺参报错退出）。H3 的另一半（`--remote-memory-configuration` 被 `allow_unrecognised_options` 吞掉）**已在本仓修复**：`CmdLineParser.cc:25` 注册、`main_online.cc:991-992` 消费。影响面限 examples 3 个示例脚本；主链路 `sh_test_mesh/run_scripts/run_online_strategy.sh:26` 用正确路径与目标名，不受影响。**双桶互证**（本会话亲验 :17/:28 原文属实）。 |

### 1.2 medium（22 条）

| # | 位置 | 问题与触发 |
|---|---|---|
| M1 | `extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.cc:170`（构造 :58-61） | 【LRU 新增面】PER_NODE_MEMORY_EXPANSION 下 `num_npus_per_node` 缺键构造为 0 且无守卫（对比 :127-131 对 `remote_mem_bw==0` 有 cerr+exit(1)），`issue()` 处 `sys_id / num_npus_per_node` **整数除零 SIGFPE**（本会话亲验 :58-61/:170 原文）；num-nodes 缺键（=0）或 npu-ids 越界时 :195 `vector<bool>::operator[]` 越界 UB 同族。构造链从命令行 `--remote-memory-configuration`（main_online.cc:991-992→:1090）直通，Workload.cc:376→461 无条件进入。触发：手写 PER_NODE 配置缺键——resolver 主路线只产 NO/PER_NPU（`config_resolver.py:226-249`），全仓无 PER_NODE 生产样例（唯一 PER_NODE 配置是 `remote_fifo_ledger_test.cc:223-230` 测试固件且键完整），属"新配置漏写即踩的雷"。 |
| M2 | `astra-sim/system/Sys.cc:1186`（构造 :215 置 0，解析 :414-415） | 【face M1 仍在】system json 缺 `preferred-dataset-splits` 键时构造值 0，`generate_collective` 第一条语句无条件调 `determine_chunk_size` 做 `size/0` 整型除零 → **SIGFPE**（调用链 Workload.cc:699/706→Sys.cc:843）。活模板含键（=6），462 份存量配置均含键，属新配置漏写即踩的雷。 |
| M3 | `astra-sim/system/Sys.hh:348` | 【face M2 仍在】`scheduling_policy` 无构造初始化（对比 :366 有 `=1`），缺 `scheduling-policy` 键时 `get_priority`（Sys.cc:1198）读未初始化枚举 **UB**：恰为 0/1/2 时按 LIFO/FIFO/EXPLICIT 静默执行（0/1 优先级方向相反），其他垃圾值在默认 Release（NDEBUG）下 assert 被编译掉直接 exit(-1)。仓内全部自带配置含键，需用户自带缺键 json 才可达；该模式继承自上游。 |
| M4 | `astra-sim/system/Sys.hh:381` | 【face M3 仍在】`collectiveOptimization` 同款缺键 UB（写点仅 Sys.cc:374/376，包裹于 :371 缺键静默跳过的 if）：Sys.cc:945 的 Baseline（逐维串行，:946-968）与 Greedy（带宽感知维度并行，:969 起）分支随机选定，**维度间调度与时延失真**（复核修正：两分支共用通信实现，失真非"通信趟数"）。现配唯 examples `custom_collective.json` 缺键且其 custom 路径在 :888 提前 return 不达 945；姊妹仓 joint 同未修。 |
| M5 | `astra-sim/system/Sys.cc:210`（intra 同款 :209 钉死 FIFO） | 【face M5 仍在，双桶互证；本会话亲验 :209-210 原文】`inter_dimension_scheduling` 唯一写点为构造常量 Ascending，initialize_sys 无任何解析键 → RoundRobin（:900-909）、OfflineGreedy/OnlineGreedy（:291-295、:850-858、:910-928、:964-969）全部死分支，`offline_greedy` 恒 nullptr，`scheduling/OfflineGreedy.cc+hh` 共 495 行（含 OfflineGreedyScheduleJournal 整个 rendezvous 记账机制）生产路径死代码；用户手写 `inter-dimension-scheduling` 键被**静默忽略**。intra 钉死 FIFO 使 insert_stream 的 RG/SmallestFirst/LessRemainingPhaseFirst 三分支（:1233-1293）恒假。 |
| M6 | `astra-sim/system/Sys.cc:1352` | 【face M6 姊妹点；face 已修、本仓未同步】`net_message_latency.back() /= net_message_counter` 无零守卫；计数器唯一递增点是 `StreamBaseline.cc:53`（PacketReceived 路径），多 phase 流中非首阶段无任何包到达时 0/0 → **NaN**，经 `DataSet.cc:47-56`→`NetworkStat.hh:30-42` 传染**平均网络延迟统计**。face 同位置已加 `&& net_message_counter != 0` 零守卫，本仓未同步（joint/wscllm 系同缺）。 |
| M7 | `astra-sim/system/SharedBusStat.hh:136-146` | 【face M6 主位点仍在】8 个 double 延时统计直接除以请求计数器（构造置 0，:46-47），无零守卫：custom collective 流全程不经 StreamBaseline::call（全仓 grep 计数器递增点仅 MemMovRequest.cc:41/StreamBaseline.cc:43/LogGP/DataSet，无一作用到 custom 流），流完成必经 `Sys.cc:1358 take_bus_stats_average()` → **8×NaN 必现**。复核补充：NaN 经 :1359→DataSet 传播进聚合统计，但 `total_shared_bus_*` 八字段全仓零消费/输出点，污染真实发生却不外显（潜伏性统计损坏，复核建议实际影响降 low；本表维持原 medium 定级）。 |
| M8 | `astra-sim/system/SimSendCaller.hh:18`（同款 `SimRecvCaller.hh:19`） | 【face M4 仍在，触发面微调】用 int 存 uint64_t 消息大小（Sys.cc:1544/1565 隐式收窄，`SimSendCaller.cc:34` 传回 uint64 形参时负值符号扩展为巨大值）。复核修正可达性：delay>0 分支潜伏（全部生产调用点 delay 均为 0），**实际可达截断点是 rendezvous 路径**（`RendezvousSendData.cc:21-23`/`RendezvousRecvData.cc:21-23`，经 `--rendezvous-protocol` 真实可达）——单条消息 count>2GiB 时按错误字节数算延迟。 |
| M9 | `astra-sim/system/astraccl/custom_collectives/CustomAlgorithm.hh:36-66`（`.cc:24,131`） | 【face M7 **本仓回归**】CustomAlgorithm 无析构函数——face 的 `virtual ~CustomAlgorithm(){ delete et_feeder; }` 经 diff 实证被本仓删除；构造 :24 `new ETFeeder`、:131 结束仅 exit()，`Sys.cc:1355 delete` 经虚析构只回收对象本身，每生成一个 custom 集合通信 phase **泄漏 ETFeeder+fd+index_map 各 1 份**；fd 耗尽后 ETFeeder 构造抛 runtime_error 被 :33 catch 后 exit(1) 优雅中止（非崩溃 abort）。条件：启用 custom 实现。 |
| M10 | `astra-sim/system/astraccl/native_collectives/logical_topology/BinaryTree.cc:65,73,81,89` | 【face M8 **本仓回归**】仅支持 2 幂节点数：构造 :22-27 折半使非 2 幂 total 实建 2^floor(log2(total)) < 成员 rank 数，四个 getter 用 `map::operator[]` 取出 nullptr 直接解引用 → **段错误**。face 的非 2 幂守卫块（face :18-26）经 diff 实证被本仓删除（含 `#include <cstdlib>`），本仓 grep "power-of-two" 为 0。触发：system JSON 非 2 幂维 + doubleBinaryTree（`CollectiveImplLookup.cc:26-27`）；Ring/HalvingDoubling 走 RingTopology 分支不受影响。 |
| M11 | `astra-sim/workload/Workload.cc:104-115`（联动 `Sys.cc:465-468`、`LocalHbmBandwidthModel.cc:35-38`） | 【**LRU 新增**】新加的 `hbm-kv-restore-bandwidth-sharing` 标志绕过 `local_mem_bw<=0` 自动降级（Sys.cc:465-467 只写 hbm_bandwidth_contention、无 kv_sharing 分支），且模型构造新增 `peak_perf<=0` 前置检查——任一不满足即从 Workload 构造函数抛**未捕获** `std::invalid_argument` → **std::terminate 启动即崩**（new Workload 在 Sys 构造内 :282/287，生产 new Sys 在 main_online.cc:1113，全链无 try/catch）：① kv-sharing:true + 缺/非正 local-mem-bw；② 缺 peak-perf（默认 0，Sys.cc:179）+ contention 默认开（Sys.cc:190），连 A/B 基线运行也崩。face 对照：face contention 默认 false 且有 force-disable、peak_perf 检查延迟到 COMP 下发——本仓引入双重差异。生成配置安全（config_resolver.py:432/438 恒写正值并上游校验），手写配置即踩雷。 |
| M12 | `astra-sim/workload/Workload.cc:530-532,567,604,625`（回退路径 :486-490；`Roofline.cc:24`） | 【face M9 **本仓回归**】roofline 模式 `num_ops==0` → operational_intensity=0 → perf=0 → `0/0`=NaN；或缺 local-mem-bw（Sys.cc:184 默认 0）→ Roofline 恒返 0 → Inf。:567 `static_cast<uint64_t>(elapsed_time*1e9)` 为 UB；:604/625 memory_utilization 写 NaN，在线 compact 端 `Statistics.cc:21-26,368-373` 对非有限值直接 **std::exit 中止整个仿真**。face 现行有两道防线（peak/bw<=0 前置 exit + node_num_ops!=0 守卫），本仓**两道皆无**（丢失 face 修复，非"原样保留"）。同根新增位点：restore 回退 :486-490 在两 HBM 标志全关的 legacy 配置缺 local-mem-bw 时 Inf → ceil → uint64 巨值（仿真实质挂死）。shipped 管线不触发（两键齐全），需畸形图（COMP 节点 num_ops=0）或手写缺键配置。 |
| M13 | `astra-sim/workload/MetricCollector.cc:511-517`（同面 :461-462） | 【face M10 **本仓回归**】load_manifest 对 event_code 仍"先按 uint8_t 窄化再校验 1..8"（nlohmann 3.10.5 `get_arithmetic_value` 为无范围检查的 static_cast），≥256 回绕成合法码（257→1=PREFILL_START_ISSUE，恰在静默区间）**静默挂错边界**；同改动面丢失负值防护：:511 负 node_id 回绕、:461-462 负 value_ns 回绕为巨大 arrival。face 同文件 :518-536/:463-469 已是"宽域 int64 先校验再窄化"（diff 亲读，注释原文即描述 256 回绕），本仓被回退。触发：metrics 启用 + manifest 含 ≥256 code 或负值，链路 CmdLineParser.cc:46→main_online.cc:1013→:293-303 完整可达。 |
| M14 | `astra-sim/workload/Statistics.cc:680,698-706` | 【-LRU 面独立报告，face 深挖未载、face 同位点同码】`get_operator_type(ETFeederNode)` switch 漏 `METADATA_NODE`：default 仅 critical+assert(false)（Release 下 no-op）后**返回未初始化枚举 stat_node_type（UB）**，垃圾类型入库 operator_statistics 并计入 type_time，撞 CPU/GPU/COMM 即污染 comp_comm_overlap 与 roofline 利用率权重。静态路径 Workload.cc:348 对每节点无条件 record_start，含 METADATA 节点的静态 ET 即触发；NodeView 重载（:731-733）正确映射 Metadata，不对称坐实疏漏。 |
| M15 | `astra-sim/workload/execution_driven/tests/calendar_reader_oracle_test.cc:96-98`（`LegacyOracleWindowedTraceReader.hh:53`） | 【face M12① 仍在】oracle 测试 legacy 对照臂注释与 CMake 注释（`analytical/CMakeLists.txt:134-137`）均声称 "UNBOUNDED arm: window 0"，实际构造未传窗口参数、落默认 `high_water=128`（仅 0 才无界，`.cc:221`）：真实队列（>128 行）下 legacy 臂只读前 128 行、calendar 臂（首泵全文件提交）读全量，长度断言**必失败误报**（默认 synthetic 8 行 <128 掩盖问题）。修复=显式传 `high_water=0`。face 的实测数字（legacy=43 vs calendar=200）引自其文档，本仓按只读纪律未复跑。 |
| M16 | `astra-sim/workload/execution_driven/tests/windowed_trace_reader_test.cc:1065`（`WindowedTraceReader.hh:136-137`） | 【face 可疑点坐实为 finding】Part Q（V4 gate 自测分区）正常路径对照臂把 128 传给第三参 **max_arrival_ns**（非旧 API 的 high_water；外层仓 dd383aa 时期旧签名第三参恰为默认 128 的 high_water，系 P0 改造前旧签名残留）：CSV 两行 turn-0 到达 1000/2000ns（:1013-1014）全部 >128 被拒收零提交，:1071-1074 `late_static_submit_count()==0` 与 `gate_ok` 空洞成立（拒收行按设计走 completion audit 不进门禁，Part H :462-464 自证）——对照臂对"把准时提交误计为 late"类 gate 回归**零覆盖**。Part Q 主断言（延迟提交令 gate 失败）不受影响。 |
| M17 | `astra-sim/network_frontend/analytical/congestion_aware/main_online.cc:757-762`（后端 `MultiDimTopology.cpp:195-198`） | 【face M11 仍在】`compute_mesh_edge_links` Ring 分支无条件 `next_id += 2*npus`（:761），但后端宽度==2 时退化 `connect_mesh_dimension` 该维实耗 npus 个 id → 多记 npus 个，其后各维 link id 基址整体偏移，**edge 归因错位**（:758-759 注释自称 "incl. the radix==2 mesh fallback" 与后端实现直接矛盾）。触发：link observer + metrics enabled（main_online.cc:1058-1071）+ 多维含宽度 2 Ring 维且其后还有 Mesh/Line 维。仅污染观测/诊断输出（edge_max_link/edge_max_bytes），不影响仿真正确性。 |
| M18 | `extern/network_backend/analytical/congestion_aware/basic-topology/Ring.cpp:19-22`（`Device.cpp:42,46`；`MultiDimTopology.cpp:195-206`） | Ring 退化宽度无守卫：1D Ring(2) 闭合边对已连 device 二次 connect → Debug assert 崩溃；Release 下 `links[id]=link` **静默覆盖**，闭合边两次 connect 使 2 条 link 成孤儿（FluidScheduler.cpp:58-67 相应多建 2 个 link_state）；1D Ring(1) 与多维 size-1 ring 维（`(a+1)%1==a` → connect(src,src)）同款（:195-198 仅对==2 有 mesh fallback 特判，守卫不一致）。`NetworkParser.cpp:184-190` 明确放行，触发配置合法可达。5 个姊妹仓同码（上游共有，非本仓引入）。 |
| M19 | `extern/graph_frontend/chakra/src/feeder_v3/protobuf_util.h:45-51`（消费 `et_feeder.cpp:88-92`） | readMessage 中 `f.read()` 失败不检查、`ParseFromArray` 返回值丢弃、恒 return true（唯一 false 出口是 :37-38 varint 失败）：截断/损坏的 .et 文件 → 部分解析/未初始化 buffer 静默 put 进 `_node_cache` 按 protobuf 默认值**失真运行，无任何报错**。加重情节：f.read 失败置 failbit 后 seekg 无法清除，此后所有 cache miss 均静默入默认节点；build 阶段截断节点仅被当作正常文件结束（et_feeder.cpp:59-61）。生产链 Workload.cc:90 `new ETFeeder` 真实可达。 |
| M20 | `sh_test_mesh/slo_tools/hopbytes.py:159-181` | 【**LRU 新增**】collect_face 未适配 -LRU 契约行列表：decode 决策的 `prefill_decode_transfer` 在生产者侧已恒为列表（`face_online_scheduler.py:1700-1701` `_transfer_rows`，行结构无 shards 键），而 hopbytes 只在 `isinstance(transfer, dict)` 时取 `.shards`——列表形态下 shards=None 且 else 路径**无任何计数语句，整个 decode P→D 分支静默产零**：slo_hopbytes_total.csv 与 per_request 的 decode 迁移分量恒缺、hop_bytes_total/bytes_with_hops 恒偏低。同文件消费者 kv_cache_adapter.py:294/:772、hbm_watermark.py:343-361 均已适配（注释明写"-LRU 新产物：契约行列表"），唯 hopbytes 失配；本仓 `REPO_VARIANT="astra-sim-face"`（plan_materializer.py:60）恒走 collect_face。 |
| M21 | `sh_test_mesh/workload/llama2_7b_inference/online/verify/ledger_reconcile.py:137-163,385-409` | load_cpp_facts/R0c 不剔除 `batch_train_` 哨兵完成事实：哨兵 watch 以 train_id+prefill 进入 completed_groups（`face_online_scheduler.py:1002-1009` 注册 → C++ `WatchRegistry.cc:88-91` fire 原样携带 train_id → `DecisionBridge.cc:229-237` 原样落盘 → journal 原样 yield），本文件全形式 grep **零** batch_train 处理 → 任何含哨兵列车（T_max=8 截断且无 drain/exit，如 a1 fixture 36-chunk prefill）的运行，R0c1/R0c3/R0c/R0c5 **必然误报失配**，对账工具恒"不平"exit 1。调度器自身 :545-549 对同组按 BATCH_TRAIN_PREFIX 路由核销，恰证对账侧缺同款处理。 |
| M22 | `sh_test_mesh/workload/llama2_7b_inference/online/verify/diff_explainability.py:328-444` | 退出码 1 契约**整体死分支**：全文件无任何 `("DEFECT", …)` 级产出点（19 处 append 全为 PASS/DIFF），defects 恒空 → :438 `elif not defects` 恒真 → :442-444 ok=False 不可达、恒 exit 0；`classify_differences` :275-278 以"排队状态差异"兜底二元穷尽归类，连 docstring :12/:43 明文"行数对不上→退出 1"的情形也归入后 exit 0——感知开关真出现排队状态差异时**工具恒放行**。全仓无自动化调用（手工工具），影响面为人工复核被恒 exit 0 虚假放行。 |

### 1.3 low（10 条，均为 bug 类）

| # | 位置 | 问题（一句话） |
|---|---|---|
| L1 | `extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.hh:42`、`.cc:24-25` | 构造函数标 `noexcept` 但内含可抛操作（:35 malformed JSON 抛 parse_error；NO 模式下 :117 类型校验被 `mem_type` 条件短路、:124 对字符串键抛 type_error.302；同类点还有 :38/:56/:60/:112）——noexcept 违例直接 std::terminate（abort 无诊断），替代了本文件既有 cerr+exit(1) 错误路径。NO 模式下 remote-mem-bw 本不被使用，属错误处理健壮性问题。 |
| L2 | `astra-sim/system/Sys.cc:225` | 【face 可疑点确认实存】`collective_impl_lookup` 每 Sys 实例泄漏：构造 new、~Sys（:300-344）逐项 delete 清单无此项，全仓零 delete 零所有权转移，lookup 无静态回收点。face 修复轮已补 delete（face 文档 §5），本仓未同步。 |
| L3 | `astra-sim/system/astraccl/native_collectives/collective_algorithm/Ring.cc:239`、`HalvingDoubling.cc:264` | 【face 已改、本仓未同步】构造 RecvPacketEventHandlerData 末参传 `packet.stream_id`，而 MyPacket 两构造均不初始化该字段、全仓零写点 → 读未初始化值（UB）。值流入的 `RecvPacketEventHandlerData::stream_id` 全仓零读点，无行为失真；face 已改传 `stream->stream_id`，同步时照改传参即可。 |
| L4 | `astra-sim/workload/MetricCollector.cc:498-504` | manifest rank 键 `std::stoi` 不传 pos 校验尾随："12abc" 静默当作 rank 12、"-1" 接受为永不匹配桶（运行时 rank 来自 sys->id 恒非负），整数溢出才走 fatal；残留事件 finalize 时静默丢失无警告。旁证：`ParsedGraphBatch.cc:358-370` 解析同类键显式校验尾随，本处恰缺。 |
| L5 | `sh_test_mesh/tests/test_config_resolver.py:135-143` | 测试防护集与被测集不同步：自建 managed_fields 漏 `remote-mem-latency`（config_resolver.py:24 真集合含它）——模板若含该键此测试不再拦截（主路径 :425-427 用真集合仍拦，仅测试防护面失真，对该字段构成静态防护零覆盖）。 |
| L6 | `sh_test_mesh/run_scripts/run_online_same_tick_milestone.sh:142,167` | 【宿主环境失配】本机 date 为 uutils coreutils 0.8.0，`date +%s%3N` 实测输出 19 位数（%3N 未按毫秒截断），脚本耗时差计算失真；同仓 `run_online_idle_fixture.sh:60-64` 已为此改用 python monotonic_ms 并注释点名该坑。差值只进信息性 echo 不断言，故降为 low。 |
| L7 | `sh_test_mesh/workload/llama2_7b_inference/metrics_schema.py:302-304` | 对非法 event_code 的报错消息称 "protocol encoding 1-7"，实际 `EVENT_FIRST_TOKEN_COMPLETE=8`（:31）也在 EVENT_EDGE_BY_CODE 合法表内，真实值域 1-8——诊断消息失实误导排查（校验逻辑本身正确，测试亦未断言该文案）。 |
| L8 | `sh_test_mesh/workload/llama2_7b_inference/test_face_tiered_eviction_sequence.py:174` | 恒真断言：注释（:169）声称验证 D4-I3"撤销最后一笔（半层 c,-80）→ 280 < 300 缺口重开"，实际写 `assertGreater(360 - 80, 0)`（280>0 恒真）——对 ensure_physical_fit 中途停机分支的**逐出量最小性守卫零验证**（正确写法 `assertLess(360-80, 300)`）。被测分支真实可达（session_kv_manager.py:1865,1960）。 |
| L9 | `sh_test_mesh/workload/llama2_7b_inference/online/face_online_scheduler.py:988-989 vs 1145-1161` | 【**LRU 新增**】WP9 拆分路径 drain_block_ends 断链：非拆分 _emit_train 在列车发射后回写 `runtime.drain_block_ends`（joiner 逐出链触发门的唯一来源，消费点 :935-936/:773-777），拆分路径 `_emit_train_remainder` 注册同样的 drain watch 却从不回写 → SH_FIRST_TOKEN_SPLIT=1 时被拆分列车 drain 的请求其后 arm 全 None，逐出 shard 丢掉对 drain 列车 end barrier 的时序依赖边且无报错。默认拆分关（SH_FIRST_TOKEN_SPLIT 缺省 "0"），生产默认路径不受影响；现有测试不覆盖。 |
| L10 | `sh_test_mesh/workload/llama2_7b_inference/online/test_train_machinery.py:75,263-275` | 空转测试：`_bare_scheduler` 置 `_ready_frontier=set()` 使被测 busy 门分支（face_online_scheduler.py:885-886）不在执行路径上——**删掉 busy 门该测试仍然通过**；且全部测试文件对 :885-886 分支零执行路径（唯一 `_plan_and_emit_trains` 调用即此处，两处 _ready_frontier 测试赋值均为空集）。 |

---

## 2. 确认的死代码（59 族，按族归类）

全部经 P14 纪律复核（全形式 grep，含 tests/CMake/脚本/宏字符串形态，排除姊妹仓与自身）；标注"编译进产物"者经根 `CMakeLists.txt:61-76` 的 file(GLOB) 确认在构建内——零引用 ≠ 未编译（P12）。

### 族 A：整类/整文件/整族零引用（8 族）

| # | 位点 | 说明 |
|---|---|---|
| 1 | `astra-sim/workload/RemoteFifoLedger.cc:18-20` + `.hh`（**LRU 新增面**） | sensing 账本整族未接线：`set_enabled` 生产零调用（唯三调用点是测试 `remote_fifo_ledger_test.cc:461,746,758`）→ `enabled_` 恒 false → 唯二生产引用 record_issue/record_completion（`AnalyticalRemoteMemory.cc:192,232`）恒 no-op；全部查询/export 接口（active_ports/port/port_ranks/rank_attribution/attributed_ranks/architecture/reset/total_* /drained/sidecar_row/set_architecture）零生产引用。`hh:51-52` 注释 "main_online.cc enables the ledger for a --sensing-enabled run" 在本仓**失实**（本仓 --sensing-enabled 实际只接线 Phase-3 injected-unfinished summary，`OnlineCli.cc:172-178`/`main_online.cc:536`），并与 hh:47-48 "blueprint repos never set_enabled(true)" 对照暴露矛盾。经根 CMakeLists.txt:61-64 GLOB 编进生产二进制但零可达。注：账本写侧调用实存，死的是没有 set_enabled 的整条观测路径。 |
| 2 | `astra-sim/system/MemEventHandlerData.hh:16-21` + `.cc` | 整类死代码：全仓唯一实质引用是 `Sys.cc:704` 死分支强转（另有 Sys.cc:19/Workload.cc:10 两处无使用 #include），触发事件 CompFinished/MemLoadFinished/MemStoreFinished 全仓零产生点；经 GLOB 编进库但永不可达。构造函数（`.cc:10-13`）显式将 workload/wlhd 置 nullptr、Sys.cc:705 有判空——即使接线也是安全 no-op（纯死代码，勘误见 §5.2）。 |
| 3 | `astra-sim/system/AstraSimDataAPI.hh` + `astra-sim/common/AstraSimDataAPI.hh` | 双副本同 guard `__ASTRA_SIM_DATA_API_HH__`（字节级相同，AstraSimDataAPI/LayerData），全仓零 include 零引用——两份均为纯死码（双桶互证）。附带隐患：两份 `Common.hh` 同 guard `__COMMON_HH__` 同 md5 且双侧均有真实包含者，单侧修改会按包含顺序被静默遮蔽（非死码，登记）。 |
| 4 | `astra-sim/system/CSVWriter.hh:17-36` + `.cc` | 整类零构造；唯一外部引用是 UsageTracker::report 死方法的形参类型（同一死链，见 #21）；类内还埋着 write_cell 的 read 返回 0 后 buf 不更新的无界追加循环（`.cc:164-170`）与 open/lockf 无界 busy-retry（:156-162）——当前不可达，删除时一并消除。经 GLOB 编入主目标。 |
| 5 | `sh_test_mesh/workload/llama2_7b_inference/metrics_integration.py:1-452` | 整模块零引用死文件：全仓 grep（含 astra-sim/ 子树、全文件类型）零结果，无 `__main__` 不可作脚本运行，`__all__` 11 个公共符号零消费（docstring 自述 "not imported by the online GraphBatch routes" 属实）；metrics_schema.py 本身活（test_metrics_contract/metrics_postprocess 消费）。 |
| 6 | `astra-sim/common/AstraComputeAPI.hh:22-58` + `.cc:10-29` | ComputeKernel/AstraComputeAPI + 4 个 create_llm_* 工厂全仓零引用；**死代码里藏错**：get_static_runtime 默认实现返回未初始化 timespec_t（hh:46-48）、ComputeKernel 默认构造不初始化 type/phase（hh:32）。经根 CMakeLists.txt:72 GLOB 编入库。face 修复已删，本仓保留。 |
| 7 | `astra-sim/system/astraccl/native_collectives/logical_topology/{Torus3D,LocalRingGlobalBinaryTree,LocalRingNodeA2AGlobalDBT}.{hh,cc}` | 6 文件全形式 grep 零引用（唯一命中 `CollectiveImpl.hh:22` 同名枚举成员，该枚举值本身亦零使用）；**死代码里藏错**：LocalRingNodeA2AGlobalDBT.cc:58-59 的 dim==2 All_Reduce 分支硬编码传 dimension=2，而 `DoubleBinaryTreeTopology.cc:40-45` 只认 0、其余返 nullptr——接回即断。face 原仓已整删（亲查同目录无此三文件），本仓保留且被 GLOB 编入。 |
| 8 | `sh_test_mesh/workload/llama2_7b_inference/online/verify/` 孤儿 fixture 族 6 文件 | bridge_cpp_death_fixture.py（docstring 引用的 run_bridge_cpp_death_fixture.sh 不存在）、bridge_streaming_fixture.py、idempotency_fixture.py、diff_explainability.py、profile_scan_audit.py、train_a1_eviction_fixture.py（usage 行还写 "bash <file>.py"）——全形式 grep 全仓零接线零引用；对照组 bridge_echo/lifecycle/wakeup/same_tick 均有真实接线。 |

### 族 B：死分支/恒假检查/恒空账本（11 族）

| # | 位点 | 说明 |
|---|---|---|
| 9 | `astra-sim/system/Sys.cc:690-692,701-708` | handleEvent 两个无喂入死分支：① NPU_to_MA/MA_to_NPU——全仓该事件唯一产生点 `MemBus.cc:54/60/78/84` 走 register_event→事件队列→Callable::call，不经 handleEvent；② CompFinished/MemLoadFinished/MemStoreFinished——三个枚举值全仓仅定义与该消费点，零产生零喂入（MemEventHandlerData 构造实为置 nullptr，接线也是安全 no-op，纯死分支）。 |
| 10 | `astra-sim/system/Sys.cc:227-230` | 构造函数中 `initialize_sys()==false` 分支恒假：函数仅 :354 exit(1)（文件打不开）与 :508 return true 两个出口（4 处 sys_panic 亦 exit(1)），永不返回 false。 |
| 11 | `astra-sim/workload/HardwareResource.cc:148-150` | ETFeederNode 版 is_available 的 else 内 `==0` 恒假赘枝（外层 :142 同条件为真已 return true，进入 else 即 counter!=0）；本仓新写的 NodeView 版（:272-289）无此赘枝可佐证其为残片。注意：face 深挖文档所记 face 位点（face :123-125）与 face 现码对不上，本赘枝实存于 face-LRU/joint/wscllm/wscllm-LRU 四仓。 |
| 12 | `astra-sim/workload/Workload.cc:409-413` | `if (true) { issue_pytorch_pg_metadata(node); } else { throw …; }` else 恒死（全仓非 extern 唯一 if(true)）。face HEAD 已删（issue_metadata 现为直接调用式），本仓保留的是 face 历史版本位点。 |
| 13 | `astra-sim/workload/LocalMemUsageTracker.cc:251-255,385` | LP64 下两处 uint64 上限比较恒假（size_t 与 uint64_t 同宽、SIZE_MAX==UINT64_MAX）；:385 为复合 if 的第二个子条件（error 分支整体仍因 fread 失败可达）；:251 检查在 ILP32 有意义，死代码结论限 64 位构建平台。与 OnlineCli 同模式（face 无、本仓有）。 |
| 14 | `astra-sim/workload/execution_driven/GraphBatchCommitter.cc:473-474,1043-1047` | comm.tag（uint32_t，`GraphSource.hh:80`）值域检查两处恒假：uint64 提升后 `>uint32max` 不可能（第一处为复合条件的 tag 子项恒假，src/dst 子项仍可达）；int64 提升后 `tag<0` 永假（第二处整块死）。两函数均在生产路径上，属活路径死检查。face 现行无此两段——**-LRU 从 sh_2.0 谱系新加回**。 |
| 15 | `astra-sim/workload/execution_driven/OnlineCli.cc:233-238` | `--request-max-arrival-ns` 的 uint64 上限检查恒假（strtoull 结果与自身类型最大值比较；:220 已拒 ERANGE，溢出唯一出路被封死）；选项路径本身活（main_online.cc:1154/1164 消费）。face 无此 6 行——**-LRU 新加回**。 |
| 16 | `astra-sim/workload/execution_driven/DecisionMailbox.hh:277-283` + `.cc:141,150` | 空转守卫/finalize 族生产恒假：`no_decision_python_callback_count` 生产零自增（唯一调用在测试）→ `main_online.cc:1992` 的 fail-closed gate **永不触发**（自检空转）；`finalize_pending_` 生产无 true 置位 → has_decision_work 的 `|| finalize_pending_` 与 drain 清理恒假（hh:246-247 注释自证 "nothing sets it in phase 1"）。 |
| 17 | `extern/graph_frontend/chakra/src/feeder_v3/dependancy_solver.cpp:49-51` + `dependancy_solver.h:63-66` | take_node 的 "already taken" 检查恒假（:42-48 先拦截不在 free 集的节点，且 free/ongoing 锁内互斥迁移永不相交）；DependancyResolver 构造 throw 不可达（唯一构造点 `et_feeder.h:37` 以 constexpr true 实参化），连带 :145/:153 两 if 恒真。 |
| 18 | `sh_test_mesh/slo_tools/hbm_watermark.py:2046-2048` | `npus_for_calibers = load_npus_per_instance(...) if not npus else npus` 恒假：capacity 仅在 `if npus:`（:2017-2018）块内赋值，进入 `if capacity is not None:`（:2032）块时 npus 必为真值 → `not npus` 恒假，load_npus_per_instance 调用不可达（若可达其异常也未被 :2053 except 捕获，与 :2013-2016 降级意图矛盾）。 |
| 19 | `sh_test_mesh/workload/llama2_7b_inference/online/graph_batch_builder.py:454,1136-1147` | pending_history 账本恒空（仅 __init__ 赋空、全仓无写入点）→ :1141 gate 恒 None、:1147 不可达，且 :1138-1146 两分支动作逐字等价，:1137 的 pending_request_by_session 读取不产生任何行为差异（账本只剩 turn-0 写入/turn>0 pop/run-end 审计消费）。 |

### 族 C：死函数/死访问器/死接口（23 族）

| # | 位点 | 说明 |
|---|---|---|
| 20 | `astra-sim/system/Sys.cc:142-149` | `SchedulerUnit::get_average_latency_per_dimension` 零调用死函数（face §2 死函数族同款仍在），内含 total_chunks_per_dimension==0 时 double 除零 inf/NaN 隐患。 |
| 21 | `astra-sim/system/UsageTracker.cc:59-107` | report/report_percentage 生产零调用（Sys 对 usage 仅 increase/decrease_usage，Sys.cc:89/116），唯一测试引用传 nullptr 且只验证 retain_history_=false 时 throw（fail-closed）——报告生成主体（含 levels==1 浮点除零理论隐患 :81/:98）全路径不可达；仓内全部以 levels=2 构造，风险纯理论。删除需同步修改 `system_history_lifecycle_test.cc:314,323`。 |
| 22 | `astra-sim/system/DataSet.hh:33` + `SendPacketEventHandlerData.cc:16-21` | ① `DataSet::is_finished()` 零调用（完成判定走 notifier 回调不走轮询；其余命中为 Workload 同名成员变量）；② `SendPacketEventHandlerData(Callable*,int)` 带参构造零调用（两处 new 均默认构造后手工补字段），且该死构造不初始化 wlhd（hh:20）——未来改用带参构造且漏补字段时激活。 |
| 23 | `astra-sim/system/Roofline.cc:13,19-21` | 单参构造 `Roofline(double)` 零调用（唯一构造点 Sys.cc:472 用双参版）且不初始化 bandwidth（hh:19）；set_bandwidth 零调用。若被启用且未先 set_bandwidth 即 get_perf（Workload.cc:531 有真实读者）则读未初始化 double——当前因构造死而未爆发。 |
| 24 | `extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.cc:244,258` + `.hh:53-54` | `architecture_name()`/`port_mapping_rule()` 全仓零调用（无扩展名过滤 grep 仅定义+声明 4 行；基类 AstraRemoteMemoryAPI 无此二符号，排除虚接口隐性引用）；hh:50-52 头注释声称 "main_online must not re-derive either string"，实际 main_online 从不消费。属已编译零引用死函数。 |
| 25 | `astra-sim/system/astraccl/native_collectives/logical_topology/BinaryTree.cc:99-125`、`RingTopology.cc:176-188`、`CollectiveImpl.hh:21-23`、`CollectiveImplLookup.hh:18` | 死函数/死枚举值族：BinaryTree::print（含递归自调用）与 RingTopology::is_enabled 零调用；死枚举值 HierarchicalRing/DoubleBinaryTreeLocalAllToAll/LocalRingNodeA2AGlobalDBT 零消费；BYPASS_PERNODE_CUSTOM 无任何生产者（头注释自认 "No current usecase"，全部 11 处 get_collective_impl 调用点无人传该值）。face 已删其中大半（但 face 仍保留 DoubleBinaryTreeLocalAllToAll）；git 历史显示本仓部分符号系改造时引入而非继承后保留。 |
| 26 | `astra-sim/workload/HardwareResource.cc:291-301` + `hh:77` | report() 零调用（全仓 `.report()` 命中均为 OnlineStatsCounters/Workload/Statistics 异族同名）；num_npus 字段仅声明与构造初始化、无读取点（Workload.cc:118 恒传字面量 1）。 |
| 27 | `astra-sim/workload/MetricCollector.hh:160-170,172-182,224-227,457-458` | 死面与失实注释（**3 个死面实例 + 1 个活函数注释失实**，非"-LRU 新增"整体定性）：① Phase-0 PerformanceCounters 整框架（结构体+enable_counters/counters_enabled/counters()+成员）全仓零调用零递增零序列化，字段恒 0；② `slo_watermark_period_ns()` 访问器零调用（emit 均直接用成员）；③ `rank_instance_conflicts` 累加后在 cc:2569 被 `(void)` 丢弃——注释却宣称 "conflicts are counted, never silently resolved"；④ hh:145-147 `on_local_hbm_restore_issue` 注释称 "In this repo no caller exists" **失实**——本仓 `Workload.cc:474-477` 在 is_local_hbm_kv_restore 路径真实调用，该函数是**活函数**（face 的 P13 死函数在本仓已接线消除），死的只是注释（face 的死函数本体已在其修复轮删除）。谱系：①②③在 face 已提交基线即含（face 修复删除尚未提交），非 LRU 自创；④的注释失实系 -LRU 接线后遗留。 |
| 28 | `astra-sim/workload/Statistics.hh:70-72,86-87,150,254-256` | 混合死面（3 实例，谱系各异）：① `retire_online_operator` 零调用（face 已删、本仓回归保留），且 hh:254-256 注释被 -LRU 改错（宣称 compact 聚合 "populated exclusively by retire_online_operator(false)"——实际填充者是 complete_online_service_operator→add_online_roofline_contribution，`Statistics.cc:204-207`；face 原注释本正确）；② `OperatorStatistics::network_bandwidth`（**-LRU 新增**）唯一生产写点 Workload.cc:216，唯一"读者"是 Statistics.cc:833-862 **整段注释掉的报告块**（连测试读的都是另一结构体的同名字段）；③ 无参 `get_operator_statistics()` map 重载零调用（face 继承、仍在）。 |
| 29 | `astra-sim/workload/execution_driven/GraphBatchCommitter.hh:311` | `was_json_id_committed` 零调用死函数（含 friend TestAccess 亦不引用），功能与 `resolve_store_id().has_value()` 完全重复。face 无、**-LRU 新增**。 |
| 30 | `astra-sim/workload/execution_driven/DecisionBridge.hh:200-202` | `FileDecisionBridge::stats()` 死访问器（字节计数实际经 `stats_report()` 打印，main_online.cc:1885），`main_online.cc:187-188` 注释宣称 "fetched through bridge->stats() at run end" 失实；stats_ 成员本身活跃（写点 DecisionBridge.cc:327-328/539-540/557）。face 无、**-LRU 新加回**。 |
| 31 | `astra-sim/workload/execution_driven/OnlineStatsCounters.hh:46-48` | `reset()` 零调用（注释称 "reset() for fixture reuse"，全部测试与 main_online 均不调用，driver_ctx.stats 只累加）。face 无、**-LRU 新增**。 |
| 32 | `astra-sim/workload/execution_driven/ServiceCoordinator.cc:172-179` | `transition_log()`/`transition_log_dropped()` 在一切可构建目标中零引用（唯一调用者是孤儿 fixture ingress_idle_fixture.cc）；且 transition_log 读成员不持 mtx_（写点 set_state 持锁），接回非同线程读取即数据竞争（两访问器加锁口径不一致）。 |
| 33 | `astra-sim/network_frontend/analytical/include/common/CmdLineParser.hh:57` + `common/CmdLineParser.cc:62-64` | `get_options()` 全仓零调用（本会话亲验：全仓仅声明+定义 2 行命中）——face §2 同名位点，face 修复已删、本仓保留且被 GLOB 编入二进制（上游 garnet 前端消费者在本仓结构性缺席）。 |
| 34 | `extern/network_backend/analytical/include/.../fluid/FluidScheduler.h:36-42,108-118` + `.cpp:45,370,534,596-631` | Phase-7 拥塞快照集群零消费：LinkCongestionSnapshot / link_congestion_snapshot() / link_state_epoch() / link_count() 全仓零调用，字段 link_state_epoch_ 的全部写点仅服务该死接口。注：joint 仓 `link_count()` 有真实消费（其 main_online.cc:1582），**跨仓删除需豁免**；本仓确认零引用（两个 fluid 测试亦不引用）。 |
| 35 | `extern/network_backend/analytical/congestion_aware/fluid/FluidScheduler.cpp:112-118,1037-1039` | `flush_pending_starts_deferred` 零调用（online 实际路径为 main_online.cc:1052 `set_deferred_flush_mode(true)` + start_flow 内联 schedule_event_deferred；face 文档同款死函数）；`get_completion_heap_size` 零调用。头注释括号中已列出 start_flow+deferred 为合规第二入口，与实现自洽（复核修正原发现"注释失实"子断言）。调用方责任层面的脆弱契约另见 §5.11，与本条不矛盾。 |
| 36 | `extern/network_backend/analytical/congestion_aware/basic-topology/BasicTopology.cpp:40-44`、`topology/Topology.cpp:83-93`、`network/Device.cpp:22-24`、`Link.h:39` | 拓扑/链路死访问器链：① `get_basic_topology_type` 零调用且守卫字段在 Ring/Switch 构造不赋值恒 Undefined（Mesh/FullyConnected 有赋值）——**接线必触发 cpp:41 assert，死代码藏错**；② `get_links_count` 全链（Topology→Device）零外部调用；③ `Link::bandwidth` 只写不读（唯一取值路径 get_bandwidth_Bpns 返回 bandwidth_Bpns）。 |
| 37 | `extern/graph_frontend/chakra/src/feeder_v3/et_feeder.h:55-62` + `dependancy_solver.h:46-47,82-84` + `et_feeder_node.h:24` | feeder/依赖求解器零调用接口族 **12 符号**：ETFeeder 的 hasNodesToIssue/getNextIssuableNode/pushBackIssuableNode/freeChildrenNodes/addNode（恒 throw）/removeNode（6 个）；DependancyResolver 的 get_data/get_ctrl/get_enabled_dependancy_mut（3 个）；_DependancyLayer 的 get_children/get_parents（2 个，get_children 内还有赋值后未用的局部变量 results）；ETFeederNode::get_attr_type（1 个）——上游主驱动接口已被本仓 NodeStore/CustomAlgorithm 直用 getDependancyResolver() 替代。 |
| 38 | `extern/graph_frontend/chakra/src/feeder_v3/et_feeder_node.h:67-71,78-82` + `.cpp:73-92,118-158` | ETFeederNode 零调用接口族：非模板老接口 5 个（num_ops/tensor_loc/tensor_size/comm_type/comm_priority 的一行委托版）+ get_inputs_shapes/get_inputs_types/get_outputs_values/get_outputs_shapes/get_outputs_types。注意同族 is_cpu_op/comm_size/comm_tag/comm_src/comm_dst/runtime/get_inputs_values 及模板版 num_ops 为**活**接口（HardwareResource.cc:64、CustomAlgorithm.cc、NodeStore.cc 消费），删除时勿误伤。 |
| 39 | `extern/graph_frontend/chakra/src/feeder_v3/cache.h:39-81` + `protobuf_util.h:54-89` + `common.h:18` | 缓存/序列化死成员族：Cache::has、get weak 版、get_or_null weak 版、remove（_node_cache 全生命周期仅 put/get_or_null_locked/get_locked 三调用）；writeVarint32/writeMessage 整条写链零调用；NO_IMPLICIT_CONVERSION 死常量零引用。 |
| 40 | `sh_test_mesh/workload/llama2_7b_inference/generate_face_trace.py:546-566,1267-1347` | B2/B3 三态化残留死函数 2 个：`_transfer_dict`（被 `_kv_transfer_dict` 取代，且其访问的 transfer.action/history_tokens/shard.relative_tp_rank 在现行 KVTransfer/KVTransferShard 数据类上已不存在，复用即 AttributeError）；`_emit_prefill_stage`（离线全管线删除后不可达——main() 本身即离线入口拒绝桩，在线 fail-closed 仅支持 request_aggregated；`graph_batch_builder.py:6` docstring 仍列它并声称直接 import——文档失实）。 |
| 41 | `sh_test_mesh/workload/llama2_7b_inference/session_kv_manager.py:376-380,727-728,778-783,43` | 死访问器/死常量 4 实例：SessionKVSnapshot.context_tokens 兼容别名（docstring 称被 manifest/fixture 使用，实测全仓 4 处 .context_tokens 访问均落在 KVCacheEvent/EvictionRecord 同名字段）；node_states property；final_session_counts；RECOMPUTE 常量死 import 链（仅两处 import 且两文件体内零使用；同源 NOC_MIGRATE 有真使用可对照）。 |
| 42 | `sh_test_mesh/workload/llama2_7b_inference/generate_trace.py:241-242`、`traces/derive_20_first_30_seconds.py:84-94` | 零散死函数 2 个：`shard_size`（同文件实际用 shard_extent）；`fnv1a64`（实际用 fnv1a64_cont+初始 basis；注意 C++ 侧 WindowedTraceReader 同名成员为无关同名，勿误判跨语言引用）。 |

### 族 D：只写不读字段/死写/注释失实（9 族）

| # | 位点 | 说明 |
|---|---|---|
| 43 | `astra-sim/system/Sys.cc:247,237` | 只写不读字段族：`stream_priorities` 唯一写点（构造填充）后全仓零读者；连带 `dim_to_break`（唯一写点恒 -1）与 `logical_broken_dims`（全仓零写点）仅有的读者是永不构造的 OfflineGreedy（OfflineGreedy.cc:92/100/103）——双重死。 |
| 44 | `astra-sim/system/BaseStream.hh:53-55,46` | 死字段族：`phase_latencies[10]` 全仓唯一出现即声明；`test`/`test2` 仅 Sys.cc:1393-1394 写 0（上方注释自称 hot fix 遗留）；`BaseStream::initial_data_size`（int）仅 StreamBaseline.cc:24 写——Sys.cc:1269 读的是 my_current_phase（CollectivePhase）的同名字段（另一字段，uint64_t）。 |
| 45 | `astra-sim/system/MyPacket.cc:27-36` | MyPacket 死回调链+死字段：`call` 全仓零调用（全仓 21 处 register_event/try_register_event 实参无一是 MyPacket*；对象只进 PacketBundle::locked_packets 且仅写 ready_time），连带 `notifier` 只写不读（set_notifier 每包调用却永不触发）；`sender` 写点值恒 null（insert_packet 实参全仓恒 nullptr）；`cycles_needed`、`ready_time` 零读者。**删除须连动去虚化**（call 是 Callable 纯虚实现，直接删函数体 MyPacket 变抽象类编译失败）。双桶互证（astraccl 桶的 sender/call 条目并入本族）。 |
| 46 | `astra-sim/system/astraccl/Algorithm.hh:27`、`Ring.hh:35,53`、`HalvingDoubling.hh:36,54`、`BasicLogicalTopology.hh:34`、`Ring.cc:130-131`、`HalvingDoubling.cc:141-142` | astraccl 只写不读字段族：Algorithm::name（4 写点零读）、total_packets_sent（=0/++ 零读）、dimension（零写零读）、basic_topology（构造存值零读）、空语句 `if (id == 0) {}` 两处。face 已清理同位置（亲证 face 的 MyPacket::call 空体、成员已删），本仓保留。 |
| 47 | `astra-sim/workload/execution_driven/GraphSource.hh:115-120` + `WindowedTraceReader.hh:262` | OnlineStatisticsState 只写不读（混合谱系）：`operation_intensity`/`is_memory_bound` 为 face 继承（唯一写点 Workload.cc:601/605；Statistics.cc:893-910 读的是 operator_statistics 的同名字段而非本结构）；`network_bandwidth`（**-LRU 新加回**，唯一"读者"是注释块）与 `WindowedTraceReader::consumed_idx_`（**-LRU 新加回**，仅被自身 watermark 单调推进自读，无外部消费者/输出引用）。 |
| 48 | `astra-sim/network_frontend/analytical/congestion_aware/main_online.cc:154-157,167,1329-1334` | OnlineDriverContext::ingress/systems/expected_requests 三字段只写不读（face §2 同位点仍在）；`expected_requests` 快照时局部变量尚为 0（CSV 行数在主循环 ：1518 才回填、从不回写字段）→ **字段恒 0**；:167 注释 "run-end assertion target" 与事实相反（实际断言读局部变量，:1948/1971/1999/2039）。 |
| 49 | `sh_test_mesh/workload/llama2_7b_inference/online/graph_batch_builder.py:1367` | 死写：`request_plan["_history_before"]=history_after` 全仓零读者（子串 grep 全仓唯一命中即写入行），仅徒增 plan dict 键。 |
| 50 | `sh_test_mesh/slo_tools/slo_stats.py:198,210` | cmd_violation 中 `deadlines: list[int] = []` 逐行 append 后无任何读取，payload 亦不含——只写不读死局部。 |
| 51 | `sh_test_mesh/workload/llama2_7b_inference/online/decision_bridge.py:179-182` | stats() **docstring 失实**：声称返回"含每 request 服务时间行列表(per_request)，供 online_stats.jsonl 合并"，实际只返回 4 个计数器 dict（per_request 由独立的 per_request_stats() 流式接口提供并在 online_service.py:289 merge join）——方法本身是活方法（online_service.py:250 生产调用），死的是注释（face 文档 §2 stats() 死访问器条的 Python 对位，本仓该方法活、唯注释说谎）。 |

### 族 E：死配置/死键/冻结参数（2 族）

| # | 位点 | 说明 |
|---|---|---|
| 52 | resolver/模板死键 3 键（**三桶互证**）：`config_resolver.py:434`（local-mem-capacity-bytes）、`:351`（logical-pool）、`sh_test_mesh/system/llama2_7b_roofline_template.json:11`（boost-mode） | P11 死键族：① `local-mem-capacity-bytes` 写入 system.json，astra-sim/extern 全部 C++ 零读取（唯一消费是两个 python 测试断言 + resolver 自身 _MANAGED_SYSTEM_FIELDS 管理性引用）；② `logical-pool` 写入 remote_memory.json，AnalyticalRemoteMemory 构造仅读 memory-type/num-nodes/num-npus-per-node/npu-ids/remote-mem-latency/remote-mem-bw 六键，未知键被 nlohmann 静默忽略；③ `boost-mode` 活模板与 inputs/ 下 10 个样例均写入（全仓 13 个 json 含该键），Sys::initialize_sys 解析键清单无它、全仓零读取——face 同款死键（face 已清理），本仓保留且进入活模板，属纯配置噪音。其中 local-mem-capacity-bytes/logical-pool 为 face 深挖未登记的新位点。 |
| 53 | `sh_test_mesh/slo_tools/slo_params_manifest.json` | 21 个冻结参数中 **11 键全仓零消费者**：epsilon、rolling_window_ns、rolling_step_ns、scan_initial_bracket、scan_convergence_width、scan_max_points、scan_repeats、wp6_overhead_budget、wp9_drift_gate、wp9_graphbatch_budget、wp9_oracle_gate（对应消费方 campaign scan_driver、rolling P99 附录、WP6/WP9 门控脚本不在本仓）；其余 10 键（alpha_*、bucket_percentiles、imbalance_bucket_ns、warmup_*、watermark_*、link_bucket_ns）有真实消费。face 同数量同族（face 修复轮评估后按"冻结留证数据"保留）。 |

### 族 F：孤儿测试/遗留残片（5 族）

| # | 位点 | 说明 |
|---|---|---|
| 54 | `astra-sim/workload/execution_driven/tests/{cli_online_test.cc:58,event_queue_deferred_test.cc:24,ingress_idle_fixture.cc:46}` | 孤儿测试三件套（face M12②）：未入任何构建（analytical CMakeLists 共 23 个 add_executable 均不含三者，全部 file(GLOB) 非递归不含 tests/），仅头注 g++ 手工可达——R1-R16 CLI 契约、EventQueue deferred 通道、IDLE 五态生命周期均无门禁保护。三处头注 Build 行仍写 `template/astra-sim-wscllm`——仓根名失实（本仓根为 astra-sim-face-LRU）。face 已接入构建，本仓未同步。**双桶互证**。 |
| 55 | `astra-sim/workload/execution_driven/ServiceCoordinator.hh:79-82` + `.cc:51-61` | fence 族死写点（face M12③）：`on_fence_scheduled`/`on_fence_resolved` 全仓零调用（仅声明+定义 4 行命中）→ `pending_fence_count_` 恒 0，watchdog 停滞诊断（main_online.cc:1557-1582，:1749 超时 fail-closed abort）的 pending_fence 字段**永远打印 0**（诊断失真）；hh 注释自述 "no behavior yet" 属预留占位（NodeStore.hh:26 注释明示计划禁止提前建通用 fence）。face 已删，本仓为 -LRU 从 sh_2.0 谱系**新加回**。**双桶互证**。 |
| 56 | `sh_test_mesh/slo_tools/*.bak_caliberfix_20260905`（5 文件） | README.md、slo_common.py、slo_params_manifest.json、slo_stats.py、tests/test_slo_contract.py 的备份残片：被 git 跟踪、全仓零引用、.bak 后缀不可被 import/构建选中——face 遗留残片（face 修复已删），本仓原样保留。 |
| 57 | `inputs/system/analytical/`（10 个上游样例 json） | tpu_v3_8/tpu_v3_32_2dtorus/tpu_v3_32_ring/dgx_v100_4gpu/dgx_v100_8gpu/hgx_h100_{2,4,8,16,32}gpu.json 全仓零引用（10 个文件名 stem 逐一 grep 零命中）——face 遗留残片族延续（face 已删整目录），与 README "裸仓两条在线路线" 的最小仓声明存在张力。 |
| 58 | 期望完成数注释矛盾 + 头注根目录失实：`tests/make_completion_fixture_et.py:36-37,184`、`tests/completion_observer_fixture_main.cc:18,214-216,222` 等 | 三方口径矛盾（代码一致、注释三处两错）：生成器注释写 "54 ranks × 3 = 162"，fixture 注释写 "54 × 5 = 270"，而两边代码实际均为 `npus*4+26=242`（:184 公式与 :222 断言一致）——162/270 均为改造前残留。另 11 处测试头注 Run/Build 根目录失实：写 `template/astra-sim-face` 的 bridge_loopback_fixture.cc:37、graph_batch_committer_test.cc:90、local_hbm_bandwidth_model_test.cc:65；写 `template/astra-sim-wscllm` 的 decision_mailbox_test.cc:23、watch_registry_test.cc:36、node_store_test.cc:35、windowed_trace_reader_test.cc:60、completion_observer_fixture_main.cc:22 及孤儿三件套（见 #54），本仓根实为 astra-sim-face-LRU。 |

### 族 G：补充登记（1 族）

| # | 位点 | 说明 |
|---|---|---|
| 59 | `astra-sim/system/RecvPacketEventHandlerData.hh:32`/`.cc:28`、`system/Common.hh:33,35,40` + `common/Common.hh`（双副本） | "只写不读字段族"深挖发现经复核部分驳回后**确认仍死的 4 字段**（见 §4 驳回 2）：① `RecvPacketEventHandlerData::message_end`（.hh:32 声明、.cc:28 写 true，全仓零读）；② `sim_request::layerNum`（双副本 Common.hh:35，全仓零读写）；③ `sim_request::reqCount`（双副本 Common.hh:33，Sys.cc:1495/1523 写、仓内零读）；④ `MetaData::timestamp`（双副本 Common.hh:40，零读写）。同族发现中的 `WorkloadLayerHandlerData::sys_id`/`workload` 两字段为**活字段**（AnalyticalRemoteMemory.cc:162/207/209、LocalHbmBandwidthModel.cc:348 有生产读点），删除时勿随删。 |

**清册边界说明（bug 伴生死码，不入 59 族计数、施工时须一并处置）**：① `astra-sim/system/scheduling/OfflineGreedy.cc+hh` 共 495 行（含 OfflineGreedyScheduleJournal 整个 rendezvous 记账机制）——死因是 §1-M5 的钉死枚举（症状级死码），修复 M5 时须二选一（补配置解析接线，或整体删除）；② `SharedBusStat` 的 `total_shared_bus_*`/`total_mem_bus_*` 八字段——唯一写点 MemMovRequest.cc:42-49、零消费零输出点（§1-M7 复核确认），属统计链死存储。仅按上方 59 族清册施工会漏删这两处。

---

## 3. 与 face 仓错误历史对照（face 86 条继承核对）

**前提**：face 深挖文档确认 86 条（high 3 / medium 12 / low 71），且 face 已于 2026-09-23 完成两轮修复（严重错误全量修复、死代码全量清除、孤儿测试接入；未 commit 的工作树还删了 MetricCollector Phase-0 死面）。本仓 face-LRU 基线**未同步**该修复，且部分位点被 -LRU 改造主动改出新形态。继承核对以**本仓亲读实码**为准，不沿用 face 文档行号——face 文档自身已有实锤的一处记载与 face 现码对不上（HardwareResource 恒假 `==0` 位点，face 文档记 face :123-125，face 现码该处无此赘枝；本仓实存，见 §2-11）。

### 3.1 face high/medium 15 条逐条对照

| face 条目 | face 修复后现状 | 本仓处置 | 说明 / 本仓位点 |
|---|---|---|---|
| H1 double free/UAF | 已修（clone_collective_impl 保型克隆） | **本仓回归** | CommunicatorGroup.cc:129-149 无 clone（grep 0 命中 vs face 2 命中，本会话亲验）→ 本仓 H1（§1.1） |
| H2 换算 vs 测试参照 | 已修（fixture 改 SI 自洽） | **演化仍在，失配面扩大** | 本仓实现同为 SI 恒等，但失配 fixture 从 2 个 fluid 测试扩至 3 个（新增常规构建目标 local_hbm_bandwidth_model_test --scenario slow）→ 本仓 H2（§1.1） |
| H3 示例脚本必失败 | 已删 3 个失效脚本 | **半修复（脚本仍在）** | 坏 build.sh/坏二进制名仍在三脚本中（且先死于缺 --online-mode）；`--remote-memory-configuration` 被吞半已修复（CmdLineParser.cc:25）→ 本仓 H3（§1.1） |
| M1 缺键除零 | 已修 | **仍在** | Sys.cc:1186（face :1165 → 本仓行号平移）→ §1-M2 |
| M2 未初始化枚举 | 已修 | **仍在** | Sys.hh:348 → §1-M3 |
| M3 未初始化枚举 | 已修 | **仍在** | Sys.hh:381 → §1-M4（后果修正为调度/时延失真） |
| M4 int 截断 | 已修 | **仍在（触发面微调）** | SimSendCaller.hh:18；delay>0 分支潜伏，实际可达点为 rendezvous 路径 → §1-M8 |
| M5 钉死枚举 | 已修（补齐配置解析+登记开关清单） | **仍在（双桶互证）** | Sys.cc:210；OfflineGreedy 495 行连带死码 → §1-M5 |
| M6 NaN 污染 | 已修（含补漏 NetworkStat.hh 守卫） | **仍在（双位点）** | SharedBusStat.hh:136-146 主位点 + Sys.cc:1352 姊妹点（face 已加零守卫、本仓未同步）→ §1-M7/M6 |
| M7 ETFeeder 泄漏 | 已修 | **本仓回归** | face 的 ~CustomAlgorithm+delete 被 diff 实证删除 → §1-M9 |
| M8 非 2 幂段错误 | 已修 | **本仓回归** | face 守卫块（:18-26）被 diff 实证删除 → §1-M10 |
| M9 roofline NaN/Inf | 已修 | **本仓回归** | face 两道防线（前置 exit + num_ops 守卫）本仓皆无 → §1-M12 |
| M10 窄化在前 | 已修（宽域先校验） | **本仓回归** | MetricCollector.cc:511-517 回退成窄化在前，且同面丢 3 处负值防护 → §1-M13 |
| M11 link id 估算失配 | 已修 | **仍在** | main_online.cc:757-762 同位点 → §1-M17 |
| M12 测试域 3 条 | ①臂改真无界 ②三测试接入 ③fence 删除 | **①②仍在、③且为新加回** | calendar_reader_oracle legacy 臂（§1-M15）；孤儿三件套（§2-54）；fence 族本仓为 -LRU 新加回（§2-55） |

小结：15 条中 **face 已修而本仓回归 5**（H1/M7/M8/M9/M10）、**仍在 9**（H3 半修复、M1-M6、M11、M12①②）、**演化出新形态 1**（H2）。

### 3.2 face low 71 条（死代码，按族）对照 → §2 全部 59 族闭环

face 的 low 71 按**实例**计数、本仓 §2 按**族**归并（59 族），粒度不同故数目不直接对齐；数量差的来源：族内归并、face 修复已清项在本仓回归、以及 -LRU 改造新增族在 face 无对应条目。以下七类处置**逐族覆盖 §2 全部 59 族**（每族恰出现一次），71→59 核对闭环：

- **① 仍在（face 同位点同码 / face 文档同登记 / face 同样零接线，20 族）**：§2-2、§2-8（6 个孤儿 fixture 经对照确认 face 同样零接线）、§2-9、§2-10、§2-17、§2-20、§2-22、§2-23、§2-26、§2-32、§2-34、§2-35、§2-36、§2-37、§2-38、§2-39、§2-43、§2-44、§2-48、§2-53（face 评估后亦保留留证）。其中 9、10、17、23、36、37、38、39、43、44 诸族另有桶级 diff 全 SAME 佐证。
- **② face 已清、本仓回归保留（13 族）**：§2-3、§2-6、§2-7、§2-12、§2-25、§2-27、§2-33、§2-45、§2-46、§2-54、§2-56、§2-57、§2-59。注：25、27 两族系 face 已提交基线含、face 修复删除尚未提交（其中 27 的成员④为活函数注释失实，非死码本体）；59 的 message_end 经 face §5.2 清单实证已删，其余 3 字段 face 清单未单列；45、46 经亲证 face 同位置已清理。
- **③ -LRU 新增 / 自 sh_2.0 谱系新加回 / B2-B3 改造残片（11 族）**：§2-1（-LRU 新增文件）、§2-13、§2-14、§2-15、§2-24、§2-29、§2-30、§2-31、§2-40、§2-41、§2-55。
- **④ 混合族（逐成员拆分，3 族）**：§2-28（①retire_online_operator=回归保留、②network_bandwidth=-LRU 新增、③get_operator_statistics()=仍在）；§2-47（operation_intensity/is_memory_bound=仍在、network_bandwidth+consumed_idx_=-LRU 新增）；§2-52（boost-mode=回归保留、local-mem-capacity-bytes/logical-pool=face 未登记的本仓新位点）。
- **⑤ face 无对照结论（sh_test_mesh python 域未被 face 深挖覆盖、或 face 文档未登记，本仓新登记，7 族）**：§2-5、§2-16、§2-18、§2-19、§2-42、§2-49、§2-50。
- **⑥ 特殊登记（3 族）**：§2-11（face 文档记载与 face 现码对不上——face 文档记 face :123-125 有恒假 `==0`，face 现码无；本赘枝在本仓实存且为 face-LRU/joint/wscllm/wscllm-LRU 四仓共有）；§2-51（对位演化：face 对位条目是 C++ FileDecisionBridge::stats() 死访问器，本仓 Python 对位方法活、唯 docstring 失实）；§2-58（改造残片级注释/头注失实，face 文档无单列对照）。
- **⑦ face 同族的本仓新位点（2 族）**：§2-4、§2-21（CSVWriter–UsageTracker 死报告链：face 文档 §2 有"死函数/死访问器族"，但此具体链为 face 深挖未单列的本仓新发现位点）。

### 3.3 本仓新形态（face 无、-LRU 改造引入或首次报告）

- **错误类**：M1（AnalyticalRemoteMemory PER_NODE 缺键除零——远端内存后端恢复接线的伴生雷）；M11（kv-restore-sharing 绕过降级 + peak_perf 构造期 terminate）；M14（get_operator_type 漏 METADATA_NODE——face 同码但 face 深挖未载，本仓首次报告）；M20（hopbytes collect_face 未适配契约行列表）；L9（拆分路径 drain_block_ends 断链）；H2 位点①（local_hbm slow fixture 失配）。
- **死代码类**：RemoteFifoLedger sensing 账本整族未接线 + hh:52 注释失实（§2-1）；architecture_name/port_mapping_rule 死 export（§2-24）；MetricCollector/Statistics 的 -LRU 死面（§2-27/28，其中 Phase-0 实为继承 face 已提交基线）。

---

## 4. 覆盖与可信度

- **覆盖**：15 个模块分桶，清单 320 文件全覆盖（glob 预验证无空桶），另含 1 件任务点名补审的清单外文件（astra-sim/common/AstraRemoteMemoryAPI.hh，逐行读完），逐行精读合计 321 件；**未分桶残留 0（应为 0）**。桶间交叠位点 9 组按唯一缺陷归并计数并标注互证（H1、H2、H3、face M5、孤儿测试三件套、fence 族、P11 死键、AstraSimDataAPI 双副本、retire/flush 死函数）。
- **独立复核口径**：每桶深挖发现全部经独立复核员全新上下文裁定，仅认亲读代码：verdict=confirmed 94 条（本文档 §1/§2）、rejected 4 条（见下方驳回留档）；复核修正了原发现多处行号/机理/谱系偏差（文中已随条标注）。死代码判定执行 P14 纪律（第二人全形式 grep：含 tests/、CMake、.py/.sh/.json/.md、宏/字符串/取地址形态，排除姊妹仓引用混判）。
- **P14 排除记录（不计入死代码族，避免误报）**：1 条 verdict=confirmed 的复核记录实为排除项——`ChunkIdGenerator::size()`（被 chunk_id_generator_test.cc:32-49 消费且目标入 analytical/CMakeLists.txt:30-33）、`set_fluid_scheduler`（main_online.cc:1044 + 5 个测试消费）、`AstraRemoteMemoryAPI`（AnalyticalRemoteMemory 实现并经 Sys.cc:183 set_sys、Workload.cc:461 issue 真实调用）均查实有真实消费者，非死代码。
- **构建基线**：**未取得**——构建链异常中止（`WorkflowError: world.run 'bash' timed out after 900000ms`），本仓全程未构建、未 ctest、未跑任何仿真/脚本。因此：全部"必失败/崩溃/NaN"结论为静态推演（除零点、无符号回绕链、口径逐段重推等证据链已闭合，但**未经实测复现**）；face 文档中的实测数据（face H2 exit 1、face M8 exit 139、face M12① 的 oracle legacy=43 vs calendar=200）仅作旁证引用，本仓未复跑；本会话仅做的独立验证是静态抽查（H1 clone 符号两仓 grep、Sys.cc:209-210、AnalyticalRemoteMemory.cc:58-61/127-131/168-172、NetworkFunction.cpp:11-19、示例脚本 :17/:28——均与发现一致）。
- **驳回留档（4 条，各一句话）**：
  1. *remote-mem-bw (0,1) 浮点穿透除零*——构造函数 :127-131 的第二层校验（非 NO_MEMORY_EXPANSION 且 remote_mem_bw==0 → exit(1)）在除法点 :275 之前干净拦截，真实最坏后果仅是 <1 GB/s 的语义合法配置被拒（过严校验，类型不一致实存但降为 low 以下）。
  2. *RecvPacketEventHandlerData 只写不读字段族（6 字段）*——sys_id 与 workload 被亲读到的生产读点直接推翻（AnalyticalRemoteMemory.cc:162/207/209、LocalHbmBandwidthModel.cc:348），按原发现删除将直接编译失败；真正死字段仅 message_end/layerNum/reqCount/timestamp 4 个，**已补录为 §2-59**（同族 sys_id/workload 为活字段的结论随条留档）。
  3. *LogGP size==0 负延迟回绕*（face 可疑点，本仓曾试图升级定论）——G 计算路径唯一入口需显式配置 `model-shared-bus:1`（全仓配置零命中）且缺省 G=0；即便人为开启，-0.0038 向零截断为 0 非 UB（[conv.fpint]），真回绕需再叠加配置 "G"≥1，属三层人为叠加的条件性缺陷，降为防御性编码建议。
  4. *Event 整类零引用*——`CallbackTrackerEntry.cc:22/31` 真实实例化 Event（`std::optional<Event>` 成员）且在在线主目标 AstraSim_Analytical_Congestion_Aware_Online 的回调跟踪链上，删除即破坏构建；其附带的 `Type.h:15 class Chunk;` 前向声明零定义零使用属实，可独立清理。
- **未覆盖项**：extern/helper 三方库与上游文件未审（仅定制甄别）；et_def.pb.h protobuf schema 未审；运行期行为/长跑稳定性/多线程未实测（纪律禁止）；Python 侧策略对 kv_actions/assignments 的内部消费语义未审（仅核对 C++ 契约与发射键集匹配）；sh_test_mesh 各 run 链路行为未实测（仅静态路径检查）；gitignored 生成配置（comm_group.json、npu-ids 等）无法仓内核对；CMake 目标级编入审计仅部分桶完成；face 仓仅 diff 对照、未独立审。

---

## 5. 可疑点登记存档（154 条 = 146 未定论 + 7 跨桶指引 + 1 已澄清勘误，均不计入确认计数）

标 **【跨桶指引】** 者为已确认结论的跨桶登记（非未定论，指回 §1/§2 对应条目）；标 **【已澄清】** 者为深挖过程中的早期怀疑、已被独立复核亲读推翻（保留勘误留档）；其余 146 条为未定论可疑点。

### 5.1 LRU 三态 KV 与远端内存核心（11 条 = 10 未定论 + 1 跨桶指引）
- `AnalyticalRemoteMemory.cc:112` remote-mem-latency 无 nonnegative/类型校验：手写负数回绕 uint64 成巨大延迟（时延失真不崩溃；resolver 侧有 nonnegative 兜底）。
- `AnalyticalRemoteMemory.cc:165-168` NO 模式 issue() 硬 exit(1)，Workload.cc:369-377 对非 kv-restore 的 MEM_LOAD/STORE 无条件走 issue_remote_mem，三个现成运行脚本全传 no_memory_expansion.json——workload 图含普通 MEM 节点即中止仿真（现成链路规避了该形态）。
- `AnalyticalRemoteMemory.cc:207` `sys_map[wlhd->sys_id]` 用 operator[]：未注册 id 插入 nullptr 即解引用，安全性全靠 main_online.cc:1115→Sys.cc:183 构造期注册的隐式时序。
- `RemoteFifoLedger.cc:35-38` in_flight 用 issued-completed 无符号差值无防御：enabled_ 中途翻转/记账乱序将 uint64 回绕成巨大 peak 值（当前生产无翻转路径）。
- `AnalyticalRemoteMemory.hh:78` 注释 "GB/sec" 与实际语义 B/ns 仅靠 SI 恒等等价（face H2/P6 高危区）；P6 核对结论：测试用例 1000-4000B→60-90ns 全为整除，与 cc:273-277 实现口径一致，未发现 H2 同款错。
- `config_resolver.py:272-273,330` hardware 源 "bandwidth-gbps" 数值直透 remote-mem-bw：源键 gbps（bit 口径?）与 C++ 消费的 B/ns（byte 口径）是否差 8 倍无法仓内裁定，需 hardware 配置文档佐证。
- **【跨桶指引】** `CmdLineParser.cc:62` get_options() 零调用（face 同名死位点；已入 §2-33）。
- `main_online.cc:536,1338,1441` sensing_enabled 只覆盖 NodeStore injected-unfinished summary（phase-3），与 RemoteFifoLedger 无关；CLI 旗标 --sensing-enabled 与账本头注释声称的启用路径撞名，极易误读为账本已接线。
- `AnalyticalRemoteMemory.cc:104` PER_NPU 的 port=npu-ids 数组索引（rank3→port0），测试注释自称 "key-label defect" 并固化该语义；与 ledger 注释一致故非错，但 port 号≠rank 号易误读。
- `AnalyticalRemoteMemory.cc:275` double 除法结果 static_cast<uint64_t> 向下截断小数 ns（40.96→40）：测试刻意用整除用例绕开，非整除 tensor_size 时延一律向下截断。
- `AnalyticalRemoteMemory.hh:82-83` num_nodes/num_npus_per_node 无默认初始化（face M2/M3 同款形态），当前仅 PER_NODE 分支先赋 0 再覆盖、其他模式不读——潜在 UB 引信。

### 5.2 system 主干 Sys（12 条 = 10 未定论 + 1 跨桶指引 + 1 已澄清）
- `Sys.cc:1186+845` All_Gather 且 size < splits 时 chunk_size=0（:1190 下限校正显式排除 All_Gather），:845 ceil(size/0)=inf 转 int UB，:930-940 size-=0 可致死循环（face 同位点可疑项）。
- `Sys.cc:1306-1317` ask_for_schedule 对 all_sys 元素直接解引用无 nullptr 守卫；~Sys 置 all_sys[id]=nullptr 后多 Sys 场景即空指针；boostedTick(:511-524) 全空时 ts=nullptr 同族。
- `Sys.cc:179+469-474` peak-perf 缺省 0，roofline-enabled 单独开启即以 0 上限构造 Roofline，下游 Workload.cc:534 除零——face M9 的 Sys 侧输入面（M9 本体见 §1-M12）。
- `Sys.cc:693-700` handleEvent RendezvousSend/Recv 分支直接解引用 ehd、无 CallEvents 分支那样的存活检查：Sys 销毁后端仍投递 rendezvous 完成为 UAF（上游遗留，现有 teardown 顺序下未找到可达路径）。
- `Sys.cc:581-648` register_event_cancellable/cancel_event 生产零调用，仅 alarm_cancellation_test.cc 消费（测试在构建内的说法未验证——未跑 CMake，故按 P14 不报死代码）。
- `Sys.cc:199-200,222-223` communication_delay/local_reduction_delay 先赋 0 又立即被 10/1 覆盖，前两次写为死存储（上游遗留清理项）。
- `Sys.cc:252-253` concurrent_streams=ceil(active_chunks/queues_per_dim[0])：空向量或首元素 0 时 UB/inf（外部输入形状无守卫）。
- **【已澄清】** `MemEventHandlerData.cc:10` 构造与 workload/wlhd 初始化——深挖早期怀疑"构造不初始化两个裸指针、接线即读未初始化"，经独立复核亲读推翻：构造函数（.cc:10-13）显式置 nullptr 且 Sys.cc:705 有判空，接线亦为安全 no-op（纯死代码，定论见 §2-9；此处保留为勘误登记）。
- `AnalyticalRemoteMemory.cc:211` start_request 用 sys_map[wlhd->sys_id]（operator[]），未注册 id 插 nullptr 后 register_event 即空指针（extern 属他桶，仅登记）。
- **【跨桶指引】** `inputs/` 整目录全仓零引用（配置桶登记，见 §2-57）。
- `Sys.cc:880-883` custom collective 路径 stream_id 先 num_streams++ 再被覆盖，前者为无效自增（无行为影响）。
- `AnalyticalRemoteMemory::sys_map` 在 ~Sys 后保留悬垂 Sys*（~Sys 不回调 unset_sys）；现有路径下未找到解引用时机，仅登记。

### 5.3 system 事件与数据组件（9 条 = 8 未定论 + 1 跨桶指引）
- `BasicEventHandlerData.cc:10-12` 默认构造只初始化 sys_id 不初始化 event：WorkloadLayerHandlerData 的基类 sys_id/event 与自身 workload 三字段恒未初始化，靠"用前必补写/永不读"约定维持（Workload.cc:418-435 issue_replay 三字段全未写），新增任一读点即 UB。
- `SharedBusStat.hh:52-73` update_bus_stats 指针形参三分支均直接解引用无空守卫；现链 data 均非空，但仓内存在 MyPacket.cc:34 传 nullptr 通知 Callable 的先例。
- **【跨桶指引】** `Sys.cc:1352` net_message_latency 0/0（Sys.cc 桶位点，已确认为 §1-M6）。
- `QueueLevels.cc:36/41/46` get_next_queue_at_level* 对 levels[level] 无越界守卫，level 来自 dim_mapper 映射，越界即 UB。
- `QueueLevelHandler.cc:50,64-65` 队列数 1/2 时 first/last allocator 落入同一队列（size()/2 回绕），语义存疑（上游如此）。
- `Sys.cc:691-692` NPU_to_MA/MA_to_NPU 死分支不 delete ehd（§2-9），删除时需一并处理内存语义。
- `DataSet.cc:70` call 将任意 CallData* 盲转 StreamStat*，契约全靠调用方保证，未来以 IntData 触发即类型混淆读越界。
- `DataSet.cc:48-51` notify_stream_finished 在 data==nullptr 时跳过统计但 finished_streams 照增 → take_stream_stats_average 的 tick/=counter 将 0 除；现生产唯一通知点传非空。
- `CSVWriter.cc:121-129` finalize_csv 的 compare（uint64_t）在首维列表先耗尽时未初始化即被 assert 读取（Debug 下 UB；方法本身已死不可达）。

### 5.4 system 通信与流组件（8 条，均未定论）
- `CommunicatorGroup.cc:123` 全节点分支 dimensions_involved 硬编码 (10,true)：physical_dims>10 时 vector<bool> 越界 UB（现实 2-3 维，低风险）。
- `CollectivePhase.cc:16-20` 构造 enabled=true 随后立即被 algorithm->enabled 覆盖，首行死写。
- `Ring.cc:157/251`、`PacketBundle.cc:67` 写 locked_packets 指针存在时序窗口，疑似 UAF 写已析构 MyPacket 的 ready_time（事件时序未完整验证，且 ready_time 本身无读者）。
- `Sys.cc:845` All_Gather chunk==0 → inf → int UB，与 size==0 同根（Sys.cc 不在本桶）。
- `OfflineGreedy.cc:132-141` dim_BW[0]==0 时 double 除零 → inf → uint64 隐式转换 UB（当前在死分支内，钉死枚举被修复后才可达）。
- `OfflineGreedy.cc:216-219,311-315` lastIndex 下扫循环无显式下界检查，安全性依赖"当前 dim 必为 involved 且 size>1"的隐式守卫。
- `CommunicatorGroup.cc:78` num_streams=id*1000000 以 int 承载，comm_group_id 极大时溢出（id 来自配置键名 stoi，低风险）。
- `LogGP.cc:149/241/264` (size/100)*local_reduction_delay 整除截断：<100B 消息处理延迟归零只剩常数（疑似设计近似，未定论）。

### 5.5 astraccl 集合通信（10 条 = 9 未定论 + 1 跨桶指引）
- `HalvingDoubling.cc:90` `log2(nodes)-1*parallel_reduce` 运算符优先级使语义为 log2(N)-parallel_reduce 而非注释意图；parallel_reduce 唯一写点钉死 1 恰好无害，一旦 >1 即错。
- HalvingDoubling 对非 2 幂不崩溃但 log2 截断（N=6→2）：stream_count/包数少算、延迟失真（face M8 守卫在 BinaryTree 层，此层仍无守卫）。
- `CollectiveImplLookup.cc:31/37` stoi(substr) 接受尾随垃圾（"direct12abc"→12）、超 5 位窗口被截断，均无校验。
- `RingTopology.cc:81/125/143` assert 在 NDEBUG(Release) 失效，未知 node_id 经 operator[] 静默返回错误节点而非报错。
- `GeneralComplexTopology.cc:133-147` get_num_of_nodes_in_dimension 越界仅 critical+assert（Release 无效）后仍越界索引；get_basic_topology_at_dimension 对 dimension 无界检。
- **【跨桶指引】** `Sys.cc:225` lookup 泄漏（已确认为 §1-L2）。
- `DoubleBinaryTreeAllReduce.cc:169,174` snd_req2.dstRank=left_child 与实际发送 dst=right_child 不一致；dstRank 全仓零读点无行为影响，属怪味死赋值。
- `CollectiveImplLookup.cc:228-230` BYPASS_ALL_CUSTOM 且缺实现键返回空 vector → GeneralComplexTopology.cc:26 size()-1 在 uint64 回绕 SIZE_MAX（当前空向量时循环不执行、后续抛 runtime_error，尚无实害）。
- `CustomAlgorithm.cc:26` 仅捕 runtime_error，ETFeeder 构造若抛其他异常类型（谱系未验证）将直接 std::terminate。
- `Ring.cc:73/80-85` 单节点环（total=1）msg_size=data_size 且 stream_count=0 的退化路径未逐分支验证终止性。

### 5.6 workload 执行与资源（9 条，均未定论）
- `HardwareResource.cc:184-189` 注释称在线 COMP 链并发、gate 被 bypass，但 Workload.cc:313 对在线节点照常 is_available 且 Compute 要求 in-flight==0 → 相互独立的在线 COMP 节点实际串行，与 count-based occupy 及测试并发占位期望矛盾——要么注释失实要么缺 bypass。
- `HardwareResource.hh:56-59` 注释仍引用 "replay bypass in Workload::issue_dep_free_nodes"，而 path-2 removal 已删 replay 路线——注释依据失实。
- `AnalyticalRemoteMemory.cc:124` 分数 GB/s 截断为 0 后被 :127 以 "must be positive" 拒绝——fail-closed 但报错误导（值本是正的）。
- `LocalMemUsageTracker.cc:197-200` 同一 tensor 二次写 assert(false)：Release 静默丢弃第二次写、Debug 直接 abort，双侧均无明确告警路径。
- `LocalMemUsageTracker.cc:98/157` 函数级 static IOinfos 跨所有 rank/实例共享——单线程无害，引入多线程即数据竞争。
- `Workload.cc:228-231` comm_group 文件缺失/不可读时 inFile>>j 抛 nlohmann 异常未捕获 → terminate（仅路径含 'empty' 特判跳过）。
- `Workload.cc:235/1241` std::stoi(comm_group_name/pg_name) 非数字键抛 invalid_argument 未捕获 → terminate（issue_pytorch_pg_metadata 内同款有 try/catch，这两处没有）。
- track-local-mem + 本仓新增 restore/MEM 节点：tracker 对每完成节点强制要求 inputs/outputs 属性（LocalMemUsageTracker.cc:102-106,161-164 抛异常），LRU 生成的 restore 节点是否总携带两属性无法静态确认。
- remote-mem-bw 双解析口径：Sys.cc:429-430 乘 1e9（秒域）与 AnalyticalRemoteMemory.hh:78/:274-277（B/ns）现行数值一致，但换算各自独立实现、无共享常量，任一侧改动即漂移（H2 族隐患）。

### 5.7 workload 指标统计（10 条，均未定论）
- WP8 直排积分不夹负（MetricCollector.cc:2075-2086）vs 水位线步进夹 0（:2139-2142）：账本中间值出现负时两口径系统性分叉，:2528-2548 的 crosscheck 会误报 mismatch>1%（face 可疑点继承，diff 此区零改动）。
- `MetricCollector.cc:2403-2405,2461-2467` emit_watermark_records 在 sim_end_tick==0 时首桶==末桶连发两条相同 hbm_watermark 记录（未去重）。
- `MetricCollector.cc:415` slo[key].get<int64_t>() 对 >INT64_MAX 无符号数值回绕为负后被 <=0 拦截回退默认锚——结果安全但属先窄化后判定的弱实例。
- `MetricCollector.cc:466-468,918-941` arrival 的 interval_ns/parent_queue_index 无负值校验、parent 可自引用（resolve 不递归无栈风险，自引用静默 unresolved）。
- `Statistics.hh:238` comp_comm_overlap 无成员初始化，正确性仅靠 post_processing() 先于 report() 的现调用约定（Workload.cc:1206-1212 满足）；绕过即读未初始化 Tick。
- `Statistics.hh:121` get_type_time 生产零调用，唯一消费者是 built 测试——非死码，生产面闲置。
- `Statistics.hh:70` comm_size 唯一读者 record_network_bandwidth 的产出 network_bandwidth 又只写不读——整条链最终无生产消费（与 §2-28 关联）。
- `MetricCollector.cc:1861-1867` i128_to_long_double 在 value==INT128_MIN 时 -value 理论 UB（byte*ns 面积不可达，纯理论）。
- 跨桶提示：Workload.cc:549-551 读 sys->remote_mem_bw——face 文档 P11 称该键为死键，本仓有 C++ 读者（本仓多桶复核确认：该键已消除，见 §3.2）。
- `Statistics.cc:436-505` OnlineExactDoubleSum::value() 最高位腿搜索下标推演亲读无越界；与 face 同款零改动，存疑排除记录。

### 5.8 execution_driven 主干（12 条，均未定论）
- `WindowedTraceReader.cc:200-205,285,304` CSV 数值列 stoi/stoull 直析，畸形 token 抛未捕获异常 → terminate，绕过 reader_fatal 通道（结构性违规有守卫、类型畸形没有；官方输入经物化器生成，风险低）。
- `ServiceCoordinator.cc:248-252` input-open 全排空回 IDLE 分支生产不可达（注释自认官方 runner 启动即 close），其契约 fixture 又是孤儿——五态迁移 IDLE 回环无可运行验证。
- `ParsedGraphBatch.cc:366-385` watch members 键 stoi 接受 "007"/"-1" 形状；num_ranks<0（fixture 模式）时负 rank 逃过域检查。
- `GraphBatchCommitter.cc:268,397` 以 json_id==UINT64_MAX 作哨兵，合法恰等于 UINT64_MAX 的 id 被误拒（record_affine_node 同值也 throw，口径一致，生成器不可达）。
- `NodeStore.cc:98,121` finish_node 的 assert 在 Release 失效，机制不一致 + --count 回绕会让子节点永不释放（依赖上游 validate，防御弱）。
- `WatchRegistry.cc:26-31` 重复 identity 注册静默返回旧 id、不合并新成员集/新状态——契约如此，调用方二次注册不同成员集被静默忽略。
- `DecisionBridge.cc:494` value("schema_version",-1) 以 int 提取，超大无符号字面量转换是实现定义路径（生产者受控）。
- `NodeStore.cc:511-512` view_of 尾部 default 注释仍列 MemLoad/MemStore，但两者已在 case 2/3 处理——注释失实（本仓还恢复了 issue_remote_mem 路径，与 face 的 removed 注释分化）。
- `DecisionMailbox.cc:52-54` counters_.other 仅非法 terminal_status 触发，正常全路径不可达（纯自检位，无消费门）。
- `WindowedTraceReader.cc:153` prov_.session_blocks_contiguous 恒 true（非连续即中途 abort），sidecar 对该键的校验因此无信息量——字段冗余。
- `main_online.cc:1685` 运行端诊断用 GraphSource::lookup 按值拷贝整个 OnlineNode（含多个 std::string）——仅诊断路径，非错误。
- `OnlineCli.cc:108-109` 非 '--' 前缀 token 一律 continue 静默忽略（位置参数），未来加位置参数语义会静默吞掉。

### 5.9 C++ 测试与构建域（12 条，均未定论）
- `windowed_trace_reader_test.cc:646` `expect(true, "fixture written")` 恒真占位断言。
- `event_queue_deferred_test.cc:423` `std::_Exit(reached_end ? 0 : 0)` 两分支恒等（父进程靠 WIFSIGNALED 判定，表达式无意义）。
- `watch_registry_test.cc:99-104` 注释称 unknown identity no-op 但实际发的是 watch 成员键 key(0,1,1)；断言 stale_count==2 仍成立，动作与注释不符。
- `tests/run_all.sh:9,12` 名为 "Running all regression tests" 实际只跑 1 个 python 文件（test_face_scheduler.py），且 run_all.sh 全仓零引用（纯手动入口）。
- `make_completion_fixture_et.py:78-81` 的 26 边缘 rank 集与 fixture main 硬编码 26 靠人工同步（生成配置 gitignored，仓内无法核对 npu-ids）。
- `local_hbm_model_test.cc:253` 注释 "1e6 ops at 1e6 ops/ns → 1 ns" 与配置 peak-perf=1000 直解（1000ns）不符；断言只绑 HBM 侧 ~3100 未受影响。
- `remote_fifo_ledger_test.cc:310-322` 等在线 fixture new Sys 后从不 delete（测试进程退出语义，无实际后果）。
- `metric_collector_anchor_spool_test.cc:476` fail-closed 契约依赖产品 clear_static_node_events 失败时 exit 非 0 而非 abort；实现改为 abort 则 :473 WIFEXITED 断言反目。
- remote_fifo_ledger_test 闭式参照（60/130/210/300）依赖 Workload 在线 MEM 终端==端口完成时刻；runtime 公式已对上 AnalyticalRemoteMemory.cc:273-277，Workload MEM 节点路径在别桶未逐行核。
- local_hbm_bandwidth_model_test fast 场景期望 146 依赖"静态单 comm slot 在 join 节点完成(125)而非网络侧完成时释放"的调度细节（HardwareResource.hh 未逐行核）。
- `LegacyOracleWindowedTraceReader.cc:5` 头注仍写 "wscllm phase 7 §10.4"（跨仓残留）。
- `local_hbm_bandwidth_model_test.cc:213-240` 无默认值共享选项靠 fixture 预注入 passthrough 规避 CmdLineParser exit(-1)（注释自述）——新增此类选项时会静默踩坑。

### 5.10 前端与通用 API（9 条，均未定论）
- `main_online.cc:1211-1215,675-683` command_fifo_reader 为分离线程（detach），run 末尾 main 作用域对象析构窗口内其 enqueue_command 重试循环可能解引用已析构的 ingress/svc——窗口窄、进程随即退出，未运行验证。
- BackendType::Analytical/NS3 枚举值全仓无人产出（默认返回 NotSpecified），唯一消费点只做 != Garnet 比较——行为无差但枚举值等同死值。
- `main_online.cc:11` 头注宣称 CLI 契约 "unit-tested in tests/cli_online_test.cc"，该文件全仓 CMake 零命中——注释宣称的构建内测试保障不成立（文件本体见 §2-54）。
- `CallbackTrackerEntry.cc:19,28` register_send/recv_callback 防重注册仅 assert 级，Release 下同键重复注册静默覆盖（现网 chunk_id 单调递增不可达）。
- `ChunkIdGenerator.cc:102-109` complete 未记录单 chunk 完成态，同一 chunk_id 二次 complete 在其余发送在途时不会被拦截（现网每 chunk_id 恰一次 complete，不可达）。
- `main_online.cc:456-458` ed_commit_cb 计数循环索引未用——纯风格。
- `Logging.cc:29` for (auto sink : default_sinks) 按值拷贝 shared_ptr——纯效率。
- `main_online.cc:591-603` service_state_name 四枚举全覆盖后 "UNKNOWN" 兜底不可达——纯风格。
- `Sys.cc:300-330` 析构不 delete remote_mem/comm_NI（所有权在 main 的 unique_ptr/vector，main_online.cc:1087-1090,2065-2067）——现网无双删，但所有权约定无注释固化，跨 main 改造时易踩。

### 5.11 extern 网络后端定制（9 条 = 8 未定论 + 1 跨桶指引）
- `EventQueue.cpp:48,63-74` strict-increase assert 被自身注释与 EventQueue.h:93 引用为 ':31'——扩展后行号漂移、注释失实。
- `FluidScheduler.cpp:70-73,476,141-148` ~FluidScheduler 只取消 wakeup 事件、不取消 PropagatingTail 流的 tail 事件；scheduler 先于 event_queue 析构且队列未排空时 tail_arrival_callback 解引用已销毁元素 → UAF（常规跑空队列再析构不触发）。
- `NetworkParser.cpp:156-208` 不拦 topology 列表为空：dims_count==0 全部校验通过，直到首次 fluid_route 才 exit(-1)，报错点位与真实原因（缺 topology 键）脱节。
- **【跨桶指引】** M11 同族跨桶提示：本仓 main_online.cc Ring 分支注释自称含 radix==2 fallback 仍按 2*npus 估算，而 MultiDimTopology.cpp:195-198 实建 npus 条（已确认为 §1-M17）。
- `FluidScheduler.cpp:90-105` start_flow 在 mark_event_loop_started 前入队 pending_starts 且自身不调度 flush，依赖调用方调用普通版 flush_pending_starts（main_online.cc:1464 现有调用）——漏调即 flow 静默滞留，属调用方责任层面的脆弱契约。口径调和：这与 §2-35 不矛盾——§2-35 所死的是 **deferred 专用入口** flush_pending_starts_deferred（online 的 deferred 模式经 set_deferred_flush_mode(true)+start_flow 内联，无需该专用入口）；本条说的是 start_flow 入队的 pending_starts 仍需调用方手动触发普通版 flush，两个 flush 入口职责不同。
- `Ring.h:18-23` 注释把 1D Ring(8) 图示画成 2D 环面并写 "devices are both 8"，与 Ring.cpp 1D 建链语义不符（上游遗留）。
- `Device.cpp:46` Release 下重复连接静默覆盖旧 link、无 error 路径——Ring 退化宽度 finding（§1-M18）的防御缺口放大器。
- `Topology.h:116` fluid_route_cache 每 (src,dest) 对常驻一个 shared_ptr 且无上界，Switch(n) 全互联 O(n²) 配对时内存持续增长——设计取舍，无淘汰。
- `FluidSchedulerLinkObserverTest.cpp:265-267` 本仓 hand_checked 的 link0 合计 14 与 SI observed totals 巧合相同（仅 bucket 分布不同）——若只看 totals 断言会掩盖失配（已并入 §1-H2）。

### 5.12 chakra-feeder 定制（10 条 = 9 未定论 + 1 跨桶指引）
- `protobuf_util.h:19` readVarint32 的 `(byte & 0x7f) << shift` 在 shift=28 且末字节≥0x08 时为 int 有符号左移溢出（C++20 前 UB；g++ wrap 语义下数值恰好正确）。
- `protobuf_util.h:95` std::mutex 定义在头文件：当前仅 et_feeder.cpp 一个 TU include 故能链接，任何第二个 TU include 即 multiple definition 链接失败。
- `et_feeder.h:69` + `et_feeder.cpp:45` _feeder_id_cnt 非原子静态计数：多线程并发构造 ETFeeder 可撞 feeder_id → _node_cache key 冲突互串节点数据（现状主线程构造不触发）。
- `et_feeder.h:73` static _node_cache 全进程共享 16384 容量：多 workload 挤占；get_raw_chakra_node 的 put→get_locked 间隙若被其它线程 put 逐出该 key 即抛 "Key not found in cache"。
- `et_feeder_node.cpp:7-12` get_chakra_node 的 expired()/lock() TOCTOU：并发下 lock() 可返回空 shared_ptr 致调用方空解引用（单线程安全）。
- `dependancy_solver.h:44-47,112-130` getter 无锁返回内部集合引用，并发 take/finish 时调用方拷贝存在 data race（现状单模拟线程不触发）。
- `et_feeder.cpp:95-118,106-108` 环/自环 trace 不被 init 拦截：graph_sanity_check 只查 free⊆index，环节点不进 free 集合 → 部分环节图 init 通过、运行期 all_done 恒假挂死。
- `et_feeder.h:35` ifstream 以 `binary|in|app` 三标志打开：app 对只读流无意义，疑上游遗留。
- **【跨桶指引】** `CustomAlgorithm.hh:64` 裸持 `Chakra::ETFeeder*` 无析构 delete——face M7 同位点（已确认为 §1-M9，属 astraccl 桶，跨桶登记）。
- `et_feeder_node.cpp:85-88` 死函数内的 static_cast<CollectiveCommType>(comm_type<uint64_t>()) 无枚举值域校验；活路径 NodeStore.cc:481 的枚举转换在 NodeStore 桶需彼处核对。

### 5.13 配置解析与 SLO 工具链（10 条，均未定论）
- `clean_build_artifacts.sh:15` 注释称"兼容三种历史布局的编译目录名"，实际 for 循环只有 build/build_congestion_aware 两个名字。
- `hbm_watermark.py:769-776` prepare 期对 decision log 全量重扫一遍，driver 头注宣称 "decision log 4 流→1"，face/S2 新产物路径下实际为 2 流（性能声明失实，非行为错误）。
- `slo_common.py:133,154` 两处 docstring 对桶语义表述不一致（边界归左桶 vs 闭区间），代码实现一致——纯文档措辞歧义。
- `hbm_watermark.py:1774` journal 校验失败消息对 after 分量缺 'after.' 前缀（before 分量有），同一条消息两分支不对称。
- `run_online_strategy*.sh` 引用的 generated/runtime_config 目录无显式存在性检查，依赖 C++ 打开 system.json 失败兜底（间接 fail-closed，报错位置深）。
- `run_golden_live.py:110-112,601-604` 清理 generated/llama2_7b_wsc_llm_inference_54npus_*，本仓物化器只产 llama2_7b_inference_54npus_*——跨仓残留清理逻辑（无害冗余）。
- `test_metrics_contract.py:591` 冻结 terminal_status 两值域 vs `slo_common.py:200-201` 五值域：两层契约域不一致，生产者（metrics_postprocess.py:785）只写两值故暂无害，仿真侧开始写 rejected/dropped/timed_out 即两侧失配。
- `slo_stats.py:208-213` cmd_violation 对 e2e=None 的非完成行仍先计算 deadline：个别 rejected 行缺桶会 fail-closed 拒绝整表——"分母=全部终态"语义下严格性可议。
- `kv_cache_adapter.py:604-607,548-551` repo_variant 查表校验逻辑重复执行两次（冗余无害）。
- `config_resolver.py:345-349` 边界 rank 收集对 rows==1 且 cols==1 的网格 npu-ids=[0] 唯一——1x1 网格在下游 AnalyticalRemoteMemory 单端口路径的行为未验证，超出本桶证据范围。

### 5.14 trace 生成与 FACE 调度（11 条，均未定论）
- `generate_face_trace.py:656-664` _config_edge_ranks 第一分支恒假（FaceTraceConfig 无 remote_memory 字段，getattr 恒 None）——行为正确（生产构造也不传 edge_ranks，同为 mesh-boundary 口径），但注释 "the resolver's remote-memory edges when present" 失实。
- `generate_face_trace.py:420-421,645-653` _stage_tag 与 TransferTagAllocator 的 tag 空间无碰撞守卫：queue_index≥1000 的请求其 stage tag 进入 KV-transfer tag 区间（30s 窗口负载数百请求不触发，纯靠规模假设）。
- `traces/derive_20_first_30_seconds.py:292` arrival_scale≠1.0 时 int(round(t0/scale)) 可能非 1000 倍数，而正式入口 generate_trace.py:214 强制 timing%1000==0 fail-closed——缩放实验物化的队列会被自己的校验门拒绝（docstring 未提示）。
- `plan_materializer.py:210-212` vs `:265-271`：_derive_manifest_requests 对缺中间 turn 的队列静默按 history=0 记账，_derive_metrics_requests 对同形状输入 fail-closed raise——两函数口径不一致，load_request_queue 不校验 turn 连续性。
- `plan_materializer.py:60` REPO_VARIANT="astra-sim-face"——本仓为 face-LRU，manifest/metrics_manifest 的 repo_variant 溯源字段标为 face 仓变体名（注释称"原样透传"系有意，按字段语义失实）。
- `session_kv_manager.py:1432-1433` _metrics_move_session_parts 对空 parts 取 parts[-1] 会 IndexError：NO_HISTORY 建会话不 add_segment；现可达性被"prefill_length≥1 → move 前必有 grow_prefill add_segment"挡住，仅防御缺失。
- `generate_face_trace.py:473-492` _paired_transfer 对 bytes=0 的 shard 不过滤（对比 _emit_kv_transfer 各分支均过滤 0），comm_size=0 被 max(1,·) 静默钳成 1 字节（face 仓原样，现负载字节量下不可达）。
- `session_kv_manager.py:2452-2455,2567-2570` PARTIAL→LOCAL 恢复分支不清 evicted_at_ns/evicted_by_request_id（仅 REMOTE 回迁分支清）——现状态机下 PARTIAL 会话该字段恒 None，属字段语义与恢复路径解耦不彻底。
- `metrics_schema.py:28-29` EVENT_MICROBENCH_ITERATION_START/END（码 5/6）python 仓内零消费者（仅存在于 EVENT_EDGE_BY_CODE 结构中）；C++ MetricCollector 侧发射点在桶外未审，若 C++ 也不发则为协议死码。
- `test_face_tiered_eviction_sequence.py:78/157/183/223/259` 直接调 ensure_physical_fit 而不经生产的 reserve_request_capacity 包装层——两段逐出核心被覆盖，但 reserve 预占记账与逐出的组合口径无专项序列测试。
- `generate_face_trace.py:80` trace_granularity 缺省默认值 token_expanded 是在线主链必然拒绝的值（graph_batch_builder.py:572-575 fail-closed raise）——默认值选了唯一不被支持的档位；test_face_scheduler.py:362-364 无回归钉住该默认值矛盾。

### 5.15 online 服务与 verify 链（12 条 = 11 未定论 + 1 跨桶指引）
- `decision_bridge.py:248,258` _new_files 探测起点=next_seq 使 contains() 条件恒真（防御性冗余，无行为影响）。
- `online_service.py:242-243` profile_sink 在 main 中恒非 None（_JsonlSink 或 _NullSink），"is None→dump_profile" 回退分支从服务入口不可达（仅测试路径可达）。
- `ledger_reconcile.py:892-894,920-921` R0/R5 报告行 PASS/FAIL 引用全局 failures/balanced 而非分项状态，任一其他项失败会把 R0/R5 行错标 FAIL（判定正确、报告误导）。
- `ledger_reconcile.py:534` R2b rank 上界硬编码 [0,53]（54-NPU），跨硬件复用假报警（README 定位 20.csv 验收工具，属已知局限）。
- `train_a1_eviction_fixture.py:131-135,179-182` fixture 进出均 rmtree generated/…_plan_*（会清掉用户现存 plan 产物；仅 trace_config.csv 有备份还原）。
- `graph_batch_builder.py:537-545` _emit_side_branch 先 stash 清空再 chain_checkpoint，checkpoint 的 pending 分量恒空，回填全靠 stash（冗余无害）。
- `face_online_scheduler.py:1138-1140` 拆分首步后 joiner 的 drain_block_ends 不清（注释自辩），残留 dict 钉到该 runtime 完成才释放（单请求级死重；与 §1-L9 相关）。
- **【跨桶指引】** `generate_face_trace.py:546` _transfer_dict online/ 目录零调用（本体归发射模块桶核查；已入 §2-40）。
- `diff_explainability.py:158-168` completed_groups 双循环使同组落 "" 与 "request_complete" 双键（_completion_tick_diffs 比较含噪声键，两侧一致无影响）。
- `test_graph_batch_builder.py:141-148` _rank_nodes 依赖"单批发射、节点 id 连续、位置==id"前提（docstring 自述 M1 适配），多发射点场景脆弱，当前用例成立。
- `face_online_scheduler.py:1933-1938` _probe_instance_capacity 用 zip(required_shards, hbm_snapshots)，长度失配会静默截断漏检（当前两侧均 tp_degree 长度强制相等，边界依赖隐式约定）。
- `online_scheduler_base.py:880` 注释 "REQUSET_COMPLETE" 拼写错误（REQUEST_COMPLETE），纯注释噪音。

---

## 6. 方法与可信度边界（结语）

**方法**：15 模块分桶并行深挖（清单 320 文件 + 1 件点名补审，逐行精读 321 件，未分桶残留 0）；桶内逐文件读码 + 符号级全形式 grep（含 tests/、CMake、.py/.sh/.json/.md，含宏/字符串/取地址形态，排除姊妹仓与 .git）；每条发现由独立复核员全新上下文裁定，9 组跨桶重复发现按唯一缺陷归并并标注互证；关键 high 结论另经本会话静态抽查亲验（H1 两仓 clone 符号对比、Sys.cc:209-210、AnalyticalRemoteMemory.cc:58-61/127-131/168-172、NetworkFunction.cpp:11-19、示例脚本 :17/:28，全部与发现一致）。

**可信度边界**：
1. **无运行期证据**——构建基线链异常中止（world.run 'bash' 900s 超时），本仓全程未构建/未 ctest/未跑仿真；所有"必失败/崩溃/NaN/泄漏"均为证据链闭合的**静态推演**，未实测复现（face 文档所载 exit 1 / exit 139 / legacy=43 vs calendar=200 等实测仅为旁证引用，本仓未复跑）。修复验证时须先补构建与 ctest 基线，重点回归 §1.1 三条 high。
2. **可信度分层**——§1/§2 的 94 条为"深挖发现 + 独立复核确认"双关口结论（其中复核修正原发现多处行号/机理/谱系偏差，文中随条标注；§2 另有清册边界说明登记的 2 处 bug 伴生死码，见 §2 收尾）；§5 的 154 条登记中，146 条为未定论可疑点（缺运行验证、跨桶未逐行核、或需配置文档/作者裁定），不应直接当实锤引用，另 7 条跨桶指引与 1 条已澄清勘误仅作导航与留档。
3. **口径边界**——死代码结论仅对本仓（face-LRU）成立；跨仓删除须按 P13 查姊妹仓消费（已知豁免：joint 仓 link_count()、face 仓保留 DoubleBinaryTreeLocalAllToAll 枚举）；死配置键的"死活"以本仓 C++ 读取点为准（remote-mem-bw/pipeline-tile-fraction 在本仓已活，与 face 文档相反）。本清单只找"错"与"死"，不评架构；完整逐条证据（含复核意见全文）见工作流"深挖审查全量证据表"与发现看板。
