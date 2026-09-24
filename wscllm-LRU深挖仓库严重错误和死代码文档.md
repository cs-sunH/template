# wscllm-LRU 仓深挖：严重错误与死代码（供姊妹仓套用）

> **审查对象**：`template/astra-sim-wscllm-LRU`（三态 KV / 两段式逐出 LRU 仓）。
> **方法**：28 个模块、316 个文件分桶逐行全覆盖（各模块声明无抽样无跳读）；每条候选由独立复核员全新上下文裁定，死代码一律按 P14 全形式 grep（符号/字符串/宏/测试/CMake/sh/py）；基线实测：`cmake` Release 配置+构建 **均 exit 0**、`ctest` 0 项、16 套 Python 单测（15 绿 1 挂，详见 §4）。
> **结果**：确认 **163 条**——错误类 **high 9 / medium 34 / low 24**，死代码 **96 条**（按族见 §2）；**驳回 11 条**误报；另有 **48 条**未送复核可疑点（§1.3 摘录）。
> **用途**：同 face 文档——§1/§2 是本仓实锤结论；§3 提炼 face 清单（P1–P14）之外的新错误模式，供其余同源仓（face-LRU / wscllm / joint 等）深挖时按模式优先排查。
> **口径**：路径相对仓根；标 **承自face** = 与 face 同位的继承性问题，标 **本仓新增** = 本仓 KV/LRU 特性或分化引入；跨模块审计同报的同一问题已合并为一条。

---

## 1. 确认的严重错误

### 1.1 high（9 条）

| # | 位置 | 问题与触发 |
|---|---|---|
| H1 | `astra-sim/system/CommunicatorGroup.cc:130-148` | **承自face H1**：无 dimensions 的子集群通信域把注册表**共享**的 `CollectiveImpl*` 交给 `implementations_should_be_removed=true` 的 plan，析构即 delete → 同 ComType 两组先后退出 **double free**；一组析构后注册表/他组经 `get_collective_impl` 继续读 → **UAF**（comm_group.json 数组形式与 torch pg 子集组为真实触发路径；sys-stream/sys-astraccl 两模块审计同报，合并） |
| H2 | `astra-sim/system/astraccl/.../logical_topology/BinaryTree.cc:65` | **承自face M8**：DBT 仅支持 2 的幂节点数，非 2 幂维度经 `map::operator[]` 取 nullptr 直接解引用 → **段错误**（face 复核实测 exit 139；本仓只读未复现，CollectiveImplLookup→GeneralComplexTopology→Sys.cc:1165 触发链为代码级推演，仓内现无配置触发） |
| H3 | `astra-sim/system/Sys.cc:1188` | **承自face M1**：缺 `preferred-dataset-splits` 键时构造默认 0，`size/0` → **SIGFPE**（现存 14 份 system json 均含键，新配置漏写即首个集合通信崩溃） |
| H4 | `astra-sim/system/Sys.cc:1200` | **承自face M2**：缺 `scheduling-policy` 键读**未初始化枚举**（UB），LIFO/FIFO/EXPLICIT 随机选定 |
| H5 | `astra-sim/system/Sys.cc:947` | **承自face M3**：缺 `collective-optimization` 键同款 UB，All_Reduce 单趟/RS+AG 双趟**随机选定**、通信量失真（examples 的 custom_collective.json 实测缺键，靠 custom 提前返回侥幸未踩） |
| H6 | `extern/.../fluid/tests/FluidSchedulerLinkObserverTest.cpp:256` | **承自face H2**：legacy 参照与 hand_checked 均按旧 2^30 口径硬编码，本仓 `bw_GBps_to_Bpns` 已 SI 恒等 → 该 opt-in 回归 fixture 开启即挂（SI 手推 link0={3,5,3,2} vs hand_checked {3,6,3,2}；face 同码实测 exit 1 旁证） |
| H7 | `extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.cc:170` | PER_NODE_MEMORY_EXPANSION 缺 `num-npus-per-node` 键构造默认 0 → `issue()` **整数除零 SIGFPE**；缺 `num-nodes` 键端口表空 → :195 越界 **UB**（同函数对 remote-mem-bw 有三重守卫 exit(1)，对照成立；remote_fifo_ledger_test 证明分支为活路径，手写漏键即踩） |
| H8 | `sh_test_mesh/workload/llama2_7b_inference/generate_wsc_llm_trace.py:562` | **本仓新增**：KV 转移 tag「与 `_stage_tag` 段错开」契约只在队列 <1000 行时成立，`_stage_tag=queue_index*10000+…` **无上界校验**；本仓 30s 窗口源 trace 实测 1177 行，joiner P→D 迁移 tag 已与 TransferTagAllocator 的 [10^7, 2^32) 独占段**重叠**，同 (src,dst,tag) 碰撞可致 p2p 错配/结果错误（且 `_stage_tag` 无 uint32 上界检查，allocator 有） |
| H9 | `examples/run_scripts/analytical/congestion_aware/`（HGX-H100-validated.sh:17 等 3 脚本） | **承自face H3**：引用的 `build/astra_analytical/build.sh` 与无 `_Online` 后缀二进制已不存在，`set -e` 下**必失败**（本仓分化：face 的 `--remote-memory-configuration` 被吞半边已消除；三模块审计同报，合并） |

### 1.2 medium（34 条）

| # | 位置 | 问题（一句话） |
|---|---|---|
| M1 | `astra-sim/system/.../GeneralComplexTopology.cc:27` | 实现数>维数的唯一守卫是 assert，默认 Release(NDEBUG) 编译掉后 `dimension_size[dim]` **越界读 UB**（对照同场景 CommunicatorGroup.cc:103 用硬 throw） |
| M2 | `astra-sim/system/Sys.cc:1353-1355` | custom 流 `net_message_counter` 恒 0（consume 永不执行），阶段完成 `back() /= counter` **0/0=NaN** 写回并污染 DataSet 统计（face M6 姊妹点；两模块审计同报） |
| M3 | `astra-sim/system/SharedBusStat.hh:135-146` | **承自face M6**：八个 delay 字段 `/=` 计数器无零守卫，custom collective 流从不递增计数器，阶段完成**必现 NaN**（face 已加守卫，本仓未同步；两模块审计同报） |
| M4 | `astra-sim/system/LogGP.cc:72` | size==0 时 `(size-1)` 回绕 -1，负延迟转无符号 Tick 成**天文数字、事件永不到来**（face 已加守卫，本仓未同步） |
| M5 | `astra-sim/.../collective_algorithm/Ring.cc:239`（HalvingDoubling.cc:264 同款） | 读**未初始化** `MyPacket::stream_id`（UB），值写入 RecvPacketEventHandlerData 后全仓零消费 |
| M6 | `astra-sim/system/PacketBundle.cc:59` | 配置缺 `local-mem-bw` 时 `local_mem_bw=0` → `size/0` 得 inf/NaN → `static_cast<uint64_t>` **UB**，巨延迟挂死 |
| M7 | `astra-sim/system/UsageTracker.cc:24` | 静态模式 `retain_history=true`：每次流增删 push 一条且 report 生产零调用 → 长跑**内存无界膨胀** |
| M8 | `astra-sim/system/astraccl/custom_collectives/CustomAlgorithm.hh:36-66` | **承自face M7**：无析构，每次 custom 集合通信**泄漏 1 个 ETFeeder**（含打开的 ET 文件 fd），耗尽后 `exit(1)` 中止仿真 |
| M9 | `astra-sim/system/astraccl/CollectiveImpl.hh:34-41` | 基类**无虚析构**，CollectivePlan 经基类指针 delete 携带 string 成员的派生实现 → **UB**（与 H1 同一析构路径） |
| M10 | `astra-sim/system/Sys.cc:210` | **承自face M5**：`inter_dimension_scheduling` 钉死 Ascending 且无配置解析，RoundRobin/OfflineGreedy/OnlineGreedy 分支全死，手写键**静默吞** |
| M11 | `astra-sim/system/Sys.cc:847` | All_Gather 且 size < splits 时 chunk 整除得 0 → `ceil(size/0)=inf` 转 int **UB**（face 可疑点在本仓坐实） |
| M12 | `astra-sim/system/Sys.cc:225` | `collective_impl_lookup` new 无 delete、类亦无析构，三 map 持有的全部 CollectiveImpl* 随每 Sys 实例**泄漏**（两模块审计同报，合并） |
| M13 | `astra-sim/workload/Workload.cc:534` | **承自face M9**：roofline 模式 num_ops==0（或两速率键缺省 0）时 0/0、x/0 → NaN/Inf，`static_cast<uint64_t>` **UB**；在线 compact 路径遇非有限值直接 **exit(1)** 中止仿真 |
| M14 | `astra-sim/workload/Workload.cc:490` | **本仓新增**：hbm 双开关全关（不建模型）时 KV restore 闭式回退按 `tensor_size/local_mem_bw` 计时，缺键 0 → **Inf → UB**（Sys 零速率守卫对该除法无保护） |
| M15 | `astra-sim/workload/MetricCollector.cc:512` | **承自face M10**：node event_code **先窄化 uint8_t 再校验**，≥256 非法码回绕成合法码（257→1），本应 fatal 的清单被静默接受、指标错挂 |
| M16 | `astra-sim/workload/MetricCollector.cc:370` | load_manifest 顶层元数据 `.value()` 无类型守卫，坏 manifest 抛 type_error 不在任何 try 内、两处调用方裸调 → **std::terminate**，绕过本文件自身 fail-closed 报错通道 |
| M17 | `astra-sim/workload/MetricCollector.cc:2075` | **承自face 可疑点坐实**：WP8 直排积分**不夹负**、与水位线走法（夹 0）口径分裂，anchor 未解析造出负中间账面 → 必触发 >1% 虚假 mismatch 一致性违规 |
| M18 | `astra-sim/workload/execution_driven/tests/windowed_trace_reader_test.cc:1065` | **承自face 可疑点坐实**：Part Q 对照臂把字面量 128 传给 max_arrival_ns 形参，两行 arrival 1000/2000 全被窗口拒绝 → 零提交下断言**恒真、对照失效** |
| M19 | `astra-sim/workload/Statistics.cc:698` | `get_operator_type` 缺 METADATA_NODE case，Release 下 assert 编译掉、critical 只记日志 → 返回**未初始化** stat_node_type，垃圾值写进算子统计（在线版同函数显式处理 Metadata，佐证疏漏） |
| M20 | `astra-sim/workload/execution_driven/GraphBatchCommitter.cc:452` | hbm_access_mode 只查非负、缺 {0,1,2} 值域上界，≥3 经 parse/validate/preflight 全链放行后**静默按写模式计费** |
| M21 | `astra-sim/workload/execution_driven/tests/calendar_reader_oracle_test.cc:98` | **承自face M12①**：legacy 臂注释称 window 0 无界，实际落默认 high_water=128 有界，run_arm 只 pump 一次且不释放窗口 → 真实队列**必误报**（8 行合成 fixture 掩蔽；两模块审计同报） |
| M22 | `astra-sim/network_frontend/analytical/common/CmdLineParser.cc:36` | `--compute-scale` **死配置键**：cxxopts 接受并存储，但 main_online 的 13 个 get 键唯独不读它，计算耗时缩放永不生效（静默吞配置） |
| M23 | `astra-sim/workload/LocalHbmBandwidthModel.cc:184` | **本仓新增**：同 tick 竞争下 issue 不摘已完成作业且会**取消其完成事件**，退役/tics 累加/完成回调整体推迟到下一转换点（时序失真） |
| M24 | `astra-sim/workload/Workload.cc:107` | **本仓新增**：`hbm-kv-restore-bandwidth-sharing` 与缺省 true 的 contention 做 `||`，标准配置下开关**对行为零影响**，策略说明 §4.4「带宽共享由该开关」语义失效（静默吞配置） |
| M25 | `astra-sim/workload/LocalHbmBandwidthModel.cc:35` | **本仓新增**：sharing 单开+缺/零速率时 Sys 零速率回退被 `||` 击穿，ctor `throw invalid_argument` 且构造链无 try/catch → **std::terminate** 中止仿真 |
| M26 | `astra-sim/network_frontend/analytical/congestion_aware/main_online.cc:758-764` | **承自face M11**：Ring 宽度==2 退化维 link id 估算仍按 `2*npus`，后端实际只耗 npus 个 id → 其后所有 Mesh 维边界 link id **整体偏移 +npus** |
| M27 | `extern/.../fluid/tests/StaleEventCancellationTest.cpp:130` | **承自face 同款 medium**：kOneBytePerNsGbps 按「换算后恰 1 B/ns」旧口径取值，SI 恒等后实际 0.9313 B/ns → 整数完成 tick 期望全失准（opt-in） |
| M28 | `sh_test_mesh/workload/llama2_7b_inference/metrics_postprocess.py:172` | collector 一致性裁决（ok/violations/dropped_events）收进 `run.consistency` 后**全仓零读者**：违规 run 照样 exit 0 并产出 raw/normalized/request_metrics 三份产物，fail-closed 门失效 |
| M29 | `sh_test_mesh/slo_tools/hopbytes.py:199-265` | collect_wscllm **不消费决策日志逐出传输字段**：逐出/回迁 NoC 字节既不进 hop_bytes 也不进 bytes_without_hops，coverage 分母整类缺失而 notes 仍称「满覆盖」 |
| M30 | `sh_test_mesh/slo_tools/tests/run_golden_live.py:606` | 场景循环的 trace_config 指针恢复不在 try/finally、except 只捕两类异常：未预期异常逃逸后**仓内 trace_config.csv 永久指向 golden 队列**，默认入口静默消费测试队列 |
| M31 | `sh_test_mesh/.../online/verify/diff_explainability.py:433` | 文档承诺的退出码 1 **不可达**：DEFECT 级别全仓零产生点 → defects 恒空、任意决策差异/行数不齐一律 exit 0（审计器恒假绿） |
| M32 | `sh_test_mesh/.../online/verify/ledger_reconcile.py:224` | R3 宣称「每 request 恰一条 completion 决策」，实际 dict 赋值 **last-wins 静默覆盖**重复行，只查缺失不查重复/总数，R3a 照样 PASS |
| M33 | `sh_test_mesh/run_scripts/run_online_strategy_sensing.sh:179` | **本仓新增**：sensing runner 归档清单**漏 kv_delta_journal**（也无 checksum 搬运，主 runner 有）→ journal/checksum 滞留 bridge/ 成孤儿，每个 sensing run 的 SLO 权威重放层静默降级 upper_bound_only |
| M34 | `sh_test_mesh/slo_tools/tests/test_driver_parity.py:445` | 基线实测挂：`Ran 13 tests … FAILED (failures=2)`，`assertEqual(rc_old, expect_rc)` 得 `1 != 0`——唯一不被「全绿」叙事覆盖的 Python 套件（见 §4 基线） |

**low 级错误 24 条**（要点）：GeneralComplexTopology.cc:26 空实现向量 `size()-1` 回绕埋雷；LocalRingNodeA2AGlobalDBT.cc:57-59 死分支接回必 nullptr；CollectiveImplLookup.cc:28-38 `direct` 前缀 stoi 未捕获 → terminate 绕过友好退出；MetricCollector.cc:500 stoi 尾随垃圾/负键静默错配；ServiceCoordinator.cc:123 deadline 守卫秒 vs ns tick 宽 ~1e9 倍、拒绝实际依赖转换 UB 饱和；event_queue_deferred_test.cc:259 每轮 ~600 个 DiffCtx new 不 delete（test-only）；main_online.cc:1211-1213 detached FIFO 线程持 main 栈对象引用，退出期窄窗口 UAF；Workload.cc:484 restore 是四条 HBM 发射路径中唯一无零字节守卫（throw 落事件栈被吞 → wlhd 泄漏+节点悬空）；LocalHbmBandwidthModel.cc:136 fail-closed throw 被 `Sys::call_events` 吞 → 有作业无转换事件静默挂死；LocalMemUsageTracker.cc:199 张量重复写仅 `assert(false)`，Release 静默沿用首次写；session_kv_manager.py:2216 `protected[0]` 静默忽略第 2+ 保护会话、:2424 净额回补无容量收敛（与策略说明 §4.2 措辞偏差）；plan_materializer.py:168 畸形队列静默 history=0 与同文件 fail-closed raise 不一致；metrics_schema.py:303 越界文案 1-7 实际值域 1-8；metrics_postprocess.py:986 行内注释残留单值旧文案；test_kv_delta_journal.py:355 checksum 门断言漏第五项 remote_account_zero；wsc_llm_online_scheduler.py:544 `batch_train_` 前缀无保留校验、:508 SH_TRAIN_MAX_ITER 负值无校验（推迟到发射期才 fail）；hbm_watermark.py:578 拷贝区第 6 函数与 manager 源漂移（数值尚等价、逐字拷贝契约失实）；slo_common.py:602 用无空格前缀匹配把 `[METRIC][ERROR]` 诊断行当数据行 → 带 violations 的 run 其 restore 全量扫描必 fail；load_imbalance.py:199-204 零长区间相位不一致计 1 桶；hopbytes.py:168-185 face decode 分支 shards/hops 不齐整条丢弃无兜底；test_kv_transfer_emission.py:334-336 合成 shard noc_path 先行后列与生产 XY 序矛盾；test_weight_passes.py 无 `__main__` 直跑静默 0 用例（P10）。

### 1.3 未送复核可疑点（共 48 条，摘录代表性 8 条，均未经独立复核）

- `MemEventHandlerData` 疑整类零实例化：scoped enum 不可隐式转换，Sys.cc:703-709 三分支疑永假；handleEvent 的 NPU_to_MA/MA_to_NPU 分支同疑永假（Sys.cc:695-697）；
- `MyPacket::call`+notifier 回调疑整链死亡（fm_id/sender/ready_time/cycles_needed 疑死字段）；
- `QueueLevels` 四参构造、`Roofline` 单参构造（留 bandwidth 未初始化）、`MemMovRequest::latency`+`set_iterator` 疑死；
- sys-core 钉死/只写不读族：`intra_dimension_scheduling` 疑钉死 FIFO 三分支永假、`comm_scale`/`stream_priorities`/`SchedulerUnit::usage` 疑只写不读、`get_collective_implementation` 疑纯死声明、`OfflineGreedy` 构造 else 疑死分支（该类整体已因 M10 生产不可达）；
- 1 维 `Ring.cpp:22` 宽度==2 疑二次 connect 致 Release 下 2 条幽灵链路（多维侧有 Ring-2→Mesh 退化而 1 维未设防）；
- `hbm_watermark.py:1391` 疑把半层比值钉死 0.5，奇数层模型虚报 restore_bytes_mismatch；`kv_cache_adapter.py:749` 疑 --reconcile 两侧回退不对称；
- `slo_params_manifest.json` 21 个冻结参数疑 11 键无脚本消费者；slo_tools 5 个 `.bak_caliberfix_20260905` 残片（face §5.2 已删同族）；
- `LocalMemUsageTracker.cc:87` 疑 `std::stoull` 负号回绕/尾随垃圾静默接受（同仓其他解析点显式拒绝，校验缺失）。

（注：个别未送复核条目已在驳回侧被部分反驳，如 `DataSet::call` 经 Workload.cc:737 注册为 Callable 可达，不列为可疑点。）

## 2. 确认的死代码（96 条，按族归类）

| 族 | 代表实例（均已独立复核零引用/恒假；标 **(med)** 者为 medium 级） |
|---|---|
| 整类/整文件零引用（12） | `Torus3D`（ctor 藏除零、CMake GLOB 仍编入）；`LocalRingGlobalBinaryTree`；`LocalRingNodeA2AGlobalDBT`（dim==2 分支接回必 nullptr，**死里藏错**）；`CSVWriter`；`AstraComputeAPI/ComputeKernel`+4 个 create_llm_* 工厂（get_static_runtime 返回未初始化聚合）；`AstraSimDataAPI/LayerData` 与 `Common.hh` 两组**双副本同 guard**（改一份另一份被静默吞）；MetricCollector Phase-0 PerformanceCounters 整框架（9 字段恒 0）；`WscRelevantKvAllocator` 整类约 240 行（含 KVAllocation/Piece、routes_for_decode）；`RemoteFifoLedger` 整册（使能门永不开、record 恒 no-op，头注释与 README 自相矛盾，**本仓新增**）；`inputs/` 整目录 10 个上游 json；`Ring_4npus.yml`（连失效消费者都没有） |
| 钉死枚举/恒假分支（12） | `CollectiveImplType` 四枚举值（AllToAll/DoubleBinaryTreeLocalAllToAll/LocalRingNodeA2AGlobalDBT/HierarchicalRing）零构造零消费；`BYPASS_PERNODE_CUSTOM` 死旁路（头注自认 "No current usecase"）；`OperatorType::REPLAY` 死分支；HardwareResource.cc:148 else 内恒假 `==0`；Workload.cc:408 `if(true)` 死 else；GraphBatchCommitter.cc:473 与 :1043 comm.tag 值域检查恒假（uint32 拷入 uint64/int64 两款）；OnlineCli.cc:233 uint64 上限恒假；Logging.cc `if constexpr` 恒假早退；LocalMemUsageTracker.cc:251 LP64 同宽恒假守卫；npus_for_calibers 不可达分支；hopbytes.py:210 prefill shard 级口径分支对本仓产物恒死（写侧零生产） |
| 只写不读字段（17） | BaseStream `test/test2/phase_latencies[10]`、`total_packets_sent`、`initial_data_size`；RecvPacketEventHandlerData `vnet/stream_id`（默认构造还不初始化）；`Algorithm::name`+Name 枚举（4 算法各写一次零读）；OnlineStatisticsState 三字段 `operation_intensity/is_memory_bound/network_bandwidth`（GraphSource.hh:117-120，承自face，三模块审计同报合并）；OperatorStatistics `comm_size/network_bandwidth` 链（唯一"读者"是**注释掉的报告块**）；`num_npus`（唯一构造点恒传 1）；`rank_instance_conflicts`（累加后 `(void)` 丢弃，与 "never silently resolved" 注释矛盾）；OnlineDriverContext `ingress/systems/expected_requests`（快照恒 0）；wsc_llm_online_scheduler `completion_ns/hbm_before_request/hbm_after_completion`；SessionKVSnapshot 8 字段族（docstring "used by manifest" 失实）；WatermarkReplay `repo_variant/mapping/tokens`；`SessionState.tokens`（6 写点零读）；`_ChangePoint.delta`；`alpha/adjusted_transfer_cost`（调参不改变任何路由）；`FluidFlow::total_bytes` |
| 死函数/死访问器（29） | `UsageTracker::report/report_percentage`；`HardwareResource::report`（连带 tics_cpu_ops/gpu_comms/hbm_dma_ops 三字段只写不读，tics_gpu_ops 因真实读取存活）；`slo_watermark_period_ns`；`retire_online_operator`（hh:254 注释指认的填充路径已失实）；`get_type_time`；`get_operator_statistics` 无参版；`was_json_id_committed`；`compute_touched_ranks`（生产弃用，仅自测引用）；**fence 记账 API (med)**：`on_fence_scheduled/on_fence_resolved` 零调用 → `pending_fence_count()` 恒 0，main_online.cc:1564 停驻诊断失真（承自face M12③）；`scheduled_future_arrival_count`；`FileDecisionBridge::stats()`（main_online 注释宣称的消费方式失实）；`NodeStore::meta_for`；`get_options()`；`final_session_counts`；时序估算族（WscLlmTimingLut/Entry、estimate_iteration_time_ns）；generate_trace.py 零生产符号群（InferenceGroup/shard_size/RemoteMemoryConfig+load_remote_memory_config/TraceBuilder）；测试专用函数族+`KV_CACHE_EVENT_COLUMNS`+`_emit_control_trigger`；ServiceMetrics 家族整族（write_manifest 链，实际 manifest 由 plan_materializer 合成）；planner-LUT 统计链（planner_lut_stats.json 全仓无写者）；`resolve_metrics_detail`；`node_count_total`；`replay_decision_log`；`percentile_from_sorted`；`ns_to_ms`；`APPENDIX_PCTS_DEFAULT`；FluidScheduler Phase-7 拥塞快照集群（link_congestion_snapshot/link_state_epoch()/link_count()）；`flush_pending_starts_deferred`；`get_completion_heap_size`；AnalyticalRemoteMemory parity 占位 `architecture_name/port_mapping_rule`（注释宣称的消费链整个缺席） |
| 孤儿测试/测试接线（10） | `cli_online_test.cc`、`event_queue_deferred_test.cc`、`ingress_idle_fixture.cc` 三件套**未入任何构建**仅头注 g++ 可达 **(med)**（承自face M12②；ingress_idle_fixture.cc:476 的 UB 依赖断言因此脱离回归网）；6/7 KV/调度 Python 单测不在任何回归入口 **(med)**（tests/run_all.sh 只跑 1 个模块且自称 "Running all regression tests"，无 CI）；全仓非 extern 仅 1 个 add_test 且挂默认 OFF option、analytical 22 个测试目标 0 个 add_test（基线 ctest 0 项互证）；`bridge_cpp_death_fixture.py` 指名驱动脚本不存在（自动断言链断裂）；8 个测试头注错指姊妹仓根 astra-sim-wscllm（fork 残留）；event_queue_deferred_test.cc:423 `_Exit(x?0:0)` 恒 0 两支同值+EventQueue 行号引用失真；windowed_trace_reader_test.cc:646 `expect(true)` 恒真占位；test_metrics_contract.py:693 恒真断言块（无被测符号参与） |
| 死配置（5） | `local-mem-capacity-bytes`（resolver 写入 system.json、C++ 零读取，仅 Python 测试断言在场）；`logical-pool`（remote_memory.json 唯一读者不解析，且被**强制必填**缺了反而报错）；`boost-mode`（模板透传进每次 run 的 system.json，face 已清同族）；`kv_cache_policy=session_lru_recompute` 旧值无对应行为（白名单放行后静默按 tiered 执行，P2 变体）；trace_config.csv 残留 "history-recompute" 字样（与「RECOMPUTE 已删」规范矛盾） |
| 遗留残片/注释失实（11） | `_RESIDENT_LEGACY/_EVICTED_LEGACY/RECOMPUTE` 兼容残片（RECOMPUTE 全仓唯一引用就是一条 import）；`JOURNAL_TIERS` 死常量；HardwareResource.cc:184/hh:58 注释失实（宣称的 COMP 门旁路已随 Path-2 删除，生产实际被 is_available 串行化）；main_online.cc:10-13 头注残留已删 replay 路由文档；前端 CMakeLists `list(REMOVE_ITEM)` 指向已删除的 main.cc（无操作残留）；test_first_token_proxy 过期行号引用（:646→实际 :1033）；test_hbm_watermark.py:1061 注释 num_heads=3 与实写 5 不符；run_golden_live.py:565 同语句连续赋值两次；test_driver_parity.py:464 `finally: pass` 每次运行泄漏 3 个 /tmp 目录；decision_kind_counts 死变量（其自然用途恰是 M32 的缺口）；pending_store_tails 第三分量 source_ack_recv_node_id 只存不读 |

## 3. 姊妹仓套用清单增补（face P1–P14 之外的新模式）

| # | 模式 | wscllm-LRU 实例 | 排查动作 |
|---|---|---|---|
| P15 | 注释/拷贝契约跨仓漂移 | hbm_watermark 拷贝区第 6 函数（manager 已重构、逐字拷贝契约失实）；MetricCollector.hh:145（face 确死的本仓已活）；HardwareResource.cc:184；main_online 头注残留 replay；metrics_schema 1-7 文案 | 凡带「逐字拷贝/md5 对齐/no caller/唯一路径」契约的注释逐条对源复核；跨仓分化（含反向：face 死→本仓活）必须同步注释 |
| P16 | 校验器自身恒绿（fail-closed 失效） | diff_explainability 退出码 1 不可达；metrics_postprocess 一致性裁决零读者；ledger_reconcile 只查缺失不查重复；test_driver_parity 挂 2 被单一入口「全绿」掩盖 | 每个承诺非零退出/违规上报的工具：找违规值的**产生点**与**读者**；回归入口列出实际覆盖清单 |
| P17 | 事件回调内 throw 被宿主吞掉 | Sys::call_events 仅 critical 后继续：LocalHbmBandwidthModel.cc:136 防御 throw → 有作业无转换事件静默挂死；Workload.cc:484 零字节 restore throw → wlhd 泄漏+节点悬空 | 对 `callable->call` 链内的 throw 追宿主 catch 行为；fail-closed 必须配资源清理或重调度 |
| P18 | 语义开关在缺省组合下退化为常量 | Workload.cc:107 `contention \|\| sharing`：contention 缺省 true 且模板同写 1 → sharing 真假行为全同（M24） | 布尔开关与缺省 true/false 的同伴做 `\|\|`/`&&` 组合时，验证两值分支是否真分叉 |

## 4. 覆盖与可信度

- **覆盖**：28 个模块 316 个文件分桶逐行全覆盖（各模块声明无抽样无跳读）；所有权/消费链跨模块同位核对（CollectivePlan 所有权链、CollectiveImplLookup、Sys 解析段、MetricCollector 发射面、resolver 写入键逐键对 C++ 读取点等）；死代码判定全部按 P14 全形式 grep；face 深挖文档与本仓 KV 策略说明全读作排查清单。
- **复核与驳回**：177 条原始候选逐条独立复核，跨模块同报合并后**确认 163**（high 9 / medium 34 / low 24 / 死代码 96）；**驳回 11 条**误报并留档，典型：隐式默认构造经派生类初始化列表真实执行、`DataSet::call` 经 Callable 注册被事件分发多态调用、pytest 发现式可达、参数化钩子在姊妹仓有真实传参——均为 face P14「只查显式形式」教训的变体。
- **基线（本机 /tmp 仓外构建，仓库保持裸仓态）**：`cmake` Release 配置 exit 0（pdlog 1.14.1、Protobuf 3.21.12）；构建 -j8 exit 0（仅 warn_unused_result 警告）；`ctest` **"No tests were found!!!"**（0 项，与 §2 测试接线发现互证）；16 套 Python 单测：15 套全绿（166 用例），`test_driver_parity` 13 用例 **FAILED (failures=2)**（即 M34）。
- **未覆盖**：受只读约束本仓未编译未运行任何仿真/测试，H2/H6/H7 等触发链为代码级推演（H2/H6 以 face 同码实测旁证）；fluid 两个 opt-in fixture 与 OfflineGreedy C++ 测试未实际执行，「必挂/不覆盖」为静态复算；长跑内存膨胀类（M7）未做时长外推；extern helper/三方库（fmt/spdlog）未审。
