/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __ANALYTICAL_MEMORY_HH__
#define __ANALYTICAL_MEMORY_HH__

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "astra-sim/common/AstraRemoteMemoryAPI.hh"
#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Sys.hh"
// PortJob 以 unique_ptr<WorkloadLayerHandlerData> 持有 wlhd（下方 PortJob /
// PortState 头内定义），其隐式实例化（如 PortState 构造的 EH 清理需要
// vector<PortJob> 的析构 => default_delete 需要 sizeof 完整类型）要求该类型
// 在每个包含本头的 TU 内完整；Sys.hh 仅前向声明不够。本头自带完整定义
// （无环：WorkloadLayerHandlerData.hh -> AstraNetworkAPI/BasicEventHandlerData，
// 均不回指本头）。
#include "astra-sim/system/WorkloadLayerHandlerData.hh"

namespace Analytical {
enum MemoryArchitectureType {
  NO_MEMORY_EXPANSION = 0,
  PER_NODE_MEMORY_EXPANSION,
  PER_NPU_MEMORY_EXPANSION,
  MEMORY_POOL
};

// 远端内存端口后端（按《SerDes片外链路并发化改造执行方案》§3.1–3.4 重写，
// 取代旧串行 FIFO 单事务端口）：
//  - 每端口流体模型：同端口已发射事务全部可并发在途；latency 重叠且不占
//    传输带宽；进入传输的流平分端口带宽 remote_mem_bw_bytes_per_ns_，任一
//    流耗尽后幸存流立即按新流数重分；端口内无 FIFO 等待队列、无人为
//    outstanding 上限。
//  - 内部按连续 ns 子步推进；可观察的 Workload 回调落在整数 Tick，
//    时刻为 ceil(fluid_finish_ns)；callback_tick 是唯一对外完成时间，
//    fluid_finish_ns 仅用于物理服务分析（§3.1，二者不得混用）。
//  - 所有端口与双零定时作业共同竞争一个挂在首次 set_sys 传入 Sys 上的
//    全局最早 deadline 可取消变迁事件；事件 payload 只带全局 generation。
//  - 数值边界 fail-closed（§3.2）：正/有限/range 检查先于全部计算；
//    不变量错误走 Sys::sys_panic 明确 fatal 路径，不得仅 throw
//    （Sys::call_events() 会捕获 std::exception 后继续）。
class AnalyticalRemoteMemory : public AstraSim::AstraRemoteMemoryAPI,
                               public AstraSim::Callable {
 public:
  AnalyticalRemoteMemory(std::string memory_configuration) noexcept;
  // 定义在 .cc：PortJob 以 unique_ptr 唯一持有 wlhd，析构需完整类型。
  ~AnalyticalRemoteMemory() override;

  void set_sys(int id, AstraSim::Sys* sys) override;
  void issue(
      uint64_t tensor_size,
      AstraSim::WorkloadLayerHandlerData* wlhd) override;
  void call(AstraSim::EventType type, AstraSim::CallData* data) override;

  // §3.4/§5.1：正常仿真结束前的无条件 fail-closed 空集检查（sensing 关闭
  // 同样生效）。逐端口校验：作业容器空（latency-waiting/活跃流/待交付
  // 完成/双零 timer/未交付 wlhd cookie 混存于同一容器，非空即有残留）、
  // 活跃流计数为零、issued/completed 的 count 与 bytes 各自相等、
  // in_flight 归零；全局校验：无派发中的交付批、变迁事件未挂起且事件
  // 句柄已空。count 相等不得掩盖 bytes 错配——两类守恒分别独立判定。
  bool is_drained() const;

  // §3.4：提前 shutdown——删除全部未交付 wlhd、经取消 deleter 释放事件
  // payload、清空端口状态（含 PortStats 累计器，见下方 PortStats 注释）。
  // 只清理后端自有 cookie，不声称清理 Workload 独立持有的 HBM join cookie。
  void shutdown();

 private:
  // ---- 常量（§3.1/§3.2） ----
  static constexpr double kTickNs = 1.0;  // 本模拟器 1 Tick = 1 ns
  // §3.2：bytes=0 且 latency=0 的独立一次性定时作业，精确延 1ns 后异步
  // 完成；不进带宽作业集合，也不从 issue 栈同步回调（相对旧版 0ns 完成
  // 的明确语义变更）。
  static constexpr double kDualZeroDelayNs = 1.0;
  static constexpr double kInfNs = std::numeric_limits<double>::infinity();

  // 作业四态可区分（§3.1/阶段 1 要求）。
  enum class PortJobState {
    // 已 issue、latency 未到期；各事务 latency 可重叠，不占带宽分母。
    kLatencyWaiting,
    // 正字节事务 latency 已到期，进入同端口传输流集合（带宽分母之一）。
    kActiveStream,
    // fluid_finish_ns 已定，等待 ceil Tick 交付；不再占用带宽分母。
    kFluidCompleteAwaitingCallback,
    // §3.2：bytes=0 且 latency=0 的独立定时作业；到期 Tick 直接交付。
    kDualZeroTimer
  };

  // 单个远端事务作业。交付前由端口容器唯一持有 wlhd（§3.4）；交付时经
  // release() 把所有权移交 Workload；提前 shutdown 由 unique_ptr 统一
  // 删除未交付项。
  struct PortJob {
    PortJobState state = PortJobState::kLatencyWaiting;
    std::size_t port_index = 0;  // 所属端口；交付批次排序键 (port_index, issue_sequence)
    uint64_t tensor_size = 0;  // 原始字节数；§3.2 对 0 有特判
    double remaining_bytes = 0.0;  // 活跃流剩余字节；随子步按端口份额衰减
    uint64_t issue_sequence = 0;  // 端口级单调序号；不假定 node_id 跨 rank 唯一
    AstraSim::Tick issue_tick = 0;
    // latency 阶段到期的连续时刻；对正字节事务同时即 stream_start_ns
    // （§5.1：多服务区间时它就是首次开始时刻）。
    double latency_ready_ns = 0.0;
    // 物理服务完成子步时刻；允许是整数 Tick 内的 double 子步时刻（§3.1）。
    double fluid_finish_ns = 0.0;
    // 交付 Tick = ceil(物理完成时刻)；GraphSource/Workload/SLO 唯一完成时间。
    AstraSim::Tick callback_tick = 0;
    std::unique_ptr<AstraSim::WorkloadLayerHandlerData> wlhd;
    // 逐事务行键恒等字段：issue 时自 wlhd 复制（wlhd 在 callback 后销毁，
    // 明细行只允许使用作业自身字段及其拷贝——§5.1 不得事后从指针补键）。
    int detail_sys_id = 0;
    uint64_t detail_node_id = 0;
  };

  // 每端口作业容器与流体状态（§3.1）。端口之间带宽互不影响。
  // deadline 一律由作业集扫描重算，不做增量缓存（性能为阶段 4 实测项）。
  struct PortState {
    explicit PortState(std::size_t index) : port_index(index) {}

    std::size_t port_index = 0;
    // 本端口全部作业；四态混存，按 state 区分。用 vector 而非 deque：
    // vector 的移动构造是 noexcept，保证外层 vector<PortState> 扩容时
    // move_if_noexcept 走移动路径（deque 移动非 noexcept 会退化为对
    // move-only PortJob 的拷贝而编译失败）。
    std::vector<PortJob> jobs;
    uint64_t next_issue_sequence = 0;  // 端口级单调 issue_sequence 计数
    std::size_t active_stream_count = 0;  // 当前传输流数 N（带宽分母）
    // 活跃流剩余字节合计；用于子步服务量结算与守恒检查
    //（端口服务总量不超过 remote_mem_bw_bytes_per_ns_ × streaming_time）。
    double active_stream_bytes_remaining = 0.0;
    // 端口连续时钟：本端口已推进到的连续 ns；与全局调度分离（§3.3）。
    double port_time_ns = 0.0;
  };

 public:
  // ---- PortStats：只读端口统计快照（§5.1，阶段 3 后端观测） ----
  // 由 [H1]-[H4] 钩子无条件下低开销累计——sensing 只门控逐事务明细，统计
  // 累计常开（is_drained() 依赖它）。结算时点：
  //   - 发射在 [H1]；完成/交付在 [H4]——严格早于任何 Workload callback，
  //     in_flight 归零发生在 callback Tick 的整批交付结算；
  //   - 流数、忙碌时长、服务字节在 [H2] 按连续服务子步区间结算（事件区
  //     间积分，无固定间隔采样；区间内流数恒定，状态迁移只在子步边界）；
  //   - redistribution_events 按“连续服务子步的流完成时刻”计：[H3] 逐完成
  //     记录、下一子步结算点落账——同一时刻多流同时结束只计一次，且仅当
  //     仍有幸存流（N-c>=1）且其份额改变（N+j-c != N，即 j != c）；新流
  //     加入导致的份额变化记入 new_stream_joins 另列，不并入本计数。
  // 不保留旧 FIFO 账本的任何 pending/queued 键（旧 pending_requests/
  // PendingMemoryRequest 已随串行 FIFO 后端删除，不复制回来，也不把旧
  // queued 数改义为 latency waiting）。
  struct PortStats {
    // 发射/完成守恒对：count 与 bytes 各自独立相等（收尾分别校验）。
    uint64_t issued_count = 0;
    uint64_t completed_count = 0;
    uint64_t issued_bytes = 0;
    uint64_t completed_bytes = 0;
    // 在途：issue 到 callback 交付（含 latency 阶段）；[H1] +1、[H4] −1。
    uint64_t in_flight_count = 0;
    uint64_t peak_in_flight = 0;
    // 实际传输流（带宽分母）：快照取端口实时活跃流数（该计数只在子步
    // 转换处变化），峰值为 [H2] 区间结算的最大值。
    std::size_t streaming_count = 0;
    std::size_t peak_streaming = 0;
    // 快照时由实时作业状态推导（§5.1：后端自身作业状态是唯一端口事实源）。
    std::size_t latency_waiting_count = 0;
    // 已定完成、等待 callback Tick 交付的作业数（含双零定时作业）。
    std::size_t completion_waiting_count = 0;
    // 流完成导致的带宽重分事件数（按完成时刻计，判定见类首注释）。
    uint64_t redistribution_events = 0;
    // 另列：新流加入传输集合的次数（其份额影响不并入
    // redistribution_events——一个计数不混两种事件）。
    uint64_t new_stream_joins = 0;
    // 事件区间积分的服务时长：port_busy_ns 为 streaming_count>=1 的实际
    // 服务时间；shared_busy_ns 为 streaming_count>=2 的实际共享服务时间。
    double port_busy_ns = 0.0;
    double shared_busy_ns = 0.0;
    // 端口按带宽实际服务的字节总量（区间积分；与交付字节 completed_bytes
    // 口径不同：前者是物理服务量积分，后者是交付事务字节）。
    double bytes_served = 0.0;
  };

  // 只读快照（§5.1）：单端口（越界 fail-closed）。不推进、不结算、
  // 不改变任何后端状态。
  PortStats port_stats(std::size_t port_index) const;

  // ---- 逐事务明细流式写出（§5.1 阶段 3 第二步；仅 sensing 开启时启用） ----
  // main 在既有 --sensing-enabled 路径上调用一次（不新增配置键/开关）：
  // 每完成一事务向 bridge_dir/remote_memory_transactions.jsonl 追加一行
  // 并逐行 flush。惰性建文件：首行才创建，无首行零 bridge 残留（wscllm
  // 端态钉死——NO_MEMORY_EXPANSION 正式负载无远端事务即零残留）。未启用
  // （sensing 关）时零逐事务记录驻留、零文件；聚合 PortStats 与无条件
  // is_drained() 不受影响。
  // 行 schema（本次新定义，nlohmann json 单行 dump，UTF-8）：
  //   schema=1, type="remote_memory_transaction",
  //   run_id（行键；沿用 metrics manifest 非空 run_id，缺失时为完整规范
  //     化 RUN_DIR 关联键，不从目录 basename 猜测——由 main 解析后传入）,
  //   sys_id, node_id（行键，issue 时自 wlhd 复制到作业）,
  //   port_index, issue_sequence（行键；端口级单调序号）,
  //   issue_tick, bytes（原始事务字节）,
  //   latency_ready_ns, stream_start_ns, fluid_finish_ns, callback_tick。
  // stream_start_ns：正字节事务在 latency_ready 恰好一次性进入传输集合
  // （§3.1），多服务区间时它就是首次开始，取 latency_ready_ns；零字节
  // 事务无传输流，写 null。fluid_finish_ns 为末字节完成时刻。中途带宽
  // 变化由端口积分字段（port_busy_ns/shared_busy_ns/bytes_served）与
  // 行区间端点表达，不在行内展开逐子步区间。
  // 写出时点：[H4] 整批交付结算处，与完成统计同一批、严格早于任何
  // Workload callback——行字段全部来自作业自身字段与 issue 时拷贝，
  // 不在 callback 后从 wlhd 指针补观测键。
  void enable_transaction_log(const std::string& bridge_dir,
                              const std::string& run_id);

  // §3.3：全局变迁事件 payload。只带注册时的全局 generation 序号；
  // 不保存 PortJob*、迭代器或任何端口数据。正常派发在 call() 入口释放
  // payload 并清空已出队句柄；取消与析构经 register_event_cancellable 的
  // deleter（release_transition_payload）释放；陈旧 generation 路径同样
  // 先释放 payload 再按无操作处理。
  // 访问性注记（2026-09-25）：payload 类型 public 化，供 nway 测试 S13 经
  // call() 注入陈旧 generation（取代旧 #define private public seam）；
  // 两个状态成员保持 private，只读经下方访问器。
  class TransitionEventData : public AstraSim::CallData {
   public:
    explicit TransitionEventData(uint64_t generation_value)
        : generation(generation_value) {}

    uint64_t generation;
  };

  // ---- 全局变迁事件状态只读访问器（§7 白盒测试观测面；本仓新增公共
  // API，非对齐姊妹仓——joint 仓该成员 private 无访问器、face-LRU 无此
  // 成员）。与 port_stats 同口径：不推进、不结算、不改变任何后端状态。
  [[nodiscard]] uint64_t transition_generation() const {
    return transition_generation_;
  }
  [[nodiscard]] bool transition_event_pending() const {
    return transition_event_pending_;
  }

 private:
  // ---- 入口与端口映射 ----
  // 端口映射落点：PER_NPU 按 per_npu_port_indices_，PER_NODE 按
  // sys_id/num_npus_per_node_，MEMORY_POOL 恒 0；NO_MEMORY_EXPANSION 的
  // fail-closed 行为保留在 issue 路径，非法 port 一律终止。
  std::size_t resolve_port_index(int sys_id) const;

  // §3.2 特殊事务分类落点（三行表逐行对应）：
  //   bytes > 0            -> kLatencyWaiting（latency 到期后加入传输流集合，
  //                           callback 在浮点完成时刻的 ceil Tick）
  //   bytes = 0, latency>0 -> kLatencyWaiting（到期即转
  //                           kFluidCompleteAwaitingCallback，不入带宽分母）
  //   bytes = 0, latency=0 -> kDualZeroTimer（精确延 1ns 异步完成）
  // 分配端口级 issue_sequence、调用 [H1]，随后 replan_transition_event()。
  void enqueue_job(
      PortState& port,
      uint64_t tensor_size,
      AstraSim::WorkloadLayerHandlerData* wlhd);

  // ---- 全局可取消变迁事件（§3.3） ----
  // 先把全部端口推进到当前整数 Tick，再重算全局最早 deadline；仅在出现
  // 更早 deadline 时才取消旧事件重注册——issue 恰逢原事件到期 Tick 且事件
  // 仍在队列中时保留该 Tick 的原事件（不得取消后推迟，callback 不得晚于
  // 其完成 Tick；新事务不计入已结束流的新带宽分母）。
  // delivery_in_progress_ 期间只允许更新状态，不重挂事件；整批交付后统一
  // 计算并注册下一变迁。
  void replan_transition_event();

  // 全局最早交付 Tick（跨端口与双零作业取 min）；无在途作业时为 kInfNs。
  double compute_global_next_event_ns() const;

  // register_event_cancellable 的取消 deleter：唯一职责是释放 payload。
  // CallData 无虚析构，必须先 static_cast<TransitionEventData*> 再 delete
  //（沿用旧版完成 payload 的删除口径）。
  static void release_transition_payload(AstraSim::CallData* data);

  // call() 的派发主体。入口已由 call() 释放 payload 并清空已出队句柄；
  // 陈旧 generation 按无操作返回。单事件到期时：先对所有端口推进连续子步，
  // 收集该 Tick 应交付的全部完成作业并整体从端口容器移除，按
  // (port_index 升序, 端口内 issue_sequence 升序) 排序；先调用 [H4] 结算
  // 后端统计，再逐个移交 Workload callback；派发期间不递归分发、不对正在
  // 派发的 completed batch 重排；整批回调后统一计算并注册下一变迁。
  void handle_transition_event(uint64_t generation);

  // ---- 连续子步推进（§3.1） ----
  // 逐连续子步推进到 to_ns：每子步先取端口均分流速率
  // remote_mem_bw_bytes_per_ns_/N（fail-closed 正值检查，避免 bytes/N
  // 整数除法），推进到下一子步边界（活跃流最早耗尽时刻或更早的
  // latency 就绪时刻）时结算 [H2]；耗尽流转 kFluidCompleteAwaitingCallback
  // 并结算 [H3]，幸存流立即按新 N 重分——已耗尽流不计入新分母；小于 1ns
  // 的正传输可在同一整数 Tick 区间内完成并让幸存流继续重分。最后一步
  // 推进到 to_ns 本身。
  void advance_port_continuous(PortState& port, double to_ns);

  // 每次外部 issue 与事件派发都把全部端口推进到当前整数 Tick（跨端口推进
  // 先于全局重排，不因另一端口触发事件而丢失期间服务）。
  void advance_all_ports(AstraSim::Tick now_tick);

  // 收割 Tick 到期作业并交付：排序与“统计先于回调”的顺序见
  // handle_transition_event；在途（in_flight）计数在 callback Tick 结算。
  void deliver_completed_batch(AstraSim::Tick now_tick);

  // ---- 数值 fail-closed 原语（§3.2；fatal 走 Sys::sys_panic，不裸 throw） ----
  // 计算 bw/N、每个 next_ns、ceil(next_ns) 与 now+delay 前的检查落点：
  // 巨大 bytes、巨大并发、浮点残差、溢出及 NaN 均 fail-closed，不得变成
  // 无限事件、NaN 延迟或零步死循环。
  static double require_positive_finite(double value, const char* what);
  static double checked_add_ns(double base, double delta, const char* what);
  // ceil + Tick 可表示范围检查；拒绝 NaN/inf/超出 Tick 上界。
  static AstraSim::Tick ceil_ns_to_tick(double ns, const char* what);

  // ---- 观测挂钩点（各一处；阶段 3 PortStats 在此累计——只写统计累计器，
  // 不读改任何流体推进状态，不改变数值行为） ----
  // [H1] issue 结算：enqueue_job 内、作业入容器并取得 issue_sequence 后调用。
  void on_stats_issue(const PortState& port, const PortJob& job);
  // [H2] 连续子步推进结算：advance_port_continuous 每个服务子步调用一次
  //（含新流加入导致份额变化的子步）；streaming/忙碌时长按子步转换在此结算。
  void on_stats_substep(
      const PortState& port,
      double from_ns,
      double to_ns,
      std::size_t streaming_count,
      double bytes_served);
  // [H3] 单流完成结算：一条流转入 kFluidCompleteAwaitingCallback 时调用；
  // redistribution_events 按连续服务子步的完成时刻在此计（同一时刻多流
  // 同时结束且幸存份额改变计一次；streams_before/after 供分母变化判定；
  // 新流加入导致的份额变化另列，不与流完成混计）。
  void on_stats_stream_completion(
      const PortState& port,
      const PortJob& job,
      std::size_t streams_before,
      std::size_t streams_after);
  // [H4] 整批交付结算：deliver_completed_batch 在排序后、任何 Workload
  // callback 前调用（统计完成结算必须早于回调；§5.1）。
  void on_stats_batch_delivery(
      AstraSim::Tick callback_tick,
      const std::vector<PortJob>& batch);

  // ---- PortStats 累计器（阶段 3；随 [H1]-[H4] 写入，口径见 PortStats 注释） ----
  // 流完成时刻记录：[H3] 逐完成写入，下一子步结算点（[H2]）落账——同一
  // 时刻的多个 [H3] 只产生一次 redistribution 判定（按完成时刻去重）。
  struct StreamCompletionMoment {
    bool active = false;
    double moment_ns = 0.0;
    std::size_t interval_streams = 0;  // 完成时刻所处服务区间的流数 N
    std::size_t completions = 0;       // 该时刻累计结束流数 c
    std::size_t joins = 0;             // 该时刻新加入流数 j（首次 [H3] 推得）
  };

  struct PortStatsAccumulator {
    PortStats totals;
    StreamCompletionMoment moment;
  };

  // 每端口累计器；与 ports_ 惰性同步（ports_ 的构造期/set_sys 追加点无需
  // 逐一同步，读端按可空处理——无累计器即全零守恒）。
  PortStatsAccumulator& stats_for(std::size_t port_index);
  // 落账进行中的流完成时刻并清记录，返回该时刻的完成流数（无则 0）。
  std::size_t flush_redistribution_moment(PortStatsAccumulator& acc);
  // 汇总单端口快照：累计器字段 + 实时作业状态推导的流数/等待计数。
  PortStats assemble_port_stats(std::size_t port_index) const;

  // ---- 逐事务明细写状态（§5.1 阶段 3 第二步；口径见 enable_transaction_log） ----
  // 单行写出：未启用即无操作（零逐事务记录）；启用时惰性打开
  // bridge_dir/remote_memory_transactions.jsonl 并逐行 flush。
  void write_transaction_row(const PortJob& job);

  bool transaction_log_enabled_ = false;
  std::string transaction_log_dir_;    // bridge_dir（main 校验非空后传入）
  std::string transaction_log_run_id_;  // 行键 run_id（main 解析后传入）
  std::ofstream transaction_log_;      // 惰性打开；shutdown 时 flush+close

  // ---- 状态 ----
  MemoryArchitectureType mem_type_ = NO_MEMORY_EXPANSION;
  // §3.2：latency 必须 finite 且非负；带宽必须为正且 finite。double 口径
  // 显式取代旧 uint64 截断语义（本仓深挖清单 §1.3 同族条目，按新后端
  // 语义取代并记录）。remote-mem-bw 配置数值按 B/ns 使用（§2.1 判定门
  // 保留原单位口径，不在本改造中重定义）。
  double remote_mem_latency_ns_ = 0.0;
  double remote_mem_bw_bytes_per_ns_ = 0.0;

  // 端口映射配置（构造期 PER_NODE 正整数校验与浮点带宽输入校验留在 .cc，
  // 本改造不覆盖工作树既有校验）。
  int num_nodes_ = 0;
  int num_npus_per_node_ = 0;
  bool per_npu_ids_configured_ = false;
  std::unordered_map<int, std::size_t> per_npu_port_indices_;

  std::unordered_map<int, AstraSim::Sys*> sys_map_;  // 各 rank 的回调宿主 Sys
  std::vector<PortState> ports_;
  // 每端口 PortStats 累计器；与 ports_ 惰性同步（无累计器的端口即全零）。
  std::vector<PortStatsAccumulator> port_stats_;

  // §3.3：全局可取消变迁事件宿主。构造早于 set_sys，构造期不缓存 host；
  // 由首次 set_sys 落定，后续 set_sys 不再更改。
  AstraSim::Sys* event_host_sys_ = nullptr;
  AstraSim::SystemEventHandle transition_event_handle_;
  bool transition_event_pending_ = false;
  // 已注册变迁事件的目标 Tick（SystemEventHandle 不暴露 event_time_）；
  // 同 Tick 保留规则据此判定：pending_tick <= 新目标时保留原事件不重挂。
  AstraSim::Tick transition_event_tick_ = 0;
  uint64_t transition_generation_ = 0;  // 全局单调 generation；payload 只带它
  // §3.3 防重入：派发期间只允许更新状态，不递归分发、不重排正在派发的批次。
  bool delivery_in_progress_ = false;
};
}  // namespace Analytical

#endif /* __ANALYTICAL_MEMORY_HH__ */
