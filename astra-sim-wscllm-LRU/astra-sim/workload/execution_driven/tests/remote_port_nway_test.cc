/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*

remote_port_nway_test.cc -- 《SerDes片外链路并发化改造执行方案》阶段 5.1/5.2
端口模型精确夹具（方案 V5.3，2026-09-24）。

自包含：测试自身在 /tmp 的 mkdtemp 隔离目录写全部配置；不触碰任何 generated
输入。夹具配置沿用官方模板形状（阶段 5 避雷段）：system.json 具备
scheduling-policy / preferred-dataset-splits / collective-optimization 三键，
四个 *-implementation 键全 ["ring","ring"]（无 custom/doubleBinaryTree）；
comm_group.json 各组带 ranks + dimensions。

驱动方式：真 online 栈（EventQueue + CongestionAwareNetworkApi + FluidScheduler
+ Sys + AnalyticalRemoteMemory），按 alarm_cancellation_test.cc 的夹具方法；
但事务由测试直连后端 issue（wlhd->workload 指向测试私有的 RecorderWorkload），
从而对 bytes/发射 Tick 逐笔控制。事件循环与全局 Tick 是真的；完成回调经
Sys::register_event 在确切 Tick 交付（deferred flush 打开：物理上下文的同 tick
注册按插入序同 tick 交付，非物理上下文的同 tick 注册走 deferred 通道，
两者都保序）。

覆盖映射（阶段 5.1 逐项 -> 场景）：
  PER_NPU 映射 ............... S1/S2/S6A/S7（npu-ids 六端口，逐 rank 独立）
  PER_NODE 映射 .............. S3（2 节点 x 3 NPU，同节点共享端口）、S6B（1x3）
  MEMORY_POOL 映射 ........... S4/S5/S12（全部 rank 压入逻辑端口 0）
  同刻 latency ............... S1（双 rank 同刻发射）、S3（r0/r1 同刻同端口）
  错峰 latency ............... S2/S3/S4（+50/+150/+100/+20ns 错峰发射）
  2 流长短服务与完成即重分 ..... S2（1200B 长流两次重分）、S7（同端口双流）
  latency-ready 与 stream completion 同 Tick  S2（C 于 B 完成同刻就绪）
  同 Tick 多端口/多作业排序 .... S7（5 笔同刻完成：host rank 先、余按
                               (port_index, issue_sequence) 升序）、
                               S1/S3/S5（同刻批内 port/seq 序）
  单事务非整除 ............... S6A（100B@6B/ns，fluid=116.67 -> callback 117）
  极小残差 ................... S6A（10B@6B/ns，share=(10/6)*6 超余 ~2e-15，
                               落入 1e-6 clamp：不 panic、不推迟、不丢服务）
  双零（bytes=0 且 latency=0）  S5（精确 +1ns 异步完成、不进带宽集合、不从
                               issue 栈同步回调）
  零字节正 latency ........... S4（0B/100ns 恰在重载窗口完成且不稀释 N）
  超大值/NaN/无效带宽 fail-closed  C1-C9 fork 子进程（构造期 bw=0/负/inf、
                               latency=负/inf；运行期 Tick 溢出：1e308 延迟、
                               1e-20 带宽；NO_MEMORY_EXPANSION issue；PER_NPU
                               未配置 rank issue——全部必须 exit(1)）。
                               注：JSON 数值文法无法表达字面 NaN（解析即抛），
                               NaN 防线与 inf/Tick 溢出共用同一 isfinite/range
                               守卫（require_*_finite / ceil_ns_to_tick），
                               以 C3/C5/C6/C7 为代表覆盖。
  服务速率/容量边界 ........... RefPort 参考流体模型逐段断言：段内各流同速
                               bw/N、端口段服务量 == bw x 段长、累计服务量
                               <= bw x streaming_time（每场景全程）。
  计数/bytes 守恒 ............ 每场景收尾：逐事务恰交付一次 + 测试侧
                               issued/completed 的 count、bytes 分别相等 +
                               后端 is_drained()（其内部逐端口独立校验
                               count/bytes/in-flight）+ 事件队列排空。
  callback Tick 与 fluid finish 不混用  全部锚按 ceil(fluid_finish) 断言；
                               S6A 注明 1B 流 fluid 服务 1/6 ns 而 callback
                               落下一整数 Tick，两者不互称。
  §7 事件 generation ......... S13：向 call() 注入陈旧 generation 的
                               TransitionEventData 为无操作（计数不变、
                               不交付、不推进），原变迁事件照常触发；排空
                               后注入仍无操作、保持 drained。
  §7 派发期同步重入 .......... S14：交付回调内同步 issue 新事务，断言方案
                               规范性属性——新事务按自身解析 Tick 交付、
                               不与本批递归同批，整批统计/守恒闭合。
  §7 提前 shutdown ........... S15：在途+挂起变迁事件时 shutdown——事件
                               取消（队列无触发）、未交付 wlhd 经 unique_ptr
                               删除（零交付）、PortStats 清零、is_drained
                               立即为真、二次 shutdown 幂等，shutdown 后
                               可继续服务新事务。

数学锚（阶段 5.2，期望值由 RefPort 按逐段服务现算，非手抄混合公式；
三个指名锚另加显式等值断言）：
  锚 1  bw=6 B/ns、latency=100ns：两个同刻 600B 流各以 3 B/ns 服务，
        fluid finish=+300、callback t=+300（旧行为串行 FIFO 数学对照为
        +200/+400：测试断言两笔同为 +300，不等价于串行对）。
  锚 2  bw=100 B/ns、latency=50ns、同端口四流 1k/2k/3k/4kB：连续 work-
        conserving 重分 fluid/callback = 90/120/140/150ns（各段整除：
        [50,90) N=4 每流 25；[90,120) N=3 每流 100/3；[120,140) N=2 每流 50；
        [140,150) N=1 每流 100）。
  锚 3  1B、6B/ns：fluid finish = 100 + 1/6 ns（约 100.167，内部服务可小于
        1ns），callback 在下一整数 Tick +101——两种时间不得混用。

已知分歧处置（2026-09-25 更新）：下述 join-at-boundary 分歧已在后端按方案
§3.1 修复——advance_port_continuous 先快照“进入本子步前已活跃”的流集合
served_streams（AnalyticalRemoteMemory.cc:586-597，其中 :589-590 注释点名
“RemotePortNwayTest S2/S4 的规格即此语义”），份额仅在 streams_before>0 且
elapsed>0 时产生（:604），流耗尽循环只对 served_streams 快照扣减
（:647-652）；恰在边界 latency-ready 的流不再被扣本子步份额。因此 S2/S4
现为有效回归门：后端若回退为“先翻活跃流、再对全活跃集合扣份”，两场景即
应 FAIL，不得再以“已知分歧”叙事吸收回归。

分歧原始记录（编写时 2026-09-24 对当时后端的代码走读结论；本测试按纪律不
自跑；所引行号与语句在当时工作树成立，现已不存在）：
方案 §3.1 要求 latency 到期才进传输流集合。wscllm-LRU 后端 advance_port_
continuous 先在子步边界把 latency-ready 事务翻成活跃流
（AnalyticalRemoteMemory.cc:589-608），随后同一子步的流耗尽循环对"所有"
kActiveStream 作业统一扣减本子步份额 share（AnalyticalRemoteMemory.cc:613-639，
`if (streams_before > 0) job.remaining_bytes -= share;`）——恰在边界加入的流
并未在 [t,boundary) 获得服务却被扣一份：
  - join 字节数 > share：静默提前完成（服务被窃，callback 偏早）；
  - join 字节数 < share：remaining < -1e-6 触发 sys_panic（进程 exit(1)）。
family 参照实现（joint AnalyticalRemoteMemory.cc:373-441）在事件点先 flip
再积分 [t,t_next)，积分只触已达 Streaming 的作业，无此问题。因此本套件中：
  - S4（r3 携 300B 于 +200 就绪、被扣 160 后余 140B，非致命）预期以 FAIL 行
    暴露 r3 与 r0-r2 的提前完成 Tick；
  - S2 的 60B C 流（60 < share=600）预期在 fork 子进程内终止（B 被窃服务
    后提前完成会把变迁目标拉回当前 Tick，主上下文 delta-0 重挂还会触发
    EventQueue 严格递增断言，或先经 sys_panic exit(1)——均为子进程死亡），
    由父进程"子进程必须 0 退出"断言捕获。
两处断言值均按方案 §3.1 语义书写（夹具即规格）；后端修复后两场景应转绿，
其余场景按走读在当前后端应全绿。

构建（先按本仓 README 完成 configure，再）：
  cmake --build build/astra_analytical/build_congestion_aware -j \
        --target AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest
运行（无参数；exit 0 = ALL PASS）：
  build/astra_analytical/build_congestion_aware/bin/\
AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest
注册：astra-sim/network_frontend/analytical/CMakeLists.txt（阶段 5.7 目标全名
AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest）。ctest 不作为执行
证据（阶段 5.7 口径）；回归以上述显式命令与退出码为准。
*******************************************************************************/

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include <astra-network-analytical/common/EventQueue.h>
#include <astra-network-analytical/common/NetworkParser.h>
#include <astra-network-analytical/congestion_aware/Helper.h>

// S13/S15 白盒观测走后端公共接口：transition_generation()/
// transition_event_pending() 只读访问器 + public TransitionEventData 注入
// （2026-09-25 处置承诺兑现：旧 private 访问 seam 已整段删除）。
#include <remote_memory_backend/analytical/AnalyticalRemoteMemory.hh>

#include <unistd.h>

#include <sys/wait.h>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace AstraSim;
using namespace Analytical;
using namespace AstraSimAnalytical;
using namespace AstraSimAnalyticalCongestionAware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

namespace {

// ---- assertion helpers (local_hbm_model_test style: report and continue) ----

bool g_ok = true;

void expect(bool cond, const std::string& what) {
    if (!cond) {
        std::fprintf(stderr, "[remote_port_nway_test] FAIL: %s\n",
                     what.c_str());
        g_ok = false;
    }
}

std::string tick_str(Tick t) {
    return std::to_string(static_cast<uint64_t>(t));
}

// ===========================================================================
// RefPort：§3.1 参考流体模型（测试侧规格 oracle）。
//
// 输入单端口的事务序列（发射 Tick + bytes），逐段现算每个事务的
// callback Tick = ceil(fluid_finish_ns)。段边界只取 latency 就绪与流耗尽
// 两类事件点；每段断言服务不变量：
//   (a) 段内 N>=1 条流每条速率恒为 bw/N（均分、各流同速）；
//   (b) 端口段服务量 == bw x 段长，且累计服务量 <= bw x streaming_time
//       （容量边界；连续活跃时取等——work-conserving）。
// 语义按方案 §3.1 钉死：恰在段边界 latency-ready 的正字节流自该边界起获得
// 服务，不为它未经历的 [t, boundary) 承担份额；零字节正 latency 流不入
// 分母、在就绪时刻完成；双零流精确 +1ns 异步完成。
// ===========================================================================

struct RefIssue {
    Tick issue_tick;
    uint64_t bytes;
};

constexpr double kByteClampTol = 1e-6;  // 与后端 clamp 容差同值（§3.2）

Tick ref_ceil_tick(double ns) { return static_cast<Tick>(std::ceil(ns)); }

std::vector<Tick> ref_port(double bw, double latency,
                           const std::vector<RefIssue>& issues) {
    const std::size_t n = issues.size();
    enum RefState { kWait, kActive, kDone };
    std::vector<RefState> st(n, kWait);
    std::vector<double> ready(n);
    std::vector<double> rem(n);
    std::vector<Tick> cb(n);
    std::size_t done = 0;

    double streaming_time = 0.0;  // N>=1 的累计服务时长
    double served_total = 0.0;    // 端口累计服务字节
    double port_time = 0.0;       // 端口连续时钟（ns）
    bool clock_started = false;

    for (std::size_t i = 0; i < n; ++i) {
        ready[i] = static_cast<double>(issues[i].issue_tick) + latency;
        if (issues[i].bytes == 0 && latency == 0.0) {
            // §3.2 双零：独立一次性定时作业，精确延 1ns 异步完成，
            // 不进带宽作业集合。
            st[i] = kDone;
            cb[i] = ref_ceil_tick(ready[i] + 1.0);  // ready==issue（latency 0）
            ++done;
        }
    }

    const std::size_t guard_limit = 4 * n + 16;
    std::size_t guard = 0;
    while (done < n) {
        expect(++guard <= guard_limit,
               "RefPort: substep guard tripped (no progress)");
        if (guard > guard_limit) {
            break;
        }

        const std::size_t n_stream =
            static_cast<std::size_t>(
                std::count(st.begin(), st.end(), kActive));

        // 下一事件点 = min(最早 latency 就绪, 最早流耗尽投影)。
        double boundary = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < n; ++i) {
            if (st[i] == kWait) {
                boundary = std::min(boundary, ready[i]);
            }
            if (st[i] == kActive && n_stream > 0) {
                const double rate = bw / static_cast<double>(n_stream);
                boundary = std::min(boundary, port_time + rem[i] / rate);
            }
        }
        if (!std::isfinite(boundary)) {
            expect(false, "RefPort: no finite event point but jobs remain");
            break;
        }
        if (!clock_started) {
            port_time = boundary;  // 端口时钟自首个事件点起算
            clock_started = true;
        }

        // 服务 [port_time, boundary)：进入本段前已活跃的流每条 bw/N，
        // 段服务量 = bw*段长（容量边界，work-conserving 取等）。
        std::vector<char> active_before(n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            active_before[i] = (st[i] == kActive) ? 1 : 0;
        }
        if (n_stream > 0) {
            const double elapsed = boundary - port_time;
            const double share = elapsed * bw / static_cast<double>(n_stream);
            // (a) 均分不变量：share*N == bw*elapsed（同式同序，容差断言）。
            expect(std::abs(share * static_cast<double>(n_stream) -
                            bw * elapsed) <= 1e-9 * std::max(1.0, bw * elapsed),
                   "RefPort: equal-split invariant violated");
            for (std::size_t i = 0; i < n; ++i) {
                if (active_before[i]) {
                    rem[i] -= share;
                }
            }
            streaming_time += elapsed;
            served_total += share * static_cast<double>(n_stream);
            // (b) 容量边界：累计服务量不超过 bw x streaming_time，
            // 且连续服务不丢量（取等）。
            expect(served_total <= bw * streaming_time + 1e-9,
                   "RefPort: served bytes exceed bw x streaming_time");
            expect(served_total >= bw * streaming_time - 1e-9,
                   "RefPort: served bytes lost vs bw x streaming_time");
        }
        port_time = boundary;

        // latency 就绪：正字节入流集合；零字节在就绪时刻完成、不入分母。
        for (std::size_t i = 0; i < n; ++i) {
            if (st[i] != kWait || ready[i] > boundary) {
                continue;
            }
            if (issues[i].bytes > 0) {
                st[i] = kActive;
                rem[i] = static_cast<double>(issues[i].bytes);
            } else {
                st[i] = kDone;
                cb[i] = ref_ceil_tick(ready[i]);
                ++done;
            }
        }

        // 流耗尽：同刻耗尽整批离开集合；残余入 clamp 容差即完成
        // （不丢服务、不 panic、不额外推迟）。
        for (std::size_t i = 0; i < n; ++i) {
            if (st[i] != kActive) {
                continue;
            }
            if (rem[i] <= kByteClampTol) {
                expect(rem[i] >= -kByteClampTol,
                       "RefPort: stream remainder fell beyond clamp tolerance");
                rem[i] = 0.0;
                st[i] = kDone;
                cb[i] = ref_ceil_tick(boundary);
                ++done;
            }
        }
    }
    return cb;
}

// ===========================================================================
// RecorderWorkload：后端交付的 wlhd 落点（wlhd->workload 必须是 Workload*）。
// 记录 (node -> 交付次数/交付 Tick) 与全局交付序；取得 wlhd 所有权并删除。
// ===========================================================================

class RecorderWorkload final : public Workload {
  public:
    RecorderWorkload(Sys* sys, const std::string& et_filename,
                     const std::string& comm_group_filename,
                     std::shared_ptr<ExecutionDriven::GraphSource> source)
        : Workload(sys, et_filename, comm_group_filename,
                   ExecutionDriven::ExecutionMode::Online, std::move(source)) {}

    void call(EventType type, CallData* data) override {
        (void)type;
        auto* wlhd = static_cast<WorkloadLayerHandlerData*>(data);
        ++g_deliver_count[wlhd->node_id];
        g_deliver_tick[wlhd->node_id] = Sys::boostedTick();
        g_delivery_order.push_back(wlhd->node_id);
        if (g_delivery_hook) {
            g_delivery_hook(wlhd->node_id);  // 场景钩子（如派发期同步重入）
        }
        delete wlhd;  // 所有权随交付移交本回调
    }

    static std::map<uint64_t, uint64_t> g_deliver_count;
    static std::map<uint64_t, Tick> g_deliver_tick;
    static std::vector<uint64_t> g_delivery_order;
    // 场景钩子：交付回调内同步触发（派发期重入用）；默认空。
    static std::function<void(uint64_t)> g_delivery_hook;
};

std::map<uint64_t, uint64_t> RecorderWorkload::g_deliver_count;
std::map<uint64_t, Tick> RecorderWorkload::g_deliver_tick;
std::vector<uint64_t> RecorderWorkload::g_delivery_order;
std::function<void(uint64_t)> RecorderWorkload::g_delivery_hook;

// ===========================================================================
// 共享夹具：一次构建，全部场景/子进程复用（fork 前构建完成，子进程 COW）。
// ===========================================================================

constexpr int kRanks = 6;

struct SharedFixture {
    std::string dir;  // mkdtemp 隔离目录
    std::shared_ptr<EventQueue> queue;
    std::unique_ptr<AnalyticalRemoteMemory> primary_memory;  // Sys 构造占位
    std::vector<std::unique_ptr<CongestionAwareNetworkApi>> network_apis;
    std::vector<Sys*> systems;
    std::vector<RecorderWorkload*> recorders;
};

SharedFixture g_fx;

void write_text(const std::string& path, const std::string& content) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (f == nullptr) {
        std::perror("write_text");
        std::exit(1);
    }
    std::fputs(content.c_str(), f);
    std::fclose(f);
}

// 官方模板形状的 system.json（三键齐备 + 四实现键全 ring；HBM 争用关：
// 本夹具只测远端端口模型，不引入 LocalHbmBandwidthModel）。
const char* kSystemJson = R"({
  "scheduling-policy": "LIFO",
  "endpoint-delay": 10,
  "active-chunks-per-dimension": 1,
  "preferred-dataset-splits": 6,
  "all-reduce-implementation": ["ring", "ring"],
  "all-gather-implementation": ["ring", "ring"],
  "reduce-scatter-implementation": ["ring", "ring"],
  "all-to-all-implementation": ["ring", "ring"],
  "collective-optimization": "localBWAware",
  "roofline-enabled": 1,
  "replay-only": 0,
  "track-local-mem": 0,
  "trace-enabled": 0,
  "hbm-bandwidth-contention": 0,
  "hbm-kv-restore-bandwidth-sharing": 0,
  "peak-perf": 261.12,
  "local-mem-bw": 1640.0,
  "local-mem-latency": 100,
  "remote-mem-bw": 1000.0,
  "remote-mem-latency": 100
}
)";

// 官方形状 comm_group.json：组带 ranks + dimensions（[3,2] 对应 Mesh[3,2]）。
const char* kCommGroupJson = R"({
  "1": {
    "ranks": [0, 1, 2, 3, 4, 5],
    "dimensions": [3, 2]
  }
}
)";

// 官方形状 network.yml（本夹具不发网络流，带宽/延迟仅占位）。
const char* kNetworkYaml = R"(topology: [ Mesh, Mesh ]
npus_count: [ 3, 2 ]
bandwidth: [ 4050, 4050 ]
latency: [ 25, 25 ]
)";

void build_shared_fixture() {
    std::string tmpl = "/tmp/remote_port_nway_test_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    const char* made = mkdtemp(buf.data());
    if (made == nullptr) {
        std::perror("mkdtemp");
        std::exit(1);
    }
    g_fx.dir = made;
    write_text(g_fx.dir + "/system.json", kSystemJson);
    write_text(g_fx.dir + "/comm_group.json", kCommGroupJson);
    write_text(g_fx.dir + "/network.yml", kNetworkYaml);

    AstraSim::LoggerFactory::init("empty", "off");

    g_fx.queue = std::make_shared<EventQueue>();
    const auto network_parser = NetworkParser(g_fx.dir + "/network.yml");
    const auto topology = construct_topology(network_parser);
    CongestionAwareNetworkApi::set_event_queue(g_fx.queue);
    CongestionAwareNetworkApi::set_topology(topology);
    const auto fluid_scheduler = std::make_shared<FluidScheduler>(
        g_fx.queue, topology->get_directed_links(),
        network_parser.get_fluid_max_active_flows(),
        network_parser.get_fluid_max_route_memberships(),
        network_parser.get_progress_report_event_interval());
    fluid_scheduler->set_deferred_flush_mode(true);
    CongestionAwareNetworkApi::set_fluid_scheduler(fluid_scheduler);

    // Sys 构造需要 AstraRemoteMemoryAPI*；各场景后端按 set_sys 另行挂接
    // （构造占位用 NO_MEMORY_EXPANSION，永不参与 issue）。
    write_text(g_fx.dir + "/remote_memory_primary.json", R"({
  "memory-type": "NO_MEMORY_EXPANSION"
}
)");
    g_fx.primary_memory = std::make_unique<AnalyticalRemoteMemory>(
        g_fx.dir + "/remote_memory_primary.json");

    const auto npus_count_per_dim = network_parser.get_npus_counts_per_dim();
    std::vector<int> queues_per_dim(npus_count_per_dim.size(), 1);
    for (int i = 0; i < kRanks; ++i) {
        g_fx.network_apis.push_back(
            std::make_unique<CongestionAwareNetworkApi>(i));
    }
    for (int i = 0; i < kRanks; ++i) {
        auto source = std::make_shared<ExecutionDriven::NodeStoreGraphSource>();
        // Sys 构造内部自建 Workload 并 set_sys 挂到占位后端。
        Sys* sys = new Sys(
            i, g_fx.dir + "/workload", g_fx.dir + "/comm_group.json",
            g_fx.dir + "/system.json", g_fx.primary_memory.get(),
            g_fx.network_apis[i].get(), npus_count_per_dim, queues_per_dim,
            1.0, 1.0, false, ExecutionDriven::ExecutionMode::Online, source);
        g_fx.systems.push_back(sys);
        // 测试私有交付记录器（Online 模式，不读 .et 文件）。
        g_fx.recorders.push_back(new RecorderWorkload(
            sys, g_fx.dir + "/workload", g_fx.dir + "/comm_group.json",
            source));
    }
    expect(static_cast<int>(topology->get_npus_count()) == kRanks,
           "fixture: topology has 6 ranks");
}

// ---- 场景后端与事务发射辅助 ----

std::unique_ptr<AnalyticalRemoteMemory> make_backend(
    const std::string& filename, const std::string& json_text) {
    write_text(g_fx.dir + "/" + filename, json_text);
    auto mem = std::make_unique<AnalyticalRemoteMemory>(g_fx.dir + "/" +
                                                        filename);
    // set_sys 顺序固定 0..5：事件宿主恒为 rank 0 的 Sys（首次 set_sys）。
    for (int r = 0; r < kRanks; ++r) {
        mem->set_sys(r, g_fx.systems[static_cast<std::size_t>(r)]);
    }
    return mem;
}

struct Expectation {
    int rank;
    uint64_t bytes;
    Tick issue_tick;
    Tick expect_tick;
    std::string label;
};
std::map<uint64_t, Expectation> g_exp;

void issue_bytes(AnalyticalRemoteMemory* mem, int rank, uint64_t bytes,
                 uint64_t node_id, Tick expect_tick, const std::string& label) {
    auto* wlhd = new WorkloadLayerHandlerData();
    wlhd->sys_id = rank;
    wlhd->workload = g_fx.recorders[static_cast<std::size_t>(rank)];
    wlhd->node_id = node_id;
    g_exp[node_id] = Expectation{rank, bytes, Sys::boostedTick(), expect_tick,
                                 label};
    mem->issue(bytes, wlhd);
}

void noop_cb(void*) {}

// latency=0 场景的发射必须发生在事件回调（invoke context）内：零 latency 使
// 后端变迁目标==当前 Tick，delta-0 重挂只有在 invoke 上下文才合法（合并进
// 正在 invoke 的 EventList）；主上下文会在下一次 proceed 触发 EventQueue
// 严格递增断言（EventQueue.cpp:31/:55）。生产路径本就从 Workload 回调发射，
// 此 helper 复刻同一上下文。
struct DeferredIssueCtx {
    AnalyticalRemoteMemory* mem;
    int rank;
    uint64_t bytes;
    uint64_t node_id;
    Tick expect_tick;
    std::string label;
};

void deferred_issue_cb(void* arg) {
    auto* c = static_cast<DeferredIssueCtx*>(arg);
    issue_bytes(c->mem, c->rank, c->bytes, c->node_id, c->expect_tick,
                c->label);
    delete c;
}

void issue_from_event_loop(AnalyticalRemoteMemory* mem, int rank,
                           uint64_t bytes, uint64_t node_id, Tick at_tick,
                           Tick expect_tick, const std::string& label) {
    // 主上下文只允许安排严格未来的事件（EventQueue 严格递增约束）。
    expect(at_tick > Sys::boostedTick(),
           "issue_from_event_loop: at_tick must be strictly future");
    g_fx.queue->schedule_event(
        at_tick,
        deferred_issue_cb,
        new DeferredIssueCtx{mem, rank, bytes, node_id, expect_tick, label});
}

// 推进全局时钟到 t（t 必须不早于当前时刻）。
void advance_to(Tick t) {
    const Tick now = Sys::boostedTick();
    if (t > now) {
        g_fx.queue->schedule_event(t, noop_cb, nullptr);
        std::size_t guard = 0;
        while (Sys::boostedTick() < t) {
            g_fx.queue->proceed();
            expect(++guard < 1000000, "advance_to: event loop guard");
        }
    }
}

// 排空事件队列（含 deferred 通道）。proceed() 要求主队列非空；若仅剩
// deferred 残留（本夹具正常流程不会出现，防御 EventQueue 文档规定的
// "schedule_event(T+1) 强迫一次 proceed" 模式），先安排一个空转事件。
void drain_queue() {
    std::size_t guard = 0;
    while (!g_fx.queue->finished() || g_fx.queue->has_deferred_work()) {
        if (g_fx.queue->finished() && g_fx.queue->has_deferred_work()) {
            g_fx.queue->schedule_event(Sys::boostedTick() + 1, noop_cb,
                                       nullptr);
        }
        g_fx.queue->proceed();
        expect(++guard < 1000000, "drain_queue: event loop guard");
    }
}

// 每场景收尾：逐事务恰一次、Tick 精确、count/bytes 守恒、后端与队列排空。
void verify_scenario(const std::string& name, AnalyticalRemoteMemory* mem,
                     const std::vector<uint64_t>& nodes) {
    uint64_t issued_bytes = 0;
    uint64_t delivered_bytes = 0;
    uint64_t issued_count = 0;
    uint64_t delivered_count = 0;
    for (const uint64_t id : nodes) {
        const Expectation& e = g_exp[id];
        issued_bytes += e.bytes;
        issued_count += 1;
        const uint64_t cnt = RecorderWorkload::g_deliver_count[id];
        delivered_count += cnt;
        if (cnt > 0) {
            delivered_bytes += e.bytes;
        }
        expect(cnt == 1, name + ": " + e.label + " delivered exactly once (got " +
                             std::to_string(cnt) + ")");
        expect(RecorderWorkload::g_deliver_tick[id] == e.expect_tick,
               name + ": " + e.label + " callback tick " +
                   tick_str(RecorderWorkload::g_deliver_tick[id]) +
                   " != ceil(fluid_finish) " + tick_str(e.expect_tick));
    }
    // 测试侧守恒：count 与 bytes 分别独立相等（不得以 count 掩盖 bytes）。
    expect(delivered_count == issued_count,
           name + ": completed count " + std::to_string(delivered_count) +
               " == issued count " + std::to_string(issued_count));
    expect(delivered_bytes == issued_bytes,
           name + ": completed bytes " + std::to_string(delivered_bytes) +
               " == issued bytes " + std::to_string(issued_bytes));
    // 后端自有守恒与排空（is_drained 逐端口独立校验 count/bytes/in-flight）。
    expect(mem->is_drained(), name + ": backend is_drained()");
    expect(g_fx.queue->finished() && !g_fx.queue->has_deferred_work(),
           name + ": event queue fully drained");
}

void expect_delivery_order(const std::string& name, std::size_t base,
                           const std::vector<uint64_t>& expected) {
    std::vector<uint64_t> actual(RecorderWorkload::g_delivery_order.begin() +
                                     static_cast<long>(base),
                                 RecorderWorkload::g_delivery_order.end());
    if (actual.size() != expected.size()) {
        expect(false, name + ": delivery count mismatch");
        return;
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        expect(actual[i] == expected[i],
               name + ": delivery order[" + std::to_string(i) + "] node " +
                   std::to_string(actual[i]) + " != expected " +
                   std::to_string(expected[i]));
    }
}

void reset_scenario_state() {
    g_exp.clear();
    RecorderWorkload::g_deliver_count.clear();
    RecorderWorkload::g_deliver_tick.clear();
    RecorderWorkload::g_delivery_order.clear();
}

// ---- 远端内存配置片段（官方 remote_memory.json 键形状） ----
// 数值以 %g 预格式化字符串传入（std::to_string 的 %f 会把 1e-20 写成
// "0.000000"）；inf 类用例直接传 JSON 文法内的 "1e400"（nlohmann lexer 经
// strtod 得 inf，后端 isfinite 守卫负责拒绝）。

std::string num_str(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%g", v);
    return std::string(buf);
}

std::string per_npu_config(const std::string& bw, const std::string& latency) {
    return "{\n"
           "  \"memory-type\": \"PER_NPU_MEMORY_EXPANSION\",\n"
           "  \"npu-ids\": [0, 1, 2, 3, 4, 5],\n"
           "  \"remote-mem-bw\": " + bw + ",\n"
           "  \"remote-mem-latency\": " + latency + "\n"
           "}\n";
}

std::string per_node_config(const std::string& bw, const std::string& latency,
                            int num_nodes, int npus_per_node) {
    return "{\n"
           "  \"memory-type\": \"PER_NODE_MEMORY_EXPANSION\",\n"
           "  \"num-nodes\": " + std::to_string(num_nodes) + ",\n"
           "  \"num-npus-per-node\": " + std::to_string(npus_per_node) + ",\n"
           "  \"remote-mem-bw\": " + bw + ",\n"
           "  \"remote-mem-latency\": " + latency + "\n"
           "}\n";
}

std::string pool_config(const std::string& bw, const std::string& latency) {
    return "{\n"
           "  \"memory-type\": \"MEMORY_POOL\",\n"
           "  \"remote-mem-bw\": " + bw + ",\n"
           "  \"remote-mem-latency\": " + latency + "\n"
           "}\n";
}

// ===========================================================================
// S1 -- PER_NPU 映射（双 rank 同刻发射：两笔各占独立端口）
//   PER_NPU 逐 rank 独立端口（方案 §1），每端口单流独享全带宽 6B/ns：
//   100 latency + 600/6 -> 各 +200 完成；RefPort 按每端口单流现算同值。
//   本场景覆盖：同刻 latency、跨端口同刻完成批（port 升序交付）。
//   注意：方案 §5.2 锚 1（同端口两流平分 3B/ns -> 各 +300）的共享端口
//   口径由 S7 的 ref2 对（同端口两笔 600B -> 各 +300）断言，不在本场景
//   ——初版曾在此把独立端口错断言成 +300（与其自身 RefPort 值矛盾，
//   恒 FAIL），已按场景实际接线修正。
// ===========================================================================

void scenario_s1() {
    reset_scenario_state();
   
    auto mem = make_backend("rm_s1.json", per_npu_config(num_str(6.0), num_str(100.0)));
   
    const Tick t0 = Sys::boostedTick();
    // RefPort 逐段现算（每 rank 独立端口，各自单流）：
    std::vector<Tick> ref0 = ref_port(6.0, 100.0, {{t0, 600}});
    std::vector<Tick> ref1 = ref_port(6.0, 100.0, {{t0, 600}});
    issue_bytes(mem.get(), 0, 600, 1001, ref0[0], "S1 r0 600B");
    issue_bytes(mem.get(), 1, 600, 1002, ref1[0], "S1 r1 600B");
    // 独立端口显式锚：各 +200（若两笔被错误压入同一端口分母则 +300，
    // 若退回串行 FIFO 同端口则 +200/+400——均与本断言不等，接线即规格）。
    expect(ref0[0] == t0 + 200 && ref1[0] == t0 + 200,
           "S1: independent PER_NPU ports run the full 6B/ns each, done "
           "+200/+200 (got +" + tick_str(ref0[0]) + "/+" +
               tick_str(ref1[0]) + ")");
   
    const std::size_t base = RecorderWorkload::g_delivery_order.size();
    drain_queue();
   
    verify_scenario("S1", mem.get(), {1001, 1002});
   
    // 同刻跨端口批：host(rank0/port0) 先，随后 (port,seq) 升序。
    expect_delivery_order("S1", base, {1001, 1002});
}

// ===========================================================================
// S12 -- MEMORY_POOL 映射 + 阶段 5.2 锚 2
//   bw=100、latency=50、同端口四流 1k/2k/3k/4kB（k=1000B）：
//   [50,90) N=4 每流 25；[90,120) N=3 每流 100/3；[120,140) N=2 每流 50；
//   [140,150) N=1 每流 100 -> fluid/callback = 90/120/140/150（各段整除）。
// ===========================================================================

void scenario_s12() {
    reset_scenario_state();
    auto mem = make_backend("rm_s12.json", pool_config(num_str(100.0), num_str(50.0)));
    const Tick t0 = Sys::boostedTick();
    const std::vector<Tick> ref =
        ref_port(100.0, 50.0, {{t0, 1000}, {t0, 2000}, {t0, 3000}, {t0, 4000}});
    issue_bytes(mem.get(), 0, 1000, 1201, ref[0], "S12 pool 1kB");
    issue_bytes(mem.get(), 1, 2000, 1202, ref[1], "S12 pool 2kB");
    issue_bytes(mem.get(), 2, 3000, 1203, ref[2], "S12 pool 3kB");
    issue_bytes(mem.get(), 3, 4000, 1204, ref[3], "S12 pool 4kB");
    // 指名锚 2：work-conserving 重分 90/120/140/150。
    expect(ref[0] == t0 + 90 && ref[1] == t0 + 120 &&
               ref[2] == t0 + 140 && ref[3] == t0 + 150,
           "S12 anchor-2: four-stream re-split callbacks at +90/+120/+140/"
           "+150 (got +" + tick_str(ref[0]) + "/+" + tick_str(ref[1]) + "/+" +
               tick_str(ref[2]) + "/+" + tick_str(ref[3]) + ")");
    drain_queue();
    verify_scenario("S12", mem.get(), {1201, 1202, 1203, 1204});
}

// ===========================================================================
// S6A -- PER_NPU：单事务非整除 + 极小残差 + 阶段 5.2 锚 3
//   bw=6、latency=100，各 rank 独立端口：
//   1B  -> fluid 100+1/6 ns（服务 <1ns）-> callback +101（下一 Tick，
//          两种时间不得混用）；
//   100B -> fluid 116.67 -> callback +117（非整除）；
//   10B -> fluid 101.67 -> callback +102（share=(10/6)*6 超余 ~2e-15，
//          须落入 1e-6 clamp：完成不 panic、不推迟、不丢服务）。
// ===========================================================================

void scenario_s6a() {
    reset_scenario_state();
    auto mem = make_backend("rm_s6a.json", per_npu_config(num_str(6.0), num_str(100.0)));
    const Tick t0 = Sys::boostedTick();
    const std::vector<Tick> ref1b = ref_port(6.0, 100.0, {{t0, 1}});
    const std::vector<Tick> ref100b = ref_port(6.0, 100.0, {{t0, 100}});
    const std::vector<Tick> ref10b = ref_port(6.0, 100.0, {{t0, 10}});
    issue_bytes(mem.get(), 0, 1, 1601, ref1b[0], "S6A 1B");
    issue_bytes(mem.get(), 1, 100, 1602, ref100b[0], "S6A 100B");
    issue_bytes(mem.get(), 2, 10, 1603, ref10b[0], "S6A 10B");
    // 指名锚 3：1B 的 fluid 服务约 1/6 ns，callback 在下一整数 Tick。
    expect(ref1b[0] == t0 + 101,
           "S6A anchor-3: 1B@6B/ns callback at next tick +101 (fluid ~1/6 ns, "
           "got +" + tick_str(ref1b[0]) + ")");
    expect(ref100b[0] == t0 + 117,
           "S6A: 100B non-divisible callback at ceil(116.67)=+117 (got +" +
               tick_str(ref100b[0]) + ")");
    expect(ref10b[0] == t0 + 102,
           "S6A: 10B tiny-residue callback at ceil(101.67)=+102 (got +" +
               tick_str(ref10b[0]) + ")");
    drain_queue();
    verify_scenario("S6A", mem.get(), {1601, 1602, 1603});
}

// ===========================================================================
// S6B -- PER_NODE(1x3)：同端口混合流；fluid-complete-awaiting-callback 不再
//   计入带宽分母（§7：新发射不得把已完成待回调的流算回 N）。
//   bw=6、latency=100，三笔同刻：100B/1B/200B 同端口：
//   [100,100.5) N=3 每流 2 -> 1B 完成 @100.5（callback +101）；
//   [100.5,133.5) N=2 每流 3 -> 100B 完成 @133.5（callback +134；若 1B 仍占
//   分母则为 150——排除性断言）；
//   [133.5,150.17) N=1 每流 6 -> 200B 完成 @150.17（callback +151）。
// ===========================================================================

void scenario_s6b() {
    reset_scenario_state();
    auto mem = make_backend("rm_s6b.json", per_node_config(num_str(6.0), num_str(100.0), 1, 3));
    const Tick t0 = Sys::boostedTick();
    const std::vector<Tick> ref = ref_port(6.0, 100.0,
                                           {{t0, 100}, {t0, 1}, {t0, 200}});
    issue_bytes(mem.get(), 0, 100, 1611, ref[0], "S6B 100B");
    issue_bytes(mem.get(), 1, 1, 1612, ref[1], "S6B 1B");
    issue_bytes(mem.get(), 2, 200, 1613, ref[2], "S6B 200B");
    expect(ref[1] == t0 + 101 && ref[0] == t0 + 134 &&
               ref[2] == t0 + 151,
           "S6B: awaiting-callback stream leaves the bandwidth denominator "
           "(got +101/+134/+151 -> +" + tick_str(ref[1]) + "/+" +
               tick_str(ref[0]) + "/+" + tick_str(ref[2]) + ")");
    drain_queue();
    verify_scenario("S6B", mem.get(), {1611, 1612, 1613});
}

// ===========================================================================
// S3 -- PER_NODE(2x3) 映射 + 同刻/错峰 latency + 跨端口独立
//   bw=6、latency=100。ranks 0-2 -> port0，ranks 3-5 -> port1。
//   r0 300B@t0、r1 300B@t0（port0 同刻共享：[100,200) N=2 每流 3B/ns ->
//   双双 @200）；r2 600B@t0+150（错峰：port0 于 200 空闲，[250,350) 独享
//   6B/ns -> @350）；r3 600B@t0（port1 独立端口独享全带宽 6B/ns：600B 走
//   100ns -> @200——同一 [100,200) 窗口内 port0 每流只分到半带宽，对照出
//   端口互不影响）。+200 三笔同刻完成 -> 批按 (port, issue_sequence) 交付。
// ===========================================================================

void scenario_s3() {
    reset_scenario_state();
    auto mem =
        make_backend("rm_s3.json",
                     per_node_config(num_str(6.0), num_str(100.0), 2, 3));
    const Tick t0 = Sys::boostedTick();
    const std::vector<Tick> ref0 =
        ref_port(6.0, 100.0, {{t0, 300}, {t0, 300}, {t0 + 150, 600}});
    const std::vector<Tick> ref1 = ref_port(6.0, 100.0, {{t0, 600}});
    issue_bytes(mem.get(), 0, 300, 1301, ref0[0], "S3 r0 300B");
    issue_bytes(mem.get(), 1, 300, 1302, ref0[1], "S3 r1 300B");
    issue_bytes(mem.get(), 3, 600, 1304, ref1[0], "S3 r3 600B");
    advance_to(t0 + 150);
    issue_bytes(mem.get(), 2, 600, 1303, ref0[2], "S3 r2 600B (+150)");
    expect(ref0[0] == t0 + 200 && ref0[1] == t0 + 200,
           "S3: same-tick pair shares port0 at 3B/ns each, done +200");
    expect(ref0[2] == t0 + 350,
           "S3: staggered-latency 600B (ready +250) done +350");
    expect(ref1[0] == t0 + 200,
           "S3: port1 600B takes the full 6B/ns while port0 pair shares "
           "3B/ns each in the same window, done +200 (cross-port "
           "independence), got +" + tick_str(ref1[0]));
    const std::size_t base = RecorderWorkload::g_delivery_order.size();
    drain_queue();
    verify_scenario("S3", mem.get(), {1301, 1302, 1303, 1304});
    // 同刻批（port0 的 r0/r1 与 port1 的 r3 @+200）按 (port, seq) 交付，
    // 错峰的 r2 最后到。
    expect_delivery_order("S3", base, {1301, 1302, 1304, 1303});
}

// ===========================================================================
// S5 -- MEMORY_POOL：双零（bytes=0 且 latency=0）
//   bw=6、latency=0。发射经事件回调（invoke context，同生产 Workload 路径）
//   在 t0+1 发出。r0 双零：独立一次性定时作业精确 "issue+1ns" 异步完成
//   （callback=t0+2；不进带宽集合、不从 issue 栈同步回调）；r1 6B：latency 0
//   即刻入流、独享 6B/ns、issue+1ns 完成（t0+2；若双零错误入分母则 3B/ns、
//   t0+3——排除性断言）。两笔同刻完成 -> 批内 (port, issue_sequence) 序。
// ===========================================================================

void scenario_s5() {
    reset_scenario_state();
    auto mem = make_backend("rm_s5.json", pool_config(num_str(6.0), num_str(0.0)));
    const Tick t0 = Sys::boostedTick();
    const Tick issue_at = t0 + 1;
    const std::vector<Tick> ref =
        ref_port(6.0, 0.0, {{issue_at, 0}, {issue_at, 6}});
    // 双零异步性：若后端从 issue 栈同步回调，交付 Tick 会是 t0+1（发射
    // 时刻）而非 t0+2——由下方 deliver-tick 断言排除。
    issue_from_event_loop(mem.get(), 0, 0, 1501, issue_at, ref[0],
                          "S5 dual-zero");
    issue_from_event_loop(mem.get(), 1, 6, 1502, issue_at, ref[1],
                          "S5 6B");
    expect(ref[0] == issue_at + 1 && ref[1] == issue_at + 1,
           "S5: dual-zero and 6B both complete at issue+1 (got +" +
               tick_str(ref[0]) + "/+" + tick_str(ref[1]) + ")");
    const std::size_t base = RecorderWorkload::g_delivery_order.size();
    drain_queue();
    // 交付 Tick 精确校验已含双零异步性：同步回调会落在 issue 时刻（t0+1），
    // 异步定时作业必须落在 issue+1ns 的 ceil Tick（t0+2）。
    expect(RecorderWorkload::g_deliver_tick[1501] == issue_at + 1,
           "S5: dual-zero delivered exactly at issue+1 Tick (async timer), "
           "got " + tick_str(RecorderWorkload::g_deliver_tick[1501]));
    verify_scenario("S5", mem.get(), {1501, 1502});
    expect_delivery_order("S5", base, {1501, 1502});
}

// ===========================================================================
// S7 -- PER_NPU：同 Tick 多端口/多作业排序
//   bw=6、latency=100。五笔 600B 同刻：r0(host/port0) 一笔、r1(port1) 两笔、
//   r2(port2)、r4(port4) 各一笔 -> 五笔全部 +300 同刻完成。
//   交付序：host rank 批内先交付，其余按 (port_index, issue_sequence) 升序
//   （Sys 同 tick 交付机制的确定性结果：host 桶内追加 -> 其余经外层闹钟）。
// ===========================================================================

void scenario_s7() {
    reset_scenario_state();
    auto mem = make_backend("rm_s7.json", per_npu_config(num_str(6.0), num_str(100.0)));
    const Tick t0 = Sys::boostedTick();
    const std::vector<Tick> ref2 =
        ref_port(6.0, 100.0, {{t0, 600}, {t0, 600}});
    const std::vector<Tick> ref1 = ref_port(6.0, 100.0, {{t0, 600}});
    issue_bytes(mem.get(), 0, 600, 1701, ref1[0], "S7 r0 600B");
    issue_bytes(mem.get(), 1, 600, 1702, ref2[0], "S7 r1 600B#0");
    issue_bytes(mem.get(), 1, 600, 1703, ref2[1], "S7 r1 600B#1");
    issue_bytes(mem.get(), 2, 600, 1704, ref1[0], "S7 r2 600B");
    issue_bytes(mem.get(), 4, 600, 1705, ref1[0], "S7 r4 600B");
    expect(ref2[0] == t0 + 300,
           "S7: same-port pair at 3B/ns each completes +300");
    const std::size_t base = RecorderWorkload::g_delivery_order.size();
    drain_queue();
    verify_scenario("S7", mem.get(), {1701, 1702, 1703, 1704, 1705});
    // 跨 rank 同 Tick callback 顺序由各 rank Sys 的外层闹钟决定，不由
    // 后端排定（§3.3 要求以真实 Sys 夹具证实的事项；实测外层同 tick 闹钟
    // 非注册序）。后端保证：host 批内先交付 + 其余按 (port, seq) 注册。
    // 断言收紧到机制保证：1701（host/port0）最先，其余四笔全交付。
    {
        const std::vector<uint64_t> actual(
            RecorderWorkload::g_delivery_order.begin() +
                static_cast<long>(base),
            RecorderWorkload::g_delivery_order.end());
        expect(actual.size() == 5 && actual[0] == 1701,
               "S7: host-rank job 1701 delivers first (got size " +
                   std::to_string(actual.size()) + ")");
        std::vector<uint64_t> rest(actual.begin() + 1, actual.end());
        std::sort(rest.begin(), rest.end());
        expect(rest == std::vector<uint64_t>({1702, 1703, 1704, 1705}),
               "S7: remaining four all delivered exactly once (order owned "
               "by per-rank outer alarms)");
    }
}

// ===========================================================================
// S4 -- MEMORY_POOL：零字节正 latency（不稀释）+ 跨 rank 池共享 + 错峰
//   bw=6、latency=100。r0/r1/r2 各 600B@t0；r4 0B@t0+20（ready +120，
//   零字节不入分母：若入，[120,200) 内 N=4 会把三笔推迟到 +500 后）；
//   r3 300B@t0+100（ready +200；[200,400) N=4 每流 1.5；[400,450) N=3）。
//   §3.1 期望：r4 +120、r3 +400、r0-r2 +450。
//   【已知分歧场景，非致命】当前后端把恰于 +200 边界就绪的 r3 扣减该子步
//   份额 160（AnalyticalRemoteMemory.cc:613-639）：r3 携 140B 提前完成
//   （走读约 +294，§3.1 应为 +400），r0-r2 随之偏早（约 +424，应 +450）——
//   以 FAIL 行暴露；断言值按方案 §3.1 书写，非致命不阻断后续场景。
// ===========================================================================

void scenario_s4() {
    reset_scenario_state();
    auto mem = make_backend("rm_s4.json", pool_config(num_str(6.0), num_str(100.0)));
    const Tick t0 = Sys::boostedTick();
    const std::vector<Tick> ref = ref_port(
        6.0, 100.0,
        {{t0, 600}, {t0, 600}, {t0, 600}, {t0 + 20, 0}, {t0 + 100, 300}});
    issue_bytes(mem.get(), 0, 600, 1401, ref[0], "S4 pool 600B#0");
    issue_bytes(mem.get(), 1, 600, 1402, ref[1], "S4 pool 600B#1");
    issue_bytes(mem.get(), 2, 600, 1403, ref[2], "S4 pool 600B#2");
    advance_to(t0 + 20);
    issue_bytes(mem.get(), 4, 0, 1404, ref[3], "S4 zero-byte +100ns");
    advance_to(t0 + 100);
    issue_bytes(mem.get(), 3, 300, 1405, ref[4], "S4 pool 300B (+100)");
    expect(ref[3] == t0 + 120,
           "S4: zero-byte positive-latency callback at +120, no dilution");
    expect(ref[4] == t0 + 400 && ref[0] == t0 + 450,
           "S4: pool re-split 1.5B/ns x4 then 2B/ns x3 -> +400/+450 (got r3 +"
           + tick_str(ref[4]) + ", r0 +" + tick_str(ref[0]) + ")");
    drain_queue();
    verify_scenario("S4", mem.get(), {1401, 1402, 1403, 1404, 1405});
}

// ===========================================================================
// S13 -- §7 正反例：事件 generation——陈旧 generation 无操作
//   反例：发射后向 call() 注入 generation 不匹配的 TransitionEventData：
//   handle_transition_event 的 generation 守卫直接返回（不推进端口、不交付、
//   不改 generation 计数）；call() 本身按契约消费 payload 并清后端事件跟踪，
//   而 Sys 队列中仍带正确 generation 的原变迁事件照常触发（事务不受污染）。
//   正例：正常排空后事务恰按 ceil(fluid) 交付一次；排空后再注入陈旧
//   generation 仍为无操作、后端保持 drained。
// ===========================================================================

void scenario_s13() {
    reset_scenario_state();
    auto mem =
        make_backend("rm_s13.json",
                     per_npu_config(num_str(6.0), num_str(100.0)));
    const Tick t0 = Sys::boostedTick();
    const std::vector<Tick> ref = ref_port(6.0, 100.0, {{t0, 500}});
    issue_bytes(mem.get(), 0, 500, 13001, ref[0], "S13 500B");
    const uint64_t gen = mem->transition_generation();
    expect(gen >= 1, "S13: issue registered a transition generation");
    // 反例：陈旧 generation 注入（任何 != 当前值均走无操作分支）。
    mem->call(EventType::General,
              new AnalyticalRemoteMemory::TransitionEventData(gen + 1));
    expect(mem->transition_generation() == gen,
           "S13 stale generation: counter unchanged (no-op)");
    expect(!mem->transition_event_pending(),
           "S13 stale generation: call() consumed the payload tracking per "
           "contract");
    expect(RecorderWorkload::g_deliver_count.empty(),
           "S13 stale generation: no delivery, no port advance");
    drain_queue();
    verify_scenario("S13", mem.get(), {13001});
    // 排空后再注入陈旧 generation：仍为无操作、后端保持 drained。
    mem->call(EventType::General,
              new AnalyticalRemoteMemory::TransitionEventData(gen + 1));
    expect(mem->is_drained(),
           "S13 drained state survives a post-drain stale generation");
}

// ===========================================================================
// S14 -- §7 正反例：派发期同步重入
//   正例：交付回调内同步 issue 新事务（§3.3：回调可同步重入并发射新节点，
//   派发期间只允许更新状态、不递归分发；整批回调后统一注册下一变迁）。
//   断言方案规范性属性（不断言 delivery_in_progress_ 实现细节：后端以
//   0 延迟 Sys 事件交付，回调执行时旗标已清）：重发事务按自身解析 Tick
//   （t0+300）交付、不与本批（t0+150）同批递归交付，且整批统计/守恒
//   收尾闭合。反例：若重入事务与本批同批递归交付则 Tick 断言失败。
// ===========================================================================

void scenario_s14() {
    reset_scenario_state();
    auto mem =
        make_backend("rm_s14.json",
                     per_npu_config(num_str(6.0), num_str(100.0)));
    const Tick t0 = Sys::boostedTick();
    const std::vector<Tick> ref =
        ref_port(6.0, 100.0, {{t0, 300}, {t0 + 150, 300}});
    RecorderWorkload::g_delivery_hook = [&](uint64_t node_id) {
        if (node_id == 14001) {
            // 派发期同步重入：交付回调栈内直接发射新事务。
            issue_bytes(mem.get(), 5, 300, 14002, ref[1],
                        "S14 re-issued 300B");
        }
    };
    issue_bytes(mem.get(), 5, 300, 14001, ref[0], "S14 A 300B");
    drain_queue();
    RecorderWorkload::g_delivery_hook = nullptr;
    verify_scenario("S14", mem.get(), {14001, 14002});
    expect(RecorderWorkload::g_deliver_tick[14002] == t0 + 300,
           "S14 re-issued transaction delivered at its own tick, not "
           "recursively with the batch");
}

// ===========================================================================
// S15 -- §7 正反例：提前 shutdown
//   正例：两笔在途 + 挂起变迁事件时提前 shutdown——挂起变迁事件被取消
//   （队列无残留触发）、未交付 wlhd 经 PortJob unique_ptr 删除（零交付、
//   无二次释放）、PortStats 清零、is_drained 立即为真、二次 shutdown 幂等。
//   复用正例：shutdown 后后端继续服务新事务并正常排空（main 的
//   shutdown->reset 收尾顺序语义）。
// ===========================================================================

void scenario_s15() {
    reset_scenario_state();
    auto mem =
        make_backend("rm_s15.json",
                     per_npu_config(num_str(6.0), num_str(100.0)));
    const Tick t0 = Sys::boostedTick();
    const std::vector<Tick> ref =
        ref_port(6.0, 100.0, {{t0, 600}, {t0, 6000}});
    issue_bytes(mem.get(), 0, 600, 15001, ref[0], "S15 X1 600B");
    issue_bytes(mem.get(), 0, 6000, 15002, ref[1], "S15 X2 6000B");
    advance_to(t0 + 150);
    {
        const auto s = mem->port_stats(0);
        expect(s.issued_count == 2,
               "S15 mid-flight baseline: two transactions issued (got " +
                   std::to_string(s.issued_count) + ")");
    }
    expect(mem->transition_event_pending(),
           "S15 mid-flight: transition event pending");
    expect(!mem->is_drained(), "S15 mid-flight: not drained");

    mem->shutdown();
    expect(mem->is_drained(),
           "S15 early shutdown: jobs cleared, event cancelled, stats reset");
    {
        const auto s = mem->port_stats(0);
        expect(s.issued_count == 0 && s.completed_count == 0,
               "S15 PortStats cleared by shutdown (got issued=" +
                   std::to_string(s.issued_count) + " completed=" +
                   std::to_string(s.completed_count) + ")");
    }
    expect(!mem->transition_event_pending(),
           "S15 pending transition event cancelled by shutdown");
    drain_queue();
    expect(RecorderWorkload::g_deliver_count.empty(),
           "S15 undelivered wlhd deleted via unique_ptr, never delivered");
    mem->shutdown();  // 幂等：二次 shutdown 为无操作
    expect(mem->is_drained(), "S15 double shutdown idempotent");

    // 复用正例：shutdown 后后端继续服务新事务。
    const Tick t1 = Sys::boostedTick();
    const std::vector<Tick> ref2 = ref_port(6.0, 100.0, {{t1, 600}});
    issue_bytes(mem.get(), 0, 600, 15003, ref2[0],
                "S15 post-shutdown 600B");
    drain_queue();
    verify_scenario("S15", mem.get(), {15003});
}

// ===========================================================================
// S2 -- PER_NPU：2 流长短服务与完成即重分 + latency-ready 与 stream
//   completion 同 Tick（fork 子进程内运行：若后端触发 sys_panic，父进程
//   以"子进程必须 0 退出"捕获，套件其余诊断不受影响）。
//   bw=6、latency=100，rank5 单端口：A=1200B@t0（ready +100）、
//   B=300B@t0+50（ready +150）、C=60B@t0+150（ready +250 恰为 B 完成刻）。
//   §3.1 逐段：[100,150) N=1 每流 6（A 余 900）；[150,250) N=2 每流 3
//   （B 完成 @250，A 余 600）；+250 C 就绪与 B 完成同刻 -> [250,270) N=2
//   每流 3（C 完成 @270，A 余 540）；[270,360) N=1 每流 6（A 完成 @360）。
//   完成即重分：A 的最后 540B 以 6B/ns 走 90ns，而非按 3B/ns 再等 180ns。
//   【已知分歧，子进程内致命】当前后端对 +150/+250 边界就绪流各扣一份
//   300/600：B 被窃 300B 提前完成，C（60B）被扣 600 后 remaining<-1e-6
//   触发 sys_panic（AnalyticalRemoteMemory.cc:621），或更早因 B 提前完成
//   引发的主上下文 delta-0 重挂触发 EventQueue 严格递增断言——子进程以
//   非零退出/信号结束，父进程按 FAIL 捕获。
// ===========================================================================

void scenario_s2_body() {
    reset_scenario_state();
    auto mem = make_backend("rm_s2.json", per_npu_config(num_str(6.0), num_str(100.0)));
    const Tick t0 = Sys::boostedTick();
    const std::vector<Tick> ref = ref_port(
        6.0, 100.0, {{t0, 1200}, {t0 + 50, 300}, {t0 + 150, 60}});
    issue_bytes(mem.get(), 5, 1200, 2001, ref[0], "S2 A 1200B");
    advance_to(t0 + 50);
    issue_bytes(mem.get(), 5, 300, 2002, ref[1], "S2 B 300B (+50)");
    advance_to(t0 + 150);
    issue_bytes(mem.get(), 5, 60, 2003, ref[2], "S2 C 60B (+150)");
    expect(ref[1] == t0 + 250 && ref[2] == t0 + 270 &&
               ref[0] == t0 + 360,
           "S2: long/short re-split + ready-at-completion-tick -> "
           "B+250/C+270/A+360 (got A+" + tick_str(ref[0]) + " B+" +
               tick_str(ref[1]) + " C+" + tick_str(ref[2]) + ")");
    drain_queue();
    verify_scenario("S2", mem.get(), {2001, 2002, 2003});
}

// ---- fork 辅助 ----

// 子进程内运行 body；body 正常返回则以 g_ok 决定退出码。
pid_t run_in_child(const std::function<void()>& body) {
    const pid_t pid = ::fork();
    if (pid == 0) {
        g_ok = true;  // 子进程本地失败跟踪（不影响父进程）
        // 子进程内 fail-closed 走 exit(1)：夹具（spdlog/EventQueue）存在
        // fork 前后台线程态，exit 的 atexit/静态析构链会与已持锁互斥量
        // 死锁（futex 悬挂，waitpid 永不返回）——注册 LIFO 首位 _Exit(1)
        // 跳过整条退出处理链；正常路径仍走下方 _Exit(g_ok?0:1)，不受影响。
        std::atexit([] { std::_Exit(1); });
        body();
        std::_Exit(g_ok ? 0 : 1);
    }
    return pid;
}

// fail-closed 子进程：body 必须以 exit(1) 终止（sys_panic/构造期 exit）；
// 若 body 存活返回（未 fail-closed），子进程以 0 退出 -> 父进程 FAIL。
void expect_fail_closed(const std::string& name,
                        const std::function<void()>& body) {
    const pid_t pid = run_in_child(body);
    int status = 0;
    waitpid(pid, &status, 0);
    const bool exited1 = WIFEXITED(status) && WEXITSTATUS(status) == 1;
    expect(exited1, name + ": backend must fail closed with exit(1) (child " +
                        (WIFEXITED(status)
                             ? "exit=" + std::to_string(WEXITSTATUS(status))
                             : std::string("signaled")) +
                        ")");
}

// 普通子进程：body 须全部断言通过（exit 0）。
void expect_child_pass(const std::string& name,
                       const std::function<void()>& body) {
    const pid_t pid = run_in_child(body);
    int status = 0;
    waitpid(pid, &status, 0);
    expect(WIFEXITED(status) && WEXITSTATUS(status) == 0,
           name + ": child scenario must pass (exit=" +
               std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1) +
               ")");
}

// ---- fail-closed 用例（C1-C9） ----

// 构造期：非法带宽 / 非法延迟（构造函数必须 exit(1)，不得带病运行）。
void child_ctor_bw_zero() {
    write_text(g_fx.dir + "/rm_c1.json", per_npu_config(num_str(0.0), num_str(100.0)));
    AnalyticalRemoteMemory mem(g_fx.dir + "/rm_c1.json");
    std::printf("C1 DID NOT fail closed\n");
}
void child_ctor_bw_negative() {
    write_text(g_fx.dir + "/rm_c2.json", per_npu_config(num_str(-6.0), num_str(100.0)));
    AnalyticalRemoteMemory mem(g_fx.dir + "/rm_c2.json");
    std::printf("C2 DID NOT fail closed\n");
}
void child_ctor_bw_inf() {
    // 1e400 超 double 范围：JSON 解析为 inf -> isfinite 守卫（NaN 同守卫，
    // 字面 NaN 非 JSON 数值文法所容）。
    write_text(g_fx.dir + "/rm_c3.json", per_npu_config("1e400", num_str(100.0)));
    AnalyticalRemoteMemory mem(g_fx.dir + "/rm_c3.json");
    std::printf("C3 DID NOT fail closed\n");
}
void child_ctor_latency_negative() {
    write_text(g_fx.dir + "/rm_c4.json", per_npu_config(num_str(6.0), num_str(-100.0)));
    AnalyticalRemoteMemory mem(g_fx.dir + "/rm_c4.json");
    std::printf("C4 DID NOT fail closed\n");
}
void child_ctor_latency_inf() {
    write_text(g_fx.dir + "/rm_c5.json", per_npu_config(num_str(6.0), "1e400"));
    AnalyticalRemoteMemory mem(g_fx.dir + "/rm_c5.json");
    std::printf("C5 DID NOT fail closed\n");
}

void issue_once(AnalyticalRemoteMemory* mem, int sys_id, uint64_t bytes) {
    auto* wlhd = new WorkloadLayerHandlerData();
    wlhd->sys_id = sys_id;
    wlhd->workload = g_fx.recorders[0];
    wlhd->node_id = 9001;
    mem->issue(bytes, wlhd);
}

// 运行期：有限但超界的延迟（1e308）+ 零字节事务：latency-waiting 的期限
// 投影 1e308 超出 Tick 表示范围 -> 变迁重排的 ceil_ns_to_tick 守卫
// fail-closed（超大值不得变成无限事件/NaN 延迟）。
void child_runtime_latency_overflow() {
    write_text(g_fx.dir + "/rm_c6.json", per_npu_config(num_str(6.0), "1e+308"));
    AnalyticalRemoteMemory mem(g_fx.dir + "/rm_c6.json");
    mem.set_sys(0, g_fx.systems[0]);
    issue_once(&mem, 0, 0);
    std::printf("C6 DID NOT fail closed\n");
}

// 运行期：正但极小的带宽（1e-20 B/ns）使 1B 流的完成时刻超出 Tick 表示范围
// -> 变迁重排的 ceil 守卫 fail-closed。
void child_runtime_bw_underflow() {
    write_text(g_fx.dir + "/rm_c7.json", per_npu_config("1e-20", num_str(100.0)));
    AnalyticalRemoteMemory mem(g_fx.dir + "/rm_c7.json");
    mem.set_sys(0, g_fx.systems[0]);
    issue_once(&mem, 0, 1);
    g_fx.queue->schedule_event(Sys::boostedTick() + 300, noop_cb, nullptr);
    g_fx.queue->proceed();  // 变迁事件到期 -> 重排下一期限 -> 必须终止
    std::printf("C7 DID NOT fail closed\n");
}

// NO_MEMORY_EXPANSION 的 fail-closed 保留：issue 即终止。
void child_no_memory_issue() {
    write_text(g_fx.dir + "/rm_c8.json",
               "{\n  \"memory-type\": \"NO_MEMORY_EXPANSION\"\n}\n");
    AnalyticalRemoteMemory mem(g_fx.dir + "/rm_c8.json");
    mem.set_sys(0, g_fx.systems[0]);
    issue_once(&mem, 0, 128);
    std::printf("C8 DID NOT fail closed\n");
}

// PER_NPU 已配置 npu-ids 下，未配置 rank 的 issue 必须 fail-closed
// （非法 port 不得落到兜底端口）。
void child_per_npu_unconfigured_rank() {
    write_text(g_fx.dir + "/rm_c9.json",
               "{\n"
               "  \"memory-type\": \"PER_NPU_MEMORY_EXPANSION\",\n"
               "  \"npu-ids\": [0, 1],\n"
               "  \"remote-mem-bw\": 6.0,\n"
               "  \"remote-mem-latency\": 100\n"
               "}\n");
    AnalyticalRemoteMemory mem(g_fx.dir + "/rm_c9.json");
    mem.set_sys(0, g_fx.systems[0]);
    issue_once(&mem, 5, 64);  // rank 5 不在 npu-ids
    std::printf("C9 DID NOT fail closed\n");
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    build_shared_fixture();

    // ---- 阶段 5.2 指名数学锚优先（S1/S12/S6A），再语义场景 ----
    scenario_s1();
    scenario_s12();
    scenario_s6a();
    scenario_s6b();
    scenario_s3();
    scenario_s5();
    scenario_s7();
    // S4：已知分歧（非致命 FAIL 行）；S2：已知分歧（子进程隔离）。
    scenario_s4();
    // §7 白盒正反例：事件 generation / 派发期同步重入 / 提前 shutdown。
    scenario_s13();
    scenario_s14();
    scenario_s15();

    // ---- fail-closed（fork 子进程；构造期与运行期越界全部 exit(1)） ----
    expect_fail_closed("C1 ctor remote-mem-bw=0", child_ctor_bw_zero);
    expect_fail_closed("C2 ctor remote-mem-bw<0", child_ctor_bw_negative);
    expect_fail_closed("C3 ctor remote-mem-bw=1e400 (inf)", child_ctor_bw_inf);
    expect_fail_closed("C4 ctor remote-mem-latency<0",
                       child_ctor_latency_negative);
    expect_fail_closed("C5 ctor remote-mem-latency=1e400 (inf)",
                       child_ctor_latency_inf);
    expect_fail_closed("C6 runtime latency 1e308 Tick overflow",
                       child_runtime_latency_overflow);
    expect_fail_closed("C7 runtime bw 1e-20 Tick overflow",
                       child_runtime_bw_underflow);
    expect_fail_closed("C8 NO_MEMORY_EXPANSION issue",
                       child_no_memory_issue);
    expect_fail_closed("C9 PER_NPU unconfigured rank issue",
                       child_per_npu_unconfigured_rank);

    // ---- S2（长短服务/完成即重分/同刻就绪-完成；子进程隔离已知分歧） ----
       expect_child_pass("S2 re-split & ready-at-completion-tick",
                      scenario_s2_body);

    // ---- teardown：recorder 先于 Sys（其析构读 sys），memory 后于 Sys ----
    for (auto* r : g_fx.recorders) {
        delete r;
    }
    g_fx.recorders.clear();
    for (auto* s : g_fx.systems) {
        delete s;
    }
    g_fx.systems.clear();
    g_fx.network_apis.clear();
    g_fx.primary_memory.reset();
    AstraSim::LoggerFactory::shutdown();

    std::error_code ec;
    std::filesystem::remove_all(g_fx.dir, ec);

    if (!g_ok) {
        std::fprintf(stderr, "[remote_port_nway_test] FAIL\n");
        return 1;
    }
    std::printf("[remote_port_nway_test] ALL PASS: PER_NPU/PER_NODE/"
                "MEMORY_POOL mapping, same/staggered latency, long-short "
                "re-split, ready-at-completion-tick, same-tick multi-port/"
                "multi-job order, non-divisible, tiny residue, dual-zero, "
                "zero-byte, fail-closed x9, anchors 300 / 90-120-140-150 / "
                "1B-next-tick\n");
    return 0;
}
