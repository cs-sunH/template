/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*

remote_port_online_gate_test.cc -- 《SerDes片外链路并发化改造执行方案》阶段 5.3
Online 系统门控夹具（方案 V5.3，2026-09-24）。

本文件是新建的独立系统路径夹具；不是任何旧 RemoteFifoLedgerTest 的扩展——
该测试及其源在本仓 2026-09-24 工作树已删除（方案 阶段 5.3 明确不得"扩展"）。
夹具骨架遵循 local_hbm_model_test.cc / 旧 ledger 测试的共同方法：自写微型
配置（mkdtemp 隔离目录，不依赖任何仓外手工产物）、真实 congestion-aware
拓扑、MEM/COMM 节点直注 per-rank NodeStore（NodeStoreGraphSource/NodeView）、
CompletionObserver hook 记录终态。

场景与断言清单（与方案阶段 5.2/5.3/阶段 2 验收一一对应）：

  场景 A（同 rank 双 MEM，无 Data 依赖）rank0：MEM_LOAD 300B（id1）+
  MEM_LOAD 600B（id2）。一次 issue pass（workload->issue_dep_free_nodes()）
  必须把两笔都发上同一端口：
    A1 门控：发射后 rank0 num_in_flight_remote_mem_ops==2 且
       num_in_flight_gpu_comm_ops==0（远端 MEM 不占 comm 单槽——阶段 2 验收
       "两个独立同 rank MEM 在第一笔事务仍在途时先后发射"）；
    A2 PortStats（port0，发射后即时）：issued_count==2 且 in_flight_count==2
       （阶段 5.3 验收原文）、peak_in_flight==2、latency_waiting_count==2、
       streaming_count==0、completion_waiting_count==0、port_busy_ns==0；
    A3 服务区间积分（跑完后 port0 快照，其 shared_busy_ns/port_busy_ns/
       bytes_served 由后端按连续子步事件区间积分结算，非采样）：
       peak_streaming==2（>=2 且逐值钉死）、shared_busy_ns==100ns（>0——
       [t0+100,t0+200) 双流共享窗口）、port_busy_ns==150、bytes_served==900
       （= 发射字节，服务量守恒）、redistribution_events==1（短流完成、
       幸存长流份额 3->6 B/ns）、new_stream_joins==2；
    A4 完成批次/守恒：issued_bytes==completed_bytes==900、completed_count==2、
       in_flight 归零；终态 tick：MEM_A@t0+200（300B@3B/ns）、
       MEM_B@t0+250（重分后 300B@6B/ns——完成即重分），各恰一次、Success。

  场景 B（独立 MEM+COMM_SEND，无依赖）rank1：MEM_LOAD 300B（id1）+
  COMM_SEND 300B->rank2 tag7（id2）；rank2：COMM_RECV（id1）配对。
    B1 门控：发射后 rank1 num_in_flight_remote_mem_ops==1 且
       num_in_flight_gpu_comm_ops==1 同时成立（MEM 与 COMM_SEND 各占各门、
       互不阻塞——阶段 2 验收 "MEM 与 COMM_SEND 不互相占用同一门"；旧 comm
       单槽门控下 SEND 会被 MEM 压住，is_available=false，此断言即失败）；
    B2 PortStats：port1 issued_count==1、in_flight_count==1（发射后即时）；
       port2（rank2）issued_count==0（COMM_RECV 不产生远端端口事务）；
    B3 终态：MEM_C@t0+150（300B 独享 6B/ns，与场景 A 同字节不同并发对照）、
       SEND/RECV 网络完成各恰一次、Success；rank1/rank2 的 remote/comm 计数
       收尾全零；后端 is_drained()。

  跨端口隔离：port0.issued==2 而 rank1 的 MEM 在 port1（发射后即查，PER_NPU
  映射逐 rank 独立，不串账）。

  全局：全部 5 个节点恰终态一次且 Success；事件队列排空；后端 is_drained()
  （sensing 关闭同样生效的无条件空集/守恒检查）。

PortStats 观测：方案阶段 5.3 验收要求直读后端 PortStats。
port_stats/enable_transaction_log 在本仓后端头的
public 段（AnalyticalRemoteMemory.hh:155-229），测试经公共接口直读。
（更正：本文件早版注释误判该访问器为 private 并引入了 include 前访问宏
seam——该 seam 已删除，公共接口即够。）

已知边界：本套件场景不触碰 5.1 套件记录的 join-at-boundary 偏差（双 MEM
同刻就绪、端口此前空闲，streams_before==0 的 join 不被扣份）；发射均经
Workload 生产路径（issue_dep_free_nodes / issue_remote_mem / issue_send_comm
/ issue_recv_comm），完成回调落 Workload::call 真实终态处理。

构建（先按本仓 README 完成 configure，再）：
  cmake --build build/astra_analytical/build_congestion_aware -j \
        --target AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest
运行（无参数；exit 0 = ALL PASS）：
  build/astra_analytical/build_congestion_aware/bin/\
AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest
注册：astra-sim/network_frontend/analytical/CMakeLists.txt；ctest 不作为执行
证据，回归以上述显式命令与退出码为准（阶段 5.7 口径）。
*******************************************************************************/

// 包含块（后端头依赖的完整类型——PortJob 以 unique_ptr<WorkloadLayerHandlerData>
// 持有 wlhd——先于后端头包含）。
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
// 后端头 PortJob 以 unique_ptr<WorkloadLayerHandlerData> 持有 wlhd（其头内
// 注释：析构需完整类型），故该头先于后端头包含。
#include "astra-sim/system/WorkloadLayerHandlerData.hh"

#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include <astra-network-analytical/common/EventQueue.h>
#include <astra-network-analytical/common/NetworkParser.h>
#include <astra-network-analytical/congestion_aware/Helper.h>

// 后端头：PortStats/port_stats/enable_transaction_log 在本头 public 段
// （AnalyticalRemoteMemory.hh:155-229），公共接口直读，无需访问宏。
#include <remote_memory_backend/analytical/AnalyticalRemoteMemory.hh>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <utility>

using namespace AstraSim;
using namespace Analytical;
using namespace AstraSimAnalytical;
using namespace AstraSimAnalyticalCongestionAware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;
using namespace AstraSim::ExecutionDriven;

namespace {

// ---- assertion helpers（report-and-continue；exit code = 整体结论） ----

bool g_ok = true;

void expect(bool cond, const std::string& what) {
    if (!cond) {
        std::fprintf(stderr,
                     "[remote_port_online_gate_test] FAIL: %s\n",
                     what.c_str());
        g_ok = false;
    }
}

void expect_near(double value, double expected, double tol,
                 const std::string& what) {
    const double diff = value > expected ? value - expected : expected - value;
    if (!(diff <= tol)) {
        std::fprintf(stderr,
                     "[remote_port_online_gate_test] FAIL: %s: got %.9f "
                     "expected %.9f (tol %.9f)\n",
                     what.c_str(), value, expected, tol);
        g_ok = false;
    }
}

std::string tick_str(Tick t) {
    return std::to_string(static_cast<uint64_t>(t));
}

// ---- 终态观察：CompletionObserver hook（真实 Workload 终态路径） ----

struct TerminalRec {
    uint64_t count = 0;
    uint64_t tick = 0;
    int status = -1;
};
std::map<uint64_t, TerminalRec> g_terminals;  // key = (rank<<32)|node_id
std::vector<uint64_t> g_terminal_order;

void terminal_hook(void* ctx, int rank, uint64_t node_id, const char*,
                   const char*, uint64_t, uint64_t tick, int status) {
    auto* seen = static_cast<std::map<uint64_t, TerminalRec>*>(ctx);
    const uint64_t key =
        (static_cast<uint64_t>(rank) << 32) | node_id;
    TerminalRec& rec = (*seen)[key];
    rec.count += 1;
    rec.tick = tick;
    rec.status = status;
    g_terminal_order.push_back(key);
}

const TerminalRec& terminal_of(int rank, uint64_t node_id) {
    return g_terminals[(static_cast<uint64_t>(rank) << 32) | node_id];
}

// ---- 共享夹具（一次构建；单次仿真覆盖两个场景） ----

constexpr int kRanks = 6;

struct Fixture {
    std::string dir;
    std::shared_ptr<EventQueue> queue;
    std::shared_ptr<FluidScheduler> fluid_scheduler;
    std::unique_ptr<AnalyticalRemoteMemory> memory;  // 全部 rank 共享
    std::vector<std::unique_ptr<CongestionAwareNetworkApi>> network_apis;
    std::vector<std::shared_ptr<NodeStoreGraphSource>> sources;
    std::vector<Sys*> systems;
};

Fixture g_fx;

void write_text(const std::string& path, const std::string& content) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (f == nullptr) {
        std::perror("write_text");
        std::exit(1);
    }
    std::fputs(content.c_str(), f);
    std::fclose(f);
}

// 官方模板形状 system.json（三键齐备 + 四实现键全 ring；HBM 两开关关：
// MEM 终态恰为端口事务完成、comm 端点不挂本地 HBM 腿，门控语义因此可逐值
// 解析）。roofline 开（无 COMP 节点，不参与）。
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
  "remote-mem-bw": 6.0,
  "remote-mem-latency": 100
}
)";

// 官方形状 comm_group.json：组带 ranks + dimensions（3x2=6 与 Mesh[3,2] 一致）。
const char* kCommGroupJson = R"({
  "1": {
    "ranks": [0, 1, 2, 3, 4, 5],
    "dimensions": [3, 2]
  }
}
)";

const char* kNetworkYaml = R"(topology: [ Mesh, Mesh ]
npus_count: [ 3, 2 ]
bandwidth: [ 4050, 4050 ]
latency: [ 25, 25 ]
)";

// PER_NPU：逐 rank 独立端口；rank0->port0（场景 A），rank1->port1（场景 B）。
const char* kRemoteMemoryJson = R"({
  "memory-type": "PER_NPU_MEMORY_EXPANSION",
  "npu-ids": [0, 1, 2, 3, 4, 5],
  "remote-mem-bw": 6.0,
  "remote-mem-latency": 100
}
)";

void build_fixture() {
    std::string tmpl = "/tmp/remote_port_online_gate_test_XXXXXX";
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
    write_text(g_fx.dir + "/remote_memory.json", kRemoteMemoryJson);

    AstraSim::LoggerFactory::init("empty", "off");
    CompletionObserver::instance().set_hook(terminal_hook, &g_terminals);

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
    CongestionAwareNetworkApi::set_fluid_scheduler(fluid_scheduler);
    g_fx.fluid_scheduler = fluid_scheduler;

    // 单一共享后端：Sys 构造期 set_sys 逐 rank 注册（PER_NPU + npu-ids 显式，
    // 端口表在构造期固定）。
    g_fx.memory = std::make_unique<AnalyticalRemoteMemory>(
        g_fx.dir + "/remote_memory.json");

    const auto npus_count_per_dim = network_parser.get_npus_counts_per_dim();
    std::vector<int> queues_per_dim(npus_count_per_dim.size(), 1);
    for (int i = 0; i < kRanks; ++i) {
        g_fx.network_apis.push_back(
            std::make_unique<CongestionAwareNetworkApi>(i));
    }
    for (int i = 0; i < kRanks; ++i) {
        auto source = std::make_shared<NodeStoreGraphSource>();
        Sys* sys = new Sys(
            i, g_fx.dir + "/workload", g_fx.dir + "/comm_group.json",
            g_fx.dir + "/system.json", g_fx.memory.get(),
            g_fx.network_apis[i].get(), npus_count_per_dim, queues_per_dim,
            1.0, 1.0, false, ExecutionDriven::ExecutionMode::Online, source);
        g_fx.systems.push_back(sys);
        g_fx.sources.push_back(source);
    }
    expect(static_cast<int>(topology->get_npus_count()) == kRanks,
           "fixture: topology has 6 ranks");
}

// ---- OnlineNode 构造助手（NodeStoreGraphSource/NodeView 路径） ----

ExecutionDriven::OnlineNode make_node(int rank, ExecutionDriven::NodeKind kind,
                                      uint64_t node_type,
                                      const std::string& name) {
    ExecutionDriven::OnlineNode node;
    node.rank = rank;
    node.kind = kind;
    node.node_type = node_type;  // ChakraProtoMsg::NodeType 原值
    node.name = name;
    return node;
}

uint64_t add_mem(int rank, uint64_t bytes, const std::string& name) {
    ExecutionDriven::OnlineNode node =
        make_node(rank, ExecutionDriven::NodeKind::MemLoad,
                  /*MEM_LOAD_NODE=*/2, name);
    node.compute.tensor_size = bytes;
    return g_fx.sources[static_cast<std::size_t>(rank)]->store().add_node(
        std::move(node));
}

uint64_t add_send(int rank, int dst, uint64_t bytes, uint32_t tag,
                  const std::string& name) {
    ExecutionDriven::OnlineNode node =
        make_node(rank, ExecutionDriven::NodeKind::CommSend,
                  /*COMM_SEND_NODE=*/5, name);
    node.comm.src = rank;
    node.comm.dst = dst;
    node.comm.bytes = bytes;
    node.comm.tag = tag;
    return g_fx.sources[static_cast<std::size_t>(rank)]->store().add_node(
        std::move(node));
}

uint64_t add_recv(int rank, int src, uint64_t bytes, uint32_t tag,
                  const std::string& name) {
    ExecutionDriven::OnlineNode node =
        make_node(rank, ExecutionDriven::NodeKind::CommRecv,
                  /*COMM_RECV_NODE=*/6, name);
    node.comm.src = src;
    node.comm.dst = rank;
    node.comm.bytes = bytes;
    node.comm.tag = tag;
    return g_fx.sources[static_cast<std::size_t>(rank)]->store().add_node(
        std::move(node));
}

// 一次 issue pass：生产入口原样调用（online 的 post-commit 路径调的就是它）。
void issue_pass(int rank) {
    g_fx.systems[static_cast<std::size_t>(rank)]
        ->workload->issue_dep_free_nodes();
}

void noop_cb(void*) {}

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

void drain_queue() {
    std::size_t guard = 0;
    while (!g_fx.queue->finished() || g_fx.queue->has_deferred_work()) {
        if (g_fx.queue->finished() && g_fx.queue->has_deferred_work()) {
            // EventQueue 文档模式：主队列空但有 deferred 残留时，
            // 安排严格未来的空转事件强迫一次 proceed 完成排空。
            g_fx.queue->schedule_event(Sys::boostedTick() + 1, noop_cb,
                                       nullptr);
        }
        g_fx.queue->proceed();
        expect(++guard < 1000000, "drain_queue: event loop guard");
    }
}

// PortStats 数值助手（公共接口直读；字段口径见后端头 §5.1 注释）。

uint64_t stat_u64(const AnalyticalRemoteMemory::PortStats& s,
                  const char* field) {
    if (std::strcmp(field, "issued_count") == 0) return s.issued_count;
    if (std::strcmp(field, "completed_count") == 0) return s.completed_count;
    if (std::strcmp(field, "issued_bytes") == 0) return s.issued_bytes;
    if (std::strcmp(field, "completed_bytes") == 0) return s.completed_bytes;
    if (std::strcmp(field, "in_flight_count") == 0) return s.in_flight_count;
    if (std::strcmp(field, "peak_in_flight") == 0) return s.peak_in_flight;
    if (std::strcmp(field, "latency_waiting_count") == 0) {
        return s.latency_waiting_count;
    }
    if (std::strcmp(field, "completion_waiting_count") == 0) {
        return s.completion_waiting_count;
    }
    // streaming_count / peak_streaming 是 std::size_t，单独映射。
    if (std::strcmp(field, "streaming_count") == 0) {
        return static_cast<uint64_t>(s.streaming_count);
    }
    if (std::strcmp(field, "peak_streaming") == 0) {
        return static_cast<uint64_t>(s.peak_streaming);
    }
    if (std::strcmp(field, "redistribution_events") == 0) {
        return s.redistribution_events;
    }
    if (std::strcmp(field, "new_stream_joins") == 0) return s.new_stream_joins;
    std::fprintf(stderr, "unknown stat field %s\n", field);
    std::exit(1);
}

void expect_stat(const AnalyticalRemoteMemory::PortStats& s, const char* field,
                 uint64_t expected, const std::string& what) {
    const uint64_t got = stat_u64(s, field);
    expect(got == expected,
           what + ": " + field + " " + std::to_string(got) +
               " != " + std::to_string(expected));
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    build_fixture();

    // ---- 图构造：两场景节点，全部无 Data 依赖（加入即可发射） ----
    // 场景 A：rank0 双 MEM（300B + 600B）。
    const uint64_t mem_a = add_mem(0, 300, "gateA_mem_300B");
    const uint64_t mem_b = add_mem(0, 600, "gateA_mem_600B");
    // 场景 B：rank1 MEM + COMM_SEND（互不依赖）；rank2 RECV 配对。
    const uint64_t mem_c = add_mem(1, 300, "gateB_mem_300B");
    const uint64_t send_b = add_send(1, 2, 300, 7, "gateB_send");
    const uint64_t recv_b = add_recv(2, 1, 300, 7, "gateB_recv");
    expect(mem_a == 1 && mem_b == 2 && mem_c == 1 && send_b == 2 && recv_b == 1,
           "fixture: per-rank NodeStore ids as expected");

    const Tick t0 = Sys::boostedTick();

    // ---- 一次 issue pass（逐 rank 生产入口） ----
    issue_pass(0);
    issue_pass(1);
    issue_pass(2);

    // ===================================================================
    // 发射后即时断言（t0，latency 未到期，任何完成都不可能发生）
    // ===================================================================

    // A1 门控：双 MEM 同端口在途、comm 槽未被 MEM 占用。
    {
        const auto* hw = g_fx.systems[0]->workload->hw_resource;
        expect(hw->num_in_flight_remote_mem_ops == 2,
               "A1 gate: two same-rank MEMs both in flight (remote slot "
               "count 2, got " +
                   std::to_string(hw->num_in_flight_remote_mem_ops) + ")");
        expect(hw->num_in_flight_gpu_comm_ops == 0,
               "A1 gate: MEM traffic holds no comm slot");
    }
    // A2 PortStats：同端口 issued/in-flight == 2（阶段 5.3 验收原文）。
    {
        const auto s = g_fx.memory->port_stats(0);
        expect_stat(s, "issued_count", 2, "A2 port0 right after issue pass");
        expect_stat(s, "in_flight_count", 2, "A2 port0 right after issue pass");
        expect_stat(s, "peak_in_flight", 2, "A2 port0 peak in flight");
        expect_stat(s, "latency_waiting_count", 2,
                    "A2 port0 both in latency stage");
        expect_stat(s, "streaming_count", 0,
                    "A2 port0 latency overlaps, no stream yet");
        expect_stat(s, "completion_waiting_count", 0,
                    "A2 port0 nothing fluid-complete yet");
        expect_stat(s, "issued_bytes", 900, "A2 port0 issued bytes");
        expect_near(s.port_busy_ns, 0.0, 1e-9,
                    "A2 port0 no service before latency expiry");
        expect_near(s.shared_busy_ns, 0.0, 1e-9,
                    "A2 port0 no shared service yet");
    }
    // 跨端口隔离：rank1 的 MEM 记在 port1，不串到 port0。
    {
        const auto p0 = g_fx.memory->port_stats(0);
        const auto p1 = g_fx.memory->port_stats(1);
        expect(p0.issued_count == 2 && p1.issued_count == 1,
               "cross-port isolation: port0 issued 2, port1 issued 1 (got " +
                   std::to_string(p0.issued_count) + "/" +
                   std::to_string(p1.issued_count) + ")");
    }
    // B1 门控：MEM 与 COMM_SEND 各占各门、同时各 1。
    {
        const auto* hw = g_fx.systems[1]->workload->hw_resource;
        expect(hw->num_in_flight_remote_mem_ops == 1,
               "B1 gate: MEM in the remote slot (got " +
                   std::to_string(hw->num_in_flight_remote_mem_ops) + ")");
        expect(hw->num_in_flight_gpu_comm_ops == 1,
               "B1 gate: COMM_SEND in the comm slot beside the in-flight MEM "
               "(got " +
                   std::to_string(hw->num_in_flight_gpu_comm_ops) + ")");
    }
    // B2 PortStats：port1 issued/in-flight == 1；port2（RECV）无远端事务。
    {
        const auto p1 = g_fx.memory->port_stats(1);
        expect_stat(p1, "issued_count", 1, "B2 port1 right after issue pass");
        expect_stat(p1, "in_flight_count", 1,
                    "B2 port1 right after issue pass");
        const auto p2 = g_fx.memory->port_stats(2);
        expect_stat(p2, "issued_count", 0,
                    "B2 port2: COMM_RECV creates no remote-port transaction");
    }

    // ---- 真实事件循环（COMM_SEND 需要 FluidScheduler 流水线活着） ----
    g_fx.fluid_scheduler->flush_pending_starts();
    g_fx.fluid_scheduler->mark_event_loop_started();
    drain_queue();
    // 推进到确定的尾点：让全部端口做最后一次 [H2] 子步结算（streaming
    // 归零、流完成时刻落账），快照因此与时间无关。
    advance_to(t0 + 1000);

    // ===================================================================
    // 排空后断言：服务区间积分 + 完成批次/守恒 + 终态
    // ===================================================================

    // A3/A4：port0 积分与守恒（shared_busy_ns/port_busy_ns/bytes_served 为
    // 后端按连续子步事件区间积分的结算值）。
    {
        const auto s = g_fx.memory->port_stats(0);
        expect_stat(s, "issued_count", 2, "A4 port0 final");
        expect_stat(s, "completed_count", 2, "A4 port0 final");
        expect_stat(s, "issued_bytes", 900, "A4 port0 bytes conservation");
        expect_stat(s, "completed_bytes", 900, "A4 port0 bytes conservation");
        expect_stat(s, "in_flight_count", 0, "A4 port0 drained");
        expect_stat(s, "peak_in_flight", 2, "A3 port0 peak in flight");
        expect_stat(s, "streaming_count", 0, "A3 port0 no stream at end");
        expect_stat(s, "peak_streaming", 2,
                    "A3 port0 two concurrent transfer streams");
        expect_stat(s, "latency_waiting_count", 0, "A3 port0 drained");
        expect_stat(s, "completion_waiting_count", 0, "A3 port0 drained");
        expect_stat(s, "redistribution_events", 1,
                    "A3 port0 short-stream completion re-splits the long one");
        expect_stat(s, "new_stream_joins", 2,
                    "A3 port0 both streams joined at latency expiry");
        expect_near(s.shared_busy_ns, 100.0, 1e-6,
                    "A3 port0 shared service window [t0+100,t0+200) (>0)");
        expect(s.shared_busy_ns > 0.0,
               "A3 port0 shared_busy_ns > 0 (stage-5.3 acceptance)");
        expect_near(s.port_busy_ns, 150.0, 1e-6,
                    "A3 port0 busy [t0+100,t0+250)");
        expect_near(s.bytes_served, 900.0, 1e-6,
                    "A3 port0 served bytes == issued bytes");
        expect(s.bytes_served <= 6.0 * s.port_busy_ns + 1e-6,
               "A3 port0 capacity bound: served <= bw x busy");
    }
    // B3：port1 积分（单流：无共享、无重分）与 port2 全零。
    {
        const auto p1 = g_fx.memory->port_stats(1);
        expect_stat(p1, "issued_count", 1, "B3 port1 final");
        expect_stat(p1, "completed_count", 1, "B3 port1 final");
        expect_stat(p1, "issued_bytes", 300, "B3 port1 bytes conservation");
        expect_stat(p1, "completed_bytes", 300, "B3 port1 bytes conservation");
        expect_stat(p1, "in_flight_count", 0, "B3 port1 drained");
        expect_stat(p1, "peak_streaming", 1, "B3 port1 solo stream");
        expect_stat(p1, "redistribution_events", 0,
                    "B3 port1 no survivor on completion");
        expect_stat(p1, "new_stream_joins", 1, "B3 port1 one join");
        expect_near(p1.shared_busy_ns, 0.0, 1e-9,
                    "B3 port1 no shared service (solo stream)");
        expect_near(p1.port_busy_ns, 50.0, 1e-6,
                    "B3 port1 busy [t0+100,t0+150) at full 6B/ns");
        const auto p2 = g_fx.memory->port_stats(2);
        expect_stat(p2, "issued_count", 0, "B3 port2 stays empty");
        expect_stat(p2, "completed_count", 0, "B3 port2 stays empty");
        expect_near(p2.bytes_served, 0.0, 1e-9, "B3 port2 served nothing");
    }

    // 终态：恰一次、Success、解析 tick。
    {
        const auto& a = terminal_of(0, mem_a);
        expect(a.count == 1 && a.status == 0,
               "terminal: MEM_A exactly once, Success");
        expect(a.tick == t0 + 200,
               "terminal: MEM_A at t0+200 (300B @ 3B/ns shared), got " +
                   tick_str(a.tick));
        const auto& b = terminal_of(0, mem_b);
        expect(b.count == 1 && b.status == 0,
               "terminal: MEM_B exactly once, Success");
        expect(b.tick == t0 + 250,
               "terminal: MEM_B at t0+250 (survivor 300B re-split to "
               "6B/ns), got " + tick_str(b.tick));
        const auto& c = terminal_of(1, mem_c);
        expect(c.count == 1 && c.status == 0,
               "terminal: MEM_C exactly once, Success");
        expect(c.tick == t0 + 150,
               "terminal: MEM_C at t0+150 (300B solo full rate; same bytes "
               "as MEM_A but no sharing), got " + tick_str(c.tick));
        const auto& snd = terminal_of(1, send_b);
        expect(snd.count == 1 && snd.status == 0,
               "terminal: COMM_SEND exactly once, Success");
        const auto& rcv = terminal_of(2, recv_b);
        expect(rcv.count == 1 && rcv.status == 0,
               "terminal: COMM_RECV exactly once, Success");
    }

    // 收尾：门控计数全零（每节点恰释放一次）、后端与队列排空。
    for (const int r : {0, 1, 2}) {
        const auto* hw = g_fx.systems[r]->workload->hw_resource;
        expect(hw->num_in_flight_remote_mem_ops == 0 &&
                   hw->num_in_flight_gpu_comm_ops == 0 &&
                   hw->num_in_flight_gpu_comp_ops == 0 &&
                   hw->num_in_flight_hbm_dma_ops == 0 &&
                   hw->num_in_flight_cpu_ops == 0,
               "rank " + std::to_string(r) +
                   " fully released (remote/comm/comp/dma/cpu all zero)");
    }
    expect(g_fx.memory->is_drained(),
           "backend is_drained() (unconditional empty-set + conservation)");
    expect(g_fx.queue->finished() && !g_fx.queue->has_deferred_work(),
           "event queue fully drained");

    // ---- teardown：Sys 先删（memory 后删，alarm 夹具同一纪律） ----
    for (auto* s : g_fx.systems) {
        delete s;
    }
    g_fx.systems.clear();
    g_fx.sources.clear();
    g_fx.network_apis.clear();
    g_fx.memory.reset();
    AstraSim::LoggerFactory::shutdown();

    std::error_code ec;
    std::filesystem::remove_all(g_fx.dir, ec);

    if (!g_ok) {
        std::fprintf(stderr, "[remote_port_online_gate_test] FAIL\n");
        return 1;
    }
    std::printf("[remote_port_online_gate_test] ALL PASS: same-rank dual-MEM "
                "one-pass issue (port issued/in-flight=2, peak_streaming=2, "
                "shared_busy_ns=100>0 by interval integration), independent "
                "MEM+COMM_SEND doors, per-port isolation, conservation, "
                "drained\n");
    return 0;
}
