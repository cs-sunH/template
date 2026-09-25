/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*

remote_port_static_gate_test.cc -- 《SerDes片外链路并发化改造执行方案》阶段 5.4
Static/ETFeeder 系统门控夹具（方案 V5.3，2026-09-24）。

与 joint 的 HbmNwayTest 无关（本仓无该文件，方案 阶段 5.5 明确不虚构）；也不是
已删除的 RemoteFifoLedgerTest 的任何扩展。静态入口复刻本仓
completion_observer_fixture_main.cc 的 main 结构（Static 模式 Sys + ETFeeder +
fire()，方案认可的唯一静态 ET 测试入口先例）。

夹具生成（契约固定，幂等）：
  python3 astra-sim/workload/execution_driven/tests/\
make_remote_port_static_fixture_et.py --out-dir <临时目录>
本二进制接受同一目录：--fixture-dir <临时目录>；从中读取
  remote_port_static_gate.0..5.et、system.json、comm_group.json、network.yml、
  remote_memory.json（生成器写入官方模板形状配置，避雷段合规）。

场景（rank0，同 rank 两个无 Data 依赖远端 MEM：id=1 300B、id=2 600B；
bw=6B/ns、latency=100ns）与断言清单（与方案阶段 5.4/阶段 4 验收对应）：

  S1 comm 单槽未阻止第二个发射：fire() 一次扫描后 rank0
     num_in_flight_remote_mem_ops==2、num_in_flight_gpu_comm_ops==0、
     remote_mem_ops_node 集合恰含 {1,2}（静态模式保留节点 id）——旧 comm
     单槽隐藏门下第二个 MEM 会被压住，本断言即失败；
  S2 PortStats（公共接口直读，同 5.3 口径）：port0 issued_count==2、
     in_flight_count==2、latency_waiting_count==2、streaming_count==0；
  S3 finish gate（阶段 4：静态 sim-finish gate 含 remote MEM 在途计数，
     Workload.cc:1160-1187）：fire() 后 workload->is_finished==false（两笔
     在途时静态结束门不得开启）；排空后 is_finished==true 且全部终态已记录
     （门等到全部终结，不提前）；
  S4 计数/集合释放各一次：收尾 remote/comm 计数为零、remote_mem_ops_node
     集合为空（释放路径带下溢 fatal 守卫，干净跑完即恰一次）、终态各恰一次
     Success；
  S5 解析终态与服务积分：MEM id1@t0+200（300B@3B/ns）、MEM id2@t0+250
     （幸存流重分 6B/ns）；port0 peak_streaming==2、shared_busy_ns==100（>0）、
     port_busy_ns==150、bytes_served==900、redistribution_events==1、
     new_stream_joins==2、bytes 900/900 守恒；
  S6 全局：其余端口零事务、后端 is_drained()、事件队列排空。

构建：cmake --build build/astra_analytical/build_congestion_aware -j \
        --target AstraSim_Analytical_Congestion_Aware_RemotePortStaticGateTest
运行：build/astra_analytical/build_congestion_aware/bin/\
        AstraSim_Analytical_Congestion_Aware_RemotePortStaticGateTest \
        --fixture-dir <临时目录>
注册：astra-sim/network_frontend/analytical/CMakeLists.txt；ctest 不作为执行
证据（阶段 5.7 口径）。
*******************************************************************************/

// 包含块（后端头依赖的完整类型先于后端头包含）。
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
// 注释：析构需完整类型），先于后端头包含。
#include "astra-sim/system/WorkloadLayerHandlerData.hh"

#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include <astra-network-analytical/common/EventQueue.h>
#include <astra-network-analytical/common/NetworkParser.h>
#include <astra-network-analytical/congestion_aware/Helper.h>

// 后端头：PortStats/port_stats/enable_transaction_log 在本头 public 段
// （AnalyticalRemoteMemory.hh:155-231），公共接口直读，无需访问宏。
#include <remote_memory_backend/analytical/AnalyticalRemoteMemory.hh>

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>

using namespace AstraSim;
using namespace Analytical;
using namespace AstraSimAnalytical;
using namespace AstraSimAnalyticalCongestionAware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

namespace {

bool g_ok = true;

void expect(bool cond, const std::string& what) {
    if (!cond) {
        std::fprintf(stderr,
                     "[remote_port_static_gate_test] FAIL: %s\n",
                     what.c_str());
        g_ok = false;
    }
}

void expect_near(double value, double expected, double tol,
                 const std::string& what) {
    const double diff = value > expected ? value - expected : expected - value;
    if (!(diff <= tol)) {
        std::fprintf(stderr,
                     "[remote_port_static_gate_test] FAIL: %s: got %.9f "
                     "expected %.9f (tol %.9f)\n",
                     what.c_str(), value, expected, tol);
        g_ok = false;
    }
}

std::string tick_str(Tick t) {
    return std::to_string(static_cast<uint64_t>(t));
}

// ---- 终态观察 ----

struct TerminalRec {
    uint64_t count = 0;
    uint64_t tick = 0;
    int status = -1;
};
std::map<uint64_t, TerminalRec> g_terminals;

void terminal_hook(void* ctx, int rank, uint64_t node_id, const char*,
                   const char*, uint64_t, uint64_t tick, int status) {
    auto* seen = static_cast<std::map<uint64_t, TerminalRec>*>(ctx);
    TerminalRec& rec =
        (*seen)[(static_cast<uint64_t>(rank) << 32) | node_id];
    rec.count += 1;
    rec.tick = tick;
    rec.status = status;
}

const TerminalRec& terminal_of(int rank, uint64_t node_id) {
    return g_terminals[(static_cast<uint64_t>(rank) << 32) | node_id];
}

// ---- 夹具 ----

constexpr int kRanks = 6;

struct Fixture {
    std::string dir;
    std::shared_ptr<EventQueue> queue;
    std::unique_ptr<AnalyticalRemoteMemory> memory;
    std::vector<std::unique_ptr<CongestionAwareNetworkApi>> network_apis;
    std::vector<Sys*> systems;
};

Fixture g_fx;

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
            g_fx.queue->schedule_event(Sys::boostedTick() + 1, noop_cb,
                                       nullptr);
        }
        g_fx.queue->proceed();
        expect(++guard < 1000000, "drain_queue: event loop guard");
    }
}

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

int main(int argc, char* argv[]) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    // ---- 契约参数：--fixture-dir <临时目录> ----
    std::string fixture_dir;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--fixture-dir=", 0) == 0) {
            fixture_dir = arg.substr(std::strlen("--fixture-dir="));
        } else if (arg == "--fixture-dir" && i + 1 < argc) {
            fixture_dir = argv[++i];
        }
    }
    if (fixture_dir.empty()) {
        std::fprintf(stderr,
                     "usage: %s --fixture-dir <dir generated by "
                     "make_remote_port_static_fixture_et.py>\n",
                     argc > 0 ? argv[0] : "RemotePortStaticGateTest");
        return 2;
    }

    AstraSim::LoggerFactory::init("empty", "off");
    // CompletionObserver 在 AstraSim::ExecutionDriven 内（静态文件未展开该
    // 命名空间，故按限定名使用）。
    ExecutionDriven::CompletionObserver::instance().set_hook(terminal_hook,
                                                             &g_terminals);

    g_fx.queue = std::make_shared<EventQueue>();
    const auto network_parser = NetworkParser(fixture_dir + "/network.yml");
    const auto topology = construct_topology(network_parser);
    CongestionAwareNetworkApi::set_event_queue(g_fx.queue);
    CongestionAwareNetworkApi::set_topology(topology);
    const auto fluid_scheduler = std::make_shared<FluidScheduler>(
        g_fx.queue, topology->get_directed_links(),
        network_parser.get_fluid_max_active_flows(),
        network_parser.get_fluid_max_route_memberships(),
        network_parser.get_progress_report_event_interval());
    CongestionAwareNetworkApi::set_fluid_scheduler(fluid_scheduler);

    g_fx.memory = std::make_unique<AnalyticalRemoteMemory>(
        fixture_dir + "/remote_memory.json");

    const auto npus_count_per_dim = network_parser.get_npus_counts_per_dim();
    std::vector<int> queues_per_dim(npus_count_per_dim.size(), 1);
    // 静态入口（与 completion_observer_fixture_main.cc 相同）：默认 Static
    // 模式 + graph_source 缺省 => Workload 自建 ETFeeder/ETFeederGraphSource，
    // 打开 <prefix>.<rank>.et。
    for (int i = 0; i < kRanks; ++i) {
        g_fx.network_apis.push_back(
            std::make_unique<CongestionAwareNetworkApi>(i));
    }
    for (int i = 0; i < kRanks; ++i) {
        Sys* sys = new Sys(
            i, fixture_dir + "/remote_port_static_gate",
            fixture_dir + "/comm_group.json", fixture_dir + "/system.json",
            g_fx.memory.get(), g_fx.network_apis[i].get(),
            npus_count_per_dim, queues_per_dim, 1.0, 1.0, false);
        g_fx.systems.push_back(sys);
    }
    expect(static_cast<int>(topology->get_npus_count()) == kRanks,
           "fixture: topology has 6 ranks");

    const Tick t0 = Sys::boostedTick();

    // ---- 静态发射：fire()（Workload::call(General,NULL) -> 首次
    //      issue_dep_free_nodes，两个 MEM 一次扫描同发） ----
    for (int i = 0; i < kRanks; ++i) {
        g_fx.systems[static_cast<std::size_t>(i)]->workload->fire();
    }

    // ===================================================================
    // 发射后即时断言（t0，latency 未到期）
    // ===================================================================

    // S1：comm 单槽未阻止第二个发射（静态保留节点 id 集合）。
    {
        const auto* hw = g_fx.systems[0]->workload->hw_resource;
        expect(hw->num_in_flight_remote_mem_ops == 2,
               "S1 gate: both same-rank MEMs issued by the first static scan "
               "(remote slot count 2, got " +
                   std::to_string(hw->num_in_flight_remote_mem_ops) + ")");
        expect(hw->num_in_flight_gpu_comm_ops == 0,
               "S1 gate: MEM traffic holds no comm slot");
        expect(hw->remote_mem_ops_node.size() == 2 &&
                   hw->remote_mem_ops_node.count(1) == 1 &&
                   hw->remote_mem_ops_node.count(2) == 1,
               "S1 gate: retained node-id set holds exactly {id1, id2}");
    }
    // S2：PortStats 同端口 issued/in-flight == 2。
    {
        const auto s = g_fx.memory->port_stats(0);
        expect_stat(s, "issued_count", 2, "S2 port0 right after fire()");
        expect_stat(s, "in_flight_count", 2, "S2 port0 right after fire()");
        expect_stat(s, "latency_waiting_count", 2,
                    "S2 port0 both in latency stage");
        expect_stat(s, "streaming_count", 0, "S2 port0 no stream yet");
        expect_stat(s, "issued_bytes", 900, "S2 port0 issued bytes");
    }
    // S3（前半）：finish gate 必须压住——两笔在途时静态结束不得开启。
    expect(g_fx.systems[0]->workload->is_finished == false,
           "S3 finish gate: static sim-finish must not fire while remote MEMs "
           "are in flight");

    // ---- 排空（静态无网络流；flush/mark 与静态入口先例一致，无害） ----
    fluid_scheduler->flush_pending_starts();
    fluid_scheduler->mark_event_loop_started();
    drain_queue();
    advance_to(t0 + 1000);

    // ===================================================================
    // 排空后断言
    // ===================================================================

    // S4/S5：计数/集合恰释放、解析终态、服务区间积分。
    {
        const auto* hw = g_fx.systems[0]->workload->hw_resource;
        expect(hw->num_in_flight_remote_mem_ops == 0,
               "S4 port0 remote slot fully released (got " +
                   std::to_string(hw->num_in_flight_remote_mem_ops) + ")");
        expect(hw->num_in_flight_gpu_comm_ops == 0,
               "S4 comm slot untouched and empty");
        expect(hw->remote_mem_ops_node.empty(),
               "S4 retained remote-MEM node-id set emptied by exactly-one "
               "release per node");
    }
    {
        const auto s = g_fx.memory->port_stats(0);
        expect_stat(s, "issued_count", 2, "S5 port0 final");
        expect_stat(s, "completed_count", 2, "S5 port0 final");
        expect_stat(s, "issued_bytes", 900, "S5 port0 bytes conservation");
        expect_stat(s, "completed_bytes", 900, "S5 port0 bytes conservation");
        expect_stat(s, "in_flight_count", 0, "S5 port0 drained");
        expect_stat(s, "peak_in_flight", 2, "S5 port0 peak in flight");
        expect_stat(s, "peak_streaming", 2,
                    "S5 port0 two concurrent transfer streams");
        expect_stat(s, "latency_waiting_count", 0, "S5 port0 drained");
        expect_stat(s, "completion_waiting_count", 0, "S5 port0 drained");
        expect_stat(s, "redistribution_events", 1,
                    "S5 port0 short-stream completion re-splits the long one");
        expect_stat(s, "new_stream_joins", 2,
                    "S5 port0 both streams joined at latency expiry");
        expect_near(s.shared_busy_ns, 100.0, 1e-6,
                    "S5 port0 shared service window (>0)");
        expect(s.shared_busy_ns > 0.0,
               "S5 port0 shared_busy_ns > 0");
        expect_near(s.port_busy_ns, 150.0, 1e-6,
                    "S5 port0 busy [t0+100,t0+250)");
        expect_near(s.bytes_served, 900.0, 1e-6,
                    "S5 port0 served bytes == issued bytes");
        expect(s.bytes_served <= 6.0 * s.port_busy_ns + 1e-6,
               "S5 port0 capacity bound: served <= bw x busy");
    }
    {
        const auto p1 = g_fx.memory->port_stats(1);
        expect_stat(p1, "issued_count", 0, "S6 port1 stays empty");
        expect_near(p1.bytes_served, 0.0, 1e-9, "S6 port1 served nothing");
    }

    // 终态：恰一次、Success、解析 tick。
    {
        const auto& a = terminal_of(0, 1);
        expect(a.count == 1 && a.status == 0,
               "terminal: MEM id1 exactly once, Success");
        expect(a.tick == t0 + 200,
               "terminal: MEM id1 at t0+200 (300B @ 3B/ns shared), got " +
                   tick_str(a.tick));
        const auto& b = terminal_of(0, 2);
        expect(b.count == 1 && b.status == 0,
               "terminal: MEM id2 exactly once, Success");
        expect(b.tick == t0 + 250,
               "terminal: MEM id2 at t0+250 (survivor 300B re-split to "
               "6B/ns), got " + tick_str(b.tick));
        // 其余终态：ranks 1-5 各一个 INVALID 填充节点（skip_invalid ->
        // Skipped），恰 5 个；Success 终态恰为两个 MEM。
        {
            uint64_t success_total = 0;
            uint64_t skipped_total = 0;
            for (const auto& [key, rec] : g_terminals) {
                if (rec.status == 0) {
                    success_total += 1;
                } else if (rec.status == 1) {
                    skipped_total += 1;
                    expect(rec.count == 1,
                           "filler node terminal exactly once");
                }
            }
            expect(success_total == 2 && skipped_total == 5,
                   "terminals: 2 Success MEMs + 5 Skipped fillers (got " +
                       std::to_string(success_total) + "/" +
                       std::to_string(skipped_total) + ")");
            expect(g_terminals.size() == 7,
                   "terminal record count 7 (got " +
                       std::to_string(g_terminals.size()) + ")");
        }
    }

    // S3（后半）：finish gate 等到全部终结后才开启。
    expect(g_fx.systems[0]->workload->is_finished == true,
           "S3 finish gate: static sim-finish fired only after every node "
           "reached whole-node terminal");
    expect(Sys::boostedTick() >= t0 + 250,
           "simulation clock passed the last terminal tick before end");

    // S6：后端与队列排空。
    expect(g_fx.memory->is_drained(),
           "backend is_drained() (unconditional empty-set + conservation)");
    expect(g_fx.queue->finished() && !g_fx.queue->has_deferred_work(),
           "event queue fully drained");

    // ---- teardown：Sys 先删（memory 后删） ----
    for (auto* s : g_fx.systems) {
        delete s;
    }
    g_fx.systems.clear();
    g_fx.network_apis.clear();
    g_fx.memory.reset();
    AstraSim::LoggerFactory::shutdown();

    std::error_code ec;
    std::filesystem::remove_all(fixture_dir, ec);

    if (!g_ok) {
        std::fprintf(stderr, "[remote_port_static_gate_test] FAIL\n");
        return 1;
    }
    std::printf("[remote_port_static_gate_test] ALL PASS: static ET path "
                "issues two same-rank dependency-free remote MEMs "
                "concurrently (issued/in-flight=2, comm slot untouched), "
                "releases counts/set exactly once, finish gate waits for "
                "all terminals, interval-integrated shared service > 0\n");
    return 0;
}
