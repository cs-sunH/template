/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*

remote_port_sensing_test.cc -- 《SerDes片外链路并发化改造执行方案》阶段 5.1/5.3
逐事务明细（sensing）正面夹具（2026-09-24，第 2 轮冒烟核验路由）。

背景：2 秒冒烟负载的远端池事务为零（kv 命中全部 local_hbm/no_history、
NOC_MIGRATE 走 NoC p2p），后端惰性建文件语义（AnalyticalRemoteMemory.cc
write_transaction_row：首行才建文件）下零事务即零文件——冒烟盘面只能证明
"零事务⇒零残留"这一正确负向，无法正面证实 enable_transaction_log 已武装、
明细行可产出、行键可复算。本夹具带真实池流量补齐正面证据。

三个臂（同一进程、真实 Sys/EventQueue 投递回路、直连后端 issue）：
  臂 1（武装 + 真实流量）PER_NPU bw=6 latency=100，port0 双流 300B/600B、
       port1 单流 300B（同 online gate 数学的共享窗口）：
    1a 明细文件 bridge_on/remote_memory_transactions.jsonl 恰 3 行、每行
       nlohmann json 可解析、schema=1、type=remote_memory_transaction、
       run_id==传入 id（武装证据）；
    1b 可复算：每行 callback_tick == ceil(fluid_finish_ns)（Tick/浮点不
       混用）、latency_ready_ns == stream_start_ns（正字节一次性入流）、
       Σbytes==1200；由行区间重算 port0 的共享窗口/忙碌/服务量
       （100/150/900 B·ns 口径）与后端 PortStats 的 shared_busy_ns/
       port_busy_ns/bytes_served 逐值相等——"无行不可复算"的反面实证；
    1c 完成分布：行内 callback_tick 多重集 {150:1, 200:1, 250:1} 与投递
       回路实测一致；port_index/issue_sequence 行键正确。
  臂 2（武装 + 零事务）：bridge_zero 永不产生文件（惰性建文件语义 =
       2 秒冒烟盘面的正确负向，武装与零事务在文件层面不可区分，正面区分
       由臂 1/臂 3 承担）。
  臂 3（未武装 + 同一真实流量）：bridge_off 无文件——证明"未武装 ⇒ 零
       明细"（与臂 1 一起把武装状态与文件存在性解耦判定）。

构建：cmake --build build/astra_analytical/build_congestion_aware -j \
        --target AstraSim_Analytical_Congestion_Aware_RemotePortSensingTest
运行（无参数；exit 0 = ALL PASS）：
  build/astra_analytical/build_congestion_aware/bin/\
AstraSim_Analytical_Congestion_Aware_RemotePortSensingTest
注册：astra-sim/network_frontend/analytical/CMakeLists.txt；sensing 开冒烟
命令在 2 秒负载冒烟之后追加本二进制，作为"武装+明细可复算"的正面证据
（冒烟本身保留零事务⇒零文件的负向证据）。
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
#include <remote_memory_backend/analytical/AnalyticalRemoteMemory.hh>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <json/json.hpp>

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
                     "[remote_port_sensing_test] FAIL: %s\n",
                     what.c_str());
        g_ok = false;
    }
}

std::string tick_str(Tick t) {
    return std::to_string(static_cast<uint64_t>(t));
}

void expect_near(double value, double expected, double tol,
                 const std::string& what) {
    const double diff = value > expected ? value - expected : expected - value;
    if (!(diff <= tol)) {
        std::fprintf(stderr,
                     "[remote_port_sensing_test] FAIL: %s: got %.9f "
                     "expected %.9f (tol %.9f)\n",
                     what.c_str(), value, expected, tol);
        g_ok = false;
    }
}

// ---- 完成投递记录器（直连后端 issue 的 wlhd 落点） ----

struct Rec {
    uint64_t count = 0;
    uint64_t tick = 0;
};
std::map<uint64_t, Rec> g_rec;

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
        Rec& rec = g_rec[wlhd->node_id];
        rec.count += 1;
        rec.tick = Sys::boostedTick();
        delete wlhd;
    }
};

// ---- 共享夹具 ----

constexpr int kRanks = 6;
constexpr const char* kRunId = "remote-port-sensing-fixture";

struct Fixture {
    std::string dir;
    std::shared_ptr<EventQueue> queue;
    std::vector<std::unique_ptr<CongestionAwareNetworkApi>> network_apis;
    std::vector<std::shared_ptr<ExecutionDriven::NodeStoreGraphSource>>
        sources;
    std::vector<Sys*> systems;
    std::vector<RecorderWorkload*> recorders;
    // 三个臂各一个后端实例（sensing 武装状态互不干扰）。
    std::unique_ptr<AnalyticalRemoteMemory> on;    // 武装 + 真实流量
    std::unique_ptr<AnalyticalRemoteMemory> zero;  // 武装 + 零事务
    std::unique_ptr<AnalyticalRemoteMemory> off;   // 未武装 + 真实流量
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
  "boost-mode": 0,
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

const char* kRemoteMemoryJson = R"({
  "memory-type": "PER_NPU_MEMORY_EXPANSION",
  "npu-ids": [0, 1, 2, 3, 4, 5],
  "remote-mem-bw": 6.0,
  "remote-mem-latency": 100
}
)";

void build_fixture() {
    std::string tmpl = "/tmp/remote_port_sensing_test_XXXXXX";
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
    std::filesystem::create_directories(g_fx.dir + "/bridge_on");
    std::filesystem::create_directories(g_fx.dir + "/bridge_zero");
    std::filesystem::create_directories(g_fx.dir + "/bridge_off");

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
    CongestionAwareNetworkApi::set_fluid_scheduler(fluid_scheduler);

    // 臂 1：武装后端（Sys 构造期 set_sys 逐 rank 注册，宿主 = rank0）。
    g_fx.on = std::make_unique<AnalyticalRemoteMemory>(
        g_fx.dir + "/remote_memory.json");
    g_fx.on->enable_transaction_log(g_fx.dir + "/bridge_on", kRunId);

    const auto npus_count_per_dim = network_parser.get_npus_counts_per_dim();
    std::vector<int> queues_per_dim(npus_count_per_dim.size(), 1);
    for (int i = 0; i < kRanks; ++i) {
        g_fx.network_apis.push_back(
            std::make_unique<CongestionAwareNetworkApi>(i));
    }
    for (int i = 0; i < kRanks; ++i) {
        auto source = std::make_shared<ExecutionDriven::NodeStoreGraphSource>();
        Sys* sys = new Sys(
            i, g_fx.dir + "/workload", g_fx.dir + "/comm_group.json",
            g_fx.dir + "/system.json", g_fx.on.get(),
            g_fx.network_apis[static_cast<std::size_t>(i)].get(),
            npus_count_per_dim, queues_per_dim, 1.0, 1.0, false,
            ExecutionDriven::ExecutionMode::Online, source);
        g_fx.systems.push_back(sys);
        g_fx.sources.push_back(source);
        g_fx.recorders.push_back(new RecorderWorkload(
            sys, g_fx.dir + "/workload", g_fx.dir + "/comm_group.json",
            source));
    }
    expect(static_cast<int>(topology->get_npus_count()) == kRanks,
           "fixture: topology has 6 ranks");

    // 臂 2/臂 3 的后端挂同一批 Sys（各自 set_sys；投递回路共用）。
    g_fx.zero = std::make_unique<AnalyticalRemoteMemory>(
        g_fx.dir + "/remote_memory.json");
    g_fx.off = std::make_unique<AnalyticalRemoteMemory>(
        g_fx.dir + "/remote_memory.json");
    g_fx.zero->enable_transaction_log(g_fx.dir + "/bridge_zero",
                                      std::string(kRunId) + "-zero");
    for (int r = 0; r < kRanks; ++r) {
        g_fx.zero->set_sys(r, g_fx.systems[static_cast<std::size_t>(r)]);
        g_fx.off->set_sys(r, g_fx.systems[static_cast<std::size_t>(r)]);
    }
}

void issue_bytes(AnalyticalRemoteMemory* mem, int rank, uint64_t bytes,
                 uint64_t node_id) {
    auto* wlhd = new WorkloadLayerHandlerData();
    wlhd->sys_id = rank;
    wlhd->workload = g_fx.recorders[static_cast<std::size_t>(rank)];
    wlhd->node_id = node_id;
    mem->issue(bytes, wlhd);
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
            g_fx.queue->schedule_event(Sys::boostedTick() + 1, noop_cb,
                                       nullptr);
        }
        g_fx.queue->proceed();
        expect(++guard < 1000000, "drain_queue: event loop guard");
    }
}

// 读回明细文件：每行一个 json 对象。
std::vector<nlohmann::json> read_rows(const std::string& path) {
    std::vector<nlohmann::json> rows;
    std::ifstream in(path);
    expect(in.is_open(), "transactions file openable: " + path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        rows.push_back(nlohmann::json::parse(line));
    }
    return rows;
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    build_fixture();

    // ================= 臂 1：武装 + 真实流量 =================
    const Tick t0 = Sys::boostedTick();
    // port0（rank0）双流共享窗口 + port1（rank1）单流。
    issue_bytes(g_fx.on.get(), 0, 300, 101);
    issue_bytes(g_fx.on.get(), 0, 600, 102);
    issue_bytes(g_fx.on.get(), 1, 300, 103);
    drain_queue();
    advance_to(t0 + 1000);
    expect(g_fx.on->is_drained(), "arm1: backend drained");

    const std::string tx_path =
        g_fx.dir + "/bridge_on/remote_memory_transactions.jsonl";
    expect(std::filesystem::exists(tx_path),
           "arm1: transactions file exists (armed + real traffic)");
    const auto rows = read_rows(tx_path);
    expect(rows.size() == 3,
           "arm1: exactly 3 transaction rows (got " +
               std::to_string(rows.size()) + ")");

    // 1a：行键/schema/武装证据 + 1b：Tick/浮点不混用 + 字节守恒。
    uint64_t bytes_total = 0;
    std::map<std::pair<int, uint64_t>, const nlohmann::json*> by_key;
    for (const auto& row : rows) {
        expect(row.at("schema") == 1, "arm1 row schema=1");
        expect(row.at("type") == "remote_memory_transaction",
               "arm1 row type");
        expect(row.at("run_id") == kRunId,
               "arm1 row run_id == armed run id (got " +
                   row.at("run_id").get<std::string>() + ")");
        const uint64_t cb = row.at("callback_tick").get<uint64_t>();
        const double finish = row.at("fluid_finish_ns").get<double>();
        // Tick/浮点不混用：可观察回调恒为 ceil(物理完成时刻)。
        expect(static_cast<double>(cb) == std::ceil(finish),
               "arm1 row callback_tick == ceil(fluid_finish_ns)");
        const uint64_t bytes = row.at("bytes").get<uint64_t>();
        bytes_total += bytes;
        if (bytes > 0) {
            // 正字节事务一次性入流：stream_start == latency_ready。
            expect(row.at("stream_start_ns") == row.at("latency_ready_ns"),
                   "arm1 row stream_start == latency_ready");
        }
        by_key[{row.at("sys_id").get<int>(),
                row.at("node_id").get<uint64_t>()}] = &row;
    }
    expect(bytes_total == 1200, "arm1 rows sum(bytes) == 1200 issued");
    expect(by_key.size() == 3, "arm1 row keys unique");

    // 行键与解析值的逐事务核对（数学锚同 online gate：共享窗口 [100,200)）。
    {
        const auto& a = *by_key.at({0, 101});
        expect(a.at("port_index") == 0 && a.at("issue_sequence") == 0,
               "arm1 A row key (port0, seq0)");
        expect(a.at("latency_ready_ns") == static_cast<double>(t0 + 100) &&
                   a.at("fluid_finish_ns") == static_cast<double>(t0 + 200) &&
                   a.at("callback_tick") == static_cast<uint64_t>(t0 + 200),
               "arm1 A row: ready +100, finish +200, callback +200");
        const auto& b = *by_key.at({0, 102});
        expect(b.at("fluid_finish_ns") == static_cast<double>(t0 + 250) &&
                   b.at("callback_tick") == static_cast<uint64_t>(t0 + 250),
               "arm1 B row: survivor re-split finish +250, callback +250");
        const auto& c = *by_key.at({1, 103});
        expect(c.at("port_index") == 1 &&
                   c.at("fluid_finish_ns") == static_cast<double>(t0 + 150) &&
                   c.at("callback_tick") == static_cast<uint64_t>(t0 + 150),
               "arm1 C row: solo full-rate finish +150, callback +150");
    }

    // 1b（可复算）：由行区间重算 port0 的共享窗口/忙碌/服务量，
    // 与后端事件区间积分的 PortStats 逐值相等。
    {
        struct Iv {
            double start;
            double end;
        };
        std::vector<Iv> ivs0;
        for (const auto& row : rows) {
            if (row.at("port_index") == 0 && row.at("bytes") > 0) {
                ivs0.push_back({row.at("stream_start_ns").get<double>(),
                                row.at("fluid_finish_ns").get<double>()});
            }
        }
        expect(ivs0.size() == 2, "arm1 recompute: two port0 stream rows");
        // 端点扫描：count>=2 的并集即共享窗口，count>=1 的并集即忙碌。
        std::vector<double> cuts;
        for (const auto& iv : ivs0) {
            cuts.push_back(iv.start);
            cuts.push_back(iv.end);
        }
        std::sort(cuts.begin(), cuts.end());
        cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
        double shared = 0.0;
        double busy = 0.0;
        double served = 0.0;
        for (std::size_t i = 0; i + 1 < cuts.size(); ++i) {
            const double mid = 0.5 * (cuts[i] + cuts[i + 1]);
            int n = 0;
            for (const auto& iv : ivs0) {
                if (iv.start <= mid && mid < iv.end) {
                    ++n;
                }
            }
            const double dt = cuts[i + 1] - cuts[i];
            if (n >= 2) {
                shared += dt;
            }
            if (n >= 1) {
                busy += dt;
                // 端口总速率恒为 bw（与流数无关）：served == bw x dt。
                served += 6.0 * dt;
            }
        }
        const auto s0 = g_fx.on->port_stats(0);
        expect_near(s0.shared_busy_ns, shared, 1e-6,
                    "arm1 recompute: shared_busy_ns from rows == PortStats");
        expect(s0.shared_busy_ns > 0.0, "arm1 shared_busy_ns > 0");
        expect_near(s0.port_busy_ns, busy, 1e-6,
                    "arm1 recompute: port_busy_ns from rows == PortStats");
        expect_near(s0.bytes_served, served, 1e-6,
                    "arm1 recompute: bytes_served from rows == PortStats");
    }

    // 1c：完成分布（行 callback_tick 多重集 == 投递回路实测 Tick 集合）。
    {
        std::map<Tick, uint64_t> from_rows;
        for (const auto& row : rows) {
            from_rows[row.at("callback_tick").get<Tick>()] += 1;
        }
        std::map<Tick, uint64_t> from_delivery;
        for (const auto& [key, rec] : g_rec) {
            from_delivery[rec.tick] += 1;
        }
        expect(from_rows == from_delivery,
               "arm1 completion distribution: rows == delivery loop");
    }

    // ================= 臂 2：武装 + 零事务 =================
    {
        advance_to(t0 + 1100);  // 一个空的推进/排空周期
        drain_queue();
        expect(g_fx.zero->is_drained(), "arm2: zero-traffic backend drained");
        const std::string p =
            g_fx.dir + "/bridge_zero/remote_memory_transactions.jsonl";
        expect(!std::filesystem::exists(p),
               "arm2: armed + zero transactions => no file (lazy create, "
               "matches the 2s smoke surface)");
    }

    // ================= 臂 3：未武装 + 同一真实流量 =================
    const Tick t3 = Sys::boostedTick();
    {
        issue_bytes(g_fx.off.get(), 0, 300, 201);
        issue_bytes(g_fx.off.get(), 1, 600, 202);
        drain_queue();
        advance_to(t0 + 1500);
        expect(g_fx.off->is_drained(), "arm3: unarmed backend drained");
        const auto& a = g_rec[201];
        expect(a.count == 1 && a.tick == t3 + 150,
               "arm3: 300B solo completes at issue+150 (300B @ 6B/ns), got " +
                   tick_str(a.tick));
        const auto& b = g_rec[202];
        expect(b.count == 1 && b.tick == t3 + 200,
               "arm3: 600B solo completes at issue+200 (600B @ 6B/ns), got " +
                   tick_str(b.tick));
        expect(!std::filesystem::exists(
                   g_fx.dir + "/bridge_off/remote_memory_transactions.jsonl"),
               "arm3: unarmed + real traffic => no transactions file");
    }

    // ---- teardown：Sys 先删（memory 后删） ----
    for (auto* r : g_fx.recorders) {
        delete r;
    }
    g_fx.recorders.clear();
    for (auto* s : g_fx.systems) {
        delete s;
    }
    g_fx.systems.clear();
    g_fx.sources.clear();
    g_fx.network_apis.clear();
    g_fx.on.reset();
    g_fx.zero.reset();
    g_fx.off.reset();
    AstraSim::LoggerFactory::shutdown();

    std::error_code ec;
    std::filesystem::remove_all(g_fx.dir, ec);

    if (!g_ok) {
        std::fprintf(stderr, "[remote_port_sensing_test] FAIL\n");
        return 1;
    }
    std::printf("[remote_port_sensing_test] ALL PASS: armed + real traffic "
                "writes 3 recomputable transaction rows (run_id, ceil tick, "
                "shared window from rows == PortStats), armed + zero "
                "transactions and unarmed + traffic both leave no file\n");
    return 0;
}
