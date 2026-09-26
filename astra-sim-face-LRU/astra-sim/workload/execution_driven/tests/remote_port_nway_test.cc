/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*

remote_port_nway_test.cc -- SerDes off-chip-link remote-port model precision
fixture (plan: 片外共享内存端口并发化改造执行方案 V5.3, stage 5.1 item 1 and
stage 5.2 math anchors; target
AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest).

Self-contained: the test writes ALL of its configs itself into one isolated
mkdtemp directory (three port-mapping configs PER_NPU / PER_NODE /
MEMORY_POOL plus anchor/stress variants and fail-closed probe configs) and
boots the real online stack (Sys + Workload + HardwareResource + NodeStore +
real AnalyticalRemoteMemory fluid port model + real Sys event queue) per
phase. Nodes are hand-built MEM_LOADs with no local-HBM flags, so each node
is exactly one remote-port transaction (Workload::issue_remote_mem).

Config discipline (plan stage 5 避雷, honored by every fixture here):
system.json follows the official template shape
(the since-removed inputs/system/analytical/dgx_v100_4gpu.json):
"scheduling-policy",
"preferred-dataset-splits" and "collective-optimization" are all present and
all four *-implementation keys are ["ring"] (one per physical dimension: the
phase networks are single-dimension Meshes, so a two-entry list would make
collective_impl.size() exceed the dimension count). No
*-implementation-custom, no doubleBinaryTree, no comm_group.json is involved
(no collective is ever issued; Sys gets the proven "empty" comm-group
argument), and the retired dead key "boost-mode" is NOT resurrected.

Math anchors -- every expected time below is derived PER SERVICE SEGMENT
(N and per-stream rate change at each transition instant; no hand-copied
fixed-rate formula). Fluid finish times are continuous ns; observable
Workload callbacks land on Tick ceil(fluid_finish_ns); the two are asserted
separately (plan sec.3.1). All integer boundaries in the scenarios below are
IEEE-754 exact or >=1 ulp-robust at their magnitudes; each assertion comment
shows its segment table.

Anchor A (bw=6 B/ns, latency=100 ns; PER_NPU phase, per rank):
  two same-tick 600B streams: ready at 100; N=2 -> 3 B/ns each for
  [100,300): 600/3 = 200 ns -> both finish at fluid 300.0, callback Tick 300
  (one batch of two, ordered by issue sequence). Old-serial reference
  200/400 is intentionally NOT produced (stated for contrast only).
Anchor B (bw=100 B/ns, latency=50 ns; dedicated phase):
  four same-tick streams 1k/2k/3k/4kB (kB = 1000 B), all ready at 50:
    [50,90)   N=4 rate 25: each serves 1000 -> 1k done  fluid 90  cb 90
    [90,120)  N=3 rate 100/3: 1k-residue survivors serve 1000 each
              -> 2k done fluid 120 cb 120
    [120,140) N=2 rate 50: survivors serve 1000 each -> 3k done 140
    [140,150] N=1 rate 100: 4k serves its last 1000 -> fluid 150 cb 150
  (each 1000 B segment divides evenly; work-conserving re-split anchors
  90/120/140/150). redistribution_events = 3 (completion instants 90/120/140
  each leave survivors), arrival_redistribution_events = 0.
Anchor C (1B @ 6 B/ns, latency 0): fluid finish ~1/6 ns (sub-Tick), callback
  on the NEXT integer Tick 1 -- the two time notions are asserted separately.

Other covered behaviors, one scenario per rank/port so the math stays exact:
  PER_NPU / PER_NODE (sys_id / num-npus-per-node) / MEMORY_POOL (single
  logical port) mapping via per-port issued/bytes counts; staggered issue
  (arrival-driven re-split listed separately from completion-driven);
  2-stream long/short service with completion-driven re-split; latency-ready
  and stream completion at the same continuous instant and Tick; same-Tick
  multi-port/multi-job completion-batch ordering (port_index asc,
  issue_sequence asc); single non-divisible transaction (ceil callback);
  sub-ns residual service with immediate survivor re-split inside one
  integer Tick; distinct continuous completion instants inside ONE callback
  Tick (redistribution counted per instant); dual-zero (bytes=0, latency=0)
  one-shot timer completing exactly issue+1 ns, never synchronously;
  zero-byte positive-latency transaction never joining the bandwidth
  denominator; huge/NaN-class/invalid bandwidth and latency fail-closed
  (fork probes: explicit exit(1) guards; JSON 1e400 overflows are rejected
  by the vendored nlohmann parser before the backend guard, which is also a
  fail-closed abort); kMaxExactBytes (2^53) byte bound; NO_MEMORY_EXPANSION
  refusing remote runtimes; count/bytes conservation and PortStats
  event-interval integrals on every port; unconditional is_drained().

Merged coverage (2026-09-24 evidence-chain repair; the passing standalone
variant and this in-repo variant are now ONE file):
  RefPort oracle -- a test-side reference fluid model (plan sec.3.1) that
  recomputes every callback from the transaction table and ASSERTS the
  per-segment invariants on the way: equal split (share * N == bw * dt),
  capacity boundary (served == bw * streaming_time while continuously
  active, never above), and the 1e-6 completion-clamp tolerance. Every
  phase below cross-checks its hand-written anchors against RefPort.
  Awaiting-callback exclusion anchor -- a 1B stream that finishes at 100.5
  leaves the bandwidth denominator: 100B completes at 134, NOT 150 (the
  denominator N drops the instant a stream is done, plan sec.7); 200B then
  runs alone to 151.
  Tiny-residue clamp -- a 10B stream whose final share overshoots by ~2e-15
  completes inside the clamp tolerance: no panic, no postponement, no lost
  service.
  Runtime fail-closed probes (on top of the constructor probes): bandwidth
  1e-20 (completion instant overflows Tick at the runtime re-arm) and
  PER_NPU issue from an unconfigured rank must exit(1) at runtime; latency
  1e308 (finite but Tick-overflowing) is forked the same way but fails
  closed earlier, at the constructor's kMaxTimeNs range guard during
  bootstrap_phase -- its issue() call is never reached.

Review-round additions (2026-09-24, plan sec.7 checklist closure):
  Synchronous callback re-entry (sec.3.3) -- phase reentry: the terminal
  callback of transaction A issues follow-up B on the SAME port from inside
  the completion dispatch; in-spot assertions pin B accepted as
  LatencyWaiting (no recursive harvest), deadlined issue=200+latency=100,
  and the re-armed transition delivers B at 400. Without a correct re-arm
  the follow-up is never delivered and the recorder/count FAILs.
  Early shutdown (sec.3.4) -- phase earlyshut: shutdown() called twice
  (idempotence) mid-stream; the undelivered transaction is NEVER delivered
  afterwards (wlhd deleted with the job, global transition event
  cancelled), the loop drains clean, and every PortStats counter/peak/
  integral is reset to zero with is_drained() true.
  Undrained destructor counter-example -- fork probe: a backend destroyed
  with a transaction in flight must sys_panic via verify_drained().
  Stale-generation dispatch branch (backend call(): generation mismatch ->
  return) is deliberately NOT directly exercised: cancel_transition_event()
  removes the queued event and its deleter releases the payload, so a stale
  payload can never reach dispatch through the queue -- the branch is
  defensive depth. Recorded here as the reviewed exemption for plan sec.7's
  "event generation" item; the generation VALUE itself is pinned indirectly
  by every cancellation path above (no double delivery, no leaked payload).
Stress (plan stage 4 evidence): thousands of concurrent streams on ONE port
  (4000) with one distinct completion instant per stream; prints the
  issue/transition/event-loop counters and wall-clock line.

Build (registered in astra-sim/network_frontend/analytical/CMakeLists.txt
next to LocalHbmModelTest):
  cmake --build build/astra_analytical/build_congestion_aware --target \
      AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest
Run:
  build/astra_analytical/build_congestion_aware/bin/\
      AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest
Exit code 0 on ALL PASS.
*/

#include <json/json.hpp>

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include <astra-network-analytical/common/EventQueue.h>
#include <astra-network-analytical/common/NetworkParser.h>
#include <astra-network-analytical/congestion_aware/Helper.h>
#include <remote_memory_backend/analytical/AnalyticalRemoteMemory.hh>

#include <unistd.h>
#include <sys/wait.h>
#include <ctime>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

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
        std::fprintf(stderr, "[remote_port_nway_test] FAIL: %s\n",
                     what.c_str());
        g_ok = false;
    }
}

void expect_near(double value, double expected, double tol,
                 const std::string& what) {
    const double diff =
        value > expected ? value - expected : expected - value;
    if (!(diff <= tol)) {
        std::fprintf(stderr,
                     "[remote_port_nway_test] FAIL: %s: got %.12f expected "
                     "%.12f (tol %.3e)\n",
                     what.c_str(), value, expected, tol);
        g_ok = false;
    }
}

// ---------------------------------------------------------------------------
// RefPort -- test-side reference fluid model (plan sec.3.1 spec oracle,
// merged from the passing standalone variant). Input: one port's issue
// table (issue Tick + bytes); output: callback Tick = ceil(fluid_finish_ns)
// per transaction. Segment boundaries are latency-ready and stream-exhaustion
// instants only. On the way it ASSERTS the per-segment invariants:
//   (a) equal split: every stream in the segment runs at exactly bw/N
//       (share * N == bw * dt);
//   (b) capacity boundary: segment service == bw * segment length, the
//       cumulative service equals bw * streaming_time while continuously
//       active and never exceeds it;
//   (c) the 1e-6 completion clamp: a final share overshooting by <= 1e-6 B
//       completes without panic, postponement or lost service.
// Semantics pinned by plan sec.3.1: a positive-byte stream that becomes
// latency-ready exactly at a segment boundary is served FROM that boundary
// (it never pays a share of [t, boundary)); a zero-byte positive-latency
// transaction completes at its ready instant and never joins the
// denominator; a dual-zero transaction completes asynchronously at
// issue + 1 ns via its own one-shot timer.
// ---------------------------------------------------------------------------

struct RefIssue {
    Tick issue_tick;
    uint64_t bytes;
};

constexpr double kByteClampTol = 1e-6;  // same value as the backend clamp

Tick ref_ceil_tick(double ns) {
    return static_cast<Tick>(std::ceil(ns));
}

std::vector<Tick> ref_port(double bw, double latency,
                           const std::vector<RefIssue>& issues) {
    const std::size_t n = issues.size();
    enum RefState { kWait, kActive, kDone };
    std::vector<RefState> st(n, kWait);
    std::vector<double> ready(n);
    std::vector<double> rem(n);
    std::vector<Tick> cb(n);
    std::size_t done = 0;

    double streaming_time = 0.0;  // cumulative N>=1 service time
    double served_total = 0.0;    // cumulative port service bytes
    double port_time = 0.0;       // port continuous clock (ns)
    bool clock_started = false;

    for (std::size_t i = 0; i < n; ++i) {
        ready[i] = static_cast<double>(issues[i].issue_tick) + latency;
        if (issues[i].bytes == 0 && latency == 0.0) {
            // Dual-zero: independent one-shot timer, exactly issue + 1 ns,
            // never a bandwidth job.
            st[i] = kDone;
            cb[i] = ref_ceil_tick(ready[i] + 1.0);
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

        const std::size_t n_stream = static_cast<std::size_t>(
            std::count(st.begin(), st.end(), kActive));

        // Next event point = min(earliest latency ready, earliest projected
        // stream exhaustion).
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
            port_time = boundary;  // port clock starts at the first event
            clock_started = true;
        }

        // Serve [port_time, boundary): only streams active BEFORE the
        // segment; each gets bw/N and the port serves bw * elapsed.
        std::vector<char> active_before(n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            active_before[i] = (st[i] == kActive) ? 1 : 0;
        }
        if (n_stream > 0) {
            const double elapsed = boundary - port_time;
            const double share =
                elapsed * bw / static_cast<double>(n_stream);
            // (a) equal-split invariant.
            expect(std::abs(share * static_cast<double>(n_stream) -
                            bw * elapsed) <=
                       1e-9 * std::max(1.0, bw * elapsed),
                   "RefPort: equal-split invariant violated");
            for (std::size_t i = 0; i < n; ++i) {
                if (active_before[i]) {
                    rem[i] -= share;
                }
            }
            streaming_time += elapsed;
            served_total += share * static_cast<double>(n_stream);
            // (b) capacity boundary: work-conserving equality, never above.
            expect(served_total <= bw * streaming_time + 1e-9,
                   "RefPort: served bytes exceed bw x streaming_time");
            expect(served_total >= bw * streaming_time - 1e-9,
                   "RefPort: served bytes lost vs bw x streaming_time");
        }
        port_time = boundary;

        // Latency ready: positive bytes join the stream set; zero bytes
        // complete at the ready instant without joining the denominator.
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

        // Exhaustion: simultaneous finishers leave as one batch; a remainder
        // within the clamp tolerance completes (no panic, no delay, no loss).
        for (std::size_t i = 0; i < n; ++i) {
            if (st[i] != kActive) {
                continue;
            }
            if (rem[i] <= kByteClampTol) {
                expect(rem[i] >= -kByteClampTol,
                       "RefPort: stream remainder fell beyond clamp "
                       "tolerance");
                rem[i] = 0.0;
                st[i] = kDone;
                cb[i] = ref_ceil_tick(boundary);
                ++done;
            }
        }
    }
    return cb;
}

void write_file(const std::string& path, const std::string& content) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
        std::fprintf(stderr, "[remote_port_nway_test] FAIL: cannot write %s\n",
                     path.c_str());
        g_ok = false;
        return;
    }
    out << content;
}

// ---------------------------------------------------------------------------
// Terminal observation: the CompletionObserver hook fires synchronously on
// the Workload::call terminal path, i.e. in the backend's delivery order
// (port_index asc, issue_sequence asc within one global completion batch).
// ---------------------------------------------------------------------------

struct Recorder {
    // (rank, node_id, tick) in exact delivery order.
    std::vector<std::tuple<int, uint64_t, uint64_t>> order;
    std::map<std::pair<int, uint64_t>, uint64_t> tick_of;
    std::map<std::pair<int, uint64_t>, uint64_t> count_of;

    void clear() {
        order.clear();
        tick_of.clear();
        count_of.clear();
    }
    uint64_t tick(int rank, uint64_t node) const {
        auto it = tick_of.find({rank, node});
        return it == tick_of.end() ? 0 : it->second;
    }
    uint64_t count(int rank, uint64_t node) const {
        auto it = count_of.find({rank, node});
        return it == count_of.end() ? 0 : it->second;
    }
} g_recorder;

void terminal_hook(void* ctx, int rank, uint64_t node_id, const char*,
                   const char*, uint64_t, uint64_t tick, int status) {
    auto* rec = static_cast<Recorder*>(ctx);
    if (status !=
        static_cast<int>(ExecutionDriven::NodeTerminalStatus::Success)) {
        std::fprintf(stderr,
                     "[remote_port_nway_test] FAIL: rank %d node %llu "
                     "terminal status %d\n",
                     rank, static_cast<unsigned long long>(node_id), status);
        g_ok = false;
    }
    rec->order.emplace_back(rank, node_id, tick);
    rec->tick_of[{rank, node_id}] = tick;
    rec->count_of[{rank, node_id}] += 1;
}

// ---------------------------------------------------------------------------
// Synchronous re-entry context (plan sec.3.3: "a completion callback may
// synchronously re-enter the Workload and issue new nodes", which re-arms
// the global transition event from inside the dispatch of the previous
// batch). phase_reentry_issue installs reentry_hook, which records the
// terminal fact first and THEN, for its configured trigger node, issues a
// follow-up MEM on the SAME port from inside the terminal path and asserts
// the backend's on-the-spot state.
// ---------------------------------------------------------------------------
struct ReentryCtx {
    bool armed = false;
    bool fired = false;
    Sys* sys = nullptr;
    ExecutionDriven::NodeStoreGraphSource* source = nullptr;
    AnalyticalRemoteMemory* mem = nullptr;
    int trigger_rank = -1;
    uint64_t trigger_node = 0;
} g_reentry;

// Defined with the issue plumbing below; forward-declared so the re-entry
// hook can live next to the recorder it extends.
ExecutionDriven::OnlineNode make_mem_node(int rank, uint64_t bytes,
                                          const std::string& name);
void add_and_issue(Sys* sys, ExecutionDriven::NodeStoreGraphSource* source,
                   ExecutionDriven::OnlineNode node);

void reentry_hook(void* ctx, int rank, uint64_t node_id, const char* name,
                  const char* stage, uint64_t generation, uint64_t tick,
                  int status) {
    // Record the terminal fact exactly like the plain hook first.
    terminal_hook(ctx, rank, node_id, name, stage, generation, tick, status);
    if (!g_reentry.armed || rank != g_reentry.trigger_rank ||
        node_id != g_reentry.trigger_node) {
        return;
    }
    g_reentry.armed = false;
    g_reentry.fired = true;
    // §3.3 re-entry: issue the follow-up transaction synchronously from the
    // completion-callback stack, then assert the backend accepted it into
    // the latency stage WITHOUT recursive dispatch of the just-issued job.
    add_and_issue(g_reentry.sys, g_reentry.source,
                  make_mem_node(rank, 600, "re_b"));
    const auto jobs = g_reentry.mem->get_port_jobs(
        static_cast<std::size_t>(rank));
    expect(jobs.size() == 1,
           "reentry: follow-up accepted synchronously (got " +
               std::to_string(jobs.size()) + " live jobs)");
    if (jobs.size() == 1) {
        expect(jobs[0].state ==
                   AnalyticalRemoteMemory::PortJobState::LatencyWaiting,
               "reentry: follow-up sits in LatencyWaiting -- the in-flight "
               "batch dispatch did NOT recursively harvest it");
        expect_near(jobs[0].issue_ns, 200.0, 1e-9,
                    "reentry: follow-up issued at the dispatch Tick 200");
        expect_near(jobs[0].ready_ns, 300.0, 1e-9,
                    "reentry: follow-up deadline = issue 200 + latency 100");
    }
}

void expect_terminal(const Recorder& rec, int rank, uint64_t node,
                     uint64_t tick, const std::string& what) {
    expect(rec.count(rank, node) == 1,
           what + ": terminal exactly once (rank " + std::to_string(rank) +
               " node " + std::to_string(node) + ", got " +
               std::to_string(rec.count(rank, node)) + ")");
    expect(rec.tick(rank, node) == tick,
           what + ": callback tick (rank " + std::to_string(rank) +
               " node " + std::to_string(node) + ") got " +
               std::to_string(rec.tick(rank, node)) + " expected " +
               std::to_string(tick));
}

// Full-sequence delivery assertion: (rank, node, tick) at every position,
// pinning both the (port asc, issue_sequence asc) batch order and the
// interleaving of consecutive batches on the global timeline.
void expect_order(const Recorder& rec,
                  const std::vector<std::tuple<int, uint64_t, uint64_t>>&
                      expected,
                  const std::string& what) {
    if (rec.order.size() != expected.size()) {
        expect(false, what + ": delivery count " +
                          std::to_string(rec.order.size()) + " expected " +
                          std::to_string(expected.size()));
        return;
    }
    for (size_t i = 0; i < expected.size(); i++) {
        const auto& got = rec.order[i];
        if (static_cast<int>(std::get<0>(got)) != std::get<0>(expected[i]) ||
            std::get<1>(got) != std::get<1>(expected[i]) ||
            std::get<2>(got) != std::get<2>(expected[i])) {
            expect(false,
                   what + ": delivery order position " +
                       std::to_string(i) + " got (rank " +
                       std::to_string(std::get<0>(got)) + " node " +
                       std::to_string(std::get<1>(got)) + " tick " +
                       std::to_string(std::get<2>(got)) +
                       ") expected (rank " +
                       std::to_string(std::get<0>(expected[i])) +
                       " node " +
                       std::to_string(std::get<1>(expected[i])) +
                       " tick " +
                       std::to_string(std::get<2>(expected[i])) + ")");
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Issue plumbing (mirrors local_hbm_model_test.cc): MEM_LOAD with no
// hbm-access-mode and no kv-restore flag == exactly one remote-port
// transaction.
// ---------------------------------------------------------------------------

ExecutionDriven::OnlineNode make_mem_node(int rank, uint64_t bytes,
                                          const std::string& name) {
    ExecutionDriven::OnlineNode node;
    node.rank = rank;
    node.kind = ExecutionDriven::NodeKind::MemLoad;
    node.node_type = 2;  // ChakraNodeType::MEM_LOAD_NODE
    node.name = name;
    node.compute.tensor_size = bytes;
    return node;
}

void issue_all_dep_free(Sys* sys,
                        ExecutionDriven::NodeStoreGraphSource* source) {
    for (const auto& nv : source->dep_free_nodes()) {
        if (sys->workload->hw_resource->is_available(nv)) {
            sys->workload->issue(nv);
        }
    }
}

void add_and_issue(Sys* sys, ExecutionDriven::NodeStoreGraphSource* source,
                   ExecutionDriven::OnlineNode node) {
    source->store().add_node(std::move(node));
    issue_all_dep_free(sys, source);
}

// Staggered issue: fires from the real Sys event queue at Tick == delay.
class StaggeredIssueEvent : public Callable {
  public:
    StaggeredIssueEvent(Sys* sys,
                        ExecutionDriven::NodeStoreGraphSource* source,
                        ExecutionDriven::OnlineNode node)
        : sys_(sys), source_(source), node_(std::move(node)) {}

    void call(EventType, CallData*) override {
        add_and_issue(sys_, source_, std::move(node_));
        delete this;
    }

  private:
    Sys* sys_;
    ExecutionDriven::NodeStoreGraphSource* source_;
    ExecutionDriven::OnlineNode node_;
};

void schedule_issue_at(Sys* sys,
                       ExecutionDriven::NodeStoreGraphSource* source,
                       ExecutionDriven::OnlineNode node, Tick delay) {
    sys->register_event(new StaggeredIssueEvent(sys, source, std::move(node)),
                        EventType::General, nullptr, delay);
}

// Mid-flight probe: runs from the real event queue at Tick == delay.
class ProbeEvent : public Callable {
  public:
    explicit ProbeEvent(std::function<void()> fn) : fn_(std::move(fn)) {}

    void call(EventType, CallData*) override {
        fn_();
        delete this;
    }

  private:
    std::function<void()> fn_;
};

void schedule_probe_at(Sys* sys, Tick delay, std::function<void()> fn) {
    sys->register_event(new ProbeEvent(std::move(fn)), EventType::General,
                        nullptr, delay);
}

// ---------------------------------------------------------------------------
// Per-phase stack assembly / teardown (fresh event queue + topology + fluid
// scheduler + memory per phase; Sys statics tolerate sequential phases).
// ---------------------------------------------------------------------------

// Official template shape (the since-removed
// inputs/system/analytical/dgx_v100_4gpu.json):
// the three discipline keys + all four *-implementation keys as
// ["ring","ring"]. local-mem keys deliberately absent (no local HBM user in
// this fixture; the N-way HBM model then auto-disables). The retired dead
// key "boost-mode" is intentionally NOT resurrected.
const char* kSystemJson =
    "{\n"
    "  \"scheduling-policy\": \"LIFO\",\n"
    "  \"endpoint-delay\": 1,\n"
    "  \"active-chunks-per-dimension\": 1,\n"
    "  \"preferred-dataset-splits\": 1,\n"
    "  \"all-reduce-implementation\": [\"ring\"],\n"
    "  \"all-gather-implementation\": [\"ring\"],\n"
    "  \"reduce-scatter-implementation\": [\"ring\"],\n"
    "  \"all-to-all-implementation\": [\"ring\"],\n"
    "  \"collective-optimization\": \"localBWAware\"\n"
    "}\n";

struct PhaseStack {
    std::shared_ptr<EventQueue> event_queue;
    std::shared_ptr<NetworkParser> parser;
    std::shared_ptr<Topology> topology;
    std::shared_ptr<FluidScheduler> fluid_scheduler;
    std::unique_ptr<AnalyticalRemoteMemory> memory;
    std::vector<std::unique_ptr<CongestionAwareNetworkApi>> network_apis;
    std::vector<Sys*> systems;
    std::vector<std::shared_ptr<ExecutionDriven::NodeStoreGraphSource>>
        sources;
};

void bootstrap_phase(const std::string& dir, const std::string& tag,
                     int ranks, const std::string& remote_cfg,
                     PhaseStack& stack) {
    // Recorder is per-phase: rank/node ids repeat across phases.
    g_recorder.clear();
    const std::string system_path = dir + "/system_" + tag + ".json";
    const std::string network_path = dir + "/network_" + tag + ".yml";
    write_file(system_path, kSystemJson);
    write_file(network_path,
               "topology: [ Mesh ]\n"
               "npus_count: [ " +
                   std::to_string(ranks) +
                   " ]\n"
                   "bandwidth: [ 4050 ]\n"
                   "latency: [ 25 ]\n");

    stack.event_queue = std::make_shared<EventQueue>();
    stack.parser = std::make_shared<NetworkParser>(network_path);
    stack.topology = construct_topology(*stack.parser);
    CongestionAwareNetworkApi::set_event_queue(stack.event_queue);
    CongestionAwareNetworkApi::set_topology(stack.topology);
    stack.fluid_scheduler = std::make_shared<FluidScheduler>(
        stack.event_queue, stack.topology->get_directed_links(),
        stack.parser->get_fluid_max_active_flows(),
        stack.parser->get_fluid_max_route_memberships(),
        stack.parser->get_progress_report_event_interval());
    CongestionAwareNetworkApi::set_fluid_scheduler(stack.fluid_scheduler);

    stack.memory = std::make_unique<AnalyticalRemoteMemory>(remote_cfg);

    const auto dims = stack.topology->get_npus_count_per_dim();
    const std::vector<int> queues_per_dim(dims.size(), 1);
    expect(static_cast<int>(stack.topology->get_npus_count()) == ranks,
           "phase " + tag + ": topology rank count");
    for (int i = 0; i < ranks; i++) {
        auto source = std::make_shared<ExecutionDriven::NodeStoreGraphSource>();
        stack.sources.push_back(source);
        auto net = std::make_unique<CongestionAwareNetworkApi>(i);
        stack.systems.push_back(new Sys(
            i, dir + "/none.et", "empty", system_path, stack.memory.get(),
            net.get(), dims, queues_per_dim, 1.0, 1.0, false,
            ExecutionDriven::ExecutionMode::Online, source));
        stack.network_apis.push_back(std::move(net));
    }
}

void teardown_phase(PhaseStack& stack) {
    for (Sys* sys : stack.systems) {
        delete sys;
    }
    stack.systems.clear();
    stack.sources.clear();
    stack.network_apis.clear();
    // ~AnalyticalRemoteMemory runs the unconditional verify_drained()
    // fail-closed audit (plan sec.3.4); an undrained backend aborts here.
    stack.memory.reset();
    stack.fluid_scheduler.reset();
    stack.topology.reset();
    stack.parser.reset();
    stack.event_queue.reset();
}

// Real Sys/network event loop with two watchdogs (convert a hypothetical
// zero-progress pathology into a failed assertion WITH diagnostics instead
// of a silent hang): a proceed-count cap and a wall-clock cap that dumps
// the in-flight state of every port before bailing out.
uint64_t run_event_loop(EventQueue& queue, const std::string& tag,
                        const AnalyticalRemoteMemory& mem,
                        std::size_t port_count) {
    const uint64_t kProceedCap = 50000000;
    const auto kWallCap = std::chrono::seconds(300);
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t proceeds = 0;
    while (!queue.finished()) {
        queue.proceed();
        proceeds++;
        const auto elapsed = std::chrono::steady_clock::now() - t0;
        if (proceeds > kProceedCap || elapsed > kWallCap) {
            std::fprintf(stderr,
                         "[remote_port_nway_test] phase %s: event loop "
                         "watchdog fired after %llu proceeds "
                         "(proceed cap) / wall clock stuck\n",
                         tag.c_str(),
                         static_cast<unsigned long long>(proceeds));
            for (std::size_t p = 0; p < port_count; p++) {
                const auto st = mem.get_port_stats(p);
                std::fprintf(stderr,
                             "  port %zu: issued=%llu completed=%llu "
                             "in_flight=%llu streaming=%llu "
                             "latency_waiting=%llu completion_waiting=%llu "
                             "dual_zero=%llu\n",
                             p,
                             static_cast<unsigned long long>(
                                 st.issued_count),
                             static_cast<unsigned long long>(
                                 st.completed_count),
                             static_cast<unsigned long long>(
                                 st.in_flight_count),
                             static_cast<unsigned long long>(
                                 st.streaming_count),
                             static_cast<unsigned long long>(
                                 st.latency_waiting_count),
                             static_cast<unsigned long long>(
                                 st.completion_waiting_count),
                             static_cast<unsigned long long>(
                                 st.dual_zero_timer_count));
            }
            std::fprintf(stderr, "  terminals=%zu recorder_done\n",
                         g_recorder.order.size());
            expect(false, "phase " + tag + ": event loop watchdog fired");
            return proceeds;
        }
    }
    return proceeds;
}

// plan sec.5.1 conservation: count/bytes pairs closed, and the served-bytes
// integral within the per-job completion residue bound (kByteEps +
// bw * kTimeEpsNs per completed transaction).
void expect_port_conservation(const AnalyticalRemoteMemory& mem,
                              std::size_t port, double bw,
                              const std::string& tag) {
    const auto st = mem.get_port_stats(port);
    expect(st.issued_count == st.completed_count,
           tag + ": port " + std::to_string(port) + " issued_count " +
               std::to_string(st.issued_count) + " == completed_count " +
               std::to_string(st.completed_count));
    expect(st.issued_bytes == st.completed_bytes,
           tag + ": port " + std::to_string(port) + " issued_bytes " +
               std::to_string(st.issued_bytes) + " == completed_bytes " +
               std::to_string(st.completed_bytes));
    expect(st.in_flight_count == 0,
           tag + ": port " + std::to_string(port) + " drained in-flight");
    const double signed_gap = static_cast<double>(st.completed_bytes) -
                              st.bytes_served;
    const double gap = signed_gap < 0 ? -signed_gap : signed_gap;
    const double bound = static_cast<double>(st.completed_count) *
                         (1e-6 + bw * 1e-9);
    expect(gap <= bound,
           tag + ": port " + std::to_string(port) + " bytes_served " +
               std::to_string(st.bytes_served) + " vs completed_bytes " +
               std::to_string(st.completed_bytes) + " within residue bound");
}

// ---------------------------------------------------------------------------
// Fail-closed fork probes (plan sec.3.2 / stage 5.1). Each probe runs in a
// forked child so an intended exit(1)/abort proves the guard without ending
// the test process. A child that survives (exit 0) means the guard is
// missing. JSON cannot carry NaN/Inf literals (RFC 8259); the reachable
// non-finite boundary is the 1e400 overflow, which the vendored nlohmann
// parser rejects with out_of_range.406 -> the child aborts: still
// fail-closed, still never accepted.
// ---------------------------------------------------------------------------

void expect_child_rejected(const std::function<void()>& body,
                           const std::string& what) {
    std::fflush(nullptr);
    const pid_t pid = fork();
    if (pid == 0) {
        // The guard paths end in the backend's exit(1). In a forked child
        // that exit runs the static-destructor pass, which deadlocks on
        // the forked-away logger state and stalls the probe until an
        // alarm fires (observed: every exit(1) probe stalled and only
        // finished via its SIGALRM). Bypass the destructor pass: turn any
        // exit(code) into an immediate _exit(1) -- atexit handlers run
        // BEFORE static destruction -- restore the default SIGALRM
        // disposition, and keep a short alarm as the last-resort kill.
        // The survived path below uses _exit(0), which does not run
        // atexit handlers.
        signal(SIGALRM, SIG_DFL);
        alarm(10);
        atexit([]() { _exit(1); });
        body();
        _exit(0);  // survived: the fail-closed guard did NOT fire
    }
    // Poll instead of a blocking waitpid: the probe child's guard path ends
    // in exit(1), whose static-destructor pass (LoggerFactory) has been
    // observed to stall in a forked child; a stalled child is SIGKILLed and
    // still counts as rejected (it never survived with exit 0).
    constexpr int kPollRounds = 3000;  // 3000 x 10ms = 30 s budget
    int status = 0;
    bool reaped = false;
    for (int i = 0; i < kPollRounds; i++) {
        const pid_t done = waitpid(pid, &status, WNOHANG);
        if (done == pid) {
            reaped = true;
            break;
        }
        timespec ts = {0, 10 * 1000 * 1000};  // 10 ms
        nanosleep(&ts, nullptr);
    }
    if (!reaped) {
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
        std::fprintf(stderr,
                     "[remote_port_nway_test] probe child stalled and was "
                     "SIGKILLed (%s)\n",
                     what.c_str());
    }
    const bool survived = WIFEXITED(status) && WEXITSTATUS(status) == 0;
    expect(!survived, "fail-closed (child must not survive): " + what);
}

void run_fail_closed_probes(const std::string& dir) {
    const auto reject_construct = [&](const std::string& cfg,
                                      const std::string& what) {
        expect_child_rejected(
            [&cfg]() { AnalyticalRemoteMemory mem(cfg); }, what);
    };
    const auto reject_runtime = [&](const std::string& cfg, uint64_t bytes,
                                    const std::string& what) {
        expect_child_rejected(
            [&cfg, bytes]() {
                AnalyticalRemoteMemory mem(cfg);
                mem.get_remote_mem_runtime(bytes);
            },
            what);
    };

    // Invalid bandwidth: 0 / negative / wrong JSON type -> explicit
    // "remote-mem-bw must be positive" exit(1) (AnalyticalRemoteMemory.cc).
    reject_construct(dir + "/pf_bw_zero.json",
                     "remote-mem-bw = 0 rejected");
    reject_construct(dir + "/pf_bw_neg.json",
                     "remote-mem-bw = -5 rejected");
    reject_construct(dir + "/pf_bw_str.json",
                     "non-numeric remote-mem-bw rejected");

    // Negative / overflow latency -> explicit guards (overflow again via the
    // parser's 406 before the isfinite guard).
    reject_construct(dir + "/pf_lat_neg.json",
                     "negative remote-mem-latency rejected");
    reject_construct(dir + "/pf_lat_ovf.json",
                     "1e400 remote-mem-latency rejected");
    reject_construct(dir + "/pf_bw_ovf.json",
                     "1e400 remote-mem-bw rejected");

    // Huge byte count: the 2^53 exact-representation boundary fails closed
    // (same kMaxExactBytes guard as issue()).
    reject_runtime(dir + "/pf_ok.json", (1ULL << 53) + 1,
                   "tensor_size > 2^53 rejected");

    // NO_MEMORY_EXPANSION has no remote port: runtimes fail closed
    // (plan sec.7 mapping judgment gate).
    reject_runtime(dir + "/pf_nomem.json", 1,
                   "NO_MEMORY_EXPANSION runtime rejected");

    // Tick-overflow / fail-closed probes (merged from the passing standalone
    // variant): finite-but-overflowing latency / bandwidth and an
    // unconfigured PER_NPU rank. Each builds its own minimal 1-rank stack
    // inside the forked child (the parent has built no Sys yet at probe
    // time; the child's atexit->_exit guard skips the teardown pass).
    // Exit sites differ: the 1e308 latency child already exits(1) inside
    // bootstrap_phase at the constructor's kMaxTimeNs range guard
    // (AnalyticalRemoteMemory.cc), while the 1e-20 bandwidth and bad-rank
    // children pass construction and fail at runtime.
    expect_child_rejected(
        [&dir]() {
            PhaseStack stack;
            const std::string cfg = dir + "/remote_pf_latovf_rt.json";
            write_file(cfg,
                       "{\n"
                       "  \"memory-type\": \"PER_NPU_MEMORY_EXPANSION\",\n"
                       "  \"npu-ids\": [0],\n"
                       "  \"remote-mem-bw\": 6,\n"
                       "  \"remote-mem-latency\": 1e+308\n"
                       "}\n");
            bootstrap_phase(dir, "pf_latovf_rt", 1, cfg, stack);
            WorkloadLayerHandlerData wlhd{};
            wlhd.sys_id = 0;
            wlhd.workload = stack.systems[0]->workload;
            wlhd.node_id = 9901;
            stack.memory->issue(0, &wlhd  // zero bytes: normally unreachable;
                                         // the 1e308 latency exits(1) at the
                                         // constructor kMaxTimeNs guard inside
                                         // bootstrap_phase (a call that got
                                         // here would still fail closed on
                                         // the ready_ns projection)
                                );
            std::printf("C6 DID NOT fail closed\n");
        },
        "runtime latency 1e308 Tick overflow rejected");
    expect_child_rejected(
        [&dir]() {
            PhaseStack stack;
            const std::string cfg = dir + "/remote_pf_bwuf_rt.json";
            write_file(cfg,
                       "{\n"
                       "  \"memory-type\": \"PER_NPU_MEMORY_EXPANSION\",\n"
                       "  \"npu-ids\": [0],\n"
                       "  \"remote-mem-bw\": 1e-20,\n"
                       "  \"remote-mem-latency\": 100\n"
                       "}\n");
            bootstrap_phase(dir, "pf_bwuf_rt", 1, cfg, stack);
            WorkloadLayerHandlerData wlhd{};
            wlhd.sys_id = 0;
            wlhd.workload = stack.systems[0]->workload;
            wlhd.node_id = 9902;
            stack.memory->issue(1, &wlhd  // 1B at 1e-20 B/ns: the projected
                                         // completion instant overflows Tick
                                         // at the transition re-arm
                                );
            while (!stack.event_queue->finished()) {
                stack.event_queue->proceed();
            }
            std::printf("C7 DID NOT fail closed\n");
        },
        "runtime bandwidth 1e-20 Tick overflow rejected");
    expect_child_rejected(
        [&dir]() {
            PhaseStack stack;
            const std::string cfg = dir + "/remote_pf_badrank.json";
            write_file(cfg,
                       "{\n"
                       "  \"memory-type\": \"PER_NPU_MEMORY_EXPANSION\",\n"
                       "  \"npu-ids\": [0],\n"
                       "  \"remote-mem-bw\": 6,\n"
                       "  \"remote-mem-latency\": 100\n"
                       "}\n");
            bootstrap_phase(dir, "pf_badrank", 1, cfg, stack);
            WorkloadLayerHandlerData wlhd{};
            wlhd.sys_id = 5;  // NOT in npu-ids [0]
            wlhd.workload = stack.systems[0]->workload;
            wlhd.node_id = 9903;
            stack.memory->issue(
                64, &wlhd  // an unconfigured rank must never fall back to a
                          // default port: fail closed
            );
            std::printf("C9 DID NOT fail closed\n");
        },
        "PER_NPU issue from an unconfigured rank rejected");

    // Undrained-destructor counter-example (plan sec.7 "remote jobs / global
    // event empty set" negative case + sec.3.4 normal-end fail-closed): a
    // backend destroyed while a transaction is still in flight must panic
    // through verify_drained() instead of silently succeeding. The child
    // issues one transaction, never runs the event loop and never calls
    // shutdown(), then lets the PhaseStack's unique_ptr run the destructor.
    expect_child_rejected(
        [&dir]() {
            PhaseStack stack;
            const std::string cfg = dir + "/remote_pf_undrained.json";
            write_file(cfg,
                       "{\n"
                       "  \"memory-type\": \"PER_NPU_MEMORY_EXPANSION\",\n"
                       "  \"npu-ids\": [0],\n"
                       "  \"remote-mem-bw\": 6,\n"
                       "  \"remote-mem-latency\": 100\n"
                       "}\n");
            bootstrap_phase(dir, "pf_undrained", 1, cfg, stack);
            ExecutionDriven::OnlineNode node = make_mem_node(0, 600, "u");
            stack.sources[0]->store().add_node(std::move(node));
            for (const auto& nv : stack.sources[0]->dep_free_nodes()) {
                if (stack.systems[0]->workload->hw_resource->is_available(nv)) {
                    stack.systems[0]->workload->issue(nv);
                }
            }
            // No event loop, no shutdown(): the in-flight transaction makes
            // is_drained() false; the destructor must sys_panic(exit(1)).
            stack.sources.clear();
            stack.memory.reset();
            std::printf("C10 DID NOT fail closed\n");
        },
        "undrained backend destructor fails closed (verify_drained)");
}

}  // namespace

// ---------------------------------------------------------------------------
// Phase P: PER_NPU anchors at bw=6 B/ns, latency=100 ns, 7 ranks
// (npu-ids 0..6, one port per rank; port_index == rank).
// ---------------------------------------------------------------------------
namespace {

void phase_pernpu_anchors(const std::string& dir) {
    const std::string tag = "pernpup";
    const std::string cfg_path = dir + "/remote_" + tag + ".json";
    write_file(cfg_path,
               "{\n"
               "  \"memory-type\": \"PER_NPU_MEMORY_EXPANSION\",\n"
               "  \"npu-ids\": [0, 1, 2, 3, 4, 5, 6],\n"
               "  \"remote-mem-bw\": 6,\n"
               "  \"remote-mem-latency\": 100\n"
               "}\n");

    PhaseStack stack;
    bootstrap_phase(dir, tag, 7, cfg_path, stack);
    auto* mem = stack.memory.get();

    // Rank 0 -- Anchor A: two same-tick 600B streams.
    // [0,100) latency (overlapped); [100,300) N=2 -> 3 B/ns each;
    // 600/3 = 200 -> both fluid 300.0, callback Tick 300 (old serial FIFO
    // would answer 200/400 -- only kept as a contrast comment).
    add_and_issue(stack.systems[0], stack.sources[0].get(),
                  make_mem_node(0, 600, "a1_s0"));
    add_and_issue(stack.systems[0], stack.sources[0].get(),
                  make_mem_node(0, 600, "a1_s1"));

    // Rank 1 -- 2-stream long/short with completion-driven re-split:
    // [100,200) N=2 rate 3: short 300 done fluid 200 (redistribution +1,
    // survivor long); long served 300, rem 1800.
    // [200,500] N=1 rate 6: 1800/6 = 300 -> long fluid 500.0.
    add_and_issue(stack.systems[1], stack.sources[1].get(),
                  make_mem_node(1, 300, "a2_short"));
    add_and_issue(stack.systems[1], stack.sources[1].get(),
                  make_mem_node(1, 2100, "a2_long"));

    // Rank 2 -- staggered issue (arrival-driven re-split):
    // A=1200B at t0: [100,200) alone at 6 -> 600 served, rem 600.
    // B=600B issued at Tick 100 (ready 200): [200,400] N=2 rate 3 ->
    // A rem 600/3 = 200 -> fluid 400.0; B 600/3 = 200 -> fluid 400.0.
    // Same-instant finishers, no survivor -> completion redist 0;
    // arrival redist +1 (B joins while A streams).
    add_and_issue(stack.systems[2], stack.sources[2].get(),
                  make_mem_node(2, 1200, "a3_A"));
    schedule_issue_at(stack.systems[2], stack.sources[2].get(),
                      make_mem_node(2, 600, "a3_B"), 100);

    // Rank 3 -- latency-ready and stream completion at the SAME continuous
    // instant and Tick: A=300B at t0 streams [100,150) alone at 6 -> 300
    // served -> fluid 150.0 exactly, cb 150; B=600B issued at Tick 50 has
    // ready 150 and is admitted at that same instant (stream_start 150.0).
    // B then runs [150,250] alone at 6 -> fluid 250.0, cb 250.
    add_and_issue(stack.systems[3], stack.sources[3].get(),
                  make_mem_node(3, 300, "a4_A"));
    schedule_issue_at(stack.systems[3], stack.sources[3].get(),
                      make_mem_node(3, 600, "a4_B"), 50);
    schedule_probe_at(
        stack.systems[3], 160, [mem]() {
            const auto jobs = mem->get_port_jobs(3);
            expect(jobs.size() == 1, "a4 probe: exactly B still active");
            if (jobs.size() == 1) {
                expect(jobs[0].state ==
                           AnalyticalRemoteMemory::PortJobState::ActiveStream,
                       "a4 probe: B is ActiveStream");
                expect_near(jobs[0].stream_start_ns, 150.0, 1e-9,
                            "a4 probe: B stream_start == A completion "
                            "instant 150");
                // Exact per-segment projection: 600 - 6 B/ns * 10 ns.
                expect_near(jobs[0].remaining_bytes, 540.0, 1e-9,
                            "a4 probe: B remaining 540 at Tick 160");
            }
        });

    // Rank 4 -- zero-byte positive-latency transaction never joins the
    // bandwidth denominator: the 0B job flips to awaiting-callback at its
    // ready instant 100 (cb 100) without entering the stream set, so the
    // 600B job runs ALONE at 6 B/ns on [100,200] -> fluid 200.0, cb 200
    // (a denominator bug would dilute it to 3 B/ns and answer 300).
    add_and_issue(stack.systems[4], stack.sources[4].get(),
                  make_mem_node(4, 600, "a5_bytes"));
    add_and_issue(stack.systems[4], stack.sources[4].get(),
                  make_mem_node(4, 0, "a5_zerobyte"));
    schedule_probe_at(
        stack.systems[4], 150, [mem]() {
            const auto jobs = mem->get_port_jobs(4);
            expect(jobs.size() == 1,
                   "a5 probe: only the 600B job remains at Tick 150");
            if (jobs.size() == 1) {
                expect(jobs[0].tensor_size == 600,
                       "a5 probe: remaining job is the 600B stream");
            }
        });

    // Rank 5 -- single non-divisible transaction: 1000B at 6 B/ns ->
    // service 1000/6 = 166.666.. ns, fluid 266.666.., callback
    // ceil(266.666..) = 267 (Tick is the ceil, not the fluid time).
    add_and_issue(stack.systems[5], stack.sources[5].get(),
                  make_mem_node(5, 1000, "a6_nondiv"));

    // Rank 6 -- second half of the same-Tick multi-port ordering batch:
    // two 600B streams like rank 0 -> both cb 300. The single global
    // transition event at Tick 300 harvests ports 0 and 6 as ONE batch of
    // four, delivered (port asc, issue_sequence asc):
    // (0,n1) (0,n2) (6,n1) (6,n2).
    add_and_issue(stack.systems[6], stack.sources[6].get(),
                  make_mem_node(6, 600, "a7_s0"));
    add_and_issue(stack.systems[6], stack.sources[6].get(),
                  make_mem_node(6, 600, "a7_s1"));

    run_event_loop(*stack.event_queue, tag, *mem, 7);

    // -- mapping: 7 ports, one per configured npu-id.
    expect(mem->get_port_count() == 7, tag + ": PER_NPU port count 7");

    // -- callbacks (Anchor A + every scenario).
    expect_terminal(g_recorder, 0, 1, 300, tag + " anchor A s0");
    expect_terminal(g_recorder, 0, 2, 300, tag + " anchor A s1");
    expect_terminal(g_recorder, 1, 1, 200, tag + " a2 short");
    expect_terminal(g_recorder, 1, 2, 500, tag + " a2 long");
    expect_terminal(g_recorder, 2, 1, 400, tag + " a3 A");
    expect_terminal(g_recorder, 2, 2, 400, tag + " a3 B");
    expect_terminal(g_recorder, 3, 1, 150, tag + " a4 A");
    expect_terminal(g_recorder, 3, 2, 250, tag + " a4 B");
    expect_terminal(g_recorder, 4, 1, 200, tag + " a5 600B");
    expect_terminal(g_recorder, 4, 2, 100, tag + " a5 0B");
    expect_terminal(g_recorder, 5, 1, 267, tag + " a6 non-divisible");
    expect_terminal(g_recorder, 6, 1, 300, tag + " a7 s0");
    expect_terminal(g_recorder, 6, 2, 300, tag + " a7 s1");

    // -- full global delivery order across every completion batch on the
    // timeline: Tick 100 (4,2 zero-byte) / 150 (3,1) / 200 batch (1,1)+(4,1)
    // / 250 (3,2) / 267 (5,1) / the Tick-300 four-job two-port batch
    // (0,1)(0,2)(6,1)(6,2) sorted (port asc, issue_sequence asc) / 400 batch
    // (2,1)(2,2) / 500 (1,2).
    expect_order(g_recorder,
                 {{4, 2, 100},
                  {3, 1, 150},
                  {1, 1, 200},
                  {4, 1, 200},
                  {3, 2, 250},
                  {5, 1, 267},
                  {0, 1, 300},
                  {0, 2, 300},
                  {6, 1, 300},
                  {6, 2, 300},
                  {2, 1, 400},
                  {2, 2, 400},
                  {1, 2, 500}},
                 tag + ": full global delivery order");

    // -- RefPort oracle cross-check: recompute every rank's callbacks from
    // the sec.3.1 reference fluid model (the oracle itself asserts the
    // equal-split / capacity / clamp invariants for every service segment)
    // and require agreement with the hand-written anchors above.
    {
        const auto r0 = ref_port(6.0, 100.0, {{0, 600}, {0, 600}});
        expect(r0 == std::vector<Tick>({300, 300}),
               tag + ": RefPort r0 == anchors 300/300");
        const auto r1 = ref_port(6.0, 100.0, {{0, 300}, {0, 2100}});
        expect(r1 == std::vector<Tick>({200, 500}),
               tag + ": RefPort r1 == anchors 200/500");
        const auto r2 = ref_port(6.0, 100.0, {{0, 1200}, {100, 600}});
        expect(r2 == std::vector<Tick>({400, 400}),
               tag + ": RefPort r2 == anchors 400/400 (staggered)");
        const auto r3 = ref_port(6.0, 100.0, {{0, 300}, {50, 600}});
        expect(r3 == std::vector<Tick>({150, 250}),
               tag + ": RefPort r3 == anchors 150/250 (ready-at-completion)");
        const auto r4 = ref_port(6.0, 100.0, {{0, 600}, {0, 0}});
        expect(r4 == std::vector<Tick>({200, 100}),
               tag + ": RefPort r4 == anchors 200/100 (zero-byte at ready)");
        const auto r5 = ref_port(6.0, 100.0, {{0, 1000}});
        expect(r5 == std::vector<Tick>({267}),
               tag + ": RefPort r5 == anchor ceil(1000/6)+100 = 267");
        const auto r6 = ref_port(6.0, 100.0, {{0, 600}, {0, 600}});
        expect(r6 == std::vector<Tick>({300, 300}),
               tag + ": RefPort r6 == anchors 300/300");
    }

    // -- per-port PortStats (event-interval integrals + peaks).
    struct PortExpect {
        std::size_t port;
        uint64_t count;
        uint64_t bytes;
        uint64_t peak_streaming;
        uint64_t peak_in_flight;
        double busy;
        double shared;
        uint64_t redist;
        uint64_t arrival;
    };
    const std::vector<PortExpect> px = {
        // r0: two streams [100,300) together; simultaneous finish, no
        // survivor -> no redistribution either way.
        {0, 2, 1200, 2, 2, 200.0, 200.0, 0, 0},
        // r1: [100,200) N=2 + [200,500] N=1; completion-driven re-split +1.
        {1, 2, 2400, 2, 2, 400.0, 100.0, 1, 0},
        // r2: [100,200) N=1 + [200,400] N=2; arrival-driven +1, no
        // completion instant leaves a survivor.
        {2, 2, 1800, 2, 2, 300.0, 200.0, 0, 1},
        // r3: [100,150) N=1 + [150,250] N=1 (same-instant handoff: A's
        // exhaustion instant IS B's admission instant, so the two streams
        // are never simultaneously active -> peak_streaming 1 / shared 0,
        // while peak_in_flight still covers both undelivered nodes).
        {3, 2, 900, 1, 2, 150.0, 0.0, 0, 0},
        // r4: zero-byte never streams; single 600B stream [100,200].
        {4, 2, 600, 1, 2, 100.0, 0.0, 0, 0},
        // r5: single 1000B stream, service 1000/6 ns.
        {5, 1, 1000, 1, 1, 1000.0 / 6.0, 0.0, 0, 0},
        // r6: mirror of r0.
        {6, 2, 1200, 2, 2, 200.0, 200.0, 0, 0},
    };
    for (const auto& e : px) {
        const auto st = mem->get_port_stats(e.port);
        const std::string pfx = tag + " port " + std::to_string(e.port);
        expect(st.issued_count == e.count,
               pfx + " issued_count " + std::to_string(st.issued_count));
        expect(st.issued_bytes == e.bytes,
               pfx + " issued_bytes " + std::to_string(st.issued_bytes));
        expect(st.peak_streaming == e.peak_streaming,
               pfx + " peak_streaming " +
                   std::to_string(st.peak_streaming));
        expect(st.peak_in_flight == e.peak_in_flight,
               pfx + " peak_in_flight " +
                   std::to_string(st.peak_in_flight));
        expect_near(st.port_busy_ns, e.busy, 1e-9, pfx + " port_busy_ns");
        expect_near(st.shared_busy_ns, e.shared, 1e-9,
                    pfx + " shared_busy_ns");
        expect(st.redistribution_events == e.redist,
               pfx + " redistribution_events " +
                   std::to_string(st.redistribution_events));
        expect(st.arrival_redistribution_events == e.arrival,
               pfx + " arrival_redistribution_events " +
                   std::to_string(st.arrival_redistribution_events));
        expect_port_conservation(*mem, e.port, 6.0, tag);
    }

    // -- sensing stays off: no per-transaction rows, no lazy file.
    expect(mem->transaction_rows_written() == 0,
           tag + ": sensing off -> zero transaction rows");
    expect(access((dir + "/txn_" + tag + ".jsonl").c_str(), F_OK) != 0,
           tag + ": sensing off -> no detail file created");

    expect(mem->is_drained(), tag + ": backend fully drained");
    teardown_phase(stack);
    std::printf("[remote_port_nway_test] phase %s: ALL PASS\n", tag.c_str());
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Phase P2: Anchor B -- bw=100 B/ns, latency=50 ns, one PER_NPU port, four
// same-tick work-conserving streams 1k/2k/3k/4kB (kB = 1000 B). Segment
// table in the file header; anchors fluid/callback 90/120/140/150 ns.
// ---------------------------------------------------------------------------
void phase_anchor_four_streams(const std::string& dir) {
    const std::string tag = "anchor4";
    const std::string cfg_path = dir + "/remote_" + tag + ".json";
    write_file(cfg_path,
               "{\n"
               "  \"memory-type\": \"PER_NPU_MEMORY_EXPANSION\",\n"
               "  \"npu-ids\": [0],\n"
               "  \"remote-mem-bw\": 100,\n"
               "  \"remote-mem-latency\": 50\n"
               "}\n");

    PhaseStack stack;
    bootstrap_phase(dir, tag, 1, cfg_path, stack);
    auto* mem = stack.memory.get();

    const uint64_t kB = 1000;
    add_and_issue(stack.systems[0], stack.sources[0].get(),
                  make_mem_node(0, 1 * kB, "b_1k"));
    add_and_issue(stack.systems[0], stack.sources[0].get(),
                  make_mem_node(0, 2 * kB, "b_2k"));
    add_and_issue(stack.systems[0], stack.sources[0].get(),
                  make_mem_node(0, 3 * kB, "b_3k"));
    add_and_issue(stack.systems[0], stack.sources[0].get(),
                  make_mem_node(0, 4 * kB, "b_4k"));

    run_event_loop(*stack.event_queue, tag, *mem, 1);

    expect_terminal(g_recorder, 0, 1, 90, tag + ": 1kB cb 90");
    expect_terminal(g_recorder, 0, 2, 120, tag + ": 2kB cb 120");
    expect_terminal(g_recorder, 0, 3, 140, tag + ": 3kB cb 140");
    expect_terminal(g_recorder, 0, 4, 150, tag + ": 4kB cb 150");
    expect_order(g_recorder,
                 {{0, 1, 90}, {0, 2, 120}, {0, 3, 140}, {0, 4, 150}},
                 tag + ": completion batch order (issue sequence)");

    // -- RefPort oracle cross-check (Anchor B): work-conserving re-split
    // 90/120/140/150 recomputed per segment; the oracle asserts the
    // equal-split invariant for [50,90)/[90,120)/[120,140)/[140,150].
    {
        const auto ref = ref_port(100.0, 50.0,
                                  {{0, 1000}, {0, 2000}, {0, 3000}, {0, 4000}});
        expect(ref == std::vector<Tick>({90, 120, 140, 150}),
               tag + ": RefPort == anchors 90/120/140/150");
    }

    const auto st = mem->get_port_stats(0);
    expect(st.issued_count == 4 && st.issued_bytes == 10000,
           tag + ": totals 4 / 10000");
    expect(st.peak_streaming == 4, tag + ": peak_streaming 4");
    expect(st.peak_in_flight == 4, tag + ": peak_in_flight 4");
    // Integrals: busy [50,150] = 100; shared = [50,90)+[90,120)+[120,140)
    // = 40+30+20 = 90 (the final 10 ns run N=1).
    expect_near(st.port_busy_ns, 100.0, 1e-9, tag + ": port_busy 100");
    expect_near(st.shared_busy_ns, 90.0, 1e-9, tag + ": shared_busy 90");
    // Completion instants 90/120/140 each leave survivors -> 3; the four
    // arrivals share one idle-port instant -> arrival 0.
    expect(st.redistribution_events == 3, tag + ": redistribution 3");
    expect(st.arrival_redistribution_events == 0, tag + ": arrival 0");
    expect_port_conservation(*mem, 0, 100.0, tag);
    expect(mem->is_drained(), tag + ": backend fully drained");
    teardown_phase(stack);
    std::printf("[remote_port_nway_test] phase %s: ALL PASS\n", tag.c_str());
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Phase Q: PER_NODE mapping -- num-nodes 3, num-npus-per-node 2, bw=6,
// latency=100, 6 ranks. port_index = sys_id / 2: ranks {0,1} share port 0,
// {2,3} share port 1, {4,5} use port 2.
// ---------------------------------------------------------------------------
void phase_per_node_mapping(const std::string& dir) {
    const std::string tag = "pernode";
    const std::string cfg_path = dir + "/remote_" + tag + ".json";
    write_file(cfg_path,
               "{\n"
               "  \"memory-type\": \"PER_NODE_MEMORY_EXPANSION\",\n"
               "  \"num-nodes\": 3,\n"
               "  \"num-npus-per-node\": 2,\n"
               "  \"remote-mem-bw\": 6,\n"
               "  \"remote-mem-latency\": 100\n"
               "}\n");

    PhaseStack stack;
    bootstrap_phase(dir, tag, 6, cfg_path, stack);
    auto* mem = stack.memory.get();

    // Port 0: ranks 0+1 issue 600B same-tick -> N=2 rate 3 -> both cb 300.
    add_and_issue(stack.systems[0], stack.sources[0].get(),
                  make_mem_node(0, 600, "pn_r0"));
    add_and_issue(stack.systems[1], stack.sources[1].get(),
                  make_mem_node(1, 600, "pn_r1"));

    // Port 1: rank 2 issues at t0 -> [100,200) alone at 6 -> cb 200;
    // rank 3 issues at Tick 100, ready 200: admitted at the exact instant
    // rank 2's stream exhausts (never two streams at once on port 1) ->
    // [200,300] alone at 6 -> cb 300.
    add_and_issue(stack.systems[2], stack.sources[2].get(),
                  make_mem_node(2, 600, "pn_r2"));
    schedule_issue_at(stack.systems[3], stack.sources[3].get(),
                      make_mem_node(3, 600, "pn_r3"), 100);

    // Port 2: rank 4 single 1200B -> 1200/6 = 200 -> cb 300.
    add_and_issue(stack.systems[4], stack.sources[4].get(),
                  make_mem_node(4, 1200, "pn_r4"));

    run_event_loop(*stack.event_queue, tag, *mem, 3);

    // -- mapping: exactly 3 node-level ports.
    expect(mem->get_port_count() == 3, tag + ": PER_NODE port count 3");
    const auto s0 = mem->get_port_stats(0);
    const auto s1 = mem->get_port_stats(1);
    const auto s2 = mem->get_port_stats(2);
    expect(s0.issued_count == 2 && s0.issued_bytes == 1200,
           tag + ": port0 aggregated both ranks");
    expect(s1.issued_count == 2 && s1.issued_bytes == 1200,
           tag + ": port1 aggregated ranks 2+3");
    expect(s2.issued_count == 1 && s2.issued_bytes == 1200,
           tag + ": port2 rank 4 only");
    expect(s0.peak_streaming == 2, tag + ": port0 peak 2 (shared node)");
    expect(s1.peak_streaming == 1,
           tag + ": port1 peak 1 (same-instant handoff)");
    expect_near(s1.shared_busy_ns, 0.0, 1e-9,
                tag + ": port1 never two streams");
    expect(s1.redistribution_events == 0 &&
               s1.arrival_redistribution_events == 0,
           tag + ": port1 no share changes (handoff at exhaustion)");
    expect_near(s0.port_busy_ns, 200.0, 1e-9, tag + ": port0 busy 200");
    expect_near(s1.port_busy_ns, 200.0, 1e-9, tag + ": port1 busy 200");
    expect_near(s2.port_busy_ns, 200.0, 1e-9, tag + ": port2 busy 200");

    // Callbacks + the Tick-300 batch: port0 seq0/seq1 then port1 seq1
    // (rank 3) then port2 seq0, sorted by (port_index, issue_sequence)
    // across THREE ports in one global event.
    expect_terminal(g_recorder, 0, 1, 300, tag + ": r0");
    expect_terminal(g_recorder, 1, 1, 300, tag + ": r1");
    expect_terminal(g_recorder, 2, 1, 200, tag + ": r2");
    expect_terminal(g_recorder, 3, 1, 300, tag + ": r3");
    expect_terminal(g_recorder, 4, 1, 300, tag + ": r4");
    expect_order(g_recorder,
                 {{2, 1, 200},
                  {0, 1, 300},
                  {1, 1, 300},
                  {3, 1, 300},
                  {4, 1, 300}},
                 tag + ": delivery order (Tick-200 batch then Tick-300 "
                       "3-port batch)");

    // -- RefPort oracle cross-check: node-shared ports recomputed per port
    // (port0 same-tick pair 300/300; port1 same-instant handoff 200/300;
    // port2 single 1200B at full rate 300).
    {
        const auto p0 = ref_port(6.0, 100.0, {{0, 600}, {0, 600}});
        expect(p0 == std::vector<Tick>({300, 300}),
               tag + ": RefPort port0 == 300/300");
        const auto p1 = ref_port(6.0, 100.0, {{0, 600}, {100, 600}});
        expect(p1 == std::vector<Tick>({200, 300}),
               tag + ": RefPort port1 == 200/300 (handoff)");
        const auto p2 = ref_port(6.0, 100.0, {{0, 1200}});
        expect(p2 == std::vector<Tick>({300}),
               tag + ": RefPort port2 == 300");
    }
    expect_port_conservation(*mem, 0, 6.0, tag);
    expect_port_conservation(*mem, 1, 6.0, tag);
    expect_port_conservation(*mem, 2, 6.0, tag);
    expect(mem->is_drained(), tag + ": backend fully drained");
    teardown_phase(stack);
    std::printf("[remote_port_nway_test] phase %s: ALL PASS\n", tag.c_str());
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Phase R: MEMORY_POOL mapping -- every rank is funneled into ONE logical
// port. bw=6, latency=100, 3 ranks.
// ---------------------------------------------------------------------------
void phase_memory_pool_mapping(const std::string& dir) {
    const std::string tag = "pool";
    const std::string cfg_path = dir + "/remote_" + tag + ".json";
    write_file(cfg_path,
               "{\n"
               "  \"memory-type\": \"MEMORY_POOL\",\n"
               "  \"remote-mem-bw\": 6,\n"
               "  \"remote-mem-latency\": 100\n"
               "}\n");

    PhaseStack stack;
    bootstrap_phase(dir, tag, 3, cfg_path, stack);
    auto* mem = stack.memory.get();

    // All three transactions share port 0:
    // [100,200) N=2 rate 3 (r0,r1 serve 300 each, rem 300 each);
    // r2 issued at Tick 100 becomes ready at 200 and joins ->
    // [200,350) N=3 rate 2 (r0/r1 serve their last 300 -> fluid 350.0;
    // r2 serves 300, rem 300);
    // [350,400] N=1 rate 6: completion-driven survivor acceleration --
    // r2 re-splits to full rate and serves its last 300 -> fluid 400.0.
    add_and_issue(stack.systems[0], stack.sources[0].get(),
                  make_mem_node(0, 600, "mp_r0"));
    add_and_issue(stack.systems[1], stack.sources[1].get(),
                  make_mem_node(1, 600, "mp_r1"));
    schedule_issue_at(stack.systems[2], stack.sources[2].get(),
                      make_mem_node(2, 600, "mp_r2"), 100);

    run_event_loop(*stack.event_queue, tag, *mem, 1);

    expect(mem->get_port_count() == 1, tag + ": MEMORY_POOL single port");
    const auto st = mem->get_port_stats(0);
    expect(st.issued_count == 3 && st.issued_bytes == 1800,
           tag + ": all ranks land on port 0");
    expect(st.peak_streaming == 3, tag + ": peak_streaming 3");
    expect_near(st.port_busy_ns, 300.0, 1e-9,
                tag + ": busy [100,400) = 300 (work-conserving: "
                      "1800 B / 6 B/ns)");
    expect_near(st.shared_busy_ns, 250.0, 1e-9,
                tag + ": shared [100,350) = 250 (pair window [100,200) "
                      "+ triple window [200,350))");
    expect(st.arrival_redistribution_events == 1,
           tag + ": r2 arrival re-split +1");
    expect(st.redistribution_events == 1,
           tag + ": r0/r1 simultaneous finish leaves r2 survivor +1");
    expect_terminal(g_recorder, 0, 1, 350, tag + ": r0");
    expect_terminal(g_recorder, 1, 1, 350, tag + ": r1");
    expect_terminal(g_recorder, 2, 1, 400, tag + ": r2 (survivor "
                                                   "acceleration)");
    expect_order(g_recorder,
                 {{0, 1, 350}, {1, 1, 350}, {2, 1, 400}},
                 tag + ": delivery order");

    // -- RefPort oracle cross-check: pool funneling recomputed as one port
    // ([100,200) N=2 rate 3, [200,350) N=3 rate 2, [350,400] survivor at 6).
    {
        const auto ref =
            ref_port(6.0, 100.0, {{0, 600}, {0, 600}, {100, 600}});
        expect(ref == std::vector<Tick>({350, 350, 400}),
               tag + ": RefPort == anchors 350/350/400");
    }
    expect_port_conservation(*mem, 0, 6.0, tag);
    expect(mem->is_drained(), tag + ": backend fully drained");
    teardown_phase(stack);
    std::printf("[remote_port_nway_test] phase %s: ALL PASS\n", tag.c_str());
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Phase T1: special transactions and sub-Tick residual service at bw=6,
// latency=0, PER_NPU [0..4], WITH the sensing-gated transaction detail
// stream armed so the fluid_finish_ns vs callback_tick distinction is
// asserted from the streamed rows (rows are written before the wlhd
// callback destroys the observation keys).
// ---------------------------------------------------------------------------
void phase_zero_latency_specials(const std::string& dir) {
    const std::string tag = "zerolat";
    const std::string cfg_path = dir + "/remote_" + tag + ".json";
    const std::string jsonl_path = dir + "/txn_" + tag + ".jsonl";
    write_file(cfg_path,
               "{\n"
               "  \"memory-type\": \"PER_NPU_MEMORY_EXPANSION\",\n"
               "  \"npu-ids\": [0, 1, 2, 3, 4],\n"
               "  \"remote-mem-bw\": 6,\n"
               "  \"remote-mem-latency\": 0\n"
               "}\n");

    PhaseStack stack;
    bootstrap_phase(dir, tag, 5, cfg_path, stack);
    auto* mem = stack.memory.get();
    mem->configure_transaction_detail(true, jsonl_path,
                                      "remote_port_nway_test");

    // Rank 0 -- Anchor C: single 1B stream: fluid 1/6 ns (sub-Tick),
    // callback on the NEXT integer Tick 1.
    add_and_issue(stack.systems[0], stack.sources[0].get(),
                  make_mem_node(0, 1, "c_1b"));

    // Rank 1 -- dual-zero (bytes=0, latency=0): an independent one-shot
    // timer at issue+1 ns; never delivered synchronously from the issue
    // stack and never a bandwidth job.
    add_and_issue(stack.systems[1], stack.sources[1].get(),
                  make_mem_node(1, 0, "c_dualzero"));
    {
        const auto jobs = mem->get_port_jobs(1);
        expect(jobs.size() == 1,
               "dual-zero: still pending right after issue");
        expect(g_recorder.order.empty(),
               "dual-zero: no synchronous delivery on the issue stack");
        if (jobs.size() == 1) {
            expect(jobs[0].state ==
                       AnalyticalRemoteMemory::PortJobState::DualZeroTimer,
                   "dual-zero: state DualZeroTimer");
            expect_near(jobs[0].ready_ns, 1.0, 1e-12,
                        "dual-zero: fires exactly at issue+1 ns");
        }
        expect(mem->get_port_stats(1).dual_zero_timer_count == 1,
               "dual-zero: timer counted");
    }

    // Rank 2 -- sub-ns completion with immediate survivor re-split inside
    // one integer Tick: A=1B and B=6B stream at 3 B/ns; A exhausts at 1/3 ns
    // (callback Tick 1), B re-splits to 6 B/ns in the continuous substep
    // (rem 6-1=5 -> 5/6 ns -> fluid 7/6, callback Tick 2).
    add_and_issue(stack.systems[2], stack.sources[2].get(),
                  make_mem_node(2, 1, "c_resA"));
    add_and_issue(stack.systems[2], stack.sources[2].get(),
                  make_mem_node(2, 6, "c_resB"));

    // Rank 3 -- distinct continuous completion instants inside ONE callback
    // Tick: P=401B Q=402B R=404B at rate 2 each:
    //   P exhausts 401/2 = 200.5      (cb 201; survivors Q,R -> redist +1)
    //   Q rem 1 at rate 3 -> +1/3 ns  (cb 201; survivor R   -> redist +1;
    //                                  both instants inside Tick 201)
    //   R rem 2 at rate 6 -> +1/3 ns  (fluid 201.166.., cb 202)
    add_and_issue(stack.systems[3], stack.sources[3].get(),
                  make_mem_node(3, 401, "c_tickP"));
    add_and_issue(stack.systems[3], stack.sources[3].get(),
                  make_mem_node(3, 402, "c_tickQ"));
    add_and_issue(stack.systems[3], stack.sources[3].get(),
                  make_mem_node(3, 404, "c_tickR"));

    // Rank 4 -- single non-divisible 1000B at 6 B/ns: fluid 166.666..,
    // callback ceil = 167.
    add_and_issue(stack.systems[4], stack.sources[4].get(),
                  make_mem_node(4, 1000, "c_nondiv"));

    run_event_loop(*stack.event_queue, tag, *mem, 5);

    expect_terminal(g_recorder, 0, 1, 1, tag + ": 1B cb Tick 1");
    expect_terminal(g_recorder, 1, 1, 1, tag + ": dual-zero cb issue+1ns");
    expect_terminal(g_recorder, 2, 1, 1, tag + ": residual A cb 1");
    expect_terminal(g_recorder, 2, 2, 2, tag + ": residual B cb 2");
    expect_terminal(g_recorder, 3, 1, 201, tag + ": P cb 201");
    expect_terminal(g_recorder, 3, 2, 201, tag + ": Q cb 201");
    expect_terminal(g_recorder, 3, 3, 202, tag + ": R cb 202");
    expect_terminal(g_recorder, 4, 1, 167, tag + ": non-divisible cb 167");
    // Full global delivery order: Tick 1 batch (0,1)(1,1)(2,1) across
    // ports 0..2, Tick 2 (2,2), Tick 167 (4,1), Tick 201 batch (3,1)(3,2)
    // [two distinct continuous instants, one callback Tick], Tick 202 (3,3).
    expect_order(g_recorder,
                 {{0, 1, 1},
                  {1, 1, 1},
                  {2, 1, 1},
                  {2, 2, 2},
                  {4, 1, 167},
                  {3, 1, 201},
                  {3, 2, 201},
                  {3, 3, 202}},
                 tag + ": global delivery order");

    // -- RefPort oracle cross-check with latency=0: dual-zero async +1 ns,
    // sub-Tick residuals and the same-Tick distinct-instant triple.
    {
        const auto r0 = ref_port(6.0, 0.0, {{0, 1}});
        expect(r0 == std::vector<Tick>({1}),
               tag + ": RefPort r0 1B -> next-Tick callback 1");
        const auto r1 = ref_port(6.0, 0.0, {{0, 0}});
        expect(r1 == std::vector<Tick>({1}),
               tag + ": RefPort r1 dual-zero -> async issue+1");
        const auto r2 = ref_port(6.0, 0.0, {{0, 1}, {0, 6}});
        expect(r2 == std::vector<Tick>({1, 2}),
               tag + ": RefPort r2 residuals -> 1/2");
        const auto r3 = ref_port(6.0, 0.0, {{0, 401}, {0, 402}, {0, 404}});
        expect(r3 == std::vector<Tick>({201, 201, 202}),
               tag + ": RefPort r3 triple -> 201/201/202");
        const auto r4 = ref_port(6.0, 0.0, {{0, 1000}});
        expect(r4 == std::vector<Tick>({167}),
               tag + ": RefPort r4 non-divisible -> 167");
    }

    // -- per-port integrals.
    {
        const auto s2 = mem->get_port_stats(2);
        expect_near(s2.port_busy_ns, 7.0 / 6.0, 1e-9,
                    tag + ": r2 busy [0,1/3)+[1/3,1)+[1,7/6) = 7/6");
        expect_near(s2.shared_busy_ns, 1.0 / 3.0, 1e-9,
                    tag + ": r2 shared [0,1/3)");
        expect(s2.redistribution_events == 1 &&
                   s2.arrival_redistribution_events == 1,
               tag + ": r2 redist 1 (A done, B survivor) + arrival 1");
        expect_near(s2.bytes_served, 7.0, 1e-6, tag + ": r2 served 7");
    }
    {
        const auto s3 = mem->get_port_stats(3);
        expect_near(s3.port_busy_ns, 201.0 + 1.0 / 6.0, 1e-9,
                    tag + ": r3 busy = 201.166..");
        expect_near(s3.shared_busy_ns, 200.5 + 1.0 / 3.0, 1e-9,
                    tag + ": r3 shared = 200.833..");
        expect(s3.redistribution_events == 2,
               tag + ": r3 two continuous instants, both counted (same "
                     "callback Tick)");
        expect(s3.arrival_redistribution_events == 2,
               tag + ": r3 latency-0 issue-time arrivals +2");
        expect(s3.peak_streaming == 3, tag + ": r3 peak 3");
    }
    {
        const auto s0 = mem->get_port_stats(0);
        expect_near(s0.port_busy_ns, 1.0 / 6.0, 1e-9,
                    tag + ": r0 busy = 1/6 ns of real service");
    }
    {
        const auto s1 = mem->get_port_stats(1);
        expect_near(s1.port_busy_ns, 0.0, 1e-12,
                    tag + ": dual-zero never occupies bandwidth");
    }
    for (std::size_t p = 0; p < 5; p++) {
        expect_port_conservation(*mem, p, 6.0, tag);
    }

    // -- sensing artifact: rows streamed before teardown closes the file.
    expect(mem->transaction_rows_written() == 8,
           tag + ": exactly 8 transaction rows, got " +
               std::to_string(mem->transaction_rows_written()));
    expect(mem->is_drained(), tag + ": backend fully drained");
    teardown_phase(stack);

    // -- verify the two time notions from the streamed rows.
    std::ifstream in(jsonl_path);
    expect(in.good(), tag + ": detail file exists");
    if (in.good()) {
        std::map<std::pair<int, uint64_t>, nlohmann::json> rows;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) {
                continue;
            }
            const auto row = nlohmann::json::parse(line);
            rows[{row["rank"].get<int>(),
                  row["node_id"].get<uint64_t>()}] = row;
        }
        expect(rows.size() == 8, tag + ": 8 JSONL rows");
        const auto row_of = [&](int rank, uint64_t node) -> const json* {
            auto it = rows.find({rank, node});
            return it == rows.end() ? nullptr : &it->second;
        };
        const auto expect_row = [&](int rank, uint64_t node, double fluid,
                                    double fluid_tol, int64_t cb,
                                    const std::string& what) {
            const json* row = row_of(rank, node);
            expect(row != nullptr, tag + ": missing row " + what);
            if (row == nullptr) {
                return;
            }
            expect((*row)["run_id"].get<std::string>() ==
                       "remote_port_nway_test",
                   tag + ": row run_id " + what);
            expect((*row)["callback_tick"].get<int64_t>() == cb,
                   tag + ": row callback_tick " + what + " got " +
                       std::to_string(
                           (*row)["callback_tick"].get<int64_t>()));
            expect_near((*row)["fluid_finish_ns"].get<double>(), fluid,
                        fluid_tol, tag + ": row fluid_finish_ns " + what);
        };
        // Anchor C: fluid 1/6 ns, callback Tick 1 -- distinct time notions.
        expect_row(0, 1, 1.0 / 6.0, 1e-9, 1, "1B anchor");
        // Dual-zero: fluid 1.0 (the +1 ns fire instant), callback Tick 1.
        expect_row(1, 1, 1.0, 1e-12, 1, "dual-zero");
        // Residual pair: 1/3 ns and 7/6 ns.
        expect_row(2, 1, 1.0 / 3.0, 1e-9, 1, "residual A");
        expect_row(2, 2, 7.0 / 6.0, 1e-9, 2, "residual B");
        // Same-Tick distinct instants: 200.5 and 200.833.. both cb 201;
        // 201.166.. -> cb 202.
        expect_row(3, 1, 200.5, 1e-9, 201, "P");
        expect_row(3, 2, 200.5 + 1.0 / 3.0, 1e-9, 201, "Q");
        expect_row(3, 3, 200.5 + 2.0 / 3.0, 1e-9, 202, "R");
        // Non-divisible: 1000/6.
        expect_row(4, 1, 1000.0 / 6.0, 1e-6, 167, "non-divisible");
        // stream_start discipline: latency-0 rows start at 0; the dual-zero
        // row never streams.
        const json* dz = row_of(1, 1);
        if (dz != nullptr) {
            expect_near((*dz)["stream_start_ns"].get<double>(), -1.0, 1e-12,
                        tag + ": dual-zero never streams");
        }
    }
    std::printf("[remote_port_nway_test] phase %s: ALL PASS\n", tag.c_str());
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Phase U: awaiting-callback exclusion anchor (merged from the passing
// standalone variant's S6B). MEMORY_POOL, bw=6, latency=100, three ranks
// same-tick 100B / 1B / 200B on the ONE logical port:
//   [100,100.5) N=3 rate 2: 1B exhausts at 100.5            -> cb 101
//   [100.5,133.5) N=2 rate 3: 100B (rem 99) exhausts 133.5  -> cb 134
//   [133.5,150.166..) N=1 rate 6: 200B (rem 100) -> 150.17  -> cb 151
// The 1B stream is FLUID-complete at 100.5 but its callback lands on Tick
// 101; the denominator drops it at the exhaustion INSTANT, not at the
// callback Tick. Exclusion anchor: if a fluid-complete-awaiting-callback
// stream still counted into N, 100B would need 99/2 = 49.5 ns -> cb 150,
// never 134 (and the backend's own drained-state transition ordering would
// differ). This pins plan sec.7 "new issues must not count finished,
// callback-pending streams back into N" from the completed-stream side.
// ---------------------------------------------------------------------------
void phase_await_callback_denominator(const std::string& dir) {
    const std::string tag = "awaitcb";
    const std::string cfg_path = dir + "/remote_" + tag + ".json";
    write_file(cfg_path,
               "{\n"
               "  \"memory-type\": \"MEMORY_POOL\",\n"
               "  \"remote-mem-bw\": 6,\n"
               "  \"remote-mem-latency\": 100\n"
               "}\n");

    PhaseStack stack;
    bootstrap_phase(dir, tag, 3, cfg_path, stack);
    auto* mem = stack.memory.get();

    add_and_issue(stack.systems[0], stack.sources[0].get(),
                  make_mem_node(0, 100, "u_100b"));
    add_and_issue(stack.systems[1], stack.sources[1].get(),
                  make_mem_node(1, 1, "u_1b"));
    add_and_issue(stack.systems[2], stack.sources[2].get(),
                  make_mem_node(2, 200, "u_200b"));

    run_event_loop(*stack.event_queue, tag, *mem, 1);

    expect(mem->get_port_count() == 1, tag + ": single pooled port");
    expect_terminal(g_recorder, 1, 1, 101, tag + ": 1B cb 101");
    expect_terminal(g_recorder, 0, 1, 134, tag + ": 100B cb 134 "
                                               "(NOT 150 -- awaiting-callback "
                                               "1B left the denominator)");
    expect_terminal(g_recorder, 2, 1, 151, tag + ": 200B cb 151");
    expect_order(g_recorder,
                 {{1, 1, 101}, {0, 1, 134}, {2, 1, 151}},
                 tag + ": delivery order");

    // -- RefPort oracle cross-check.
    {
        const auto ref = ref_port(6.0, 100.0, {{0, 100}, {0, 1}, {0, 200}});
        expect(ref == std::vector<Tick>({134, 101, 151}),
               tag + ": RefPort == anchors 134/101/151");
    }

    // -- port integrals: busy = (100+1+200)/6 = 50.166.. (work-conserving);
    // shared = [100,100.5) N=3 + [100.5,133.5) N=2 = 33.5; completion
    // instants 100.5 and 133.5 both leave survivors -> redistribution 2;
    // all three arrivals hit an idle port together -> arrival 0.
    const auto st = mem->get_port_stats(0);
    expect(st.peak_streaming == 3, tag + ": peak_streaming 3");
    expect_near(st.port_busy_ns, 301.0 / 6.0, 1e-9,
                tag + ": busy 301/6 (work-conserving)");
    expect_near(st.shared_busy_ns, 33.5, 1e-9, tag + ": shared 33.5");
    expect(st.redistribution_events == 2,
           tag + ": two survivor completion instants");
    expect(st.arrival_redistribution_events == 0,
           tag + ": idle-port same-tick arrivals -> 0");
    expect_port_conservation(*mem, 0, 6.0, tag);
    expect(mem->is_drained(), tag + ": backend fully drained");
    teardown_phase(stack);
    std::printf("[remote_port_nway_test] phase %s: ALL PASS\n", tag.c_str());
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Phase V: tiny-residue clamp (merged from the passing standalone variant's
// S6A residue case). PER_NPU single rank, bw=6, latency=100, ONE 10B
// transaction: the final substep share (elapsed * bw) carries a ~2e-15
// floating-point overshoot past the 10B remainder, which must land inside
// the backend's 1e-6 completion clamp -- the transaction completes at
// ceil(100 + 10/6) = 102 with no panic, no postponement and no lost service
// (bytes_served conservation closes at the same tolerance).
// ---------------------------------------------------------------------------
void phase_tiny_residue_clamp(const std::string& dir) {
    const std::string tag = "tinyres";
    const std::string cfg_path = dir + "/remote_" + tag + ".json";
    write_file(cfg_path,
               "{\n"
               "  \"memory-type\": \"PER_NPU_MEMORY_EXPANSION\",\n"
               "  \"npu-ids\": [0],\n"
               "  \"remote-mem-bw\": 6,\n"
               "  \"remote-mem-latency\": 100\n"
               "}\n");

    PhaseStack stack;
    bootstrap_phase(dir, tag, 1, cfg_path, stack);
    auto* mem = stack.memory.get();

    add_and_issue(stack.systems[0], stack.sources[0].get(),
                  make_mem_node(0, 10, "v_10b"));

    run_event_loop(*stack.event_queue, tag, *mem, 1);

    expect_terminal(g_recorder, 0, 1, 102,
                    tag + ": 10B cb ceil(101.666..) = 102 (clamp path, no "
                           "panic)");
    // -- RefPort oracle cross-check (its own clamp assertion covers the
    // remainder overshoot).
    {
        const auto ref = ref_port(6.0, 100.0, {{0, 10}});
        expect(ref == std::vector<Tick>({102}),
               tag + ": RefPort == anchor 102");
    }
    {
        const auto st = mem->get_port_stats(0);
        expect_near(st.bytes_served, 10.0, 1e-6,
                    tag + ": bytes_served == 10 within the clamp tolerance "
                           "(no lost service)");
        expect_near(st.port_busy_ns, 10.0 / 6.0, 1e-9,
                    tag + ": busy 10/6 ns of real service");
    }
    expect_port_conservation(*mem, 0, 6.0, tag);
    expect(mem->is_drained(), tag + ": backend fully drained");
    teardown_phase(stack);
    std::printf("[remote_port_nway_test] phase %s: ALL PASS\n", tag.c_str());
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Phase W: synchronous re-entry (plan sec.3.3 positive case, stage 7
// checklist item). PER_NPU, bw=6, latency=100, ONE rank. Transaction A
// (600B) completes at Tick 200; reentry_hook (installed for this phase)
// issues follow-up B (600B, same port) SYNCHRONOUSLY from inside A's
// terminal callback. The in-spot assertions in the hook pin the sec.3.3
// dispatch rules: B is accepted while the batch dispatch is still on the
// stack, sits in LatencyWaiting (no recursive harvest of the just-issued
// job), and is deadlined at issue=200 + latency=100. After the loop:
//   A cb 200, B alone on the idle port [300,400] -> cb 400,
//   delivery order {A@200, B@400}, issued==completed==2, busy 200 ns
//   ([100,200) A + [300,400) B), no redistribution either way, conserved,
//   drained. A backend that drops the re-entrant issue or fails to re-arm
//   the global transition event never delivers B -> recorder/count FAIL
//   (or the watchdog fires).
// ---------------------------------------------------------------------------
void phase_reentry_issue(const std::string& dir) {
    const std::string tag = "reentry";
    const std::string cfg_path = dir + "/remote_" + tag + ".json";
    write_file(cfg_path,
               "{\n"
               "  \"memory-type\": \"PER_NPU_MEMORY_EXPANSION\",\n"
               "  \"npu-ids\": [0],\n"
               "  \"remote-mem-bw\": 6,\n"
               "  \"remote-mem-latency\": 100\n"
               "}\n");

    PhaseStack stack;
    bootstrap_phase(dir, tag, 1, cfg_path, stack);
    auto* mem = stack.memory.get();

    // Install the re-entry hook (plain recorder behavior + one synchronous
    // follow-up issue) and arm it for A's terminal.
    g_reentry = ReentryCtx{};
    g_reentry.armed = true;
    g_reentry.sys = stack.systems[0];
    g_reentry.source = stack.sources[0].get();
    g_reentry.mem = mem;
    g_reentry.trigger_rank = 0;
    g_reentry.trigger_node = 1;
    ExecutionDriven::CompletionObserver::instance().set_hook(reentry_hook,
                                                             &g_recorder);

    add_and_issue(stack.systems[0], stack.sources[0].get(),
                  make_mem_node(0, 600, "w_a"));

    run_event_loop(*stack.event_queue, tag, *mem, 1);

    // Restore the plain hook BEFORE any expectation can fail-run more
    // terminals (nothing else issues here, but restore unconditionally).
    ExecutionDriven::CompletionObserver::instance().set_hook(terminal_hook,
                                                             &g_recorder);

    expect(g_reentry.fired, tag + ": re-entry hook fired exactly once");
    expect(g_recorder.count(0, 2) == 1,
           tag + ": follow-up B delivered exactly once, got " +
               std::to_string(g_recorder.count(0, 2)));
    expect_terminal(g_recorder, 0, 1, 200, tag + ": A cb 200");
    expect_terminal(g_recorder, 0, 2, 400,
                    tag + ": B cb 400 (re-armed transition delivered the "
                           "re-entrant issue)");
    expect_order(g_recorder,
                 {{0, 1, 200}, {0, 2, 400}},
                 tag + ": delivery order A then B");

    const auto st = mem->get_port_stats(0);
    expect(st.issued_count == 2 && st.completed_count == 2,
           tag + ": both the original and the re-entrant issue landed "
                  "(issued " + std::to_string(st.issued_count) +
                       " completed " + std::to_string(st.completed_count) +
                       ")");
    expect(st.issued_bytes == 1200 && st.completed_bytes == 1200,
           tag + ": bytes 1200/1200");
    expect(st.peak_streaming == 1,
           tag + ": B arrives on an idle port -> never two streams "
                  "(peak_streaming 1)");
    expect(st.redistribution_events == 0 &&
               st.arrival_redistribution_events == 0,
           tag + ": no share change (idle-port arrival, empty-port "
                  "completion)");
    expect_near(st.port_busy_ns, 200.0, 1e-9,
                tag + ": busy [100,200) + [300,400) = 200");
    expect_near(st.shared_busy_ns, 0.0, 1e-9, tag + ": never shared");
    expect_port_conservation(*mem, 0, 6.0, tag);
    expect(mem->is_drained(), tag + ": backend fully drained");
    g_reentry = ReentryCtx{};
    teardown_phase(stack);
    std::printf("[remote_port_nway_test] phase %s: ALL PASS\n", tag.c_str());
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Phase X: early shutdown (plan sec.3.4 positive case, stage 7 checklist
// item "early shutdown deletes undelivered wlhds, cancels the global event
// and resets the counters; Workload-owned HBM join cookies are untouched").
// PER_NPU, bw=6, latency=100, ONE rank: A/B (600B each, same tick) run
// [100,150) N=2 then [150,375) N=3 against C -> both cb 375; C (1200B,
// issued at Tick 50, ready 150) survives with 750B and would run
// [375,500) alone at 6 -> cb 500. At Tick 400 the probe calls shutdown()
// TWICE (idempotence) while C is mid-stream (600B left).
// Asserted afterwards:
//   - C is NEVER delivered (its wlhd was deleted with the job; the global
//     transition event was cancelled -- a leaked event or wlhd would
//     deliver C at 500 and fail the count);
//   - the event loop drains to finished with no further work (cancelled
//     event leaves no alarm);
//   - the counters are RESET: all PortStats fields zero, no live jobs,
//     is_drained() true (also what lets the phase's own teardown pass the
//     destructor's verify_drained audit);
//   - A/B keep their exactly-once deliveries at 375.
// ---------------------------------------------------------------------------
void phase_early_shutdown(const std::string& dir) {
    const std::string tag = "earlyshut";
    const std::string cfg_path = dir + "/remote_" + tag + ".json";
    write_file(cfg_path,
               "{\n"
               "  \"memory-type\": \"PER_NPU_MEMORY_EXPANSION\",\n"
               "  \"npu-ids\": [0],\n"
               "  \"remote-mem-bw\": 6,\n"
               "  \"remote-mem-latency\": 100\n"
               "}\n");

    PhaseStack stack;
    bootstrap_phase(dir, tag, 1, cfg_path, stack);
    auto* mem = stack.memory.get();

    add_and_issue(stack.systems[0], stack.sources[0].get(),
                  make_mem_node(0, 600, "x_a"));
    add_and_issue(stack.systems[0], stack.sources[0].get(),
                  make_mem_node(0, 600, "x_b"));
    schedule_issue_at(stack.systems[0], stack.sources[0].get(),
                      make_mem_node(0, 1200, "x_c"), 50);

    // Pre-shutdown in-spot evidence at Tick 400: exactly C remains.
    schedule_probe_at(stack.systems[0], 400, [mem]() {
        const auto st = mem->get_port_stats(0);
        expect(st.completed_count == 2,
               "earlyshut: A/B delivered before the shutdown (got " +
                   std::to_string(st.completed_count) + ")");
        expect(st.in_flight_count == 1,
               "earlyshut: C still in flight at Tick 400");
        expect(st.streaming_count == 1,
               "earlyshut: C is an active stream at Tick 400");
        mem->shutdown();
        mem->shutdown();  // idempotence
        expect(mem->is_drained(),
               "earlyshut: is_drained() right after shutdown()");
        expect(mem->get_port_jobs(0).empty(),
               "earlyshut: no live jobs after shutdown()");
    });

    run_event_loop(*stack.event_queue, tag, *mem, 1);

    // C must never be delivered: shutdown deleted its undelivered wlhd and
    // cancelled the transition event that would have fired at 500.
    expect(g_recorder.count(0, 3) == 0,
           "earlyshut: C never delivered after shutdown (got " +
               std::to_string(g_recorder.count(0, 3)) + " deliveries)");
    expect(g_recorder.order.size() == 2,
           "earlyshut: exactly two terminals (A,B), got " +
               std::to_string(g_recorder.order.size()));
    expect_terminal(g_recorder, 0, 1, 375, "earlyshut: A cb 375");
    expect_terminal(g_recorder, 0, 2, 375, "earlyshut: B cb 375");

    // Counters reset to zero across the board.
    const auto st = mem->get_port_stats(0);
    expect(st.issued_count == 0 && st.completed_count == 0 &&
               st.issued_bytes == 0 && st.completed_bytes == 0,
           "earlyshut: count/bytes counters reset");
    expect(st.peak_in_flight == 0 && st.peak_streaming == 0,
           "earlyshut: peaks reset");
    expect(st.redistribution_events == 0 &&
               st.arrival_redistribution_events == 0,
           "earlyshut: redistribution counters reset");
    expect(st.port_busy_ns == 0.0 && st.shared_busy_ns == 0.0 &&
               st.bytes_served == 0.0,
           "earlyshut: integrals reset");
    expect(mem->get_port_jobs(0).empty(),
           "earlyshut: job set still empty at phase end");
    expect(mem->is_drained(),
           "earlyshut: drained at phase end (destructor's verify_drained "
           "passes on the normal teardown)");
    g_reentry = ReentryCtx{};
    teardown_phase(stack);
    std::printf("[remote_port_nway_test] phase %s: ALL PASS\n", tag.c_str());
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Phase S: stress evidence (plan stage 4) -- thousands of concurrent streams
// on ONE port, one distinct continuous completion instant per stream.
// Work-conserving equal split: the port re-splits to the survivors at every
// completion instant, so while ANY stream remains the port runs at FULL
// rate bw without a gap -- the run ends when the last byte is carried:
// port_busy_ns == total_bytes / bw (total = N*10000 + 10*N*(N-1)/2 =
// 119,980,000 B at N=4000 -> 1199.8 ns over [100, 1299.8)), and the
// completion instants follow the cumulative-byte order: stream 0 (10000 B)
// completes first at 100 + 10000/(bw/N) = 500, the heaviest stream (49990
// B) last at ceil(100 + total/bw) = 1300. Stream count is 4000 (workflow
// re-run budget; the evidence is the counters and the timing line, not the
// raw stream count).
// ---------------------------------------------------------------------------
void phase_stress(const std::string& dir) {
    const std::string tag = "stress";
    const std::string cfg_path = dir + "/remote_" + tag + ".json";
    write_file(cfg_path,
               "{\n"
               "  \"memory-type\": \"PER_NPU_MEMORY_EXPANSION\",\n"
               "  \"npu-ids\": [0],\n"
               "  \"remote-mem-bw\": 100000,\n"
               "  \"remote-mem-latency\": 100\n"
               "}\n");

    const int kStreams = 4000;
    PhaseStack stack;
    bootstrap_phase(dir, tag, 1, cfg_path, stack);
    auto* mem = stack.memory.get();
    auto* sys = stack.systems[0];
    auto* source = stack.sources[0].get();

    // Add all nodes first, then ONE issue pass at Tick 0 (per-node
    // add+issue would re-copy the growing dep-free set, O(N^2)).
    for (int i = 0; i < kStreams; i++) {
        source->store().add_node(
            make_mem_node(0, 10000 + 10 * i, "stress_" + std::to_string(i)));
    }
    issue_all_dep_free(sys, source);

    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t proceeds = run_event_loop(*stack.event_queue, tag, *mem, 1);
    const double wall_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0)
            .count();

    const auto st = mem->get_port_stats(0);
    expect(st.issued_count == static_cast<uint64_t>(kStreams),
           tag + ": all issues landed");
    expect(st.peak_streaming == static_cast<uint64_t>(kStreams),
           tag + ": peak_streaming = full stream count in flight");
    expect(st.peak_in_flight == static_cast<uint64_t>(kStreams),
           tag + ": peak_in_flight = full stream count");
    // Every completion instant up to the last one leaves a survivor -> N-1.
    expect(st.redistribution_events == static_cast<uint64_t>(kStreams) - 1,
           tag + ": one redistribution per survivor instant");
    expect(st.arrival_redistribution_events == 0,
           tag + ": single same-instant arrival wave -> 0");
    // Work-conserving anchor: the equal-split port re-splits to the
    // survivors at EVERY completion instant, so while any stream remains
    // the port runs at FULL rate bw without a gap -- the run ends when the
    // last byte is carried: port_busy_ns == total_bytes / bw exactly
    // (total = N*10000 + 10*N*(N-1)/2 = 119,980,000 B at N=4000 ->
    // 1199.8 ns over [100, 1299.8)). Individual completion instants follow
    // the cumulative-byte order: stream 0 (10000 B) completes first at
    // 100 + 10000/(bw/N) = 100 + 400 = 500, the heaviest stream last at
    // ceil(100 + total/bw) = 1300. Shared is positive (thousands of
    // multi-stream instants) and just under busy.
    const double total_bytes =
        static_cast<double>(kStreams) * 10000.0 +
        10.0 * static_cast<double>(kStreams) *
            (static_cast<double>(kStreams) - 1.0) / 2.0;
    const double busy_expect = total_bytes / 100000.0;
    expect_near(st.port_busy_ns, busy_expect, 1e-6,
                tag + ": busy == total_bytes / bw (work-conserving: the "
                       "port never idles while a stream remains)");
    expect(st.shared_busy_ns > 0.0 && st.shared_busy_ns < busy_expect,
           tag + ": shared window positive and below busy");
    // Deterministic completion lattice: stream 0 (10000 B) is the FIRST to
    // complete at 100 + 10000/(bw/N) = 500 (its 25 B/ns share carries it
    // in 400 ns); the heaviest stream (49990 B) lands last at
    // ceil(100 + total/bw) = 1300. The middle stream's exact instant
    // needs the survivor-acceleration history, so it is only
    // range-checked.
    expect_terminal(g_recorder, 0, 1, 500, tag + ": first stream (anchor: "
                                               "100 + 10000/(bw/N))");
    expect_terminal(g_recorder, 0, kStreams,
                    100 + static_cast<uint64_t>(std::ceil(busy_expect)),
                    tag + ": last stream (anchor: ceil(100 + total/bw))");
    {
        const uint64_t mid_tick =
            g_recorder.tick(0, static_cast<uint64_t>(kStreams) / 2 + 1);
        expect(g_recorder.count(0, static_cast<uint64_t>(kStreams) / 2 + 1) ==
                   1,
               tag + ": middle stream completed exactly once");
        const uint64_t last_tick =
            100 + static_cast<uint64_t>(std::ceil(busy_expect));
        expect(mid_tick > 500 && mid_tick <= last_tick,
               tag + ": middle stream tick " + std::to_string(mid_tick) +
                   " inside (500," + std::to_string(last_tick) + "]");
    }
    uint64_t distinct_ticks = 0;
    uint64_t last_tick = 0;
    for (const auto& rec : g_recorder.order) {
        const uint64_t tick = std::get<2>(rec);
        if (tick != last_tick || distinct_ticks == 0) {
            distinct_ticks++;
            last_tick = tick;
        }
    }
    expect_port_conservation(*mem, 0, 100000.0, tag);
    expect(mem->is_drained(), tag + ": backend fully drained");

    std::printf(
        "[remote_port_nway_test] stress: streams=%d issues=%llu "
        "completion_batch_ticks=%llu redistribution_events=%llu "
        "event_loop_proceeds=%llu wall_ms=%.1f peak_streaming=%llu\n",
        kStreams,
        static_cast<unsigned long long>(st.issued_count),
        static_cast<unsigned long long>(distinct_ticks),
        static_cast<unsigned long long>(st.redistribution_events),
        static_cast<unsigned long long>(proceeds), wall_ms,
        static_cast<unsigned long long>(st.peak_streaming));
    std::fflush(stdout);

    teardown_phase(stack);
    std::printf("[remote_port_nway_test] phase %s: ALL PASS\n", tag.c_str());
    std::fflush(stdout);
}

}  // namespace

// Last-resort hang detector: the proceed/wall-clock watchdogs between
// event-loop iterations cannot see a stuck SINGLE dispatch (e.g. a delayed
// re-arm loop inside one Sys bucket). This alarm turns such a hang into a
// diagnostic exit well inside the workflow budget instead of a timeout.
void on_stuck_alarm(int) {
    std::fprintf(stderr,
                 "[remote_port_nway_test] ALARM: run stuck, likely inside a "
                 "single event dispatch; aborting for diagnosis\n");
    std::_Exit(3);
}

int main() {
    std::signal(SIGALRM, on_stuck_alarm);
    alarm(600);
    char dir_template[] = "/tmp/remote_port_nway_test_XXXXXX";
    const char* dir_name = mkdtemp(dir_template);
    if (dir_name == nullptr) {
        std::perror("mkdtemp");
        return 1;
    }
    const std::string dir(dir_name);

    AstraSim::LoggerFactory::init("empty", "off");
    ExecutionDriven::CompletionObserver::instance().set_hook(terminal_hook,
                                                             &g_recorder);

    // Probe configs first: fork children must run before any phase builds
    // Sys/backend state they would inherit.
    write_file(dir + "/pf_bw_zero.json",
               "{\n"
               "  \"memory-type\": \"PER_NPU_MEMORY_EXPANSION\",\n"
               "  \"npu-ids\": [0],\n"
               "  \"remote-mem-bw\": 0,\n"
               "  \"remote-mem-latency\": 100\n"
               "}\n");
    write_file(dir + "/pf_bw_neg.json",
               "{\n"
               "  \"memory-type\": \"PER_NPU_MEMORY_EXPANSION\",\n"
               "  \"npu-ids\": [0],\n"
               "  \"remote-mem-bw\": -5,\n"
               "  \"remote-mem-latency\": 100\n"
               "}\n");
    write_file(dir + "/pf_bw_str.json",
               "{\n"
               "  \"memory-type\": \"PER_NPU_MEMORY_EXPANSION\",\n"
               "  \"npu-ids\": [0],\n"
               "  \"remote-mem-bw\": \"fast\",\n"
               "  \"remote-mem-latency\": 100\n"
               "}\n");
    write_file(dir + "/pf_lat_neg.json",
               "{\n"
               "  \"memory-type\": \"PER_NPU_MEMORY_EXPANSION\",\n"
               "  \"npu-ids\": [0],\n"
               "  \"remote-mem-bw\": 6,\n"
               "  \"remote-mem-latency\": -1\n"
               "}\n");
    write_file(dir + "/pf_lat_ovf.json",
               "{\n"
               "  \"memory-type\": \"PER_NPU_MEMORY_EXPANSION\",\n"
               "  \"npu-ids\": [0],\n"
               "  \"remote-mem-bw\": 6,\n"
               "  \"remote-mem-latency\": 1e400\n"
               "}\n");
    write_file(dir + "/pf_bw_ovf.json",
               "{\n"
               "  \"memory-type\": \"PER_NPU_MEMORY_EXPANSION\",\n"
               "  \"npu-ids\": [0],\n"
               "  \"remote-mem-bw\": 1e400,\n"
               "  \"remote-mem-latency\": 100\n"
               "}\n");
    write_file(dir + "/pf_ok.json",
               "{\n"
               "  \"memory-type\": \"PER_NPU_MEMORY_EXPANSION\",\n"
               "  \"npu-ids\": [0],\n"
               "  \"remote-mem-bw\": 6,\n"
               "  \"remote-mem-latency\": 100\n"
               "}\n");
    write_file(dir + "/pf_nomem.json",
               "{\n"
               "  \"memory-type\": \"NO_MEMORY_EXPANSION\"\n"
               "}\n");

    run_fail_closed_probes(dir);
    std::printf("[remote_port_nway_test] fail-closed probes: done\n");
    std::fflush(stdout);

    phase_pernpu_anchors(dir);
    phase_anchor_four_streams(dir);
    phase_per_node_mapping(dir);
    phase_memory_pool_mapping(dir);
    phase_zero_latency_specials(dir);
    phase_await_callback_denominator(dir);
    phase_tiny_residue_clamp(dir);
    phase_reentry_issue(dir);
    phase_early_shutdown(dir);
    phase_stress(dir);

    ExecutionDriven::CompletionObserver::instance().set_hook(nullptr,
                                                             nullptr);
    AstraSim::LoggerFactory::shutdown();

    // Workspace hygiene: remove every file this test created, then the
    // temp directory itself.
    const char* files[] = {
        "system_pernpup.json", "network_pernpup.yml",
        "system_anchor4.json", "network_anchor4.yml",
        "system_pernode.json", "network_pernode.yml",
        "system_pool.json",    "network_pool.yml",
        "system_zerolat.json", "network_zerolat.yml",
        "system_awaitcb.json", "network_awaitcb.yml",
        "system_tinyres.json", "network_tinyres.yml",
        "system_reentry.json", "network_reentry.yml",
        "system_earlyshut.json", "network_earlyshut.yml",
        "system_stress.json",  "network_stress.yml",
        "remote_pernpup.json", "remote_anchor4.json",
        "remote_pernode.json", "remote_pool.json",
        "remote_zerolat.json", "remote_awaitcb.json",
        "remote_tinyres.json", "remote_stress.json",
        "remote_reentry.json", "remote_earlyshut.json",
        "remote_pf_latovf_rt.json", "remote_pf_bwuf_rt.json",
        "remote_pf_badrank.json", "remote_pf_undrained.json",
        "txn_zerolat.jsonl",
        "pf_bw_zero.json", "pf_bw_neg.json", "pf_bw_str.json",
        "pf_lat_neg.json", "pf_lat_ovf.json", "pf_bw_ovf.json",
        "pf_ok.json", "pf_nomem.json",
    };
    for (const char* f : files) {
        std::remove((dir + "/" + f).c_str());
    }
    ::rmdir(dir.c_str());

    if (!g_ok) {
        std::fprintf(stderr, "[remote_port_nway_test] FAIL\n");
        return 1;
    }
    std::printf("[remote_port_nway_test] ALL PASS: PER_NPU/PER_NODE/"
                "MEMORY_POOL mapping, same/staggered latency, long/short "
                "re-split, same-Tick ready+completion and multi-port "
                "ordering, non-divisible/residual/dual-zero/zero-byte "
                "edges, awaiting-callback exclusion 134/101/151, tiny-"
                "residue clamp, RefPort oracle cross-check, synchronous "
                "callback re-entry (re-armed transition), early shutdown "
                "(undelivered wlhd deleted, event cancelled, counters "
                "reset, idempotent), fail-closed x12 (ctor + runtime Tick "
                "overflow + unconfigured rank + undrained destructor), "
                "conservation, 4000x10000B single-port stress\n");
    std::fflush(stdout);
    return 0;
}
