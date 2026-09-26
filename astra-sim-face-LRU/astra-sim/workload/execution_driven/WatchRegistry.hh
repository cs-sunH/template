/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

WatchRegistry -- execution-driven mechanism layer (wscllm phase 1).
Implementation (方案 §4 步骤 1-5).

Stage-completion watches over an exact member set, driven by the terminal
facts recorded by CompletionObserver (step 1-3). The wscllm boundary mapping
(步骤 1-5 操作 1) is: prefill stage all-nodes-terminal -> PREFILL_DRAIN,
decode stage all-nodes-terminal -> DECODE_COMPLETION. The registry itself is
stage-agnostic -- the stage string is opaque, and the reason mapping
(stage -> DecisionMailbox reason) is the step-1-6 wiring concern.

Semantics (all explicit, none implied):
  - CompletionKey{rank, node_id, generation} identifies one node terminal.
  - register_stage_watch freezes the exact expected member set. Duplicate
    registration of the same (request_id, stage, generation) identity
    returns the existing watch_id (register once, fire once; a double commit
    must not double-fire). An empty expected set NEVER fires -- that is a
    programming error surfaced by the run-end stale audit, not a silent
    immediate fire.
  - on_node_terminal uses a direct CompletionKey -> watch-id index (read-only
    with respect to the graph -- it never touches dependency state;
    GraphSource::finish_node remains the sole dependency release, called only
    by Workload::call). No-ops: an unindexed/non-member terminal, a duplicate
    terminal of an already-completed member, or a watch that already fired.
    Only the FIRST terminal of an exact member may enter completed_members.
  - Whether a terminal status satisfies a watch is the watch's own explicit
    policy (satisfying_statuses set at registration) -- Skipped is NEVER
    implicitly upgraded to Success. Phase-1 wscllm stage watches register
    {Success, Skipped}: a stage ends when its last node is terminal,
    regardless of which path produced the terminal fact (real traces
    contain INVALID_NODE / metadata nodes that are always skipped).
  - A watch fires ONCE, synchronously inside on_node_terminal, when every
    expected member has a satisfying terminal; the fire is queued for
    fired_and_drain() and reported to the fire notifier (step 1-6 wires the
    notifier to the DecisionMailbox push).
  - Lifecycle: register once, fire once, remove_watches_for_request on
    request end / batch abort (plus drain_fired_sentinels for the train
    sentinel watches); the run-end audit only asserts empty() &&
    stale_count() == 0 (every registered watch either fired or was
    removed). remove_all() exists for the tests; the production run end
    never calls it.

The online CompletionObserver hook (online_completion_hook below) does
EXACTLY two things (步骤 1-5 操作 3): (1) record the completion fact via
WatchRegistry::on_node_terminal and (2) write the DecisionMailbox when a
watch fires -- the notifier wiring lands in step 1-6. It NEVER calls
NodeStore::finish_node. The static binary never installs a hook (zero
overhead contract, CompletionObserver.hh).
*******************************************************************************/

#ifndef EXECUTION_DRIVEN_WATCHREGISTRY_HH
#define EXECUTION_DRIVEN_WATCHREGISTRY_HH

#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/DecisionMailbox.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"

namespace NetworkAnalytical {
class EventQueue;
}  // namespace NetworkAnalytical

namespace AstraSim {
class Workload;
namespace ExecutionDriven {

/// Identity of one completed node in the watch context.
struct CompletionKey {
    int rank = 0;
    uint64_t node_id = 0;
    uint64_t generation = 0;

    bool operator==(const CompletionKey& other) const {
        return rank == other.rank && node_id == other.node_id &&
               generation == other.generation;
    }
    bool operator!=(const CompletionKey& other) const {
        return !(*this == other);
    }
    bool operator<(const CompletionKey& other) const {
        if (rank != other.rank) {
            return rank < other.rank;
        }
        if (node_id != other.node_id) {
            return node_id < other.node_id;
        }
        return generation < other.generation;
    }
};

struct CompletionKeyHash {
    size_t operator()(const CompletionKey& key) const {
        size_t hash = std::hash<int>{}(key.rank);
        hash ^= std::hash<uint64_t>{}(key.node_id) + 0x9e3779b9 +
                (hash << 6) + (hash >> 2);
        hash ^= std::hash<uint64_t>{}(key.generation) + 0x9e3779b9 +
                (hash << 6) + (hash >> 2);
        return hash;
    }
};

/// A fired stage-completion watch (步骤 1-5 操作 2). The reason mapping
/// (stage -> PREFILL_DRAIN / DECODE_COMPLETION) is the step-1-6 mailbox
/// wiring concern, not the registry's.
struct WatchFire {
    uint64_t watch_id = 0;
    std::string request_id;
    std::string stage;
    uint64_t generation = 0;
    size_t member_count = 0;  // completed members at fire time (== expected)
    // Phase 4 (schema v1): the frozen member RANKS of the fired watch
    // (distinct ranks of expected_members, ascending). Feeds
    // StateDelta.affected_ranks (the affected-rank set of one delivery
    // epoch = union of its completed_groups' member ranks).
    std::vector<int> member_ranks;
};

/// One registered stage-completion watch. expected_members is frozen at
/// registration; only exact members' FIRST satisfying terminal may enter
/// completed_members.
struct StageWatch {
    uint64_t watch_id = 0;
    std::string request_id;
    std::string stage;
    uint64_t generation = 0;
    std::set<CompletionKey> expected_members;
    std::set<CompletionKey> completed_members;
    std::set<NodeTerminalStatus> satisfying_statuses;
    bool fired = false;
};

class WatchRegistry {
  public:
    using WatchFireNotifier = std::function<void(const WatchFire&)>;

    /// Register a stage-completion watch over an exact member set (frozen).
    /// Duplicate identity (request_id, stage, generation) returns the
    /// existing watch_id (register once / fire once). Empty expected_members
    /// never fires (programming error caught by the end audit).
    uint64_t register_stage_watch(
        const std::string& request_id, const std::string& stage,
        uint64_t generation, std::set<CompletionKey> expected_members,
        std::set<NodeTerminalStatus> satisfying_statuses);

    /// Completion-fact recording, called by the online hook (read-only with
    /// respect to the graph; never releases dependencies). The direct member
    /// index means ordinary non-watch terminals never construct/copy the
    /// request_id/stage metadata. A key may belong to multiple watches; each
    /// retains its own duplicate/status/fire semantics.
    void on_node_terminal(const CompletionKey& key, NodeTerminalStatus status);

    /// Fired watches since the last call in no-notifier fixture/polling mode;
    /// production notifier mode retains no polling queue.
    std::vector<WatchFire> fired_and_drain();

    /// Registered-but-not-fired watches. Run-end audit must be 0 (aborted
    /// requests must remove_watch before finishing).
    size_t stale_count() const;

    /// Phase 4: total registered watches (fired or not). The phase-4 run-end
    /// audit requires the registry to be EMPTY (every watch of every
    /// completed request is removed at REQUEST_COMPLETE commit).
    [[nodiscard]] size_t size() const { return watches_.size(); }
    [[nodiscard]] bool empty() const { return watches_.empty(); }

    /// Lifecycle: request end / batch abort / run finish.
    void remove_watch(uint64_t watch_id);

    /// Phase 4: remove every registered watch of one request (all stages /
    /// generations). Wired at the REQUEST_COMPLETE commit -- by then both
    /// stage watches of the request have fired; removal keeps the run-end
    /// registry empty (the phase-4 end audit: mailbox/watch/heap/ready set
    /// all empty).
    void remove_watches_for_request(const std::string& request_id);

    /// 拼 batch 适配(2026-08-22):回收已 fire 的列车哨兵 watch
    /// (request_id 前缀 "batch_train_")。O(存活哨兵数)。
    void drain_fired_sentinels();

    void remove_all();

    /// Fire notification (step 1-6 wires this to the DecisionMailbox push).
    void set_fire_notifier(WatchFireNotifier notifier);

  private:
    using Identity = std::tuple<std::string, std::string, uint64_t>;

    uint64_t next_watch_id_ = 1;
    std::unordered_map<uint64_t, StageWatch> watches_;
    std::map<Identity, uint64_t> identity_to_watch_;
    // Direct terminal routing. Each expected member appends its watch id at
    // registration; remove_watch removes that id from every corresponding
    // vector. Multiple logical watches may intentionally share one key.
    std::unordered_map<CompletionKey, std::vector<uint64_t>, CompletionKeyHash>
        member_to_watch_ids_;
    // Phase 4: per-request watch-id index (for the REQUEST_COMPLETE-commit
    // removal; keeps removal O(watches of the request) instead of a full
    // scan).
    std::unordered_map<std::string, std::vector<uint64_t>>
        request_to_watch_ids_;
    // 拼 batch 适配(2026-08-22):哨兵 watch id 索引。
    std::vector<uint64_t> sentinel_watch_ids_;
    std::vector<WatchFire> fired_;
    WatchFireNotifier notifier_;
};

/// Context for the online CompletionObserver hook (owns the registry and,
/// step 1-8, the post-commit deferred issue-pass wiring).
struct OnlineCompletionHookContext {
    using IssuePassCallback = void (*)(void* context, int rank);

    OnlineCompletionHookContext() = default;
    OnlineCompletionHookContext(const OnlineCompletionHookContext&) = delete;
    OnlineCompletionHookContext& operator=(const OnlineCompletionHookContext&) =
        delete;

    WatchRegistry* registry = nullptr;
    /// Step 1-8: the per-tick deferred drain (post-commit phase). The hook
    /// schedules one per-rank issue pass into the SAME tick's deferred drain;
    /// the drain runs after the completion callback (and its
    /// finish_node dependency release) has fully executed, so the freed
    /// children are in the store's free set when the pass runs. This is the
    /// plan's "ready 集合 -> post-commit deferred 后继发射" wiring
    /// (方案 §4 步骤 1-4 操作 4 / 仿真加速分析.md §3.3): Workload::call
    /// itself never calls issue_dep_free_nodes in online mode (no static
    /// auto-advance), yet the pipeline advances continuously -- a freed
    /// child is issued in the same tick, keeping the offline completion
    /// times. Until configure_issue_passes() is called, the hook only records
    /// facts (fixture mode).
    ///
    /// The scheduler coalesces repeated terminal callbacks for the same rank
    /// into one pending pass. Its callback arguments are allocated once at
    /// configuration time and remain stable for the entire run. The pending
    /// bit is cleared before invoking the callback so nested completions can
    /// append a new same-rank pass to the current deferred drain.
    void configure_issue_passes(
        NetworkAnalytical::EventQueue* event_queue, size_t rank_count,
        IssuePassCallback callback, void* callback_context);
    bool schedule_issue_pass(int rank);
    [[nodiscard]] size_t pending_issue_pass_count() const;
    /// Phase 4: the per-delivery completed-facts collector
    /// (StateDelta.completed_nodes). Default mode aggregates audit-relevant
    /// count/rank facts before the tick-end hand-off; exact mode is opt-in.
    /// Driver-owned (main_online.cc); null = facts not recorded (fixture
    /// mode).
    CompletedFactAccumulator* completed_facts = nullptr;

  private:
    struct IssuePassArg {
        OnlineCompletionHookContext* owner = nullptr;
        int rank = -1;
    };

    static void issue_pass_trampoline(void* arg);

    NetworkAnalytical::EventQueue* event_queue_ = nullptr;
    IssuePassCallback issue_pass_callback_ = nullptr;
    void* issue_pass_callback_context_ = nullptr;
    std::vector<uint8_t> issue_pass_pending_;
    std::vector<IssuePassArg> issue_pass_args_;
};

/// Online-mode CompletionObserver hook (步骤 1-5 操作 3). Does ONLY:
///   (1) record the completion fact via WatchRegistry::on_node_terminal;
///   (2) (step 1-6) write the DecisionMailbox when a watch fires -- the
///       registry's fire notifier is the wiring point;
///   (3) (step 1-8) schedule the per-rank post-commit deferred issue pass
///       (ready-set drain; runs after finish_node in the same tick).
/// NEVER calls GraphSource::finish_node: dependency release belongs
/// exclusively to Workload::call. The static binary never installs this
/// hook (static hook stays null, byte-exact).
void online_completion_hook(void* ctx, int rank, uint64_t node_id,
                            const char* request_id, const char* stage,
                            uint64_t generation, uint64_t tick,
                            int terminal_status);

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_WATCHREGISTRY_HH
