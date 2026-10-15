#pragma once
#include "coarse_die/Plan.hh"
#include <array>
#include <cstddef>
#include <functional>
#include <map>
#include <optional>

namespace coarse_die {
struct Owner {
    std::uint64_t request_id=0, node_id=0, execution_id=0, generation=1;
    std::string key() const;
};

// Lifetime served-work buckets. These are classification labels only; they
// do not add resource domains or alter the existing 13-domain scheduler.
enum class ServedCategory : std::size_t {
    Comp = 0,
    CommSend,
    CommRecv,
    Restore,
    Pool,
    Die,
    Count
};

inline constexpr std::size_t served_category_count =
    static_cast<std::size_t>(ServedCategory::Count);

struct LiveActivity {
    Activity definition;
    std::size_t epoch_index=0;
    double progress=0, rate=0, alpha=0;
    bool done=false;
    std::array<std::array<long double,2>,2> hbm_served{};
    long double int4_served=0, vec_served=0;
};
struct Execution {
    std::uint64_t handle=0, issue_ns=0, finish_ns=0, ready_ns=0;
    Owner owner;
    ServedCategory served_category=ServedCategory::Comp;
    std::shared_ptr<const Plan> plan;
    std::size_t epoch=0;
    std::vector<LiveActivity> activities;
    std::array<double,2> credit{}, credit_capacity{};
    std::array<std::array<long double,2>,2> hbm_served{};
    long double int4_served=0, vec_served=0;
    bool terminal=false, cancelled=false;
};

struct ServedCounters {
    std::array<std::array<long double, 2>, 2> hbm_bytes{};
    long double int4_ops = 0.0L;
    long double vec_eqops = 0.0L;
};

class Runtime {
 public:
    explicit Runtime(Hardware hardware);
    explicit Runtime(std::shared_ptr<const Hardware> hardware);
    std::uint64_t submit(std::shared_ptr<const Plan>, Owner,
                         std::uint64_t now_ns);
    std::uint64_t submit_endpoint(const nlohmann::json&, Owner,
                                  std::uint64_t now_ns);
    void advance_to(std::uint64_t absolute_ns);
    void cancel(std::uint64_t handle, std::uint64_t now_ns);
    void update_capacity(std::size_t resource, double rate,
                         std::uint64_t now_ns);
    std::optional<std::uint64_t> next_event() const;
    nlohmann::json query_progress(std::uint64_t handle) const;
    // Returns null while the execution is live and its terminal timestamp
    // after completion or cancellation. Unknown/destroyed handles throw the
    // same stale_handle error as query_progress.
    std::optional<std::uint64_t> terminal_finish_ns(std::uint64_t handle) const;
    nlohmann::json counters() const;
    nlohmann::json export_snapshot() const;
    static std::unique_ptr<Runtime> import_snapshot(const Hardware&,
                                                  const nlohmann::json&);
    std::vector<std::uint64_t> execution_handles() const;
    bool drained() const;
    void destroy_execution(std::uint64_t handle);
    std::uint64_t now_ns() const { return now_ns_; }
    const Hardware& hardware() const { return *hardware_; }
 private:
    friend std::uint64_t detail::estimate_candidate_latency_ns(
        std::shared_ptr<const Hardware>, std::shared_ptr<const Plan>);
    std::uint64_t submit_impl(std::shared_ptr<const Plan>, Owner,
                             std::uint64_t now_ns, bool require_fingerprint);
    std::shared_ptr<const Hardware> hardware_;
    std::map<std::uint64_t,Execution> executions_;
    std::array<double,resource_count> capacities_{};
    std::uint64_t now_ns_=0, next_handle_=1, generation_=0;
    std::uint64_t allocation_updates_=0, internal_events_=0;
    std::map<std::uint64_t,std::uint64_t> endpoint_registry_;
    std::map<std::uint64_t,std::pair<Owner,std::size_t>> transaction_owners_;
    long double hbm_busy_ns_=0, hbm_shared_ns_=0, compute_busy_ns_=0;
    std::array<long double,2> row_busy_ns_{};
    std::array<long double,2> row_shared_ns_{};
    std::array<std::array<long double,2>,2> hbm_served_{};
    std::array<ServedCounters, served_category_count> served_by_category_{};
    std::array<std::array<long double, 2>, 2> legacy_unclassified_hbm_{};
    // v2 snapshots lack the cumulative categorized totals, and destroyed
    // execution handles make those totals unrecoverable. Such imports keep
    // known per-handle lower bounds plus unclassified historic HBM, and must
    // remain visibly incomplete through subsequent snapshots.
    bool accounting_history_complete_ = true;
    void settle();
    void allocate();
    void integrate(long double delta_ns);
    // Keep absolute integer time separate from sub-nanosecond integration.
    // Adding a short deadline to a large absolute long double can otherwise
    // erase its fractional part and change the public ceil by one ns.
    std::uint64_t exact_origin_ns_=0;
    long double exact_ns_=0;  // fractional ns in [0,1)
    long double time_until(std::uint64_t absolute_ns) const;
    std::uint64_t ceil_exact_time() const;
    void set_exact_time(std::uint64_t absolute_ns);
};
}
