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
    long double progress=0;
    double rate=0, alpha=0;
    bool done=false;
    std::array<std::array<long double,2>,2> hbm_served{};
    long double int4_served=0, vec_served=0;
    // Declared generative repeat instance index of this live copy. Always 0
    // for legacy plans, so the round filters are identity there.
    std::size_t round=0;
};
struct Execution {
    std::uint64_t handle=0, issue_ns=0, finish_ns=0, ready_ns=0;
    Owner owner;
    ServedCategory served_category=ServedCategory::Comp;
    std::shared_ptr<const Plan> plan;
    std::size_t epoch=0;
    // Current declared generative repeat instance (see PlanRepeat). The
    // epoch cursor rewinds to 0 when a round advances; past-round activities
    // stay done in the vector and are excluded from ledger sums by round.
    std::size_t round=0;
    std::vector<LiveActivity> activities;
    std::array<double,2> credit{}, credit_capacity{};
    // Directed outward bounds of the exact affine credit ledger at the
    // represented progress, maintained by recompute_credit. Equal to credit
    // far from a boundary; straddling bounds mean the true value may sit on
    // the boundary. Decision-only state: never exported on the wire.
    std::array<long double,2> credit_interval_lo{}, credit_interval_hi{};
    // Decision caches maintained by recompute_credit alongside the interval
    // arrays (and copied with them on import): the flight band every boundary
    // decision shares, whether the active epoch still sits at its declared
    // initial progress, and that epoch's curve presence (cache_epoch records
    // which epoch index the curved flag was computed for). Refreshing them in
    // one place keeps at_low/at_high, boundary rows, and crossing gates on the
    // same stored band without rescanning the activity vector per query. They
    // are never read before the first recompute_credit of a live state and
    // never exported on the wire.
    std::array<long double,2> flight_band{};
    bool epoch_curved=false;
    bool epoch_all_progress_zero=true;
    std::size_t cache_epoch=static_cast<std::size_t>(-1);
    // Per-epoch static band ingredients for that same cache_epoch,
    // view-relative index j addresses activities[view_begin + j] (the
    // current (round, epoch) window holds at most three entries):
    // |credit_delta[b]|*rate_bound[b] and |credit_delta[b]|*representation
    // epsilon. They let integrate()'s fused staging sweep refresh the band
    // without re-deriving rate bounds.
    std::array<std::array<long double,2>,3> band_delta_rate{};
    std::array<std::array<long double,2>,3> band_delta_eps{};
    // False whenever credit, the interval bounds, and every decision cache
    // above already reflect the current progress/done/epoch state, so a
    // second recompute_credit on an unchanged state can be skipped.
    bool credit_stale=true;
    std::array<std::array<long double,2>,2> hbm_served{};
    long double int4_served=0, vec_served=0;
    bool terminal=false, cancelled=false;
    // Current (round, epoch) scheduling window: the half-open activity index
    // range [view_begin, view_end). Event-path scans read only this window;
    // past rounds/epochs stay done in the vector for lifetime accounting and
    // explicit full queries. The layout is round-major then plan-epoch order,
    // so the window is exactly the activities with epoch_index==epoch and
    // round==round, done ones included (they still feed the credit ledger).
    // Windows only move forward and consecutive windows are contiguous.
    // Derived state: built at submit/import, advanced at every epoch/round
    // switch; never exported on the wire.
    std::size_t view_begin=0, view_end=0;
    // Window the previous allocate() may have published rates into. The next
    // allocate clears rate/alpha over [publish_view_begin,
    // max(publish_view_end, view_end)) instead of the whole history; every
    // activity outside that union is already zero because its window was
    // cleared when it was left. Derived, never on the wire.
    std::size_t publish_view_begin=0, publish_view_end=0;
    // Lifetime count of done activities. Monotone (only settle marks done),
    // appended rounds add only fresh items, so it feeds advance_to's settle
    // change digest. Derived, never on the wire.
    std::size_t done_total=0;
    // Cached has_credit_curves(plan): the plan is immutable after submit, and
    // advance_to consulted the free function per event iteration. Derived at
    // submit/import from the plan itself; never exported on the wire, so
    // imports rebuild it instead of trusting a foreign cache field.
    bool plan_curved=false;
};

struct ServedCounters {
    std::array<std::array<long double, 2>, 2> hbm_bytes{};
    long double int4_ops = 0.0L;
    long double vec_eqops = 0.0L;
};

// Ephemeral diagnostics of the most recent arbitration call. These describe
// solver cost, not physical simulation state, and are not snapshot counters.
struct ArbitrationDiagnostics {
    enum class Path { Empty, Singleton, Packing, SignedLP, Ratio };
    Path path=Path::Empty;
    std::size_t activities=0, constraint_rows=0, pivots=0, tangent_rounds=0;
};

// Exact dyadic accumulator for busy-time counters (P2 canonical semantics,
// 2026-10-17). Long double deltas are dyadic rationals; the running sum is
// carried exactly as significand * 2^exponent with a 128-bit significand.
// When an addition would need more than 128 significand bits, the value is
// rounded to nearest even at the 128th bit (about 38 significant decimal
// digits, far below the 18-digit decimal_measure contract). Public
// projections are correctly rounded from the exact value, so busy outputs
// are independent of the event partitioning that produced them.
class DyadicBusySum {
  public:
    void add(long double delta);
    long double value() const;
    double value_double() const;
    bool zero() const { return mantissa_ == 0; }
    // Canonical wire form: 32-hex-digit significand + exponent ("0",0 for
    // zero). import->export is the identity on canonical forms.
    nlohmann::json to_json() const;
    static DyadicBusySum from_json(const nlohmann::json&);
    // Snapshot import: prefer the exact dyadic field, fall back to the exact
    // value of the legacy decimal text (older snapshots).
    static DyadicBusySum from_snapshot(const nlohmann::json& parent,
                                       const char* dyadic_key,
                                       long double fallback);
  private:
    unsigned __int128 mantissa_ = 0;
    int exponent_ = 0;  // value = mantissa_ * 2^exponent_
    void normalize();
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
    // §4.2 observation counters (roofline batch): pure uint64 diagnostics of
    // scheduler/query activity — next_event queries, advance_to calls,
    // allocate calls, the arbitration path distribution, and total solver
    // pivots. They are call-history facts about this object only: they never
    // enter snapshot wire state, never change numeric behavior, and are
    // deliberately kept out of counters() because existing equivalence
    // tests compare counters() wholesale between runtimes and their snapshot
    // forks, whose call histories legitimately differ.
    nlohmann::json observation_counters() const;
    nlohmann::json export_snapshot() const;
    static std::unique_ptr<Runtime> import_snapshot(const Hardware&,
                                                  const nlohmann::json&);
    std::vector<std::uint64_t> execution_handles() const;
    bool drained() const;
    void destroy_execution(std::uint64_t handle);
    std::uint64_t now_ns() const { return now_ns_; }
    const Hardware& hardware() const { return *hardware_; }
    ArbitrationDiagnostics last_arbitration() const { return last_arbitration_; }
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
    ArbitrationDiagnostics last_arbitration_;
    // Backing storage for observation_counters(): diagnostics only, never on
    // the snapshot wire. next_event_calls_ is mutable because next_event()
    // is a const query.
    mutable std::uint64_t next_event_calls_=0;
    std::uint64_t advance_to_calls_=0;
    std::uint64_t allocate_calls_=0;
    std::uint64_t arbitration_path_empty_=0, arbitration_path_singleton_=0;
    std::uint64_t arbitration_path_packing_=0, arbitration_path_ratio_=0;
    std::uint64_t arbitration_path_signed_lp_=0;
    std::uint64_t solver_pivots_total_=0;
    std::map<std::uint64_t,std::uint64_t> endpoint_registry_;
    std::map<std::uint64_t,std::pair<Owner,std::size_t>> transaction_owners_;
    DyadicBusySum hbm_busy_ns_, compute_busy_ns_;
    long double hbm_shared_ns_=0;
    std::array<DyadicBusySum,2> row_busy_ns_{};
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
    void allocate(bool redundant_initial_boundaries,
                  std::size_t* import_pivots);
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
