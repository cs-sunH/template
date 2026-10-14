#pragma once

#include <functional>
#include <list>
#include <map>
#include <memory>
#include <string>

#include "coarse_die/Execution.hh"
#include "coarse_die/Memory.hh"
#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Common.hh"

namespace AstraSim {

class Sys;

// Per-rank bridge from the Sys nanosecond event loop to the shared coarse
// die Runtime. COMP plans and external local-HBM endpoints share this one
// Runtime instance; callbacks only signal their owner after an outer
// execution becomes terminal.
class CoarseDieRuntime final : public Callable {
  public:
    using Completion = std::function<void(uint64_t, const nlohmann::json&)>;

    CoarseDieRuntime(Sys* sys, coarse_die::Hardware hardware);
    ~CoarseDieRuntime() override;

    CoarseDieRuntime(const CoarseDieRuntime&) = delete;
    CoarseDieRuntime& operator=(const CoarseDieRuntime&) = delete;

    std::shared_ptr<const coarse_die::Plan> compile(
        const nlohmann::json& task);
    uint64_t submit_plan(std::shared_ptr<const coarse_die::Plan> plan,
                         coarse_die::Owner owner,
                         Completion completion);
    uint64_t submit_endpoint(const nlohmann::json& endpoint,
                             coarse_die::Owner owner,
                             Completion completion);
    void cancel(uint64_t handle, uint64_t now_ns, bool notify = false);
    nlohmann::json query_progress(uint64_t handle) const;
    // Sample one execution from an isolated Runtime copy. This does not
    // advance the live event loop or dispatch Workload callbacks.
    nlohmann::json query_progress_at(uint64_t handle,
                                     uint64_t now_ns) const;
    nlohmann::json counters() const;
    // Sample known die state without advancing the live event loop or
    // dispatching graph/network callbacks. Time reversal is rejected.
    nlohmann::json counters_at(uint64_t now_ns) const;
    nlohmann::json export_snapshot(uint64_t now_ns);
    const coarse_die::Hardware& hardware() const { return hardware_; }
    const nlohmann::json& hardware_summary() const { return hardware_.derived; }
    bool has_active_executions() const { return !completion_.empty(); }
    nlohmann::json plan_cache_counters() const;

    void call(EventType type, CallData* data) override;

  private:
    struct CachedPlan {
        std::shared_ptr<const coarse_die::Plan> plan;
        std::size_t accounted_bytes = 0;
        std::list<std::string>::iterator lru_position;
    };
    struct EventData final : public CallData {
        explicit EventData(uint64_t generation_in) : generation(generation_in) {}
        uint64_t generation;
    };
    static void destroy_event_data(CallData* data);
    void schedule_next();
    void dispatch_terminal();
    void retire(uint64_t handle);

    Sys* sys_;
    coarse_die::Hardware hardware_;
    coarse_die::Runtime runtime_;
    std::map<std::string, CachedPlan> plans_;
    std::list<std::string> plan_lru_;
    std::size_t plan_cache_accounted_bytes_ = 0;
    std::map<uint64_t, Completion> completion_;
    uint64_t event_generation_ = 0;
    SystemEventHandle event_;
    bool dispatching_ = false;
};

}  // namespace AstraSim
