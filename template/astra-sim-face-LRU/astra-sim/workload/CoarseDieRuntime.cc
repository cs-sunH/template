#include "astra-sim/workload/CoarseDieRuntime.hh"
#include "astra-sim/workload/CoarseDiePrediction.hh"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>
#include <exception>

#include "astra-sim/system/Sys.hh"

namespace AstraSim {
namespace {
constexpr std::size_t kPlanCacheMaxEntries = 4096;
constexpr std::size_t kPlanCacheMaxAccountedBytes = 64u * 1024u * 1024u;
}

CoarseDieRuntime::CoarseDieRuntime(Sys* sys, coarse_die::Hardware hardware)
    : sys_(sys), hardware_(std::move(hardware)), runtime_(hardware_) {
    if (sys_ == nullptr) {
        throw std::runtime_error("coarse die runtime requires a Sys owner");
    }
}

CoarseDieRuntime::~CoarseDieRuntime() {
    if (event_.valid()) {
        static_cast<void>(sys_->cancel_event(event_));
    }
    for (const auto& entry : completion_) {
        try {
            runtime_.cancel(entry.first, runtime_.now_ns());
            runtime_.destroy_execution(entry.first);
        } catch (...) {
            // Teardown must not call into Workload callbacks. The Runtime
            // owns no Sys objects, so its destructor safely drops any
            // remaining model state if cancellation was already terminal.
        }
    }
    completion_.clear();
    // Observation-only teardown summary (roofline batch G2a/G2c): cumulative
    // plan-compile hit/miss counts, miss compile time, semantic key time,
    // scheduling/observation call counts, and arbitration path mix for this
    // runtime. Never parsed by production code.
    const nlohmann::json observation = runtime_.observation_counters();
    const auto obs = [&observation](const char* key,
                                    const char* nested) -> unsigned long long {
        const nlohmann::json* value = &observation;
        if (nested != nullptr) {
            value = &observation.at("arbitration_paths");
        }
        const auto it = value->find(key);
        if (it == value->end() || !it->is_number_unsigned()) return 0ULL;
        return static_cast<unsigned long long>(it->get<std::uint64_t>());
    };
    std::fprintf(stderr,
                 "[coarse-counters] plan_compile hits=%llu misses=%llu "
                 "compile_ns_total=%llu key_ns_total=%llu schedule_next=%llu "
                 "next_event=%llu advance_to=%llu allocate=%llu "
                 "arb[empty=%llu singleton=%llu packing=%llu ratio=%llu "
                 "signed_lp=%llu] pivots=%llu\n",
                 static_cast<unsigned long long>(plan_compile_hits_),
                 static_cast<unsigned long long>(plan_compile_misses_),
                 static_cast<unsigned long long>(plan_compile_ns_total_),
                 static_cast<unsigned long long>(plan_key_ns_total_),
                 static_cast<unsigned long long>(schedule_next_calls_),
                 obs("next_event_calls", nullptr),
                 obs("advance_to_calls", nullptr),
                 obs("allocate_calls", nullptr),
                 obs("empty", "arbitration_paths"),
                 obs("singleton", "arbitration_paths"),
                 obs("packing", "arbitration_paths"),
                 obs("ratio", "arbitration_paths"),
                 obs("signed_lp", "arbitration_paths"),
                 obs("solver_pivots_total", nullptr));
}

std::shared_ptr<const coarse_die::Plan> CoarseDieRuntime::compile(
    const nlohmann::json& task) {
    // Semantic compile-cache key (roofline batch P2-C): the canonical task
    // text from coarse_die::semantic_task_key() already dropped tracing-only
    // span identities (request ids/prefixes/first indices) and wrapper
    // provenance/config hashes, so plans are reused across train identities
    // whose tasks compile identically. The hardware fingerprint keeps
    // distinct hardware from sharing entries.
    const auto key_started = std::chrono::steady_clock::now();
    const std::string canonical =
        coarse_die::semantic_task_key(hardware_, task);
    const std::string key = coarse_die::stable_hash(
        hardware_.fingerprint + "\n" + canonical);
    plan_key_ns_total_ +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - key_started)
            .count();
    const auto found = plans_.find(key);
    if (found != plans_.end()) {
        ++plan_compile_hits_;
        // Same guard as the C ABI compile cache: a hash hit must still carry
        // the exact normalized task so a collision can never cross plans.
        if (found->second.plan->task.dump() != canonical) {
            throw std::runtime_error(
                "numerical_failure: normalized compile cache key collision");
        }
        coarse_die::require_complete_prediction(*found->second.plan);
        plan_lru_.splice(plan_lru_.begin(), plan_lru_,
                         found->second.lru_position);
        found->second.lru_position = plan_lru_.begin();
        return found->second.plan;
    }
    ++plan_compile_misses_;
    const auto compile_started = std::chrono::steady_clock::now();
    auto plan = coarse_die::compile(hardware_, task);
    plan_compile_ns_total_ +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - compile_started)
            .count();
    coarse_die::require_complete_prediction(*plan);
    // Conservative retained-allocation accounting (container capacity and
    // allocator allowance), explicitly distinct from host allocator RSS.
    const std::size_t accounted_bytes =
        coarse_die::retained_plan_bytes(*plan) + key.capacity() +
        sizeof(CachedPlan) + 512;
    if (accounted_bytes > kPlanCacheMaxAccountedBytes) {
        return plan;
    }
    plan_lru_.push_front(key);
    plans_.emplace(key, CachedPlan{plan, accounted_bytes, plan_lru_.begin()});
    plan_cache_accounted_bytes_ += accounted_bytes;
    while (plans_.size() > kPlanCacheMaxEntries ||
           plan_cache_accounted_bytes_ > kPlanCacheMaxAccountedBytes) {
        const std::string evicted_key = plan_lru_.back();
        plan_lru_.pop_back();
        const auto evicted = plans_.find(evicted_key);
        if (evicted != plans_.end()) {
            plan_cache_accounted_bytes_ -= evicted->second.accounted_bytes;
            plans_.erase(evicted);
        }
    }
    return plan;
}

nlohmann::json CoarseDieRuntime::plan_cache_counters() const {
    return nlohmann::json{
        {"plan_cache_entries", plans_.size()},
        {"plan_cache_compile_hits", plan_compile_hits_},
        {"plan_cache_compile_misses", plan_compile_misses_},
        {"plan_cache_compile_ns_total", plan_compile_ns_total_},
        {"plan_cache_key_ns_total", plan_key_ns_total_},
        {"schedule_next_calls", schedule_next_calls_},
        {"plan_cache_accounted_bytes", plan_cache_accounted_bytes_},
        {"plan_cache_accounting",
         "coarse_die::retained_plan_bytes(plan)+key.capacity()+sizeof(cache "
         "entry)+512 conservative metadata allowance; not allocator RSS"},
    };
}

uint64_t CoarseDieRuntime::submit_plan(
    std::shared_ptr<const coarse_die::Plan> plan,
    coarse_die::Owner owner,
    Completion completion) {
    if (!plan || !completion) {
        throw std::runtime_error(
            "coarse die submit_plan requires a plan and completion callback");
    }
    coarse_die::require_complete_prediction(*plan);
    const uint64_t now = Sys::boostedTick();
    const uint64_t handle = runtime_.submit(std::move(plan), owner, now);
    if (!completion_.emplace(handle, std::move(completion)).second) {
        throw std::runtime_error("coarse die Runtime reused an execution handle");
    }
    schedule_next();
    return handle;
}

uint64_t CoarseDieRuntime::submit_endpoint(
    const nlohmann::json& endpoint,
    coarse_die::Owner owner,
    Completion completion) {
    if (!completion) {
        throw std::runtime_error(
            "coarse die submit_endpoint requires a completion callback");
    }
    const uint64_t now = Sys::boostedTick();
    const uint64_t handle = runtime_.submit_endpoint(endpoint, owner, now);
    if (!completion_.emplace(handle, std::move(completion)).second) {
        throw std::runtime_error("coarse die Runtime reused an endpoint handle");
    }
    schedule_next();
    return handle;
}

void CoarseDieRuntime::cancel(const uint64_t handle,
                             const uint64_t now_ns,
                             const bool notify) {
    const auto found = completion_.find(handle);
    if (found == completion_.end()) {
        return;
    }
    if (!notify) {
        completion_.erase(found);
    }
    // Lightweight terminality check: building the full progress JSON here
    // would cost a complete serialization just to read one state field. A
    // terminal timestamp of 0 is legal, so only the optional's presence
    // decides (has_value), never the numeric truth of the timestamp.
    if (!runtime_.terminal_finish_ns(handle).has_value()) {
        runtime_.cancel(handle, now_ns);
    }
    std::exception_ptr first_exception;
    if (notify) {
        try {
            dispatch_terminal();
        } catch (...) {
            first_exception = std::current_exception();
        }
    } else {
        retire(handle);
    }
    try {
        schedule_next();
    } catch (...) {
        if (!first_exception) first_exception = std::current_exception();
    }
    if (first_exception) std::rethrow_exception(first_exception);
}

nlohmann::json CoarseDieRuntime::query_progress(const uint64_t handle) const {
    return runtime_.query_progress(handle);
}

nlohmann::json CoarseDieRuntime::query_progress_at(
    const uint64_t handle, const uint64_t now_ns) const {
    // Match counters_at(): advance a copy of the same Runtime, preserving
    // its time-reversal and unknown-handle checks without mutating the live
    // scheduler or dispatching callbacks.
    coarse_die::Runtime sampled(runtime_);
    sampled.advance_to(now_ns);
    return sampled.query_progress(handle);
}

nlohmann::json CoarseDieRuntime::export_snapshot(uint64_t now_ns) {
    if (dispatching_) {
        throw std::runtime_error(
            "coarse_die_v2 snapshot cannot be exported during terminal callback dispatch");
    }
    if (now_ns != Sys::boostedTick()) {
        throw std::runtime_error(
            "coarse_die_v2 snapshot tick must match the active Sys decision tick");
    }
    runtime_.advance_to(now_ns);
    std::exception_ptr first_exception;
    try {
        dispatch_terminal();
    } catch (...) {
        first_exception = std::current_exception();
    }
    try {
        schedule_next();
    } catch (...) {
        if (!first_exception) first_exception = std::current_exception();
    }
    if (first_exception) std::rethrow_exception(first_exception);
    return nlohmann::json{
        {"hardware_config", hardware_.input},
        {"runtime_snapshot", runtime_.export_snapshot()},
        {"exported_at_ns", now_ns},
    };
}

nlohmann::json CoarseDieRuntime::counters_at(const uint64_t now_ns) const {
    // Runtime copies mutable execution/accounting state and shares only the
    // immutable Hardware/Plans. The sample owns no external callbacks.
    coarse_die::Runtime sampled(runtime_);
    sampled.advance_to(now_ns);
    nlohmann::json value = sampled.counters();
    // Observation counters come from the live runtime: the sample's own
    // sampling advance is not part of the real call history.
    value.update(runtime_.observation_counters());
    value.update(plan_cache_counters());
    return value;
}

nlohmann::json CoarseDieRuntime::counters() const {
    nlohmann::json value = runtime_.counters();
    value.update(runtime_.observation_counters());
    value.update(plan_cache_counters());
    return value;
}

void CoarseDieRuntime::destroy_event_data(CallData* data) {
    delete static_cast<EventData*>(data);
}

void CoarseDieRuntime::schedule_next() {
    ++schedule_next_calls_;
    if (dispatching_) {
        return;
    }
    if (event_.valid()) {
        static_cast<void>(sys_->cancel_event(event_));
    }
    auto next = runtime_.next_event();
    if (!next.has_value()) {
        // A zero-work plan may be terminal at submit time. Defer its
        // callback until the next Sys tick so it cannot release graph
        // dependencies re-entrantly inside issue_comp().
        if (completion_.empty()) {
            return;
        }
        next = runtime_.now_ns() + 1;
    }
    const uint64_t now = Sys::boostedTick();
    const uint64_t delta = *next > now ? *next - now : 1;
    ++event_generation_;
    event_ = sys_->register_event_cancellable(
        this, EventType::General, new EventData(event_generation_), delta,
        &destroy_event_data);
}

void CoarseDieRuntime::dispatch_terminal() {
    if (dispatching_) {
        return;
    }
    dispatching_ = true;
    struct DispatchReset final {
        explicit DispatchReset(bool& value_in) : value(value_in) {}
        ~DispatchReset() { value = false; }
        bool& value;
    } reset(dispatching_);
    struct ReadyCompletion final {
        uint64_t handle;
        Completion completion;
        nlohmann::json progress;
    };
    std::exception_ptr first_exception;
    std::vector<ReadyCompletion> ready;
    ready.reserve(completion_.size());
    for (auto it = completion_.begin(); it != completion_.end();) {
        // Cheap terminality gate first: while the execution is live the tick
        // must not pay for a full query_progress JSON serialization. Stale
        // handles still surface the same out_of_range as query_progress.
        std::optional<uint64_t> finish_ns;
        try {
            finish_ns = runtime_.terminal_finish_ns(it->first);
        } catch (...) {
            if (!first_exception) first_exception = std::current_exception();
            ++it;
            continue;
        }
        // A terminal timestamp of 0 is legal, so the optional's presence
        // decides terminality, never the numeric truth of the timestamp.
        if (!finish_ns.has_value()) {
            ++it;
            continue;
        }
        nlohmann::json progress;
        try {
            progress = runtime_.query_progress(it->first);
        } catch (...) {
            if (!first_exception) first_exception = std::current_exception();
            ++it;
            continue;
        }
        ready.push_back(
            ReadyCompletion{it->first, std::move(it->second),
                            std::move(progress)});
        it = completion_.erase(it);
    }
    for (auto& entry : ready) {
        try {
            retire(entry.handle);
        } catch (...) {
            if (!first_exception) first_exception = std::current_exception();
            continue;
        }
        try {
            entry.completion(entry.handle, entry.progress);
        } catch (...) {
            if (!first_exception) first_exception = std::current_exception();
        }
    }
    if (first_exception) std::rethrow_exception(first_exception);
}

void CoarseDieRuntime::retire(const uint64_t handle) {
    runtime_.destroy_execution(handle);
}

void CoarseDieRuntime::call(EventType, CallData* data_ptr) {
    auto* data = static_cast<EventData*>(data_ptr);
    const uint64_t generation = data->generation;
    delete data;
    if (generation != event_generation_) {
        return;
    }
    event_.reset();
    runtime_.advance_to(Sys::boostedTick());
    std::exception_ptr first_exception;
    try {
        dispatch_terminal();
    } catch (...) {
        first_exception = std::current_exception();
    }
    try {
        schedule_next();
    } catch (...) {
        if (!first_exception) first_exception = std::current_exception();
    }
    if (first_exception) std::rethrow_exception(first_exception);
}

}  // namespace AstraSim
