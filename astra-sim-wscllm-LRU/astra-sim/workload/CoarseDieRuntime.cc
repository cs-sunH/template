#include "astra-sim/workload/CoarseDieRuntime.hh"
#include "astra-sim/workload/CoarseDiePrediction.hh"

#include <stdexcept>
#include <utility>
#include <vector>
#include <exception>

#include "astra-sim/system/Sys.hh"

namespace AstraSim {
namespace {
constexpr std::size_t kPlanCacheMaxEntries = 4096;
constexpr std::size_t kPlanCacheMaxAccountedBytes = 64u * 1024u * 1024u;

bool terminal_state(const nlohmann::json& progress) {
    const auto it = progress.find("state");
    return it != progress.end() && it->is_string() &&
        (it->get<std::string>() == "finished" ||
         it->get<std::string>() == "cancelled");
}
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
}

std::shared_ptr<const coarse_die::Plan> CoarseDieRuntime::compile(
    const nlohmann::json& task) {
    const std::string key = coarse_die::stable_hash(
        hardware_.fingerprint + "\n" + task.dump());
    const auto found = plans_.find(key);
    if (found != plans_.end()) {
        coarse_die::require_complete_prediction(*found->second.plan);
        plan_lru_.splice(plan_lru_.begin(), plan_lru_,
                         found->second.lru_position);
        found->second.lru_position = plan_lru_.begin();
        return found->second.plan;
    }
    auto plan = coarse_die::compile(hardware_, task);
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
    const nlohmann::json progress = runtime_.query_progress(handle);
    if (!terminal_state(progress)) {
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
    value.update(plan_cache_counters());
    return value;
}

nlohmann::json CoarseDieRuntime::counters() const {
    nlohmann::json value = runtime_.counters();
    value.update(plan_cache_counters());
    return value;
}

void CoarseDieRuntime::destroy_event_data(CallData* data) {
    delete static_cast<EventData*>(data);
}

void CoarseDieRuntime::schedule_next() {
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
        nlohmann::json progress;
        try {
            progress = runtime_.query_progress(it->first);
        } catch (...) {
            if (!first_exception) first_exception = std::current_exception();
            ++it;
            continue;
        }
        if (!terminal_state(progress)) {
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
