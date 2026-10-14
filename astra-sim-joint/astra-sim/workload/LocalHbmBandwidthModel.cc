/******************************************************************************
This source code is licensed under the MIT license found in the root directory
of this source tree.
*******************************************************************************/

#include "astra-sim/workload/LocalHbmBandwidthModel.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/workload/CoarseDieRuntime.hh"
#include "astra-sim/workload/FailClosed.hh"
#include "astra-sim/workload/HardwareResource.hh"
#include "astra-sim/workload/Workload.hh"

using namespace AstraSim;

namespace {
constexpr double kEpsilon = 1e-9;

std::pair<double, double> coarse_hbm_read_write_bytes(
    const nlohmann::json& progress) {
    const nlohmann::json matrix = progress.value(
        "hbm_served_bytes_exact",
        progress.value("hbm_served_bytes", nlohmann::json::array()));
    if (!matrix.is_array() || matrix.size() != 2) {
        throw std::runtime_error(
            "coarse_die_v2 progress is missing the 2x2 HBM served matrix");
    }
    long double read_bytes = 0.0L;
    long double write_bytes = 0.0L;
    for (const auto& row : matrix) {
        if (!row.is_array() || row.size() != 2) {
            throw std::runtime_error(
                "coarse_die_v2 HBM served matrix row is malformed");
        }
        const auto as_number = [](const nlohmann::json& value) -> long double {
            if (value.is_string()) {
                const std::string exact = value.get<std::string>();
                size_t consumed = 0;
                const long double parsed = std::stold(exact, &consumed);
                if (consumed != exact.size() || !std::isfinite(parsed) ||
                    parsed < 0.0L) {
                    throw std::runtime_error(
                        "coarse_die_v2 HBM served byte string is invalid");
                }
                return parsed;
            }
            if (value.is_number()) {
                const long double parsed = value.get<long double>();
                if (!std::isfinite(parsed) || parsed < 0.0L) {
                    throw std::runtime_error(
                        "coarse_die_v2 HBM served byte value is invalid");
                }
                return parsed;
            }
            throw std::runtime_error(
                "coarse_die_v2 HBM served byte is neither number nor string");
        };
        read_bytes += as_number(row[0]);
        write_bytes += as_number(row[1]);
    }
    return {static_cast<double>(read_bytes), static_cast<double>(write_bytes)};
}

uint64_t coarse_endpoint_identity(const std::string& value) {
    uint64_t hash = 14695981039346656037ull;
    for (const unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    return hash == 0 ? 1 : hash;
}
}

LocalHbmBandwidthModel::LocalHbmBandwidthModel(Sys* sys, Workload* workload)
    : sys(sys),
      workload(workload),
      last_update_tick(Sys::boostedTick()),
      event_generation(0),
      scheduled_transition_event() {
    if (sys == nullptr || workload == nullptr) {
        throw std::invalid_argument("local HBM model requires Sys and Workload");
    }
    if (sys->coarse_die_runtime == nullptr &&
        (!std::isfinite(sys->local_mem_bw) ||
        !std::isfinite(sys->peak_perf) || sys->local_mem_bw <= 0 ||
        sys->peak_perf <= 0)) {
        throw std::invalid_argument(
            "local HBM sharing requires positive finite bandwidth and peak "
            "compute");
    }
}

LocalHbmBandwidthModel::~LocalHbmBandwidthModel() {
    cancel_scheduled_transition();
    if (sys != nullptr && sys->coarse_die_runtime != nullptr) {
        while (!coarse_jobs_.empty()) {
            const uint64_t handle = coarse_jobs_.begin()->first;
            const CoarseJob job = coarse_jobs_.begin()->second;
            try {
                const uint64_t now_ns = Sys::boostedTick();
                const auto progress =
                    sys->coarse_die_runtime->query_progress_at(handle, now_ns);
                const auto [read_bytes, write_bytes] =
                    coarse_hbm_read_write_bytes(progress);
                if (job.kind == JobKind::CONTROL_MARKER) {
                    coarse_control_marker_bytes_served_ += read_bytes;
                } else if (job.kind == JobKind::COMM_READ) {
                    coarse_comm_read_served_ += read_bytes;
                } else if (job.kind == JobKind::COMM_WRITE) {
                    coarse_comm_write_served_ += write_bytes;
                } else if (job.kind == JobKind::RESTORE) {
                    coarse_restore_served_ += write_bytes;
                } else if (job.kind == JobKind::POOL_READ) {
                    coarse_pool_read_served_ += read_bytes;
                } else if (job.kind == JobKind::POOL_WRITE) {
                    coarse_pool_write_served_ += write_bytes;
                }
                sys->coarse_die_runtime->cancel(handle, now_ns, false);
            } catch (...) {
                std::terminate();
            }
            if (job.wlhd != nullptr) {
                workload->abandon_hbm_endpoint_join(job.wlhd->node_id);
                delete job.wlhd;
            }
            coarse_jobs_.erase(handle);
        }
    }
    // A normal completion transfers each cookie to Workload::call(), but an
    // early shutdown can destroy the model while jobs are still queued. The
    // model owns those outstanding cookies until that callback, so release
    // them after cancelling the only event that could have delivered one.
    for (size_t slot = 0; slot < jobs_.size(); ++slot) {
        if (slot_alive_[slot] == 0) {
            continue;
        }
        Job& job = jobs_[slot];
        delete job.wlhd;
        job.wlhd = nullptr;
        // DIE_COMPUTE payloads have no WorkloadLayerHandlerData; the model
        // owned them from issue_die_memory until here (the sink is gone or
        // going away with the Workload -- never call it from a destructor).
        delete job.die_data;
        job.die_data = nullptr;
    }
    jobs_.clear();
}

bool LocalHbmBandwidthModel::has_active_jobs() const {
    return live_job_count_ > 0 || !coarse_jobs_.empty();
}

double LocalHbmBandwidthModel::compute_bytes_served() const {
    double served = bytes_served_by_kind_[kind_index(JobKind::COMPUTE)] +
                    coarse_control_marker_bytes_served_;
    if (sys == nullptr || sys->coarse_die_runtime == nullptr) {
        return served;
    }
    const uint64_t now_ns = Sys::boostedTick();
    for (const auto& entry : coarse_jobs_) {
        if (entry.second.kind != JobKind::CONTROL_MARKER) continue;
        const auto progress =
            sys->coarse_die_runtime->query_progress_at(entry.first, now_ns);
        const auto [read_bytes, ignored_write_bytes] =
            coarse_hbm_read_write_bytes(progress);
        (void)ignored_write_bytes;
        served += read_bytes;
    }
    return served;
}

size_t LocalHbmBandwidthModel::resource_observer_count() const noexcept {
    size_t count = 0;
    for (const ObserverEntry& entry : resource_observers_) {
        if (entry.id != 0) {
            ++count;
        }
    }
    return count;
}

size_t LocalHbmBandwidthModel::resource_observer_tombstone_count()
    const noexcept {
    return resource_observers_.size() - resource_observer_count();
}

LocalHbmBandwidthModel::MutationObserverId
LocalHbmBandwidthModel::add_resource_mutation_observer(
    ResourceMutationObserver observer) {
    if (!observer || next_observer_id_ == 0 ||
        next_observer_id_ == std::numeric_limits<MutationObserverId>::max()) {
        throw std::invalid_argument("invalid or exhausted HBM mutation observer");
    }
    const MutationObserverId id = next_observer_id_++;
    resource_observers_.push_back(ObserverEntry{id, 0, std::move(observer)});
    return id;
}

void LocalHbmBandwidthModel::remove_resource_mutation_observer(
    const MutationObserverId observer_id) noexcept {
    for (auto it = resource_observers_.begin();
         it != resource_observers_.end(); ++it) {
        if (it->id != observer_id) {
            continue;
        }
        if (observer_pass_depth_ == 0) {
            // No pass is iterating: drop the entry outright (the old
            // copy-based dispatch made a removal between passes invisible
            // to every later pass; erasing now reproduces that exactly).
            resource_observers_.erase(it);
        } else {
            // A pass is active. Tombstone: passes that started BEFORE this
            // removal still visit the entry (their snapshot copy included
            // it); passes that start later skip it via dead_pass. The
            // callback object stays alive until compaction so those
            // in-flight visits remain legal.
            it->id = 0;
            it->dead_pass = observer_pass_serial_;
            observers_compaction_pending_ = true;
        }
        return;
    }
}

void LocalHbmBandwidthModel::compact_observers() {
    if (!observers_compaction_pending_) {
        return;
    }
    // v8 reclamation fix: compact IN PLACE. The pass guard invokes this from
    // its destructor (outermost pass end -- including exception unwind), so
    // the compaction must not allocate: only noexcept move-assignments and
    // tail destruction happen here. Stable order is preserved.
    static_assert(
        std::is_nothrow_move_assignable<ObserverEntry>::value,
        "observer compaction move must be noexcept (guard destructor)");
    size_t write = 0;
    for (size_t read = 0; read < resource_observers_.size(); ++read) {
        if (resource_observers_[read].id != 0) {
            if (write != read) {
                resource_observers_[write] =
                    std::move(resource_observers_[read]);
            }
            ++write;
        }
    }
    resource_observers_.erase(resource_observers_.begin() +
                                  static_cast<std::ptrdiff_t>(write),
                              resource_observers_.end());
    observers_compaction_pending_ = false;
}

namespace {
// (The pass guard itself is a private nested class of the model -- nested
// classes reach the private compaction boundary; see the header.)
}  // namespace

LocalHbmBandwidthModel::ObserverPassGuard::ObserverPassGuard(
    LocalHbmBandwidthModel& model) noexcept : model_(model) {
    ++model_.observer_pass_depth_;
}

LocalHbmBandwidthModel::ObserverPassGuard::~ObserverPassGuard() {
    if (--model_.observer_pass_depth_ == 0) {
        // v8 reclamation fix: the OUTERMOST pass owns the compaction
        // boundary. At this point no pass is iterating (nested passes
        // already unwound), so tombstoned entries -- whose captured state
        // must merely outlive the passes that could still visit them --
        // can be destroyed exactly here. This mirrors the old per-notify
        // copy, which released its captured objects at the same
        // outermost-pass end. compact_observers() is allocation free, so
        // this stays legal during exception unwinding.
        model_.compact_observers();
    }
}

void LocalHbmBandwidthModel::dispatch_mutation_observers(
    const bool before_change, const Tick now, const uint64_t version) {
    // Snapshot dispatch (v7): each pass captures its own element bound and
    // serial. Entries appended after the bound are not visited (matching
    // the old per-notify vector copy); tombstoned entries are visited only
    // by passes that started before their removal (dead_pass >= my serial
    // means the removal happened at or after this pass started, so this
    // pass's snapshot still held the entry).
    const uint64_t my_serial = ++observer_pass_serial_;
    ObserverPassGuard depth_guard(*this);
    const size_t bound = resource_observers_.size();
    for (size_t i = 0; i < bound; ++i) {
        ObserverEntry& entry = resource_observers_[i];
        if (entry.id == 0 && entry.dead_pass < my_serial) {
            continue;  // removed before this pass started
        }
        entry.callback(before_change, now, version);
    }
    // v8 reclamation fix: compaction moved to the pass guard -- it fires
    // when the OUTERMOST pass ends (depth back to zero, including the
    // exception-unwind path). Checking the depth here, while this pass's
    // own guard is still alive, could never observe zero.
}

void LocalHbmBandwidthModel::notify_resource_mutation_before(const Tick now) {
    dispatch_mutation_observers(true, now, resource_version_);
}

void LocalHbmBandwidthModel::publish_resource_mutation(const Tick now) {
    if (resource_version_ == std::numeric_limits<uint64_t>::max()) {
        fail_closed("local HBM resource version overflows uint64");
    }
    ++resource_version_;
    dispatch_mutation_observers(false, now, resource_version_);
}

bool LocalHbmBandwidthModel::phase_erase(std::vector<size_t>& members,
                                         const size_t slot) {
    const auto it = std::lower_bound(members.begin(), members.end(), slot);
    if (it != members.end() && *it == slot) {
        members.erase(it);
        return true;
    }
    return false;
}

void LocalHbmBandwidthModel::phase_insert(std::vector<size_t>& members,
                                          const size_t slot) {
    members.insert(
        std::lower_bound(members.begin(), members.end(), slot),
        slot);
}

void LocalHbmBandwidthModel::register_job_phase(Job& job) {
    const size_t slot = job.slot;
    if (job.memory_latency_ns > kEpsilon) {
        phase_insert(latency_members_, slot);
    } else if (remaining_drain(job) > kEpsilon) {
        phase_insert(stream_members_, slot);
    }
    if (remaining_drain(job) > kEpsilon) {
        ++drain_pending_count_;
    }
    if (job.kind == JobKind::COMPUTE) {
        phase_insert(compute_members_, slot);
    }
    if (complete(job)) {
        mark_completion_candidate(job);
    }
}

void LocalHbmBandwidthModel::unregister_job_phase(const size_t slot) {
    phase_erase(latency_members_, slot);
    phase_erase(stream_members_, slot);
    phase_erase(compute_members_, slot);
}

void LocalHbmBandwidthModel::mark_completion_candidate(Job& job) {
    if (!job.in_candidates) {
        job.in_candidates = true;
        phase_insert(completion_candidates_, job.slot);
    }
}

double LocalHbmBandwidthModel::remaining_drain(const Job& job) {
    return job.kind == JobKind::DIE_COMPUTE ? job.remaining_service
                                            : job.remaining_bytes;
}

std::vector<LocalHbmBandwidthModel::JobSnapshotEntry>
LocalHbmBandwidthModel::job_snapshot() {
    // W1 (joint work ledger): per-job execution-truth remainder. Issue order
    // is the container order and the identity is (sys_id, node_id, kind) --
    // a node issues at most one job per HBM leg (mark_issued consumes the
    // node from the free set), so the export needs no extra keying.
    // A tick-end callback can be triggered by an unrelated event before this
    // model's own completion alarm.  Integrate that interval here so the
    // exported residual/as_of_tick is the actual decision boundary rather
    // than the last HBM transition.
    advance_to(Sys::boostedTick());
    std::vector<JobSnapshotEntry> snapshot;
    snapshot.reserve(live_job_count_);
    for (size_t slot = 0; slot < jobs_.size(); ++slot) {
        if (slot_alive_[slot] == 0) {
            continue;
        }
        const Job& job = jobs_[slot];
        if (job.kind == JobKind::DIE_COMPUTE) {
            // Internal memory stage of a still-running outer die COMP node:
            // no wlhd, no ledger node identity -- excluded by contract (see
            // the job_snapshot note in the header).
            continue;
        }
        JobSnapshotEntry entry;
        entry.kind = static_cast<int>(job.kind);
        entry.sys_id = job.wlhd != nullptr ? job.wlhd->sys_id
                                           : job.transfer_sys_id;
        entry.node_id = job.wlhd != nullptr ? job.wlhd->node_id
                                            : job.transfer_node_id;
        entry.job_id = job.issue_seq;
        entry.transfer_sink = job.transfer_completion != nullptr;
        entry.total_ops = job.total_ops;
        entry.total_bytes = job.total_bytes;
        entry.total_latency_ns = job.total_latency_ns;
        entry.remaining_ops = job.remaining_ops;
        entry.remaining_bytes = job.remaining_bytes;
        entry.memory_latency_ns = job.memory_latency_ns;
        entry.start_tick = job.start_tick;
        snapshot.push_back(entry);
    }
    return snapshot;
}

nlohmann::json LocalHbmBandwidthModel::coarse_snapshot_bindings(
    const nlohmann::json& local_sample) const {
    using Json = nlohmann::json;
    const auto reject = [](const std::string& reason) {
        throw std::invalid_argument(
            "coarse_die_v2 local HBM sample: " + reason);
    };
    const auto member = [&](const Json& object,
                            const char* key,
                            const std::string& path) -> const Json& {
        if (!object.is_object() || !object.contains(key)) {
            reject(path + " is missing " + key);
        }
        return object.at(key);
    };
    const auto uint_value = [&](const Json& value,
                                const std::string& path) -> uint64_t {
        if (value.is_number_unsigned()) {
            return value.get<uint64_t>();
        }
        if (value.is_number_integer()) {
            const int64_t parsed = value.get<int64_t>();
            if (parsed >= 0) return static_cast<uint64_t>(parsed);
        }
        reject(path + " must be a nonnegative integer");
        return 0;
    };
    const auto bool_value = [&](const Json& value,
                                const std::string& path) -> bool {
        if (!value.is_boolean()) reject(path + " must be boolean");
        return value.get<bool>();
    };
    const auto string_value = [&](const Json& value,
                                  const std::string& path) -> std::string {
        if (!value.is_string()) reject(path + " must be a string");
        return value.get<std::string>();
    };

    if (sys == nullptr || sys->coarse_die_runtime == nullptr) {
        reject("requires an active coarse DIE Runtime");
    }
    if (!local_sample.is_object() ||
        string_value(member(local_sample, "schema", "sample"),
                     "sample.schema") != "coarse_die_local_sample_v1") {
        reject("incompatible sample schema");
    }
    if (string_value(member(local_sample, "scope", "sample"),
                     "sample.scope") != "rank_local_runtime_only" ||
        bool_value(member(local_sample, "callbacks_dispatched", "sample"),
                   "sample.callbacks_dispatched") ||
        bool_value(member(local_sample,
                          "production_world_import_supported", "sample"),
                   "sample.production_world_import_supported")) {
        reject("sample scope or callback status is incompatible");
    }

    const Json& rank_wire = member(local_sample, "rank", "sample");
    int sample_rank = 0;
    if (rank_wire.is_number_unsigned()) {
        const uint64_t parsed = rank_wire.get<uint64_t>();
        if (parsed > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            reject("rank is out of range");
        }
        sample_rank = static_cast<int>(parsed);
    } else if (rank_wire.is_number_integer()) {
        const int64_t parsed = rank_wire.get<int64_t>();
        if (parsed < std::numeric_limits<int>::min() ||
            parsed > std::numeric_limits<int>::max()) {
            reject("rank is out of range");
        }
        sample_rank = static_cast<int>(parsed);
    } else {
        reject("rank must be an integer");
    }
    if (sample_rank != sys->id) reject("rank does not match the local Sys");

    const uint64_t sampled_at_ns = uint_value(
        member(local_sample, "sampled_at_ns", "sample"),
        "sample.sampled_at_ns");
    if (sampled_at_ns != Sys::boostedTick()) {
        reject("sample tick does not match the active Sys decision tick");
    }
    const uint64_t live_runtime_now_ns = uint_value(
        member(local_sample, "live_runtime_now_ns", "sample"),
        "sample.live_runtime_now_ns");
    if (live_runtime_now_ns > sampled_at_ns) {
        reject("live Runtime clock is after the sample tick");
    }

    const auto& hardware = sys->coarse_die_runtime->hardware();
    const std::string expected_fingerprint = hardware.fingerprint;
    if (string_value(member(local_sample, "hardware_fingerprint", "sample"),
                     "sample.hardware_fingerprint") != expected_fingerprint ||
        member(local_sample, "hardware_config", "sample") != hardware.input) {
        reject("sample hardware does not match the local Runtime");
    }

    const Json& runtime = member(local_sample, "runtime_snapshot", "sample");
    if (!runtime.is_object() ||
        string_value(member(runtime, "schema", "runtime_snapshot"),
                     "runtime_snapshot.schema") != "coarse_die_snapshot_v3" ||
        uint_value(member(runtime, "snapshot_version", "runtime_snapshot"),
                   "runtime_snapshot.snapshot_version") != 3 ||
        string_value(member(runtime, "hardware_fingerprint", "runtime_snapshot"),
                     "runtime_snapshot.hardware_fingerprint") != expected_fingerprint ||
        uint_value(member(runtime, "now_ns", "runtime_snapshot"),
                   "runtime_snapshot.now_ns") != sampled_at_ns) {
        reject("Runtime snapshot schema, hardware, or clock does not match the sample");
    }

    const Json& executions = member(runtime, "executions", "runtime_snapshot");
    const Json& bindings = member(local_sample, "completion_bindings", "sample");
    if (!executions.is_array() || !bindings.is_array()) {
        reject("Runtime executions and completion bindings must be arrays");
    }
    std::map<uint64_t, const Json*> executions_by_handle;
    std::map<uint64_t, const Json*> bindings_by_handle;
    for (const auto& execution : executions) {
        if (!execution.is_object()) reject("Runtime execution must be an object");
        const uint64_t handle = uint_value(
            member(execution, "handle", "runtime execution"),
            "runtime execution.handle");
        if (!executions_by_handle.emplace(handle, &execution).second) {
            reject("Runtime snapshot contains a duplicate handle");
        }
    }
    for (const auto& binding : bindings) {
        if (!binding.is_object()) reject("completion binding must be an object");
        const uint64_t handle = uint_value(
            member(binding, "handle", "completion binding"),
            "completion binding.handle");
        if (!bindings_by_handle.emplace(handle, &binding).second) {
            reject("completion bindings contain a duplicate handle");
        }
    }
    if (executions_by_handle.size() != bindings_by_handle.size()) {
        reject("Runtime execution and callback-binding sets differ");
    }
    for (const auto& entry : executions_by_handle) {
        const auto binding_it = bindings_by_handle.find(entry.first);
        if (binding_it == bindings_by_handle.end()) {
            reject("Runtime execution has no completion binding");
        }
        const Json& execution = *entry.second;
        const Json& binding = *binding_it->second;
        const Json& execution_owner = member(execution, "owner", "runtime execution");
        if (member(binding, "owner", "completion binding") != execution_owner) {
            reject("completion binding owner differs from its Runtime execution");
        }
        const bool terminal = bool_value(
            member(execution, "terminal", "runtime execution"),
            "runtime execution.terminal");
        const bool cancelled = bool_value(
            member(execution, "cancelled", "runtime execution"),
            "runtime execution.cancelled");
        const bool terminal_at_sample = bool_value(
            member(binding, "terminal_at_sample", "completion binding"),
            "completion binding.terminal_at_sample");
        const Json& execution_finish = member(
            execution, "finish_ns", "runtime execution");
        const Json& binding_finish = member(
            binding, "finish_ns_at_sample", "completion binding");
        if (terminal != terminal_at_sample ||
            terminal != !execution_finish.is_null() ||
            execution_finish.is_null() != binding_finish.is_null()) {
            reject("completion binding terminal state differs from its Runtime execution");
        }
        if (!execution_finish.is_null() &&
            uint_value(execution_finish, "runtime execution.finish_ns") !=
                uint_value(binding_finish,
                           "completion binding.finish_ns_at_sample")) {
            reject("completion binding finish tick differs from its Runtime execution");
        }
        const std::string state = string_value(
            member(binding, "state_at_sample", "completion binding"),
            "completion binding.state_at_sample");
        if ((terminal && state != (cancelled ? "cancelled" : "finished")) ||
            (!terminal && (state == "cancelled" || state == "finished"))) {
            reject("completion binding state differs from its Runtime execution");
        }
    }

    struct Metadata {
        const char* name;
        const char* owner_kind;
        const char* direction;
        const char* path_class;
        bool initial_delay;
    };
    const auto kind_metadata = [&](JobKind kind) -> Metadata {
        switch (kind) {
            case JobKind::RESTORE:
                return Metadata{"restore", "restore", "write", "edge_to_hbm", false};
            case JobKind::COMM_READ:
                return Metadata{"comm_read", "comm", "read", "hbm_to_edge", false};
            case JobKind::COMM_WRITE:
                return Metadata{"comm_write", "comm", "write", "edge_to_hbm", false};
            case JobKind::POOL_READ:
                return Metadata{"pool_read", "pool", "read", "hbm_to_edge", false};
            case JobKind::POOL_WRITE:
                return Metadata{"pool_write", "pool", "write", "edge_to_hbm", false};
            case JobKind::CONTROL_MARKER:
                return Metadata{"control_marker", "die", "read", "hbm_to_edge", true};
            default:
                reject("coarse job has no external endpoint kind");
                return Metadata{"", "", "", "", false};
        }
    };

    Json result = Json::array();
    for (const auto& job_entry : coarse_jobs_) {
        const uint64_t handle = job_entry.first;
        const CoarseJob& job = job_entry.second;
        if (job.has_transfer_sink != (job.wlhd == nullptr)) {
            reject("coarse job callback ownership differs from its saved identity");
        }
        const auto execution_it = executions_by_handle.find(handle);
        const auto binding_it = bindings_by_handle.find(handle);
        if (execution_it == executions_by_handle.end() ||
            binding_it == bindings_by_handle.end()) {
            reject("active coarse job has no matching sampled callback handle");
        }
        const Json& execution = *execution_it->second;
        const Json& binding = *binding_it->second;
        const Json& plan = member(execution, "plan", "runtime execution");
        if (string_value(member(plan, "hardware_fingerprint", "execution plan"),
                         "execution plan.hardware_fingerprint") != expected_fingerprint) {
            reject("execution plan hardware differs from the sampled Runtime");
        }
        const Json& task = member(plan, "task", "execution plan");
        if (string_value(member(task, "schema", "execution plan task"),
                         "execution plan task.schema") != "coarse_die_endpoint_v2") {
            reject("active local-HBM handle is not an endpoint Plan");
        }
        const Json& endpoint = member(task, "endpoint", "execution plan task");
        const Metadata metadata = kind_metadata(job.kind);
        const uint64_t expected_endpoint_id = coarse_endpoint_identity(
            std::to_string(job.completion_sys_id) + ":" +
            std::to_string(job.node_id) + ":" +
            std::to_string(static_cast<int>(job.kind)));
        const uint64_t expected_transaction_id = coarse_endpoint_identity(
            std::to_string(job.node_id));
        const Json& owner = member(execution, "owner", "runtime execution");
        const coarse_die::Owner expected_owner_value{
            expected_transaction_id, job.node_id, job.node_id, 1};
        const Json expected_owner = {
            {"request_id", expected_transaction_id},
            {"node_id", job.node_id},
            {"execution_id", job.node_id},
            {"generation", 1},
            {"key", expected_owner_value.key()}};
        if (owner != expected_owner ||
            member(binding, "owner", "completion binding") != expected_owner) {
            reject("sampled owner differs from the coarse job's saved identity");
        }
        if (uint_value(member(endpoint, "endpoint_id", "endpoint"),
                       "endpoint.endpoint_id") != expected_endpoint_id ||
            uint_value(member(endpoint, "transaction_id", "endpoint"),
                       "endpoint.transaction_id") != expected_transaction_id ||
            string_value(member(endpoint, "owner_kind", "endpoint"),
                         "endpoint.owner_kind") != metadata.owner_kind ||
            string_value(member(endpoint, "direction", "endpoint"),
                         "endpoint.direction") != metadata.direction ||
            string_value(member(endpoint, "path_class", "endpoint"),
                         "endpoint.path_class") != metadata.path_class) {
            reject("sampled endpoint identity or kind metadata differs from its coarse job");
        }
        const Json& route = member(endpoint, "route", "endpoint");
        if (!route.is_object() ||
            member(route, "internal_route", "endpoint.route") != false) {
            reject("sampled endpoint route differs from the local-HBM endpoint");
        }
        if (metadata.initial_delay) {
            if (uint_value(member(endpoint, "initial_delay_ns", "endpoint"),
                           "endpoint.initial_delay_ns") !=
                static_cast<uint64_t>(sys->local_mem_latency)) {
                reject("control-marker endpoint latency differs from the local HBM configuration");
            }
        } else if (endpoint.contains("initial_delay_ns")) {
            reject("non-marker endpoint unexpectedly has an initial delay");
        }

        result.push_back({
            {"handle", handle},
            {"domain_rank", sys->id},
            {"completion_rank", job.completion_sys_id},
            {"node_id", job.node_id},
            {"job_kind", metadata.name},
            {"transfer_sink", job.has_transfer_sink},
            {"endpoint", endpoint},
            {"owner", owner},
            {"terminal_at_sample",
             member(binding, "terminal_at_sample", "completion binding")},
            {"finish_ns_at_sample",
             member(binding, "finish_ns_at_sample", "completion binding")}});
    }
    return result;
}

double LocalHbmBandwidthModel::bandwidth_bytes_per_ns() const {
    return sys->local_mem_bw / 1e9;
}

bool LocalHbmBandwidthModel::memory_done(const Job& job) {
    if (job.kind == JobKind::DIE_COMPUTE) {
        // No latency phase (contract §7.4: the DieExecution charges
        // local_mem_latency itself as a one-shot TIMER before the first
        // memory request); the service stream is the only memory quantity.
        return job.remaining_service <= kEpsilon;
    }
    return job.memory_latency_ns <= kEpsilon &&
        job.remaining_bytes <= kEpsilon;
}

bool LocalHbmBandwidthModel::complete(const Job& job) {
    return memory_done(job) && job.remaining_ops <= kEpsilon;
}

void LocalHbmBandwidthModel::advance_to(Tick now) {
    if (now < last_update_tick) {
        fail_closed("local HBM model time moved backwards");
    }
    const double elapsed_ns = static_cast<double>(now - last_update_tick);
    if (elapsed_ns <= 0) {
        return;
    }
    // v7: the active-streamer membership "before" set is implicit. The old
    // implementation captured it as an index vector and compared an "after"
    // vector at the end; only the comparison OUTCOME drives the
    // notify/publish decisions below, and it flips exactly when a latency
    // job crosses into streaming or a streamer's drain crosses out. Those
    // crossings are recorded where they happen.
    bool membership_changed = false;
    const bool predicted_boundary = has_next_transition_tick_ &&
        next_transition_changes_membership_ && now >= next_transition_tick_;
    if (predicted_boundary) {
        notify_resource_mutation_before(now);
    }

    // Compute FLOPs always progress at peak rate in parallel with the HBM
    // fluid (Roofline max(compute, memory) semantics), for every COMPUTE
    // job regardless of how many jobs share the bus.
    const double compute_rate_ops_per_ns = sys->peak_perf / 1e9;
    for (const size_t slot : compute_members_) {
        Job& job = jobs_[slot];
        const double ops_before = job.remaining_ops;
        job.remaining_ops = std::max(
            0.0,
            job.remaining_ops -
                compute_rate_ops_per_ns * elapsed_ns);
        if (ops_before > kEpsilon && job.remaining_ops <= kEpsilon) {
            // Ops crossed the completion window: the job may now satisfy
            // complete(); queue it for the next detach.
            mark_completion_candidate(job);
        }
    }

    const double full_rate_bytes_per_ns = sys->local_mem_bw / 1e9;
    double remaining_ns = elapsed_ns;
    while (remaining_ns > kEpsilon) {
        // v7: bandwidth users are the maintained streaming list (latency
        // done, drain above tolerance) in slot order, so the min scans and
        // the serve/apply loops below keep the original issue-order float
        // sequence. The old call-local scratch vector is gone; the list IS
        // the membership.
        double step_ns = remaining_ns;
        for (const size_t slot : latency_members_) {
            const Job& job = jobs_[slot];
            step_ns = std::min(step_ns, job.memory_latency_ns);
        }

        // N-way strict equal split: every byte-streaming job receives
        // full_rate / N where N is the number of concurrent streamers.
        const double per_user_rate = stream_members_.empty()
            ? 0.0
            : full_rate_bytes_per_ns /
                static_cast<double>(stream_members_.size());
        if (per_user_rate > 0) {
            for (const size_t slot : stream_members_) {
                const Job& job = jobs_[slot];
                step_ns = std::min(
                    step_ns, remaining_drain(job) / per_user_rate);
            }
        }

        if (step_ns <= kEpsilon) {
            // A subtraction at the previous transition can leave a byte
            // residue whose service time is below the numerical time
            // tolerance, while the byte value itself is still above the
            // quantity tolerance.  Breaking here would preserve that residue
            // forever and create one transition per nanosecond.  Clamp every
            // quantity that can complete within the time tolerance, then
            // recompute rates without consuming measurable simulation time.
            // (v7: this is the rare numerical-residue path; it still scans
            // the live slots -- the hot substep path above is the one that
            // must not.)
            bool clamped_residue = false;
            std::vector<size_t> clamp_drained;
            for (size_t slot = 0; slot < jobs_.size(); ++slot) {
                if (slot_alive_[slot] == 0) {
                    continue;
                }
                Job& job = jobs_[slot];
                if (job.memory_latency_ns > 0 &&
                    job.memory_latency_ns <= kEpsilon) {
                    // Zeroing a sub-tolerance latency changes no phase
                    // membership (the job is already past the latency gate)
                    // and no completion predicate (memory_done already
                    // treats <= epsilon as done); it only exports 0 instead
                    // of the residue.
                    job.memory_latency_ns = 0;
                    clamped_residue = true;
                }
                if (job.remaining_bytes > 0 &&
                    job.remaining_bytes <= kEpsilon) {
                    job.remaining_bytes = 0;
                    clamped_residue = true;
                    mark_completion_candidate(job);
                }
                if (job.remaining_service > 0 &&
                    job.remaining_service <= kEpsilon) {
                    job.remaining_service = 0;
                    clamped_residue = true;
                    clamp_drained.push_back(slot);
                    mark_completion_candidate(job);
                }
            }
            if (per_user_rate > 0) {
                std::vector<size_t> user_clamp_drained;
                for (const size_t slot : stream_members_) {
                    Job& job = jobs_[slot];
                    if (remaining_drain(job) / per_user_rate <= kEpsilon) {
                        if (job.kind == JobKind::DIE_COMPUTE) {
                            job.remaining_service = 0;
                        } else {
                            job.remaining_bytes = 0;
                        }
                        clamped_residue = true;
                        user_clamp_drained.push_back(slot);
                        mark_completion_candidate(job);
                    }
                }
                for (const size_t slot : user_clamp_drained) {
                    if (phase_erase(stream_members_, slot)) {
                        membership_changed = true;
                    }
                }
            }
            for (const size_t slot : clamp_drained) {
                if (phase_erase(stream_members_, slot)) {
                    membership_changed = true;
                }
            }
            if (!clamped_residue) {
                fail_closed("local HBM model made no progress at a transition");
            }
            continue;
        }

        // Per-substep apply: value updates iterate the phase lists captured
        // for THIS substep; membership transitions (latency expiry, drain
        // completion) are recorded and committed after the serve/side-band
        // loops. The old implementation had the same effect by rescanning
        // every job per substep: within a substep the user set is frozen
        // (a job that expires or drains in this substep was still a member
        // for its serve and side-band accounting), and the next substep
        // sees the updated membership.
        std::vector<size_t> latency_expired;
        std::vector<size_t> stream_drained;
        for (const size_t slot : latency_members_) {
            Job& job = jobs_[slot];
            const double latency_before = job.memory_latency_ns;
            job.memory_latency_ns =
                std::max(0.0, job.memory_latency_ns - step_ns);
            if (latency_before > kEpsilon &&
                job.memory_latency_ns <= kEpsilon) {
                latency_expired.push_back(slot);
            }
        }
        for (const size_t slot : stream_members_) {
            Job& job = jobs_[slot];
            const double served = per_user_rate * step_ns;
            const double drain_before = remaining_drain(job);
            if (job.kind == JobKind::DIE_COMPUTE) {
                // Contract §7.4: service units are drained; the actual-byte
                // advance is served * eta and is bookkeeping only.
                job.remaining_service =
                    std::max(0.0, job.remaining_service - served);
            } else {
                job.remaining_bytes =
                    std::max(0.0, job.remaining_bytes - served);
            }
            if (drain_before > kEpsilon && remaining_drain(job) <= kEpsilon) {
                stream_drained.push_back(slot);
            }
        }
        // Read-only side-band accumulation (doc sec.9.4): reuse the step_ns
        // and per_user_rate already computed above.  These counters must not
        // change any value used for the next transition.  The iteration
        // order (stream list = slot order = issue order) preserves the
        // historical accumulation order, including the per-job eta factor
        // of die_actual_bytes_served_.  Like the historical bandwidth_users
        // scratch vector, the list still holds jobs that drained to zero in
        // THIS substep -- they were members for this substep's accounting.
        if (!stream_members_.empty()) {
            this->hbm_busy_ns_ += step_ns;
            if (stream_members_.size() > 1) {
                this->hbm_shared_ns_ += step_ns;
            }
            for (const size_t slot : stream_members_) {
                const Job& job = jobs_[slot];
                if (job.kind == JobKind::DIE_COMPUTE) {
                    // Die side-band (kept separate from the per-kind byte
                    // array): service units drained + actual bytes advanced
                    // (= service * eta). Neither ever feeds a byte-based
                    // consumer.
                    this->die_service_units_served_ +=
                        per_user_rate * step_ns;
                    this->die_actual_bytes_served_ +=
                        per_user_rate * step_ns * job.service_eta;
                    continue;
                }
                this->bytes_served_by_kind_[kind_index(job.kind)] +=
                    per_user_rate * step_ns;
            }
        }
        // Commit the membership transitions recorded above.
        for (const size_t slot : latency_expired) {
            phase_erase(latency_members_, slot);
            Job& job = jobs_[slot];
            if (remaining_drain(job) > kEpsilon) {
                phase_insert(stream_members_, slot);
                membership_changed = true;
            }
            mark_completion_candidate(job);
        }
        for (const size_t slot : stream_drained) {
            phase_erase(stream_members_, slot);
            --drain_pending_count_;
            membership_changed = true;
            mark_completion_candidate(jobs_[slot]);
        }
        remaining_ns -= step_ns;
    }
    last_update_tick = now;
    if (membership_changed) {
        if (!predicted_boundary) {
            // This can occur when a read-only snapshot advances over a
            // transition whose event callback has not run yet. The model
            // still publishes the version before any completion callback;
            // callers must preserve the pending same-tick callback itself.
            notify_resource_mutation_before(now);
        }
        publish_resource_mutation(now);
    }
}

void LocalHbmBandwidthModel::cancel_scheduled_transition() {
    if (scheduled_transition_event.valid()) {
        static_cast<void>(sys->cancel_event(scheduled_transition_event));
    }
    scheduled_transition_event.reset();
    has_next_transition_tick_ = false;
    next_transition_changes_membership_ = false;
}

void LocalHbmBandwidthModel::destroy_transition_data(CallData* data) {
    delete static_cast<TransitionData*>(data);
}

void LocalHbmBandwidthModel::schedule_next_transition() {
    // Every reallocation replaces the old prediction. The old callback would
    // only hit its generation guard, so delete its payload/list node now.
    cancel_scheduled_transition();
    ++event_generation;
    if (!has_active_jobs()) {
        return;
    }

    size_t bandwidth_user_count = stream_members_.size();
    double next_ns = std::numeric_limits<double>::infinity();
    double next_membership_ns = std::numeric_limits<double>::infinity();
    for (const size_t slot : latency_members_) {
        const Job& job = jobs_[slot];
        next_ns = std::min(next_ns, job.memory_latency_ns);
        if (remaining_drain(job) > kEpsilon) {
            next_membership_ns =
                std::min(next_membership_ns, job.memory_latency_ns);
        }
    }

    const double full_rate_bytes_per_ns = sys->local_mem_bw / 1e9;
    const double per_user_rate = bandwidth_user_count == 0
        ? 0.0
        : full_rate_bytes_per_ns /
            static_cast<double>(bandwidth_user_count);
    if (per_user_rate > 0) {
        // Stream members in slot order (issue order), applying the same
        // per-user division and minimum updates as the historical scan.
        for (const size_t slot : stream_members_) {
            const Job& job = jobs_[slot];
            const double drain = remaining_drain(job);
            if (!(drain > kEpsilon)) {
                continue;
            }
            const double drain_ns = drain / per_user_rate;
            next_ns = std::min(next_ns, drain_ns);
            next_membership_ns = std::min(next_membership_ns, drain_ns);
        }
    }

    // A COMPUTE job whose HBM side is already done may still be streaming
    // FLOPs; its ops completion is a transition of its own.
    const double compute_rate_ops_per_ns = sys->peak_perf / 1e9;
    for (const size_t slot : compute_members_) {
        const Job& job = jobs_[slot];
        if (job.kind == JobKind::COMPUTE && memory_done(job) &&
            job.remaining_ops > kEpsilon) {
            next_ns = std::min(
                next_ns, job.remaining_ops / compute_rate_ops_per_ns);
        }
    }

    if (!std::isfinite(next_ns)) {
        fail_closed("local HBM model has no schedulable transition");
    }
    const double delay_ns = std::ceil(next_ns);
    if (!std::isfinite(delay_ns) ||
        delay_ns >= static_cast<double>(std::numeric_limits<Tick>::max())) {
        fail_closed("local HBM transition exceeds the representable Tick range");
    }
    Tick delay = static_cast<Tick>(delay_ns);
    delay = std::max<Tick>(1, delay);
    const Tick now = Sys::boostedTick();
    if (delay > std::numeric_limits<Tick>::max() - now) {
        fail_closed("local HBM transition tick overflows Tick");
    }
    has_next_transition_tick_ = true;
    next_transition_tick_ = now + delay;
    next_transition_changes_membership_ =
        std::isfinite(next_membership_ns) &&
        next_membership_ns <= next_ns + kEpsilon;
    scheduled_transition_event = sys->register_event_cancellable(
        this,
        EventType::General,
        new TransitionData(event_generation),
        delay,
        destroy_transition_data);
}

void LocalHbmBandwidthModel::issue_job(
    JobKind kind,
    uint64_t num_ops,
    uint64_t tensor_size,
    WorkloadLayerHandlerData* wlhd,
    std::function<void()> transfer_completion,
    int transfer_sys_id,
    uint64_t transfer_node_id) {
    if (sys->coarse_die_runtime != nullptr) {
        if (kind == JobKind::COMPUTE || kind == JobKind::DIE_COMPUTE) {
            throw std::runtime_error(
                "coarse_die_v2 COMP HBM work must be owned by its typed "
                "Plan, not the legacy scalar queue");
        }
        if (tensor_size == 0 ||
            (wlhd == nullptr && !transfer_completion)) {
            throw std::runtime_error(
                "coarse_die_v2 endpoint requires nonzero bytes and a "
                "completion owner");
        }
        const bool is_marker = kind == JobKind::CONTROL_MARKER;
        const bool is_read = kind == JobKind::COMM_READ ||
                             kind == JobKind::POOL_READ || is_marker;
        const bool is_comm = kind == JobKind::COMM_READ ||
                             kind == JobKind::COMM_WRITE;
        const bool is_pool = kind == JobKind::POOL_READ ||
                             kind == JobKind::POOL_WRITE;
        const uint64_t node_id = wlhd != nullptr
            ? wlhd->node_id : transfer_node_id;
        const int endpoint_sys_id = wlhd != nullptr
            ? wlhd->sys_id : transfer_sys_id;
        const std::string identity = std::to_string(endpoint_sys_id) + ":" +
            std::to_string(node_id) + ":" +
            std::to_string(static_cast<int>(kind));
        coarse_die::Owner owner;
        owner.request_id = coarse_endpoint_identity(std::to_string(node_id));
        owner.node_id = node_id;
        owner.execution_id = node_id;
        owner.generation = 1;
        const auto& performance =
            sys->coarse_die_runtime->hardware().input.at("performance_profile");
        const uint64_t endpoint_id = coarse_endpoint_identity(identity);
        const uint64_t transaction_id =
            coarse_endpoint_identity(std::to_string(node_id));
        nlohmann::json endpoint = {
            {"endpoint_id", endpoint_id},
            {"transaction_id", transaction_id},
            {"owner_kind", is_marker ? "die" :
                (is_comm ? "comm" : (is_pool ? "pool" : "restore"))},
            {"direction", is_read ? "read" : "write"},
            {"physical_bytes", tensor_size},
            {"path_class", is_read ? "hbm_to_edge" : "edge_to_hbm"},
            {"eta_hbm", performance.at("eta_hbm_matrix")},
            {"route", {{"internal_route", false}}},
        };
        if (is_marker) {
            endpoint["initial_delay_ns"] = sys->local_mem_latency;
        }
        const uint64_t handle = sys->coarse_die_runtime->submit_endpoint(
            endpoint, owner,
            [this, wlhd, kind,
             completion = std::move(transfer_completion)](
                const uint64_t completed_handle,
                const nlohmann::json& progress) mutable {
                coarse_jobs_.erase(completed_handle);
                const bool cancelled =
                    progress.value("state", std::string()) == "cancelled";
                const auto [read_bytes, write_bytes] =
                    coarse_hbm_read_write_bytes(progress);
                if (kind == JobKind::CONTROL_MARKER) {
                    coarse_control_marker_bytes_served_ += read_bytes;
                }
                if (cancelled) {
                    if (wlhd != nullptr) {
                        workload->abandon_hbm_endpoint_join(wlhd->node_id);
                        delete wlhd;
                    }
                    return;
                }
                if (kind == JobKind::COMM_READ) {
                    coarse_comm_read_served_ += read_bytes;
                } else if (kind == JobKind::COMM_WRITE) {
                    coarse_comm_write_served_ += write_bytes;
                } else if (kind == JobKind::RESTORE) {
                    coarse_restore_served_ += write_bytes;
                } else if (kind == JobKind::POOL_READ) {
                    coarse_pool_read_served_ += read_bytes;
                } else if (kind == JobKind::POOL_WRITE) {
                    coarse_pool_write_served_ += write_bytes;
                }
                if (completion) {
                    completion();
                } else {
                    workload->call(EventType::General, wlhd);
                }
            });
        try {
            if (!coarse_jobs_.emplace(handle, CoarseJob{wlhd, kind,
                    endpoint_sys_id, node_id, wlhd == nullptr})
                     .second) {
                throw std::runtime_error(
                    "coarse_die_v2 endpoint handle collision");
            }
        } catch (...) {
            try {
                sys->coarse_die_runtime->cancel(
                    handle, Sys::boostedTick(), false);
            } catch (...) {
                std::terminate();
            }
            throw;
        }
        return;
    }
    advance_to(Sys::boostedTick());
    if (tensor_size == 0) {
        // Zero-byte endpoints never create an HBM job (the Workload layer
        // already guards this); reaching here is a wiring bug, not a data
        // property: fail closed instead of silently stalling the node.
        fail_closed("local HBM model refused a zero-byte job");
    }
    {
        // Same-tick zombies out first (4.2 zombie fix, same as the wscllm
        // model): a job fully served by advance_to() but not yet detached
        // (its transition event still queued later in the same tick) used
        // to stay in `jobs` while a same-tick issue_job ran -- inflating
        // peak_concurrent_jobs_ and counting a phantom join against
        // redistribution_events_. Their completion callbacks fire here --
        // same tick, before this new job's membership is accounted.
        const std::vector<Job> zombies = detach_completed_jobs();
        for (const Job& job : zombies) {
            on_job_complete(job, Sys::boostedTick());
        }
    }
    // Definition-aligned counting: a join counts only when at least one
    // remaining LIVE bandwidth user exists whose share the join reshapes.
    const bool joins_active_jobs =
        live_job_count_ > 0 && drain_pending_count_ > 0;
    notify_resource_mutation_before(Sys::boostedTick());
    Job job{
        kind,
        wlhd,
        static_cast<double>(num_ops),
        static_cast<double>(tensor_size),
        static_cast<double>(sys->local_mem_latency),
        static_cast<double>(num_ops),
        static_cast<double>(tensor_size),
        static_cast<double>(sys->local_mem_latency),
        Sys::boostedTick(),
        0.0,
        0.0,
        nullptr,
        nullptr,
        std::move(transfer_completion),
        transfer_sys_id,
        transfer_node_id,
        next_issue_seq_,
        jobs_.size(),
        false};
    jobs_.push_back(std::move(job));
    slot_alive_.push_back(1);
    ++live_job_count_;
    ++next_issue_seq_;
    register_job_phase(jobs_.back());
    publish_resource_mutation(Sys::boostedTick());
    if (live_job_count_ > peak_concurrent_jobs_) {
        peak_concurrent_jobs_ = static_cast<uint64_t>(live_job_count_);
    }
    if (joins_active_jobs) {
        // Every new streamer immediately dilutes the equal split for the
        // already-active ones -- one redistribution per joining job.
        ++redistribution_events_;
    }
    schedule_next_transition();
}

void LocalHbmBandwidthModel::issue_compute(
    uint64_t num_ops,
    uint64_t tensor_size,
    WorkloadLayerHandlerData* wlhd) {
    issue_job(JobKind::COMPUTE, num_ops, tensor_size, wlhd);
}

void LocalHbmBandwidthModel::issue_control_marker(
    WorkloadLayerHandlerData* wlhd) {
    if (sys->coarse_die_runtime == nullptr) {
        issue_compute(1, 1, wlhd);
        return;
    }
    issue_job(JobKind::CONTROL_MARKER, 0, 1, wlhd);
}

void LocalHbmBandwidthModel::issue_restore(
    uint64_t tensor_size,
    WorkloadLayerHandlerData* wlhd) {
    issue_job(JobKind::RESTORE, 0, tensor_size, wlhd);
}

void LocalHbmBandwidthModel::issue_comm_read(
    uint64_t bytes,
    WorkloadLayerHandlerData* wlhd) {
    issue_job(JobKind::COMM_READ, 0, bytes, wlhd);
}

void LocalHbmBandwidthModel::issue_comm_write(
    uint64_t bytes,
    WorkloadLayerHandlerData* wlhd) {
    issue_job(JobKind::COMM_WRITE, 0, bytes, wlhd);
}

void LocalHbmBandwidthModel::issue_transfer(
    JobKind kind,
    uint64_t bytes,
    int sys_id,
    uint64_t node_id,
    std::function<void()> completion) {
    if ((kind != JobKind::COMM_READ && kind != JobKind::COMM_WRITE) ||
        !completion) {
        throw std::invalid_argument(
            "local HBM transfer requires COMM_READ/COMM_WRITE and a completion");
    }
    issue_job(kind, 0, bytes, nullptr, std::move(completion), sys_id, node_id);
}

void LocalHbmBandwidthModel::issue_pool_read(
    uint64_t bytes,
    WorkloadLayerHandlerData* wlhd) {
    issue_job(JobKind::POOL_READ, 0, bytes, wlhd);
}

void LocalHbmBandwidthModel::issue_pool_write(
    uint64_t bytes,
    WorkloadLayerHandlerData* wlhd) {
    issue_job(JobKind::POOL_WRITE, 0, bytes, wlhd);
}

void LocalHbmBandwidthModel::issue_die_memory(double service_units,
                                              double actual_bytes,
                                              Callable* sink,
                                              CallData* data) {
    // Contract §7.4: no latency phase, no FLOP leg; the drained quantity is
    // service_units. eta is derived from the served amounts
    // (actual/service) so the side-band actual-byte counter matches the
    // request triple the DieExecution received from the plan.
    if (!(service_units > 0.0) || !(actual_bytes >= 0.0) ||
        sink == nullptr || data == nullptr) {
        delete data;
        throw std::runtime_error(
            "local HBM model refused a malformed die memory request");
    }
    if (actual_bytes > 0.0 && service_units <= actual_bytes) {
        // eta <= 1 would mean the access is at least as efficient as an
        // ideal eta=1 stream -- the die calibration etas are all < 1.
        delete data;
        throw std::runtime_error(
            "local HBM model refused a die memory request with eta >= 1 "
            "(service_units must exceed actual_bytes)");
    }
    const double eta = actual_bytes / service_units;

    const Tick now = Sys::boostedTick();
    advance_to(now);
    {
        // Same-tick zombie detach as issue_job: a job fully served by
        // advance_to() but not yet detached must not inflate the membership
        // counters against this new DIE job.
        const std::vector<Job> completed = detach_completed_jobs();
        for (const Job& job : completed) {
            on_job_complete(job, now);
        }
    }
    const bool joins_active_jobs =
        live_job_count_ > 0 && drain_pending_count_ > 0;
    notify_resource_mutation_before(now);
    Job job{};
    job.kind = JobKind::DIE_COMPUTE;
    job.wlhd = nullptr;
    job.total_ops = 0.0;
    job.total_bytes = 0.0;
    job.total_latency_ns = 0.0;
    job.remaining_ops = 0.0;
    job.remaining_bytes = 0.0;
    job.memory_latency_ns = 0.0;
    job.start_tick = now;
    job.remaining_service = service_units;
    job.service_eta = eta;
    job.die_sink = sink;
    job.die_data = data;
    job.issue_seq = next_issue_seq_;
    job.slot = jobs_.size();
    jobs_.push_back(std::move(job));
    slot_alive_.push_back(1);
    ++live_job_count_;
    ++next_issue_seq_;
    register_job_phase(jobs_.back());
    publish_resource_mutation(now);
    if (live_job_count_ > peak_concurrent_jobs_) {
        peak_concurrent_jobs_ = static_cast<uint64_t>(live_job_count_);
    }
    if (joins_active_jobs) {
        ++redistribution_events_;
    }
    schedule_next_transition();
}

void LocalHbmBandwidthModel::cancel_die_jobs_of(Callable* sink) {
    const Tick now = Sys::boostedTick();
    // Account for the old share before removing this sink. Completed jobs
    // are detached before any user callback, matching the normal transition
    // path; callbacks for the sink being destroyed are reclaimed below.
    advance_to(now);
    const std::vector<Job> completed = detach_completed_jobs();
    bool has_match = false;
    for (size_t slot = 0; slot < jobs_.size(); ++slot) {
        if (slot_alive_[slot] == 0) {
            continue;
        }
        const Job& job = jobs_[slot];
        if (job.kind == JobKind::DIE_COMPUTE && job.die_sink == sink) {
            has_match = true;
            break;
        }
    }
    if (has_match) {
        notify_resource_mutation_before(now);
    }
    bool removed = false;
    for (size_t slot = 0; slot < jobs_.size(); ++slot) {
        if (slot_alive_[slot] == 0) {
            continue;
        }
        Job& job = jobs_[slot];
        if (job.kind == JobKind::DIE_COMPUTE && job.die_sink == sink) {
            delete job.die_data;
            job.die_data = nullptr;
            if (remaining_drain(job) > kEpsilon) {
                --drain_pending_count_;
            }
            job.in_candidates = false;
            unregister_job_phase(slot);
            slot_alive_[slot] = 0;
            --live_job_count_;
            ++dead_slot_count_;
            removed = true;
        }
    }
    if (removed) {
        completion_candidates_.erase(
            std::remove_if(completion_candidates_.begin(),
                           completion_candidates_.end(),
                           [this](const size_t slot) {
                               return slot >= slot_alive_.size() ||
                                      slot_alive_[slot] == 0;
                           }),
            completion_candidates_.end());
        publish_resource_mutation(now);
    }
    for (const Job& job : completed) {
        if (job.kind == JobKind::DIE_COMPUTE && job.die_sink == sink) {
            delete job.die_data;
            continue;
        }
        on_job_complete(job, now);
    }
    // A wiped job may have been the next transition's reason; re-arm the
    // prediction for whatever remains (a no-op when nothing is left).
    schedule_next_transition();
    maybe_compact_slots();
}

void LocalHbmBandwidthModel::call(EventType, CallData* data) {
    ++transition_event_count_;
    TransitionData* transition = static_cast<TransitionData*>(data);
    const uint64_t generation = transition->generation;
    delete transition;
    if (generation != event_generation) {
        return;
    }
    // Sys popped this callback before invoking us, so it is no longer
    // cancellable. Forget the consumed handle before a completion callback can
    // install the next prediction. A defensive stale-generation callback must
    // leave a newer handle untouched.
    scheduled_transition_event.reset();

    const Tick now = Sys::boostedTick();
    advance_to(now);
    // v7: only completion candidates (jobs whose quantities landed at a
    // boundary since the last detach) can satisfy complete(); the old
    // any_of scan over the whole vector visits exactly the same jobs.
    std::vector<size_t> candidates = std::move(completion_candidates_);
    completion_candidates_.clear();
    bool has_completed = false;
    for (const size_t slot : candidates) {
        if (slot_alive_[slot] != 0 && complete(jobs_[slot])) {
            has_completed = true;
            break;
        }
    }
    if (has_completed) {
        notify_resource_mutation_before(now);
    }
    std::vector<Job> completed;
    for (const size_t slot : candidates) {
        if (slot_alive_[slot] == 0) {
            continue;
        }
        Job& job = jobs_[slot];
        job.in_candidates = false;
        if (!complete(job)) {
            continue;
        }
        completed.push_back(std::move(job));
        slot_alive_[slot] = 0;
        --live_job_count_;
        ++dead_slot_count_;
        if (remaining_drain(completed.back()) > kEpsilon) {
            --drain_pending_count_;
        }
        unregister_job_phase(slot);
    }
    if (!completed.empty()) {
        publish_resource_mutation(now);
    }

    // 中-4④ unified batch-level rule (face semantics): the whole completion
    // batch leaves the survivors' shares re-split (full_rate/N ->
    // full_rate/(N-k)); the batch counts ONCE regardless of how many jobs
    // completed together (previously each completed job counted separately,
    // diverging from the other four repos).
    if (!completed.empty() && drain_pending_count_ > 0) {
        ++redistribution_events_;
    }

    for (const Job& job : completed) {
        on_job_complete(job, now);
    }

    // A completion callback may immediately issue the next job (compute,
    // restore, comm, or pool), which schedules a transition and advances
    // the generation.  Do not invalidate that event by scheduling the same
    // state a second time.
    if (event_generation == generation) {
        schedule_next_transition();
    }
    maybe_compact_slots();
}

std::vector<LocalHbmBandwidthModel::Job>
LocalHbmBandwidthModel::detach_completed_jobs() {
    // 4.2 zombie fix (same as the wscllm model): see issue_job.
    std::vector<size_t> candidates = std::move(completion_candidates_);
    completion_candidates_.clear();
    bool has_completed = false;
    for (const size_t slot : candidates) {
        if (slot_alive_[slot] != 0 && complete(jobs_[slot])) {
            has_completed = true;
            break;
        }
    }
    if (has_completed) {
        notify_resource_mutation_before(Sys::boostedTick());
    }
    std::vector<Job> completed;
    for (const size_t slot : candidates) {
        if (slot_alive_[slot] == 0) {
            continue;
        }
        Job& job = jobs_[slot];
        job.in_candidates = false;
        if (!complete(job)) {
            continue;
        }
        completed.push_back(std::move(job));
        slot_alive_[slot] = 0;
        --live_job_count_;
        ++dead_slot_count_;
        if (remaining_drain(completed.back()) > kEpsilon) {
            --drain_pending_count_;
        }
        unregister_job_phase(slot);
    }
    if (!completed.empty()) {
        publish_resource_mutation(Sys::boostedTick());
    }
    return completed;
}

void LocalHbmBandwidthModel::maybe_compact_slots() {
    // Bulk dead-slot reclamation at a safe point (no observer pass active,
    // advance_to not running, no iteration in flight). Order-preserving:
    // every per-job apply loop keeps the original issue order afterwards.
    if (dead_slot_count_ < 16 || dead_slot_count_ < live_job_count_) {
        return;
    }
    std::vector<Job> alive;
    std::vector<uint8_t> alive_flags;
    alive.reserve(live_job_count_);
    alive_flags.reserve(live_job_count_);
    for (size_t slot = 0; slot < jobs_.size(); ++slot) {
        if (slot_alive_[slot] == 0) {
            continue;
        }
        alive.push_back(std::move(jobs_[slot]));
        alive_flags.push_back(1);
    }
    jobs_.swap(alive);
    slot_alive_.swap(alive_flags);
    dead_slot_count_ = 0;
    latency_members_.clear();
    stream_members_.clear();
    compute_members_.clear();
    completion_candidates_.clear();
    for (size_t slot = 0; slot < jobs_.size(); ++slot) {
        Job& job = jobs_[slot];
        job.slot = slot;
        const bool was_candidate = job.in_candidates;
        job.in_candidates = false;
        if (job.memory_latency_ns > kEpsilon) {
            phase_insert(latency_members_, slot);
        } else if (remaining_drain(job) > kEpsilon) {
            phase_insert(stream_members_, slot);
        }
        if (job.kind == JobKind::COMPUTE) {
            phase_insert(compute_members_, slot);
        }
        if (was_candidate || complete(job)) {
            mark_completion_candidate(job);
        }
    }
}

void LocalHbmBandwidthModel::on_job_complete(const Job& job,
                                             Tick now) {
    const uint64_t elapsed = now - job.start_tick;
    if (job.kind == JobKind::DIE_COMPUTE) {
        // Contract §7.4/§6.3: the sink is the DieExecution (an internal
        // memory stage of a still-running outer node -- NOT an outer COMP
        // completion). The DieExecution accounts the outer node's GPU
        // occupancy itself at its own completion; this callback only hands
        // back the memory port. No tics, no Workload layer, no join.
        CallData* const data = job.die_data;
        job.die_sink->call(EventType::General, data);
        return;
    }
    if (job.transfer_completion) {
        job.transfer_completion();
        return;
    }
    // RESTORE / COMM_* / POOL_* jobs are HBM transfers and do not count
    // toward the GPU busy-tick accumulator (the old sibling
    // tics_hbm_dma_ops counter was write-only and was removed).
    if (job.kind == JobKind::COMPUTE) {
        workload->hw_resource->tics_gpu_ops += elapsed;
    }
    job.wlhd->workload->call(EventType::General, job.wlhd);
}
