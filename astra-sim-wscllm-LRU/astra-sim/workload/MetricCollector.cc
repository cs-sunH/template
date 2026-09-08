/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/workload/MetricCollector.hh"

#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/LocalHbmBandwidthModel.hh"
#include "astra-sim/workload/Statistics.hh"
#include "astra-sim/workload/Workload.hh"

#include <json/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <unistd.h>

using namespace AstraSim;
using json = nlohmann::json;

namespace {

// Utilization values may only exceed 1.0 by a tiny floating-point epsilon
// (doc sec.12.6); raw numerator/denominator are always printed alongside.
constexpr double kUtilEpsilon = 1e-6;

[[noreturn]] void fatal_metrics_error(const std::string& message) {
    std::cerr << "[METRIC][ERROR] " << message << std::endl;
    exit(EXIT_FAILURE);
}

[[noreturn]] void fatal_metric_write(const int error_number) noexcept {
    if (error_number == 0) {
        std::fprintf(stderr,
                     "[METRIC][FATAL] write to stdout made no progress\n");
    } else {
        std::fprintf(stderr,
                     "[METRIC][FATAL] write to stdout failed: %s (errno=%d)\n",
                     std::strerror(error_number), error_number);
    }
    // Do not route this through MetricCollector or buffered iostreams: a
    // write-channel failure must fail closed without recursively emitting.
    ::_exit(EXIT_FAILURE);
}

void write_all_or_die(const char* bytes, size_t remaining) noexcept {
    while (remaining > 0) {
        const ssize_t written = ::write(STDOUT_FILENO, bytes, remaining);
        if (written > 0) {
            bytes += written;
            remaining -= static_cast<size_t>(written);
            continue;
        }
        if (written < 0 && errno == EINTR) {
            continue;
        }
        fatal_metric_write(written == 0 ? 0 : errno);
    }
}

std::string u128_to_string(unsigned __int128 value) {
    if (value == 0) {
        return "0";
    }
    std::string digits;
    while (value > 0) {
        digits.push_back(static_cast<char>('0' + value % 10));
        value /= 10;
    }
    std::reverse(digits.begin(), digits.end());
    return digits;
}

long double u128_to_long_double(unsigned __int128 value) {
    const long double hi =
        static_cast<long double>(static_cast<uint64_t>(value >> 64));
    const long double lo =
        static_cast<long double>(static_cast<uint64_t>(value));
    return hi * 18446744073709551616.0L + lo;
}

// Nearest-rank percentile (doc sec.3.5): index = ceil(p * N) - 1.
Tick nearest_rank_percentile(const std::vector<Tick>& sorted, double p) {
    const size_t n = sorted.size();
    size_t index = static_cast<size_t>(std::ceil(p * static_cast<double>(n)));
    if (index == 0) {
        index = 1;
    }
    if (index > n) {
        index = n;
    }
    return sorted[index - 1];
}

json tick_or_null(const std::optional<Tick>& value) {
    if (value.has_value()) {
        return json(value.value());
    }
    return json(nullptr);
}

}  // namespace

MetricCollector& MetricCollector::instance() {
    static MetricCollector collector;
    return collector;
}

MetricCollector::~MetricCollector() {
    // Destructors cannot report a failed close safely.  All operational close
    // paths use close_memory_anchor_spool_or_die(); this is a final guard for
    // abnormal test/lifecycle exits before finalize().
    if (this->memory_anchor_spool_ != nullptr) {
        std::fclose(this->memory_anchor_spool_);
        this->memory_anchor_spool_ = nullptr;
    }
}

void MetricCollector::reset_for_initialize() {
    // initialize() is documented as one-shot in production, but the singleton
    // is also used by focused fixtures.  Drain a prior buffered record and
    // close the old anonymous file before dropping every state member so a
    // repeated initialize has neither stale anchors nor a leaked descriptor.
    flush_emit_buffer();
    close_memory_anchor_spool_or_die();
    *this = MetricCollector();
}

void MetricCollector::ensure_online_memory_anchor_spool() {
    if (this->memory_anchor_spool_ != nullptr) {
        return;
    }
    errno = 0;
    this->memory_anchor_spool_ = std::tmpfile();
    if (this->memory_anchor_spool_ == nullptr) {
        const int error_number = errno;
        fatal_metrics_error(
            "cannot create anonymous online memory-anchor spool: " +
            std::string(error_number == 0 ? "unknown error"
                                          : std::strerror(error_number)));
    }
}

void MetricCollector::close_memory_anchor_spool_or_die() {
    if (this->memory_anchor_spool_ == nullptr) {
        return;
    }
    std::FILE* const spool = this->memory_anchor_spool_;
    this->memory_anchor_spool_ = nullptr;
    if (std::fclose(spool) != 0) {
        const int error_number = errno;
        fatal_metrics_error(
            "failed to close anonymous online memory-anchor spool: " +
            std::string(error_number == 0 ? "unknown error"
                                          : std::strerror(error_number)));
    }
}

void MetricCollector::rebuild_online_transfer_anchor_interest() {
    this->online_transfer_anchor_keys_needed_.clear();
    this->online_transfer_anchor_subjects_needed_.clear();
    this->online_transfer_anchor_ticks_.clear();
    this->online_transfer_anchor_by_subject_.clear();
    for (const auto& action : this->memory_actions_) {
        if (action.anchor_kind != "transfer_complete" ||
            action.trigger_queue_index < 0) {
            continue;
        }
        this->online_transfer_anchor_subjects_needed_.insert(
            action.trigger_queue_index);
        this->online_transfer_anchor_keys_needed_.insert(
            TransferAnchorKey{action.trigger_queue_index, action.rank});
    }
}

void MetricCollector::append_online_memory_anchor(
    const MemoryAnchorTick& anchor) {
    ensure_online_memory_anchor_spool();
    if (std::fwrite(&anchor, sizeof(anchor), 1, this->memory_anchor_spool_) !=
        1) {
        const int error_number = errno;
        fatal_metrics_error(
            "failed to write anonymous online memory-anchor spool: " +
            std::string(error_number == 0 ? "unknown error"
                                          : std::strerror(error_number)));
    }
    if (this->memory_anchor_spool_record_count_ ==
        std::numeric_limits<uint64_t>::max()) {
        fatal_metrics_error("online memory-anchor spool record count overflow");
    }
    this->memory_anchor_spool_record_count_++;

    // Keep only the two maxima required by transfer_complete resolution.
    // Events for subjects with no transfer action remain in the spool for
    // output, but intentionally consume no resident index memory.
    if (this->online_transfer_anchor_subjects_needed_.count(
            anchor.subject_id) == 0) {
        return;
    }
    const auto subject_it =
        this->online_transfer_anchor_by_subject_.find(anchor.subject_id);
    if (subject_it == this->online_transfer_anchor_by_subject_.end() ||
        anchor.tick > subject_it->second) {
        this->online_transfer_anchor_by_subject_[anchor.subject_id] =
            anchor.tick;
    }

    const TransferAnchorKey key{anchor.subject_id, anchor.rank};
    if (this->online_transfer_anchor_keys_needed_.count(key) == 0) {
        return;
    }
    const auto rank_it = this->online_transfer_anchor_ticks_.find(key);
    if (rank_it == this->online_transfer_anchor_ticks_.end() ||
        anchor.tick > rank_it->second) {
        this->online_transfer_anchor_ticks_[key] = anchor.tick;
    }
}

void MetricCollector::emit_memory_anchor_record(
    const MemoryAnchorTick& anchor) const {
    json record;
    record["schema"] = 1;
    record["type"] = "memory_anchor";
    record["source"] = "simulator";
    record["repo_variant"] = this->repo_variant_;
    record["run_id"] = this->run_id_;
    record["subject_id"] = anchor.subject_id;
    record["rank"] = anchor.rank;
    record["node_id"] = anchor.node_id;
    record["tick_ns"] = anchor.tick;
    emit_record(record.dump());
}

void MetricCollector::emit_online_memory_anchor_spool_records() {
    if (this->memory_anchor_spool_ == nullptr) {
        return;
    }
    if (std::fflush(this->memory_anchor_spool_) != 0) {
        const int error_number = errno;
        fatal_metrics_error(
            "failed to flush anonymous online memory-anchor spool: " +
            std::string(error_number == 0 ? "unknown error"
                                          : std::strerror(error_number)));
    }
    if (std::fseek(this->memory_anchor_spool_, 0, SEEK_SET) != 0) {
        const int error_number = errno;
        fatal_metrics_error(
            "failed to rewind anonymous online memory-anchor spool: " +
            std::string(error_number == 0 ? "unknown error"
                                          : std::strerror(error_number)));
    }
    for (;;) {
        MemoryAnchorTick anchor{};
        const size_t bytes = std::fread(&anchor, 1, sizeof(anchor),
                                        this->memory_anchor_spool_);
        if (bytes == sizeof(anchor)) {
            emit_memory_anchor_record(anchor);
            continue;
        }
        if (bytes == 0 && std::feof(this->memory_anchor_spool_) != 0) {
            break;
        }
        std::string message =
            "failed to read anonymous online memory-anchor spool";
        if (std::ferror(this->memory_anchor_spool_) != 0) {
            const int error_number = errno;
            message += ": ";
            message += error_number == 0 ? "unknown error"
                                         : std::strerror(error_number);
        } else {
            message += ": truncated record";
        }
        fatal_metrics_error(message);
    }
}

void MetricCollector::initialize(const std::string& manifest_path,
                                 const std::string& detail_level) {
    reset_for_initialize();
    if (detail_level != "off" && detail_level != "summary" &&
        detail_level != "full") {
        fatal_metrics_error("invalid --metrics-detail value: " + detail_level +
                            " (expected off|summary|full)");
    }
    this->detail_level_ = detail_level;
    if (detail_level == "off") {
        // Metrics disabled: do not touch the filesystem at all, so the
        // simulation behavior and output stay byte-identical (doc sec.12.5).
        return;
    }
    if (manifest_path.empty() || manifest_path == "empty") {
        fatal_metrics_error(
            "--metrics-detail=" + detail_level +
            " requires --metrics-configuration=<manifest path>");
    }
    load_manifest(manifest_path);
    this->enabled_ = true;

    json init_record;
    init_record["schema"] = 1;
    init_record["type"] = "init";
    init_record["source"] = "simulator";
    init_record["repo_variant"] = this->repo_variant_;
    init_record["run_mode"] = this->run_mode_;
    init_record["run_id"] = this->run_id_;
    init_record["detail_level"] = this->detail_level_;
    init_record["manifest_path"] = manifest_path;
    init_record["schema_version"] = this->schema_version_;
    init_record["input_requests"] = this->requests_.size();
    if (this->trace_digest_.has_value()) {
        // The C++ collector cannot recompute the Python-side ET digest
        // recipe; doc sec.10 requires the run script to verify it before
        // launching. Surface the value and the verification status
        // explicitly in the log instead of silently accepting it.
        init_record["trace_digest"] = this->trace_digest_.value();
        init_record["trace_digest_verified"] = false;
        init_record["trace_digest_note"] =
            "not recomputed by the C++ collector; must be verified by the run "
            "script against the current ET before launch (doc sec.10)";
    }
    if (this->request_mapping_digest_.has_value()) {
        init_record["request_mapping_digest"] =
            this->request_mapping_digest_.value();
    }
    if (this->kv_event_digest_.has_value()) {
        init_record["kv_event_digest"] = this->kv_event_digest_.value();
    }
    // SLO sampling anchors echo (CPP_SPEC §A): the values actually in
    // effect, plus the fallback warning when the manifest carried no
    // usable slo_sampling node (never fatal).
    json slo_echo;
    slo_echo["watermark_period_ns"] = this->watermark_period_ns_;
    slo_echo["link_bucket_ns"] = this->link_bucket_ns_;
    slo_echo["provisional"] = this->slo_sampling_provisional_;
    slo_echo["source"] = this->slo_sampling_from_manifest_
        ? "manifest:slo_sampling"
        : "provisional-default:5000000ns";
    if (!this->slo_sampling_from_manifest_) {
        slo_echo["warning"] =
            "manifest has no slo_sampling node; using the documented "
            "provisional anchor (5,000,000 ns) for the WP8 watermark period "
            "and the WP6 link bucket; batch B4 replaces both with derived "
            "values";
    }
    init_record["slo_sampling"] = slo_echo;
    fflush(stdout);
    emit_record(init_record.dump());
}

void MetricCollector::load_manifest(const std::string& manifest_path) {
    std::ifstream manifest_file(manifest_path);
    if (!manifest_file.is_open()) {
        fatal_metrics_error("cannot open metrics manifest: " + manifest_path);
    }
    json manifest;
    try {
        manifest_file >> manifest;
    } catch (const std::exception& e) {
        fatal_metrics_error("failed to parse metrics manifest " +
                            manifest_path + ": " + e.what());
    }

    this->schema_version_ = manifest.value("schema_version", 0);
    if (this->schema_version_ != 1) {
        // Doc sec.12.1: unsupported schema versions must be rejected.
        fatal_metrics_error(
            "unsupported metrics manifest schema_version: " +
            std::to_string(this->schema_version_) + " (expected 1)");
    }
    this->repo_variant_ = manifest.value("repo_variant", "unknown");
    this->run_mode_ = manifest.value("run_mode", "service");
    this->run_id_ = manifest.value("run_id", "");
    if (manifest.contains("npus_count") && manifest["npus_count"].is_number()) {
        this->manifest_npus_count_ =
            manifest["npus_count"].get<int64_t>();
    }
    if (manifest.contains("trace_digest") &&
        manifest["trace_digest"].is_string()) {
        this->trace_digest_ = manifest["trace_digest"].get<std::string>();
    }
    if (manifest.contains("request_mapping_digest") &&
        manifest["request_mapping_digest"].is_string()) {
        this->request_mapping_digest_ =
            manifest["request_mapping_digest"].get<std::string>();
    }
    if (manifest.contains("kv_event_digest") &&
        manifest["kv_event_digest"].is_string()) {
        this->kv_event_digest_ =
            manifest["kv_event_digest"].get<std::string>();
    }

    // SLO pipeline sampling anchors (CPP_SPEC §A): the manifest's
    // ``slo_sampling`` node carries the WP8 watermark period and the WP6
    // link bucket length. A missing node or null values fall back to the
    // documented provisional anchor (5,000,000 ns; coordinator ruling
    // 2026-08-26 -- 1 ms measured 31,895 bucket records on the S3 2s
    // window) with provisional=true; this never fails the run (batch B4
    // replaces the anchors with derived values, and every related record
    // echoes the period actually used so the fallback is
    // downstream-visible).
    if (manifest.contains("slo_sampling") &&
        manifest["slo_sampling"].is_object()) {
        const json& slo = manifest["slo_sampling"];
        const auto read_ns = [&slo](const char* key) -> std::optional<uint64_t> {
            if (!slo.contains(key) || !slo[key].is_number()) {
                return std::nullopt;
            }
            const int64_t value = slo[key].get<int64_t>();
            if (value <= 0) {
                return std::nullopt;
            }
            return static_cast<uint64_t>(value);
        };
        const auto watermark = read_ns("watermark_period_ns");
        const auto link_bucket = read_ns("link_bucket_ns");
        if (watermark.has_value()) {
            this->watermark_period_ns_ = watermark.value();
        }
        if (link_bucket.has_value()) {
            this->link_bucket_ns_ = link_bucket.value();
        }
        this->slo_sampling_from_manifest_ = true;
        this->slo_sampling_provisional_ =
            !watermark.has_value() || !link_bucket.has_value() ||
            (slo.contains("provisional") && slo["provisional"].is_boolean() &&
             slo["provisional"].get<bool>());
    } else {
        this->slo_sampling_from_manifest_ = false;
        this->slo_sampling_provisional_ = true;
    }

    // Requests (doc sec.4.2). queue_index must be unique (doc sec.4.5).
    const json requests = manifest.value("requests", json::array());
    for (const auto& entry : requests) {
        RequestMetricState state;
        try {
            state.queue_index = entry.at("queue_index").get<int64_t>();
            state.request_id = entry.value(
                "request_id", "q" + std::to_string(state.queue_index));
            state.session_id = entry.value("session_id", "");
            state.turn_index = entry.value("turn_index", 0);
            state.prefill_instance = entry.value("prefill_instance", -1);
            state.decode_instance = entry.value("decode_instance", -1);
            for (const auto& rank : entry.at("prefill_ranks")) {
                state.prefill_ranks.insert(rank.get<int>());
            }
            for (const auto& rank : entry.at("decode_ranks")) {
                state.decode_ranks.insert(rank.get<int>());
            }
            const auto& arrival = entry.at("arrival");
            const std::string kind = arrival.at("kind").get<std::string>();
            if (kind == "absolute") {
                state.arrival.kind = ArrivalSpec::Kind::ABSOLUTE;
                state.arrival.value_ns =
                    arrival.at("value_ns").get<Tick>();
            } else if (kind == "after_request") {
                state.arrival.kind = ArrivalSpec::Kind::AFTER_REQUEST;
                state.arrival.parent_queue_index =
                    arrival.at("parent_queue_index").get<int64_t>();
                state.arrival.interval_ns =
                    arrival.at("interval_ns").get<Tick>();
            } else {
                fatal_metrics_error("unknown arrival kind: " + kind);
            }
            // WP9 (CPP_SPEC §C): optional request decode length for the
            // first-token invariant; synthetic online manifests may omit
            // it, in which case finalize skips the invariant and notes
            // manifest_decode_length_missing.
            if (entry.contains("decode_length") &&
                entry["decode_length"].is_number()) {
                state.decode_length =
                    entry["decode_length"].get<int64_t>();
            }
        } catch (const std::exception& e) {
            fatal_metrics_error(std::string("malformed request entry in ") +
                                manifest_path + ": " + e.what());
        }
        if (this->request_index_by_queue_index_.count(state.queue_index) > 0) {
            fatal_metrics_error("duplicate queue_index in manifest: " +
                                std::to_string(state.queue_index));
        }
        this->request_index_by_queue_index_[state.queue_index] =
            this->requests_.size();
        this->requests_.push_back(std::move(state));
    }

    // Sparse node events (doc sec.4.3): [node_id, event_code, subject_id].
    const json events_by_rank =
        manifest.value("node_events_by_rank", json::object());
    for (auto it = events_by_rank.begin(); it != events_by_rank.end(); ++it) {
        int rank;
        try {
            rank = std::stoi(it.key());
        } catch (const std::exception&) {
            fatal_metrics_error("non-integer rank key in node_events_by_rank: "
                                + it.key());
        }
        for (const auto& triple : it.value()) {
            if (!triple.is_array() || triple.size() != 3) {
                fatal_metrics_error(
                    "node event must be a [node_id, event_code, subject_id] "
                    "triple");
            }
            const uint64_t node_id = triple[0].get<uint64_t>();
            const uint8_t event_code = triple[1].get<uint8_t>();
            const int64_t subject_id = triple[2].get<int64_t>();
            if (event_code < 1 || event_code > 8) {
                fatal_metrics_error("invalid node event code: " +
                                    std::to_string(event_code));
            }
            NodeMetricEvent event{event_code, subject_id};
            const bool is_issue_edge =
                (event_code ==
                 static_cast<uint8_t>(EventCode::PREFILL_START_ISSUE)) ||
                (event_code ==
                 static_cast<uint8_t>(EventCode::DECODE_START_ISSUE)) ||
                (event_code ==
                 static_cast<uint8_t>(EventCode::ITERATION_START_ISSUE));
            auto& event_map =
                is_issue_edge ? this->issue_events_ : this->complete_events_;
            event_map[rank][node_id].push_back(event);
        }
    }

    // Planner memory ledger deltas (doc sec.7.1/7.8). Anchors stay symbolic
    // here; they are resolved to actual ASTRA ticks at finalize time.
    const json memory_actions = manifest.value("memory_actions", json::array());
    for (const auto& entry : memory_actions) {
        MemoryAction action;
        try {
            action.sequence_index = entry.value("sequence_index", int64_t(0));
            action.anchor_kind = entry.at("anchor_kind").get<std::string>();
            action.anchor_quality =
                entry.value("anchor_quality", std::string("unknown"));
            if (entry.contains("trigger_queue_index") &&
                !entry["trigger_queue_index"].is_null()) {
                action.trigger_queue_index =
                    entry["trigger_queue_index"].get<int64_t>();
            }
            action.rank = entry.at("rank").get<int>();
            action.weight_delta_bytes =
                entry.value("weight_delta_bytes", int64_t(0));
            action.resident_kv_delta_bytes =
                entry.value("resident_kv_delta_bytes", int64_t(0));
            action.reserved_kv_delta_bytes =
                entry.value("reserved_kv_delta_bytes", int64_t(0));
            action.cause = entry.value("cause", std::string(""));
        } catch (const std::exception& e) {
            fatal_metrics_error(std::string("malformed memory action in ") +
                                manifest_path + ": " + e.what());
        }
        this->memory_actions_.push_back(std::move(action));
    }

    // Planner memory peaks (doc sec.7.7), kept raw for passthrough with
    // annotations at finalize.
    const json peaks = manifest.value("planner_memory_peaks", json::array());
    for (const auto& entry : peaks) {
        this->planner_memory_peaks_.push_back(entry);
    }

    // Microbenchmark point descriptor (doc sec.8 extension key). Exactly one
    // point per ET/run, so the point's active-rank totals are the iteration
    // numerators; no windowed attribution is needed.
    if (manifest.contains("microbenchmark") &&
        manifest["microbenchmark"].is_object()) {
        const json& point = manifest["microbenchmark"];
        MicrobenchPointSpec spec;
        try {
            spec.benchmark_point_id =
                point.at("benchmark_point_id").get<int64_t>();
            spec.tp_degree = point.at("tp_degree").get<int64_t>();
            for (const auto& rank : point.at("ranks")) {
                spec.active_ranks.push_back(rank.get<int>());
            }
            spec.raw = point;
        } catch (const std::exception& e) {
            fatal_metrics_error(std::string("malformed microbenchmark point "
                                            "in ") +
                                manifest_path + ": " + e.what());
        }
        this->microbench_point_ = std::move(spec);
    }
}

void MetricCollector::on_node_issue(int rank, uint64_t node_id, Tick tick) {
    const auto rank_it = this->issue_events_.find(rank);
    if (rank_it == this->issue_events_.end()) {
        return;
    }
    const auto node_it = rank_it->second.find(node_id);
    if (node_it == rank_it->second.end()) {
        return;
    }
    for (const auto& event : node_it->second) {
        apply_event(event, rank, node_id, tick);
    }
    // R2 one-shot erase (frozen plan §4-B2, 2026-08-29): every
    // (rank, node) edge fires at most once (GraphSource::take_node consumes
    // the free set, online store ids are never reused, and anchor
    // registration (Phase B-1.5) precedes the issue pass (B-5)), so this
    // bucket's routing duty is complete once applied. Erase it immediately;
    // drop the rank bucket when it empties. Only the ROUTING tables are
    // erased: apply_event has already copied everything finalize consumes
    // (memory_anchor_ticks_ / spool + transfer maxima / requests_ /
    // request_boundaries_ / first_token_ticks_ / iterations_) into its own
    // derived storage.
    rank_it->second.erase(node_it);
    if (rank_it->second.empty()) {
        this->issue_events_.erase(rank_it);
    }
}

void MetricCollector::on_node_complete(int rank, uint64_t node_id, Tick tick) {
    const auto rank_it = this->complete_events_.find(rank);
    if (rank_it == this->complete_events_.end()) {
        return;
    }
    const auto node_it = rank_it->second.find(node_id);
    if (node_it == rank_it->second.end()) {
        return;
    }
    for (const auto& event : node_it->second) {
        apply_event(event, rank, node_id, tick);
    }
    // R2 one-shot erase: see on_node_issue.
    rank_it->second.erase(node_it);
    if (rank_it->second.empty()) {
        this->complete_events_.erase(rank_it);
    }
}

void MetricCollector::apply_event(const NodeMetricEvent& event, int rank,
                                  uint64_t node_id, Tick tick) {
    const auto code = static_cast<EventCode>(event.event_code);
    switch (code) {
    case EventCode::PREFILL_START_ISSUE:
    case EventCode::PREFILL_END_COMPLETE:
    case EventCode::DECODE_START_ISSUE:
    case EventCode::COMPLETION_COMPLETE: {
        const auto req_it =
            this->request_index_by_queue_index_.find(event.subject_id);
        if (req_it == this->request_index_by_queue_index_.end()) {
            this->dropped_events_++;
            return;
        }
        RequestMetricState& state = this->requests_[req_it->second];
        switch (code) {
        case EventCode::PREFILL_START_ISSUE:
            if (!state.prefill_start.has_value() ||
                tick < state.prefill_start.value()) {
                state.prefill_start = tick;
            }
            state.prefill_start_ranks_seen.insert(rank);
            break;
        case EventCode::PREFILL_END_COMPLETE:
            if (!state.prefill_end.has_value() ||
                tick > state.prefill_end.value()) {
                state.prefill_end = tick;
            }
            state.prefill_end_ranks_seen.insert(rank);
            break;
        case EventCode::DECODE_START_ISSUE:
            if (!state.decode_start.has_value() ||
                tick < state.decode_start.value()) {
                state.decode_start = tick;
            }
            state.decode_start_ranks_seen.insert(rank);
            break;
        case EventCode::COMPLETION_COMPLETE:
            if (!state.completion.has_value() ||
                tick > state.completion.value()) {
                state.completion = tick;
            }
            state.completion_ranks_seen.insert(rank);
            break;
        default:
            break;
        }
        return;
    }
    case EventCode::ITERATION_START_ISSUE:
    case EventCode::ITERATION_END_COMPLETE: {
        IterationMetricState& state = this->iterations_[event.subject_id];
        state.benchmark_point_id = event.subject_id;
        if (code == EventCode::ITERATION_START_ISSUE) {
            if (!state.iteration_start.has_value() ||
                tick < state.iteration_start.value()) {
                state.iteration_start = tick;
            }
            state.start_ranks_seen.insert(rank);
        } else {
            if (!state.iteration_end.has_value() ||
                tick > state.iteration_end.value()) {
                state.iteration_end = tick;
            }
            state.end_ranks_seen.insert(rank);
        }
        return;
    }
    case EventCode::MEMORY_ANCHOR_COMPLETE: {
        const MemoryAnchorTick anchor{event.subject_id, rank, node_id, tick};
        if (this->online_mode_) {
            append_online_memory_anchor(anchor);
        } else {
            this->memory_anchor_ticks_.push_back(anchor);
        }
        return;
    }
    case EventCode::FIRST_TOKEN_COMPLETE: {
        // WP9 (WP9_CONTRACT §1): min tick across ranks and re-registrations;
        // unknown subjects count as dropped like every other request event.
        const auto req_it =
            this->request_index_by_queue_index_.find(event.subject_id);
        if (req_it == this->request_index_by_queue_index_.end()) {
            this->dropped_events_++;
            return;
        }
        const auto existing = this->first_token_ticks_.find(event.subject_id);
        if (existing == this->first_token_ticks_.end() ||
            tick < existing->second) {
            this->first_token_ticks_[event.subject_id] = tick;
        }
        return;
    }
    }
}

void MetricCollector::on_compute_issue(int rank, uint64_t num_ops,
                                       uint64_t local_tensor_bytes) {
    RankMetricState& state = this->rank_states_[rank];
    state.total_num_ops += num_ops;
    state.total_local_bytes += local_tensor_bytes;
}

void MetricCollector::on_local_hbm_restore_issue(int rank, uint64_t bytes) {
    this->rank_states_[rank].local_hbm_restore_bytes += bytes;
}

// ---------------------------------------------------------------------------
// Phase-7 §10.3: online dynamic anchor registration (see the header comment).
// All entry points no-op when metrics are disabled; none of them touches the
// static manifest path (the static main never calls them).
// ---------------------------------------------------------------------------

void MetricCollector::online_register_request(
    int64_t queue_index, const std::string& request_id,
    const std::string& session_id, int64_t turn_index, bool absolute_arrival,
    Tick arrival_value_ns, int64_t arrival_parent_queue_index,
    Tick arrival_interval_ns) {
    if (!this->enabled_) {
        return;
    }
    const auto existing = this->request_index_by_queue_index_.find(queue_index);
    if (existing == this->request_index_by_queue_index_.end()) {
        // Fresh dynamic request (the manifest already covered the static
        // requests; anything not covered is an online-injected one).
        RequestMetricState state;
        state.queue_index = queue_index;
        state.request_id = request_id;
        state.session_id = session_id;
        state.turn_index = turn_index;
        state.arrival.kind = absolute_arrival
                                 ? ArrivalSpec::Kind::ABSOLUTE
                                 : ArrivalSpec::Kind::AFTER_REQUEST;
        state.arrival.value_ns = arrival_value_ns;
        state.arrival.parent_queue_index = arrival_parent_queue_index;
        state.arrival.interval_ns = arrival_interval_ns;
        this->request_index_by_queue_index_[queue_index] = this->requests_.size();
        this->request_index_by_request_id_[request_id] = this->requests_.size();
        this->requests_.push_back(std::move(state));
        return;
    }
    // Idempotent re-registration: the online CSV data is authoritative for
    // the arrival/session fields; overwrite them (the manifest values are
    // planner-side, the CSV row is the actual online input).
    RequestMetricState& state = this->requests_[existing->second];
    state.request_id = request_id;
    state.session_id = session_id;
    state.turn_index = turn_index;
    state.arrival.kind = absolute_arrival ? ArrivalSpec::Kind::ABSOLUTE
                                          : ArrivalSpec::Kind::AFTER_REQUEST;
    state.arrival.value_ns = arrival_value_ns;
    state.arrival.parent_queue_index = arrival_parent_queue_index;
    state.arrival.interval_ns = arrival_interval_ns;
    this->request_index_by_request_id_[request_id] = existing->second;
}

void MetricCollector::clear_static_node_events() {
    // Online mode only (see the header comment): the manifest's static
    // node-event tables are keyed by the offline ET node-id space, which
    // overlaps the online per-rank id space and would fire on unrelated
    // online nodes. Clearing keeps the dynamic anchors as the sole event
    // source. Requests/arrivals/memory actions stay loaded (the manifest
    // request list is authoritative for arrivals; anchors are not).
    if (!this->enabled_) {
        return;
    }
    this->issue_events_.clear();
    this->complete_events_.clear();
    // D3 (2026-08-28): latch online mode for emit_watermark_records (the
    // hbm_watermark bucket lines are the documented all-zero series there
    // -- see the header comment on online_mode_).
    if (!this->online_mode_) {
        this->online_mode_ = true;
        rebuild_online_transfer_anchor_interest();
    }
    // Open at the online/static transition, before a terminal event can be
    // observed.  A temp-file failure is therefore deterministic and
    // fail-closed instead of silently falling back to an unbounded vector.
    ensure_online_memory_anchor_spool();
}

void MetricCollector::online_register_ranks(const std::string& request_id,
                                            bool prefill,
                                            const std::vector<int>& ranks) {
    if (!this->enabled_) {
        return;
    }
    const auto it = this->request_index_by_request_id_.find(request_id);
    if (it == this->request_index_by_request_id_.end()) {
        // Unknown request (anchor registration before request registration
        // is a driver wiring bug, not a metrics-data property); ignore.
        return;
    }
    RequestMetricState& state = this->requests_[it->second];
    // REPLACE, not union: the manifest ranks are the OFFLINE planner's
    // instance assignment, the online watch members are the ranks the live
    // dynamic graph actually emitted on. The two differ by an instance
    // offset (measured: prefill {2,3,8,9,14,15} -> {38,39,44,45,50,51},
    // +36; decode {18,19,24,25,30,31} -> {22,23,28,29,34,35}, +4), so a
    // union would inflate the expected rank set (6+6=12) while the anchors
    // only cover the emitted ranks -- completion could never be satisfied.
    // The watch members are authoritative for the online run.
    std::set<int>& target =
        prefill ? state.prefill_ranks : state.decode_ranks;
    target.clear();
    target.insert(ranks.begin(), ranks.end());
}

bool MetricCollector::online_register_node_anchor(int rank, uint64_t node_id,
                                                  const std::string& request_id,
                                                  const std::string& kind,
                                                  bool transfer_anchor) {
    // R2 (2026-08-29): returns whether the (rank, node) HAS an anchor on
    // some edge after this call -- true also for an idempotent duplicate
    // hit (the OnlineNode fast-path flag must be set in both cases). The
    // disabled / unknown-request / unknown-kind paths return false because
    // no routing entry exists, matching the Workload-side enabled() gate.
    if (!this->enabled_) {
        return false;
    }
    uint8_t event_code = 0;
    if (kind == "prefill_start") {
        event_code = static_cast<uint8_t>(EventCode::PREFILL_START_ISSUE);
    } else if (kind == "prefill_end") {
        event_code = static_cast<uint8_t>(EventCode::PREFILL_END_COMPLETE);
    } else if (kind == "decode_start") {
        event_code = static_cast<uint8_t>(EventCode::DECODE_START_ISSUE);
    } else if (kind == "completion") {
        event_code = static_cast<uint8_t>(EventCode::COMPLETION_COMPLETE);
    } else if (kind == "first_token") {
        // WP9 (WP9_CONTRACT §1): complete edge, subject=request, min tick
        // per subject across ranks.
        event_code = static_cast<uint8_t>(EventCode::FIRST_TOKEN_COMPLETE);
    } else if (transfer_anchor) {
        event_code = static_cast<uint8_t>(EventCode::MEMORY_ANCHOR_COMPLETE);
    } else {
        return false;  // unknown kind: nothing to register
    }
    const auto it = this->request_index_by_request_id_.find(request_id);
    if (it == this->request_index_by_request_id_.end()) {
        return false;  // unknown request (see online_register_ranks)
    }
    // apply_event resolves subjects through request_index_by_queue_index_,
    // so the event subject is the queue index (the manifest semantics).
    const int64_t subject_id =
        this->requests_[it->second].queue_index;
    const NodeMetricEvent event{event_code, subject_id};
    const bool is_issue =
        event_code == static_cast<uint8_t>(EventCode::PREFILL_START_ISSUE) ||
        event_code == static_cast<uint8_t>(EventCode::DECODE_START_ISSUE);
    std::vector<NodeMetricEvent>& table =
        is_issue ? this->issue_events_[rank][node_id]
                 : this->complete_events_[rank][node_id];
    // Idempotent: the same (rank, node_id, event) may be re-registered (e.g.
    // a transfer node that is also a stage boundary); it must not double the
    // event stream.
    for (const auto& existing : table) {
        if (existing.event_code == event.event_code &&
            existing.subject_id == event.subject_id) {
            return true;  // duplicate: the anchor exists -- flag stays set
        }
    }
    table.push_back(event);
    // A2 (2026-08-28): remember the (rank, node) pair so
    // online_release_request_anchors can erase the routing entry at the
    // request's REQUEST_COMPLETE commit (duplicate pairs erase
    // idempotently; the reverse entry itself is freed with the request).
    this->online_anchor_nodes_[request_id].emplace_back(rank, node_id);
    return true;
}

void MetricCollector::check_consistency(bool condition,
                                        const std::string& message) {
    if (!condition) {
        this->consistency_violations_.push_back(message);
    }
}

std::optional<Tick> MetricCollector::resolve_arrival(size_t request_index) {
    RequestMetricState& state = this->requests_[request_index];
    if (state.resolved_arrival.has_value()) {
        return state.resolved_arrival;
    }
    if (state.arrival.kind == ArrivalSpec::Kind::ABSOLUTE) {
        state.resolved_arrival = state.arrival.value_ns;
        return state.resolved_arrival;
    }
    // after_request: parent actual completion + interval (doc sec.3.1/4.4).
    // The parent completion is an observed boundary tick and does not depend
    // on the parent's own arrival, so no recursion is needed here.
    const auto parent_it =
        this->request_index_by_queue_index_.find(state.arrival.parent_queue_index);
    if (parent_it == this->request_index_by_queue_index_.end()) {
        return std::nullopt;
    }
    const RequestMetricState& parent = this->requests_[parent_it->second];
    if (!parent.completion.has_value()) {
        return std::nullopt;
    }
    state.resolved_arrival = parent.completion.value() + state.arrival.interval_ns;
    return state.resolved_arrival;
}

void MetricCollector::emit_record(const std::string& json_line) const {
    // Single-line JSON with a fixed prefix (doc sec.5.9). C5 (2026-08-28):
    // records accumulate in emit_buffer_ and are drained at 1 MiB (which
    // bounds both memory and crash-loss windows) instead of one syscall per
    // record. write_all_or_die handles short writes and EINTR; a permanent
    // stdout failure is fail-closed rather than silently dropping records.
    this->emit_buffer_ += "[METRIC] ";
    this->emit_buffer_ += json_line;
    this->emit_buffer_ += "\n";
    if (this->emit_buffer_.size() >= (1u << 20)) {
        write_all_or_die(this->emit_buffer_.data(), this->emit_buffer_.size());
        this->emit_buffer_.clear();
    }
}

void MetricCollector::flush_emit_buffer() const {
    // C5 (2026-08-28): drain the buffered [METRIC] channel. finalize() calls
    // this at its end; the link-observer emission path calls it once after its
    // last record.
    if (this->emit_buffer_.empty()) {
        return;
    }
    write_all_or_die(this->emit_buffer_.data(), this->emit_buffer_.size());
    this->emit_buffer_.clear();
}

void MetricCollector::online_release_request_anchors(
    const std::string& request_id) {
    // A2 (2026-08-28): the completed request's anchor routing entries are
    // dead weight (see the header comment). requests_ / first_token_ticks_
    // stay -- finalize emits the request row and the first-token boundary
    // from them.
    if (!this->enabled_) {
        return;
    }
    const auto it = this->online_anchor_nodes_.find(request_id);
    if (it == this->online_anchor_nodes_.end()) {
        return;
    }
    for (const auto& [rank, node_id] : it->second) {
        const auto issue_rank = this->issue_events_.find(rank);
        if (issue_rank != this->issue_events_.end()) {
            issue_rank->second.erase(node_id);
            if (issue_rank->second.empty()) {
                this->issue_events_.erase(issue_rank);
            }
        }
        const auto complete_rank = this->complete_events_.find(rank);
        if (complete_rank != this->complete_events_.end()) {
            complete_rank->second.erase(node_id);
            if (complete_rank->second.empty()) {
                this->complete_events_.erase(complete_rank);
            }
        }
    }
    this->online_anchor_nodes_.erase(it);
}

void MetricCollector::emit_observer_record(const std::string& json_line) const {
    // SLO pipeline B2 (CPP_SPEC §D): observers outside the workload layer
    // (the congestion-aware FluidScheduler link observer) emit through the
    // same single-write channel. Read-only; no collector state is touched.
    emit_record(json_line);
}

void MetricCollector::finalize(const std::vector<Sys*>& systems,
                               Tick sim_end_tick) {
    if (!this->enabled_ || this->finalized_) {
        return;
    }
    const bool full_detail = (this->detail_level_ == "full");

    // Drain any logger output still sitting in the stdout FILE buffer so the
    // metric records appear after it, in one contiguous block.
    fflush(stdout);

    // Resolve arrivals now that actual completions are known (doc sec.4.4).
    for (size_t i = 0; i < this->requests_.size(); i++) {
        resolve_arrival(i);
    }

    // Per-rank compute/DRAM records (doc sec.3.7 and 3.8), sorted by rank.
    std::vector<Sys*> sorted_systems = systems;
    std::sort(sorted_systems.begin(), sorted_systems.end(),
              [](const Sys* a, const Sys* b) { return a->id < b->id; });
    for (const Sys* sys : sorted_systems) {
        const Statistics* stats = sys->workload->stats;
        const Tick wall_time = stats->get_wall_time();
        const Tick gpu_busy_ns = stats->calculate_type_time_in_window(
            Statistics::OperatorStatistics::OperatorType::GPU, 0,
            sim_end_tick);
        const RankMetricState rank_state =
            this->rank_states_.count(sys->id) > 0
                ? this->rank_states_[sys->id]
                : RankMetricState{};

        json record;
        record["schema"] = 1;
        record["type"] = "rank_compute";
        record["source"] = "simulator";
        record["repo_variant"] = this->repo_variant_;
        record["run_id"] = this->run_id_;
        record["rank"] = sys->id;
        record["utilization_window_ns"] = sim_end_tick;
        record["utilization_window"] = "[0, sim_end_ns]";
        record["wall_time_ns"] = wall_time;
        record["gpu_busy_ns"] = gpu_busy_ns;
        const long double window_seconds =
            static_cast<long double>(sim_end_tick) / 1e9L;

        // die_compute_busy_util = union(GPU_COMP intervals) / window
        // (doc sec.3.7).
        if (sim_end_tick > 0) {
            const long double busy_util =
                static_cast<long double>(gpu_busy_ns) /
                static_cast<long double>(sim_end_tick);
            record["die_compute_busy_util"] =
                static_cast<double>(busy_util);
            check_consistency(
                busy_util <= 1.0L + kUtilEpsilon,
                "rank " + std::to_string(sys->id) +
                    " die_compute_busy_util out of range: numerator=" +
                    std::to_string(gpu_busy_ns) +
                    " denominator=" + std::to_string(sim_end_tick));
        } else {
            record["die_compute_busy_util"] = nullptr;
        }

        record["total_num_ops"] = u128_to_string(rank_state.total_num_ops);
        record["total_local_bytes"] =
            u128_to_string(rank_state.total_local_bytes);
        record["local_hbm_restore_bytes"] =
            rank_state.local_hbm_restore_bytes;

        // achieved_compute_util = sum(num_ops) / (peak_flops * window_s)
        // (doc sec.3.7). Hardware constants come from the parsed system
        // configuration, never hardcoded.
        record["peak_flops_per_second"] = sys->peak_perf;
        record["peak_flops_per_second_source"] =
            "system-configuration:peak-perf";
        if (sys->roofline_enabled && sys->peak_perf > 0 && sim_end_tick > 0) {
            const long double achieved =
                u128_to_long_double(rank_state.total_num_ops) /
                (static_cast<long double>(sys->peak_perf) * window_seconds);
            record["achieved_compute_util"] = static_cast<double>(achieved);
            check_consistency(
                achieved <= 1.0L + kUtilEpsilon,
                "rank " + std::to_string(sys->id) +
                    " achieved_compute_util out of range: numerator=" +
                    u128_to_string(rank_state.total_num_ops) +
                    " denominator=peak_flops_per_second*window_seconds");
        } else {
            record["achieved_compute_util"] = nullptr;
            record["achieved_compute_util_note"] =
                "unavailable: roofline model disabled or zero peak-perf/window";
        }

        // dram_bw_util = local_bytes_served / (local_hbm_bw * window_s)
        // (doc sec.3.8, source=roofline_local_bytes).
        record["dram_bw_util_source"] = "roofline_local_bytes";
        record["local_hbm_bw_bytes_per_second"] = sys->local_mem_bw;
        record["local_hbm_bw_source"] = "system-configuration:local-mem-bw";
        if (sys->roofline_enabled && sys->local_mem_bw > 0 &&
            sim_end_tick > 0) {
            const long double bw_util =
                u128_to_long_double(rank_state.total_local_bytes) /
                (static_cast<long double>(sys->local_mem_bw) * window_seconds);
            record["dram_bw_util"] = static_cast<double>(bw_util);
            check_consistency(
                bw_util <= 1.0L + kUtilEpsilon,
                "rank " + std::to_string(sys->id) +
                    " dram_bw_util out of range: numerator=" +
                    u128_to_string(rank_state.total_local_bytes) +
                    " denominator=local_hbm_bw*window_seconds");
        } else {
            record["dram_bw_util"] = nullptr;
            record["dram_bw_util_note"] =
                "unavailable: roofline model disabled or zero "
                "local-mem-bw/window";
        }

        // Global-window roofline active-kernel utilization (doc sec.8.7):
        // same weighting formula as the existing "Average compute
        // utilization" log line, under distinct field names.
        record["active_kernel_roofline_source"] = "simulator";
        record["active_kernel_roofline_definition"] =
            "duration-weighted average of roofline per-node utilization over "
            "comp nodes clipped to the utilization window; same weighting "
            "formula as 'Average compute utilization' (doc sec.8.7)";
        if (sys->roofline_enabled) {
            const auto windowed =
                stats->calculate_roofline_utilization_in_window(0,
                                                                sim_end_tick);
            record["active_kernel_roofline_compute_util"] =
                windowed.compute_utilization_weighted_sum /
                static_cast<double>(windowed.total_comp_time);
            record["active_kernel_roofline_memory_util"] =
                windowed.memory_utilization_weighted_sum /
                static_cast<double>(windowed.total_comp_time);
        } else {
            record["active_kernel_roofline_compute_util"] = nullptr;
            record["active_kernel_roofline_memory_util"] = nullptr;
            record["active_kernel_roofline_note"] =
                "unavailable: roofline model disabled";
        }
        emit_record(record.dump());

        // SH2-only second DRAM bandwidth accounting (doc sec.3.8): when this
        // rank runs the LocalHbmBandwidthModel
        // (hbm_bandwidth_contention / hbm_kv_restore_bandwidth_sharing),
        // report the runtime model's own read-only counters. The plain
        // roofline_local_bytes record above is still emitted unchanged.
        const LocalHbmBandwidthModel* hbm_model =
            sys->workload->local_hbm_bandwidth_model.get();
        if (hbm_model != nullptr) {
            const double hbm_busy_ns = hbm_model->hbm_busy_ns();
            const double hbm_shared_ns = hbm_model->hbm_shared_ns();
            const double compute_bytes_served =
                hbm_model->compute_bytes_served();
            const double restore_bytes_served =
                hbm_model->restore_bytes_served();
            const double comm_read_bytes_served =
                hbm_model->comm_read_bytes_served();
            const double comm_write_bytes_served =
                hbm_model->comm_write_bytes_served();
            const double pool_read_bytes_served =
                hbm_model->pool_read_bytes_served();
            const double pool_write_bytes_served =
                hbm_model->pool_write_bytes_served();
            const long double served_bytes =
                static_cast<long double>(compute_bytes_served) +
                static_cast<long double>(restore_bytes_served) +
                static_cast<long double>(comm_read_bytes_served) +
                static_cast<long double>(comm_write_bytes_served) +
                static_cast<long double>(pool_read_bytes_served) +
                static_cast<long double>(pool_write_bytes_served);

            json hbm_record;
            hbm_record["schema"] = 1;
            hbm_record["type"] = "local_hbm";
            hbm_record["source"] = "shared_hbm_runtime_model";
            hbm_record["repo_variant"] = this->repo_variant_;
            hbm_record["run_id"] = this->run_id_;
            hbm_record["rank"] = sys->id;
            hbm_record["model"] =
                "LocalHbmBandwidthModel: N-way strict equal split of the "
                "rank HBM bandwidth across all active HBM users (COMP, "
                "KV-restore DMA, NoC p2p comm read/write endpoints, "
                "off-chip pool read/write endpoints); full_rate/N per job, "
                "event-driven reallocation on every completion; reads and "
                "writes share one bus and one total bandwidth (no "
                "peak/sustained distinction)";
            hbm_record["utilization_window_ns"] = sim_end_tick;
            hbm_record["utilization_window"] = "[0, sim_end_ns]";
            hbm_record["hbm_busy_ns"] = hbm_busy_ns;
            hbm_record["hbm_busy_definition"] =
                "sum of advance_to() steps during which at least one job was "
                "streaming HBM bytes (memory-latency-only steps excluded)";
            hbm_record["hbm_shared_ns"] = hbm_shared_ns;
            hbm_record["hbm_shared_definition"] =
                "subset of hbm_busy_ns during which two or more jobs shared "
                "the rank HBM bandwidth (equal split)";
            hbm_record["compute_bytes_served"] = compute_bytes_served;
            hbm_record["restore_bytes_served"] = restore_bytes_served;
            hbm_record["comm_read_bytes_served"] = comm_read_bytes_served;
            hbm_record["comm_write_bytes_served"] = comm_write_bytes_served;
            hbm_record["pool_read_bytes_served"] = pool_read_bytes_served;
            hbm_record["pool_write_bytes_served"] = pool_write_bytes_served;
            hbm_record["bytes_served_definition"] =
                "sum of per-step per_user_rate*step_ns accumulated inside "
                "advance_to(); does not alter transition scheduling";
            hbm_record["peak_concurrent_jobs"] =
                hbm_model->peak_concurrent_jobs();
            hbm_record["peak_concurrent_jobs_definition"] =
                "largest number of simultaneously active HBM jobs observed "
                "at a membership change (issue or completion)";
            hbm_record["redistribution_events"] =
                hbm_model->redistribution_events();
            hbm_record["redistribution_events_definition"] =
                "count of equal-share recomputations: every membership "
                "change (job joining a non-empty set, completion leaving "
                "survivors) with at least one remaining bandwidth user";
            hbm_record["local_hbm_restore_bytes_issued"] =
                rank_state.local_hbm_restore_bytes;
            hbm_record["local_hbm_bw_bytes_per_second"] = sys->local_mem_bw;
            hbm_record["local_hbm_bw_source"] =
                "system-configuration:local-mem-bw";
            if (sim_end_tick > 0) {
                const long double busy_util =
                    static_cast<long double>(hbm_busy_ns) /
                    static_cast<long double>(sim_end_tick);
                hbm_record["hbm_busy_util"] = static_cast<double>(busy_util);
                check_consistency(
                    busy_util <= 1.0L + kUtilEpsilon,
                    "rank " + std::to_string(sys->id) +
                        " hbm_busy_util out of range: numerator=" +
                        std::to_string(hbm_busy_ns) +
                        " denominator=" + std::to_string(sim_end_tick));
                hbm_record["hbm_shared_fraction_of_window"] =
                    static_cast<double>(
                        static_cast<long double>(hbm_shared_ns) /
                        static_cast<long double>(sim_end_tick));
            } else {
                hbm_record["hbm_busy_util"] = nullptr;
                hbm_record["hbm_shared_fraction_of_window"] = nullptr;
            }
            if (sys->local_mem_bw > 0 && sim_end_tick > 0) {
                const long double bw_util =
                    served_bytes /
                    (static_cast<long double>(sys->local_mem_bw) *
                     window_seconds);
                hbm_record["dram_bw_util"] = static_cast<double>(bw_util);
                hbm_record["dram_bw_util_numerator_bytes"] =
                    static_cast<double>(served_bytes);
                hbm_record["dram_bw_util_denominator"] =
                    "local_hbm_bw*window_seconds";
                check_consistency(
                    bw_util <= 1.0L + kUtilEpsilon,
                    "rank " + std::to_string(sys->id) +
                        " shared_hbm_runtime_model dram_bw_util out of range");
            } else {
                hbm_record["dram_bw_util"] = nullptr;
                hbm_record["dram_bw_util_note"] =
                    "undefined: zero local-mem-bw/window";
            }
            emit_record(hbm_record.dump());
        }
    }

    // Per-request records (doc sec.5.9 field list), sorted by queue_index.
    std::vector<size_t> request_order(this->requests_.size());
    for (size_t i = 0; i < request_order.size(); i++) {
        request_order[i] = i;
    }
    std::sort(request_order.begin(), request_order.end(), [&](size_t a, size_t b) {
        return this->requests_[a].queue_index < this->requests_[b].queue_index;
    });

    std::vector<Tick> completed_e2e;
    uint64_t completed_unique_requests = 0;
    std::optional<Tick> first_arrival;
    std::optional<Tick> last_completion;

    for (const size_t index : request_order) {
        RequestMetricState& state = this->requests_[index];
        const bool prefill_start_done =
            state.prefill_start.has_value() &&
            std::includes(state.prefill_start_ranks_seen.begin(),
                          state.prefill_start_ranks_seen.end(),
                          state.prefill_ranks.begin(),
                          state.prefill_ranks.end());
        const bool prefill_end_done =
            state.prefill_end.has_value() &&
            std::includes(state.prefill_end_ranks_seen.begin(),
                          state.prefill_end_ranks_seen.end(),
                          state.prefill_ranks.begin(),
                          state.prefill_ranks.end());
        const bool decode_start_done =
            state.decode_start.has_value() &&
            std::includes(state.decode_start_ranks_seen.begin(),
                          state.decode_start_ranks_seen.end(),
                          state.decode_ranks.begin(),
                          state.decode_ranks.end());
        const bool completion_done =
            state.completion.has_value() &&
            std::includes(state.completion_ranks_seen.begin(),
                          state.completion_ranks_seen.end(),
                          state.decode_ranks.begin(),
                          state.decode_ranks.end());
        const bool completed = state.resolved_arrival.has_value() &&
            prefill_start_done && prefill_end_done && decode_start_done &&
            completion_done;

        // Stash observed boundaries for the doc sec.7.8 memory anchor
        // resolution, which runs after this request pass.
        RequestBoundaryInfo boundaries;
        boundaries.prefill_start = state.prefill_start;
        boundaries.decode_start = state.decode_start;
        boundaries.completion = state.completion;
        boundaries.prefill_done = prefill_start_done;
        boundaries.decode_done = decode_start_done;
        boundaries.completion_done = completion_done;
        this->request_boundaries_[state.queue_index] = boundaries;

        json record;
        record["schema"] = 1;
        record["type"] = "request";
        record["source"] = "simulator";
        record["repo_variant"] = this->repo_variant_;
        record["run_id"] = this->run_id_;
        record["queue_index"] = state.queue_index;
        record["session_id"] = state.session_id;
        record["turn_index"] = state.turn_index;
        record["request_id"] = state.request_id;
        record["prefill_instance"] = state.prefill_instance;
        record["decode_instance"] = state.decode_instance;
        record["completed"] = completed;
        record["arrival_ns"] = tick_or_null(state.resolved_arrival);
        record["prefill_start_ns"] = tick_or_null(state.prefill_start);
        record["prefill_end_ns"] = tick_or_null(state.prefill_end);
        record["decode_start_ns"] = tick_or_null(state.decode_start);
        record["completion_ns"] = tick_or_null(state.completion);

        // WP9 (WP9_CONTRACT §1 + §6 2026-08-27 ruling): first-token
        // boundary. Null when no code-8 event ever fired for the request;
        // ORDER violations (arrival <= first_token <= completion) null the
        // field, count a consistency violation, and leave an instruction
        // note. The former decode_length==1 equality (first_token ==
        // completion) was relaxed per the ruling: code 8 aggregates the min
        // tick across TP ranks while completion (code 4) aggregates the
        // max, so even a same-node dual anchor disagrees by the TP
        // completion skew -- the residual is reported as the informational
        // first_token_completion_skew_ns field instead of a violation.
        {
            const auto ft_it =
                this->first_token_ticks_.find(state.queue_index);
            std::optional<Tick> first_token;
            std::string first_token_note;
            if (ft_it != this->first_token_ticks_.end()) {
                first_token = ft_it->second;
                const std::string prefix =
                    "queue_index " + std::to_string(state.queue_index) + ": ";
                bool ok = true;
                if (state.resolved_arrival.has_value() &&
                    first_token.value() < state.resolved_arrival.value()) {
                    check_consistency(
                        false, prefix + "arrival > first_token");
                    ok = false;
                }
                if (state.completion.has_value() &&
                    first_token.value() > state.completion.value()) {
                    check_consistency(
                        false, prefix + "first_token > completion");
                    ok = false;
                }
                if (!ok) {
                    first_token.reset();
                    first_token_note =
                        "first_token_ns nulled by invariant violation (see "
                        "consistency record)";
                }
            }
            record["first_token_ns"] = tick_or_null(first_token);
            if (first_token.has_value() && state.completion.has_value() &&
                state.decode_length.has_value() &&
                state.decode_length.value() == 1) {
                // WP9_CONTRACT §6: informational only (completion aggregated
                // as max, first_token as min -> skew >= 0 is expected).
                record["first_token_completion_skew_ns"] =
                    state.completion.value() - first_token.value();
            }
            if (!first_token_note.empty()) {
                record["first_token_note"] = first_token_note;
            } else if (completed && !state.decode_length.has_value()) {
                record["first_token_note"] =
                    "manifest_decode_length_missing: decode-length==1 skew "
                    "field not evaluable";
            } else if (!first_token.has_value()) {
                record["first_token_note"] =
                    "no code-8 first-token event observed for this request";
            }
        }

        if (state.resolved_arrival.has_value()) {
            if (!first_arrival.has_value() ||
                state.resolved_arrival.value() < first_arrival.value()) {
                first_arrival = state.resolved_arrival;
            }
        }

        if (completed) {
            const Tick arrival = state.resolved_arrival.value();
            const Tick prefill_start = state.prefill_start.value();
            const Tick prefill_end = state.prefill_end.value();
            const Tick decode_start = state.decode_start.value();
            const Tick completion = state.completion.value();

            // Self-consistency (doc sec.12.6); violations are reported, not
            // silently clamped.
            const std::string prefix =
                "queue_index " + std::to_string(state.queue_index) + ": ";
            check_consistency(arrival <= prefill_start,
                              prefix + "arrival > prefill_start");
            check_consistency(prefill_start <= prefill_end,
                              prefix + "prefill_start > prefill_end");
            check_consistency(prefill_end <= decode_start,
                              prefix + "prefill_end > decode_start");
            check_consistency(decode_start <= completion,
                              prefix + "decode_start > completion");
            const bool ordering_ok = arrival <= prefill_start &&
                prefill_start <= prefill_end && prefill_end <= decode_start &&
                decode_start <= completion;

            if (ordering_ok) {
                // Stage breakdown (doc sec.3.3).
                record["queue_ns"] = prefill_start - arrival;
                record["prefill_ns"] = prefill_end - prefill_start;
                record["prefill_decode_gap_ns"] = decode_start - prefill_end;
                record["decode_ns"] = completion - decode_start;
                record["e2e_ns"] = completion - arrival;
                completed_e2e.push_back(completion - arrival);
            } else {
                // Withhold derived fields instead of emitting underflowed
                // unsigned arithmetic; the violation is reported above.
                record["queue_ns"] = nullptr;
                record["prefill_ns"] = nullptr;
                record["prefill_decode_gap_ns"] = nullptr;
                record["decode_ns"] = nullptr;
                record["e2e_ns"] = nullptr;
                record["stage_breakdown_note"] =
                    "withheld: stage ordering violation (see consistency "
                    "record)";
            }

            completed_unique_requests++;
            if (!last_completion.has_value() ||
                completion > last_completion.value()) {
                last_completion = completion;
            }
        } else {
            record["queue_ns"] = nullptr;
            record["prefill_ns"] = nullptr;
            record["prefill_decode_gap_ns"] = nullptr;
            record["decode_ns"] = nullptr;
            record["e2e_ns"] = nullptr;
            if (!state.resolved_arrival.has_value()) {
                record["incomplete_reason"] =
                    "arrival could not be resolved (missing parent "
                    "completion)";
            } else {
                record["incomplete_reason"] =
                    "missing stage boundary events for one or more expected "
                    "ranks";
            }
        }
        if (full_detail) {
            emit_record(record.dump());
        }
    }

    // Microbenchmark iteration boundary records (doc sec.4.3 codes 5/6 and
    // sec.8.6), plus the doc sec.8.7 per-iteration utilization fields.
    // Full detail only; service runs normally have none.
    if (full_detail) {
        std::unordered_map<int, const Sys*> systems_by_id;
        for (const Sys* sys : systems) {
            systems_by_id[sys->id] = sys;
        }
        for (const auto& [point_id, state] : this->iterations_) {
            json record;
            record["schema"] = 1;
            record["type"] = "iteration";
            record["source"] = "simulator";
            record["repo_variant"] = this->repo_variant_;
            record["run_id"] = this->run_id_;
            record["benchmark_point_id"] = point_id;
            record["iteration_start_ns"] = tick_or_null(state.iteration_start);
            record["iteration_end_ns"] = tick_or_null(state.iteration_end);
            if (state.iteration_start.has_value() &&
                state.iteration_end.has_value()) {
                record["iteration_time_ns"] =
                    state.iteration_end.value() - state.iteration_start.value();
            } else {
                record["iteration_time_ns"] = nullptr;
            }
            record["start_ranks_seen"] = state.start_ranks_seen.size();
            record["end_ranks_seen"] = state.end_ranks_seen.size();

            // Doc sec.8.7 utilization fields. A microbenchmark run carries
            // exactly one point per ET, so the per-rank num_ops/byte totals
            // accumulated by on_compute_issue over the point's active ranks
            // are precisely this iteration's numerators (no windowed
            // attribution needed); idle sentinel ranks contribute nothing.
            const bool have_spec = this->microbench_point_.has_value() &&
                this->microbench_point_->benchmark_point_id == point_id &&
                this->microbench_point_->tp_degree > 0;
            const bool have_window = state.iteration_start.has_value() &&
                state.iteration_end.has_value() &&
                state.iteration_end.value() > state.iteration_start.value();
            if (have_spec) {
                record["point"] = this->microbench_point_->raw;
                record["tp_degree"] = this->microbench_point_->tp_degree;
            }
            if (!have_spec || !have_window) {
                record["num_ops_total"] = nullptr;
                record["local_tensor_bytes_total"] = nullptr;
                record["compute_util"] = nullptr;
                record["hbm_bw_util"] = nullptr;
                record["active_kernel_roofline_compute_util"] = nullptr;
                record["active_kernel_roofline_memory_util"] = nullptr;
                record["utilization_note"] =
                    "unavailable: no matching microbenchmark point "
                    "descriptor in the manifest, or zero-length iteration "
                    "window";
            } else {
                const MicrobenchPointSpec& spec = this->microbench_point_.value();
                unsigned __int128 num_ops_total = 0;
                unsigned __int128 local_bytes_total = 0;
                double compute_util_weighted_sum = 0;
                double memory_util_weighted_sum = 0;
                Tick total_comp_time = 0;
                const Sys* reference_sys = nullptr;
                for (const int rank : spec.active_ranks) {
                    const auto state_it = this->rank_states_.find(rank);
                    if (state_it != this->rank_states_.end()) {
                        num_ops_total += state_it->second.total_num_ops;
                        local_bytes_total += state_it->second.total_local_bytes;
                    }
                    const auto sys_it = systems_by_id.find(rank);
                    if (sys_it == systems_by_id.end()) {
                        continue;
                    }
                    if (reference_sys == nullptr) {
                        reference_sys = sys_it->second;
                    }
                    const auto windowed = sys_it->second->workload->stats
                        ->calculate_roofline_utilization_in_window(
                            state.iteration_start.value(),
                            state.iteration_end.value());
                    compute_util_weighted_sum +=
                        windowed.compute_utilization_weighted_sum;
                    memory_util_weighted_sum +=
                        windowed.memory_utilization_weighted_sum;
                    total_comp_time += windowed.total_comp_time;
                }

                record["num_ops_total"] = u128_to_string(num_ops_total);
                record["local_tensor_bytes_total"] =
                    u128_to_string(local_bytes_total);
                record["numerator_attribution"] =
                    "sum of per-rank on_compute_issue totals over the "
                    "point's active_ranks (one point per ET/run)";
                record["compute_util_source"] = "simulator_microbenchmark";
                record["compute_util_definition"] =
                    "num_ops_total / (tp_degree * peak_flops_per_second * "
                    "iteration_time_seconds)";
                record["hbm_bw_util_source"] = "simulator_microbenchmark";
                record["hbm_bw_util_definition"] =
                    "local_tensor_bytes_total / (tp_degree * "
                    "local_hbm_bw_bytes_per_second * "
                    "iteration_time_seconds)";
                record["active_kernel_roofline_source"] = "simulator";
                record["active_kernel_roofline_definition"] =
                    "duration-weighted average of roofline per-node "
                    "utilization over comp nodes clipped to the iteration "
                    "window, aggregated across active ranks; same weighting "
                    "formula as 'Average compute utilization' (doc sec.8.7)";

                const long double iteration_seconds =
                    static_cast<long double>(
                        state.iteration_end.value() -
                        state.iteration_start.value()) /
                    1e9L;
                const double peak_flops =
                    reference_sys != nullptr ? reference_sys->peak_perf : 0;
                const double local_hbm_bw =
                    reference_sys != nullptr ? reference_sys->local_mem_bw : 0;
                const bool roofline_on = reference_sys != nullptr &&
                    reference_sys->roofline_enabled;
                record["peak_flops_per_second"] = peak_flops;
                record["peak_flops_per_second_source"] =
                    "system-configuration:peak-perf";
                record["local_hbm_bw_bytes_per_second"] = local_hbm_bw;
                record["local_hbm_bw_source"] =
                    "system-configuration:local-mem-bw";

                const std::string prefix = "benchmark_point_id " +
                    std::to_string(point_id) + ": ";
                if (roofline_on && peak_flops > 0) {
                    const long double denominator =
                        static_cast<long double>(spec.tp_degree) *
                        static_cast<long double>(peak_flops) *
                        iteration_seconds;
                    const long double compute_util =
                        u128_to_long_double(num_ops_total) / denominator;
                    record["compute_util"] =
                        static_cast<double>(compute_util);
                    check_consistency(
                        compute_util >= -kUtilEpsilon &&
                            compute_util <= 1.0L + kUtilEpsilon,
                        prefix + "compute_util out of range: numerator=" +
                            u128_to_string(num_ops_total) +
                            " denominator=tp_degree*peak_flops_per_second*"
                            "iteration_time_seconds");
                } else {
                    record["compute_util"] = nullptr;
                    record["compute_util_note"] =
                        "unavailable: roofline model disabled or zero "
                        "peak-perf";
                }
                if (roofline_on && local_hbm_bw > 0) {
                    const long double denominator =
                        static_cast<long double>(spec.tp_degree) *
                        static_cast<long double>(local_hbm_bw) *
                        iteration_seconds;
                    const long double hbm_bw_util =
                        u128_to_long_double(local_bytes_total) / denominator;
                    record["hbm_bw_util"] =
                        static_cast<double>(hbm_bw_util);
                    check_consistency(
                        hbm_bw_util >= -kUtilEpsilon &&
                            hbm_bw_util <= 1.0L + kUtilEpsilon,
                        prefix + "hbm_bw_util out of range: numerator=" +
                            u128_to_string(local_bytes_total) +
                            " denominator=tp_degree*local_hbm_bw*"
                            "iteration_time_seconds");
                } else {
                    record["hbm_bw_util"] = nullptr;
                    record["hbm_bw_util_note"] =
                        "unavailable: roofline model disabled or zero "
                        "local-mem-bw";
                }
                if (roofline_on && total_comp_time > 0) {
                    const double compute_roofline =
                        compute_util_weighted_sum /
                        static_cast<double>(total_comp_time);
                    const double memory_roofline =
                        memory_util_weighted_sum /
                        static_cast<double>(total_comp_time);
                    record["active_kernel_roofline_compute_util"] =
                        compute_roofline;
                    record["active_kernel_roofline_memory_util"] =
                        memory_roofline;
                    check_consistency(
                        compute_roofline <= 1.0 + kUtilEpsilon,
                        prefix +
                            "active_kernel_roofline_compute_util out of "
                            "range");
                    check_consistency(
                        memory_roofline <= 1.0 + kUtilEpsilon,
                        prefix +
                            "active_kernel_roofline_memory_util out of "
                            "range");
                } else {
                    record["active_kernel_roofline_compute_util"] = nullptr;
                    record["active_kernel_roofline_memory_util"] = nullptr;
                }
            }
            emit_record(record.dump());
        }
        if (this->online_mode_) {
            // The raw anonymous spool is replayed exactly here, preserving
            // the legacy memory_anchor block's fields, multiplicity, and
            // event-arrival order without retaining it in RAM.
            emit_online_memory_anchor_spool_records();
        } else {
            for (const auto& anchor : this->memory_anchor_ticks_) {
                emit_memory_anchor_record(anchor);
            }
        }
    }
    // Summary detail deliberately suppresses raw anchor records too; close
    // the file in either detail mode before the later ledger replay.
    if (this->online_mode_) {
        close_memory_anchor_spool_or_die();
    }

    // Capacity-time integral over the planner memory ledger (doc sec.7.8),
    // anchored to the actual ASTRA ticks collected above. Emits one
    // capacity_timeavg record per rank in every non-off mode; the raw
    // planner peaks passthrough is full-detail only.
    std::vector<int> all_ranks;
    all_ranks.reserve(systems.size());
    for (const Sys* sys : systems) {
        all_ranks.push_back(sys->id);
    }
    const MemoryReplayTotals memory_totals =
        emit_memory_records(sim_end_tick, full_detail, all_ranks);

    // Summary record (doc sec.5.9 field list, sec.3.4/3.5 windows).
    const uint64_t input_requests = this->requests_.size();
    check_consistency(completed_unique_requests <= input_requests,
                      "completed_unique_requests > input_requests");

    json summary;
    summary["schema"] = 1;
    summary["type"] = "summary";
    summary["source"] = "simulator";
    summary["repo_variant"] = this->repo_variant_;
    summary["run_mode"] = this->run_mode_;
    summary["run_id"] = this->run_id_;
    summary["detail_level"] = this->detail_level_;
    summary["input_requests"] = input_requests;
    summary["completed_unique_requests"] = completed_unique_requests;
    summary["incomplete_requests"] =
        input_requests - completed_unique_requests;
    summary["first_arrival_ns"] = tick_or_null(first_arrival);
    summary["last_completion_ns"] = tick_or_null(last_completion);
    summary["sim_end_ns"] = sim_end_tick;
    summary["percentile_method"] = "nearest_rank";

    std::sort(completed_e2e.begin(), completed_e2e.end());
    if (!completed_e2e.empty()) {
        long double sum = 0;
        for (const Tick value : completed_e2e) {
            sum += static_cast<long double>(value);
        }
        summary["mean_e2e_ns"] =
            static_cast<double>(sum / completed_e2e.size());
        summary["min_e2e_ns"] = completed_e2e.front();
        summary["max_e2e_ns"] = completed_e2e.back();
        summary["p50_e2e_ns"] = nearest_rank_percentile(completed_e2e, 0.50);
        summary["p95_e2e_ns"] = nearest_rank_percentile(completed_e2e, 0.95);
        summary["p99_e2e_ns"] = nearest_rank_percentile(completed_e2e, 0.99);
    } else {
        summary["mean_e2e_ns"] = nullptr;
        summary["min_e2e_ns"] = nullptr;
        summary["max_e2e_ns"] = nullptr;
        summary["p50_e2e_ns"] = nullptr;
        summary["p95_e2e_ns"] = nullptr;
        summary["p99_e2e_ns"] = nullptr;
    }

    // drain_tput_rps (doc sec.3.4): numerator is the number of unique
    // requests with an actual completion, never the input request count.
    summary["drain_tput_numerator"] = completed_unique_requests;
    summary["drain_tput_window"] =
        "last_completion_ns - first_arrival_ns";
    summary["sim_window_tput_numerator"] = completed_unique_requests;
    summary["sim_window_tput_window"] = "[0, sim_end_ns]";
    if (completed_unique_requests > 0 && first_arrival.has_value() &&
        last_completion.has_value() &&
        last_completion.value() > first_arrival.value()) {
        const Tick denominator =
            last_completion.value() - first_arrival.value();
        summary["drain_tput_denominator_ns"] = denominator;
        summary["drain_tput_rps"] =
            static_cast<double>(completed_unique_requests) * 1e9 /
            static_cast<double>(denominator);
    } else {
        summary["drain_tput_denominator_ns"] = nullptr;
        summary["drain_tput_rps"] = nullptr;
        summary["drain_tput_note"] =
            "undefined: empty or zero-length drain window";
    }
    if (completed_unique_requests > 0 && sim_end_tick > 0) {
        summary["sim_window_tput_denominator_ns"] = sim_end_tick;
        summary["sim_window_tput_rps"] =
            static_cast<double>(completed_unique_requests) * 1e9 /
            static_cast<double>(sim_end_tick);
    } else {
        summary["sim_window_tput_denominator_ns"] = nullptr;
        summary["sim_window_tput_rps"] = nullptr;
        summary["sim_window_tput_note"] =
            "undefined: no completed request or zero sim window";
    }

    // Doc sec.7.8 capacity-time replay totals (details are in the per-rank
    // capacity_timeavg records).
    summary["memory_actions_total"] = memory_totals.actions_total;
    summary["memory_actions_replayed"] = memory_totals.actions_replayed;
    summary["memory_actions_unresolved"] = memory_totals.actions_unresolved;
    summary["memory_actions_unresolved_by_reason"] =
        memory_totals.unresolved_by_reason;
    summary["planner_memory_peaks_count"] = memory_totals.peaks_count;
    summary["memory_replay_final_state_mismatches"] =
        memory_totals.final_state_mismatches;
    summary["memory_transfer_anchor_request_level_fallback"] =
        memory_totals.transfer_anchor_request_level_fallback;
    emit_record(summary.dump());

    // Consistency record (doc sec.12.6).
    if (this->manifest_npus_count_.has_value()) {
        check_consistency(
            this->manifest_npus_count_.value() ==
                static_cast<int64_t>(systems.size()),
            "manifest npus_count != simulated systems count");
    }
    json consistency;
    consistency["schema"] = 1;
    consistency["type"] = "consistency";
    consistency["source"] = "simulator";
    consistency["repo_variant"] = this->repo_variant_;
    consistency["run_id"] = this->run_id_;
    consistency["ok"] = this->consistency_violations_.empty() &&
        this->dropped_events_ == 0;
    consistency["violations"] = this->consistency_violations_;
    consistency["dropped_events"] = this->dropped_events_;
    // Unresolved memory anchors are expected whenever a run has incomplete
    // requests; they are reported here (never silently dropped) but do not
    // flag the run inconsistent on their own.
    consistency["memory_actions_unresolved"] = memory_totals.actions_unresolved;
    consistency["memory_actions_unresolved_by_reason"] =
        memory_totals.unresolved_by_reason;
    emit_record(consistency.dump());

    for (const auto& violation : this->consistency_violations_) {
        std::cerr << "[METRIC][ERROR] consistency violation: " << violation
                  << std::endl;
    }
    if (this->dropped_events_ > 0) {
        std::cerr << "[METRIC][ERROR] " << this->dropped_events_
                  << " node events referenced unknown subjects" << std::endl;
    }
    // C5 (2026-08-28): all finalize-time records are buffered -- drain the
    // channel so the [METRIC] block is complete before main_online's
    // run-end log lines (the post-finalize link-observer path reuses the
    // same buffer and flushes again when done).
    flush_emit_buffer();
    this->finalized_ = true;
}

namespace {

// Signed 128-bit helpers for the capacity-time integral: per-rank ledger
// states are bounded by capacity, but byte*ns areas are not.
std::string i128_to_string(__int128 value) {
    if (value == 0) {
        return "0";
    }
    std::string sign;
    if (value < 0) {
        sign = "-";
        value = -value;
    }
    return sign + u128_to_string(static_cast<unsigned __int128>(value));
}

long double i128_to_long_double(__int128 value) {
    if (value < 0) {
        return -u128_to_long_double(
            static_cast<unsigned __int128>(-value));
    }
    return u128_to_long_double(static_cast<unsigned __int128>(value));
}

}  // namespace

MetricCollector::MemoryReplayTotals MetricCollector::emit_memory_records(
    Tick sim_end_tick, bool full_detail, const std::vector<int>& fallback_ranks) {
    MemoryReplayTotals totals;
    totals.actions_total = this->memory_actions_.size();
    totals.peaks_count = this->planner_memory_peaks_.size();
    if (this->memory_actions_.empty() && this->planner_memory_peaks_.empty()) {
        // WP8 (CPP_SPEC §B): no planner ledger rows at all (the online
        // synthetic manifests carry none). Still emit the per-rank
        // watermark summaries over the simulated ranks so the flat-zero
        // replay is explicit downstream instead of a missing record.
        std::map<int, WatermarkSeries> flat_series;
        for (const int rank : fallback_ranks) {
            WatermarkSeries series;
            series.replayed = false;
            flat_series.emplace(rank, std::move(series));
        }
        emit_watermark_records(flat_series, sim_end_tick, full_detail);
        return totals;
    }

    // Code-7 transfer anchors (doc sec.4.3): (subject queue_index, rank) ->
    // last transfer node complete tick, plus the per-request last transfer
    // node complete tick across ranks (doc sec.7.8 transfer row).
    std::map<std::pair<int64_t, int>, Tick> transfer_anchor_ticks;
    std::map<int64_t, Tick> transfer_anchor_by_subject;
    if (!this->online_mode_) {
        for (const auto& anchor : this->memory_anchor_ticks_) {
            const auto key = std::make_pair(anchor.subject_id, anchor.rank);
            auto it = transfer_anchor_ticks.find(key);
            if (it == transfer_anchor_ticks.end() || anchor.tick > it->second) {
                transfer_anchor_ticks[key] = anchor.tick;
            }
            auto subj_it = transfer_anchor_by_subject.find(anchor.subject_id);
            if (subj_it == transfer_anchor_by_subject.end() ||
                anchor.tick > subj_it->second) {
                transfer_anchor_by_subject[anchor.subject_id] = anchor.tick;
            }
        }
    }

    // Per-rank capacity and final planner ledger from the peaks payload.
    std::map<int, const json*> peaks_by_rank;
    for (const auto& entry : this->planner_memory_peaks_) {
        const int rank = entry.value("rank", -1);
        if (rank >= 0) {
            peaks_by_rank[rank] = &entry;
        }
    }

    struct AnchoredDelta {
        Tick tick;
        int64_t sequence_index;
        int64_t weight_delta;
        int64_t resident_delta;
        int64_t reserved_delta;
    };
    std::map<int, std::vector<AnchoredDelta>> deltas_by_rank;
    std::map<int, std::map<std::string, uint64_t>> quality_counts_by_rank;
    std::map<int, uint64_t> unresolved_by_rank;
    std::map<int, uint64_t> total_by_rank;
    std::map<int, uint64_t> transfer_anchor_fallback_by_rank;

    // Anchor resolution table (doc sec.7.8). Actions whose anchor cannot be
    // resolved (e.g. the triggering request stayed incomplete) are skipped
    // but counted by reason, never silently dropped.
    for (const auto& action : this->memory_actions_) {
        total_by_rank[action.rank]++;
        std::optional<Tick> anchor_tick;
        std::string reason;
        if (action.anchor_kind == "tick_zero") {
            anchor_tick = 0;
        } else if (action.trigger_queue_index < 0) {
            reason = "missing_trigger_queue_index";
        } else if (action.anchor_kind == "prefill_start" ||
                   action.anchor_kind == "decode_start" ||
                   action.anchor_kind == "completion") {
            const auto it =
                this->request_boundaries_.find(action.trigger_queue_index);
            if (it == this->request_boundaries_.end()) {
                reason = "unknown_trigger_request";
            } else {
                const RequestBoundaryInfo& boundaries = it->second;
                if (action.anchor_kind == "prefill_start") {
                    if (boundaries.prefill_done) {
                        anchor_tick = boundaries.prefill_start;
                    } else {
                        reason = "prefill_boundary_incomplete";
                    }
                } else if (action.anchor_kind == "decode_start") {
                    if (boundaries.decode_done) {
                        anchor_tick = boundaries.decode_start;
                    } else {
                        reason = "decode_boundary_incomplete";
                    }
                } else {
                    if (boundaries.completion_done) {
                        anchor_tick = boundaries.completion;
                    } else {
                        reason = "completion_boundary_incomplete";
                    }
                }
            }
        } else if (action.anchor_kind == "transfer_complete") {
            std::optional<Tick> rank_anchor;
            std::optional<Tick> subject_anchor;
            if (this->online_mode_) {
                const auto rank_it = this->online_transfer_anchor_ticks_.find(
                    TransferAnchorKey{action.trigger_queue_index, action.rank});
                if (rank_it != this->online_transfer_anchor_ticks_.end()) {
                    rank_anchor = rank_it->second;
                }
                const auto subject_it =
                    this->online_transfer_anchor_by_subject_.find(
                        action.trigger_queue_index);
                if (subject_it !=
                    this->online_transfer_anchor_by_subject_.end()) {
                    subject_anchor = subject_it->second;
                }
            } else {
                const auto rank_it = transfer_anchor_ticks.find(
                    std::make_pair(action.trigger_queue_index, action.rank));
                if (rank_it != transfer_anchor_ticks.end()) {
                    rank_anchor = rank_it->second;
                }
                const auto subject_it = transfer_anchor_by_subject.find(
                    action.trigger_queue_index);
                if (subject_it != transfer_anchor_by_subject.end()) {
                    subject_anchor = subject_it->second;
                }
            }
            if (rank_anchor.has_value()) {
                anchor_tick = rank_anchor;
            } else if (subject_anchor.has_value()) {
                // Not every ledger target rank owns an ET transfer node
                // (e.g. ranks outside the request's decode set). Fall back
                // to the request's last transfer node complete tick across
                // ranks (doc sec.7.8 transfer row), counted separately.
                anchor_tick = subject_anchor;
                transfer_anchor_fallback_by_rank[action.rank]++;
                totals.transfer_anchor_request_level_fallback++;
            } else {
                reason = "missing_transfer_anchor";
            }
        } else {
            reason = "unknown_anchor_kind";
        }

        if (!anchor_tick.has_value()) {
            totals.actions_unresolved++;
            totals.unresolved_by_reason[reason]++;
            unresolved_by_rank[action.rank]++;
            continue;
        }
        Tick tick = anchor_tick.value();
        if (tick > sim_end_tick) {
            // Observed boundary ticks should never exceed the sim window;
            // clamp defensively instead of corrupting the integral.
            tick = sim_end_tick;
        }
        deltas_by_rank[action.rank].push_back(
            AnchoredDelta{tick, action.sequence_index,
                          action.weight_delta_bytes,
                          action.resident_kv_delta_bytes,
                          action.reserved_kv_delta_bytes});
        quality_counts_by_rank[action.rank][action.anchor_quality]++;
        totals.actions_replayed++;
    }

    // Per-rank replay over [0, sim_end_ns] (doc sec.7.8). Resident covers
    // weight + resident_kv, committed covers weight + resident_kv +
    // reserved_kv (doc sec.3.8).
    std::map<int, WatermarkSeries> watermark_series_by_rank;
    for (auto& [rank, deltas] : deltas_by_rank) {
        std::sort(deltas.begin(), deltas.end(),
                  [](const AnchoredDelta& a, const AnchoredDelta& b) {
                      if (a.tick != b.tick) {
                          return a.tick < b.tick;
                      }
                      return a.sequence_index < b.sequence_index;
                  });

        // Per-rank capacity and final planner ledger from the peaks
        // payload. Resolved BEFORE the integrals: the WP8 watermark walk
        // needs the capacity for its violation counters too.
        const auto peak_it = peaks_by_rank.find(rank);
        std::optional<int64_t> capacity;
        std::optional<int64_t> planner_physical;
        std::optional<int64_t> planner_committed;
        if (peak_it != peaks_by_rank.end() &&
            peak_it->second->contains("ledger")) {
            const json& ledger = (*peak_it->second)["ledger"];
            capacity = ledger.value("capacity_bytes", int64_t(0));
            planner_physical =
                ledger.value("physical_used_bytes", int64_t(0));
            planner_committed =
                ledger.value("committed_used_bytes", int64_t(0));
        }

        __int128 weight = 0;
        __int128 resident = 0;
        __int128 reserved = 0;
        __int128 resident_area = 0;
        __int128 committed_area = 0;
        Tick previous = 0;
        for (const auto& delta : deltas) {
            const Tick elapsed = delta.tick - previous;
            resident_area += (weight + resident) * elapsed;
            committed_area += (weight + resident + reserved) * elapsed;
            weight += delta.weight_delta;
            resident += delta.resident_delta;
            reserved += delta.reserved_delta;
            previous = delta.tick;
        }
        const Tick tail = sim_end_tick - previous;
        resident_area += (weight + resident) * tail;
        committed_area += (weight + resident + reserved) * tail;

        // WP8 (CPP_SPEC §B): bucketed watermark sampling of the same step
        // function, filled by an independent walk that splits segments at
        // every delta tick (event-driven) AND every watermark bucket
        // boundary (periodic bottom line), so a bucket fully inside a long
        // constant segment still gets sampled. Peaks are sparse: buckets
        // that stayed at zero are not stored (emit treats missing as 0).
        {
            WatermarkSeries series;
            series.replayed = true;
            series.capacity_known = capacity.has_value();
            series.capacity_bytes = capacity.value_or(0);
            const uint64_t period =
                this->watermark_period_ns_ > 0 ? this->watermark_period_ns_ : 1;
            size_t next_delta = 0;
            Tick seg_start = 0;
            __int128 wm_weight = 0;
            __int128 wm_resident = 0;
            __int128 wm_reserved = 0;
            while (seg_start < sim_end_tick) {
                // Zero-length step points: deltas at/before seg_start
                // apply before the segment is evaluated.
                while (next_delta < deltas.size() &&
                       deltas[next_delta].tick <= seg_start) {
                    wm_weight += deltas[next_delta].weight_delta;
                    wm_resident += deltas[next_delta].resident_delta;
                    wm_reserved += deltas[next_delta].reserved_delta;
                    // A delta applying inside this bucket is occupancy
                    // activity even when the resulting value stays zero
                    // (e.g. a full release).
                    series.changed_buckets.insert(
                        deltas[next_delta].tick / period);
                    next_delta++;
                }
                Tick seg_end = sim_end_tick;
                if (next_delta < deltas.size() &&
                    deltas[next_delta].tick < seg_end) {
                    seg_end = deltas[next_delta].tick;
                }
                const uint64_t bucket = seg_start / period;
                const Tick bucket_end =
                    static_cast<Tick>(bucket + 1) * static_cast<Tick>(period);
                if (bucket_end < seg_end) {
                    seg_end = bucket_end;
                }
                if (seg_end <= seg_start) {
                    break;  // defensive: never expected
                }
                const Tick length = seg_end - seg_start;
                const __int128 resident_value = wm_weight + wm_resident;
                const __int128 committed_value =
                    wm_weight + wm_resident + wm_reserved;
                const __int128 resident_pos =
                    resident_value > 0 ? resident_value : 0;
                const __int128 committed_pos =
                    committed_value > 0 ? committed_value : 0;
                const uint64_t resident_peak =
                    resident_pos > static_cast<__int128>(
                                       std::numeric_limits<uint64_t>::max())
                        ? std::numeric_limits<uint64_t>::max()
                        : static_cast<uint64_t>(resident_pos);
                const uint64_t committed_peak =
                    committed_pos > static_cast<__int128>(
                                        std::numeric_limits<uint64_t>::max())
                        ? std::numeric_limits<uint64_t>::max()
                        : static_cast<uint64_t>(committed_pos);
                if (resident_peak > 0 || committed_peak > 0) {
                    std::pair<uint64_t, uint64_t>& peaks =
                        series.bucket_peaks[bucket];
                    if (resident_peak > peaks.first) {
                        peaks.first = resident_peak;
                    }
                    if (committed_peak > peaks.second) {
                        peaks.second = committed_peak;
                    }
                    if (resident_peak > series.resident_peak) {
                        series.resident_peak = resident_peak;
                    }
                    if (committed_peak > series.committed_peak) {
                        series.committed_peak = committed_peak;
                    }
                    // Nonzero occupancy is activity even without a delta
                    // inside this bucket (e.g. weights resident from the
                    // first delta on).
                    series.changed_buckets.insert(bucket);
                }
                series.resident_area += resident_pos * length;
                series.committed_area += committed_pos * length;
                if (series.capacity_known && series.capacity_bytes > 0 &&
                    (resident_value > series.capacity_bytes ||
                     committed_value > series.capacity_bytes)) {
                    series.capacity_violations++;
                }
                series.sample_count++;
                seg_start = seg_end;
            }
            series.direct_resident_area = resident_area;
            series.direct_committed_area = committed_area;
            watermark_series_by_rank[rank] = std::move(series);
        }

        json record;
        record["schema"] = 1;
        record["type"] = "capacity_timeavg";
        record["source"] = "planner_memory_ledger";
        record["repo_variant"] = this->repo_variant_;
        record["run_id"] = this->run_id_;
        record["rank"] = rank;
        record["scope"] = "npu_local_hbm";
        record["utilization_window"] = "[0, sim_end_ns]";
        record["utilization_window_ns"] = sim_end_tick;
        record["resident_definition"] = "weight + resident_kv";
        record["committed_definition"] =
            "weight + resident_kv + reserved_kv";
        record["capacity_source"] =
            "planner_memory_peaks.ledger.capacity_bytes";
        if (capacity.has_value()) {
            record["capacity_bytes"] = capacity.value();
        } else {
            record["capacity_bytes"] = nullptr;
        }
        record["resident_byte_ns"] = i128_to_string(resident_area);
        record["committed_byte_ns"] = i128_to_string(committed_area);
        if (capacity.has_value() && capacity.value() > 0 && sim_end_tick > 0) {
            const long double denominator =
                static_cast<long double>(capacity.value()) *
                static_cast<long double>(sim_end_tick);
            const long double resident_util =
                i128_to_long_double(resident_area) / denominator;
            const long double committed_util =
                i128_to_long_double(committed_area) / denominator;
            record["resident_capacity_timeavg_util"] =
                static_cast<double>(resident_util);
            record["committed_capacity_timeavg_util"] =
                static_cast<double>(committed_util);
            const std::string prefix =
                "rank " + std::to_string(rank) + ": ";
            check_consistency(
                resident_util >= -kUtilEpsilon &&
                    resident_util <= 1.0L + kUtilEpsilon,
                prefix + "resident_capacity_timeavg_util out of range: "
                "numerator=" + i128_to_string(resident_area) +
                    " denominator=capacity_bytes*window_ns");
            check_consistency(
                committed_util >= -kUtilEpsilon &&
                    committed_util <= 1.0L + kUtilEpsilon,
                prefix + "committed_capacity_timeavg_util out of range: "
                "numerator=" + i128_to_string(committed_area) +
                    " denominator=capacity_bytes*window_ns");
        } else {
            record["resident_capacity_timeavg_util"] = nullptr;
            record["committed_capacity_timeavg_util"] = nullptr;
            record["timeavg_util_note"] =
                "undefined: zero capacity/window or missing planner peaks "
                "entry for this rank";
        }
        record["actions_total"] = total_by_rank[rank];
        record["actions_replayed"] =
            static_cast<uint64_t>(deltas.size());
        record["actions_unresolved"] = unresolved_by_rank[rank];
        record["anchor_quality_counts"] = quality_counts_by_rank[rank];
        record["transfer_anchor_request_level_fallback"] =
            transfer_anchor_fallback_by_rank[rank];

        // Cross-check the replayed final ledger against the planner ledger.
        // Only meaningful when no action of this rank was skipped.
        record["replay_final_physical_used_bytes"] =
            i128_to_string(weight + resident);
        record["replay_final_committed_used_bytes"] =
            i128_to_string(weight + resident + reserved);
        if (unresolved_by_rank[rank] == 0 && planner_physical.has_value()) {
            const bool matches =
                weight + resident == planner_physical.value() &&
                weight + resident + reserved == planner_committed.value();
            record["planner_final_physical_used_bytes"] =
                planner_physical.value();
            record["planner_final_committed_used_bytes"] =
                planner_committed.value();
            record["final_state_matches_planner"] = matches;
            if (!matches) {
                totals.final_state_mismatches++;
                check_consistency(
                    false,
                    "rank " + std::to_string(rank) +
                        ": replayed final ledger differs from planner ledger "
                        "(replay physical=" +
                        i128_to_string(weight + resident) +
                        " planner physical=" +
                        std::to_string(planner_physical.value()) + ")");
            }
        } else {
            record["final_state_matches_planner"] = nullptr;
            record["final_state_check_note"] =
                "skipped: unresolved anchors or missing planner peaks entry";
        }
        emit_record(record.dump());
    }

    // WP8: extend the watermark universe with simulated ranks that have no
    // replayed deltas (capacity still picked up from the planner peaks
    // when present), then emit the watermark records.
    for (const int rank : fallback_ranks) {
        if (watermark_series_by_rank.count(rank) > 0) {
            continue;
        }
        WatermarkSeries series;
        series.replayed = false;
        const auto peak_it = peaks_by_rank.find(rank);
        if (peak_it != peaks_by_rank.end() &&
            peak_it->second->contains("ledger")) {
            series.capacity_bytes = (*peak_it->second)["ledger"].value(
                "capacity_bytes", int64_t(0));
            series.capacity_known = true;
        }
        watermark_series_by_rank.emplace(rank, std::move(series));
    }
    emit_watermark_records(watermark_series_by_rank, sim_end_tick,
                           full_detail);

    // Planner memory peaks passthrough (doc sec.7.7). Full detail only; the
    // summary record always carries the count. The chiplet breakdown is a
    // measurement-only equal-striping projection of the per-NPU aggregate
    // ledger (doc sec.7.6/16), not natively simulated chiplet hardware.
    if (full_detail) {
        std::vector<const json*> sorted_peaks;
        for (const auto& entry : this->planner_memory_peaks_) {
            sorted_peaks.push_back(&entry);
        }
        std::sort(sorted_peaks.begin(), sorted_peaks.end(),
                  [](const json* a, const json* b) {
                      return a->value("rank", -1) < b->value("rank", -1);
                  });
        for (const json* entry : sorted_peaks) {
            json record;
            record["schema"] = 1;
            record["type"] = "planner_memory_peaks";
            record["source"] = "planner_memory_ledger";
            record["repo_variant"] = this->repo_variant_;
            record["run_id"] = this->run_id_;
            record["rank"] = entry->value("rank", -1);
            record["scope"] = "npu_local_hbm";
            record["chiplet_scope"] = "physical_chiplet_projection";
            record["projection"] = "measurement_only_equal_striping";
            record["projection_note"] =
                "measurement-only equal striping of the per-NPU aggregate "
                "ledger; the simulator does not model per-chiplet "
                "allocators, fragmentation, placement feedback, or port "
                "contention (doc sec.7.6/16)";
            if (entry->contains("ledger")) {
                record["ledger"] = (*entry)["ledger"];
            }
            if (entry->contains("peak_physical")) {
                record["peak_physical"] = (*entry)["peak_physical"];
            }
            if (entry->contains("peak_committed")) {
                record["peak_committed"] = (*entry)["peak_committed"];
            }
            if (entry->contains("chiplets")) {
                record["chiplets"] = (*entry)["chiplets"];
            }
            emit_record(record.dump());
        }
    }

    return totals;
}

void MetricCollector::emit_watermark_records(
    const std::map<int, WatermarkSeries>& series_by_rank,
    Tick sim_end_tick, bool full_detail) {
    // WP8 (CPP_SPEC §B). Instance projection: ranks are grouped by the
    // manifest requests' instance assignments (prefill ranks -> prefill
    // instance, decode ranks -> decode instance; first assignment wins,
    // conflicts are counted, never silently resolved). Ranks no request
    // covers map to instance -1 -- online synthetic manifests carry only
    // placeholder instance-0 rank sets, so -1 is the honest label there.
    std::map<int, int64_t> instance_by_rank;
    uint64_t rank_instance_conflicts = 0;
    for (const auto& state : this->requests_) {
        for (const int rank : state.prefill_ranks) {
            const auto it = instance_by_rank.find(rank);
            if (it == instance_by_rank.end()) {
                instance_by_rank[rank] = state.prefill_instance;
            } else if (it->second != state.prefill_instance) {
                rank_instance_conflicts++;
            }
        }
        for (const int rank : state.decode_ranks) {
            const auto it = instance_by_rank.find(rank);
            if (it == instance_by_rank.end()) {
                instance_by_rank[rank] = state.decode_instance;
            } else if (it->second != state.decode_instance) {
                rank_instance_conflicts++;
            }
        }
    }
    const auto instance_of = [&instance_by_rank](int rank) -> int64_t {
        const auto it = instance_by_rank.find(rank);
        return it == instance_by_rank.end() ? int64_t(-1) : it->second;
    };

    // Per-(instance, bucket) peaks = max over the instance's ranks. Full
    // detail only (CPP_SPEC §B). Coordinator ruling 2026-08-26: a bucket
    // record is emitted only when the instance had occupancy activity in
    // it (some rank changed), plus the FIRST and LAST bucket of the window
    // which always emit -- empty middle buckets are omitted to keep the
    // log inside the WP6/WP8 disk budget.
    // D3 (2026-08-28): online mode skips the bucket lines entirely -- the
    // online synthetic manifests carry no planner ledger, so every bucket
    // is the documented flat-zero series whose authoritative WP8 source is
    // the ledger.jsonl replay (slo_tools/hbm_watermark.py header) and
    // metrics_postprocess consumes none of them. The per-rank summary
    // records below stay (they carry the "why zero" note).
    if (full_detail && !this->online_mode_) {
        const uint64_t period =
            this->watermark_period_ns_ > 0 ? this->watermark_period_ns_ : 1;
        const uint64_t last_bucket = sim_end_tick > 0
            ? (sim_end_tick - 1) / period
            : 0;
        std::map<std::pair<int64_t, uint64_t>, std::pair<uint64_t, uint64_t>>
            aggregated;
        std::map<int64_t, std::set<uint64_t>> changed_by_instance;
        std::set<int64_t> known_instances;
        for (const auto& [rank, series] : series_by_rank) {
            const int64_t instance = instance_of(rank);
            known_instances.insert(instance);
            auto& changed = changed_by_instance[instance];
            for (const uint64_t bucket : series.changed_buckets) {
                changed.insert(bucket);
            }
            for (const auto& [bucket, peaks] : series.bucket_peaks) {
                auto& agg = aggregated[{instance, bucket}];
                if (peaks.first > agg.first) {
                    agg.first = peaks.first;
                }
                if (peaks.second > agg.second) {
                    agg.second = peaks.second;
                }
            }
        }
        if (known_instances.empty()) {
            // No rank mapped at all (defensive): still emit the boundary
            // buckets under the unmapped-instance label.
            known_instances.insert(int64_t(-1));
        }
        for (const int64_t instance : known_instances) {
            const auto& changed = changed_by_instance[instance];
            auto emit_bucket = [&](uint64_t bucket) {
                const auto it = aggregated.find({instance, bucket});
                const std::pair<uint64_t, uint64_t> peaks =
                    it != aggregated.end()
                        ? it->second
                        : std::pair<uint64_t, uint64_t>{0, 0};
                json record;
                record["schema"] = 1;
                record["type"] = "hbm_watermark";
                record["source"] = "planner_memory_ledger";
                record["repo_variant"] = this->repo_variant_;
                record["run_id"] = this->run_id_;
                record["instance"] = instance;
                record["bucket_start_ns"] = bucket * period;
                record["resident_peak_bytes"] = peaks.first;
                record["committed_peak_bytes"] = peaks.second;
                record["watermark_period_ns"] = this->watermark_period_ns_;
                record["provisional"] = this->slo_sampling_provisional_;
                if (instance < 0) {
                    record["instance_note"] =
                        "rank->instance projection unavailable for this "
                        "rank (no manifest request covers it; online "
                        "synthetic manifests carry placeholder instance-0 "
                        "rank sets)";
                }
                emit_record(record.dump());
            };
            if (changed.empty()) {
                // Flat series (e.g. empty online planner ledger): only the
                // window boundary buckets, both zero, note on the summary
                // explains why.
                emit_bucket(0);
                emit_bucket(last_bucket);
                continue;
            }
            for (const uint64_t bucket : changed) {
                emit_bucket(bucket);
            }
            if (changed.count(0) == 0) {
                emit_bucket(0);
            }
            if (changed.count(last_bucket) == 0) {
                emit_bucket(last_bucket);
            }
        }
    }

    // Per-rank summary (every non-off detail level). violation counts must
    // be zero: a nonzero count is a simulation-correctness defect, output
    // as-is AND counted as a consistency violation.
    for (const auto& [rank, series] : series_by_rank) {
        json record;
        record["schema"] = 1;
        record["type"] = "hbm_watermark_summary";
        record["source"] = "planner_memory_ledger";
        record["repo_variant"] = this->repo_variant_;
        record["run_id"] = this->run_id_;
        record["rank"] = rank;
        record["watermark_period_ns"] = this->watermark_period_ns_;
        record["provisional"] = this->slo_sampling_provisional_;
        record["capacity_bytes"] =
            series.capacity_known ? json(series.capacity_bytes) : json(nullptr);
        record["resident_peak_bytes"] = series.resident_peak;
        record["committed_peak_bytes"] = series.committed_peak;
        record["nonzero_bucket_count"] = series.bucket_peaks.size();
        record["sample_count"] = series.sample_count;
        if (sim_end_tick > 0) {
            const long double window_ns =
                static_cast<long double>(sim_end_tick);
            record["resident_timeavg_bytes"] = static_cast<double>(
                i128_to_long_double(series.resident_area) / window_ns);
            record["committed_timeavg_bytes"] = static_cast<double>(
                i128_to_long_double(series.committed_area) / window_ns);
            if (series.capacity_known && series.capacity_bytes > 0) {
                const long double denominator =
                    static_cast<long double>(series.capacity_bytes) * window_ns;
                record["resident_capacity_timeavg_util"] = static_cast<double>(
                    i128_to_long_double(series.resident_area) / denominator);
                record["committed_capacity_timeavg_util"] =
                    static_cast<double>(i128_to_long_double(
                        series.committed_area) / denominator);
            } else {
                record["resident_capacity_timeavg_util"] = nullptr;
                record["committed_capacity_timeavg_util"] = nullptr;
            }
        } else {
            record["resident_timeavg_bytes"] = nullptr;
            record["committed_timeavg_bytes"] = nullptr;
            record["resident_capacity_timeavg_util"] = nullptr;
            record["committed_capacity_timeavg_util"] = nullptr;
        }

        // A-class cross-check: the watermark-walk integral must agree with
        // the direct delta-loop integral of the same replay to <=1%.
        if (series.replayed && sim_end_tick > 0) {
            const long double watermark =
                i128_to_long_double(series.resident_area);
            const long double direct =
                i128_to_long_double(series.direct_resident_area);
            const long double scale = direct > 0.0L ? direct : 1.0L;
            long double relative = (watermark - direct) / scale;
            if (relative < 0.0L) {
                relative = -relative;
            }
            record["timeavg_crosscheck_abs_relative_diff"] =
                static_cast<double>(relative);
            record["timeavg_crosscheck"] =
                relative > 0.01L ? "mismatch>1%" : "ok<=1%";
            check_consistency(
                relative <= 0.01L,
                "rank " + std::to_string(rank) +
                    ": WP8 watermark resident timeavg differs from the "
                    "capacity_timeavg replay integral by >1% (watermark=" +
                    i128_to_string(series.resident_area) + " direct=" +
                    i128_to_string(series.direct_resident_area) + ")");
        } else {
            record["timeavg_crosscheck"] =
                "skipped: no planner ledger replay for this rank";
        }

        record["capacity_violation_count"] = series.capacity_violations;
        if (series.capacity_violations > 0) {
            check_consistency(
                false,
                "rank " + std::to_string(rank) + ": WP8 watermark observed " +
                    std::to_string(series.capacity_violations) +
                    " sample points above the planner HBM capacity");
        }
        if (!series.replayed) {
            record["note"] =
                "no memory actions/planner peaks replayed for this rank; "
                "flat-zero watermark over [0, sim_end_ns]";
        }
        emit_record(record.dump());
    }
    (void)rank_instance_conflicts;
}
