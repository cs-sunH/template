/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/workload/MetricCollector.hh"

#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/CoarseDieRuntime.hh"
#include "astra-sim/workload/LocalHbmBandwidthModel.hh"
#include "astra-sim/workload/Statistics.hh"
#include "astra-sim/workload/Workload.hh"

#include <json/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
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

// Strict JSON load for the metrics manifest: istream operator>> parses in
// NON-strict mode -- it silently consumes the FIRST JSON value and leaves
// trailing bytes unread, and the DOM build last-wins on duplicate object
// keys. The manifest is machine-written by paired tools, so trailing
// content or a duplicate key is producer corruption, not a legal shape.
// json::parse in its default strict mode rejects trailing bytes outright;
// the stateful parse callback rejects a repeated key inside one object.
nlohmann::json strict_json_load(std::istream& in, const std::string& what) {
    struct KeyGuard {
        std::vector<std::set<std::string>> stack;
        std::string duplicate;
        bool operator()(int, nlohmann::json::parse_event_t event,
                        nlohmann::json& parsed) {
            using EVT = nlohmann::json::parse_event_t;
            if (event == EVT::object_start) {
                stack.emplace_back();
            } else if (event == EVT::key) {
                if (!stack.back().insert(parsed.get<std::string>()).second) {
                    duplicate = parsed.get<std::string>();
                }
            } else if (event == EVT::object_end && !stack.empty()) {
                stack.pop_back();
            }
            return true;  // observe only; never discard subtree
        }
    } guard;
    nlohmann::json value;
    try {
        value = nlohmann::json::parse(in, guard);
    } catch (const nlohmann::json::exception& e) {
        fatal_metrics_error("invalid JSON in " + what + ": " + e.what());
    }
    if (!guard.duplicate.empty()) {
        fatal_metrics_error("duplicate JSON key \"" + guard.duplicate +
                            "\" in " + what);
    }
    return value;
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

using ExactHbmMatrix = std::array<std::array<long double, 2>, 2>;

long double parse_runtime_exact_counter(const json& value,
                                        const std::string& field) {
    if (!value.is_string()) {
        throw std::runtime_error("coarse_die Runtime counter " + field +
                                 " is not an exact decimal string");
    }
    const std::string text = value.get<std::string>();
    char* end = nullptr;
    errno = 0;
    const long double parsed = std::strtold(text.c_str(), &end);
    if (errno == ERANGE || end != text.c_str() + text.size() ||
        !std::isfinite(parsed) || parsed < 0.0L) {
        throw std::runtime_error("coarse_die Runtime counter " + field +
                                 " is not a finite non-negative decimal");
    }
    return parsed;
}

std::string exact_runtime_decimal(long double value) {
    std::ostringstream stream;
    stream << std::setprecision(std::numeric_limits<long double>::max_digits10)
           << value;
    return stream.str();
}

ExactHbmMatrix parse_runtime_hbm_matrix(const json& value,
                                       const std::string& field) {
    if (!value.is_array() || value.size() != 2) {
        throw std::runtime_error("coarse_die Runtime counter " + field +
                                 " must have two rows");
    }
    ExactHbmMatrix matrix{};
    for (size_t row = 0; row < 2; ++row) {
        if (!value.at(row).is_array() || value.at(row).size() != 2) {
            throw std::runtime_error("coarse_die Runtime counter " + field +
                                     " has a malformed row");
        }
        for (size_t direction = 0; direction < 2; ++direction) {
            matrix[row][direction] = parse_runtime_exact_counter(
                value.at(row).at(direction), field + "[" +
                    std::to_string(row) + "][" +
                    std::to_string(direction) + "]");
        }
    }
    return matrix;
}

long double hbm_matrix_total(const ExactHbmMatrix& matrix) {
    long double total = 0.0L;
    for (const auto& row : matrix) {
        for (const long double direction : row) total += direction;
    }
    return total;
}

long double hbm_category_total(const json& runtime_counters,
                              const char* category) {
    const json& served = runtime_counters.at("served_by_category").at(category);
    return hbm_matrix_total(parse_runtime_hbm_matrix(
        served.at("hbm_served_bytes_by_row_direction_exact"),
        std::string("served_by_category.") + category +
            ".hbm_served_bytes_by_row_direction_exact"));
}

json coarse_die_runtime_metric_record(const json& runtime_counters,
                                      Tick window_ns) {
    const bool accounting_complete =
        runtime_counters.value("accounting_history_complete", false);
    const bool typed_complete =
        runtime_counters.value("typed_totals_history_complete", false);
    const bool history_complete = accounting_complete && typed_complete;
    const json& raw = runtime_counters.at("hardware_raw");
    const bool raw_available = raw.value("available", false);
    const long double window_seconds =
        static_cast<long double>(window_ns) / 1e9L;

    const json& categories = runtime_counters.at("served_by_category");
    const json& comp = categories.at("comp");
    const long double int4_ops = parse_runtime_exact_counter(
        comp.at("int4_ops_exact_decimal"),
        "served_by_category.comp.int4_ops_exact_decimal");
    const long double vec_eqops = parse_runtime_exact_counter(
        comp.at("vec_eqops_exact_decimal"),
        "served_by_category.comp.vec_eqops_exact_decimal");
    const ExactHbmMatrix total_hbm = parse_runtime_hbm_matrix(
        runtime_counters.at("hbm_served_bytes_by_row_direction_exact"),
        "hbm_served_bytes_by_row_direction_exact");
    const long double total_hbm_bytes = hbm_matrix_total(total_hbm);
    const auto& row_peaks = raw.at("hbm_row_peak_Bps");
    long double int4_denominator = 0.0L;
    long double vec_denominator = 0.0L;
    std::array<long double, 2> hbm_row_denominators{};
    bool int4_peak_valid = raw_available &&
        raw.at("peak_int4_ops_per_s").is_number();
    bool vec_peak_valid = raw_available &&
        raw.at("peak_vec_fp32_eq_ops_per_s").is_number();
    if (int4_peak_valid) {
        const long double peak = raw.at("peak_int4_ops_per_s").get<double>();
        int4_peak_valid = peak > 0.0L && std::isfinite(peak);
        int4_denominator = peak * window_seconds;
    }
    if (vec_peak_valid) {
        const long double peak =
            raw.at("peak_vec_fp32_eq_ops_per_s").get<double>();
        vec_peak_valid = peak > 0.0L && std::isfinite(peak);
        vec_denominator = peak * window_seconds;
    }
    bool hbm_peaks_valid = raw_available && row_peaks.is_array() &&
        row_peaks.size() == 2;
    std::array<long double, 2> row_peak_values{};
    if (hbm_peaks_valid) {
        for (size_t row = 0; row < 2; ++row) {
            if (!row_peaks.at(row).is_number()) {
                hbm_peaks_valid = false;
                break;
            }
            row_peak_values[row] = row_peaks.at(row).get<double>();
            if (!(row_peak_values[row] > 0.0L) ||
                !std::isfinite(row_peak_values[row])) {
                hbm_peaks_valid = false;
                break;
            }
            hbm_row_denominators[row] =
                row_peak_values[row] * window_seconds;
        }
    }

    const char* unavailable_reason = nullptr;
    if (!history_complete) {
        unavailable_reason = "incomplete_runtime_counter_history";
    } else if (window_ns == 0) {
        unavailable_reason = "zero_utilization_window";
    } else if (!raw_available) {
        unavailable_reason = "raw_hardware_denominators_unavailable";
    }
    const auto utilization = [history_complete, window_ns](long double numerator,
                                                             long double denominator,
                                                             bool valid_peak) {
        if (!history_complete || window_ns == 0 || !valid_peak ||
            !(denominator > 0.0L) || !std::isfinite(denominator)) {
            return json(nullptr);
        }
        return json(static_cast<double>(numerator / denominator));
    };

    json int4 = {
        {"numerator_ops_exact_decimal", exact_runtime_decimal(int4_ops)},
        {"denominator_ops", int4_denominator},
        {"denominator_definition", "raw_peak_int4_ops_per_s*window_seconds"},
        {"utilization", utilization(int4_ops, int4_denominator,
                                     int4_peak_valid)},
        {"status", !history_complete ? "incomplete_runtime_counter_history"
                   : window_ns == 0 ? "zero_utilization_window"
                   : !int4_peak_valid ? "raw_peak_unavailable"
                                      : "available"}};
    json vec = {
        {"numerator_eqops_exact_decimal", exact_runtime_decimal(vec_eqops)},
        {"denominator_eqops", vec_denominator},
        {"denominator_definition",
         "raw_peak_vec_fp32_eq_ops_per_s*window_seconds"},
        {"utilization", utilization(vec_eqops, vec_denominator,
                                     vec_peak_valid)},
        {"status", !history_complete ? "incomplete_runtime_counter_history"
                   : window_ns == 0 ? "zero_utilization_window"
                   : !vec_peak_valid ? "raw_peak_unavailable"
                                     : "available"}};
    json hbm_by_row = json::array();
    for (size_t row = 0; row < 2; ++row) {
        const long double row_bytes = total_hbm[row][0] + total_hbm[row][1];
        hbm_by_row.push_back({
            {"row", row},
            {"numerator_bytes_exact_decimal", exact_runtime_decimal(row_bytes)},
            {"denominator_bytes", hbm_row_denominators[row]},
            {"denominator_definition",
             "raw_hbm_row_peak_Bps*window_seconds"},
            {"utilization", utilization(row_bytes,
                                         hbm_row_denominators[row],
                                         hbm_peaks_valid)},
            {"status", !history_complete ? "incomplete_runtime_counter_history"
                       : window_ns == 0 ? "zero_utilization_window"
                       : !hbm_peaks_valid ? "raw_peak_unavailable"
                                          : "available"}});
    }
    const long double hbm_denominator =
        hbm_row_denominators[0] + hbm_row_denominators[1];
    const json hbm_total_util = utilization(total_hbm_bytes, hbm_denominator,
                                            hbm_peaks_valid);
    json hbm_total = {
        {"numerator_bytes_exact_decimal",
         exact_runtime_decimal(total_hbm_bytes)},
        {"denominator_bytes", hbm_denominator},
        {"denominator_definition",
         "sum(raw_hbm_row_peak_Bps)*window_seconds"},
        {"utilization", hbm_total_util},
        {"status", !history_complete ? "incomplete_runtime_counter_history"
                   : window_ns == 0 ? "zero_utilization_window"
                   : !hbm_peaks_valid ? "raw_peak_unavailable"
                                      : "available"}};

    return {{"source", "coarse_die::Runtime::counters"},
            {"counter_scope",
             "rank_lifetime_served_work_including_active_cancelled_and_retired_executions"},
            {"utilization_window_ns", window_ns},
            {"utilization_window", "[0, sim_end_ns]"},
            {"accounting_history_complete", accounting_complete},
            {"typed_totals_history_complete", typed_complete},
            {"totals_semantics", runtime_counters.value(
                 "served_category_totals_semantics", "unknown")},
            {"served_by_category", categories},
            {"hbm_direction_order",
             runtime_counters.value("hbm_direction_order", json::array())},
            {"hbm_served_bytes_by_row_direction_exact",
             runtime_counters.at("hbm_served_bytes_by_row_direction_exact")},
            {"legacy_unclassified_hbm_bytes_by_row_direction_exact",
             runtime_counters.at(
                 "legacy_unclassified_hbm_bytes_by_row_direction_exact")},
            {"raw_hardware_denominators", raw},
            {"utilization_status", unavailable_reason == nullptr
                 ? "available" : unavailable_reason},
            {"utilization", {{"int4", std::move(int4)},
                             {"vec_fp32_eq", std::move(vec)},
                             {"hbm_total", std::move(hbm_total)},
                             {"hbm_by_row", std::move(hbm_by_row)}}}};
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
    // C2 (2026-09-28): snapshot for the finalize cross-reference. The online
    // driver registers CSV rows after this point, so the summary's
    // input_requests legitimately differs in online mode; see the summary
    // emission (input_requests_at_init).
    this->input_requests_at_initialize_ = this->requests_.size();
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
            "and the WP6 link bucket; batch B4 has landed and normally fills "
            "both with derived values";
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
    json manifest = strict_json_load(manifest_file, "metrics manifest " +
                                                          manifest_path);

    // Nesting-depth gate: the json::value(key, default) reads below copy
    // the stored value through nlohmann's RECURSIVE copy constructor, so a
    // hostile manifest with ~1100 nested containers overflows the stack
    // inside an uncancellable copy (no exception reaches the catch, no
    // [METRIC][ERROR] record, silent stderr death). Walk the parsed
    // document ITERATIVELY (explicit stack -- this walk itself must not
    // recurse) and fail-closed past a bound far above any legitimate
    // manifest (the schema's deepest legitimate node is a handful of
    // levels).
    {
        constexpr size_t kMaxManifestDepth = 64;
        std::vector<std::pair<const json*, size_t>> pending;
        pending.emplace_back(&manifest, 0);
        while (!pending.empty()) {
            const json* node = pending.back().first;
            const size_t depth = pending.back().second;
            pending.pop_back();
            if (node->is_array()) {
                if (depth >= kMaxManifestDepth) {
                    fatal_metrics_error(
                        "metrics manifest nesting depth exceeds " +
                        std::to_string(kMaxManifestDepth) + " (" +
                        manifest_path + ")");
                }
                for (const json& child : *node) {
                    pending.emplace_back(&child, depth + 1);
                }
            } else if (node->is_object()) {
                if (depth >= kMaxManifestDepth) {
                    fatal_metrics_error(
                        "metrics manifest nesting depth exceeds " +
                        std::to_string(kMaxManifestDepth) + " (" +
                        manifest_path + ")");
                }
                for (auto it = node->begin(); it != node->end(); ++it) {
                    pending.emplace_back(&it.value(), depth + 1);
                }
            }
        }
    }

    // Top-level metadata: guard the manifest's own type BEFORE any value()
    // read -- a top level that is not a JSON object (e.g. a bare array)
    // would throw a nlohmann type_error that no caller of load_manifest
    // catches (std::terminate), bypassing this file's own fail-closed
    // channel (fatal_metrics_error). Then guard each key's type BEFORE
    // .value() -- a wrong-typed key would throw the same uncaught
    // type_error. Same guard pattern as the optional keys below.
    if (!manifest.is_object()) {
        fatal_metrics_error(
            "metrics manifest top level must be a JSON object");
    }
    if (manifest.contains("schema_version") &&
        !manifest["schema_version"].is_number_integer()) {
        fatal_metrics_error(
            "metrics manifest schema_version must be an integer");
    }
    this->schema_version_ = manifest.value("schema_version", 0);
    if (this->schema_version_ != 1) {
        // Doc sec.12.1: unsupported schema versions must be rejected.
        fatal_metrics_error(
            "unsupported metrics manifest schema_version: " +
            std::to_string(this->schema_version_) + " (expected 1)");
    }
    if (manifest.contains("repo_variant") &&
        !manifest["repo_variant"].is_string()) {
        fatal_metrics_error("metrics manifest repo_variant must be a string");
    }
    this->repo_variant_ = manifest.value("repo_variant", "unknown");
    if (manifest.contains("run_mode") && !manifest["run_mode"].is_string()) {
        fatal_metrics_error("metrics manifest run_mode must be a string");
    }
    this->run_mode_ = manifest.value("run_mode", "service");
    if (manifest.contains("run_id") && !manifest["run_id"].is_string()) {
        fatal_metrics_error("metrics manifest run_id must be a string");
    }
    this->run_id_ = manifest.value("run_id", "");
    if (manifest.contains("npus_count") &&
        manifest["npus_count"].is_number_integer()) {
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
    // has landed and fills the anchors with derived values, and every
    // related record echoes the period actually used so the fallback is
    // downstream-visible).
    if (manifest.contains("slo_sampling") &&
        manifest["slo_sampling"].is_object()) {
        const json& slo = manifest["slo_sampling"];
        const auto read_ns = [&slo](const char* key) -> std::optional<uint64_t> {
            // Integer-only (same full-width discipline as the rank elements
            // and node events): a float 5000000.9 would silently truncate
            // in get<int64_t> under a wide is_number() gate.
            if (!slo.contains(key) || !slo[key].is_number_integer()) {
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

    // Requests (doc sec.4.2). queue_index must be unique (doc sec.4.5) and
    // request_id must be unique too: the online registration path keys
    // rank/anchor lookups by request_id (request_index_by_request_id_,
    // last-wins on re-registration), so two manifest entries sharing a
    // request_id would silently hang the online anchors and watch rank sets
    // on whichever entry registered last.
    const json requests = manifest.value("requests", json::array());
    if (!requests.is_array()) {
        fatal_metrics_error("metrics manifest requests must be an array");
    }
    std::unordered_set<std::string> seen_request_ids;
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
            // Type-guard the rank sets BEFORE iterating: nlohmann iterates a
            // JSON scalar as a single-element sequence, so a scalar
            // prefill_ranks/decode_ranks (e.g. 3) would silently become the
            // one-element set {3} -- the request could then never satisfy
            // its expected-rank completion checks and would be reported
            // incomplete with zero diagnostics. Elements must be integers
            // too (same wide-domain discipline as the node-event codes
            // below: a float like 3.5 would silently truncate in get<int>).
            if (!entry.contains("prefill_ranks") ||
                !entry["prefill_ranks"].is_array()) {
                fatal_metrics_error(
                    "metrics manifest request prefill_ranks must be an array "
                    "(queue_index " + std::to_string(state.queue_index) + ")");
            }
            if (!entry.contains("decode_ranks") ||
                !entry["decode_ranks"].is_array()) {
                fatal_metrics_error(
                    "metrics manifest request decode_ranks must be an array "
                    "(queue_index " + std::to_string(state.queue_index) + ")");
            }
            for (const auto& rank : entry["prefill_ranks"]) {
                if (!rank.is_number_integer()) {
                    fatal_metrics_error(
                        "metrics manifest request prefill_ranks elements must "
                        "be integers (queue_index " +
                        std::to_string(state.queue_index) + ")");
                }
                // Validate in the wide int64 domain BEFORE narrowing (same
                // discipline as the node-event codes below): >2^31-1 would
                // silently alias to another rank in get<int> and a negative
                // rank would poison the completion checks.
                const int64_t rank_wide = rank.get<int64_t>();
                if (rank_wide < 0 || rank_wide > 2147483647LL) {
                    fatal_metrics_error(
                        "metrics manifest request prefill_ranks element out "
                        "of int rank domain [0, 2^31-1] (queue_index " +
                        std::to_string(state.queue_index) + "): " +
                        std::to_string(rank_wide));
                }
                state.prefill_ranks.insert(static_cast<int>(rank_wide));
            }
            for (const auto& rank : entry["decode_ranks"]) {
                if (!rank.is_number_integer()) {
                    fatal_metrics_error(
                        "metrics manifest request decode_ranks elements must "
                        "be integers (queue_index " +
                        std::to_string(state.queue_index) + ")");
                }
                const int64_t rank_wide = rank.get<int64_t>();
                if (rank_wide < 0 || rank_wide > 2147483647LL) {
                    fatal_metrics_error(
                        "metrics manifest request decode_ranks element out "
                        "of int rank domain [0, 2^31-1] (queue_index " +
                        std::to_string(state.queue_index) + "): " +
                        std::to_string(rank_wide));
                }
                state.decode_ranks.insert(static_cast<int>(rank_wide));
            }
            const auto& arrival = entry.at("arrival");
            const std::string kind = arrival.at("kind").get<std::string>();
            // Validate in the wide int64 domain BEFORE narrowing (same
            // discipline as the node-event codes below): a negative
            // value_ns/interval_ns would silently wrap into a huge Tick.
            if (kind == "absolute") {
                state.arrival.kind = ArrivalSpec::Kind::ABSOLUTE;
                const int64_t value_ns_wide =
                    arrival.at("value_ns").get<int64_t>();
                if (value_ns_wide < 0) {
                    fatal_metrics_error("negative arrival value_ns: " +
                                        std::to_string(value_ns_wide));
                }
                state.arrival.value_ns =
                    static_cast<Tick>(value_ns_wide);
            } else if (kind == "after_request") {
                state.arrival.kind = ArrivalSpec::Kind::AFTER_REQUEST;
                state.arrival.parent_queue_index =
                    arrival.at("parent_queue_index").get<int64_t>();
                const int64_t interval_ns_wide =
                    arrival.at("interval_ns").get<int64_t>();
                if (interval_ns_wide < 0) {
                    fatal_metrics_error("negative arrival interval_ns: " +
                                        std::to_string(interval_ns_wide));
                }
                state.arrival.interval_ns =
                    static_cast<Tick>(interval_ns_wide);
            } else {
                fatal_metrics_error("unknown arrival kind: " + kind);
            }
            // WP9 (CPP_SPEC §C): optional request decode length for the
            // first-token invariant; synthetic online manifests may omit
            // it, in which case finalize skips the invariant and notes
            // manifest_decode_length_missing.
            if (entry.contains("decode_length") &&
                entry["decode_length"].is_number_integer()) {
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
        if (!seen_request_ids.insert(state.request_id).second) {
            fatal_metrics_error("duplicate request_id in manifest: " +
                                state.request_id);
        }
        this->request_index_by_queue_index_[state.queue_index] =
            this->requests_.size();
        this->requests_.push_back(std::move(state));
    }

    // Sparse node events (doc sec.4.3): [node_id, event_code, subject_id].
    const json events_by_rank =
        manifest.value("node_events_by_rank", json::object());
    if (!events_by_rank.is_object()) {
        fatal_metrics_error(
            "metrics manifest node_events_by_rank must be an object");
    }
    for (auto it = events_by_rank.begin(); it != events_by_rank.end(); ++it) {
        // Full-width strict parse (same fail-closed discipline as the event
        // codes below): std::stoi would silently accept trailing garbage
        // (e.g. "12abc" -> rank 12) and attach the events to the wrong
        // rank bucket, and a negative key would build a bucket no node ever
        // matches (anchors silently lost, no dropped_events_ accounting).
        int rank = 0;
        const char* key_first = it.key().data();
        const char* key_last = key_first + it.key().size();
        const std::from_chars_result rank_result =
            std::from_chars(key_first, key_last, rank);
        if (rank_result.ec != std::errc() || rank_result.ptr != key_last ||
            rank < 0) {
            fatal_metrics_error("invalid rank key in node_events_by_rank: "
                                + it.key());
        }
        if (!it.value().is_array()) {
            fatal_metrics_error("node events for rank " + it.key() +
                                " must be an array");
        }
        for (const auto& triple : it.value()) {
            // Type-guard every element BEFORE get<int64_t>: a string/bool/
            // null/array element would throw a nlohmann type_error that no
            // caller catches (std::terminate), bypassing the fail-closed
            // channel. Same guard pattern as the top-level metadata above;
            // short-circuit keeps triple[x] inside the size==3 domain.
            if (!triple.is_array() || triple.size() != 3 ||
                !triple[0].is_number_integer() ||
                !triple[1].is_number_integer() ||
                !triple[2].is_number_integer()) {
                fatal_metrics_error(
                    "node event must be a [node_id, event_code, subject_id] "
                    "triple of integers");
            }
            // Validate in the wide int64 domain BEFORE narrowing: an event
            // code like 256 wraps to a legal-looking uint8_t and a negative
            // node id wraps to a huge uint64_t, so both must be rejected at
            // full width (a wrapped code would silently attach the metric to
            // the wrong event edge).
            const int64_t node_id_wide = triple[0].get<int64_t>();
            if (node_id_wide < 0) {
                fatal_metrics_error("negative node id in node event: " +
                                    std::to_string(node_id_wide));
            }
            const uint64_t node_id = static_cast<uint64_t>(node_id_wide);
            const int64_t event_code_wide = triple[1].get<int64_t>();
            if (event_code_wide < 1 || event_code_wide > 8) {
                fatal_metrics_error("invalid node event code: " +
                                    std::to_string(event_code_wide));
            }
            const uint8_t event_code =
                static_cast<uint8_t>(event_code_wide);
            const int64_t subject_id = triple[2].get<int64_t>();
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
    // sequence_index must be strictly increasing in array order: the array
    // is a projection of the planner's ledger journal, and the authoritative
    // Python replay (slo_tools/hbm_watermark.py) fail-closes on a sequence
    // regression ("sequence 回退") -- a manifest whose rows are out of
    // sequence order is the same ledger corruption and must not be silently
    // re-sorted into a clean-looking replay here.
    const json memory_actions = manifest.value("memory_actions", json::array());
    if (!memory_actions.is_array()) {
        fatal_metrics_error(
            "metrics manifest memory_actions must be an array");
    }
    int64_t previous_sequence_index = -1;
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
            // Wide-domain gate before narrowing (same discipline as the
            // request rank sets above): a negative rank would poison the
            // per-rank anchors and >2^31-1 would silently alias to another
            // rank in get<int>.
            const int64_t memory_action_rank_wide =
                entry.at("rank").get<int64_t>();
            if (memory_action_rank_wide < 0 ||
                memory_action_rank_wide > 2147483647LL) {
                fatal_metrics_error(
                    "memory action rank out of int domain [0, 2^31-1]: " +
                    std::to_string(memory_action_rank_wide));
            }
            action.rank = static_cast<int>(memory_action_rank_wide);
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
        if (action.sequence_index <= previous_sequence_index) {
            fatal_metrics_error(
                "memory_actions sequence_index is not strictly increasing: " +
                std::to_string(action.sequence_index) + " after " +
                std::to_string(previous_sequence_index) +
                " (ledger journal order corruption; the Python journal replay "
                "fail-closes on the same stream)");
        }
        previous_sequence_index = action.sequence_index;
        this->memory_actions_.push_back(std::move(action));
    }

    // Planner memory peaks (doc sec.7.7), kept raw for passthrough with
    // annotations at finalize. Type-guard HERE, at the single write point
    // of planner_memory_peaks_, so every later value("rank")/
    // ledger.value(...) read (peaks_by_rank, the ledger capacity lookup,
    // the passthrough sort comparator) is safe by construction -- a
    // wrong-typed field would otherwise throw a nlohmann type_error at
    // finalize that nothing catches (std::terminate, after the whole
    // simulation has already run), bypassing the fail-closed channel.
    const json peaks = manifest.value("planner_memory_peaks", json::array());
    if (!peaks.is_array()) {
        fatal_metrics_error(
            "metrics manifest planner_memory_peaks must be an array");
    }
    for (const auto& entry : peaks) {
        if (!entry.is_object()) {
            fatal_metrics_error(
                "planner_memory_peaks entry must be an object");
        }
        if (entry.contains("rank") && !entry["rank"].is_number_integer()) {
            fatal_metrics_error(
                "planner_memory_peaks entry rank must be an integer");
        }
        // Wide-domain gate at this single load point (wl-F4 fix, same
        // discipline as the request rank sets and memory_actions above):
        // every later value("rank", -1) read (peaks_by_rank, the finalize
        // annotations, the passthrough comparator) is safe by construction;
        // a negative rank would only ever match the finalize-time ghost
        // consistency check, after the whole run -- a negative peak entry
        // is load-time corruption and fails closed here.
        if (entry.contains("rank") &&
            (entry["rank"].get<int64_t>() < 0 ||
             entry["rank"].get<int64_t>() > 2147483647LL)) {
            fatal_metrics_error(
                "planner_memory_peaks entry rank out of int domain "
                "[0, 2^31-1]: " +
                std::to_string(entry["rank"].get<int64_t>()));
        }
        if (entry.contains("ledger")) {
            const json& ledger = entry["ledger"];
            if (!ledger.is_object()) {
                fatal_metrics_error(
                    "planner_memory_peaks ledger must be an object");
            }
            for (const char* ledger_key : {"capacity_bytes",
                                           "physical_used_bytes",
                                           "committed_used_bytes"}) {
                if (ledger.contains(ledger_key) &&
                    !ledger[ledger_key].is_number_integer()) {
                    fatal_metrics_error(
                        "planner_memory_peaks ledger field must be an "
                        "integer: " + std::string(ledger_key));
                }
            }
        }
        this->planner_memory_peaks_.push_back(entry);
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
    // R2 one-shot erase: the semantics (why every (rank, node) edge fires
    // at most once, and which finalize consumers are NOT touched) are
    // documented on the on_node_issue/on_node_complete header comment in
    // MetricCollector.hh; just drop the routing bucket here.
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
            // Unreachable today (the outer switch narrowed the code set);
            // kept as the -Wswitch guard for future EventCode additions.
            break;
        }
        return;
    }
    case EventCode::MEMORY_ANCHOR_COMPLETE: {
        const MemoryAnchorTick anchor{event.subject_id, rank, node_id, tick};
        if (this->online_mode_) {
            append_online_memory_anchor(anchor);
        } else {
            // Static-path baseline (testOnly): the online main always
            // clears the static node-event tables right after initialize,
            // so only the metric regression fixtures drive this in-memory
            // vector arm; it is kept for the static/online
            // byte-equivalence red line (metric_collector_anchor_spool_test).
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
    default:
        // Codes 5/6 are schema-frozen but have no live producer in this
        // repo (see EventCode); they are accepted and intentionally
        // ignored (no dropped_events_ accounting, matching the old
        // no-op-without-subject behavior).
        break;
    }
}

void MetricCollector::on_compute_issue(int rank, uint64_t num_ops,
                                       uint64_t local_tensor_bytes) {
    RankMetricState& state = this->rank_states_[rank];
    state.total_num_ops += num_ops;
    state.total_local_bytes += local_tensor_bytes;
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
        // F1-CPP C1-4 fix (2026-09-29): a fresh queue_index whose request_id
        // is ALREADY bound (to a different queue index -- impossible to
        // rebind to the same one, or this queue_index would have been found
        // above) is a cross-source identity collision: the manifest or an
        // earlier dynamic registration already owns "req-A" at queue 0, and
        // this row claims "req-A" at queue 5. The unconditional insert below
        // used to rebind request_index_by_request_id_ last-wins, silently
        // moving every "req-A" node/rank anchor to the wrong subject while
        // the real subject's stage boundaries starved (completed=false +
        // misattributed accounting). Same fail-closed contract and channel
        // as load_manifest's duplicate request_id gate: the official
        // materializer guarantees manifest and CSV bindings are consistent,
        // so reaching this point means the inputs disagree and the run must
        // not report numbers.
        const auto id_binding =
            this->request_index_by_request_id_.find(request_id);
        if (id_binding != this->request_index_by_request_id_.end()) {
            fatal_metrics_error(
                "duplicate request_id across sources: '" + request_id +
                "' is already bound to queue_index " +
                std::to_string(
                    this->requests_[id_binding->second].queue_index) +
                "; refusing to rebind it to queue_index " +
                std::to_string(queue_index));
        }
        // Fresh dynamic request (the manifest already covered the static
        // requests; anything not covered is an online-injected one), now
        // guaranteed to own its request_id binding exclusively.
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
    // C1-4 sibling (2026-09-30): the same queue_index arriving under a
    // DIFFERENT request_id is the same-source twin of the cross-source
    // identity collision gated above -- the row would silently rebind the
    // queue's record to the new id while the old reverse-lookup key keeps
    // pointing at this queue_index (anchors registered under the old id
    // keep routing here). Fail closed on the same channel instead.
    if (state.request_id != request_id) {
        fatal_metrics_error(
            "request identity drift on re-registration: queue_index " +
            std::to_string(queue_index) + " is already bound to '" +
            state.request_id + "'; refusing to rebind it to '" +
            request_id + "'");
    }
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
    // Return-value semantics are documented on the declaration in
    // MetricCollector.hh.
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
        // The direct write(2) bypasses the stdout FILE buffer, so drain
        // that buffer first to keep the log's FIFO record order (an
        // unflushed earlier log line must not land after this [METRIC]
        // block).
        std::fflush(stdout);
        write_all_or_die(this->emit_buffer_.data(), this->emit_buffer_.size());
        this->emit_buffer_.clear();
    }
}

void MetricCollector::flush_emit_buffer() const {
    // C5 (2026-08-28): drain the buffered [METRIC] channel. finalize() calls
    // this at its end; the link-observer emission path calls it once after its
    // last record. Same stdout-FILE-buffer ordering guard as emit_record.
    if (this->emit_buffer_.empty()) {
        return;
    }
    std::fflush(stdout);
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

        json coarse_counters;
        json coarse_metrics;
        if (sys->coarse_die_runtime != nullptr) {
            coarse_counters = sys->coarse_die_runtime->counters_at(sim_end_tick);
            coarse_metrics = coarse_die_runtime_metric_record(coarse_counters,
                                                              sim_end_tick);
        }
        json record;
        record["schema"] = 1;
        record["type"] = "rank_compute";
        record["source"] = "simulator";
        record["repo_variant"] = this->repo_variant_;
        record["run_id"] = this->run_id_;
        record["rank"] = sys->id;
        record["utilization_window_ns"] = sim_end_tick;
        record["utilization_window"] = "[0, sim_end_ns]";
        record["die_cancelled_comp_count"] =
            sys->workload->die_cancelled_comp_count();
        if (!coarse_counters.is_null()) {
            record["coarse_die_runtime"] = coarse_metrics;
        }
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
        // This repo has no local HBM restore model, so this stays 0 (kept
        // for cross-repo [METRIC] output schema stability).
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
        // Domain assumption (wl-F8, registered): the numerator counts
        // tensor bytes booked at COMP issue time. A COMP node carrying a
        // calibrated runtime_ns completes through the closed-form event and
        // its tensor bytes BYPASS the local HBM fluid model, so a
        // hand-built GraphBatch mixing calibrated nodes whose tensor_size
        // exceeds what could stream through the HBM within their
        // runtime_ns window can raise this check as a FALSE POSITIVE (the
        // model's own "bytes must fit the window" assumption is what the
        // mixed input violates). Production producers never write
        // runtime_ns (all COMP nodes take the fluid path), so official runs
        // cannot hit this; the violation is recorded in consistency, never
        // fatal.
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
            // total_comp_time carries a permanent 1ns floor (the accumulator
            // struct's initial value, mirrored by the online running total;
            // both only ever grow), so it is strictly positive in every
            // reachable state -- the former null+"zero total comp time" arm
            // here was dead code and has been removed. Actual behavior: a
            // roofline-enabled rank with no completed comp exports 0.0
            // (zero weighted sums over the 1ns floor), indistinguishable
            // from a genuinely zero utilization. That is the export
            // contract (a null-signal redesign would be a downstream-visible
            // change and is deliberately not done).
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

        // Die-backend active-kernel utilization (contract §7.6 caliber):
        // the same duration-weighted window aggregation as the roofline
        // pair above, populated by DieExecution::fill_utilizations --
        // compute = PE FLOP / (PE_peak_die * node elapsed), memory =
        // actual bytes / (B_mem * node elapsed), vector = eq-ops /
        // (V_peak * node elapsed).  Gated on the die backend itself (the
        // roofline keys above keep their own legacy gate untouched), so a
        // legacy run reports them null with a note instead.
        record["active_kernel_die_source"] = "simulator";
        record["active_kernel_die_definition"] =
            "duration-weighted average of die per-node utilization over comp "
            "nodes clipped to the utilization window (contract §7.6: "
            "compute=PE FLOP/(PE_peak_die*elapsed), memory=actual bytes/"
            "(B_mem*elapsed), vector=eq-ops/(V_peak*elapsed))";
        if (sys->compute_cost_model == ComputeCostModel::Die &&
            sim_end_tick > 0) {
            const auto windowed =
                stats->calculate_roofline_utilization_in_window(0,
                                                                sim_end_tick);
            if (windowed.total_comp_time > 0) {
                const double die_compute =
                    windowed.compute_utilization_weighted_sum /
                    static_cast<double>(windowed.total_comp_time);
                const double die_memory =
                    windowed.memory_utilization_weighted_sum /
                    static_cast<double>(windowed.total_comp_time);
                const double die_vector =
                    windowed.vector_utilization_weighted_sum /
                    static_cast<double>(windowed.total_comp_time);
                record["active_kernel_die_compute_util"] = die_compute;
                record["active_kernel_die_memory_util"] = die_memory;
                record["active_kernel_die_vector_util"] = die_vector;
                check_consistency(
                    die_compute >= -kUtilEpsilon &&
                        die_compute <= 1.0L + kUtilEpsilon,
                    "rank " + std::to_string(sys->id) +
                        " active_kernel_die_compute_util out of range");
                check_consistency(
                    die_memory >= -kUtilEpsilon &&
                        die_memory <= 1.0L + kUtilEpsilon,
                    "rank " + std::to_string(sys->id) +
                        " active_kernel_die_memory_util out of range");
                check_consistency(
                    die_vector >= -kUtilEpsilon &&
                        die_vector <= 1.0L + kUtilEpsilon,
                    "rank " + std::to_string(sys->id) +
                        " active_kernel_die_vector_util out of range");
            } else {
                record["active_kernel_die_compute_util"] = nullptr;
                record["active_kernel_die_memory_util"] = nullptr;
                record["active_kernel_die_vector_util"] = nullptr;
                record["active_kernel_die_note"] =
                    "unavailable: no comp nodes in the utilization window";
            }
        } else {
            record["active_kernel_die_compute_util"] = nullptr;
            record["active_kernel_die_memory_util"] = nullptr;
            record["active_kernel_die_vector_util"] = nullptr;
            record["active_kernel_die_note"] =
                "unavailable: compute-cost-model is not die";
        }
        emit_record(record.dump());

        // Multi-user local-HBM bandwidth contention export (sh_2.0
        // local_hbm record pattern, generalized to N users): when this rank
        // runs the LocalHbmBandwidthModel (hbm-bandwidth-contention),
        // report the runtime model's own read-only counters.  The
        // plain roofline_local_bytes record above is still emitted
        // unchanged (dram_bw_util and every legacy key keep their meaning).
        const LocalHbmBandwidthModel* hbm_model =
            sys->workload->local_hbm_bandwidth_model.get();
        if (!coarse_counters.is_null()) {
            const auto category_direction = [&](const char* category,
                                                  size_t direction) {
                const ExactHbmMatrix matrix = parse_runtime_hbm_matrix(
                    coarse_counters.at("served_by_category").at(category)
                        .at("hbm_served_bytes_by_row_direction_exact"), category);
                return matrix[0][direction] + matrix[1][direction];
            };
            const long double comp_read = category_direction("comp", 0) +
                                          category_direction("die", 0);
            const long double comp_write = category_direction("comp", 1) +
                                           category_direction("die", 1);
            const auto& hbm_total = coarse_metrics.at("utilization").at("hbm_total");
            const auto& raw = coarse_counters.at("hardware_raw");
            double raw_bandwidth = 0.0;
            if (raw.value("available", false)) {
                const auto& peaks = raw.at("hbm_row_peak_Bps");
                raw_bandwidth = peaks[0].get<double>() + peaks[1].get<double>();
            }
            json hbm_record = {
                {"schema", 1}, {"type", "local_hbm"},
                {"source", "shared_coarse_die_runtime_model"},
                {"model", "13-domain per-rank coarse Runtime"},
                {"repo_variant", this->repo_variant_}, {"run_id", this->run_id_},
                {"rank", sys->id}, {"utilization_window_ns", sim_end_tick},
                {"utilization_window", "[0, sim_end_ns]"},
                {"hbm_busy_ns", coarse_counters.at("hbm_busy_ns")},
                {"hbm_busy_definition", "union of intervals with positive HBM service rate"},
                {"hbm_shared_ns", coarse_counters.at("hbm_shared_ns")},
                {"shared_definition", "at least two distinct owners with positive service on the same HBM row"},
                // Preserve the scalar-queue COMP category; typed DIE work is
                // separate and contributes to the global total exactly once.
                {"compute_bytes_served", hbm_model ? hbm_model->compute_bytes_served() : 0.0},
                {"die_compute_read_bytes_served", static_cast<double>(comp_read)},
                {"die_compute_write_bytes_served", static_cast<double>(comp_write)},
                {"die_compute_read_bytes_served_exact_decimal", exact_runtime_decimal(comp_read)},
                {"die_compute_write_bytes_served_exact_decimal", exact_runtime_decimal(comp_write)},
                {"comm_read_bytes_served", static_cast<double>(hbm_category_total(coarse_counters, "comm_send"))},
                {"comm_write_bytes_served", static_cast<double>(hbm_category_total(coarse_counters, "comm_recv"))},
                {"restore_bytes_served", static_cast<double>(hbm_category_total(coarse_counters, "restore"))},
                {"pool_read_bytes_served", static_cast<double>(category_direction("pool", 0))},
                {"pool_write_bytes_served", static_cast<double>(category_direction("pool", 1))},
                {"dram_bw_util", hbm_total.at("utilization")},
                {"dram_bw_util_numerator_bytes", static_cast<double>(parse_runtime_exact_counter(hbm_total.at("numerator_bytes_exact_decimal"), "hbm_total_bytes"))},
                {"dram_bw_util_numerator_bytes_exact_decimal", hbm_total.at("numerator_bytes_exact_decimal")},
                {"dram_bw_util_denominator", "sum(raw_hbm_row_peak_Bps)*window_seconds"},
                {"denominator_source", "coarse_die::Hardware.derived.hardware_summary.hbm_row_peak_Bps"},
                {"local_hbm_bw_bytes_per_second", raw_bandwidth},
                {"local_hbm_bw_source", "coarse_die raw hardware descriptor"},
                {"served_by_category", coarse_counters.at("served_by_category")},
                {"bytes_served_definition", "integrated physical payload; includes active, cancelled and retired executions"},
                {"accounting_history_complete", coarse_counters.at("accounting_history_complete")},
                {"arbiter_active_activities", coarse_counters.at("active_activities")},
                {"allocation_updates", coarse_counters.at("allocation_updates")},
                {"peak_concurrent_jobs", hbm_model ? hbm_model->peak_concurrent_jobs() : 0},
                {"peak_concurrent_jobs_definition", "legacy scalar-queue peak; coarse logical-owner peak unavailable"},
                {"redistribution_events", hbm_model ? hbm_model->redistribution_events() : 0},
                {"redistribution_events_definition", "legacy scalar-queue membership changes; coarse allocation_updates reported separately"},
                {"legacy_queue_counter_scope", "legacy scalar queue only; coarse logical-owner peaks are not instrumented"}
            };
            hbm_record["hbm_busy_util"] = sim_end_tick > 0
                ? json(coarse_counters.at("hbm_busy_ns").get<double>() / sim_end_tick)
                : json(nullptr);
            hbm_record["hbm_shared_fraction_of_window"] = sim_end_tick > 0
                ? json(coarse_counters.at("hbm_shared_ns").get<double>() / sim_end_tick)
                : json(nullptr);
            emit_record(hbm_record.dump());
        } else if (hbm_model != nullptr) {
            const double hbm_busy_ns = hbm_model->hbm_busy_ns();
            const double compute_bytes_served =
                hbm_model->compute_bytes_served();
            const double comm_read_bytes_served =
                hbm_model->comm_read_bytes_served();
            const double comm_write_bytes_served =
                hbm_model->comm_write_bytes_served();
            const uint64_t peak_concurrent_jobs =
                hbm_model->peak_concurrent_jobs();
            const uint64_t redistribution_events =
                hbm_model->redistribution_events();
            const long double served_bytes =
                static_cast<long double>(compute_bytes_served) +
                static_cast<long double>(comm_read_bytes_served) +
                static_cast<long double>(comm_write_bytes_served);

            json hbm_record;
            hbm_record["schema"] = 1;
            hbm_record["type"] = "local_hbm";
            hbm_record["source"] = "shared_hbm_runtime_model";
            hbm_record["repo_variant"] = this->repo_variant_;
            hbm_record["run_id"] = this->run_id_;
            hbm_record["rank"] = sys->id;
            hbm_record["model"] =
                "LocalHbmBandwidthModel: N concurrent per-rank HBM users "
                "(COMP roofline traffic, p2p send COMM_READ, p2p recv "
                "COMM_WRITE, collective endpoints COMM_READ for all four "
                "comm types: ALL_REDUCE/ALL_TO_ALL/ALL_GATHER/"
                "REDUCE_SCATTER -- endpoint symmetry since 2026-09-29) "
                "strictly equally share local-mem-bw; any completion "
                "immediately redistributes the rate "
                "(hbm-bandwidth-contention)";
            hbm_record["utilization_window_ns"] = sim_end_tick;
            hbm_record["utilization_window"] = "[0, sim_end_ns]";
            hbm_record["hbm_busy_ns"] = hbm_busy_ns;
            hbm_record["hbm_busy_definition"] =
                "sum of advance_to() steps during which at least one job was "
                "streaming HBM bytes (memory-latency-only steps excluded)";
            hbm_record["compute_bytes_served"] = compute_bytes_served;
            hbm_record["comm_read_bytes_served"] = comm_read_bytes_served;
            hbm_record["comm_write_bytes_served"] = comm_write_bytes_served;
            hbm_record["bytes_served_definition"] =
                "sum of per-step per_user_rate*step_ns accumulated inside "
                "advance_to(); does not alter transition scheduling";
            hbm_record["peak_concurrent_jobs"] = peak_concurrent_jobs;
            hbm_record["peak_concurrent_jobs_definition"] =
                "largest number of simultaneously active HBM jobs observed "
                "at a membership change (issue or completion)";
            hbm_record["redistribution_events"] = redistribution_events;
            hbm_record["redistribution_events_definition"] =
                "count of equal-share recomputations: every membership "
                "change (job joining a non-empty set, completion leaving "
                "survivors) with at least one remaining bandwidth user";
            hbm_record["local_hbm_bw_bytes_per_second"] = sys->local_mem_bw;
            hbm_record["local_hbm_bw_source"] =
                "system-configuration:local-mem-bw";
            if (sim_end_tick > 0) {
                const long double busy_util =
                    static_cast<long double>(hbm_busy_ns) /
                    static_cast<long double>(sim_end_tick);
                hbm_record["hbm_busy_util"] =
                    static_cast<double>(busy_util);
                check_consistency(
                    busy_util <= 1.0L + kUtilEpsilon,
                    "rank " + std::to_string(sys->id) +
                        " hbm_busy_util out of range: numerator=" +
                        std::to_string(hbm_busy_ns) +
                        " denominator=" + std::to_string(sim_end_tick));
            } else {
                hbm_record["hbm_busy_util"] = nullptr;
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
                        " local_hbm dram_bw_util out of range: numerator=" +
                        std::to_string(static_cast<double>(served_bytes)) +
                        " denominator=local_hbm_bw*window_seconds");
            } else {
                hbm_record["dram_bw_util"] = nullptr;
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
            // Note priority: an invariant-violation note wins, then the
            // missing code-8 event (the primary cause of a null
            // first_token_ns), then the missing manifest decode length.
            if (!first_token_note.empty()) {
                record["first_token_note"] = first_token_note;
            } else if (!first_token.has_value()) {
                record["first_token_note"] =
                    "no code-8 first-token event observed for this request";
            } else if (completed && !state.decode_length.has_value()) {
                record["first_token_note"] =
                    "manifest_decode_length_missing: decode-length==1 skew "
                    "field not evaluable";
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

    // Raw anchor records (full detail only). The online anonymous spool
    // is replayed exactly here, preserving the legacy memory_anchor
    // block's fields, multiplicity, and event-arrival order without
    // retaining it in RAM.
    if (full_detail) {
        if (this->online_mode_) {
            emit_online_memory_anchor_spool_records();
        } else {
            // Static-path baseline (testOnly): see apply_event; the
            // in-memory vector arm exists only for the metric regression
            // fixtures (static/online byte-equivalence red line).
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
    // capacity_timeavg record per rank in every non-off mode -- except the
    // empty-ledger early path inside emit_memory_records, which has no
    // planner rows at all and emits only the flat watermark summaries; the
    // raw planner peaks passthrough is full-detail only.
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
    // C2 (2026-09-28): cross-reference the init record's count. Online mode
    // registers CSV rows between initialize() and finalize(), so a larger
    // final count is the documented dynamic path -- echo the init value so
    // the two same-named fields are explicitly reconcilable. Outside online
    // mode there is no dynamic registration path at all: a divergence means
    // an unexpected writer into requests_ and is a consistency violation,
    // not a silent same-name-different-value pair.
    if (this->input_requests_at_initialize_.has_value() &&
        this->input_requests_at_initialize_.value() != input_requests) {
        summary["input_requests_at_init"] =
            this->input_requests_at_initialize_.value();
        if (!this->online_mode_) {
            check_consistency(
                false,
                "summary input_requests (" + std::to_string(input_requests) +
                    ") differs from the init record's input_requests (" +
                    std::to_string(
                        this->input_requests_at_initialize_.value()) +
                    ") outside online mode (no dynamic registration is "
                    "expected on the static path)");
        }
    }
    summary["completed_unique_requests"] = completed_unique_requests;
    // The violation (if any) is already recorded by check_consistency
    // above; clamp here so the unsigned subtraction cannot underflow into
    // a huge bogus count in the emitted record.
    summary["incomplete_requests"] =
        input_requests >= completed_unique_requests
            ? input_requests - completed_unique_requests
            : 0;
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

    // C2 (2026-09-28): rank-universe cross-check. The metric record
    // families key on different rank sources -- rank_compute/hbm_watermark_
    // summary on the simulated systems, capacity_timeavg on the ledger
    // action rows (plus the simulated fallback), planner_memory_peaks on
    // the peaks rows -- and nothing checked the two universes against each
    // other: a ledger row for a rank that is not a simulated system id
    // emitted a ghost capacity_timeavg row with no rank_compute /
    // hbm_watermark_summary counterpart, and a ghost peaks row silently
    // dropped out of the per-rank joins. fallback_ranks IS the simulated
    // rank set (finalize passes every Sys id). An empty fallback set means
    // a manifest-only fixture replay (no simulated universe exists to be
    // ghosted against), so the check applies only to real runs.
    if (!fallback_ranks.empty()) {
        const std::set<int> simulated_ranks(fallback_ranks.begin(),
                                            fallback_ranks.end());
        for (const auto& action : this->memory_actions_) {
            check_consistency(
                simulated_ranks.count(action.rank) > 0,
                "memory action sequence " +
                    std::to_string(action.sequence_index) + " targets rank " +
                    std::to_string(action.rank) +
                    " which is not a simulated rank (ghost "
                    "capacity_timeavg row; metric record families would "
                    "disagree on the rank universe)");
        }
        for (const auto& entry : this->planner_memory_peaks_) {
            const int rank = entry.value("rank", -1);
            check_consistency(
                rank >= 0 && simulated_ranks.count(rank) > 0,
                "planner_memory_peaks entry targets rank " +
                    std::to_string(rank) +
                    " which is not a simulated rank (ghost passthrough row "
                    "with no capacity_timeavg/hbm_watermark_summary "
                    "counterpart)");
        }
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
        // C2 (2026-09-28): the (tick, sequence) order below must already be
        // the array order of the resolved rows. The ledger journal is
        // sequence-ordered (load_manifest enforces strictly increasing
        // sequence_index) and the planner applies its mutations in time
        // order, so a re-sort here means anchored ticks that invert the
        // ledger sequence -- exactly the corruption the Python journal
        // replay fails closed on ("planner_time_ns 回退"). The C++ anchors
        // carry cross-rank aggregation noise (min/max boundary ticks), so
        // the run is not killed; the reorder is a reported consistency
        // violation instead of a silent re-sort. Equal ticks are fine: the
        // sort then only orders by sequence, which the array already is.
        for (size_t i = 1; i < deltas.size(); ++i) {
            if (deltas[i - 1].tick > deltas[i].tick) {
                check_consistency(
                    false,
                    "rank " + std::to_string(rank) +
                        ": ledger replay required a re-sort -- anchored tick "
                        "of sequence " +
                        std::to_string(deltas[i].sequence_index) + " (" +
                        std::to_string(deltas[i].tick) +
                        " ns) precedes sequence " +
                        std::to_string(deltas[i - 1].sequence_index) + " (" +
                        std::to_string(deltas[i - 1].tick) +
                        " ns); replay proceeds in (tick, sequence) order but "
                        "the ledger/anchor chain looks corrupt");
            }
        }
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
        // C2 (2026-09-28): a negative intermediate book value (a release
        // with no matching acquire on this rank's replay chain) is ledger
        // corruption -- the authoritative Python journal replay fail-closes
        // on the same stream ("after 各分量非负", exit 2). It must not be
        // silently clamped to 0 and integrated into plausible-looking
        // byte*ns areas while Python rejects the run: that exact stream
        // would produce C++ numbers and a Python exit code 2. Fail closed
        // through the shared channel instead; the occupancy step function
        // is therefore provably non-negative at every sample below.
        for (const auto& delta : deltas) {
            const Tick elapsed = delta.tick - previous;
            resident_area += (weight + resident) * elapsed;
            committed_area += (weight + resident + reserved) * elapsed;
            weight += delta.weight_delta;
            resident += delta.resident_delta;
            reserved += delta.reserved_delta;
            if (weight < 0 || resident < 0 || reserved < 0) {
                fatal_metrics_error(
                    "rank " + std::to_string(rank) +
                    ": ledger replay went negative (weight=" +
                    i128_to_string(weight) + " resident=" +
                    i128_to_string(resident) + " reserved=" +
                    i128_to_string(reserved) + ") after sequence " +
                    std::to_string(delta.sequence_index) +
                    " (the Python journal replay fail-closes on the same "
                    "stream; refusing to clamp and integrate a corrupt "
                    "ledger)");
            }
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
            // P2 (2026-09-28): one instantaneous sample point of the step
            // function -- peaks, changed buckets, area and the violation
            // counter. Per-delta samples are zero-length (no area), so a
            // delta applied inside a bucket updates the peak/violation state
            // exactly like the Python journal replay's per-row check while
            // the segment walk keeps the periodic bottom line.
            const auto sample_point = [&series, period](
                uint64_t bucket, __int128 resident_value,
                __int128 committed_value, Tick length) {
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
            };
            size_t next_delta = 0;
            Tick seg_start = 0;
            __int128 wm_weight = 0;
            __int128 wm_resident = 0;
            __int128 wm_reserved = 0;
            while (seg_start < sim_end_tick) {
                // Zero-length step points: each same-tick delta is applied
                // and sampled INDIVIDUALLY, so a transient that overflows
                // capacity between two deltas of one tick is seen (it used
                // to be folded into the post-group value and certified
                // clean -- the Python journal replay saw it).
                while (next_delta < deltas.size() &&
                       deltas[next_delta].tick <= seg_start) {
                    wm_weight += deltas[next_delta].weight_delta;
                    wm_resident += deltas[next_delta].resident_delta;
                    wm_reserved += deltas[next_delta].reserved_delta;
                    // A delta applying inside this bucket is occupancy
                    // activity even when the resulting value stays zero
                    // (e.g. a full release).
                    const uint64_t delta_bucket =
                        deltas[next_delta].tick / period;
                    series.changed_buckets.insert(delta_bucket);
                    sample_point(delta_bucket, wm_weight + wm_resident,
                                 wm_weight + wm_resident + wm_reserved, 0);
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
                sample_point(bucket, wm_weight + wm_resident,
                             wm_weight + wm_resident + wm_reserved, length);
                seg_start = seg_end;
            }
            // Deltas anchored exactly at the window end: the walk above
            // stops at sim_end_tick, but the direct integral and the
            // final-ledger cross-check both include them. Sample each
            // post-delta instant into the last window bucket (zero-length,
            // no area -- matching the direct integral's zero tail).
            while (next_delta < deltas.size()) {
                wm_weight += deltas[next_delta].weight_delta;
                wm_resident += deltas[next_delta].resident_delta;
                wm_reserved += deltas[next_delta].reserved_delta;
                const uint64_t delta_bucket = sim_end_tick > 0
                    ? static_cast<uint64_t>((sim_end_tick - 1) / period)
                    : 0;
                series.changed_buckets.insert(delta_bucket);
                sample_point(delta_bucket, wm_weight + wm_resident,
                             wm_weight + wm_resident + wm_reserved, 0);
                next_delta++;
            }
            series.direct_resident_area = resident_area;
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
    // when present), then emit the watermark records. The doc sec.7.8
    // promise is one capacity_timeavg record per rank in every non-off
    // mode, so these ranks also get an explicit all-zero capacity_timeavg
    // record and per-rank joins downstream never see gaps.
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
        if (series.capacity_known && series.capacity_bytes > 0 &&
            sim_end_tick > 0) {
            record["capacity_bytes"] = series.capacity_bytes;
            // All-zero occupancy over the whole window.
            record["resident_capacity_timeavg_util"] = 0.0;
            record["committed_capacity_timeavg_util"] = 0.0;
        } else {
            record["capacity_bytes"] =
                series.capacity_known ? json(series.capacity_bytes)
                                      : json(nullptr);
            record["resident_capacity_timeavg_util"] = nullptr;
            record["committed_capacity_timeavg_util"] = nullptr;
            record["timeavg_util_note"] =
                "undefined: zero capacity/window or missing planner peaks "
                "entry for this rank";
        }
        record["resident_byte_ns"] = "0";
        record["committed_byte_ns"] = "0";
        // C2 (2026-09-28): a fallback rank can still carry unresolved ledger
        // rows (every action of the rank failed to resolve, so it never
        // reached deltas_by_rank). Hardcoding 0/0 contradicted the summary
        // record's memory_actions_total/memory_actions_unresolved for the
        // same run; report the rank's real per-rank totals (actions_
        // replayed is genuinely 0 here -- any resolved row would have put
        // the rank into the replayed branch above).
        record["actions_total"] = total_by_rank[rank];
        record["actions_replayed"] = 0;
        record["actions_unresolved"] = unresolved_by_rank[rank];
        record["anchor_quality_counts"] = quality_counts_by_rank[rank];
        record["transfer_anchor_request_level_fallback"] =
            transfer_anchor_fallback_by_rank[rank];
        record["replay_final_physical_used_bytes"] = "0";
        record["replay_final_committed_used_bytes"] = "0";
        record["final_state_matches_planner"] = nullptr;
        record["final_state_check_note"] =
            "skipped: no replayed deltas for this rank";
        emit_record(record.dump());
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
    // later conflicting assignments are silently ignored). Ranks no request
    // covers map to instance -1 -- online synthetic manifests carry only
    // placeholder instance-0 rank sets, so -1 is the honest label there.
    std::map<int, int64_t> instance_by_rank;
    for (const auto& state : this->requests_) {
        for (const int rank : state.prefill_ranks) {
            if (instance_by_rank.find(rank) == instance_by_rank.end()) {
                instance_by_rank[rank] = state.prefill_instance;
            }
        }
        for (const int rank : state.decode_ranks) {
            if (instance_by_rank.find(rank) == instance_by_rank.end()) {
                instance_by_rank[rank] = state.decode_instance;
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
                // C2 (2026-09-28): declare the sampling window explicitly.
                // The bucket right endpoint is derived from the finalize
                // sim_end_tick -- a DIFFERENT source than the WP6
                // link_bucket rows (the FluidScheduler observer's own last
                // tick) -- so downstream joins must reconcile via these
                // fields, not assume the two bucket families share a window.
                record["window_start_ns"] = 0;
                record["window_end_ns"] = sim_end_tick;
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
                // explains why. A degenerate window (sim_end_tick <= period)
                // collapses last_bucket onto 0; emit that bucket once.
                emit_bucket(0);
                if (last_bucket != 0) {
                    emit_bucket(last_bucket);
                }
                continue;
            }
            // Emit in ascending bucket order (the boundary buckets slot in
            // around the changed set, not after it): downstream joins and
            // the documented log contract read the bucket family as a
            // time-ordered series, so a bucket-0 record appended after the
            // ascending run was the one non-monotonic wrinkle. The changed
            // set iterates ascending (std::set), so bucket 0 emits first
            // when present; the pre-loop arm covers it when absent.
            if (changed.count(0) == 0) {
                emit_bucket(0);
            }
            for (const uint64_t bucket : changed) {
                emit_bucket(bucket);
            }
            // last_bucket==0 collapses onto the bucket-0 emission above; a
            // second emit would duplicate the identical record.
            if (changed.count(last_bucket) == 0 && last_bucket != 0) {
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
        // C2 (2026-09-28): same explicit window declaration as the
        // hbm_watermark bucket rows (see there for the link_bucket source
        // divergence note).
        record["window_start_ns"] = 0;
        record["window_end_ns"] = sim_end_tick;
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
}
