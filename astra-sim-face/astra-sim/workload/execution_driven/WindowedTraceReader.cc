/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

WindowedTraceReader -- execution-driven mechanism layer (wscllm phase 7 §10.4).
Implementation. Row semantics are byte-for-byte the phase-1 full loader
(main_online.cc load_request_queue_csv): same CSV schema, same turn-0 Submit /
turn>0 future-alarm split, same queue_index and metrics registration order.
The only difference is WHEN rows leave the file: the window tops up to
high_water un-consumed rows per pump instead of one full pass.
*******************************************************************************/

#include "astra-sim/workload/execution_driven/WindowedTraceReader.hh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include <json/json.hpp>

#include "astra-sim/workload/MetricCollector.hh"

namespace AstraSim {
namespace ExecutionDriven {

namespace {

uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

}  // namespace

WindowedTraceReader::WindowedTraceReader(const std::string& csv_path,
                                         RequestIngress& ingress,
                                         const size_t high_water,
                                         const uint64_t max_arrival_ns)
    : ingress_(ingress),
      high_water_(high_water),
      max_arrival_ns_(max_arrival_ns) {
    if (csv_path.empty()) {
        // No --request-queue-csv: request-neutral IDLE start. The reader is
        // a no-op (already at EOF); the main entry handles the empty path.
        eof_ = true;
        return;
    }
    file_.open(csv_path);
    if (!file_.is_open()) {
        std::cerr << "[Error] (execution_driven/windowed_reader) cannot open "
                  << "--request-queue-csv: " << csv_path << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

size_t WindowedTraceReader::occupancy() const {
    // Un-consumed rows = rows read so far minus the consumed prefix. The
    // consumed prefix is (consumed_idx_ + 1) because queue indices are
    // 0-based and CSV order == queue index order.
    if (consumed_idx_ < 0) {
        return rows_read_;
    }
    const uint64_t consumed = static_cast<uint64_t>(consumed_idx_) + 1;
    return (rows_read_ >= consumed) ? (rows_read_ - consumed) : 0;
}

void WindowedTraceReader::read_one_row() {
    // Sample the stream position BEFORE the read: this is where the next
    // un-read row starts (the checkpoint's file_pos).
    if (file_.is_open()) {
        const std::streampos pos = file_.tellg();
        if (pos != std::streampos(-1)) {
            last_file_pos_ = static_cast<int64_t>(pos);
        }
    }
    std::string line;
    if (!std::getline(file_, line)) {
        eof_ = true;
        return;
    }
    if (line.empty()) {
        return;  // blank line: not a data row, keep the same position
    }
    if (!header_seen_) {
        header_seen_ = true;
        return;
    }
    std::istringstream row(line);
    std::string session_id, turn_index_s, request_id, prefill_s, decode_s,
        arrival_s, interval_s;
    std::getline(row, session_id, ',');
    std::getline(row, turn_index_s, ',');
    std::getline(row, request_id, ',');
    std::getline(row, prefill_s, ',');
    std::getline(row, decode_s, ',');
    std::getline(row, arrival_s, ',');
    std::getline(row, interval_s, ',');
    ++data_rows_;

    RequestEnvelope env;
    env.session_id = session_id;
    env.turn_index = std::stoi(turn_index_s);
    env.request_id = request_id;
    env.prefill_length = std::stoull(prefill_s);
    env.decode_length = std::stoull(decode_s);
    env.inter_request_interval_ns =
        interval_s.empty() ? 0 : std::stoull(interval_s);
    // Frozen queue index (CSV data-row order, 0-based). Registered for EVERY
    // data row -- turn-0 and skipped turn>0 alike -- so future-arrival
    // scheduling can fill the envelope from the ingress map.
    const int64_t queue_index = static_cast<int64_t>(data_rows_) - 1;
    env.queue_index = queue_index;
    ingress_.register_queue_index(request_id, queue_index);
    // Dynamic request registration (every data row, turn-0 and skipped
    // turn>0 alike, so the request state exists for the node/rank anchors
    // registered at commit time). The online CSV row is authoritative for
    // arrival/session fields. No-op when metrics are disabled. This must
    // keep the phase-1 loader's call order (row order) -- the window only
    // batches the reads, never reorders the registrations.
    if (MetricCollector::instance().enabled()) {
        if (arrival_s.empty()) {
            // Turn>0 row: AFTER_REQUEST, parent = previous same-session row
            // (the future_alarm scheduling path's parent).
            MetricCollector::instance().online_register_request(
                queue_index, request_id, session_id, env.turn_index,
                /*absolute_arrival=*/false, 0,
                last_queue_index_by_session_[session_id],
                env.inter_request_interval_ns);
        } else {
            MetricCollector::instance().online_register_request(
                queue_index, request_id, session_id, env.turn_index,
                /*absolute_arrival=*/true, std::stoull(arrival_s),
                /*arrival_parent_queue_index=*/-1, 0);
        }
        last_queue_index_by_session_[session_id] = queue_index;
    }
    if (arrival_s.empty()) {
        // Turn>0 row: the future_alarm path schedules this arrival from the
        // REQUEST_COMPLETE commit. Never submit it directly.
        ++rows_read_;
        return;
    }
    // Turn-0 row (arrival column non-empty): counted separately for the
    // run-end accepted-accounting invariant (accepted + dropped == turn-0
    // rows; backport fix 2026-08-16).
    ++turn0_rows_;
    const uint64_t arrival_ns = std::stoull(arrival_s);
    env.arrival_world_ns = arrival_ns;
    ++rows_read_;
    if (max_arrival_ns_ > 0 && arrival_ns > max_arrival_ns_) {
        // Phase-7 §10.4: simulation-out-of-range rejection. Backport fix
        // (2026-08-16, sh_2.0测试 §5.1): max_arrival_ns_ == 0 = UNBOUNDED
        // (the production default -- no 30s acceptance-window assumption is
        // burned into the code); a nonzero value is the explicit experiment
        // knob. Counted and never submitted; the row still got its
        // queue_index and metrics request registered above ("every data row
        // registered" stays true). The run-end completion audit fail-closes
        // on any nonzero count (the drop is visible, never silent).
        ++rejected_out_of_range_;
        return;
    }
    IngressCommand cmd;
    cmd.kind = IngressCommandKind::Submit;
    cmd.envelope = std::move(env);
    if (!ingress_.enqueue_command(cmd)) {
        std::cerr << "[Error] (execution_driven/windowed_reader) ingress "
                     "command queue full while reading "
                  << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

bool WindowedTraceReader::pump() {
    if (eof_) {
        return false;
    }
    const uint64_t start = now_ns();
    bool read_any = false;
    // high_water_ == 0 = unbounded (the full-pass control arm of the
    // phase-7 §10.4 window benchmark): one pump reads the whole file.
    while (!eof_ && (high_water_ == 0 || occupancy() < high_water_)) {
        read_one_row();
        read_any = true;
    }
    io_read_ns_ += now_ns() - start;
    if (read_any) {
        ++read_pumps_;
    }
    if (occupancy() > peak_occupancy_) {
        peak_occupancy_ = occupancy();
    }
    return !eof_;
}

void WindowedTraceReader::notify_consumed(const int64_t queue_index) {
    if (queue_index > consumed_idx_) {
        consumed_idx_ = queue_index;
    }
}

uint64_t WindowedTraceReader::count_remaining_data_rows() {
    // Backport fix (2026-08-16, sh_2.0测试 §5.1): count-only tail scan. A
    // rejected row is never consumed (its arrival alarm never fires), so a
    // drop-laden run can pin the window occupancy at high_water and stall
    // pump() before EOF. The completion audit's denominator must be the
    // file's TOTAL data rows, so this scans the remaining lines WITHOUT
    // admitting them to the window: no Submit commands, no queue_index or
    // metrics registration -- counting only (the same blank-line/header
    // skips as read_one_row). Idempotent; eof() is true afterwards.
    if (eof_) {
        return 0;
    }
    uint64_t counted = 0;
    uint64_t counted_turn0 = 0;
    std::string line;
    while (std::getline(file_, line)) {
        if (line.empty()) {
            continue;
        }
        if (!header_seen_) {
            header_seen_ = true;
            continue;
        }
        ++counted;
        // Turn-0-ness of a tail row (6th CSV field non-empty) -- same field
        // split as read_one_row; the fields before the arrival column
        // never contain commas, so the prefix split is safe.
        std::istringstream row(line);
        std::string f1, f2, f3, f4, f5, arrival;
        std::getline(row, f1, ',');
        std::getline(row, f2, ',');
        std::getline(row, f3, ',');
        std::getline(row, f4, ',');
        std::getline(row, f5, ',');
        std::getline(row, arrival, ',');
        if (!arrival.empty()) {
            ++counted_turn0;
        }
    }
    eof_ = true;
    tail_rows_ = counted;
    tail_turn0_rows_ = counted_turn0;
    return counted;
}

bool WindowedTraceReader::write_checkpoint(const std::string& path) const {
    nlohmann::json cp;
    cp["schema"] = 1;
    cp["kind"] = "windowed_reader_checkpoint";
    cp["high_water"] = high_water_;
    cp["max_arrival_ns"] = max_arrival_ns_;
    cp["header_seen"] = header_seen_;
    cp["eof"] = eof_;
    cp["data_rows"] = data_rows_;
    cp["consumed_idx"] = consumed_idx_;
    cp["rows_read"] = rows_read_;
    cp["rejected_out_of_range"] = rejected_out_of_range_;
    cp["read_pumps"] = read_pumps_;
    cp["peak_occupancy"] = peak_occupancy_;
    // Raw byte position of the next un-read line (sampled by read_one_row;
    // the stream tellg is not const-usable from a const method). Meaningful
    // for a restore over the SAME file (same-process restart / same-machine
    // audit); a cross-process restore is out of scope (the stream handle is
    // not persistent), and read_checkpoint fail-closes on an unusable
    // position.
    if (file_.is_open() && last_file_pos_ >= 0) {
        cp["file_pos"] = last_file_pos_;
    }
    cp["written_at_wall_ns"] = now_ns();
    const std::string tmp = path + ".tmp";
    std::ofstream out(tmp, std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "[Error] (execution_driven/windowed_reader) cannot "
                     "write checkpoint "
                  << tmp << std::endl;
        return false;
    }
    out << cp.dump() << "\n";
    out.flush();
    out.close();
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::cerr << "[Error] (execution_driven/windowed_reader) atomic "
                     "rename of checkpoint failed: "
                  << path << std::endl;
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

bool WindowedTraceReader::read_checkpoint(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return false;  // missing checkpoint: nothing to restore (fail-closed
                       // = "no state", not "silent corruption")
    }
    nlohmann::json cp;
    try {
        in >> cp;
    } catch (const nlohmann::json::exception& e) {
        std::cerr << "[Error] (execution_driven/windowed_reader) corrupt "
                     "checkpoint "
                  << path << ": " << e.what() << std::endl;
        return false;  // corrupt checkpoint: fail-closed, state untouched
    }
    if (!cp.is_object() || cp.value("schema", 0) != 1 ||
        cp.value("kind", "") != "windowed_reader_checkpoint") {
        std::cerr << "[Error] (execution_driven/windowed_reader) checkpoint "
                     "schema/kind mismatch: "
                  << path << std::endl;
        return false;
    }
    // Configuration disagreement is fail-closed: a checkpoint written for a
    // different window/max-arrival configuration must not be applied.
    if (cp.value("high_water", static_cast<uint64_t>(~0ULL)) != high_water_ ||
        cp.value("max_arrival_ns", static_cast<uint64_t>(~0ULL)) !=
            max_arrival_ns_) {
        std::cerr << "[Error] (execution_driven/windowed_reader) checkpoint "
                     "configuration mismatch (high_water/max_arrival_ns): "
                  << path << std::endl;
        return false;
    }
    // Rewind the stream to the checkpointed position BEFORE adopting the
    // bookkeeping: the raw position and header_seen_/data_rows_ must agree
    // (the checkpoint was taken over this same file). A missing/unusable
    // position is fail-closed.
    if (!cp.contains("file_pos") || !file_.is_open()) {
        std::cerr << "[Error] (execution_driven/windowed_reader) checkpoint "
                     "has no usable file position: "
                  << path << std::endl;
        return false;
    }
    const int64_t file_pos = cp["file_pos"].get<int64_t>();
    if (file_pos < 0) {
        std::cerr << "[Error] (execution_driven/windowed_reader) checkpoint "
                     "file position invalid: "
                  << path << std::endl;
        return false;
    }
    file_.clear();
    file_.seekg(static_cast<std::streamoff>(file_pos));
    if (!file_) {
        std::cerr << "[Error] (execution_driven/windowed_reader) seekg to "
                     "checkpointed position failed: "
                  << path << std::endl;
        return false;
    }
    header_seen_ = cp.value("header_seen", header_seen_);
    eof_ = cp.value("eof", eof_);
    data_rows_ = cp.value("data_rows", data_rows_);
    consumed_idx_ = cp.value("consumed_idx", consumed_idx_);
    rows_read_ = cp.value("rows_read", rows_read_);
    rejected_out_of_range_ =
        cp.value("rejected_out_of_range", rejected_out_of_range_);
    read_pumps_ = cp.value("read_pumps", read_pumps_);
    peak_occupancy_ = cp.value("peak_occupancy", peak_occupancy_);
    return true;
}

void WindowedTraceReader::report(std::ostream& os) const {
    os << "[online] windowed reader: high_water=" << high_water_
       << " max_arrival_ns=" << max_arrival_ns_
       << (max_arrival_ns_ == 0 ? " (unbounded)" : "") << " rows=" << rows_read_
       << " total_rows=" << total_data_rows()
       << " turn0_rows=" << turn0_data_rows() << " rejected_out_of_range="
       << rejected_out_of_range_ << " read_pumps=" << read_pumps_
       << " peak_window_occupancy=" << peak_occupancy_
       << " io_read_ns=" << io_read_ns_;
    if (io_read_ns_ > 0) {
        os << " read_throughput_rows_per_s="
           << (static_cast<double>(rows_read_) * 1e9 /
               static_cast<double>(io_read_ns_));
    }
    os << std::endl;
}

CompletionAuditVerdict audit_completion(const CompletionAuditCounts& counts) {
    // Backport fix (2026-08-16, sh_2.0测试 §5.1). The OLD audit
    // (completed == rows-the-window-read) let a stalled/dropping run PASS
    // silently: with the 30e9 default and a longer input, over-window
    // requests were rejected, the window stalled on the un-consumable
    // rejects, EOF (and with it the expected-rows bookkeeping) was never
    // reached, and the audit was skipped entirely -- exit 0 "PASS" with
    // 12.5% of the input silently missing (sh_2.0 strategy-20 measured:
    // 2091 input rows, 1830 completed). Fail-closed order: drops first (the
    // loudest, most actionable), then the turn-0 accepted accounting, then
    // the completed-vs-total shortfall.
    if (counts.dropped > 0) {
        return CompletionAuditVerdict::Dropped;
    }
    if (counts.accepted + counts.dropped != counts.turn0_rows) {
        return CompletionAuditVerdict::AccountMismatch;
    }
    if (counts.completed + counts.dropped != counts.total_rows) {
        return CompletionAuditVerdict::Incomplete;
    }
    return CompletionAuditVerdict::Ok;
}

const char* completion_audit_why(const CompletionAuditVerdict verdict) {
    switch (verdict) {
        case CompletionAuditVerdict::Ok:
            return "ok";
        case CompletionAuditVerdict::Dropped:
            return "input rows rejected by the explicit arrival window "
                   "(dropped_out_of_range > 0; visible drop, fail-closed)";
        case CompletionAuditVerdict::AccountMismatch:
            return "accepted + dropped != turn-0 data rows (a turn-0 row "
                   "was neither accepted by the service nor rejected)";
        case CompletionAuditVerdict::Incomplete:
            return "completed + dropped != total data rows (requests "
                   "missing from the run)";
    }
    return "unknown";
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
