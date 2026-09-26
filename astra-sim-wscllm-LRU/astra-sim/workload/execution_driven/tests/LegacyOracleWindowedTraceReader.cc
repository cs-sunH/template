/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

LegacyOracleWindowedTraceReader -- execution-driven mechanism layer (wscllm phase 7 §10.4).
Implementation. Row semantics mirror the phase-1 full loader this oracle
stands in for: same CSV schema, same turn-0 Submit / turn>0 future-alarm
split, same queue_index and metrics registration order (live equivalence
anchors: WindowedTraceReader.hh item 4, tests/windowed_trace_reader_test.cc
Part B). The only difference is WHEN rows leave the file: the window tops up
to high_water un-consumed rows per pump instead of one full pass.
*******************************************************************************/

#include "astra-sim/workload/execution_driven/tests/LegacyOracleWindowedTraceReader.hh"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

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

LegacyOracleWindowedTraceReader::LegacyOracleWindowedTraceReader(const std::string& csv_path,
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
    // A non-empty CSV is the production indexed-input path. Enable strict
    // one-shot resolution before reading any row so a future alarm cannot
    // silently fall back to queue_index=-1 merely because no turn>0 row has
    // been registered yet. Standalone no-CSV fixtures keep legacy mode.
    ingress_.enable_queue_index_tracking();
    file_.open(csv_path);
    if (!file_.is_open()) {
        std::cerr << "[Error] (execution_driven/windowed_reader) cannot open "
                  << "--request-queue-csv: " << csv_path << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

size_t LegacyOracleWindowedTraceReader::occupancy() const {
    // A later CSV row may arrive before an earlier turn>0 row. Counting from
    // max(consumed queue index) would treat that hole as a consumed prefix,
    // let pump read past high_water, and allow the future-index map to grow
    // with the trace. Track exact outstanding rows instead.
    return outstanding_rows_.size();
}

void LegacyOracleWindowedTraceReader::read_one_row() {
    std::string line;
    if (!std::getline(file_, line)) {
        eof_ = true;
        // The collector copies every AFTER_REQUEST parent queue index during
        // registration.  Once this finite CSV reaches EOF no later row can
        // consult the per-session tails, so retaining them only extends their
        // lifetime to run end.
        last_queue_index_by_session_.clear();
        return;
    }
    // Checkpoints resume at the NEXT unread line. The old pre-read sample
    // pointed at the row just consumed and duplicated it after restore.
    std::streampos next_pos = file_.tellg();
    if (next_pos == std::streampos(-1) && file_.eof()) {
        // A final line without a trailing newline may set eofbit even though
        // getline succeeded. Clear it and recover the byte position at EOF;
        // the next getline still fails normally from that position.
        file_.clear();
        file_.seekg(0, std::ios::end);
        next_pos = file_.tellg();
    }
    if (next_pos != std::streampos(-1)) {
        last_file_pos_ = static_cast<int64_t>(next_pos);
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
    // Frozen queue index (CSV data-row order, 0-based). Turn-0 carries it
    // directly in the Submit envelope. A turn>0 row registers a one-shot
    // lookup that is erased when its future arrival is scheduled, so the map
    // is bounded by the reader's unconsumed window rather than total rows.
    const int64_t queue_index = static_cast<int64_t>(data_rows_) - 1;
    env.queue_index = queue_index;
    if (!outstanding_rows_.insert(queue_index).second) {
        std::cerr << "[Error] (execution_driven/windowed_reader) duplicate "
                     "queue index while reading: "
                  << queue_index << std::endl;
        std::abort();
    }
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
            const auto parent = last_queue_index_by_session_.find(session_id);
            if (parent == last_queue_index_by_session_.end()) {
                std::cerr
                    << "[Error] (execution_driven/windowed_reader) turn>0 "
                       "metrics row has no preceding same-session row: session_id="
                    << session_id << " request_id=" << request_id << std::endl;
                std::abort();
            }
            MetricCollector::instance().online_register_request(
                queue_index, request_id, session_id, env.turn_index,
                /*absolute_arrival=*/false, 0,
                parent->second,
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
        ingress_.register_queue_index(request_id, queue_index);
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
        // Phase-7 §10.4: simulation-out-of-range rejection (EXPLICIT window
        // only; default 0 = unbounded, backport fix 2026-08-16 对比报告
        // §5.1; 推进机制统一 2026-08-20 中-3). Counted and never submitted;
        // the row still got its frozen queue_index value and metrics request
        // registration above, but (as a turn-0 row) no future lookup entry.
        // The rejected row
        // is marked consumed right here: it will never fire an arrival
        // alarm, so leaving it un-consumed would pin the window occupancy
        // at high_water and stall pump() before EOF. Rows are read strictly
        // in order, so this row currently holds the highest queue index and
        // advancing the consumed prefix to it is exact. The run-end
        // completion audit fail-closes on any nonzero count (the drop is
        // visible, never silent).
        ++rejected_out_of_range_;
        if (queue_index > consumed_idx_) {
            consumed_idx_ = queue_index;
        }
        outstanding_rows_.erase(queue_index);
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

bool LegacyOracleWindowedTraceReader::pump() {
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

void LegacyOracleWindowedTraceReader::notify_consumed(const int64_t queue_index) {
    // Duplicate notifications for an already-consumed row are harmless; an
    // unknown future index is fail-closed because it would corrupt the window
    // bound. Older fixture notifications may repeat a retired row.
    if (queue_index >= static_cast<int64_t>(rows_read_)) {
        std::cerr << "[Error] (execution_driven/windowed_reader) consumed "
                     "queue index was never read: "
                  << queue_index << " rows_read=" << rows_read_ << std::endl;
        std::abort();
    }
    outstanding_rows_.erase(queue_index);
    if (queue_index > consumed_idx_) {
        consumed_idx_ = queue_index;
    }
}

void LegacyOracleWindowedTraceReader::report(std::ostream& os) const {
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

}  // namespace ExecutionDriven
}  // namespace AstraSim
