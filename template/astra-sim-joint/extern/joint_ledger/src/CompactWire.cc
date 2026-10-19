// CompactWire.cc -- workledger-compact-v1 encoder + read primitives.
// Record-level decoding lives in CompactReader.cc (one sequential walk).
#include "joint_ledger/CompactWire.hh"

#include <cstring>
#include <unordered_map>

#include "joint_ledger/LedgerTypes.hh"

namespace jointledger {

const char* const kCompactSchemaName = "workledger-compact-v1";

namespace {

[[noreturn]] void wire_fail(const std::string& what) {
    throw LedgerError("compact wire: " + what);
}

// ------------------------------------------------------------ writer ------

class Writer {
public:
    explicit Writer(std::string& out) : out_(out) {}

    void u8(std::uint8_t value) { out_.push_back(static_cast<char>(value)); }
    void raw(const char* data, std::size_t length) {
        out_.append(data, length);
    }
    void u32_at(std::size_t at, std::uint32_t value) {
        for (int i = 0; i < 4; ++i) {
            out_[at + static_cast<std::size_t>(i)] =
                static_cast<char>((value >> (8 * i)) & 0xFF);
        }
    }
    void f64_le(double value) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof bits);
        for (int i = 0; i < 8; ++i) {
            out_.push_back(static_cast<char>((bits >> (8 * i)) & 0xFF));
        }
    }
    void var(std::uint64_t value) {
        while (value >= 0x80) {
            out_.push_back(static_cast<char>((value & 0x7F) | 0x80));
            value >>= 7;
        }
        out_.push_back(static_cast<char>(value));
    }
    void str(const std::string& text) {  // length-prefixed literal
        var(text.size());
        out_.append(text);
    }
    std::uint64_t intern(const std::string& text) {
        const auto found = interned_.find(text);
        if (found != interned_.end()) return found->second;
        const std::uint64_t ref = static_cast<std::uint64_t>(pool_.size());
        pool_.push_back(text);
        interned_.emplace(text, ref);
        return ref;
    }
    const std::vector<std::string>& pool() const { return pool_; }
    void flush_pool() {  // exactly once, after every intern call
        var(pool_.size());
        for (const std::string& entry : pool_) str(entry);
    }

private:
    std::string& out_;
    std::vector<std::string> pool_;
    std::unordered_map<std::string, std::uint64_t> interned_;
};

void put_num(Writer& w, const WireNum& value) {
    switch (value.kind) {
        case WireNum::Kind::Null:
            w.u8(0);
            return;
        case WireNum::Kind::Int:
            w.u8(1);
            w.var(value.integer);
            return;
        case WireNum::Kind::Float:
            w.u8(2);
            w.f64_le(value.real);
            return;
    }
    wire_fail("unknown number tag");
}

void put_owner(Writer& w, const SlotOwner& owner) {
    w.var(w.intern(owner.segment_id));
    w.var(owner.group_ordinal);
    w.var(owner.tile_ordinal);
}

void put_slot_row(Writer& w, const SlotRow& row) {
    w.u8(row.occupied ? 1 : 0);
    if (row.occupied) {
        if (!row.owner) wire_fail("occupied slot row lacks its owner");
        put_owner(w, *row.owner);
    }
    w.u8(row.consumed ? 1 : 0);
    w.var(row.target_shards_to_consume.size());
    for (std::uint64_t shard : row.target_shards_to_consume) w.var(shard);
    w.var(row.consumed_target_shards.size());
    for (std::uint64_t shard : row.consumed_target_shards) w.var(shard);
}

void put_slot_patch(Writer& w, const SlotPatch& patch) {
    if (patch.is_target_delta) {
        w.u8(1);
        w.var(w.intern(patch.key.domain));
        w.var(patch.key.rank);
        w.var(patch.key.slot_index);
        w.var(patch.before_slot_generation);
        w.var(patch.predecessor_logical_seq);
        w.var(patch.predecessor_consumed_count);
        w.var(patch.operations.size());
        for (const TargetOp& op : patch.operations) {
            w.u8(static_cast<std::uint8_t>(op.op));
            w.var(op.slot_generation);
            switch (op.op) {
                case TargetOp::Op::Initialize:
                case TargetOp::Op::RequestCloseClear:
                case TargetOp::Op::Erase:
                    break;
                case TargetOp::Op::Claim:
                    if (!op.owner) {
                        wire_fail("claim operation lacks its owner");
                    }
                    put_owner(w, *op.owner);
                    w.var(op.target_shards.size());
                    for (std::uint64_t shard : op.target_shards) w.var(shard);
                    break;
                case TargetOp::Op::Consume:
                    w.var(op.terminal_rank);
                    w.var(op.terminal_store_id);
                    w.var(op.consume_run_index);
                    w.var(op.add_shards.size());
                    for (std::uint64_t shard : op.add_shards) w.var(shard);
                    w.u8(op.cleared ? 1 : 0);
                    break;
            }
        }
        return;
    }
    w.u8(0);
    // Source quad form (the legacy wire restricts full-row quads to source
    // slots; the compact form is source-only by construction).
    w.var(w.intern(patch.key.domain));
    w.var(patch.key.rank);
    w.u8(patch.key.shard >= 0 ? 1 : 0);
    if (patch.key.shard >= 0) {
        w.var(static_cast<std::uint64_t>(patch.key.shard));
    }
    w.var(patch.key.slot_index);
    w.u8(patch.before ? 1 : 0);
    if (patch.before) put_slot_row(w, *patch.before);
    w.u8(patch.after ? 1 : 0);
    if (patch.after) put_slot_row(w, *patch.after);
}

void put_credit_patch(Writer& w, const CreditPatch& patch) {
    w.var(w.intern(patch.key.domain));
    w.var(patch.key.source_rank);
    w.var(patch.key.shard);
    w.u8(patch.before ? 1 : 0);
    if (patch.before) {
        w.var(patch.before->available);
        w.var(patch.before->outstanding_ack_count);
    }
    w.u8(patch.after ? 1 : 0);
    if (patch.after) {
        w.var(patch.after->available);
        w.var(patch.after->outstanding_ack_count);
    }
}

void put_obligation_patch(Writer& w, const ObligationPatch& patch) {
    // Stable identity (segment + "/run/" + run_index) is carried as the
    // (segment, run_index) coordinates; the canonical id is reconstructed.
    const std::size_t cut = patch.obligation_id.rfind("/run/");
    if (cut == std::string::npos || cut == 0
            || cut + 5 == patch.obligation_id.size()) {
        wire_fail("obligation id is not in canonical segment/run form");
    }
    std::uint64_t run_index = 0;
    for (const char c : patch.obligation_id.substr(cut + 5)) {
        if (c < '0' || c > '9') {
            wire_fail("obligation id is not in canonical segment/run form");
        }
        run_index = run_index * 10 + static_cast<std::uint64_t>(c - '0');
    }
    const std::string segment = patch.obligation_id.substr(0, cut);
    const auto row_coords = [&](const ObligationRow& row) {
        if (row.obligation_id != patch.obligation_id
                || row.segment_id != segment) {
            wire_fail("obligation row identity differs from its patch key");
        }
    };
    w.var(w.intern(segment));
    w.var(run_index);
    w.u8(patch.before ? 1 : 0);
    if (patch.before) {
        row_coords(*patch.before);
        w.var(patch.before->first_store_id);
        w.var(patch.before->remaining_endpoint_count);
        w.u8(static_cast<std::uint8_t>(patch.before->state));
    }
    w.u8(patch.after ? 1 : 0);
    if (patch.after) {
        row_coords(*patch.after);
        w.var(patch.after->first_store_id);
        w.var(patch.after->remaining_endpoint_count);
        w.u8(static_cast<std::uint8_t>(patch.after->state));
    }
}

void put_cursor_patch(Writer& w, const CursorPatch& patch) {
    w.var(w.intern(patch.segment_id));
    w.var(patch.cursor_version);
    w.u8(patch.retired ? 1 : 0);
    w.var(patch.runs.size());
    for (const auto& entry : patch.runs) {
        w.var(entry.first);
        w.u8(entry.second ? 1 : 0);
        if (!entry.second) continue;
        const CursorRun& run = *entry.second;
        w.var(run.completed_endpoint_count);
        w.var(run.cancelled_endpoint_count);
        w.var(run.tile_offset);
        w.var(run.shard_index);
        w.var(run.leg_index);
        w.var(run.completed_out_of_order.size());
        for (const auto& range : run.completed_out_of_order) {
            w.var(range.first);
            w.var(range.second);
        }
        w.var(run.active_endpoints.size());
        for (const ActiveEndpoint& endpoint : run.active_endpoints) {
            w.var(endpoint.ordinal);
            w.var(endpoint.transfer_id);
            w.u8(endpoint.flow_bound ? 1 : 0);
            if (endpoint.flow_bound) w.var(endpoint.flow_id);
            w.u8(endpoint.network_done ? 1 : 0);
            w.u8(endpoint.hbm_done ? 1 : 0);
        }
    }
}

void put_resource_row(Writer& w, const ResourceRow& row) {
    w.u8(static_cast<std::uint8_t>(row.kind));
    w.var(w.intern(row.request_id));
    w.var(w.intern(row.stage));
    w.var(row.generation);
    w.var(row.rank);
    w.var(row.node_id);
    w.var(w.intern(row.operation));
    w.var(w.intern(row.leg));
    w.u8(row.state_index());
    put_num(w, row.total.ops);
    put_num(w, row.total.bytes);
    put_num(w, row.total.latency_ns);
    put_num(w, row.remaining.ops);
    put_num(w, row.remaining.bytes);
    put_num(w, row.remaining.latency_ns);
    w.var(row.route.size());
    for (std::uint64_t hop : row.route) w.var(hop);
    w.var(row.as_of_tick);
    w.u8(row.has_issued_tick ? 1 : 0);
    if (row.has_issued_tick) w.var(row.issued_tick);
    w.u8(row.bound ? 1 : 0);
    if (row.bound) {
        w.var(w.intern(row.descriptor_segment_id));
        w.var(row.endpoint_selector.group_ordinal);
        w.var(row.endpoint_selector.tile_ordinal);
        w.var(row.endpoint_selector.shard_ordinal);
        w.var(w.intern(row.endpoint_selector.endpoint));
    }
    switch (row.kind) {
        case ResourceKind::NocFlow:
            w.var(row.flow_id);
            put_num(w, row.current_rate_Bpns);
            put_num(w, row.last_rate_update_time);
            break;
        case ResourceKind::HbmJob:
            w.var(w.intern(row.job_kind));
            w.var(row.start_tick);
            break;
        case ResourceKind::PoolJob:
            w.var(row.port_index);
            w.var(row.issue_seq);
            put_num(w, row.latency_ready_ns);
            break;
    }
}

void put_run(Writer& w, const PipelineRun& run) {
    w.var(run.first_logical_seq);
    w.u8(static_cast<std::uint8_t>(run.transition_kind));
    w.var(w.intern(run.descriptor_segment_id));
    std::uint8_t flags = 0;
    if (run.have_selector) flags |= 0x01;
    if (run.resource_leg != ResourceLeg::None) flags |= 0x02;
    w.u8(flags);
    if (run.resource_leg != ResourceLeg::None) {
        w.u8(static_cast<std::uint8_t>(run.resource_leg));
    }
    if (run.have_selector) {
        w.var(run.endpoint_selector.group_ordinal);
        w.var(run.endpoint_selector.tile_ordinal);
        w.var(run.endpoint_selector.shard_ordinal);
        w.var(w.intern(run.endpoint_selector.endpoint));
        w.var(run.selector_rank);
    }
    w.var(run.tick_first);
    w.var(run.cursor_version_start);
    w.var(run.cursor_version_end);
    w.var(run.slot_patches.size());
    for (const SlotPatch& patch : run.slot_patches) put_slot_patch(w, patch);
    w.var(run.credit_patches.size());
    for (const CreditPatch& patch : run.credit_patches) {
        put_credit_patch(w, patch);
    }
    w.var(run.obligation_patches.size());
    for (const ObligationPatch& patch : run.obligation_patches) {
        put_obligation_patch(w, patch);
    }
}

}  // namespace

bool is_compact_frame(const char* data, std::size_t length) {
    return length >= 4 && std::memcmp(data, kCompactMagic, 4) == 0;
}

std::size_t compact_frame_length(const char* data, std::size_t length) {
    if (!is_compact_frame(data, length)) {
        wire_fail("segment does not start with the compact magic");
    }
    if (length < 10) wire_fail("compact header is truncated");
    std::uint32_t total = 0;
    for (int i = 0; i < 4; ++i) {
        total |= static_cast<std::uint32_t>(
            static_cast<unsigned char>(data[6 + i])) << (8 * i);
    }
    const std::uint64_t whole = static_cast<std::uint64_t>(total) + 10;
    if (whole > static_cast<std::uint64_t>(length)) {
        wire_fail("compact frame is truncated");
    }
    return static_cast<std::size_t>(whole);
}

void encode_compact_frame(const CompactFrameInput& input, std::string& out) {
    if (input.epoch.empty()) wire_fail("epoch must be nonempty");
    // Pass 1: build the record body with a scratch writer whose pool ends up
    // in first-use order (descriptors -> nodes -> runs -> cursors ->
    // resources); pass 2 emits epoch/schema + pool + body into the frame.
    std::string body;
    body.reserve(1024);
    Writer bw(body);
    bw.u8(1);   // TAG_DESCRIPTORS
    bw.var(input.descriptor_commits_json.size());
    for (const std::string& declaration : input.descriptor_commits_json) {
        bw.str(declaration);
    }
    bw.u8(2);   // TAG_NODES
    bw.var(input.nodes.size());
    for (const CompactNodeRecord& node : input.nodes) {
        if (!node.payload.is_object()) {
            wire_fail("node payload must be an object");
        }
        bw.u8(static_cast<std::uint8_t>(node.kind));
        bw.var(node.logical_seq);
        bw.var(node.tick);
        bw.var(bw.intern(node.work_id));
        bw.str(node.payload.dump());
    }
    bw.u8(3);   // TAG_RUNS
    bw.var(input.runs.size());
    for (const PipelineRun& run : input.runs) put_run(bw, run);
    bw.u8(4);   // TAG_CURSORS
    bw.var(input.cursors.size());
    for (const CursorPatch& patch : input.cursors) put_cursor_patch(bw, patch);
    bw.u8(5);   // TAG_RESOURCES
    bw.var(input.resources.size());
    for (const ResourceRow& row : input.resources) put_resource_row(bw, row);

    out.clear();
    out.reserve(body.size() + 256);
    out.append(kCompactMagic, 4);
    out.push_back(static_cast<char>(kCompactWireVersion));
    out.push_back(static_cast<char>(input.full ? 0x01 : 0x00));
    out.append(4, '\0');   // total_len placeholder
    Writer h(out);
    h.var(input.delivery_sequence);
    h.var(input.tick);
    h.var(input.base_watermark);
    h.var(input.seq_start);
    h.var(input.seq_end);
    h.var(input.event_count);
    h.var(input.base_state_version);
    h.var(input.result_state_version);
    h.str(input.epoch);
    h.str(kCompactSchemaName);
    const std::vector<std::string>& pool = bw.pool();
    h.var(pool.size());
    for (const std::string& entry : pool) h.str(entry);
    out.append(body);
    h.u32_at(6, static_cast<std::uint32_t>(out.size() - 10));
}

// ------------------------------------------------------------- reader -----

std::uint8_t WireCursor::u8(const char* what) {
    if (pos_ >= length_) wire_fail(std::string(what) + " truncated");
    return static_cast<std::uint8_t>(data_[pos_++]);
}

std::uint64_t WireCursor::var(const char* what) {
    std::uint64_t value = 0;
    int shift = 0;
    for (int group = 0; group < 9; ++group) {
        if (pos_ >= length_) {
            wire_fail(std::string(what) + " truncated");
        }
        const std::uint8_t byte = static_cast<std::uint8_t>(data_[pos_++]);
        if (group == 8 && (byte & 0xFE) != 0) {
            wire_fail(std::string(what) + " overflows uint64");
        }
        value |= static_cast<std::uint64_t>(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0) return value;
        shift += 7;
    }
    wire_fail(std::string(what) + " overflows uint64");
}

double WireCursor::f64(const char* what) {
    if (length_ - pos_ < 8) wire_fail(std::string(what) + " truncated");
    std::uint64_t bits = 0;
    for (int i = 0; i < 8; ++i) {
        bits |= static_cast<std::uint64_t>(
            static_cast<unsigned char>(data_[pos_ + i])) << (8 * i);
    }
    pos_ += 8;
    double out = 0.0;
    std::memcpy(&out, &bits, sizeof out);
    return out;
}

void WireCursor::bytes(std::size_t count, std::string& out,
                       const char* what) {
    if (length_ < pos_ || length_ - pos_ < count) {
        wire_fail(std::string(what) + " truncated");
    }
    out.append(data_ + pos_, count);
    pos_ += count;
}

std::string WireCursor::raw_bytes(std::size_t count, const char* what) {
    std::string out;
    out.reserve(count);
    bytes(count, out, what);
    return out;
}

void WireCursor::load_pool(std::vector<std::string>& pool) {
    const std::uint64_t count = var("string pool count");
    pool.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        const std::uint64_t length = var("pool entry length");
        std::string entry;
        bytes(static_cast<std::size_t>(length), entry, "pool entry");
        pool.push_back(std::move(entry));
    }
}

const std::string& WireCursor::pool_str(const std::vector<std::string>& pool,
                                        std::uint64_t ref, const char* what) {
    if (ref >= pool.size()) {
        wire_fail(std::string(what) + " references an undeclared string");
    }
    return pool[static_cast<std::size_t>(ref)];
}

}  // namespace jointledger
