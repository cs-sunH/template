// EnvelopeReader.cc -- v4 work_ledger segment SAX reader (joint v4
// acceleration round).
//
// One nlohmann::json::sax_parse pass over the raw segment bytes
// (pipeline_ledger.py _parse_envelope/_validate_envelope_header/
// _parse_descriptor_commits and the node_transitions section of
// work_ledger.py _typed_v3_node_event_from_validated_scalars):
//
//   * top-level object keys arrive in any order; scalar header fields are
//     taken directly (duplicate keys last-wins, Python dict semantics);
//   * the five row arrays (descriptor_commits/node_transitions/
//     pipeline_transition_runs/cursor_patches/active_resources) stream
//     element-by-element through a per-element mini-DOM builder that is
//     discarded as soon as the element's row validator has run -- no
//     whole-envelope DOM is retained;
//   * rows whose validators depend on the descriptor registry are buffered
//     until descriptor_commits closes (or the document ends), so key order
//     never changes accept/reject outcomes;
//   * the header itself (key set, identity constants, replay token,
//     sequence/count interval, full/version constraints, optional_sha256
//     canonical-bytes digest) validates once the document is complete, and
//     node-row tick checks that depend on a later-arriving header tick are
//     deferred to that point.
//
// Row validators live in RowParsers.cc (namespace rowport); their
// declarations are mirrored below.
#include "joint_ledger/PipelineRows.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace jointledger {

// ---------------------------------------------------------------- rowport --
// Internal cross-TU row-parse API, defined in RowParsers.cc.  The view
// structs are defined with identical tokens there (ODR-safe).

namespace rowport {

struct DescriptorView {
    const std::unordered_map<std::string, DescriptorPtr>* declared = nullptr;
    const std::unordered_map<std::string, DescriptorPtr>* retained = nullptr;
    const DescriptorPtr* find(const std::string& segment) const {
        if (declared != nullptr) {
            auto it = declared->find(segment);
            if (it != declared->end()) return &it->second;
        }
        if (retained != nullptr) {
            auto it = retained->find(segment);
            if (it != retained->end()) return &it->second;
        }
        return nullptr;
    }
};

struct SelectorView {
    const std::unordered_map<std::string, SelectorIndex>* declared = nullptr;
    const std::unordered_map<std::string, SelectorIndex>* retained = nullptr;
    const SelectorIndex* find(const std::string& segment) const {
        if (declared != nullptr) {
            auto it = declared->find(segment);
            if (it != declared->end()) return &it->second;
        }
        if (retained != nullptr) {
            auto it = retained->find(segment);
            if (it != retained->end()) return &it->second;
        }
        return nullptr;
    }
};

bool python_equal(const Json& left, const Json& right);
std::string canonical_json_dumps(const Json& value);
std::string sha256_hex(const std::string& bytes);

struct RunSeqCursor {
    bool has_previous = false;
    std::uint64_t previous_seq = 0;
};

struct CursorSegmentCursor {
    bool has_previous = false;
    std::string previous_segment;
};

using ResourceSortKey = std::tuple<std::string, std::uint64_t,
                                   std::uint64_t, std::string, std::string>;

PipelineRun parse_one_pipeline_run(const Json& raw, const std::string& p,
                                  const std::string& section_path,
                                  const DescriptorView& descriptors,
                                  const SelectorView& selector_indexes,
                                  RunSeqCursor* cursor);
CursorPatch parse_cursor_patch_one(const Json& raw, const std::string& p,
                                   const DescriptorView& descriptors,
                                   CursorSegmentCursor* cursor);
ResourceRow parse_resource_row(const Json& raw, const std::string& p,
                               const DescriptorView& descriptors,
                               const SelectorView& selector_indexes,
                               std::optional<ResourceSortKey>* previous);

}  // namespace rowport

namespace {

// ------------------------------------------------------------ constants ----

const char* const kTopKeys[] = {
    "producer", "schema_version", "protocol_version", "bridge_protocol",
    "work_ledger_schema", "epoch", "delivery_sequence", "base_watermark",
    "logical_seq_start", "logical_seq_end", "logical_event_count",
    "replay_token", "descriptor_version", "base_state_version",
    "result_state_version", "snapshot_kind", "tick", "descriptor_commits",
    "node_transitions", "pipeline_transition_runs", "cursor_patches",
    "active_resources", "optional_sha256"};

const char* const kNodeEventKinds[] = {"commit", "issue", "progress",
                                       "completion", "ack", "release"};

// work_ledger.py EVENT_PAYLOAD_WHITELIST minus {kind, work_id, tick,
// event_id} (the row carries those already).
const char* const kAllowedPayloadKeys[] = {
    "delta_served", "state", "request_id", "stage", "generation", "rank",
    "node_id", "operation", "leg", "num_ops", "bytes", "runtime_ns",
    "tensor_size", "total_bytes", "dependencies", "parents", "service_unit",
    "route", "flow_id", "port_index", "issue_seq", "total",
    "total_latency_ns", "release_conditions", "remaining_ops",
    "remaining_bytes", "remaining_latency_ns"};

const char* const kRemainingKeys[] = {"remaining_ops", "remaining_bytes",
                                      "remaining_latency_ns"};

const char* const kOwnerKeys[] = {"segment_id", "pipeline_generation",
                                  "stage_epoch", "staging_allocation_id",
                                  "group_ordinal", "tile_ordinal"};

// ------------------------------------------------------- mini-DOM builder --

struct MiniFrame {
    Json node;
    std::string pending_key;
    bool has_pending_key = false;
};

// Per-element DOM assembler.  Object keys overwrite (duplicate keys
// last-wins, exactly like Python json.loads building a dict).
class MiniDomBuilder {
public:
    bool done() const { return done_; }
    const Json& root() const { return root_; }
    void reset() {
        frames_.clear();
        root_ = Json();
        done_ = false;
    }
    void start_object() {
        frames_.push_back(MiniFrame{Json::object(), {}, false});
    }
    void start_array() {
        frames_.push_back(MiniFrame{Json::array(), {}, false});
    }
    void key(std::string key) {
        frames_.back().pending_key = std::move(key);
        frames_.back().has_pending_key = true;
    }
    void value(Json v) {
        if (frames_.empty()) {
            root_ = std::move(v);
            done_ = true;
            return;
        }
        attach(std::move(v));
    }
    void end_object() {
        Json node = std::move(frames_.back().node);
        frames_.pop_back();
        finish(std::move(node));
    }
    void end_array() {
        Json node = std::move(frames_.back().node);
        frames_.pop_back();
        finish(std::move(node));
    }

private:
    void attach(Json v) {
        MiniFrame& frame = frames_.back();
        if (frame.node.is_object()) {
            if (!frame.has_pending_key) {
                throw LedgerError("internal: object member without key");
            }
            frame.node[frame.pending_key] = std::move(v);  // last-wins
            frame.has_pending_key = false;
        } else {
            frame.node.push_back(std::move(v));
        }
    }
    void finish(Json node) {
        if (frames_.empty()) {
            root_ = std::move(node);
            done_ = true;
        } else {
            attach(std::move(node));
        }
    }
    std::vector<MiniFrame> frames_;
    Json root_;
    bool done_ = false;
};

// --------------------------------------------------------------- router ----

enum class State {
    Start,          // before the top-level value
    TopAwaitKey,    // inside the top object, awaiting a key or its end
    TopValue,       // building the value of a non-section top key
    SectionArray,   // inside one of the five row arrays, awaiting an element
    Element,        // building one row element
    TopOther,       // top-level value is not an object (rejected at the end)
    Done
};

enum class Section {
    None, DescriptorCommits, NodeTransitions, PipelineRuns, CursorPatches,
    ActiveResources
};

Section section_of(const std::string& key) {
    if (key == "descriptor_commits") return Section::DescriptorCommits;
    if (key == "node_transitions") return Section::NodeTransitions;
    if (key == "pipeline_transition_runs") return Section::PipelineRuns;
    if (key == "cursor_patches") return Section::CursorPatches;
    if (key == "active_resources") return Section::ActiveResources;
    return Section::None;
}

struct DeferredTickCheck {
    std::string path;
    std::uint64_t tick;
};

std::string python_list_of_strings(const std::vector<std::string>& values) {
    std::string out = "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out += ",";
        out += "'" + values[i] + "'";
    }
    out += "]";
    return out;
}

class EnvelopeSax {
public:
    EnvelopeSax(const RetainedRegistry& retained, bool allow_retained_replay,
                std::uint64_t expected_delivery, std::uint64_t expected_tick,
                const char* data, std::size_t length)
        : retained_(retained),
          allow_retained_replay_(allow_retained_replay),
          expected_delivery_(expected_delivery),
          expected_tick_(expected_tick),
          data_(data),
          length_(length) {}

    bool entered_top_object() const { return entered_top_object_; }
    bool finished() const { return state_ == State::Done; }
    ParsedEnvelope take_result() { return std::move(result_); }

    // ---- nlohmann json_sax interface ----
    bool null() { return route_scalar(Json()); }
    bool boolean(bool value) { return route_scalar(Json(value)); }
    bool number_integer(Json::number_integer_t value) {
        return route_scalar(Json(value));
    }
    bool number_unsigned(Json::number_unsigned_t value) {
        return route_scalar(Json(value));
    }
    bool number_float(Json::number_float_t value, const std::string&) {
        return route_scalar(Json(value));
    }
    bool string(std::string& value) { return route_scalar(Json(value)); }
    bool binary(Json::binary_t&) {
        throw LedgerError("work_ledger segment contains unsupported binary");
    }

    bool start_object(std::size_t) {
        switch (state_) {
            case State::Start:
                entered_top_object_ = true;
                state_ = State::TopAwaitKey;
                break;
            case State::TopValue:
            case State::Element:
                builder_.start_object();
                break;
            case State::SectionArray:
                state_ = State::Element;
                builder_.start_object();
                break;
            case State::TopOther:
                ++other_depth_;
                break;
            default:
                throw LedgerError("internal: unexpected object start");
        }
        return true;
    }

    bool key(std::string& k) {
        if (state_ == State::Element) {
            builder_.key(std::move(k));
            return true;
        }
        if (state_ == State::TopOther) return true;
        if (state_ != State::TopAwaitKey) {
            throw LedgerError("internal: unexpected object key");
        }
        begin_top_key(k);
        return true;
    }

    bool start_array(std::size_t) {
        switch (state_) {
            case State::Start:
                entered_top_object_ = false;
                state_ = State::TopOther;
                other_depth_ = 1;
                break;
            case State::TopValue:
                if (current_section_ != Section::None) {
                    on_section_start();
                    state_ = State::SectionArray;
                } else {
                    builder_.start_array();
                }
                break;
            case State::SectionArray:
                state_ = State::Element;
                builder_.start_array();
                break;
            case State::Element:
                builder_.start_array();
                break;
            case State::TopOther:
                ++other_depth_;
                break;
            default:
                throw LedgerError("internal: unexpected array start");
        }
        return true;
    }

    bool end_array() {
        switch (state_) {
            case State::TopValue:
                builder_.end_array();
                if (builder_.done()) finish_top_value();
                break;
            case State::Element:
                builder_.end_array();
                if (builder_.done()) finish_element();
                break;
            case State::SectionArray:
                on_section_end();
                state_ = State::TopAwaitKey;
                break;
            case State::TopOther:
                if (--other_depth_ == 0) state_ = State::Done;
                break;
            default:
                throw LedgerError("internal: unexpected array end");
        }
        return true;
    }

    bool end_object() {
        switch (state_) {
            case State::TopValue:
                builder_.end_object();
                if (builder_.done()) finish_top_value();
                break;
            case State::Element:
                builder_.end_object();
                if (builder_.done()) finish_element();
                break;
            case State::TopAwaitKey:
                finish_document();
                state_ = State::Done;
                break;
            case State::TopOther:
                if (--other_depth_ == 0) state_ = State::Done;
                break;
            default:
                throw LedgerError("internal: unexpected object end");
        }
        return true;
    }

    bool parse_error(std::size_t position, const std::string&,
                     const Json::exception& error) {
        throw LedgerError(
            std::string("work_ledger segment JSON parse error at byte ")
            + std::to_string(position) + ": " + error.what());
    }

private:
    bool route_scalar(Json value) {
        switch (state_) {
            case State::Start:
                // envelope is a bare scalar: rejected as non-object
                state_ = State::Done;
                break;
            case State::TopValue:
                builder_.value(std::move(value));
                if (builder_.done()) finish_top_value();
                break;
            case State::SectionArray: {
                Json element = std::move(value);
                dispatch_element(std::move(element));
                break;
            }
            case State::Element:
                builder_.value(std::move(value));
                break;
            case State::TopOther:
                break;  // already rejected as non-object
            default:
                throw LedgerError("internal: unexpected scalar value");
        }
        return true;
    }

    void begin_top_key(const std::string& key) {
        current_key_ = key;
        seen_top_keys_.insert(key);
        current_section_ = section_of(key);
        if (current_section_ != Section::None) {
            reset_section(current_section_);  // duplicate key: last array wins
        }
        state_ = State::TopValue;
    }

    void reset_section(Section section) {
        switch (section) {
            case Section::DescriptorCommits:
                commit_list_.clear();
                seen_segments_.clear();
                declared_.clear();
                selector_additions_.clear();
                dependency_additions_.clear();
                descriptor_commits_closed_ = false;
                descriptor_index_ = 0;
                break;
            case Section::NodeTransitions:
                node_events_.clear();
                deferred_tick_checks_.clear();
                node_index_ = 0;
                break;
            case Section::PipelineRuns:
                runs_.clear();
                pending_runs_.clear();
                run_cursor_ = rowport::RunSeqCursor{};
                run_index_ = 0;
                break;
            case Section::CursorPatches:
                cursor_patches_.clear();
                pending_cursor_.clear();
                cursor_cursor_ = rowport::CursorSegmentCursor{};
                cursor_index_ = 0;
                break;
            case Section::ActiveResources:
                resources_.clear();
                pending_resources_.clear();
                resource_previous_.reset();
                resource_index_ = 0;
                break;
            case Section::None:
                break;
        }
    }

    void on_section_start() {}

    void on_section_end() {
        if (current_section_ == Section::DescriptorCommits) {
            finalize_commits();
        }
    }

    void finish_top_value() {
        Json value = builder_.root();
        builder_.reset();
        if (current_section_ != Section::None) {
            throw LedgerError(current_key_ + " must be an array");
        }
        header_fields_[current_key_] = std::move(value);
        state_ = State::TopAwaitKey;
    }

    void finish_element() {
        Json element = builder_.root();
        builder_.reset();
        dispatch_element(std::move(element));
        state_ = State::SectionArray;
    }

    void dispatch_element(Json element) {
        switch (current_section_) {
            case Section::DescriptorCommits:
                handle_descriptor_commit(element,
                                         "descriptor_commits["
                                         + std::to_string(descriptor_index_++)
                                         + "]");
                break;
            case Section::NodeTransitions:
                handle_node_row(element, "node_transitions["
                                 + std::to_string(node_index_++) + "]");
                break;
            case Section::PipelineRuns:
                if (descriptor_commits_closed_) {
                    validate_run_element(
                        element, "pipeline_transition_runs["
                        + std::to_string(run_index_) + "]");
                } else {
                    pending_runs_.emplace_back(
                        "pipeline_transition_runs["
                        + std::to_string(run_index_) + "]", std::move(element));
                }
                ++run_index_;
                break;
            case Section::CursorPatches:
                if (descriptor_commits_closed_) {
                    validate_cursor_element(
                        element, "cursor_patches["
                        + std::to_string(cursor_index_) + "]");
                } else {
                    pending_cursor_.emplace_back(
                        "cursor_patches["
                        + std::to_string(cursor_index_) + "]", std::move(element));
                }
                ++cursor_index_;
                break;
            case Section::ActiveResources:
                if (descriptor_commits_closed_) {
                    validate_resource_element(
                        element, "active_resources["
                        + std::to_string(resource_index_) + "]");
                } else {
                    pending_resources_.emplace_back(
                        "active_resources["
                        + std::to_string(resource_index_) + "]", std::move(element));
                }
                ++resource_index_;
                break;
            case Section::None:
                throw LedgerError("internal: element outside a section");
        }
    }

    rowport::DescriptorView descriptor_view() const {
        return rowport::DescriptorView{&declared_,
                                       retained_.descriptors};
    }

    rowport::SelectorView selector_view() const {
        return rowport::SelectorView{&selector_additions_,
                                     retained_.selector_indexes};
    }

    void validate_run_element(Json element, const std::string& path) {
        runs_.push_back(rowport::parse_one_pipeline_run(
            std::move(element), path, "pipeline_transition_runs",
            descriptor_view(), selector_view(), &run_cursor_));
    }

    void validate_cursor_element(Json element, const std::string& path) {
        cursor_patches_.push_back(rowport::parse_cursor_patch_one(
            std::move(element), path, descriptor_view(), &cursor_cursor_));
    }

    void validate_resource_element(Json element,
                                   const std::string& path) {
        resources_.push_back(rowport::parse_resource_row(
            std::move(element), path, descriptor_view(), selector_view(),
            &resource_previous_));
    }

    // pipeline_ledger.py _parse_descriptor_commits, per-element part.
    void handle_descriptor_commit(Json element, const std::string& p) {
        const Json& commit = read_mapping(element, p);
        const std::vector<const char*> key_list = {
            "identity", "descriptor", "commit_tick", "cursor_version",
            "service_done", "settle_done", "retired"};
        require_keys(commit, p, key_list, key_list);
        DescriptorPtr descriptor =
            PipelineDescriptor::from_wire(commit.at("descriptor"));
        const PipelineIdentity& ident = descriptor->identity;
        const Json& identity = read_mapping(commit.at("identity"),
                                            p + ".identity");
        const std::vector<const char*> identity_keys = {
            "request_id", "stage", "generation", "stage_epoch",
            "segment_ordinal", "rank", "staging_allocation_id", "segment_id",
            "buffer_domain_id"};
        require_keys(identity, p + ".identity", identity_keys, identity_keys);
        if (!rowport::python_equal(identity,
                                   descriptor->to_wire().at("identity"))) {
            throw LedgerError(p + ".identity differs from immutable descriptor");
        }
        if (seen_segments_.count(ident.segment_id) != 0) {
            throw LedgerError("duplicate descriptor declaration '"
                              + ident.segment_id + "'");
        }
        seen_segments_.insert(ident.segment_id);
        read_uint(commit.at("commit_tick"), p + ".commit_tick");
        read_uint(commit.at("cursor_version"), p + ".cursor_version");
        read_bool(commit.at("service_done"), p + ".service_done");
        read_bool(commit.at("settle_done"), p + ".settle_done");
        read_bool(commit.at("retired"), p + ".retired");
        commit_list_.emplace_back(std::move(descriptor), std::move(element));
    }

    // pipeline_ledger.py _parse_descriptor_commits, registry-merge part.
    void finalize_commits() {
        for (const auto& entry : commit_list_) {
            const std::string& segment_id = entry.first->identity.segment_id;
            const DescriptorPtr* prior = nullptr;
            if (retained_.descriptors != nullptr) {
                const auto found = retained_.descriptors->find(segment_id);
                if (found != retained_.descriptors->end()) {
                    prior = &found->second;
                }
            }
            if (prior != nullptr
                    && !descriptor_wire_equal(**prior, *entry.first)) {
                throw LedgerError(
                    "immutable descriptor changed after declaration");
            }
            if (prior != nullptr && !allow_retained_replay_) {
                throw LedgerError(
                    "descriptor segment was declared more than once");
            }
            declared_[segment_id] = entry.first;
        }
        for (const auto& entry : declared_) {
            const bool has_retained_index =
                retained_.selector_indexes != nullptr
                && retained_.selector_indexes->find(entry.first)
                    != retained_.selector_indexes->end();
            if (!has_retained_index) {
                selector_additions_.emplace(
                    entry.first, build_selector_index(*entry.second));
            }
        }
        for (const auto& entry : declared_) {
            const bool has_retained_descriptor =
                retained_.descriptors != nullptr
                && retained_.descriptors->find(entry.first)
                    != retained_.descriptors->end();
            if (!has_retained_descriptor) {
                dependency_additions_.emplace(
                    entry.first, build_dependency_index(*entry.second));
            }
        }
        descriptor_commits_closed_ = true;
        flush_pending();
    }

    void flush_pending() {
        for (auto& pending : pending_runs_) {
            validate_run_element(std::move(pending.second), pending.first);
        }
        pending_runs_.clear();
        for (auto& pending : pending_cursor_) {
            validate_cursor_element(std::move(pending.second), pending.first);
        }
        pending_cursor_.clear();
        for (auto& pending : pending_resources_) {
            validate_resource_element(std::move(pending.second), pending.first);
        }
        pending_resources_.clear();
    }

    // ---- node rows (work_ledger.py _typed_v3_node_event_from_validated_
    // scalars behind pipeline_ledger.py _parse_envelope's section) ----
    void handle_node_row(Json element, const std::string& p) {
        const Json& event = read_mapping(element, p);
        const std::vector<const char*> row_keys = {
            "logical_seq", "tick", "kind", "work_id", "payload"};
        require_keys(event, p, row_keys, row_keys);
        const std::uint64_t seq = read_uint(event.at("logical_seq"),
                                            p + ".logical_seq");
        const std::uint64_t etick = read_uint(event.at("tick"), p + ".tick");
        if (seq == 0) {
            throw LedgerError(p + " logical sequence/tick is invalid");
        }
        const auto tick_field = header_fields_.find("tick");
        const bool tick_known = tick_field != header_fields_.end();
        if (tick_known) {
            const std::uint64_t envelope_tick =
                read_uint(tick_field->second, "tick");
            if (etick > envelope_tick) {
                throw LedgerError(p + " logical sequence/tick is invalid");
            }
        } else {
            deferred_tick_checks_.push_back(DeferredTickCheck{p, etick});
        }
        const std::string kind = read_string(event.at("kind"), p + ".kind");
        bool kind_ok = false;
        for (const char* candidate : kNodeEventKinds) {
            if (kind == candidate) {
                kind_ok = true;
                break;
            }
        }
        if (!kind_ok) {
            throw LedgerError(p + ".kind is unknown");
        }
        const std::string work_id = read_string(event.at("work_id"),
                                                p + ".work_id");
        const Json& payload = event.at("payload");
        if (!payload.is_object()) {
            throw LedgerError(p + ".payload must be a string-keyed object");
        }
        std::vector<std::string> unknown;
        for (auto member = payload.begin(); member != payload.end();
             ++member) {
            bool allowed = false;
            for (const char* key : kAllowedPayloadKeys) {
                if (member.key() == key) {
                    allowed = true;
                    break;
                }
            }
            if (!allowed) unknown.push_back(member.key());
        }
        if (!unknown.empty()) {
            std::sort(unknown.begin(), unknown.end());
            throw LedgerError(p + ".payload has unknown keys "
                              + python_list_of_strings(unknown));
        }
        NodeEvent typed;
        typed.kind = kind;
        typed.work_id = work_id;
        typed.tick = etick;
        typed.producer = "cpp_online";
        typed.local_seq = seq;
        // One retained copy of the payload subtree (the move experiment
        // showed downstream readers still alias the element, so the safe
        // form is a single copy handed over by value).
        Json payload_retained = payload;
        try {
            build_typed_node_event(kind, payload,
                                    std::move(payload_retained),
                                    &typed);
        } catch (const LedgerError& error) {
            throw LedgerError(p + " is not a valid typed node event: "
                              + error.what());
        }
        if (!node_events_.empty() && seq <= node_events_.back().first) {
            throw LedgerError(
                "node_transitions must preserve unique logical sequence "
                "order");
        }
        node_events_.emplace_back(seq, std::move(typed));
    }

    // work_ledger.py _typed_v3_node_event_from_validated_scalars body: the
    // payload is retained as-is (including delta_served; the application
    // core does not read it back), delta_served is parsed into a WorkAmount.
    void build_typed_node_event(const std::string& kind, const Json& payload,
                                Json&& payload_storage, NodeEvent* out) {
        if (kind != "commit" && kind != "progress") {
            for (const char* key : kRemainingKeys) {
                if (payload.find(key) != payload.end()) {
                    throw LedgerError(
                        "remaining_* fields are only valid on commit/progress "
                        "events");
                }
            }
        }
        for (const char* key : kRemainingKeys) {
            const auto field = payload.find(key);
            if (field == payload.end()) continue;
            check_amount_number(*field, std::string("event.") + key);
        }
        WorkAmount delta;
        const auto delta_field = payload.find("delta_served");
        if (delta_field != payload.end() && !delta_field->is_null()) {
            delta = parse_delta_served(*delta_field);
        }
        out->delta_served = delta;
        // dict(payload): delta_served stays inline. The element is a
        // discarded temporary (dispatch passes it by value), so the payload
        // subtree is moved rather than deep-copied.
        out->payload = std::move(payload_storage);
    }

    static void check_amount_number(const Json& value, const std::string&) {
        // work_ledger.py _amount_number: bool rejected, int/float finite.
        if (value.is_boolean()
                || !(value.is_number_integer() || value.is_number_unsigned()
                     || value.is_number_float())) {
            throw LedgerError("work amount must be a finite number");
        }
        if (value.is_number_float() && !std::isfinite(value.get<double>())) {
            throw LedgerError("work amount must be a finite number");
        }
    }

    static WorkAmount parse_delta_served(const Json& value) {
        if (!value.is_object()) {
            throw LedgerError("unknown work-amount keys");
        }
        static const char* const dims[] = {"ops", "bytes", "latency_ns"};
        std::vector<std::string> unknown;
        for (auto member = value.begin(); member != value.end(); ++member) {
            bool known = false;
            for (const char* dim : dims) {
                if (member.key() == dim) {
                    known = true;
                    break;
                }
            }
            if (!known) unknown.push_back(member.key());
        }
        if (!unknown.empty()) {
            std::sort(unknown.begin(), unknown.end());
            throw LedgerError("unknown work-amount keys "
                              + python_list_of_strings(unknown));
        }
        WorkAmount amount;
        std::optional<double> WorkAmount::*const members[3] = {
            &WorkAmount::ops, &WorkAmount::bytes, &WorkAmount::latency_ns};
        for (int i = 0; i < 3; ++i) {
            const auto member = value.find(dims[i]);
            if (member == value.end() || member->is_null()) continue;
            check_amount_number(*member, dims[i]);
            const double number = member->get<double>();
            if (number < -1e-6) {
                throw LedgerError(std::string("work amount '") + dims[i]
                                  + "' must be non-negative");
            }
            amount.*members[i] = number;
        }
        return amount;
    }

    // ---- document completion: header validation + pending flush ----
    void finish_document() {
        if (!entered_top_object_) {
            throw LedgerError("pipeline ledger envelope must be an object");
        }
        cross_check_outer_frame();
        validate_key_set();
        const EnvelopeHeader header = validate_header_fields();
        for (auto& entry : node_events_) {
            entry.second.delivery_sequence = header.delivery_sequence;
        }
        for (const DeferredTickCheck& check : deferred_tick_checks_) {
            if (check.tick > header.tick) {
                throw LedgerError(check.path
                                  + " logical sequence/tick is invalid");
            }
        }
        if (!descriptor_commits_closed_) {
            finalize_commits();  // no commits section: flush buffered rows
        }
        result_.header = header;
        result_.descriptors = std::move(commit_list_);
        result_.selector_index_additions = std::move(selector_additions_);
        result_.dependency_index_additions = std::move(dependency_additions_);
        result_.node_events = std::move(node_events_);
        result_.pipeline_runs = std::move(runs_);
        result_.cursor_patches = std::move(cursor_patches_);
        result_.resources = std::move(resources_);
    }

    // The Python coordinator validates the payload delivery/tick against the
    // outer frame before any ledger parsing (work_ledger.py
    // _validate_payload_envelope).
    void cross_check_outer_frame() {
        const auto delivery = header_fields_.find("delivery_sequence");
        if (delivery == header_fields_.end()) {
            throw LedgerError("work_ledger delivery_sequence -1 != delta "
                              "delivery_sequence "
                              + std::to_string(expected_delivery_));
        }
        const std::uint64_t delivery_value =
            read_uint(delivery->second, "delivery_sequence");
        if (delivery_value != expected_delivery_) {
            throw LedgerError(
                "work_ledger delivery_sequence "
                + std::to_string(delivery_value)
                + " != delta delivery_sequence "
                + std::to_string(expected_delivery_));
        }
        const auto tick = header_fields_.find("tick");
        if (tick == header_fields_.end()) {
            throw LedgerError("work_ledger tick -1 != delta tick "
                              + std::to_string(expected_tick_));
        }
        const std::uint64_t tick_value = read_uint(tick->second, "tick");
        if (tick_value != expected_tick_) {
            throw LedgerError("work_ledger tick "
                              + std::to_string(tick_value) + " != delta tick "
                              + std::to_string(expected_tick_));
        }
    }

    void validate_key_set() const {
        std::vector<std::string> missing;
        std::vector<std::string> unknown;
        for (const char* key : kTopKeys) {
            if (std::string(key) == "optional_sha256") continue;
            if (seen_top_keys_.find(key) == seen_top_keys_.end()) {
                missing.push_back(key);
            }
        }
        for (const std::string& key : seen_top_keys_) {
            bool known = false;
            for (const char* candidate : kTopKeys) {
                if (key == candidate) {
                    known = true;
                    break;
                }
            }
            if (!known) unknown.push_back(key);
        }
        if (!missing.empty() || !unknown.empty()) {
            std::sort(missing.begin(), missing.end());
            std::sort(unknown.begin(), unknown.end());
            throw LedgerError(
                "pipeline ledger envelope keys invalid (missing="
                + python_list_of_strings(missing) + ", unknown="
                + python_list_of_strings(unknown) + ")");
        }
    }

    // pipeline_ledger.py _validate_envelope_header, protocol_version=4.
    EnvelopeHeader validate_header_fields() {
        const auto field = [this](const char* key) -> const Json& {
            const auto found = header_fields_.find(key);
            if (found == header_fields_.end()) {
                throw LedgerError(std::string("internal: missing field ")
                                  + key);
            }
            return found->second;
        };
        const Json& producer = field("producer");
        if (!(producer.is_string()
                && producer.get_ref<const std::string&>() == "cpp_online")) {
            throw LedgerError("unsupported v4 producer/schema/protocol");
        }
        if (read_uint(field("schema_version"), "schema_version") != 4) {
            throw LedgerError("unsupported v4 producer/schema/protocol");
        }
        if (read_uint(field("protocol_version"), "protocol_version") != 4) {
            throw LedgerError("unsupported v4 producer/schema/protocol");
        }
        const Json& bridge = field("bridge_protocol");
        if (!(bridge.is_string()
                && bridge.get_ref<const std::string&>()
                    == "bridge-v4-20261011")) {
            throw LedgerError("unsupported v4 bridge or work-ledger schema");
        }
        const Json& ledger_schema = field("work_ledger_schema");
        if (!(ledger_schema.is_string()
                && ledger_schema.get_ref<const std::string&>()
                    == "workledger-v4-20261011")) {
            throw LedgerError("unsupported v4 bridge or work-ledger schema");
        }
        if (read_uint(field("descriptor_version"), "descriptor_version") != 1) {
            throw LedgerError("unsupported descriptor_version");
        }
        const std::string epoch = read_string(field("epoch"), "epoch");
        const std::uint64_t delivery =
            read_uint(field("delivery_sequence"), "delivery_sequence");
        const std::uint64_t tick = read_uint(field("tick"), "tick");
        const Json& kind = field("snapshot_kind");
        if (!(kind.is_string()
                && (kind.get_ref<const std::string&>() == "full"
                    || kind.get_ref<const std::string&>() == "delta"))) {
            throw LedgerError("snapshot_kind must be full or delta");
        }
        const bool full = kind.get_ref<const std::string&>() == "full";
        const std::uint64_t watermark =
            read_uint(field("base_watermark"), "base_watermark");
        const std::uint64_t seq_start =
            read_uint(field("logical_seq_start"), "logical_seq_start");
        const std::uint64_t seq_end =
            read_uint(field("logical_seq_end"), "logical_seq_end");
        const std::uint64_t count =
            read_uint(field("logical_event_count"), "logical_event_count");
        const std::uint64_t base_version =
            read_uint(field("base_state_version"), "base_state_version");
        const std::uint64_t result_version =
            read_uint(field("result_state_version"), "result_state_version");
        const std::string replay =
            read_string(field("replay_token"), "replay_token");
        if (replay != "cpp_online/" + epoch + "/" + std::to_string(delivery)) {
            throw LedgerError(
                "replay_token does not match producer/epoch/delivery");
        }
        const bool seq_start_ok =
            watermark < U64_MAX && seq_start == watermark + 1;
        if (count == 0) {
            if (!seq_start_ok || seq_end != watermark) {
                throw LedgerError(
                    "empty delta sequence interval is invalid");
            }
        } else {
            const bool interval_ok = seq_start_ok && seq_end >= seq_start
                && seq_end - seq_start + 1 == count;
            if (!interval_ok) {
                throw LedgerError(
                    "logical sequence interval/count is inconsistent");
            }
        }
        const bool version_ok = result_version == base_version
            || (base_version < U64_MAX
                && result_version == base_version + 1);
        if ((full && watermark != 0) || !version_ok) {
            throw LedgerError(
                "full watermark or state-version transition is invalid");
        }
        const auto digest_field = header_fields_.find("optional_sha256");
        if (digest_field != header_fields_.end()) {
            const Json& digest = digest_field->second;
            bool digest_ok = digest.is_string();
            if (digest_ok) {
                const std::string& text =
                    digest.get_ref<const std::string&>();
                digest_ok = text.size() == 64;
                for (char c : text) {
                    if (!((c >= '0' && c <= '9')
                            || (c >= 'a' && c <= 'f'))) {
                        digest_ok = false;
                        break;
                    }
                }
            }
            if (!digest_ok) {
                throw LedgerError(
                    "optional_sha256 must be 64 lowercase hexadecimal "
                    "characters");
            }
            Json dom;
            try {
                dom = Json::parse(data_, data_ + length_);
            } catch (const Json::parse_error& error) {
                throw LedgerError(
                    std::string("envelope cannot be parsed as canonical JSON "
                                "at byte ")
                    + std::to_string(error.byte) + ": " + error.what());
            }
            dom.erase("optional_sha256");
            if (rowport::sha256_hex(rowport::canonical_json_dumps(dom))
                    != digest.get_ref<const std::string&>()) {
                throw LedgerError(
                    "optional_sha256 does not match canonical envelope JSON");
            }
        }
        EnvelopeHeader header;
        header.replay_token = replay;
        header.epoch = epoch;
        header.delivery_sequence = delivery;
        header.full = full;
        header.tick = tick;
        header.base_watermark = watermark;
        header.seq_start = seq_start;
        header.seq_end = seq_end;
        header.event_count = count;
        header.base_version = base_version;
        header.result_version = result_version;
        return header;
    }

    // ---- router state ----
    State state_ = State::Start;
    bool entered_top_object_ = false;
    std::size_t other_depth_ = 0;
    std::string current_key_;
    Section current_section_ = Section::None;
    std::set<std::string> seen_top_keys_;
    std::map<std::string, Json> header_fields_;
    MiniDomBuilder builder_;

    // ---- section accumulators ----
    std::size_t descriptor_index_ = 0;
    std::size_t node_index_ = 0;
    std::size_t run_index_ = 0;
    std::size_t cursor_index_ = 0;
    std::size_t resource_index_ = 0;
    bool descriptor_commits_closed_ = false;
    std::vector<std::pair<DescriptorPtr, Json>> commit_list_;
    std::unordered_set<std::string> seen_segments_;
    std::unordered_map<std::string, DescriptorPtr> declared_;
    std::unordered_map<std::string, SelectorIndex> selector_additions_;
    std::unordered_map<std::string, DependencyIndex> dependency_additions_;
    std::vector<std::pair<std::uint64_t, NodeEvent>> node_events_;
    std::vector<DeferredTickCheck> deferred_tick_checks_;
    std::vector<PipelineRun> runs_;
    std::vector<CursorPatch> cursor_patches_;
    std::vector<ResourceRow> resources_;
    std::vector<std::pair<std::string, Json>> pending_runs_;
    std::vector<std::pair<std::string, Json>> pending_cursor_;
    std::vector<std::pair<std::string, Json>> pending_resources_;
    rowport::RunSeqCursor run_cursor_;
    rowport::CursorSegmentCursor cursor_cursor_;
    std::optional<rowport::ResourceSortKey> resource_previous_;
    ParsedEnvelope result_;

    // ---- parse inputs ----
    const RetainedRegistry& retained_;
    bool allow_retained_replay_;
    std::uint64_t expected_delivery_;
    std::uint64_t expected_tick_;
    const char* data_;
    std::size_t length_;
};

}  // namespace

ParsedEnvelope parse_envelope_v4(const char* data, std::size_t length,
                                 const RetainedRegistry& retained,
                                 bool allow_retained_replay,
                                 std::uint64_t expected_delivery,
                                 std::uint64_t expected_tick) {
    EnvelopeSax handler(retained, allow_retained_replay, expected_delivery,
                        expected_tick, data, length);
    bool parsed = false;
    try {
        parsed = Json::sax_parse(data, data + length, &handler);
    } catch (const Json::parse_error& error) {
        throw LedgerError(
            std::string("work_ledger segment JSON parse error at byte ")
            + std::to_string(error.byte) + ": " + error.what());
    }
    if (!parsed) {
        throw LedgerError("work_ledger segment JSON parse failed");
    }
    if (!handler.entered_top_object()) {
        throw LedgerError("pipeline ledger envelope must be an object");
    }
    if (!handler.finished()) {
        throw LedgerError("work_ledger segment ended unexpectedly");
    }
    return handler.take_result();
}

}  // namespace jointledger
