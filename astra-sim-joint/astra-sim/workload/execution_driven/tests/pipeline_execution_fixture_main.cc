/******************************************************************************
Cold execution-fixture producer (exec doc 19.1 / 19.1A, 2026-10-06).

CLI (fixed):
  JointPipelineExecutionFixture --state STATE.json --events EVENTS.json \
      --precision CONTRACT.json --out RESULT.json

The producer reconstructs a physical cold run from SHA256-verified asset
bytes (system/network/remote-memory configs), samples the initial
certificate from the freshly constructed Sys/backends, then drives the
event tape's closed kind set (graph_commit / external_resource /
request_complete; tick and gate_terminal are triggers, never kinds) through
the real GraphBatchCommitter / RemoteReadExecutor / FluidScheduler /
LocalHbmBandwidthModel / AnalyticalRemoteMemory stack.

Stage-A raw output (19.1A): boundary_records carry causal_prestate=null and
planner_state=null with top-level canonicalization_pending=true;
audit_records keep the ordered physical facts plus one raw_v2_frame audit
record per real boundary (full WorkLedgerV2Publisher envelope; despite the
publisher's historical V2 class name, its nested work-ledger schema and
protocol are v4; same-tick WorkLedgerV2RuntimeSnapshot sample, producer_build
with binary SHA256 and the frozen nine-field kernel contract). The isolated M01 micro integration
case also carries a bounded root-level delivery_frames array with each exact
publisher envelope (including non-boundary close/retire deliveries); the
focused audit entries remain one-per-boundary. Canonical six-domain prestates
are derived by the Python acceptance driver (Stage B), never faked here.

Convention for assets beyond the hardware block: the state asset list may
carry asset_id "comm_group" (the Sys communicator-group JSON). The workload
directory handed to Sys is a fresh empty private temp directory, exactly
like the proven seven-rank fixture.
******************************************************************************/

#include "astra-sim/common/Logging.hh"
#include "astra-sim/common/Common.hh"
#include "astra-sim/system/SendPacketEventHandlerData.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/workload/LocalHbmBandwidthModel.hh"
#include "astra-sim/workload/MetricCollector.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/DecisionMailbox.hh"
#include "astra-sim/workload/execution_driven/GraphBatchCommitter.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "astra-sim/workload/execution_driven/RemoteReadExecutor.hh"
#include "astra-sim/workload/execution_driven/TileTransportAdapter.hh"
#include "astra-sim/workload/execution_driven/WatchRegistry.hh"
#include "astra-sim/workload/execution_driven/WorkLedgerV2Publisher.hh"
#include "astra-sim/workload/execution_driven/WorkLedgerV2RuntimeSnapshot.hh"
#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include "extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.hh"
#include <astra-network-analytical/common/EventQueue.h>
#include <astra-network-analytical/common/NetworkParser.h>
#include <astra-network-analytical/congestion_aware/Helper.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace AstraSim;
using namespace AstraSim::ExecutionDriven;
using namespace Analytical;
using namespace AstraSimAnalytical;
using namespace AstraSimAnalyticalCongestionAware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

namespace {

/*****************************************************************************
 * Small utilities: file IO, SHA256, strict JSON access, fail-closed result.
 *****************************************************************************/

struct RunContext;  // forward

[[noreturn]] void fail(const std::string& message) {
    std::fprintf(stderr, "[pipeline_execution_fixture] FAIL: %s\n",
                 message.c_str());
    std::fflush(stderr);
    std::exit(1);
}

std::string read_whole_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot open " + path);
    std::string bytes((std::istreambuf_iterator<char>(in)),
                      std::istreambuf_iterator<char>());
    if (!in.eof() && in.fail()) fail("short read on " + path);
    return bytes;
}

nlohmann::json parse_json_file(const std::string& path,
                               const std::string& label) {
    nlohmann::json value;
    try {
        value = nlohmann::json::parse(read_whole_file(path));
    } catch (const std::exception& error) {
        fail(label + " is not valid JSON (" + path + "): " + error.what());
    }
    if (!value.is_object()) fail(label + " must be a JSON object");
    return value;
}

void write_json_file(const std::string& path, const nlohmann::json& value) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) fail("cannot create output " + path);
    out << value.dump(2) << '\n';
    out.close();
    if (!out.good()) fail("failed writing output " + path);
}

// ---------------------------------------------------------------------------
// SHA256 (FIPS 180-4), self-contained: asset digest verification and the
// producer_build binary hash have no shared helper in the vendored deps.
// ---------------------------------------------------------------------------
class Sha256 {
  public:
    static std::string hex(const std::string& bytes) {
        Sha256 hash;
        hash.update(bytes);
        return hash.final_hex();
    }

  private:
    void update(const std::string& bytes) {
        total_bits_ += static_cast<uint64_t>(bytes.size()) * 8;
        for (const char c : bytes) {
            buffer_[fill_++] = static_cast<uint8_t>(c);
            if (fill_ == 64) {
                transform(buffer_.data());
                fill_ = 0;
            }
        }
    }

    std::string final_hex() {
        const uint64_t message_bits = total_bits_;  // padding must not count
        const size_t pad_bytes = (fill_ < 56 ? 56 - fill_ : 120 - fill_);
        update(std::string(1, static_cast<char>(0x80)) +
               std::string(pad_bytes - 1, '\0'));
        std::array<uint8_t, 8> length{};
        for (size_t i = 0; i < 8; ++i) {
            length[i] = static_cast<uint8_t>(message_bits >> (56 - 8 * i));
        }
        update(std::string(reinterpret_cast<const char*>(length.data()), 8));
        std::string out;
        out.reserve(64);
        static const char* kHex = "0123456789abcdef";
        for (const uint32_t word : state_) {
            for (int shift = 28; shift >= 0; shift -= 4) {
                out.push_back(kHex[(word >> shift) & 0xF]);
            }
        }
        return out;
    }

    explicit Sha256() : state_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                               0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19} {}

    static uint32_t rotr(const uint32_t value, const int bits) {
        return (value >> bits) | (value << (32 - bits));
    }

    void transform(const uint8_t* block) {
        static const std::array<uint32_t, 64> k = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b,
            0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
            0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7,
            0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152,
            0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
            0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
            0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
            0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
            0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        std::array<uint32_t, 64> w{};
        for (size_t i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
                   (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
                   static_cast<uint32_t>(block[i * 4 + 3]);
        }
        for (size_t i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^
                                (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^
                                (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (size_t i = 0; i < 64; ++i) {
            const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = h + s1 + ch + k[i] + w[i];
            const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = s0 + maj;
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    std::array<uint32_t, 8> state_;
    std::array<uint8_t, 64> buffer_{};
    size_t fill_ = 0;
    uint64_t total_bits_ = 0;
};

std::string self_binary_sha256() {
    return Sha256::hex(read_whole_file("/proc/self/exe"));
}

/*****************************************************************************
 * Strict wire checks for the three input documents.
 *****************************************************************************/

const nlohmann::json& require_field(const nlohmann::json& object,
                                    const char* key, const std::string& where,
                                    const char* type) {
    const auto it = object.find(key);
    if (it == object.end()) fail(std::string(where) + " misses '" + key + "'");
    if (type == std::string("object") && !it->is_object()) {
        fail(std::string(where) + "." + key + " must be an object");
    }
    if (type == std::string("array") && !it->is_array()) {
        fail(std::string(where) + "." + key + " must be an array");
    }
    if (type == std::string("string") && !it->is_string()) {
        fail(std::string(where) + "." + key + " must be a string");
    }
    if (type == std::string("uint") &&
        (!it->is_number_unsigned() || it->get<uint64_t>() < 0)) {
        fail(std::string(where) + "." + key + " must be a non-negative integer");
    }
    return *it;
}

void require_exact_keys(const nlohmann::json& object,
                        const std::vector<std::string>& keys,
                        const std::string& where) {
    if (!object.is_object()) fail(where + " must be an object");
    std::set<std::string> expected(keys.begin(), keys.end());
    for (auto it = object.begin(); it != object.end(); ++it) {
        if (expected.count(it.key()) == 0) {
            fail(where + " has unknown key '" + it.key() + "'");
        }
    }
    for (const std::string& key : keys) {
        if (object.find(key) == object.end()) {
            fail(where + " misses key '" + key + "'");
        }
    }
}

// Nine-field kernel contract (exec doc section 7): strict, unknown keys
// rejected, integers where the wire says integers.
nlohmann::json parse_precision_contract(const std::string& path) {
    const nlohmann::json contract = parse_json_file(path, "precision contract");
    require_exact_keys(contract,
                       {"projection_abi_version", "descriptor_version",
                        "pipeline_protocol_version", "model_version",
                        "timer_whitelist_version", "window_tile_limit",
                        "window_local_error_ppm", "window_refine_margin_ppm",
                        "prediction_config_sha256"},
                       "precision contract");
    for (const char* key : {"projection_abi_version", "descriptor_version",
                            "pipeline_protocol_version", "window_tile_limit",
                            "window_local_error_ppm",
                            "window_refine_margin_ppm"}) {
        if (!require_field(contract, key, "precision contract", "uint")
                 .is_number_unsigned()) {
            fail(std::string("precision contract.") + key + " must be uint");
        }
    }
    for (const char* key : {"model_version", "timer_whitelist_version",
                            "prediction_config_sha256"}) {
        require_field(contract, key, "precision contract", "string");
    }
    const std::string digest =
        contract.at("prediction_config_sha256").get<std::string>();
    bool hex_ok = digest.size() == 64;
    for (const char c : digest) {
        hex_ok = hex_ok && ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
    }
    if (!hex_ok) {
        fail("precision contract.prediction_config_sha256 must be 64 lowercase hex");
    }
    return contract;
}

/*****************************************************************************
 * State parsing: assets verified against real bytes before any Sys build.
 *****************************************************************************/

struct AssetEntry {
    std::string asset_id;
    std::string path;    // resolved absolute path
    std::string sha256;
};

struct FixtureInputs {
    nlohmann::json state;
    nlohmann::json tape;
    nlohmann::json precision;
    std::map<std::string, AssetEntry> assets;
    std::string request_id;
    std::string fixed_instance_id;
    std::string fixed_action;
    uint64_t seed = 0;
    int rank_count = 0;
    std::string case_id;
};

std::map<std::string, AssetEntry> verify_assets(const nlohmann::json& state,
                                                const std::string& state_path) {
    const nlohmann::json& list = require_field(state, "assets", "state", "array");
    const std::filesystem::path base =
        std::filesystem::path(state_path).parent_path();
    std::map<std::string, AssetEntry> by_id;
    for (const auto& entry : list) {
        if (!entry.is_object()) fail("asset entry must be an object");
        require_exact_keys(entry, {"asset_id", "path", "sha256"}, "asset entry");
        AssetEntry asset;
        asset.asset_id = entry.at("asset_id").get<std::string>();
        asset.sha256 = entry.at("sha256").get<std::string>();
        const std::string relative = entry.at("path").get<std::string>();
        if (asset.asset_id.empty() || relative.empty()) {
            fail("asset id/path must be non-empty");
        }
        if (by_id.count(asset.asset_id) != 0) {
            fail("duplicate asset id " + asset.asset_id);
        }
        const std::filesystem::path relative_path(relative);
        if (relative_path.is_absolute() ||
            relative.find("..") != std::string::npos) {
            fail("asset path must stay inside the state directory: " + relative);
        }
        std::filesystem::path resolved =
            std::filesystem::weakly_canonical(base / relative_path);
        const std::string bytes = read_whole_file(resolved.string());
        const std::string digest = Sha256::hex(bytes);
        if (digest != asset.sha256) {
            fail("asset digest mismatch for " + relative + ": expected " +
                 asset.sha256 + ", real bytes hash " + digest);
        }
        asset.path = resolved.string();
        by_id.emplace(asset.asset_id, std::move(asset));
    }
    if (by_id.empty()) fail("state must carry at least one asset");
    return by_id;
}

FixtureInputs load_inputs(const std::string& state_path,
                          const std::string& events_path,
                          const std::string& precision_path) {
    FixtureInputs inputs;
    inputs.state = parse_json_file(state_path, "fixture state");
    inputs.tape = parse_json_file(events_path, "event tape");
    inputs.precision = parse_precision_contract(precision_path);

    require_exact_keys(inputs.state,
                       {"version", "kind", "case_id", "cluster_id", "scenario_id",
                        "seed", "request_id", "fixed_instance_id", "fixed_action",
                        "causal_input", "assets", "hardware", "initial_state",
                        "python_runtime_asset_id"},
                       "fixture state");
    if (inputs.state.at("version").get<std::string>() !=
            "execution-fixture-v1-20261006" ||
        inputs.state.at("kind").get<std::string>() != "cold_initial") {
        fail("fixture state version/kind must be execution-fixture-v1-20261006 / cold_initial");
    }
    inputs.case_id = inputs.state.at("case_id").get<std::string>();
    inputs.request_id = inputs.state.at("request_id").get<std::string>();
    inputs.fixed_instance_id =
        inputs.state.at("fixed_instance_id").get<std::string>();
    inputs.fixed_action = inputs.state.at("fixed_action").get<std::string>();
    inputs.seed = inputs.state.at("seed").get<uint64_t>();
    const nlohmann::json& hardware =
        require_field(inputs.state, "hardware", "state", "object");
    inputs.rank_count =
        static_cast<int>(require_field(hardware, "rank_count", "state.hardware", "uint")
                             .get<uint64_t>());
    if (inputs.rank_count < 1) fail("rank_count must be positive");
    inputs.assets = verify_assets(inputs.state, state_path);

    // Event tape structural check (closed trigger set; kind checked in the loop).
    require_exact_keys(inputs.tape, {"version", "case_id", "events"}, "event tape");
    if (inputs.tape.at("version").get<std::string>() !=
        "execution-event-tape-v1-20261006") {
        fail("event tape version must be execution-event-tape-v1-20261006");
    }
    if (inputs.tape.at("case_id").get<std::string>() != inputs.case_id) {
        fail("event tape case_id differs from the state");
    }
    int64_t previous_ordinal = -1;
    for (const auto& event : require_field(inputs.tape, "events", "event tape", "array")) {
        if (!event.is_object()) fail("tape event must be an object");
        require_exact_keys(event, {"ordinal", "trigger", "kind", "payload"},
                           "tape event");
        const int64_t ordinal = event.at("ordinal").get<int64_t>();
        if (ordinal <= previous_ordinal) {
            fail("tape ordinals must strictly increase");
        }
        previous_ordinal = ordinal;
        const nlohmann::json& trigger = event.at("trigger");
        if (!trigger.is_object() || !trigger.contains("kind") ||
            !trigger.at("kind").is_string()) {
            fail("tape event trigger must carry a string kind");
        }
        const std::string kind = trigger.at("kind").get<std::string>();
        if (kind == "tick") {
            require_exact_keys(trigger, {"kind", "tick_ns"}, "tick trigger");
        } else if (kind == "gate_terminal") {
            require_exact_keys(trigger, {"kind", "segment_id", "role"},
                               "gate_terminal trigger");
        } else {
            fail("unknown trigger kind " + kind);
        }
        if (!event.at("kind").is_string() || event.at("kind").empty()) {
            fail("tape event kind must be a non-empty string");
        }
        if (!event.at("payload").is_object()) {
            fail("tape event payload must be an object");
        }
    }
    return inputs;
}

/*****************************************************************************
 * Physical runtime assembled from the verified assets.
 *****************************************************************************/

struct TransferObservation {
    TileTransportAdapter::ActiveTransfer transfer;
    uint64_t issue_tick = 0;
    bool completed = false;
};

struct StageRun {
    std::string stage_label;           // "P" or "D"
    std::string wire_stage;            // "prefill" or "decode"
    std::string fixed_instance_id;
    std::string fixed_action;
    std::string segment_id;
    uint64_t boundary_ordinal = 0;
    uint64_t start_tick = 0;
    bool service_recorded = false;
    uint64_t service_tick = 0;
    bool settle_recorded = false;
    uint64_t settle_tick = 0;
    uint64_t logical_events_at_start = 0;
    uint64_t proceed_steps_at_start = 0;
    uint64_t logical_events_at_settle = 0;
    uint64_t proceed_steps_at_settle = 0;
    bool measurement_emitted = false;
};

struct TerminalRecord {
    int rank = -1;
    uint64_t node_id = 0;
    uint64_t tick = 0;
};

std::string temporary_directory() {
    const std::string pattern = "/tmp/joint_pipeline_fixture_XXXXXX";
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    const char* const path = ::mkdtemp(buffer.data());
    if (path == nullptr) fail("mkdtemp failed");
    return std::string(path);
}

// Background external_resource drain counter: counts real completions and
// takes ownership of any WorkloadLayerHandlerData the completion carries
// (the remote-port path transfers the cookie to the sink exactly once).
struct BackgroundSink final : public Callable {
    explicit BackgroundSink(std::string label)
        : label_(std::move(label)) {}
    void call(EventType, CallData* data) override {
        ++completions;
        if (data != nullptr) {
            delete static_cast<WorkloadLayerHandlerData*>(data);
        }
    }
    std::string label_;
    uint64_t completions = 0;
};

// Deferred issue-pass plumbing: plain functions so the CompletionObserver /
// hook registrations receive real function pointers (same shape as the
// proven seven-rank fixture).
struct FixtureIssuePassState {
    std::vector<Workload*>* workloads = nullptr;
};

void fixture_issue_pass_callback(void* opaque, const int rank) {
    auto* state = static_cast<FixtureIssuePassState*>(opaque);
    if (state == nullptr || state->workloads == nullptr || rank < 0 ||
        static_cast<size_t>(rank) >= state->workloads->size()) {
        fail("corrupt deferred issue pass");
    }
    (*state->workloads)[rank]->issue_dep_free_nodes();
}

struct FixtureTerminalHookContext {
    std::vector<TerminalRecord>* records = nullptr;
    OnlineCompletionHookContext* online_hook = nullptr;
    nlohmann::json* audit = nullptr;
};

void fixture_terminal_hook(void* opaque, const int rank, const uint64_t node_id,
                           const char* request_id, const char* stage,
                           const uint64_t generation, const uint64_t tick,
                           const int terminal_status) {
    auto* context = static_cast<FixtureTerminalHookContext*>(opaque);
    if (context == nullptr || context->records == nullptr ||
        context->online_hook == nullptr ||
        terminal_status != static_cast<int>(NodeTerminalStatus::Success)) {
        fail("invalid physical terminal callback");
    }
    context->records->push_back(TerminalRecord{rank, node_id, tick});
    context->audit->push_back({{"kind", "node_terminal"},
                               {"rank", rank},
                               {"node_id", node_id},
                               {"tick", tick},
                               {"request_id", request_id},
                               {"stage", stage},
                               {"generation", generation}});
    online_completion_hook(context->online_hook, rank, node_id, request_id,
                           stage, generation, tick, terminal_status);
}

/*****************************************************************************
 * main
 *****************************************************************************/

int protected_main(int argc, char** argv) {
    std::string state_path;
    std::string events_path;
    std::string precision_path;
    std::string out_path;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const bool has_value = index + 1 < argc;
        if (argument == "--state" && has_value) {
            state_path = argv[++index];
        } else if (argument == "--events" && has_value) {
            events_path = argv[++index];
        } else if (argument == "--precision" && has_value) {
            precision_path = argv[++index];
        } else if (argument == "--out" && has_value) {
            out_path = argv[++index];
        } else if (argument == "--help") {
            std::printf("usage: JointPipelineExecutionFixture --state STATE.json "
                        "--events EVENTS.json --precision CONTRACT.json "
                        "--out RESULT.json\n");
            return 0;
        } else {
            fail("unknown or incomplete argument: " + argument);
        }
    }
    if (state_path.empty() || events_path.empty() || precision_path.empty() ||
        out_path.empty()) {
        fail("all of --state/--events/--precision/--out are required");
    }

    setvbuf(stdout, nullptr, _IONBF, 0);
    LoggerFactory::init("empty", "off");
    MetricCollector::instance().initialize("empty", "off");

    FixtureInputs inputs = load_inputs(state_path, events_path, precision_path);

    // The failure result carries everything gathered so far; the harness
    // still writes the output file so the driver can inspect the reason.
    const std::string binary_sha = self_binary_sha256();
    const auto asset_path = [&inputs](const std::string& id) -> const std::string& {
        const auto it = inputs.assets.find(id);
        if (it == inputs.assets.end()) {
            fail("state references missing asset '" + id + "'");
        }
        return it->second.path;
    };

    const nlohmann::json& hardware = inputs.state.at("hardware");
    const std::string& system_asset = asset_path(
        hardware.at("system_asset_id").get<std::string>());
    const std::string& network_asset = asset_path(
        hardware.at("network_asset_id").get<std::string>());
    const std::string& remote_memory_asset = asset_path(
        hardware.at("remote_memory_asset_id").get<std::string>());
    if (!hardware.at("die_hardware_asset_id").is_null()) {
        // Verified like every asset; the die backend itself is initialised by
        // Sys from the system configuration, nothing extra to wire here.
        asset_path(hardware.at("die_hardware_asset_id").get<std::string>());
    }
    asset_path(inputs.state.at("python_runtime_asset_id").get<std::string>());
    const auto comm_group_it = inputs.assets.find("comm_group");
    if (comm_group_it == inputs.assets.end()) {
        fail("state must carry a SHA-verified 'comm_group' asset for the Sys build");
    }
    const std::string& comm_group_asset = comm_group_it->second.path;

    const std::string work_directory = temporary_directory();
    std::filesystem::create_directories(work_directory + "/workload");
    struct TempDirCleanup {
        std::string path;
        ~TempDirCleanup() {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
        }
    } temp_dir_cleanup{work_directory};

    // ---- real backends from verified bytes --------------------------------
    auto event_queue = std::make_shared<EventQueue>();
    const NetworkParser network_parser(network_asset);
    const auto topology = construct_topology(network_parser);
    if (topology->get_npus_count() != inputs.rank_count) {
        fail("network asset rank count differs from state.hardware.rank_count");
    }
    CongestionAwareNetworkApi::set_event_queue(event_queue);
    CongestionAwareNetworkApi::set_topology(topology);
    auto fluid_scheduler = std::make_shared<FluidScheduler>(
        event_queue, topology->get_directed_links(),
        network_parser.get_fluid_max_active_flows(),
        network_parser.get_fluid_max_route_memberships(),
        network_parser.get_progress_report_event_interval());
    fluid_scheduler->set_deferred_flush_mode(true);
    CongestionAwareNetworkApi::set_fluid_scheduler(fluid_scheduler);

    auto memory_api =
        std::make_unique<AnalyticalRemoteMemory>(remote_memory_asset);
    std::vector<std::unique_ptr<CongestionAwareNetworkApi>> network_apis;
    std::vector<std::shared_ptr<NodeStoreGraphSource>> graph_sources;
    std::vector<std::unique_ptr<Sys>> sys_owners;
    std::vector<Sys*> systems;
    std::vector<Workload*> workloads;
    const auto npus_per_dim = network_parser.get_npus_counts_per_dim();
    std::vector<int> queues_per_dim(npus_per_dim.size(), 1);
    for (int rank = 0; rank < inputs.rank_count; ++rank) {
        network_apis.push_back(
            std::make_unique<CongestionAwareNetworkApi>(rank));
        auto source = std::make_shared<NodeStoreGraphSource>();
        auto system = std::make_unique<Sys>(
            rank, work_directory + "/workload", comm_group_asset, system_asset,
            memory_api.get(), network_apis.back().get(), npus_per_dim,
            queues_per_dim, 1.0, 1.0, false, ExecutionMode::Online, source);
        systems.push_back(system.get());
        workloads.push_back(system->workload);
        graph_sources.push_back(std::move(source));
        sys_owners.push_back(std::move(system));
    }

    // ---- initial certificate from the fresh components --------------------
    // Physical facts are sampled from the live objects; only the Python-owned
    // directory/capacity/history JSON is carried through uninterpreted.
    const auto export_initial_certificate = [&]() -> nlohmann::json {
        nlohmann::json ranks = nlohmann::json::array();
        for (int rank = 0; rank < inputs.rank_count; ++rank) {
            ranks.push_back({{"rank", rank},
                             {"next_store_id",
                              graph_sources[rank]->store().next_auto_id()}});
        }
        nlohmann::json components = nlohmann::json::array();
        const auto directed_links = topology->get_directed_links();
        for (size_t link = 0; link < directed_links.size(); ++link) {
            components.push_back(
                {{"identity", {{"kind", "noc_link"},
                               {"link_id", link},
                               {"fluid_version",
                                fluid_scheduler->link_resource_version(link)}}},
                  {"version", fluid_scheduler->link_resource_version(link)},
                  {"state", {{"active_flows", nlohmann::json::array()},
                             {"propagation_tails", nlohmann::json::array()}}}});
        }
        for (int rank = 0; rank < inputs.rank_count; ++rank) {
            components.push_back(
                {{"identity", {{"kind", "hbm_rank"}, {"rank", rank}}},
                  {"version", 0},
                  {"state", {{"hbm_jobs", nlohmann::json::array()}}}});
            components.push_back(
                {{"identity", {{"kind", "node_store"}, {"rank", rank}}},
                  {"version", graph_sources[rank]->store().next_auto_id() - 1},
                  {"state", {{"nodes", nlohmann::json::array()},
                             {"ready", nlohmann::json::array()},
                             {"deferred", nlohmann::json::array()},
                             {"pending_events", nlohmann::json::array()},
                             {"watches", nlohmann::json::array()},
                             {"descriptors", nlohmann::json::array()},
                             {"cursors", nlohmann::json::array()},
                             {"buffers", nlohmann::json::array()},
                             {"obligations", nlohmann::json::array()},
                             {"hardware_inflight", 0}}}});
        }
        for (size_t port = 0; port < memory_api->port_count(); ++port) {
            components.push_back(
                {{"identity", {{"kind", "remote_port"},
                               {"port_index", port}}},
                  {"version", 0},
                  {"state", {{"pool_jobs", nlohmann::json::array()}}}});
        }
        const nlohmann::json& cold = inputs.state.at("initial_state");
        return {{"tick_ns", Sys::boostedTick()},
                {"next_flow_id",
                 static_cast<uint64_t>(fluid_scheduler->get_active_flow_count())},
                {"next_tag", 1},
                {"ranks", std::move(ranks)},
                {"components", std::move(components)},
                {"directory", cold.at("directory")},
                {"capacity", cold.at("capacity")},
                {"history", cold.at("history")}};
    };

    // ---- observer / publisher wiring (mirrors the proven fixture) ---------
    std::vector<TerminalRecord> terminals;
    nlohmann::json audit_records = nlohmann::json::array();
    const bool capture_micro_delivery_frames =
        inputs.case_id == "M01-micro-p2-d3-seed000";
    nlohmann::json delivery_frames = nlohmann::json::array();
    std::vector<StageRun> stages;
    nlohmann::json drained_blocked_snapshot = nullptr;
    // Pre-close live view taken at request_complete entry (audit-only). The
    // failure-path capture below overwrites drained_blocked_snapshot with the
    // post-close (cursor-force-exhausted) view; the pre-close view is the one
    // that carries live cursor positions, so it is kept in its own field.
    nlohmann::json pre_close_blocked_snapshot = nullptr;
    // Same-entry NodeStore work-ledger capture of every UNFINISHED node with
    // its blocking parents (adjudicates missed-wake vs wire cycle: the
    // pre-close blocking chain of the stalled ordinary nodes, before the
    // close's force-exhaustion pump mutates anything).
    nlohmann::json pre_close_unfinished_ledger = nullptr;
    nlohmann::json pre_close_resource_occupancy = nullptr;
    // Round-22 calibration surfaces for the drain-first window: steps
    // consumed and ordinary terminals produced while the window ran
    // (UINT64_MAX = the window never ran: the executor was already
    // drained at request_complete entry). Deterministic values; they let
    // the fall-through budget be calibrated from the resolved cases'
    // actual work instead of a magic constant, and let a bounded
    // fall-through's FAIL certificate show whether the engine produced
    // far more terminals than the tape declared (regeneration signature).
    uint64_t pre_close_drain_steps = std::numeric_limits<uint64_t>::max();
    uint64_t pre_close_drain_terminal_base = 0;
    uint64_t pre_close_drain_terminals = 0;
    bool pre_close_drain_mem_capped = false;
    bool post_close_drain_mem_capped = false;
    std::map<uint64_t, std::set<std::pair<int, uint64_t>>>
        boundary_node_keys;  // ordinary (descriptor-less) boundaries
    uint64_t physical_steps = 0;
    std::map<uint64_t, TransferObservation> transfer_observations;

    WatchRegistry watch_registry;
    OnlineCompletionHookContext completion_context;
    completion_context.registry = &watch_registry;
    FixtureIssuePassState issue_pass_state{&workloads};
    completion_context.configure_issue_passes(
        event_queue.get(), workloads.size(), &fixture_issue_pass_callback,
        &issue_pass_state);
    FixtureTerminalHookContext terminal_context{&terminals, &completion_context,
                                                &audit_records};
    CompletionObserver::instance().set_hook(&fixture_terminal_hook,
                                            &terminal_context);

    TileTransportAdapter transport(systems, workloads);
    RemoteReadExecutor executor(
        graph_sources, workloads, &transport,
        [&completion_context](const int rank) {
            completion_context.schedule_issue_pass(rank);
        });
    WorkLedgerV2Publisher v2_publisher("joint-pipeline-execution-fixture");
    uint64_t next_delivery_sequence = 0;
    nlohmann::json last_published_frame;
    const auto publish_v2_frame = [&]() {
        const uint64_t tick = Sys::boostedTick();
        nlohmann::json cursor_patches =
            executor.take_cursor_patches(tick);
        const nlohmann::json frame = v2_publisher.publish(
            next_delivery_sequence, tick, std::move(cursor_patches),
            collect_work_ledger_v2_active_resources(
                graph_sources, systems, memory_api.get(), fluid_scheduler,
                &executor, &transport, tick));
        audit_records.push_back(
            {{"kind", "v2_frame_published"},
             {"delivery_sequence", next_delivery_sequence},
             {"tick_ns", tick},
             {"logical_seq_start", frame.at("logical_seq_start")},
             {"logical_seq_end", frame.at("logical_seq_end")},
             {"logical_event_count", frame.at("logical_event_count")}});
        ++next_delivery_sequence;
        last_published_frame = frame;
        if (capture_micro_delivery_frames) delivery_frames.push_back(frame);
        return frame;
    };
    executor.set_ledger_publishers(
        [&v2_publisher, &graph_sources, &executor](
            const std::string& kind, const int rank, const uint64_t node_id,
            const OnlineNode& node, const std::vector<uint64_t>& parents) {
            append_work_ledger_v2_node_transition(
                v2_publisher, graph_sources, &executor, kind, rank, node_id,
                node, parents, Sys::boostedTick());
        },
        [&v2_publisher](nlohmann::json transition) {
            v2_publisher.append_pipeline_transition(std::move(transition));
        },
        [&v2_publisher](nlohmann::json declaration) {
            v2_publisher.append_descriptor_commit(std::move(declaration));
        });
    for (Workload* workload : workloads) {
        workload->set_ledger_transition_observer(
            [&v2_publisher, &graph_sources, &executor](
                const std::string& kind, const NodeView& node) {
                append_work_ledger_v2_node_transition(
                    v2_publisher, graph_sources, &executor, kind, node.rank,
                    node.global_id, node, {}, Sys::boostedTick());
            });
    }
    executor.set_watch_reference_counter(
        [&watch_registry](const auto& progressions) {
            std::vector<WatchRegistry::MemberProgression> ranges;
            ranges.reserve(progressions.size());
            for (const auto& progression : progressions) {
                ranges.push_back({progression.rank, progression.first_store_id,
                                  progression.count, progression.stride});
            }
            return watch_registry.member_reference_count(ranges);
        });
    for (Workload* workload : workloads) {
        workload->set_remote_read_executor(&executor);
    }
    const auto flow_bound_audit =
        [&transport, &audit_records](const AstraNetworkAPI::LogicalFlowKey& key,
                                     const FlowId flow_id) {
        audit_records.push_back(
            {{"kind", "flow_bound"},
             {"flow_id", flow_id},
             {"tick_ns", Sys::boostedTick()},
             {"descriptor_identity", key.descriptor_identity},
             {"generation", key.generation},
             {"leg", key.leg},
             {"tile", key.tile},
             {"shard", key.shard},
             {"rank", key.rank},
             {"tag", key.tag}});
        transport.on_flow_bound(key, flow_id);
    };
    CongestionAwareNetworkApi::set_flow_bound_callback(flow_bound_audit);
    struct FlowCallbackReset {
        ~FlowCallbackReset() {
            CongestionAwareNetworkApi::set_flow_bound_callback({});
        }
    } flow_callback_reset;
    const auto watch_fire_notifier = [&executor, &audit_records](
                                          const WatchFire& fire) {
        audit_records.push_back({{"kind", "watch_fire"},
                                 {"request_id", fire.request_id},
                                 {"stage", fire.stage},
                                 {"generation", fire.generation},
                                 {"tick_ns", Sys::boostedTick()}});
        executor.notify_service_gate(fire.request_id, fire.stage,
                                      fire.generation);
        executor.notify_watch_registry_changed();
    };
    watch_registry.set_fire_notifier(watch_fire_notifier);

    GraphBatchCommitter::Context committer_context;
    committer_context.num_ranks = inputs.rank_count;
    committer_context.graph_sources = &graph_sources;
    committer_context.watch_registry = &watch_registry;
    committer_context.publish_node_transition =
        [&v2_publisher, &graph_sources, &executor](
            const std::string& kind, const int rank, const uint64_t node_id,
            const OnlineNode& node, const std::vector<uint64_t>& parents) {
            append_work_ledger_v2_node_transition(
                v2_publisher, graph_sources, &executor, kind, rank, node_id,
                node, parents, Sys::boostedTick());
        };
    committer_context.issue_rank = [&workloads](const int rank) {
        workloads.at(static_cast<size_t>(rank))->issue_dep_free_nodes();
    };
    // Collective-liveness preflight: numeric pg names resolve against the
    // real Workload comm_groups (declared by the comm_group asset); the
    // production graph emits TP groups under symbolic names (e.g.
    // "tp_decode"), so those cross-check the batch's own collective
    // participant sets for self-consistency. NOTE: Workload's execution-time
    // extract_comm_group still requires numeric keys -- symbolic-pg batches
    // commit but cannot yet execute collectives; the divergence is reported
    // through the evidence-package handoff for the main agent's ruling.
    committer_context.communicator_members_for_pg =
        [&workloads](const std::string& pg_name,
                     const std::vector<int>& participant_ranks) {
            bool numeric = !pg_name.empty() &&
                pg_name.find_first_not_of("0123456789") == std::string::npos;
            if (numeric) {
                const int pg_id = std::atoi(pg_name.c_str());
                for (const Workload* workload : workloads) {
                    const auto it = workload->comm_groups.find(pg_id);
                    if (it == workload->comm_groups.end() || !it->second) {
                        return std::optional<std::vector<int>>(std::nullopt);
                    }
                }
            }
            std::vector<int> sorted = participant_ranks;
            std::sort(sorted.begin(), sorted.end());
            sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
            return std::optional<std::vector<int>>(sorted);
        };
    committer_context.prepare_pipeline_batch =
        [&executor](const GraphBatch& batch,
                    const GraphBatchCommitter& committer) {
            return executor.prepare_batch(batch, committer);
        };
    committer_context.commit_pipeline_batch =
        [&executor](std::shared_ptr<void> plan, const GraphBatch& batch,
                    const GraphBatchCommitter& committer) {
            executor.commit_batch(std::move(plan), batch, committer);
        };
    committer_context.commit_kernel_contract =
        [&workloads](std::shared_ptr<const PipelineKernelContract> contract) {
            for (Workload* workload : workloads) {
                workload->set_pipeline_kernel_contract(contract);
            }
        };
    GraphBatchCommitter committer(std::move(committer_context));

    // ---- result assembly helpers ------------------------------------------
    std::string failure_reason;
    const auto sample_transfers = [&]() {
        const uint64_t tick = Sys::boostedTick();
        const auto active = transport.active_snapshot();
        std::set<uint64_t> active_ids;
        for (const auto& transfer : active) {
            active_ids.insert(transfer.transfer_id);
            auto it = transfer_observations.find(transfer.transfer_id);
            if (it == transfer_observations.end()) {
                transfer_observations.emplace(
                    transfer.transfer_id,
                    TransferObservation{transfer, transfer.submit_tick, false});
            } else {
                it->second.transfer = transfer;
            }
        }
        for (auto& entry : transfer_observations) {
            if (!entry.second.completed &&
                active_ids.count(entry.first) == 0) {
                entry.second.completed = true;
            }
        }
    };
    // Delivery 0: the initial protocol-2 FULL frame doubles as the first
    // boundary's prestate so the Stage-B canonicalizer can bootstrap the
    // double ledger from a full snapshot.
    publish_v2_frame();

    const auto descriptor_done = [](const nlohmann::json& snapshot,
                                    const std::string& segment,
                                    const char* field) {
        for (const auto& entry : snapshot.at("descriptor_commits")) {
            if (entry.at("identity").at("segment_id").get<std::string>() ==
                segment) {
                return entry.at(field).get<bool>();
            }
        }
        // A segment absent from the active snapshot has completed and been
        // retired (or was administratively closed with the staging lease):
        // both service and settle are done for it.
        (void)field;
        return true;
    };

    uint64_t boundary_counter = 0;
    const auto record_stage_transitions = [&]() {
        // Service/settle ticks are sampled from the real descriptor snapshot
        // after every physical step; a flip is recorded exactly once.
        if (stages.empty() || !event_queue) return;
        const uint64_t tick = Sys::boostedTick();
        const nlohmann::json snapshot = executor.authoritative_snapshot(tick);
        std::set<std::pair<int, uint64_t>> terminal_keys;
        for (const TerminalRecord& record : terminals) {
            terminal_keys.insert({record.rank, record.node_id});
        }
        for (auto& stage : stages) {
            if (stage.settle_recorded) continue;
            bool service = false;
            bool settle = false;
            if (stage.segment_id.empty()) {
                // Ordinary (descriptor-less) boundary: completion is the
                // full terminal set of its committed nodes. Service and
                // settle coincide (no executor cursor separates them); the
                // stage watch / physical drain distinctions of descriptor
                // batches do not exist for plain graphs.
                const auto keys = boundary_node_keys.find(
                    stage.boundary_ordinal);
                if (keys == boundary_node_keys.end() ||
                    keys->second.empty()) {
                    continue;
                }
                settle = std::includes(terminal_keys.begin(),
                                       terminal_keys.end(),
                                       keys->second.begin(),
                                       keys->second.end());
                service = settle;
            } else {
                service = !stage.service_recorded &&
                    descriptor_done(snapshot, stage.segment_id,
                                    "service_done");
                settle = descriptor_done(snapshot, stage.segment_id,
                                         "settle_done");
            }
            if (service && !stage.service_recorded) {
                stage.service_recorded = true;
                stage.service_tick = tick;
            }
            if (settle && !stage.settle_recorded) {
                stage.settle_recorded = true;
                stage.settle_tick = tick;
                // Snapshot the stage-local counters at settle so the
                // measurement row covers THIS stage's window only.
                stage.logical_events_at_settle =
                    v2_publisher.next_logical_sequence() - 1;
                stage.proceed_steps_at_settle = physical_steps;
            }
        }
    };

    uint64_t last_sampled_terminal_count = 0;
    const auto pump_once = [&]() {
        event_queue->proceed();
        ++physical_steps;
        sample_transfers();
        // Service/settle flips only happen at terminal (or watch-fire)
        // activity, both of which append to `terminals` -- sampling the
        // authoritative snapshot ONLY then keeps the per-step cost
        // independent of the active-descriptor count (a per-step full
        // snapshot embeds every descriptor wire and stalled multi-train
        // tapes: signature 4, M03/M04/M07).
        if (terminals.size() != last_sampled_terminal_count) {
            last_sampled_terminal_count = terminals.size();
            record_stage_transitions();
        }
    };
    const auto pump_until = [&](const std::function<bool()>& done,
                                const std::string& label) {
        uint64_t guard = 0;
        while (!done()) {
            if (event_queue->finished()) {
                if (!event_queue->has_deferred_work()) break;
                // Defect-C shape (EventQueue contract): the main map is
                // empty but the same-tick deferred queue still holds
                // unexecuted events (scheduled outside a proceed context).
                // finished() is blind to them and no proceed() would ever
                // run to drain them, so the tape would freeze with issued
                // work pending. Force the explicit next decision boundary
                // (the same mechanism main_online.cc and the drain
                // harnesses use) so a proceed() executes and drains the
                // deferred queue. No wait gate is added and the emission
                // order is unchanged -- this only lets the already-started
                // deferred work actually run.
                event_queue->schedule_event(
                    event_queue->get_current_time() + 1,
                    [](void*) {}, nullptr);
            }
            pump_once();
            if (++guard > 2000000) fail(label + " exceeded event guard");
        }
        record_stage_transitions();
    };
    const auto advance_to_tick = [&](const uint64_t target_tick) {
        while (Sys::boostedTick() < target_tick && !event_queue->finished()) {
            pump_once();
        }
    };

    bool event_loop_started = false;
    const auto ensure_event_loop = [&]() {
        fluid_scheduler->flush_pending_starts();
        if (!event_loop_started) {
            fluid_scheduler->mark_event_loop_started();
            event_loop_started = true;
        }
    };

    // Background external_resource sinks; kept alive until the end.
    std::vector<std::unique_ptr<BackgroundSink>> background_sinks;

    // ---- event tape execution ---------------------------------------------
    const auto commit_graph_batch = [&](const nlohmann::json& payload,
                                         uint64_t event_ordinal) {
        require_exact_keys(payload, {"batch", "decision_events", "boundary_record"},
                           "graph_commit payload");
        nlohmann::json batch = payload.at("batch");
        if (!batch.is_object()) fail("graph_commit batch must be an object");
        if (!batch.contains("schema_version") ||
            batch.at("schema_version").get<uint64_t>() != 2) {
            fail("graph_commit batch must be schema_version=2");
        }
        // Producer injects ONLY the frozen precision contract and protocol
        // identity; nodes/IDs/tags/quantities/order stay untouched.
        batch["kernel_contract"] = inputs.precision;
        if (!batch.contains("batch_id") || !batch.contains("source_delivery_sequence")) {
            fail("graph_commit batch must carry batch_id/source_delivery_sequence");
        }
        const uint64_t batch_id = batch.at("batch_id").get<uint64_t>();
        if (batch.at("source_delivery_sequence").get<uint64_t>() != batch_id) {
            fail("graph_commit batch_id != source_delivery_sequence");
        }

        std::vector<DecisionEvent> decision_events;
        for (const auto& wire_event : payload.at("decision_events")) {
            if (!wire_event.is_object()) {
                fail("decision event must be an object");
            }
            require_exact_keys(wire_event,
                               {"seq", "reason", "request_id", "stage",
                                "generation", "payload"},
                               "decision event");
            DecisionEvent event;
            event.seq = wire_event.at("seq").get<uint64_t>();
            const std::string reason =
                wire_event.at("reason").get<std::string>();
            if (reason == "ARRIVAL") {
                event.reason = DecisionReason::ARRIVAL;
            } else if (reason == "PREFILL_DRAIN") {
                event.reason = DecisionReason::PREFILL_DRAIN;
            } else if (reason == "DECODE_COMPLETION") {
                event.reason = DecisionReason::DECODE_COMPLETION;
            } else if (reason == "REQUEST_COMPLETE") {
                event.reason = DecisionReason::REQUEST_COMPLETE;
            } else {
                fail("unknown decision reason " + reason);
            }
            event.request_id = wire_event.at("request_id").get<std::string>();
            event.stage = wire_event.at("stage").get<std::string>();
            event.generation = wire_event.at("generation").get<uint64_t>();
            const nlohmann::json& pl = wire_event.at("payload");
            if (!pl.is_object()) fail("decision event payload must be an object");
            if (pl.contains("session_id")) {
                event.payload.session_id = pl.at("session_id").get<std::string>();
            }
            if (pl.contains("turn_index")) {
                event.payload.turn_index = pl.at("turn_index").get<int>();
            }
            if (pl.contains("prefill_length")) {
                event.payload.prefill_length =
                    pl.at("prefill_length").get<uint64_t>();
            }
            if (pl.contains("decode_length")) {
                event.payload.decode_length =
                    pl.at("decode_length").get<uint64_t>();
            }
            if (pl.contains("inter_request_interval_ns")) {
                event.payload.inter_request_interval_ns =
                    pl.at("inter_request_interval_ns").get<uint64_t>();
            }
            if (pl.contains("arrival_world_ns")) {
                event.payload.arrival_world_ns =
                    pl.at("arrival_world_ns").get<uint64_t>();
            }
            if (pl.contains("ingress_seq")) {
                event.payload.ingress_seq = pl.at("ingress_seq").get<uint64_t>();
            }
            if (pl.contains("queue_index")) {
                event.payload.queue_index = pl.at("queue_index").get<int64_t>();
            }
            if (pl.contains("watch_member_count")) {
                event.payload.watch_member_count =
                    pl.at("watch_member_count").get<uint64_t>();
            }
            decision_events.push_back(std::move(event));
        }
        if (decision_events.empty()) {
            fail("graph_commit requires at least one real decision event");
        }

        const uint64_t commit_tick = Sys::boostedTick();
        const StateDelta delta = build_state_delta(
            std::move(decision_events), commit_tick, batch_id);

        // Stage A raw frame: the boundary's prestate. For the first boundary
        // that is the initial protocol-2 full frame (delivery 0) already
        // sealed at wiring; later boundaries seal a fresh frame that carries
        // the previous stage's transitions. The same-tick runtime snapshot
        // accompanies it.
        const uint64_t boundary_ordinal = boundary_counter + 1;
        const nlohmann::json prestate_frame =
            boundary_counter == 0 ? last_published_frame : publish_v2_frame();
        const nlohmann::json runtime_snapshot = {
            {"tick_ns", commit_tick},
            {"active_resources", collect_work_ledger_v2_active_resources(
                 graph_sources, systems, memory_api.get(), fluid_scheduler,
                 &executor, &transport, commit_tick)}};

        const ParsedGraphBatch parsed = parse_graph_batch(batch, inputs.rank_count);
        const auto commit_result = committer.validate_and_commit(delta, parsed);
        if (commit_result.error.has_value()) {
            failure_reason = "graph_commit rejected batch " +
                             std::to_string(batch_id) + ": " + *commit_result.error;
            return;
        }
        ++boundary_counter;

        const bool empty_batch =
            (!batch.contains("nodes") || batch.at("nodes").empty()) &&
            (!batch.contains("pipeline_descriptors") ||
             batch.at("pipeline_descriptors").empty());
        if (empty_batch) {
            // Control-only delivery (no nodes, no descriptors): commit the
            // envelope for its delivery semantics, audit it, and skip the
            // stage/measurement machinery (no physical work exists).
            audit_records.push_back({{"kind", "graph_commit"},
                                     {"event_ordinal", event_ordinal},
                                     {"batch_id", batch_id},
                                     {"tick_ns", commit_tick},
                                     {"empty_batch", true}});
            ensure_event_loop();
            return;
        }
        StageRun stage;
        // The committed batch is the stage authority. Descriptor batches
        // classify via identity.stage (prefill/decode; a PREFILL_DRAIN event
        // legitimately carries the *prefill* stage while committing the
        // decode continuation). Ordinary action batches (stay/recompute/
        // copy without remote-read work) carry NO pipeline descriptors:
        // their stage comes from the committed nodes themselves and their
        // completion is measured over the node set (no executor cursor).
        const bool has_descriptors =
            batch.contains("pipeline_descriptors") &&
            !batch.at("pipeline_descriptors").empty();
        std::string wire_stage;
        if (has_descriptors) {
            wire_stage = batch.at("pipeline_descriptors")
                             .back()
                             .at("identity")
                             .at("stage")
                             .get<std::string>();
        } else {
            wire_stage = batch.at("nodes")
                             .front()
                             .at("stage")
                             .get<std::string>();
        }
        if (wire_stage != "prefill" && wire_stage != "decode") {
            fail("batch stage must be prefill or decode");
        }
        stage.wire_stage = wire_stage;
        stage.stage_label = wire_stage == "prefill" ? "P" : "D";
        stage.segment_id = [&]() {
            if (!has_descriptors) return std::string();
            const nlohmann::json& descriptors =
                batch.at("pipeline_descriptors");
            if (!descriptors.is_array() || descriptors.empty()) {
                fail("committed batch declared no pipeline descriptor");
            }
            const std::string segment = descriptors.back()
                                             .at("identity")
                                             .at("segment_id")
                                             .get<std::string>();
            // Cross-check the descriptor is really live in the executor.
            const nlohmann::json snapshot =
                executor.authoritative_snapshot(Sys::boostedTick());
            for (const auto& entry : snapshot.at("descriptor_commits")) {
                if (entry.at("identity").at("segment_id").get<std::string>() ==
                    segment) {
                    return segment;
                }
            }
            fail("committed descriptor " + segment +
                 " is absent from the executor registry");
        }();
        const nlohmann::json& boundary = payload.at("boundary_record");
        if (!boundary.is_object()) {
            fail("graph_commit boundary_record must be an object");
        }
        require_exact_keys(boundary,
                           {"stage", "causal_metadata", "fixed_instance_id",
                            "fixed_action"},
                           "boundary_record");
        if (boundary.at("stage").get<std::string>() != stage.stage_label) {
            fail("boundary_record stage differs from the batch's stage");
        }
        stage.fixed_instance_id =
            boundary.at("fixed_instance_id").get<std::string>();
        stage.fixed_action = boundary.at("fixed_action").get<std::string>();
        stage.boundary_ordinal = boundary_ordinal;
        stage.start_tick = commit_tick;
        stage.logical_events_at_start = v2_publisher.next_logical_sequence() - 1;
        stage.proceed_steps_at_start = physical_steps;
        stages.push_back(stage);

        audit_records.push_back({{"kind", "graph_commit"},
                                 {"event_ordinal", event_ordinal},
                                 {"boundary_ordinal", boundary_ordinal},
                                 {"batch_id", batch_id},
                                 {"tick_ns", commit_tick},
                                 {"stage", stage.stage_label}});
        audit_records.push_back(
            {{"kind", "raw_v2_frame"},
             {"boundary_ordinal", boundary_ordinal},
             {"frame", prestate_frame},
             {"runtime_snapshot", runtime_snapshot},
             {"producer_build", {{"binary_sha256", binary_sha},
                                 {"kernel_contract", inputs.precision}}}});
        // Explicit ordinary-node identity map for this boundary's committed
        // batch (rank/json_id -> affine store id + name/type): the K1
        // logical-view comparison joins node_terminal audit records (store
        // ids) back to the wire's logical ids through exactly this map.
        {
            nlohmann::json map_rows = nlohmann::json::array();
            for (const auto& node : batch.at("nodes")) {
                const int rank = node.at("rank").get<int>();
                const uint64_t json_id = node.at("id").get<uint64_t>();
                const auto store_id = committer.resolve_store_id(rank, json_id);
                if (!store_id.has_value()) {
                    fail("committed node missing from the affine map");
                }
                nlohmann::json row =
                    {{"rank", rank},
                     {"json_id", json_id},
                     {"store_id", *store_id},
                     {"name", node.at("name")},
                     {"type", node.at("type")},
                     {"request_id", node.at("request_id")}};
                if (node.at("type").get<int>() == 4 &&
                    node.contains("compute")) {
                    row["num_ops"] = node.at("compute").at("num_ops");
                    row["runtime_ns"] =
                        node.at("compute").at("runtime_ns");
                }
                boundary_node_keys[boundary_ordinal].insert(
                    {rank, *store_id});
                map_rows.push_back(std::move(row));
            }
            audit_records.push_back(
                {{"kind", "committed_node_map"},
                 {"boundary_ordinal", boundary_ordinal},
                 {"batch_id", batch_id},
                 {"nodes", std::move(map_rows)}});
        }
        ensure_event_loop();
    };

    const auto apply_external_resource = [&](const nlohmann::json& payload,
                                             uint64_t event_ordinal) {
        require_exact_keys(payload, {"resource", "mutation", "target", "amount"},
                           "external_resource payload");
        const std::string resource = payload.at("resource").get<std::string>();
        const std::string mutation = payload.at("mutation").get<std::string>();
        const nlohmann::json& target = payload.at("target");
        const nlohmann::json& amount = payload.at("amount");
        const uint64_t tick = Sys::boostedTick();
        if (resource == "noc_link" && mutation == "join") {
            require_exact_keys(target, {"src_rank", "dst_rank"}, "noc target");
            require_exact_keys(amount, {"bytes", "tag"}, "noc amount");
            const int src = target.at("src_rank").get<int>();
            const int dst = target.at("dst_rank").get<int>();
            const uint64_t bytes = amount.at("bytes").get<uint64_t>();
            const int tag = amount.at("tag").get<int>();
            if (src < 0 || src >= inputs.rank_count || dst < 0 ||
                dst >= inputs.rank_count || src == dst) {
                fail("noc_link join target is outside the physical rank range");
            }
            if (bytes == 0) fail("noc_link join requires positive bytes");
            auto sink = std::make_unique<BackgroundSink>("noc_background");
            BackgroundSink* sink_ptr = sink.get();
            background_sinks.push_back(std::move(sink));
            // Real background flow through the ordinary send path (same
            // handler plumbing as TileTransportAdapter); the physical FlowId
            // is allocated by the original scheduler allocation.
            auto* send_data = new SendPacketEventHandlerData;
            send_data->callable = sink_ptr;
            send_data->wlhd = nullptr;
            send_data->event = EventType::PacketSent;
            sim_request request{};
            workloads[src]->sys->front_end_sim_send(
                0, Sys::dummy_data, bytes, UINT8, dst, tag, &request,
                Sys::FrontEndSendRecvType::NATIVE, &Sys::handleEvent,
                send_data);
            audit_records.push_back(
                {{"kind", "external_resource"},
                 {"event_ordinal", event_ordinal},
                 {"tick_ns", tick},
                 {"resource", resource},
                 {"mutation", mutation},
                 {"src_rank", src},
                 {"dst_rank", dst},
                 {"bytes", bytes},
                 {"tag", tag}});
            ensure_event_loop();
            return;
        }
        if (resource == "hbm_rank" && mutation == "join") {
            require_exact_keys(target, {"rank"}, "hbm target");
            require_exact_keys(amount, {"kind", "bytes"}, "hbm amount");
            const int rank = target.at("rank").get<int>();
            const std::string kind = amount.at("kind").get<std::string>();
            const uint64_t bytes = amount.at("bytes").get<uint64_t>();
            if (rank < 0 || rank >= inputs.rank_count) {
                fail("hbm_rank join target is outside the physical rank range");
            }
            if (bytes == 0) fail("hbm_rank join requires positive bytes");
            LocalHbmBandwidthModel::JobKind job_kind;
            if (kind == "COMM_READ") {
                job_kind = LocalHbmBandwidthModel::JobKind::COMM_READ;
            } else if (kind == "COMM_WRITE") {
                job_kind = LocalHbmBandwidthModel::JobKind::COMM_WRITE;
            } else {
                fail("hbm_rank join amount.kind must be COMM_READ|COMM_WRITE");
            }
            auto sink = std::make_unique<BackgroundSink>("hbm_background");
            BackgroundSink* sink_ptr = sink.get();
            background_sinks.push_back(std::move(sink));
            workloads[rank]->local_hbm_bandwidth_model->issue_transfer(
                job_kind, bytes, rank, /*node_id=*/0,
                [sink_ptr]() { sink_ptr->call(EventType::General, nullptr); });
            audit_records.push_back(
                {{"kind", "external_resource"},
                 {"event_ordinal", event_ordinal},
                 {"tick_ns", tick},
                 {"resource", resource},
                 {"mutation", mutation},
                 {"rank", rank},
                 {"job_kind", kind},
                 {"bytes", bytes}});
            return;
        }
        if (resource == "remote_port" && mutation == "join") {
            require_exact_keys(target, {"edge_rank", "port_index"},
                               "remote_port target");
            require_exact_keys(amount, {"bytes", "latency_ns"},
                               "remote_port amount");
            const int edge_rank = target.at("edge_rank").get<int>();
            const uint64_t bytes = amount.at("bytes").get<uint64_t>();
            if (edge_rank < 0 || edge_rank >= inputs.rank_count) {
                fail("remote_port join target is outside the physical rank range");
            }
            if (bytes == 0) fail("remote_port join requires positive bytes");
            auto sink = std::make_unique<BackgroundSink>("remote_port_background");
            BackgroundSink* sink_ptr = sink.get();
            background_sinks.push_back(std::move(sink));
            auto* wlhd = new WorkloadLayerHandlerData();
            wlhd->sys_id = edge_rank;
            wlhd->workload = workloads[edge_rank];
            wlhd->node_id = 0;
            wlhd->pipeline_sink = sink_ptr;
            memory_api->issue(bytes, wlhd);
            audit_records.push_back(
                {{"kind", "external_resource"},
                 {"event_ordinal", event_ordinal},
                 {"tick_ns", tick},
                 {"resource", resource},
                 {"mutation", mutation},
                 {"edge_rank", edge_rank},
                 {"port_index", target.at("port_index")},
                 {"bytes", bytes},
                 {"latency_ns", amount.at("latency_ns")}});
            return;
        }
        failure_reason = "external_resource combination resource=" + resource +
                         " mutation=" + mutation +
                         " has no corresponding real component API (fail-closed)";
    };

    // Audit-only: every UNFINISHED store node with its live blocking-parent
    // set (NodeStore work-ledger view).
    const auto collect_unfinished_ledger = [&graph_sources]() {
        nlohmann::json unfinished = nlohmann::json::array();
        for (size_t rank = 0; rank < graph_sources.size(); ++rank) {
            for (const auto& row :
                 graph_sources[rank]->store().work_ledger_view()) {
                if (row.finished) continue;
                unfinished.push_back({
                    {"rank", static_cast<int>(rank)},
                    {"store_id", row.node_id},
                    {"request_id", row.request_id},
                    {"stage", row.stage},
                    {"issued", row.issued},
                    {"unresolved_parents", row.unresolved_parents},
                    {"blocking_parents", row.blocking_parents},
                });
            }
        }
        return unfinished;
    };

    const auto handle_request_complete = [&](const nlohmann::json& payload,
                                             uint64_t event_ordinal) {
        require_exact_keys(payload, {"request_id"}, "request_complete payload");
        const std::string request_id =
            payload.at("request_id").get<std::string>();
        // Pre-close capture (entry-state only; zero behavior change): if
        // the queue already drained at request_complete entry, snapshot
        // the blocked set before the close mutates cursors. NOTE: a full
        // pre-close drain was tried and REJECTED -- without the close the
        // multi-train tapes LIVELOCK (the queue keeps producing deferred
        // work and never drains within 200s), which is itself a decisive
        // diagnostic fact recorded in STATE round_16.
        if (event_queue->finished() && !executor.is_drained()) {
            drained_blocked_snapshot =
                executor.blocked_endpoints_diagnostics();
        }
        if (!executor.is_drained()) {
            pre_close_blocked_snapshot =
                executor.blocked_endpoints_diagnostics();
            pre_close_unfinished_ledger = collect_unfinished_ledger();
            // Per-rank physical-occupancy view (audit-only): reconciles an
            // issued-but-unfinished node against the hardware/HBM/fluid
            // resources that owe its completion.
            nlohmann::json occupancy = nlohmann::json::array();
            for (size_t rank = 0; rank < workloads.size(); ++rank) {
                occupancy.push_back({
                    {"rank", static_cast<int>(rank)},
                    {"gpu_comp_in_flight",
                     workloads[rank]->hw_resource->num_in_flight_gpu_comp_ops},
                    {"gpu_comm_in_flight",
                     workloads[rank]->hw_resource->num_in_flight_gpu_comm_ops},
                    {"hbm_active_jobs",
                     workloads[rank]->local_hbm_bandwidth_model ==
                             nullptr
                         ? -1
                         : (workloads[rank]
                                    ->local_hbm_bandwidth_model
                                    ->has_active_jobs()
                                ? 1
                                : 0)},
                });
            }
            pre_close_resource_occupancy = occupancy;
            // Bounded natural-drain window (round-21 fix for the
            // deps-closure class): the request_complete tick can arrive
            // while the multi-train compute chain is still legitimately in
            // flight (its own HBM/compute events pending). Closing at that
            // instant destroys the live cursors -- never-issued legs and
            // pipeline anchors can never dispatch afterwards -- converting
            // a completing tape into a permanent FAIL. Let the tape's OWN
            // events finish the work first: no wait gate is added and the
            // emission order is unchanged. Bounded (round-16 lesson): a
            // tape whose queue never drains by itself (the M03 livelock
            // shape) falls through to the unchanged close semantics at
            // the guard. The step/terminal counters feed the certificate
            // calibration fields (see the declarations above).
            //
            // Round-22 calibrated budget floor: the ten naturally-completed
            // cases of the regenerated library needed at most 19,586
            // window steps (family ladder 345/1239/2426/5804/6344/19586;
            // terminals/steps ratio uniform at ~0.344), so 40,000 steps
            // is ~2x the largest observed legitimate demand of that
            // family. Round-23 refinement: larger tapes legitimately need
            // proportionally more (the ~35k-declared-node family implies
            // ~100k steps at the same ratio), so the budget scales with
            // the work committed so far -- 2,000 + 6 x committed ordinary
            // nodes (~1.7x the implied demand at the 0.344 ratio), with
            // the calibrated 40,000 floor for small tapes. A tape whose
            // queue still regenerates work at the budget falls through to
            // the unchanged close semantics and its FAIL certificate
            // carries the step/terminal counters (a regeneration tape
            // shows window terminals far above its declared remaining
            // work; a clipped-but-finite tape shows a coherent near-
            // complete ledger). Deterministic: the bound is a step count,
            // never host timing or memory.
            constexpr uint64_t kDrainFirstStepBudgetFloor = 40000;
            uint64_t committed_nodes = 0;
            for (const auto& audit : audit_records) {
                if (audit.value("kind", std::string()) ==
                    "committed_node_map") {
                    committed_nodes += audit.value(
                        "nodes", std::vector<nlohmann::json>{}).size();
                }
            }
            const uint64_t drain_step_budget = std::max<uint64_t>(
                kDrainFirstStepBudgetFloor, 2000 + 6 * committed_nodes);
            // Host-safety valve (round 22): regeneration tapes allocate
            // 1-2 MB per window step (measured 60-70 GB before 40k steps
            // on M02-seed001/M12-seed002), so the step budget alone is
            // NOT a memory bound. Break the window when the process peak
            // RSS passes a ceiling above any legitimate tape's observed
            // demand (the most memory-hungry finite tape measured ~1.2
            // MB/step and would peak near 35 GiB) while still containing
            // the worst regeneration tape (measured 62 GiB uncapped) on a
            // 78 GiB host under the serial-only run discipline.
            // FAIL-path-only semantics:
            // PASS tapes never approach it; on livelock tapes the FAIL
            // certificate's exact stopping point is host-dependent by
            // construction (documented), while the deterministic step
            // budget remains the reproducible fall-through bound.
            // 40 GiB expressed in KB (ru_maxrss unit on Linux).
            constexpr uint64_t kDrainFirstMemCeilingKb = 40ull * 1024 * 1024;
            uint64_t drain_guard = 0;
            bool drain_mem_capped = false;
            pre_close_drain_terminal_base = terminals.size();
            while (!event_queue->finished() && !executor.is_drained()) {
                pump_once();
                if (++drain_guard > drain_step_budget) break;
                if ((drain_guard & 0xFF) == 0) {
                    struct rusage usage;
                    if (getrusage(RUSAGE_SELF, &usage) == 0 &&
                        static_cast<uint64_t>(usage.ru_maxrss) >
                            kDrainFirstMemCeilingKb) {
                        drain_mem_capped = true;
                        break;
                    }
                }
            }
            pre_close_drain_steps = drain_guard;
            pre_close_drain_terminals =
                terminals.size() - pre_close_drain_terminal_base;
            pre_close_drain_mem_capped = drain_mem_capped;
        }
        // Administrative close BEFORE the natural drain (main-agent ruling
        // 2026-10-06): the staging lease retires with the request --
        // unconsumed slots and never-issued tail endpoints retire with it,
        // consume watchers stop gating settle, while every issued physical
        // leg keeps draining naturally below.
        std::set<std::string> closed_requests{request_id};
        for (const auto& entry : boundary_node_keys) {
            (void)entry;
        }
        for (const auto& audit : audit_records) {
            if (audit.value("kind", std::string()) == "committed_node_map") {
                for (const auto& node : audit.at("nodes")) {
                    const std::string node_request =
                        node.value("request_id", std::string());
                    if (!node_request.empty()) {
                        closed_requests.insert(node_request);
                    }
                }
            }
        }
        for (const std::string& closed : closed_requests) {
            executor.notify_request_closed(closed);
        }
        const auto naturally_complete = [&]() {
            return executor.is_drained() &&
                   transport.active_count() == 0 &&
                   fluid_scheduler->get_active_flow_count() == 0 &&
                   memory_api->is_drained() && event_queue->finished();
        };
        // Post-close natural drain, memory-bounded (round 22): on the
        // stall-class tapes this finite churn allocates 1-2 MB per event
        // with no terminal progress (measured 76 GiB peak on M02-seed001,
        // 38 GiB on M13-seed002) before the queue finally drains and the
        // run lands on its FAIL certificate. Cap the churn by peak RSS so
        // a bare fixture invocation stays contained; healthy tapes
        // complete far below the ceiling, and the deterministic step
        // guard below is unchanged. The capped run records
        // post_close_drain_mem_capped in its certificate.
        constexpr uint64_t kPostCloseMemCeilingKb = 40ull * 1024 * 1024;
        {
            uint64_t churn_guard = 0;
            while (!naturally_complete()) {
                if (event_queue->finished()) {
                    if (!event_queue->has_deferred_work()) {
                        // Provably-no-future-events moment: no flow can
                        // still bind and no packet can arrive, so any
                        // active leg with a never-bound pending network
                        // part is stranded by construction (round 23:
                        // the close retires never-issued send endpoints
                        // whose already-submitted partner receives then
                        // have no completion source). Retire them through
                        // their ordinary completions, then re-evaluate:
                        // the retirements may schedule legitimate
                        // follow-up work (children dispatch, settles).
                        transport.retire_stranded_transfers();
                        if (transport.active_count() == 0) {
                            // Round-24 terminal-gate retirement (main-
                            // agent ruling, rounds 5+9 synthesis): with
                            // every transport leg gone, consume
                            // obligations of CLOSED leases that are
                            // still pending can no longer receive any
                            // delivery -- dead bookkeeping that retires
                            // with the lease, audited per descriptor.
                            // The executor re-checks its guard; the
                            // queue-finished/no-deferred facts above are
                            // the caller side of the contract.
                            executor.retire_dead_consume_obligations();
                            // Never-firing stage watches of closed leases
                            // unpin here too (round 5): their member
                            // nodes belong to requests that already
                            // ended, so removal at the no-future-events
                            // moment is the same retirement, one step
                            // earlier than the post-completion site
                            // below (which stays as the idempotent
                            // backstop).
                            for (const std::string& closed :
                                 closed_requests) {
                                watch_registry.remove_watches_for_request(
                                    closed);
                            }
                            executor.notify_watch_registry_changed();
                        }
                        if (naturally_complete() ||
                            (event_queue->finished() &&
                             !event_queue->has_deferred_work())) {
                            break;
                        }
                    }
                    // Defect-C boundary (same as pump_until): proceed()
                    // requires a non-empty main map; force the next
                    // decision boundary so the deferred queue (fluid
                    // flush callbacks among it) actually drains. Round-23
                    // fix: this loop originally called pump_once()
                    // directly on the finished-with-deferred state,
                    // violating proceed()'s precondition (the Release
                    // build compiles the assert out -- UB on the empty
                    // map's end iterator).
                    event_queue->schedule_event(
                        event_queue->get_current_time() + 1,
                        [](void*) {}, nullptr);
                }
                if ((++churn_guard & 0x3FF) == 0) {
                    struct rusage usage;
                    if (getrusage(RUSAGE_SELF, &usage) == 0 &&
                        static_cast<uint64_t>(usage.ru_maxrss) >
                            kPostCloseMemCeilingKb) {
                        post_close_drain_mem_capped = true;
                        break;
                    }
                }
                pump_once();
                if (churn_guard > 2000000) {
                    fail("request_complete natural drain exceeded event guard");
                }
            }
        }
        if (!naturally_complete()) {
            // Final-state capture BEFORE the administrative close mutates
            // the cursors: this is the adjudication snapshot (wire cycle
            // vs mapping gap) for the drain-time blocked endpoint set.
            drained_blocked_snapshot =
                executor.blocked_endpoints_diagnostics();
            failure_reason =
                "request_complete: request " + request_id +
                " did not reach natural physical completion before the queue drained";
            return;
        }
        for (Workload* workload : workloads) {
            if (workload->hw_resource->num_in_flight_gpu_comm_ops != 0 ||
                workload->hw_resource->num_in_flight_gpu_comp_ops != 0 ||
                workload->local_hbm_bandwidth_model->has_active_jobs()) {
                failure_reason =
                    "request_complete: hardware/HBM occupancy did not drain";
                return;
            }
        }
        for (const std::string& closed : closed_requests) {
            watch_registry.remove_watches_for_request(closed);
        }
        executor.notify_watch_registry_changed();
        audit_records.push_back({{"kind", "request_complete"},
                                 {"event_ordinal", event_ordinal},
                                 {"request_id", request_id},
                                 {"tick_ns", Sys::boostedTick()}});
    };

    // Initial certificate check AFTER wiring but BEFORE any commit: the
    // publisher's initial full frame (delivery 0) is already out, and no
    // physical mutation has happened.
    const nlohmann::json exported_certificate = export_initial_certificate();
    const bool initial_matches =
        exported_certificate == inputs.state.at("initial_state");
    if (!initial_matches && failure_reason.empty()) {
        nlohmann::json diff;
        for (auto it = exported_certificate.begin();
             it != exported_certificate.end(); ++it) {
            if (inputs.state.at("initial_state").contains(it.key()) &&
                inputs.state.at("initial_state").at(it.key()) != it.value()) {
                diff[it.key()] = "certificate/input mismatch";
            }
        }
        failure_reason =
            "initial_certificate differs from the cold input state (fields: " +
            diff.dump() + ")";
    }

    // ---- drive the tape -----------------------------------------------------
    for (const auto& event : inputs.tape.at("events")) {
        if (!failure_reason.empty()) break;
        const uint64_t ordinal = event.at("ordinal").get<uint64_t>();
        const nlohmann::json& trigger = event.at("trigger");
        const std::string trigger_kind = trigger.at("kind").get<std::string>();
        if (trigger_kind == "tick") {
            const uint64_t tick_ns = trigger.at("tick_ns").get<uint64_t>();
            advance_to_tick(tick_ns);
        } else {  // gate_terminal
            const std::string segment_id =
                trigger.at("segment_id").get<std::string>();
            const std::string role = trigger.at("role").get<std::string>();
            if (role != "service" && role != "settle") {
                failure_reason = "gate_terminal role must be service|settle";
                break;
            }
            const auto gate_done = [&]() {
                const nlohmann::json snapshot =
                    executor.authoritative_snapshot(Sys::boostedTick());
                return descriptor_done(snapshot, segment_id,
                                       role == "service" ? "service_done"
                                                         : "settle_done");
            };
            pump_until(gate_done,
                       "gate_terminal " + segment_id + "/" + role);
            if (!gate_done()) {
                failure_reason =
                    "gate_terminal referenced " + segment_id + "/" + role +
                    " which never really completed";
                break;
            }
        }
        const std::string kind = event.at("kind").get<std::string>();
        const nlohmann::json& payload = event.at("payload");
        if (kind == "graph_commit") {
            commit_graph_batch(payload, ordinal);
        } else if (kind == "external_resource") {
            apply_external_resource(payload, ordinal);
        } else if (kind == "request_complete") {
            handle_request_complete(payload, ordinal);
        } else if (kind == "tick" || kind == "gate_terminal") {
            failure_reason =
                "event kind '" + kind +
                "' is a trigger dimension and cannot appear as event.kind";
        } else {
            failure_reason = "unknown event kind '" + kind + "' (fail-closed)";
        }
    }

    // ---- settle any outstanding stages, then seal ---------------------------
    pump_until([&]() { return event_queue->finished(); }, "final drain");
    sample_transfers();
    record_stage_transitions();
    if (failure_reason.empty()) {
        for (const auto& stage : stages) {
            if (!stage.settle_recorded) {
                failure_reason = "stage " + stage.stage_label +
                                 " (segment " + stage.segment_id +
                                 ") never reached settle";
                break;
            }
        }
    }
    // Seal the terminal frame (delivery N): state_version bookkeeping and the
    // final v2_frame_published audit anchor.
    publish_v2_frame();

    // ---- measurements -------------------------------------------------------
    nlohmann::json measurements = nlohmann::json::array();
    nlohmann::json boundary_records = nlohmann::json::array();
    const nlohmann::json& state_causal = inputs.state.at("causal_input");
    const auto state_context = [&]() -> uint64_t {
        for (const char* key : {"context_tokens", "ctx"}) {
            if (state_causal.contains(key)) {
                return state_causal.at(key).get<uint64_t>();
            }
        }
        fail("state causal_input carries no context");
    }();
    const uint64_t state_horizon = [&]() -> uint64_t {
        for (const char* key : {"horizon_tokens", "horizon"}) {
            if (state_causal.contains(key)) {
                return state_causal.at(key).get<uint64_t>();
            }
        }
        fail("state causal_input carries no horizon");
    }();
    for (size_t index = 0; index < stages.size(); ++index) {
        const StageRun& stage = stages[index];
        const uint64_t window_end =
            index + 1 < stages.size() ? stages[index + 1].start_tick
                                      : Sys::boostedTick();
        uint64_t data_flows = 0, data_bytes = 0, ack_flows = 0, ack_bytes = 0,
                 control_markers = 0, total_transfers = 0;
        for (const auto& entry : transfer_observations) {
            const auto& observation = entry.second;
            const std::string& leg = observation.transfer.logical_key.leg;
            const bool in_window = observation.issue_tick >= stage.start_tick &&
                                   observation.issue_tick < window_end;
            if (!in_window) continue;
            ++total_transfers;
            if (observation.transfer.control_marker) {
                ++control_markers;
            } else if (leg == "data_send" || leg == "data_recv") {
                ++data_flows;
                data_bytes += observation.transfer.total_bytes;
            } else {
                ++ack_flows;
                ack_bytes += observation.transfer.total_bytes;
            }
        }
        const uint64_t settle_bound =
            stage.settle_recorded ? stage.settle_tick : Sys::boostedTick();
        measurements.push_back(
            {{"stage", stage.stage_label},
             {"fixed_instance_id", stage.fixed_instance_id},
             {"fixed_action", stage.fixed_action},
             {"boundary_ordinal", stage.boundary_ordinal},
             {"start_tick_ns", stage.start_tick},
             {"service_tick_ns",
              stage.service_recorded ? stage.service_tick : settle_bound},
             {"settle_tick_ns", settle_bound},
             {"logical_event_count",
              stage.logical_events_at_settle - stage.logical_events_at_start},
             {"native_physical_steps",
              stage.proceed_steps_at_settle - stage.proceed_steps_at_start},
             {"window_stats",
              {{"window_tile_limit",
                inputs.precision.at("window_tile_limit")},
               {"eligible_template_windows", 0},
               {"averaged_windows", 0},
               {"refinements", 0},
               {"mode", "exact"}}},
             {"charged_totals",
              {{"total_transfers", total_transfers},
               {"data_flows", data_flows},
               {"data_bytes", data_bytes},
               {"ack_flows", ack_flows},
               {"ack_bytes", ack_bytes},
               {"control_markers", control_markers}}}});
        boundary_records.push_back(
            {{"request_id", inputs.request_id},
             {"stage", stage.stage_label},
             {"boundary_ordinal", stage.boundary_ordinal},
             {"tick_ns", stage.start_tick},
             {"fixed_instance_id", stage.fixed_instance_id},
             {"fixed_action", stage.fixed_action},
             {"causal_prestate", nullptr},
             {"causal_metadata",
              {{"context_tokens", state_context},
               {"horizon_tokens", state_horizon}}},
             {"planner_state", nullptr}});
    }

    const bool completed = failure_reason.empty() && !stages.empty() &&
                           event_queue->finished() && executor.is_drained() &&
                           transport.active_count() == 0 &&
                           fluid_scheduler->get_active_flow_count() == 0 &&
                           memory_api->is_drained();
    if (failure_reason.empty() && (!completed || stages.empty())) {
        failure_reason = "fixture did not reach natural completion";
    }
    const bool passed = failure_reason.empty() && completed;

    nlohmann::json terminal_certificate = nlohmann::json::object();
    {
        const uint64_t final_tick = Sys::boostedTick();
        terminal_certificate = {
            {"tick_ns", final_tick},
            {"executor_drained", executor.is_drained()},
            {"transport_active_count", transport.active_count()},
            {"fluid_active_flow_count",
             fluid_scheduler->get_active_flow_count()},
            {"remote_memory_drained", memory_api->is_drained()},
            {"event_queue_finished", event_queue->finished()},
            {"watch_registry_empty", watch_registry.empty()},
            {"active_resources",
             collect_work_ledger_v2_active_resources(
                 graph_sources, systems, memory_api.get(), fluid_scheduler,
                 &executor, &transport, final_tick)},
            {"obligations",
             executor.authoritative_snapshot(final_tick).at("obligations")},
            {"final_delivery_sequence", next_delivery_sequence - 1},
            {"final_watermark", v2_publisher.watermark()},
            {"final_state_version", v2_publisher.state_version()},
            {"blocked_endpoints_diagnostics",
             drained_blocked_snapshot.is_null()
                 ? executor.blocked_endpoints_diagnostics()
                 : drained_blocked_snapshot},
            {"external_resource_completions",
             [&]() {
                 nlohmann::json rows = nlohmann::json::array();
                 for (const auto& sink : background_sinks) {
                     rows.push_back({{"label", sink->label_},
                                     {"completions", sink->completions}});
                 }
                 return rows;
             }()},
            {"bound_flow_count",
             [&]() {
                 uint64_t flows = 0;
                 for (const auto& record : audit_records) {
                     if (record.is_object() &&
                         record.value("kind", std::string()) == "flow_bound") {
                         ++flows;
                     }
                 }
                 return flows;
             }()}};
        // Emitted only when a pre-close view exists (FAIL paths): keeps
        // PASS-result bytes stable while retaining the live pre-close
        // cursor positions for adjudication.
        if (!pre_close_blocked_snapshot.is_null()) {
            terminal_certificate["pre_close_blocked_diagnostics"] =
                pre_close_blocked_snapshot;
        }
        // Round-22 drain-first calibration fields: emitted only when the
        // window actually ran (executor not yet drained at
        // request_complete entry). Tapes that were already drained keep
        // byte-identical certificates. The values are deterministic
        // (step count / terminal count), so double-run determinism holds.
        if (pre_close_drain_steps !=
            std::numeric_limits<uint64_t>::max()) {
            terminal_certificate["drain_first_steps"] = pre_close_drain_steps;
            terminal_certificate["drain_first_terminals"] =
                pre_close_drain_terminals;
            if (pre_close_drain_mem_capped) {
                terminal_certificate["drain_first_mem_capped"] = true;
            }
        }
        // Round-24 terminal-gate consume-obligation retirement audit:
        // emitted only when the executor actually retired dead consume
        // obligations (per-descriptor counts -- never silent zeroing).
        // Tapes whose leases paid every consume obligation keep
        // byte-identical certificates.
        {
            const nlohmann::json retired_consume =
                executor.administratively_retired_consume_summary();
            if (!retired_consume.is_null()) {
                terminal_certificate[
                    "administratively_retired_consume_obligations"] =
                    retired_consume;
            }
        }
        // FAIL-path adjudication aid: every UNFINISHED store node with its
        // live blocking-parent set (NodeStore work-ledger view). A node that
        // is wire-parentless yet blocked here had parents installed at
        // commit; an unissued node with zero blocking parents is a dispatch
        // gap; finished-but-uncollected rows are excluded.
        if (!failure_reason.empty()) {
            terminal_certificate["unfinished_nodes_ledger"] =
                collect_unfinished_ledger();
            if (!pre_close_unfinished_ledger.is_null()) {
                terminal_certificate["pre_close_unfinished_nodes_ledger"] =
                    pre_close_unfinished_ledger;
            }
            if (!pre_close_resource_occupancy.is_null()) {
                terminal_certificate["pre_close_resource_occupancy"] =
                    pre_close_resource_occupancy;
            }
            // Host memory peak at failure (getrusage, Linux reports KB):
            // diagnostic for the drain-window heap hazard; deliberately
            // FAIL-only so PASS-result determinism never depends on host
            // allocator behavior.
            struct rusage usage;
            if (getrusage(RUSAGE_SELF, &usage) == 0) {
                terminal_certificate["failure_mem_peak_rss_kb"] =
                    static_cast<uint64_t>(usage.ru_maxrss);
            }
            if (post_close_drain_mem_capped) {
                terminal_certificate["post_close_drain_mem_capped"] = true;
            }
        }
    }

    nlohmann::json result = {
        {"version", "execution-result-v1-20261006"},
        {"case_id", inputs.case_id},
        {"cluster_id", inputs.state.at("cluster_id")},
        {"scenario_id", inputs.state.at("scenario_id")},
        {"seed", inputs.seed},
        {"request_id", inputs.request_id},
        {"fixed_instance_id", inputs.fixed_instance_id},
        {"fixed_action", inputs.fixed_action},
        {"kernel_contract", inputs.precision},
        {"status", passed ? "PASS" : "FAIL"},
        {"reason", failure_reason.empty() ? nlohmann::json(nullptr)
                                          : nlohmann::json(failure_reason)},
        {"completed", passed},
        {"canonicalization_pending", true},
        {"initial_certificate", exported_certificate},
        {"terminal_certificate", terminal_certificate},
        {"measurements", measurements},
        {"boundary_records", boundary_records},
        {"audit_records", audit_records}};
    if (capture_micro_delivery_frames) {
        if (delivery_frames.size() != 5) {
            fail("micro fixture expected all five actual delivery envelopes");
        }
        result["delivery_frames"] = std::move(delivery_frames);
    }
    write_json_file(out_path, result);

    CompletionObserver::instance().set_hook(nullptr, nullptr);
    CongestionAwareNetworkApi::set_flow_bound_callback({});
    if (!passed) {
        std::fprintf(stderr, "[pipeline_execution_fixture] status=FAIL reason=%s\n",
                     failure_reason.c_str());
        return 1;
    }
    std::printf("[pipeline_execution_fixture] PASS case=%s boundaries=%llu "
                "tick=%llu deliveries=%llu steps=%llu\n",
                inputs.case_id.c_str(),
                static_cast<unsigned long long>(boundary_counter),
                static_cast<unsigned long long>(Sys::boostedTick()),
                static_cast<unsigned long long>(next_delivery_sequence),
                static_cast<unsigned long long>(physical_steps));
    return 0;
}

}  // namespace

#include <execinfo.h>
#include <signal.h>

static void crash_handler(int signal_number) {
    void* frames[64];
    const int count = backtrace(frames, 64);
    std::fprintf(stderr, "SIGSEGV/SIGABRT backtrace (%d frames):\n", count);
    backtrace_symbols_fd(frames, count, STDERR_FILENO);
    _exit(128 + signal_number);
}

int main(int argc, char** argv) {
    ::signal(SIGSEGV, crash_handler);
    ::signal(SIGABRT, crash_handler);
    try {
        return protected_main(argc, argv);
    } catch (const std::exception& error) {
        fail(std::string("unhandled exception: ") + error.what());
    } catch (...) {
        fail("unhandled non-standard exception");
    }
}
