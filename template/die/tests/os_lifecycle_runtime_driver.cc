#include "coarse_die/Execution.hh"
#include "../src/GemmOSGeometry.hh"
#include "../src/GemmOSResourceFacts.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef COARSE_DIE_SOURCE_DIR
#error "COARSE_DIE_SOURCE_DIR must point at die/"
#endif

namespace geometry = coarse_die::detail::gemm_os_geometry;
namespace access = coarse_die::detail::gemm_os_access_counter;
using coarse_die::Activity;
using coarse_die::Hardware;
using coarse_die::Owner;
using coarse_die::Plan;
using coarse_die::Role;
using coarse_die::Runtime;
using u64 = std::uint64_t;
using u128 = unsigned __int128;
using json = nlohmann::json;

namespace {

constexpr std::size_t kLoad = 0;
constexpr std::size_t kCompute = 1;
constexpr std::size_t kStore = 2;
constexpr std::size_t kStageCount = 3;
constexpr std::size_t kInputSlots = 2;
constexpr std::size_t kOutputSlots = 2;
constexpr double kHbmRowCapacityBps = 50000000.0;
constexpr double kNonbindingCapacity = 1.0e30;
constexpr double kVectorEfficiency = 0.02;
constexpr u64 kDefaultStageSubmissionCap = 256;

const std::array<const char*, kStageCount> kStageNames = {
    "load", "compute", "store"};

struct Options {
  geometry::Input geometry;
  u64 stage_submission_cap = kDefaultStageSubmissionCap;
  bool diagnostic_cap = false;
};

std::string u128_string(u128 value) {
  if (value == 0) return "0";
  std::string result;
  while (value != 0) {
    result.push_back(static_cast<char>('0' + value % 10));
    value /= 10;
  }
  std::reverse(result.begin(), result.end());
  return result;
}

std::string long_double_hex(long double value) {
  std::ostringstream out;
  out << std::hexfloat << value;
  return out.str();
}

u64 parse_u64(const std::string& value, const char* name) {
  if (value.empty() || value.front() == '-')
    throw std::invalid_argument(std::string(name) + " must be a nonnegative integer");
  std::size_t consumed = 0;
  const unsigned long long parsed = std::stoull(value, &consumed, 10);
  if (consumed != value.size() ||
      parsed > std::numeric_limits<u64>::max()) {
    throw std::invalid_argument(std::string(name) + " is outside uint64 range");
  }
  return static_cast<u64>(parsed);
}

double parse_double(const std::string& value, const char* name) {
  std::size_t consumed = 0;
  const double parsed = std::stod(value, &consumed);
  if (consumed != value.size() || !std::isfinite(parsed) || parsed < 0.0)
    throw std::invalid_argument(std::string(name) + " must be finite and nonnegative");
  return parsed;
}

access::StorageDType parse_dtype(const std::string& value, const char* name) {
  if (value == "fp32") return access::StorageDType::Fp32;
  if (value == "bf16") return access::StorageDType::Bf16;
  if (value == "int4") return access::StorageDType::Int4;
  if (value == "int32") return access::StorageDType::Int32;
  throw std::invalid_argument(std::string(name) +
                              " must be fp32, bf16, int4, or int32");
}

const char* dtype_name(access::StorageDType dtype) {
  switch (dtype) {
    case access::StorageDType::Fp32: return "fp32";
    case access::StorageDType::Bf16: return "bf16";
    case access::StorageDType::Int4: return "int4";
    case access::StorageDType::Int32: return "int32";
  }
  return "unknown";
}

json resource_names_json() {
  static constexpr const char* names[coarse_die::resource_count] = {
      "COMPUTE_GATE", "SRAM_READ", "SRAM_WRITE", "LOCAL", "HBM0",
      "HBM1", "INJECT0", "INJECT1", "XPOS", "XNEG", "YPOS", "YNEG",
      "HOP"};
  json result = json::array();
  for (const char* name : names) result.push_back(name);
  return result;
}

Options parse_options(int argc, char** argv) {
  Options result;
  result.geometry.M = 17;
  result.geometry.N = 65;
  result.geometry.K = 129;
  result.geometry.tm = 16;
  result.geometry.tn = 64;
  result.geometry.tk = 64;
  result.geometry.core_count = 1;
  result.geometry.a_storage = access::StorageDType::Fp32;
  result.geometry.w_storage = access::StorageDType::Fp32;
  result.geometry.output_storage = access::StorageDType::Fp32;
  result.geometry.matrix_path = access::MatrixPath::StrictFp32Vec;
  result.geometry.kappa_quant = 4.0;
  result.geometry.kappa_acc = 2.0;

  const auto usage = [] {
    std::cout
        << "usage: os_lifecycle_runtime_driver [options]\n"
        << "  --m N --n N --k N --tm N --tn N --tk N\n"
        << "  --path strict|int4 --a-dtype fp32|bf16|int4|int32\n"
        << "  --w-dtype fp32|bf16|int4|int32 --output-dtype fp32|bf16|int4|int32\n"
        << "  --kappa-quant X --kappa-acc X --stage-submission-cap N\n"
        << "  --cap N (alias; default 256 stage submissions)\n"
        << "  --diagnostic-cap N (alias for a larger small-shape diagnostic cap)\n";
  };

  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--help" || option == "-h") {
      usage();
      std::exit(0);
    }
    if (i + 1 >= argc) throw std::invalid_argument(option + " requires a value");
    const std::string value = argv[++i];
    if (option == "--m") result.geometry.M = parse_u64(value, "M");
    else if (option == "--n") result.geometry.N = parse_u64(value, "N");
    else if (option == "--k") result.geometry.K = parse_u64(value, "K");
    else if (option == "--tm") result.geometry.tm = parse_u64(value, "tm");
    else if (option == "--tn") result.geometry.tn = parse_u64(value, "tn");
    else if (option == "--tk") result.geometry.tk = parse_u64(value, "tk");
    else if (option == "--cap" || option == "--stage-submission-cap")
      result.stage_submission_cap = parse_u64(value, "stage-submission-cap");
    else if (option == "--diagnostic-cap") {
      result.stage_submission_cap = parse_u64(value, "diagnostic-cap");
      result.diagnostic_cap = true;
    } else if (option == "--path") {
      if (value == "strict")
        result.geometry.matrix_path = access::MatrixPath::StrictFp32Vec;
      else if (value == "int4")
        result.geometry.matrix_path = access::MatrixPath::Int4PE;
      else
        throw std::invalid_argument("--path must be strict or int4");
    } else if (option == "--a-dtype") {
      result.geometry.a_storage = parse_dtype(value, "--a-dtype");
    } else if (option == "--w-dtype") {
      result.geometry.w_storage = parse_dtype(value, "--w-dtype");
    } else if (option == "--output-dtype") {
      result.geometry.output_storage = parse_dtype(value, "--output-dtype");
    } else if (option == "--kappa-quant") {
      result.geometry.kappa_quant = parse_double(value, "--kappa-quant");
    } else if (option == "--kappa-acc") {
      result.geometry.kappa_acc = parse_double(value, "--kappa-acc");
    } else {
      throw std::invalid_argument("unknown option: " + option);
    }
  }
  if (result.stage_submission_cap == 0 && result.diagnostic_cap)
    throw std::invalid_argument("--diagnostic-cap must be positive");
  return result;
}

json read_json(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open " + path);
  return json::parse(input);
}

Hardware synthetic_hardware() {
  json input = read_json(std::string(COARSE_DIE_SOURCE_DIR) +
                         "/configs/example_hardware.json");
  input["architecture"]["sram_mib_per_core"] = 0.25;
  input["architecture"]["noc_width_bits"] = 64;
  input["architecture"]["core_columns_nx"] = 1;
  input["architecture"]["core_rows_ny"] = 1;
  input["architecture"]["die_geometry"] = "isotropic";
  input["performance_profile"]["eta_vec"] = kVectorEfficiency;
  input["performance_profile"]["eta_hbm_matrix"] = 1.0;
  input["performance_profile"]["granularity_validation"] = "not_run";

  Hardware hardware = coarse_die::resolve_hardware(input);
  const std::string source_fingerprint = hardware.fingerprint;
  for (std::size_t resource = 0; resource < coarse_die::resource_count;
       ++resource) {
    auto& descriptor = hardware.derived.at("resources").at(resource);
    const bool enabled = descriptor.at("enabled").get<bool>();
    double capacity = 0.0;
    if (enabled) {
      if (resource == coarse_die::HBM0 || resource == coarse_die::HBM1)
        capacity = kHbmRowCapacityBps;
      else if (resource == coarse_die::COMPUTE_GATE)
        capacity = 1.0;
      else
        capacity = kNonbindingCapacity;
    }
    hardware.rates[resource] = capacity;
    descriptor["rate"] = capacity;
  }
  const json identity = {
      {"kind", "synthetic_model_to_model_single_core_os_lifecycle_v1"},
      {"single_core", true},
      {"hbm_row_capacity_Bps", kHbmRowCapacityBps},
      {"compute_gate_capacity", 1.0},
      {"nonbinding_enabled_resource_capacity", kNonbindingCapacity},
      {"eta_vec", kVectorEfficiency},
      {"eta_hbm_matrix", 1.0}};
  hardware.fingerprint = "synthetic-os-lifecycle:" +
                         coarse_die::stable_hash(source_fingerprint +
                                                 identity.dump());
  return hardware;
}

const char* role_name(Role role) {
  switch (role) {
    case Role::Load: return kStageNames[kLoad];
    case Role::Compute: return kStageNames[kCompute];
    case Role::Store: return kStageNames[kStore];
    case Role::Endpoint: return "endpoint";
  }
  return "unknown";
}

json activity_json(const Activity& activity) {
  json work = json::array();
  for (double value : activity.work) work.push_back(value);
  json hbm = json::array();
  for (const auto& row : activity.hbm_bytes)
    hbm.push_back({row[0], row[1]});
  const json exact_fraction = activity.vec_eqops_fraction == 0.0L
                                  ? json(nullptr)
                                  : coarse_die::equivalent_fraction_to_json(
                                        activity.vec_eqops_fraction);
  const json typed_ops = {
      {"int4_ops", activity.int4_ops},
      {"vec_eqops_whole", activity.vec_eqops},
      {"vec_eqops_fraction", exact_fraction},
      {"vec_eqops_fraction_binary64",
       static_cast<double>(activity.vec_eqops_fraction)}};
  return {{"id", activity.id},
          {"role", role_name(activity.role)},
          {"resource_work_13", work},
          {"work_by_resource", std::move(work)},
          {"resource_names", resource_names_json()},
          {"hbm_row_payload_bytes", hbm},
          {"hbm_bytes_by_row_direction", std::move(hbm)},
          {"int4_ops", activity.int4_ops},
          {"vec_eqops", activity.vec_eqops},
          {"typed_ops", typed_ops},
          {"vec_eqops_fraction_hex", long_double_hex(activity.vec_eqops_fraction)},
          {"canonical_q", activity.canonical_q},
          {"private_time_s", activity.private_time_s},
          {"credit_delta", activity.credit_delta}};
}

json ledger_json(const geometry::Ledger& value) {
  return {{"hbm_read_bytes", u128_string(value.hbm_read_bytes)},
          {"hbm_write_bytes", u128_string(value.hbm_write_bytes)},
          {"load_sram_write_bytes", u128_string(value.load_sram_write_bytes)},
          {"compute_sram_read_bytes", u128_string(value.compute_sram_read_bytes)},
          {"compute_sram_write_bytes", u128_string(value.compute_sram_write_bytes)},
          {"store_sram_read_bytes", u128_string(value.store_sram_read_bytes)},
          {"sram_read_bytes", u128_string(value.sram_read_bytes)},
          {"sram_write_bytes", u128_string(value.sram_write_bytes)},
          {"local_bytes", u128_string(value.local_bytes)},
          {"matrix_int4_ops", u128_string(value.matrix_int4_ops)},
          {"vec_equivalent_ops_whole", u128_string(value.vec_equivalent_ops.whole())},
          {"vec_equivalent_ops_fraction_hex",
           long_double_hex(value.vec_equivalent_ops.fraction())},
          {"conversion_elements", u128_string(value.conversion_elements)},
          {"source_math_ops", u128_string(value.source_math_ops)}};
}

std::shared_ptr<const Plan> make_stage_plan(const Hardware& hardware,
                                            const geometry::Input& input,
                                            const geometry::AccessClass& cls,
                                            std::size_t class_index,
                                            std::size_t stage,
                                            const Activity& activity) {
  auto plan = std::make_shared<Plan>();
  plan->task = {{"schema", "os_lifecycle_runtime_stage_v1"},
                {"op_kind", "gemm_os_lifecycle_stage"},
                {"m", input.M}, {"n", input.N}, {"k", input.K},
                {"tm", input.tm}, {"tn", input.tn}, {"tk", input.tk},
                {"class_index", class_index},
                {"class_m", cls.m}, {"class_n", cls.n}, {"class_k", cls.k},
                {"first_k", cls.first_k}, {"final_k", cls.final_k},
                {"stage", kStageNames[stage]}};
  plan->summary = {{"kind", "os_lifecycle_runtime_stage"},
                   {"hbm_workspace_peak_bytes", 0},
                   {"geometry_class_index", class_index}};
  plan->hardware_fingerprint = hardware.fingerprint;
  coarse_die::Epoch epoch;
  epoch.name = std::string("os_") + kStageNames[stage];
  epoch.activities.push_back(activity);
  plan->epochs.push_back(std::move(epoch));
  plan->fingerprint = coarse_die::plan_to_json(*plan)
                          .at("fingerprint").get<std::string>();
  return plan;
}

struct ClassCache {
  std::array<Activity, kStageCount> facts{};
  std::array<std::shared_ptr<const Plan>, kStageCount> plans{};
  std::array<double, coarse_die::resource_count> hardware_capacities{};
};

std::array<ClassCache, geometry::kMaxAccessClasses> make_class_cache(
    const Hardware& hardware, const geometry::GemmOSGeometry& geom) {
  std::array<ClassCache, geometry::kMaxAccessClasses> cache{};
  for (std::size_t i = 0; i < geom.class_count(); ++i) {
    const auto& cls = geom.access_class(i);
    cache[i].hardware_capacities = hardware.rates;
    access::Input input;
    input.m = cls.m;
    input.n = cls.n;
    input.k = cls.k;
    input.first_k = cls.first_k;
    input.final_k = cls.final_k;
    input.a_storage = geom.input().a_storage;
    input.w_storage = geom.input().w_storage;
    input.output_storage = geom.input().output_storage;
    input.matrix_path = geom.input().matrix_path;
    input.dataflow = access::Dataflow::OutputStationary;
    input.kappa_quant = geom.input().kappa_quant;
    input.kappa_acc = geom.input().kappa_acc;
    cache[i].facts = coarse_die::detail::gemm_os_single_core_resource_facts(
        hardware, input);
    for (std::size_t stage = 0; stage < kStageCount; ++stage) {
      cache[i].plans[stage] = make_stage_plan(
          hardware, geom.input(), cls, i, stage, cache[i].facts[stage]);
    }
  }
  return cache;
}

enum class InputState : std::uint8_t { Free, Loading, Ready, Computing };
enum class OutputState : std::uint8_t { Free, Reserved, Ready, Storing };

const char* input_state_name(InputState state) {
  switch (state) {
    case InputState::Free: return "free";
    case InputState::Loading: return "loading";
    case InputState::Ready: return "ready";
    case InputState::Computing: return "computing";
  }
  return "unknown";
}

const char* output_state_name(OutputState state) {
  switch (state) {
    case OutputState::Free: return "free";
    case OutputState::Reserved: return "reserved";
    case OutputState::Ready: return "ready";
    case OutputState::Storing: return "storing";
  }
  return "unknown";
}

struct InputSlot {
  InputState state = InputState::Free;
  u128 ordinal = 0;
  std::size_t class_index = 0;
  u64 handle = 0;
};

struct OutputSlot {
  OutputState state = OutputState::Free;
  u128 q = 0;
  u128 final_ordinal = 0;
  std::size_t class_index = 0;
  u64 handle = 0;
};

class LifecycleDriver {
 public:
  LifecycleDriver(const Hardware& hardware,
                  const geometry::GemmOSGeometry& geom,
                  std::array<ClassCache, geometry::kMaxAccessClasses> cache,
                  u64 stage_submission_cap, bool diagnostic_cap)
      : hardware_(hardware), geom_(geom), cache_(std::move(cache)),
        stage_submission_cap_(stage_submission_cap),
        diagnostic_cap_(diagnostic_cap), runtime_(hardware) {
    total_consumptions_ = geom_.consumption_wave_count();
    if (geom_.input().core_count != 1)
      throw std::invalid_argument("offline readiness lifecycle requires one core");
    owner_.request_id = 1;
    owner_.node_id = 901;
    owner_.execution_id = 901;
    owner_.generation = 1;
  }

  json run() {
    pump();
    for (;;) {
      collect_completions();
      pump();
      if (fully_complete()) {
        status_ = "complete";
        break;
      }
      const auto next = runtime_.next_event();
      if (!next) {
        if (submitted_stages_ >= stage_submission_cap_) {
          status_ = "stage_submission_cap_reached";
          break;
        }
        if (has_live_or_pending_work())
          throw std::runtime_error("no public Runtime event can advance a live lifecycle");
        status_ = "complete";
        break;
      }
      if (*next <= runtime_.now_ns())
        throw std::runtime_error("public Runtime next_event did not advance integer ns time");
      runtime_.advance_to(*next);
      ++runtime_advances_;
      record_runtime_activity_peak();
    }
    collect_completions();
    update_queue_peaks();
    return report();
  }

 private:
  struct EventRef {
    u128 ordinal = 0;
    u128 q = 0;
    u64 k_tile = 0;
    std::size_t class_index = 0;
  };

  EventRef event_ref(u128 ordinal) const {
    const u64 k_tiles = geom_.k_tiles();
    const u128 q = ordinal / k_tiles;
    const u64 k_tile = static_cast<u64>(ordinal % k_tiles);
    const auto fact = geom_.wave_fact(q, k_tile);
    return {ordinal, q, k_tile, fact.class_index};
  }

  std::size_t input_occupancy() const {
    return static_cast<std::size_t>(std::count_if(
        input_slots_.begin(), input_slots_.end(),
        [](const InputSlot& slot) { return slot.state != InputState::Free; }));
  }

  std::size_t output_occupancy() const {
    return static_cast<std::size_t>(std::count_if(
        output_slots_.begin(), output_slots_.end(),
        [](const OutputSlot& slot) { return slot.state != OutputState::Free; }));
  }

  void update_queue_peaks() {
    max_input_occupancy_ = std::max(max_input_occupancy_, input_occupancy());
    max_output_occupancy_ = std::max(max_output_occupancy_, output_occupancy());
    if (input_occupancy() > kInputSlots || output_occupancy() > kOutputSlots)
      throw std::logic_error("finite readiness queue exceeded two-slot capacity");
  }

  int first_free_input_slot() const {
    for (std::size_t i = 0; i < input_slots_.size(); ++i)
      if (input_slots_[i].state == InputState::Free) return static_cast<int>(i);
    return -1;
  }

  int ready_input_for_next_compute() const {
    for (std::size_t i = 0; i < input_slots_.size(); ++i) {
      if (input_slots_[i].state == InputState::Ready &&
          input_slots_[i].ordinal == next_compute_ordinal_) {
        return static_cast<int>(i);
      }
    }
    return -1;
  }

  int first_free_output_slot() const {
    for (std::size_t i = 0; i < output_slots_.size(); ++i)
      if (output_slots_[i].state == OutputState::Free) return static_cast<int>(i);
    return -1;
  }

  int first_ready_output_slot() const {
    int selected = -1;
    for (std::size_t i = 0; i < output_slots_.size(); ++i) {
      if (output_slots_[i].state == OutputState::Ready &&
          (selected < 0 || output_slots_[i].q <
                               output_slots_[static_cast<std::size_t>(selected)].q)) {
        selected = static_cast<int>(i);
      }
    }
    return selected;
  }

  bool can_submit_stage() const noexcept {
    return submitted_stages_ < stage_submission_cap_;
  }

  void record_runtime_activity_peak() {
    const json counters = runtime_.counters();
    const std::size_t active = counters.at("active_activities").get<std::size_t>();
    max_runtime_active_activities_ =
        std::max(max_runtime_active_activities_, active);
  }

  void append_stage_event(const char* phase, const char* stage,
                          const EventRef& ref, std::size_t slot,
                          u64 handle, const json& extra = json::object()) {
    json item = {{"phase", phase}, {"stage", stage},
                 {"time_ns", runtime_.now_ns()},
                 {"ordinal", u128_string(ref.ordinal)},
                 {"output_q", u128_string(ref.q)}, {"k_tile", ref.k_tile},
                 {"class_index", ref.class_index}, {"slot", slot},
                 {"handle", handle}, {"extra", extra}};
    events_.push_back(std::move(item));
  }

  u64 submit_stage(std::size_t class_index, std::size_t stage) {
    if (stage >= kStageCount || class_index >= geom_.class_count())
      throw std::logic_error("invalid cached stage selection");
    return runtime_.submit(cache_[class_index].plans[stage], owner_,
                           runtime_.now_ns());
  }

  bool start_one_store() {
    if (!can_submit_stage()) return false;
    if (std::any_of(output_slots_.begin(), output_slots_.end(),
                    [](const OutputSlot& slot) {
                      return slot.state == OutputState::Storing;
                    })) return false;
    const int index = first_ready_output_slot();
    if (index < 0) return false;
    auto& slot = output_slots_[static_cast<std::size_t>(index)];
    const EventRef ref{slot.final_ordinal, slot.q,
                       static_cast<u64>(slot.final_ordinal % geom_.k_tiles()),
                       slot.class_index};
    const u64 handle = submit_stage(slot.class_index, kStore);
    slot.state = OutputState::Storing;
    slot.handle = handle;
    ++submitted_stages_;
    ++submitted_by_stage_[kStore];
    record_runtime_activity_peak();
    append_stage_event("start", kStageNames[kStore], ref,
                       static_cast<std::size_t>(index), handle,
                       {{"queue_occupancy", output_occupancy()}});
    update_queue_peaks();
    return true;
  }

  bool start_compute() {
    if (!can_submit_stage() || compute_active_) return false;
    const int input_index = ready_input_for_next_compute();
    if (input_index < 0) return false;
    auto& input_slot = input_slots_[static_cast<std::size_t>(input_index)];
    const EventRef ref = event_ref(input_slot.ordinal);
    const bool final_k = ref.k_tile + 1 == geom_.k_tiles();
    int output_index = -1;
    if (final_k) {
      output_index = first_free_output_slot();
      if (output_index < 0) return false;
      auto& output = output_slots_[static_cast<std::size_t>(output_index)];
      output.state = OutputState::Reserved;  // reserve at final C start
      output.q = ref.q;
      output.final_ordinal = ref.ordinal;
      output.class_index = ref.class_index;
      output.handle = 0;
      ++outputs_reserved_;
    }

    const u64 handle = submit_stage(ref.class_index, kCompute);
    input_slot.state = InputState::Computing;
    input_slot.handle = handle;
    compute_active_ = true;
    compute_input_slot_ = input_index;
    compute_output_slot_ = output_index;
    compute_ordinal_ = ref.ordinal;
    ++submitted_stages_;
    ++submitted_by_stage_[kCompute];
    record_runtime_activity_peak();
    append_stage_event("start", kStageNames[kCompute], ref,
                       static_cast<std::size_t>(input_index), handle,
                       {{"reserved_output_slot", output_index},
                        {"input_queue_occupancy", input_occupancy()},
                        {"output_queue_occupancy", output_occupancy()}});
    update_queue_peaks();
    return true;
  }

  bool start_one_load() {
    if (!can_submit_stage() || next_load_ordinal_ >= total_consumptions_)
      return false;
    if (std::any_of(input_slots_.begin(), input_slots_.end(),
                    [](const InputSlot& slot) {
                      return slot.state == InputState::Loading;
                    })) return false;
    const int index = first_free_input_slot();
    if (index < 0) return false;
    const EventRef ref = event_ref(next_load_ordinal_);
    const u64 handle = submit_stage(ref.class_index, kLoad);
    auto& slot = input_slots_[static_cast<std::size_t>(index)];
    slot.state = InputState::Loading;
    slot.ordinal = ref.ordinal;
    slot.class_index = ref.class_index;
    slot.handle = handle;
    ++next_load_ordinal_;
    ++submitted_stages_;
    ++submitted_by_stage_[kLoad];
    record_runtime_activity_peak();
    append_stage_event("start", kStageNames[kLoad], ref,
                       static_cast<std::size_t>(index), handle,
                       {{"queue_occupancy", input_occupancy()}});
    update_queue_peaks();
    return true;
  }

  void pump() {
    bool changed = true;
    while (changed) {
      changed = false;
      if (start_one_store()) {
        changed = true;
        continue;
      }
      if (start_compute()) {
        changed = true;
        continue;
      }
      if (start_one_load()) {
        changed = true;
        continue;
      }
    }
  }

  json completion_extra(u64 handle, u64 finish_ns) const {
    const json progress = runtime_.query_progress(handle);
    return {{"finish_ns", finish_ns},
            {"observed_ns", runtime_.now_ns()},
            {"terminal_state", progress.at("state")},
            {"elapsed_ns", progress.at("elapsed_ns")},
            {"served", progress.at("served")},
            {"activities", progress.at("activities")}};
  }

  void finish_load_slot(std::size_t index) {
    auto& slot = input_slots_[index];
    const auto finish = runtime_.terminal_finish_ns(slot.handle);
    if (!finish) return;
    const EventRef ref = event_ref(slot.ordinal);
    const u64 handle = slot.handle;
    append_stage_event("terminal", kStageNames[kLoad], ref, index, handle,
                       completion_extra(handle, *finish));
    runtime_.destroy_execution(handle);
    slot.state = InputState::Ready;
    slot.handle = 0;
    ++terminal_by_stage_[kLoad];
    ++destroyed_executions_;
  }

  void finish_compute() {
    if (!compute_active_) return;
    const auto finish = runtime_.terminal_finish_ns(
        input_slots_[static_cast<std::size_t>(compute_input_slot_)].handle);
    if (!finish) return;
    auto& input_slot = input_slots_[static_cast<std::size_t>(compute_input_slot_)];
    const EventRef ref = event_ref(compute_ordinal_);
    const u64 handle = input_slot.handle;
    append_stage_event("terminal", kStageNames[kCompute], ref,
                       static_cast<std::size_t>(compute_input_slot_), handle,
                       completion_extra(handle, *finish));
    runtime_.destroy_execution(handle);
    input_slot.state = InputState::Free;
    input_slot.handle = 0;
    input_slot.ordinal = 0;
    ++terminal_by_stage_[kCompute];
    ++destroyed_executions_;
    ++next_compute_ordinal_;

    if (compute_output_slot_ >= 0) {
      auto& output = output_slots_[static_cast<std::size_t>(compute_output_slot_)];
      if (output.state != OutputState::Reserved || output.q != ref.q)
        throw std::logic_error("final Compute lost its reserved output slot");
      output.state = OutputState::Ready;
      ++outputs_produced_;
    }
    compute_active_ = false;
    compute_input_slot_ = -1;
    compute_output_slot_ = -1;
    compute_ordinal_ = 0;
  }

  void finish_store_slot(std::size_t index) {
    auto& slot = output_slots_[index];
    const auto finish = runtime_.terminal_finish_ns(slot.handle);
    if (!finish) return;
    const u64 handle = slot.handle;
    const EventRef ref{slot.final_ordinal, slot.q,
                       static_cast<u64>(slot.final_ordinal % geom_.k_tiles()),
                       slot.class_index};
    append_stage_event("terminal", kStageNames[kStore], ref, index, handle,
                       completion_extra(handle, *finish));
    runtime_.destroy_execution(handle);
    slot = OutputSlot{};
    ++terminal_by_stage_[kStore];
    ++destroyed_executions_;
    ++outputs_retired_;
  }

  void collect_completions() {
    for (std::size_t i = 0; i < input_slots_.size(); ++i) {
      if (input_slots_[i].state == InputState::Loading)
        finish_load_slot(i);
    }
    finish_compute();
    for (std::size_t i = 0; i < output_slots_.size(); ++i) {
      if (output_slots_[i].state == OutputState::Storing)
        finish_store_slot(i);
    }
    update_queue_peaks();
  }

  bool all_inputs_free() const {
    return std::all_of(input_slots_.begin(), input_slots_.end(),
                       [](const InputSlot& slot) {
                         return slot.state == InputState::Free;
                       });
  }

  bool all_outputs_free() const {
    return std::all_of(output_slots_.begin(), output_slots_.end(),
                       [](const OutputSlot& slot) {
                         return slot.state == OutputState::Free;
                       });
  }

  bool fully_complete() const {
    return next_load_ordinal_ == total_consumptions_ &&
           next_compute_ordinal_ == total_consumptions_ &&
           !compute_active_ && all_inputs_free() && all_outputs_free() &&
           runtime_.execution_handles().empty();
  }

  bool has_live_or_pending_work() const {
    return next_load_ordinal_ < total_consumptions_ ||
           next_compute_ordinal_ < total_consumptions_ || compute_active_ ||
           !all_inputs_free() || !all_outputs_free() ||
           !runtime_.execution_handles().empty();
  }

  json slot_query(u64 handle) const {
    if (!handle) return nullptr;
    return runtime_.query_progress(handle);
  }

  json input_slots_json() const {
    json result = json::array();
    for (std::size_t i = 0; i < input_slots_.size(); ++i) {
      const auto& slot = input_slots_[i];
      json item = {{"slot", i}, {"state", input_state_name(slot.state)},
                   {"handle", slot.handle}};
      if (slot.state != InputState::Free) {
        const EventRef ref = event_ref(slot.ordinal);
        item["ordinal"] = u128_string(ref.ordinal);
        item["output_q"] = u128_string(ref.q);
        item["k_tile"] = ref.k_tile;
        item["class_index"] = ref.class_index;
      }
      if (slot.handle) item["runtime_progress"] = slot_query(slot.handle);
      result.push_back(std::move(item));
    }
    return result;
  }

  json output_slots_json() const {
    json result = json::array();
    for (std::size_t i = 0; i < output_slots_.size(); ++i) {
      const auto& slot = output_slots_[i];
      json item = {{"slot", i}, {"state", output_state_name(slot.state)},
                   {"handle", slot.handle}};
      if (slot.state != OutputState::Free) {
        item["output_q"] = u128_string(slot.q);
        item["final_ordinal"] = u128_string(slot.final_ordinal);
        item["class_index"] = slot.class_index;
      }
      if (slot.handle) item["runtime_progress"] = slot_query(slot.handle);
      result.push_back(std::move(item));
    }
    return result;
  }

  json class_facts_json() const {
    json result = json::array();
    for (std::size_t i = 0; i < geom_.class_count(); ++i) {
      const auto& cls = geom_.access_class(i);
      const json load_fact = activity_json(cache_[i].facts[kLoad]);
      const json compute_fact = activity_json(cache_[i].facts[kCompute]);
      const json store_fact = activity_json(cache_[i].facts[kStore]);
      json capacities = json::array();
      for (double value : cache_[i].hardware_capacities)
        capacities.push_back(value);
      result.push_back({{"index", i},
                        {"m_tile_begin", cls.m_tile_begin},
                        {"m_tile_count", cls.m_tile_count},
                        {"n_tile_begin", cls.n_tile_begin},
                        {"n_tile_count", cls.n_tile_count},
                        {"k_tile_begin", cls.k_tile_begin},
                        {"k_tile_count", cls.k_tile_count},
                        {"actual_m", cls.m}, {"actual_n", cls.n},
                        {"actual_k", cls.k},
                        {"m_tail", cls.m_tail}, {"n_tail", cls.n_tail},
                        {"k_tail", cls.k_tail},
                        {"first_k", cls.first_k}, {"final_k", cls.final_k},
                        {"repeat_count", u128_string(cls.repeat_count)},
                        {"single_consumption_counter",
                         ledger_json(geometry::detail::scale(cls.per_consumption, 1))},
                        {"resource_names", resource_names_json()},
                        {"hardware_capacities_13", capacities},
                        {"load_fact", load_fact},
                        {"compute_fact", compute_fact},
                        {"store_fact", store_fact},
                        {"stage_plan_fingerprints",
                         {cache_[i].plans[kLoad]->fingerprint,
                          cache_[i].plans[kCompute]->fingerprint,
                          cache_[i].plans[kStore]->fingerprint}}});
    }
    return result;
  }

  json report() const {
    const auto& input = geom_.input();
    json active_handles = json::array();
    for (u64 handle : runtime_.execution_handles()) {
      active_handles.push_back(runtime_.query_progress(handle));
    }
    const u128 total_outputs = geom_.output_tile_count();
    const u128 stores_planned = total_outputs;
    const u128 nonfinal_store_suppressions =
        total_consumptions_ - total_outputs;
    const json stage_submission_cap = {
        {"limit", stage_submission_cap_},
        {"unit", "submitted L/C/S stage instances"},
        {"diagnostic_small_shape_cap", diagnostic_cap_},
        {"per_output_reset", false},
        {"production_event_cap_certificate", false},
        {"production_cycle_skipping", false},
        {"semantics", "offline fixture submission budget only; not a production descriptor or event-discovery bound"}};
    const json geometry_report = {
        {"M", input.M}, {"N", input.N}, {"K", input.K},
        {"tm", input.tm}, {"tn", input.tn}, {"tk", input.tk},
        {"core_count", input.core_count},
        {"path", input.matrix_path == access::MatrixPath::StrictFp32Vec
                     ? "strict_fp32_vec" : "int4_pe"},
        {"a_storage", dtype_name(input.a_storage)},
        {"w_storage", dtype_name(input.w_storage)},
        {"output_storage", dtype_name(input.output_storage)},
        {"kappa_quant", input.kappa_quant}, {"kappa_acc", input.kappa_acc},
        {"m_tiles", geom_.m_tiles()}, {"n_tiles", geom_.n_tiles()},
        {"k_tiles", geom_.k_tiles()},
        {"output_tile_count", u128_string(geom_.output_tile_count())},
        {"output_round_count", u128_string(geom_.output_round_count())},
        {"consumption_wave_count", u128_string(geom_.consumption_wave_count())},
        {"total_k_consumptions", u128_string(total_consumptions_)},
        {"access_class_count", geom_.class_count()},
        {"access_class_limit", geometry::kMaxAccessClasses},
        {"cached_resource_fact_count", geom_.class_count()},
        {"classes", class_facts_json()}};
    const json live_state = {
        {"status", status_},
        {"now_ns", runtime_.now_ns()},
        {"next_load_ordinal", u128_string(next_load_ordinal_)},
        {"next_compute_ordinal", u128_string(next_compute_ordinal_)},
        {"compute_active", compute_active_},
        {"input_slots", input_slots_json()},
        {"output_slots", output_slots_json()},
        {"public_runtime_live_executions", active_handles.size()},
        {"live_runtime_progress", std::move(active_handles)},
        {"runtime_drained", runtime_.drained()}};
    const json lifetime_counters = {
        {"submitted_stages", submitted_stages_},
        {"runtime_advances", runtime_advances_},
        {"submitted_by_stage", {{"load", submitted_by_stage_[kLoad]},
                                 {"compute", submitted_by_stage_[kCompute]},
                                 {"store", submitted_by_stage_[kStore]}}},
        {"terminal_by_stage", {{"load", terminal_by_stage_[kLoad]},
                                {"compute", terminal_by_stage_[kCompute]},
                                {"store", terminal_by_stage_[kStore]}}},
        {"destroyed_executions", destroyed_executions_},
        {"outputs_reserved_at_final_compute_start", u128_string(outputs_reserved_)},
        {"outputs_produced_at_final_compute_end", u128_string(outputs_produced_)},
        {"outputs_retired_after_store", u128_string(outputs_retired_)},
        {"store_stages_planned", u128_string(stores_planned)},
        {"nonfinal_k_store_stages_suppressed", u128_string(nonfinal_store_suppressions)},
        {"max_input_queue_occupancy", max_input_occupancy_},
        {"max_output_queue_occupancy", max_output_occupancy_},
        {"max_runtime_active_activities", max_runtime_active_activities_},
        {"input_queue_capacity", kInputSlots},
        {"output_queue_capacity", kOutputSlots}};
    return {{"schema_version", 1},
            {"measurement_source", "single_core_gemm_os_geometry_facts_and_public_cpp_runtime"},
            {"fixture_identity", "synthetic_model_to_model_not_hardware_measurement"},
            {"production_bounded_descriptor_claim", false},
            {"integer_clock_note", "driver advances only through public Runtime next_event()/advance_to() integer ns; sub-ns exact clock remains private; no phase epsilon or cycle skipping"},
            {"lifecycle_contract", "independent one-activity L/C/S Plans, same node/execution/generation owner; at most one active Load, Compute, and Store stage; L terminal before C; final C reserves output at start; nonfinal C uses independently modeled accumulation state; Store starts after final C terminal; Store terminal retires output; terminal executions are destroyed"},
            {"runtime_owner", {{"request_id", owner_.request_id},
                               {"node_id", owner_.node_id},
                               {"execution_id", owner_.execution_id},
                               {"generation", owner_.generation}}},
            {"stage_submission_cap", stage_submission_cap},
            {"synthetic_hardware", {{"hardware_fingerprint", hardware_.fingerprint},
                                    {"core_count", hardware_.derived.at("die").at("core_count")},
                                    {"resource_names", resource_names_json()},
                                    {"hardware_capacities_13", runtime_.counters().at("capacity")},
                                    {"eta_vec", hardware_.input.at("performance_profile").at("eta_vec")},
                                    {"eta_hbm_matrix", hardware_.input.at("performance_profile").at("eta_hbm_matrix")},
                                    {"hbm_row_capacity_Bps", kHbmRowCapacityBps},
                                    {"compute_gate_capacity", 1.0},
                                    {"other_enabled_resource_capacity", kNonbindingCapacity}}},
            {"geometry", geometry_report},
            {"source_counters_full_single_gemm", ledger_json(geom_.total_ledger())},
            {"live_state", live_state},
            {"lifetime_counters", lifetime_counters},
            {"runtime_internal_events", runtime_.counters().at("internal_events")},
            {"runtime_counters", runtime_.counters()},
            {"events", events_}};
  }

  const Hardware& hardware_;
  const geometry::GemmOSGeometry& geom_;
  std::array<ClassCache, geometry::kMaxAccessClasses> cache_;
  u64 stage_submission_cap_;
  bool diagnostic_cap_;
  Runtime runtime_;
  Owner owner_;
  std::array<InputSlot, kInputSlots> input_slots_{};
  std::array<OutputSlot, kOutputSlots> output_slots_{};
  u128 total_consumptions_ = 0;
  u128 next_load_ordinal_ = 0;
  u128 next_compute_ordinal_ = 0;
  bool compute_active_ = false;
  int compute_input_slot_ = -1;
  int compute_output_slot_ = -1;
  u128 compute_ordinal_ = 0;
  u64 submitted_stages_ = 0;
  u64 runtime_advances_ = 0;
  std::array<u64, kStageCount> submitted_by_stage_{};
  std::array<u64, kStageCount> terminal_by_stage_{};
  u64 destroyed_executions_ = 0;
  u128 outputs_reserved_ = 0;
  u128 outputs_produced_ = 0;
  u128 outputs_retired_ = 0;
  std::size_t max_input_occupancy_ = 0;
  std::size_t max_output_occupancy_ = 0;
  std::size_t max_runtime_active_activities_ = 0;
  std::string status_ = "running";
  json events_ = json::array();
};

json resource_facts_input(const geometry::Input& input) {
  return {{"M", input.M}, {"N", input.N}, {"K", input.K},
          {"tm", input.tm}, {"tn", input.tn}, {"tk", input.tk},
          {"core_count", input.core_count},
          {"a_storage", dtype_name(input.a_storage)},
          {"w_storage", dtype_name(input.w_storage)},
          {"output_storage", dtype_name(input.output_storage)},
          {"matrix_path", input.matrix_path == access::MatrixPath::StrictFp32Vec
                               ? "strict_fp32_vec" : "int4_pe"},
          {"kappa_quant", input.kappa_quant}, {"kappa_acc", input.kappa_acc}};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const Hardware hardware = synthetic_hardware();
    geometry::GemmOSGeometry geom(options.geometry);
    if (geom.class_count() > geometry::kMaxAccessClasses)
      throw std::runtime_error("geometry exceeds the fixed access-class cache");
    const auto cache = make_class_cache(hardware, geom);
    LifecycleDriver driver(hardware, geom, cache, options.stage_submission_cap,
                           options.diagnostic_cap);
    json output = driver.run();
    output["resource_facts_input"] = resource_facts_input(options.geometry);
    std::cout << output.dump(2) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "os_lifecycle_runtime_driver: " << error.what() << '\n';
    return 1;
  }
}
