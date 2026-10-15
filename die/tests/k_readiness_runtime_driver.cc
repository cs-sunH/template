// Diagnostic-only finite-slot K-readiness measurement. The included compiler
// driver supplies the same synthetic hardware, selected mapping, input shape,
// and JSON helpers as the archived probe. Its compiled whole-GEMM Plan is used
// only as the mapping/ledger source; the Runtime below receives per-K stage
// facts from GemmOSResourceFacts.
#define main compiled_gemm_oracle_driver_entry
#include "compiled_gemm_oracle_driver.cc"
#undef main

#include "../src/GemmOSReadiness.hh"

#include <algorithm>
#include <array>
#include <deque>
#include <map>
#include <optional>
#include <tuple>
#include <vector>

namespace k_readiness {
namespace geometry = coarse_die::detail::gemm_os_geometry;
namespace access = coarse_die::detail::gemm_os_access_counter;
namespace readiness = coarse_die::detail::gemm_os_readiness;
using u64 = std::uint64_t;
using u128 = unsigned __int128;
using Key = std::tuple<u64, u64, u64, bool, bool>;
using StagePlans = std::array<std::shared_ptr<const Plan>, 3>;

constexpr std::size_t kLoad = 0;
constexpr std::size_t kCompute = 1;
constexpr std::size_t kStore = 2;
constexpr std::size_t kInputSlots = 2;
constexpr std::size_t kOutputSlots = 2;
constexpr u64 kDiagnosticEventLimit = 1000000;

enum class InputState : std::uint8_t { Free, Loading, Ready, Computing };
enum class OutputState : std::uint8_t { Free, Reserved, Ready, Storing };

struct Token {
  u64 ordinal = 0;
  u64 q = 0;
  u64 k_tile = 0;
  bool first = false;
  bool final = false;
  std::size_t class_index = 0;
  Key key{};
};

struct InputSlot {
  InputState state = InputState::Free;
  Token token;
  u64 handle = 0;
};

struct OutputSlot {
  OutputState state = OutputState::Free;
  Token token;
  u64 handle = 0;
};

struct Request {
  u64 arrival = 0;
  u64 id = 0;
  u64 next_load = 0;
  u64 next_compute = 0;
  u64 outputs_stored = 0;
  u64 finish_ns = 0;
  bool started = false;
  bool done = false;
  std::array<InputSlot, kInputSlots> inputs{};
  std::array<OutputSlot, kOutputSlots> outputs{};
  std::optional<std::size_t> loading;
  std::optional<std::size_t> computing;
  std::optional<std::size_t> storing;
  std::optional<std::size_t> compute_output;
  std::optional<u64> last_final_wait_ordinal;
  std::optional<readiness::Cursor> cursor;
};

u64 checked_u64(u128 value, const char* field) {
  if (value > static_cast<u128>(std::numeric_limits<u64>::max()))
    throw std::overflow_error(std::string("diagnostic overflow: ") + field);
  return static_cast<u64>(value);
}

u64 count_as_u64(u128 value, const char* field) {
  return checked_u64(value, field);
}

geometry::Input geometry_input(const Hardware& hardware, const Plan& mapping) {
  if (mapping.summary.at("dataflow_id") != "output_stationary_v2")
    throw std::runtime_error(
        "K-readiness diagnostic requires the selected output-stationary mapping");
  const auto& shape = mapping.task.at("tensor_shapes");
  geometry::Input input;
  input.M = shape.at("m").get<u64>();
  input.N = shape.at("n").get<u64>();
  input.K = shape.at("k").get<u64>();
  input.tm = mapping.summary.at("tm").get<u64>();
  input.tn = mapping.summary.at("tn").get<u64>();
  input.tk = mapping.summary.at("tk").get<u64>();
  input.core_count = 1;
  input.a_storage = access::StorageDType::Fp32;
  input.w_storage = access::StorageDType::Fp32;
  input.output_storage = access::StorageDType::Fp32;
  input.matrix_path = access::MatrixPath::StrictFp32Vec;
  input.kappa_quant = hardware.input.at("performance_profile")
                          .value("kappa_quant_eqops", 4.0);
  input.kappa_acc = hardware.input.at("performance_profile")
                        .value("kappa_acc_convert_eqops", 2.0);
  return input;
}

std::shared_ptr<const Plan> make_stage_plan(const Hardware& hardware,
                                            const Token& token,
                                            const Activity& activity,
                                            const char* label) {
  auto plan = std::make_shared<Plan>();
  plan->task = {{"schema", "k_readiness_diagnostic_stage_v1"},
                {"stage", label}, {"q", token.q},
                {"k_tile", token.k_tile}, {"final_k", token.final}};
  plan->summary = {{"kind", "k_readiness_diagnostic_stage"},
                   {"hbm_workspace_peak_bytes", 0}};
  plan->hardware_fingerprint = hardware.fingerprint;
  coarse_die::Epoch epoch;
  epoch.name = std::string("diagnostic_") + label;
  epoch.activities.push_back(activity);
  plan->epochs.push_back(std::move(epoch));
  plan->fingerprint = coarse_die::plan_to_json(*plan).at("fingerprint");
  return plan;
}

class Kernel {
 public:
  Kernel(const Hardware& hardware, geometry::Input input, bool manual_fixture)
      : hardware_(hardware), input_(input), descriptor_(hardware, input),
        geometry_(descriptor_.geometry()),
        manual_fixture_(manual_fixture) {
    if (geometry_.class_count() > geometry::kMaxAccessClasses)
      throw std::runtime_error("GEMM OS geometry exceeds the fixed access-class limit");
    output_count_ = count_as_u64(geometry_.output_tile_count(),
                                 "output tile count");
    wave_count_ = count_as_u64(geometry_.consumption_wave_count(),
                               "consumption wave count");
  }

  Token token(u64 ordinal) const {
    if (ordinal >= wave_count_)
      throw std::out_of_range("diagnostic K-consumption ordinal");
    const u64 kt = geometry_.k_tiles();
    const u128 q128 = static_cast<u128>(ordinal) / kt;
    const u64 q = checked_u64(q128, "output tile ordinal");
    const u64 k_tile = ordinal % kt;
    const auto fact = geometry_.wave_fact(q128, k_tile);
    const Key key{fact.output.m, fact.output.n, fact.k,
                  fact.first_k, fact.final_k};
    return {ordinal, q, k_tile, fact.first_k, fact.final_k,
            fact.class_index, key};
  }

  const StagePlans& plans(const Token& token) {
    const auto found = plans_.find(token.key);
    if (found != plans_.end()) return found->second;

    StagePlans result;
    if (manual_fixture_) {
      result = fixture_plans(token);
    } else {
      const auto& facts = descriptor_.facts(token.ordinal);
      static constexpr const char* names[] = {"load", "compute", "store"};
      for (std::size_t stage = 0; stage < result.size(); ++stage) {
        if (facts[stage].credit_delta != std::array<double, 2>{0.0, 0.0})
          throw std::runtime_error(
              "per-K resource fact unexpectedly carries fluid buffer credit");
        result[stage] = make_stage_plan(hardware_, token, facts[stage],
                                        names[stage]);
      }
    }
    return plans_.emplace(token.key, std::move(result)).first->second;
  }

  const geometry::GemmOSGeometry& geometry() const { return geometry_; }
  const readiness::Descriptor& descriptor() const { return descriptor_; }
  u64 output_count() const { return output_count_; }
  u64 wave_count() const { return wave_count_; }
  std::size_t fact_class_count() const { return plans_.size(); }
  const geometry::Input& input() const { return input_; }
  bool manual_fixture() const { return manual_fixture_; }

 private:
  StagePlans fixture_plans(const Token& token) const {
    // Small reachable counterexample: three output tiles, two K consumptions
    // each, and a deliberately slow store. These are work-fixture values, not
    // architecture facts or a timing claim.
    static constexpr const char* names[] = {"load", "compute", "store"};
    static constexpr Role roles[] = {Role::Load, Role::Compute, Role::Store};
    StagePlans result;
    for (std::size_t stage = 0; stage < result.size(); ++stage) {
      Activity activity;
      activity.role = roles[stage];
      if (stage == kLoad) {
        activity.work[coarse_die::HBM1] = 1.0;
        activity.hbm_bytes[1][0] = 1;
        activity.canonical_q = 1.0;
      } else if (stage == kCompute) {
        activity.work[coarse_die::COMPUTE_GATE] = 1.0;
        activity.canonical_q = 1.0;
      } else {
        activity.work[coarse_die::HBM0] = 100.0;
        activity.hbm_bytes[0][1] = 100;
        activity.canonical_q = 100.0;
      }
      result[stage] = make_stage_plan(hardware_, token, activity, names[stage]);
    }
    return result;
  }

  const Hardware& hardware_;
  geometry::Input input_;
  readiness::Descriptor descriptor_;
  const geometry::GemmOSGeometry& geometry_;
  bool manual_fixture_ = false;
  u64 output_count_ = 0;
  u64 wave_count_ = 0;
  std::map<Key, StagePlans> plans_;
};

class Driver {
 public:
  Driver(const Hardware& hardware, Kernel& kernel,
         std::vector<u64> arrivals, bool trace_enabled = false)
      : kernel_(kernel), runtime_(hardware), trace_enabled_(trace_enabled) {
    requests_.reserve(arrivals.size());
    for (std::size_t i = 0; i < arrivals.size(); ++i) {
      Request request;
      request.arrival = arrivals[i];
      request.id = static_cast<u64>(i + 1);
      request.cursor.emplace(kernel_.descriptor());
      requests_.push_back(std::move(request));
    }
  }

  json run() {
    for (;;) {
      collect_terminals();
      for (auto& request : requests_) {
        if (!request.started && request.arrival <= runtime_.now_ns())
          request.started = true;
      }
      pump();
      if (all_done()) break;

      auto next = runtime_.next_event();
      for (const auto& request : requests_) {
        if (!request.started && (!next || request.arrival < *next))
          next = request.arrival;
      }
      if (!next || *next <= runtime_.now_ns())
        throw std::runtime_error("readiness diagnostic deadlocked without an advancing event");
      runtime_.advance_to(*next);
      ++runtime_advances_;
      if (runtime_advances_ > kDiagnosticEventLimit)
        throw std::runtime_error("diagnostic exceeded its fixed O(events) observation limit");
    }
    collect_terminals();
    verify_finished_invariants();
    return report();
  }

 private:
  std::size_t input_occupancy(const Request& request) const {
    return static_cast<std::size_t>(std::count_if(
        request.inputs.begin(), request.inputs.end(), [](const InputSlot& slot) {
          return slot.state != InputState::Free;
        }));
  }

  std::size_t output_occupancy(const Request& request) const {
    return static_cast<std::size_t>(std::count_if(
        request.outputs.begin(), request.outputs.end(), [](const OutputSlot& slot) {
          return slot.state != OutputState::Free;
        }));
  }

  void verify_capacity(const Request& request) {
    const std::size_t in = input_occupancy(request);
    const std::size_t out = output_occupancy(request);
    max_input_occupancy_ = std::max(max_input_occupancy_, in);
    max_output_occupancy_ = std::max(max_output_occupancy_, out);
    if (in > kInputSlots || out > kOutputSlots)
      throw std::logic_error("two-slot readiness capacity invariant failed");
    if (request.cursor->input_occupancy() != in ||
        request.cursor->output_occupancy() != out ||
        request.cursor->state().load_started != request.next_load ||
        request.cursor->state().compute_completed != request.next_compute ||
        request.cursor->state().store_completed != request.outputs_stored)
      throw std::logic_error("compact cursor differs from explicit slot state");
    auto restored = *request.cursor;
    restored.restore(request.cursor->state());
    ++capacity_invariant_checks_;
  }

  int first_free_input(const Request& request) const {
    for (std::size_t i = 0; i < request.inputs.size(); ++i) {
      if (request.inputs[i].state == InputState::Free)
        return static_cast<int>(i);
    }
    return -1;
  }

  int first_ready_input(const Request& request) const {
    for (std::size_t i = 0; i < request.inputs.size(); ++i) {
      const auto& slot = request.inputs[i];
      if (slot.state == InputState::Ready &&
          slot.token.ordinal == request.next_compute)
        return static_cast<int>(i);
    }
    return -1;
  }

  int first_free_output(const Request& request) const {
    for (std::size_t i = 0; i < request.outputs.size(); ++i) {
      if (request.outputs[i].state == OutputState::Free)
        return static_cast<int>(i);
    }
    return -1;
  }

  int first_ready_output(const Request& request) const {
    int selected = -1;
    for (std::size_t i = 0; i < request.outputs.size(); ++i) {
      const auto& slot = request.outputs[i];
      if (slot.state == OutputState::Ready &&
          (selected < 0 || slot.token.q < request.outputs[
              static_cast<std::size_t>(selected)].token.q))
        selected = static_cast<int>(i);
    }
    return selected;
  }

  void record_trace(const char* event, const Request& request,
                    const Token& token, std::size_t input_index = kInputSlots,
                    std::size_t output_index = kOutputSlots) {
    if (!trace_enabled_) return;
    trace_.push_back({{"event", event}, {"time_ns", runtime_.now_ns()},
                      {"request_id", request.id}, {"ordinal", token.ordinal},
                      {"output_q", token.q}, {"k_tile", token.k_tile},
                      {"input_slot", input_index}, {"output_slot", output_index},
                      {"input_occupancy", input_occupancy(request)},
                      {"output_occupancy", output_occupancy(request)}});
  }

  bool start_store(Request& request) {
    if (request.storing) return false;
    const int slot_index = first_ready_output(request);
    if (slot_index < 0) return false;
    auto& slot = request.outputs[static_cast<std::size_t>(slot_index)];
    const Token token = slot.token;
    if (!token.final)
      throw std::logic_error("Store admission received a nonfinal K token");
    const u64 handle = runtime_.submit(
        kernel_.plans(token)[kStore], owner(request.id), runtime_.now_ns());
    slot.state = OutputState::Storing;
    slot.handle = handle;
    request.storing = static_cast<std::size_t>(slot_index);
    if (request.cursor->admit(Role::Store) != std::optional<u64>(token.ordinal))
      throw std::logic_error("compact Store admission differs from explicit slots");
    ++submitted_by_stage_[kStore];
    verify_capacity(request);
    record_trace("store_start", request, token, kInputSlots,
                 static_cast<std::size_t>(slot_index));
    return true;
  }

  bool start_compute(Request& request) {
    if (request.computing) return false;
    const int input_index = first_ready_input(request);
    if (input_index < 0) return false;
    auto& input_slot = request.inputs[static_cast<std::size_t>(input_index)];
    const Token token = input_slot.token;
    if (token.ordinal != request.next_compute)
      throw std::logic_error("Compute admission skipped the oldest ready input");

    const std::size_t output_used = output_occupancy(request);
    if (!token.final && output_used == kOutputSlots) {
      ++output_full_nonfinal_ready_observations_;
      if (first_output_full_nonfinal_witness_.is_null()) {
        first_output_full_nonfinal_witness_ = json{
            {"time_ns", runtime_.now_ns()}, {"request_id", request.id},
            {"output_q", token.q}, {"k_tile", token.k_tile},
            {"output_occupancy", output_used}};
      }
    }

    int output_index = -1;
    if (token.final) {
      output_index = first_free_output(request);
      if (output_index < 0) {
        if (request.last_final_wait_ordinal != std::optional<u64>(token.ordinal)) {
          request.last_final_wait_ordinal = token.ordinal;
          ++final_compute_wait_observations_;
          record_trace("final_compute_wait", request, token,
                       static_cast<std::size_t>(input_index));
        }
        return false;
      }
      auto& output_slot = request.outputs[static_cast<std::size_t>(output_index)];
      output_slot.state = OutputState::Reserved;
      output_slot.token = token;
      output_slot.handle = 0;
      ++final_output_reservations_;
    }

    const u64 handle = runtime_.submit(
        kernel_.plans(token)[kCompute], owner(request.id), runtime_.now_ns());
    if (request.cursor->admit(Role::Compute) != std::optional<u64>(token.ordinal))
      throw std::logic_error("compact Compute admission differs from explicit slots");
    input_slot.state = InputState::Computing;
    input_slot.handle = handle;
    request.computing = static_cast<std::size_t>(input_index);
    if (output_index >= 0)
      request.compute_output = static_cast<std::size_t>(output_index);
    else if (request.compute_output)
      throw std::logic_error("nonfinal Compute unexpectedly owns an output slot");
    ++submitted_by_stage_[kCompute];
    if (!token.final && output_used == kOutputSlots)
      ++nonfinal_computes_admitted_at_full_output_;
    verify_capacity(request);
    record_trace("compute_start", request, token,
                 static_cast<std::size_t>(input_index),
                 output_index < 0 ? kOutputSlots :
                     static_cast<std::size_t>(output_index));
    return true;
  }

  bool start_load(Request& request) {
    if (request.loading || request.next_load >= kernel_.wave_count())
      return false;
    const int slot_index = first_free_input(request);
    if (slot_index < 0) return false;
    const Token token = kernel_.token(request.next_load);
    const u64 handle = runtime_.submit(
        kernel_.plans(token)[kLoad], owner(request.id), runtime_.now_ns());
    auto& slot = request.inputs[static_cast<std::size_t>(slot_index)];
    slot.state = InputState::Loading;
    slot.token = token;
    slot.handle = handle;
    request.loading = static_cast<std::size_t>(slot_index);
    ++request.next_load;
    if (request.cursor->admit(Role::Load) != std::optional<u64>(token.ordinal))
      throw std::logic_error("compact Load admission differs from explicit slots");
    ++submitted_by_stage_[kLoad];
    verify_capacity(request);
    record_trace("load_start", request, token,
                 static_cast<std::size_t>(slot_index));
    return true;
  }

  void pump() {
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto& request : requests_) {
        if (!request.started || request.done) continue;
        const auto check = [&](Role role, auto start) {
          auto probe = *request.cursor;
          const bool expected = probe.admit(role).has_value();
          const bool actual = (this->*start)(request);
          if (actual != expected)
            throw std::logic_error("compact readiness differs from explicit stage eligibility");
          changed = changed || actual;
        };
        check(Role::Store, &Driver::start_store);
        check(Role::Compute, &Driver::start_compute);
        check(Role::Load, &Driver::start_load);
      }
    }
  }

  void collect_terminals() {
    for (auto& request : requests_) {
      if (!request.loading) continue;
      const std::size_t index = *request.loading;
      auto& slot = request.inputs[index];
      const auto finish = runtime_.terminal_finish_ns(slot.handle);
      if (!finish) continue;
      if (slot.state != InputState::Loading || slot.token.ordinal >= request.next_load)
        throw std::logic_error("Load terminal lost its admitted input token");
      request.cursor->complete(Role::Load, slot.token.ordinal);
      runtime_.destroy_execution(slot.handle);
      slot.state = InputState::Ready;
      slot.handle = 0;
      request.loading.reset();
      ++terminal_by_stage_[kLoad];
      verify_capacity(request);
      record_trace("load_terminal", request, slot.token, index);
    }

    for (auto& request : requests_) {
      if (!request.computing) continue;
      const std::size_t index = *request.computing;
      auto& input_slot = request.inputs[index];
      const auto finish = runtime_.terminal_finish_ns(input_slot.handle);
      if (!finish) continue;
      const Token token = input_slot.token;
      if (input_slot.state != InputState::Computing ||
          token.ordinal != request.next_compute)
        throw std::logic_error("Compute terminal violated input FIFO order");
      request.cursor->complete(Role::Compute, token.ordinal);
      runtime_.destroy_execution(input_slot.handle);
      input_slot = InputSlot{};
      request.computing.reset();
      ++request.next_compute;
      ++terminal_by_stage_[kCompute];

      if (token.final) {
        if (!request.compute_output)
          throw std::logic_error("final Compute terminal has no reserved output slot");
        auto& output = request.outputs[*request.compute_output];
        if (output.state != OutputState::Reserved ||
            output.token.ordinal != token.ordinal)
          throw std::logic_error("final Compute lost its output reservation");
        output.state = OutputState::Ready;
        ++final_outputs_produced_;
      } else if (request.compute_output) {
        throw std::logic_error("nonfinal Compute terminal held an output reservation");
      }
      request.compute_output.reset();
      verify_capacity(request);
      record_trace("compute_terminal", request, token, index);
    }

    for (auto& request : requests_) {
      if (!request.storing) continue;
      const std::size_t index = *request.storing;
      auto& output = request.outputs[index];
      const auto finish = runtime_.terminal_finish_ns(output.handle);
      if (!finish) continue;
      if (output.state != OutputState::Storing || !output.token.final)
        throw std::logic_error("Store terminal does not own a final output slot");
      request.cursor->complete(Role::Store, output.token.ordinal);
      runtime_.destroy_execution(output.handle);
      const Token token = output.token;
      output = OutputSlot{};
      request.storing.reset();
      ++terminal_by_stage_[kStore];
      ++store_terminal_slot_releases_;
      ++request.outputs_stored;
      if (request.outputs_stored == kernel_.output_count()) {
        request.done = true;
        request.finish_ns = *finish;
      }
      verify_capacity(request);
      record_trace("store_terminal_release", request, token,
                   kInputSlots, index);
    }
  }

  bool all_done() const {
    return std::all_of(requests_.begin(), requests_.end(),
                       [](const Request& request) { return request.done; });
  }

  void verify_finished_invariants() const {
    if (!runtime_.drained())
      throw std::logic_error("diagnostic completed with live Runtime executions");
    for (const auto& request : requests_) {
      if (request.next_load != kernel_.wave_count() ||
          request.next_compute != kernel_.wave_count() ||
          request.outputs_stored != kernel_.output_count() ||
          request.loading || request.computing || request.storing ||
          request.compute_output)
        throw std::logic_error("terminal stage counts or dependencies did not drain");
      if (input_occupancy(request) != 0 || output_occupancy(request) != 0)
        throw std::logic_error("readiness slots remained occupied at completion");
    }
    const u64 request_count = static_cast<u64>(requests_.size());
    if (submitted_by_stage_[kLoad] != request_count * kernel_.wave_count() ||
        submitted_by_stage_[kCompute] != request_count * kernel_.wave_count() ||
        submitted_by_stage_[kStore] != request_count * kernel_.output_count() ||
        terminal_by_stage_ != submitted_by_stage_ ||
        final_output_reservations_ != request_count * kernel_.output_count() ||
        final_outputs_produced_ != request_count * kernel_.output_count() ||
        store_terminal_slot_releases_ != request_count * kernel_.output_count())
      throw std::logic_error("load/compute/final-store conservation invariant failed");
  }

  json report() const {
    const u64 first_finish = requests_.empty() ? 0 : requests_.front().finish_ns;
    json extra = {
        {"diagnostic_only", true},
        {"complexity", "uncompressed O(Runtime event advances); no production efficiency claim"},
        {"production_E5_or_E8_certificate", false},
        {"production_minimum_E_certificate", false},
        {"finite_cycle_skipping", false},
        {"input_slots", kInputSlots}, {"output_slots", kOutputSlots},
        {"fact_class_count", kernel_.fact_class_count()},
        {"runtime_advances", runtime_advances_},
        {"stage_submissions", {{"load", submitted_by_stage_[kLoad]},
                                {"compute", submitted_by_stage_[kCompute]},
                                {"store", submitted_by_stage_[kStore]}}},
        {"stage_terminals", {{"load", terminal_by_stage_[kLoad]},
                              {"compute", terminal_by_stage_[kCompute]},
                              {"store", terminal_by_stage_[kStore]}}},
        {"final_output_reservations_at_compute_admission", final_output_reservations_},
        {"final_outputs_produced_at_compute_terminal", final_outputs_produced_},
        {"store_terminal_slot_releases", store_terminal_slot_releases_},
        {"final_compute_wait_observations", final_compute_wait_observations_},
        {"output_full_nonfinal_ready_observations", output_full_nonfinal_ready_observations_},
        {"nonfinal_computes_admitted_while_output_full", nonfinal_computes_admitted_at_full_output_},
        {"max_input_slot_occupancy", max_input_occupancy_},
        {"max_output_slot_occupancy", max_output_occupancy_},
        {"capacity_invariant_checks", capacity_invariant_checks_},
        {"slot_and_order_invariants", {
            {"maximum_input_occupancy_le_two", max_input_occupancy_ <= kInputSlots},
            {"maximum_output_occupancy_le_two", max_output_occupancy_ <= kOutputSlots},
            {"compute_uses_oldest_terminal_load", true},
            {"only_final_compute_reserves_output", true},
            {"only_final_compute_terminal_produces_output", true},
            {"store_terminal_releases_output_slot", true},
            {"compact_cursor_matches_explicit_slots_and_admissions", true},
            {"all_submitted_stages_reached_terminal", runtime_.drained()}}},
        {"first_output_full_nonfinal_admission", first_output_full_nonfinal_witness_},
        {"time_quantization", "public Runtime integer-nanosecond next_event()/advance_to()"}};
    if (trace_enabled_) extra["manual_fixture_stage_trace"] = trace_;

    const json counters = counters_json(runtime_);
    if (requests_.size() == 1) {
      return {{"request", "solo"}, {"arrival_ns", requests_[0].arrival},
              {"finish_ns", first_finish},
              {"finish_s", static_cast<double>(first_finish) / kNsPerSecond},
              {"event_advances", runtime_advances_}, {"counters", counters},
              {"readiness_diagnostic", std::move(extra)}};
    }
    json finishes_ns = json::object();
    json finishes_s = json::object();
    for (std::size_t i = 0; i < requests_.size(); ++i) {
      const std::string name = "comp" + std::to_string(i);
      finishes_ns[name] = requests_[i].finish_ns;
      finishes_s[name] = static_cast<double>(requests_[i].finish_ns) / kNsPerSecond;
    }
    const u64 offset = requests_.size() > 1 ? requests_[1].arrival : 0;
    return {{"request_names", [&] {
               json names = json::array();
               for (std::size_t i = 0; i < requests_.size(); ++i)
                 names.push_back("comp" + std::to_string(i));
               return names;
             }()},
            {"phase_offset_ns", offset},
            {"phase_offset_s", static_cast<double>(offset) / kNsPerSecond},
            {"finish_ns_by_request", std::move(finishes_ns)},
            {"finish_s_by_request", std::move(finishes_s)},
            {"event_advances", runtime_advances_}, {"counters", counters},
            {"readiness_diagnostic", std::move(extra)}};
  }

  Kernel& kernel_;
  Runtime runtime_;
  std::vector<Request> requests_;
  bool trace_enabled_ = false;
  json trace_ = json::array();
  u64 runtime_advances_ = 0;
  std::array<u64, 3> submitted_by_stage_{};
  std::array<u64, 3> terminal_by_stage_{};
  u64 final_output_reservations_ = 0;
  u64 final_outputs_produced_ = 0;
  u64 store_terminal_slot_releases_ = 0;
  u64 final_compute_wait_observations_ = 0;
  u64 output_full_nonfinal_ready_observations_ = 0;
  u64 nonfinal_computes_admitted_at_full_output_ = 0;
  u64 capacity_invariant_checks_ = 0;
  std::size_t max_input_occupancy_ = 0;
  std::size_t max_output_occupancy_ = 0;
  json first_output_full_nonfinal_witness_ = nullptr;
};

json run_case(const Hardware& hardware, const Plan& mapping,
              const std::vector<u64>& arrivals) {
  Kernel kernel(hardware, geometry_input(hardware, mapping), false);
  Driver driver(hardware, kernel, arrivals);
  return driver.run();
}

json run_manual_self_test() {
  Hardware hardware = synthetic_hardware();
  hardware.rates[coarse_die::HBM0] = 25.0;
  hardware.rates[coarse_die::HBM1] = 1.0;
  hardware.derived.at("resources").at(coarse_die::HBM0)["rate"] = 25.0;
  hardware.derived.at("resources").at(coarse_die::HBM1)["rate"] = 1.0;
  hardware.fingerprint = "synthetic-k-readiness-manual-stage-fixture";
  geometry::Input input;
  input.M = 3;
  input.N = 1;
  input.K = 2;
  input.tm = 1;
  input.tn = 1;
  input.tk = 1;
  input.core_count = 1;
  input.matrix_path = access::MatrixPath::StrictFp32Vec;
  input.a_storage = access::StorageDType::Fp32;
  input.w_storage = access::StorageDType::Fp32;
  input.output_storage = access::StorageDType::Fp32;
  Kernel kernel(hardware, input, true);
  Driver driver(hardware, kernel, {0}, true);
  const json actual = driver.run();
  const json diagnostic = actual.at("readiness_diagnostic");
  const bool admitted_nonfinal_at_full =
      diagnostic.at("nonfinal_computes_admitted_while_output_full").get<u64>() > 0;
  const bool final_waited =
      diagnostic.at("final_compute_wait_observations").get<u64>() > 0;
  const bool store_released_all =
      diagnostic.at("store_terminal_slot_releases").get<u64>() == 3;
  const bool all_stored = actual.at("finish_ns").get<u64>() == 15000000000ULL &&
      diagnostic.at("stage_terminals").at("store").get<u64>() == 3;
  const json trace = diagnostic.at("manual_fixture_stage_trace");
  json final_wait = nullptr;
  json final_compute_after_wait = nullptr;
  json store_releases = json::array();
  for (const auto& event : trace) {
    if (event.at("event") == "final_compute_wait" && final_wait.is_null())
      final_wait = event;
    if (event.at("event") == "compute_start" && event.at("output_q") == 2 &&
        event.at("k_tile") == 1)
      final_compute_after_wait = event;
    if (event.at("event") == "store_terminal_release")
      store_releases.push_back({{"output_q", event.at("output_q")},
                                {"time_ns", event.at("time_ns")}});
  }
  const json full_nonfinal_witness =
      diagnostic.at("first_output_full_nonfinal_admission");
  const bool expected_sequence =
      full_nonfinal_witness.at("time_ns") == 5000000000ULL &&
      full_nonfinal_witness.at("output_q") == 2 &&
      full_nonfinal_witness.at("k_tile") == 0 &&
      final_wait.at("time_ns") == 6000000000ULL &&
      final_wait.at("output_q") == 2 && final_wait.at("k_tile") == 1 &&
      final_compute_after_wait.at("time_ns") == 7000000000ULL &&
      store_releases == json::array({
          {{"output_q", 0}, {"time_ns", 7000000000ULL}},
          {{"output_q", 1}, {"time_ns", 11000000000ULL}},
          {{"output_q", 2}, {"time_ns", 15000000000ULL}}});
  const bool passed = admitted_nonfinal_at_full && final_waited &&
      store_released_all && all_stored && expected_sequence;
  return {{"fixture", "three output tiles x two K consumptions; synthetic manual stage work"},
          {"expected", {{"output_full_nonfinal_compute_admitted", true},
                        {"final_compute_waited_for_free_output_slot", true},
                        {"store_terminal_released_each_output_slot", true},
                        {"three_outputs_stored", true},
                        {"finish_ns", 15000000000ULL},
                        {"output_full_nonfinal_at_ns", 5000000000ULL},
                        {"final_wait_at_ns", 6000000000ULL},
                        {"final_compute_resumed_at_ns", 7000000000ULL}}},
          {"actual", {{"output_full_nonfinal_ready_observations",
                        diagnostic.at("output_full_nonfinal_ready_observations")},
                       {"first_output_full_nonfinal_admission",
                        full_nonfinal_witness},
                       {"nonfinal_computes_admitted_while_output_full",
                        diagnostic.at("nonfinal_computes_admitted_while_output_full")},
                       {"final_compute_wait_observations",
                        diagnostic.at("final_compute_wait_observations")},
                       {"first_final_compute_wait", final_wait},
                       {"final_compute_after_store_release",
                        final_compute_after_wait},
                       {"store_terminal_slot_releases",
                        diagnostic.at("store_terminal_slot_releases")},
                       {"store_terminal_release_times", store_releases},
                       {"finish_ns", actual.at("finish_ns")},
                       {"manual_fixture_stage_trace",
                        diagnostic.at("manual_fixture_stage_trace")},
                       {"slot_and_order_invariants",
                        diagnostic.at("slot_and_order_invariants")}}},
          {"passed", passed}};
}

}  // namespace k_readiness

int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i) {
      if (std::string_view(argv[i]) == "--self-test") {
        const json result = k_readiness::run_manual_self_test();
        std::cout << result.dump(2) << '\n';
        return result.at("passed").get<bool>() ? 0 : 1;
      }
    }
    const Shape shape = parse_shape(argc, argv);
    const Hardware hardware = synthetic_hardware();
    const auto mapping = compile(hardware, gemm_task(shape));
    json two_comp = json::array();
    for (std::uint64_t offset : {0ULL, 50000000ULL})
      two_comp.push_back(k_readiness::run_case(hardware, *mapping, {0, offset}));
    const json output = {
        {"schema_version", 2},
        {"measurement_source", "diagnostic_cpp_k_ready_facts_and_actual_public_runtime"},
        {"diagnostic_only", true},
        {"baseline_compiled_plan_executed", false},
        {"mapping_plan_role", "selected shape/mapping and independent workload ledger source only"},
        {"validation_scope", "readiness/slot diagnostic; not production E5/E8 or efficiency evidence"},
        {"complexity", "uncompressed O(Runtime event advances); fixed diagnostic limit 1000000"},
        {"synthetic_hardware", synthetic_capacity_json(hardware)},
        {"compiled_plan", plan_json(*mapping)},
        {"standalone", k_readiness::run_case(hardware, *mapping, {0})},
        {"two_comp", std::move(two_comp)}};
    std::cout << output.dump(2) << '\n';
  } catch (const std::exception& error) {
    std::cerr << "k-readiness runtime diagnostic: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
