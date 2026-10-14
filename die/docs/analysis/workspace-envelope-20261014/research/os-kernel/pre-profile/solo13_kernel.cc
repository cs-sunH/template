#include "solo13_kernel.hh"

#include <fstream>
#include <iostream>
#include <string>

namespace {

using namespace solo13;

std::string status_name(Kernel::Status status) {
  switch (status) {
    case Kernel::Status::Passed: return "passed";
    case Kernel::Status::Paused: return "paused";
    case Kernel::Status::FailedClosed: return "failed_closed";
  }
  return "unknown";
}

json ledger_json(const Ledger& ledger) {
  json service = json::array(), busy = json::array(), hbm = json::array();
  for (std::size_t row = 0; row < kRows; ++row) {
    service.push_back(ledger.service[row].to_string());
    busy.push_back(ledger.busy[row].to_string());
  }
  for (std::size_t row = 0; row < 2; ++row) {
    hbm.push_back({ledger.hbm_payload[row][0].to_string(),
                   ledger.hbm_payload[row][1].to_string()});
  }
  return {{"elapsed_s", ledger.elapsed.to_string()},
          {"resource_service_13", std::move(service)},
          {"resource_busy_s_13", std::move(busy)},
          {"hbm_payload_row_direction", std::move(hbm)},
          {"int4_ops", ledger.int4_ops.to_string()},
          {"vec_eqops", ledger.vec_eqops.to_string()},
          {"compute_busy_s", ledger.compute_busy.to_string()},
          {"hbm0_busy_s", ledger.hbm_row_busy[0].to_string()},
          {"hbm1_busy_s", ledger.hbm_row_busy[1].to_string()},
          {"hbm_busy_s", ledger.hbm_busy.to_string()},
          {"hbm_shared_s", "0/1"}};
}

json state_json(const Kernel& kernel, Kernel::Status status) {
  const auto& owner = kernel.owner();
  const auto& facts = kernel.facts();
  const auto& input = facts.input;
  json active = json::array(), loaded = json::array(), computed = json::array();
  for (std::size_t stage = 0; stage < kStages; ++stage) {
    if (!owner.active[stage].present) continue;
    active.push_back({{"stage", stage_name(stage)},
                      {"wave", u128_string(owner.active[stage].wave)},
                      {"progress", owner.active[stage].progress.to_string()}});
  }
  for (std::size_t i = 0; i < owner.loaded_count; ++i)
    loaded.push_back(u128_string(owner.loaded[i]));
  for (std::size_t i = 0; i < owner.computed_count; ++i)
    computed.push_back(u128_string(owner.computed[i]));
  json capacities = json::array();
  for (const auto& capacity : kernel.capacities())
    capacities.push_back(capacity.to_string());
  const auto dtype_name = [](coarse_die::detail::gemm_os_access_counter::StorageDType dtype) {
    using DType = coarse_die::detail::gemm_os_access_counter::StorageDType;
    switch (dtype) {
      case DType::Fp32: return "fp32";
      case DType::Bf16: return "bf16";
      case DType::Int4: return "int4";
      case DType::Int32: return "int32";
    }
    return "unknown";
  };
  return {{"status", status_name(status)},
          {"reason", kernel.error()},
          {"expanded_events", kernel.events()},
          {"skipped_waves", u128_string(kernel.skipped_waves())},
          {"skipped_outputs", u128_string(kernel.skipped_outputs())},
          {"history_entries", kernel.history_count()},
          {"identity_skips_enabled", kernel.identity_skips_enabled()},
          {"capacity_invalidations", kernel.capacity_invalidations()},
          {"peak_active_stages", kernel.peak_active()},
          {"peak_input_slots", kernel.peak_input()},
          {"peak_output_slots", kernel.peak_output()},
          {"nonfinal_compute_started_while_output_full",
           kernel.nonfinal_compute_at_full_output()},
          {"final_compute_waited_for_full_output_slot",
           kernel.final_compute_blocked_on_full_output()},
          {"capacities_13", std::move(capacities)},
          {"geometry", {{"M", input.M}, {"N", input.N}, {"K", input.K},
                        {"tm", input.tm}, {"tn", input.tn}, {"tk", input.tk},
                        {"core_count", input.core_count},
                        {"output_tiles", u128_string(facts.geometry.output_tile_count())},
                        {"consumption_waves", u128_string(facts.geometry.consumption_wave_count())},
                        {"access_class_count", facts.class_count},
                        {"class_cache_capacity", kMaxClasses},
                        {"path", input.matrix_path ==
                            coarse_die::detail::gemm_os_access_counter::MatrixPath::StrictFp32Vec
                                ? "strict_fp32_vec" : "int4_pe"},
                        {"dtypes", {dtype_name(input.a_storage), dtype_name(input.w_storage),
                                    dtype_name(input.output_storage)}},
                        {"kappa_quant", input.kappa_quant},
                        {"kappa_acc", input.kappa_acc}}},
          {"bounded_kernel", {{"active_stage_capacity", kStages},
                               {"input_queue_capacity", kQueueSlots},
                               {"output_queue_capacity", kQueueSlots},
                               {"global_expanded_event_cap", kGlobalEventCap},
                               {"history_snapshot_capacity", kHistory},
                               {"tile_or_core_arrays", false},
                               {"time_model", "exact_continuous_internal_fraction"},
                               {"public_integer_ns_runtime_clock_used", false}}},
          {"live_state", {{"next_load_wave", u128_string(owner.next_load)},
                           {"retired_outputs", u128_string(owner.retired)},
                           {"loaded_waves", std::move(loaded)},
                           {"computed_final_waves", std::move(computed)},
                           {"active", std::move(active)},
                           {"finish", owner.finished}}},
          {"ledger", ledger_json(kernel.ledger())}};
}

Rational scale(const Rational& value, const Rational& factor) {
  return r_mul(value, factor);
}

json execute(const json& facts, const std::string& scenario) {
  if (scenario == "full") {
    Kernel kernel(facts);
    const auto status = kernel.run_until();
    return {{"scenario", scenario}, {"result", state_json(kernel, status)}};
  }
  if (scenario == "unpowered-cap") {
    Kernel kernel(facts, false);
    const auto status = kernel.run_until();
    return {{"scenario", scenario}, {"result", state_json(kernel, status)}};
  }
  if (scenario == "integer-stop") {
    Kernel kernel(facts);
    const Rational stop = r_fraction("17/1000000000");
    const auto first = kernel.run_until(&stop);
    const json paused = state_json(kernel, first);
    const auto resumed = kernel.resume_until();
    return {{"scenario", scenario}, {"pause", paused},
            {"result", state_json(kernel, resumed)}};
  }
  if (scenario == "pause-resume") {
    Kernel reference(facts);
    const auto reference_status = reference.run_until();
    if (reference_status != Kernel::Status::Passed)
      throw std::runtime_error("unstopped reference did not finish");
    const Rational finish = reference.ledger().elapsed;
    const Rational one_third = r_fraction("1/3");
    const Rational two_thirds = r_fraction("2/3");
    const Rational first_stop = scale(finish, one_third);
    const Rational second_stop = scale(finish, two_thirds);
    Kernel kernel(facts);
    const auto first = kernel.run_until(&first_stop);
    const json pause1 = state_json(kernel, first);
    const u64 before_resume1 = kernel.events();
    const auto second = kernel.resume_until(&second_stop);
    const json pause2 = state_json(kernel, second);
    const u64 before_resume2 = kernel.events();
    const auto final = kernel.resume_until();
    return {{"scenario", scenario}, {"pause1", pause1}, {"pause2", pause2},
            {"events_before_resume1", before_resume1},
            {"events_before_resume2", before_resume2},
            {"unstopped_reference", state_json(reference, reference_status)},
            {"result", state_json(kernel, final)}};
  }
  if (scenario == "capacity-stop") {
    Kernel reference(facts);
    const auto reference_status = reference.run_until();
    if (reference_status != Kernel::Status::Passed)
      throw std::runtime_error("unstopped reference did not finish");
    const Rational finish = reference.ledger().elapsed;
    const Rational first_stop = scale(finish, r_fraction("1/3"));
    const Rational second_stop = scale(finish, r_fraction("1/2"));
    const Rational backwards_stop = first_stop;
    Kernel kernel(facts);
    const auto first = kernel.run_until(&first_stop);
    const json pause1 = state_json(kernel, first);
    std::array<Rational, kRows> changed = kernel.capacities();
    const Rational factor = r_fraction("7/8");
    changed[4] = r_mul(changed[4], factor);
    changed[5] = r_mul(changed[5], factor);
    const u64 events_before_change = kernel.events();
    const auto second = kernel.resume_until(&second_stop, &changed);
    const json pause2 = state_json(kernel, second);
    const u64 events_after_change = kernel.events();
    const Rational clock_before_reverse = kernel.ledger().elapsed;
    const auto backwards = kernel.resume_until(&backwards_stop);
    const json reverse = state_json(kernel, backwards);
    const bool clock_preserved = kernel.ledger().elapsed.equal(clock_before_reverse);
    const auto final = kernel.resume_until();
    return {{"scenario", scenario}, {"pause1", pause1}, {"pause2", pause2},
            {"backwards", reverse}, {"clock_preserved_after_backwards", clock_preserved},
            {"events_before_capacity_change", events_before_change},
            {"events_after_capacity_change", events_after_change},
            {"capacity_invalidations_after_resume", kernel.capacity_invalidations()},
            {"unstopped_reference", state_json(reference, reference_status)},
            {"result", state_json(kernel, final)}};
  }
  throw std::invalid_argument("scenario must be full, integer-stop, pause-resume, or capacity-stop");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string facts_path, scenario = "full";
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--help" || option == "-h") {
        std::cout << "usage: solo13_kernel --facts FILE [--scenario full|integer-stop|pause-resume|capacity-stop|unpowered-cap]\n";
        return 0;
      }
      if (i + 1 >= argc) throw std::invalid_argument(option + " requires an argument");
      const std::string value = argv[++i];
      if (option == "--facts") facts_path = value;
      else if (option == "--scenario") scenario = value;
      else throw std::invalid_argument("unknown option: " + option);
    }
    if (facts_path.empty()) throw std::invalid_argument("--facts is required");
    std::ifstream input(facts_path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open facts JSON: " + facts_path);
    const json facts = json::parse(input);
    const json result = execute(facts, scenario);
    std::cout << result.dump(2) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "failed_closed: " << error.what() << '\n';
    return 2;
  }
}
