#include "coarse_die/Capi.h"

#include <json/json.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using nlohmann::json;

static void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

static json take(const char* value) {
  check(value != nullptr, die_coarse_last_error_v2());
  auto result = json::parse(value);
  die_coarse_free_result_v2(value);
  return result;
}

static double hbm_read_bytes(const json& progress) {
  const auto& rows = progress.at("hbm_served_bytes");
  return rows.at(0).at(0).get<double>() + rows.at(1).at(0).get<double>();
}

int main() {
  try {
    check(die_coarse_abi_version() == 2, "ABI version");
    std::ifstream in(std::string(COARSE_DIE_SOURCE_DIR) + "/configs/example_hardware.json");
    check(static_cast<bool>(in), "example hardware configuration exists");
    json cfg;
    in >> cfg;

    // Structural hardware errors use the same status at namespace roots and
    // leaf fields, and a rejected model must not publish a handle.
    for (unsigned kind=0; kind<7; ++kind) {
      auto invalid_cfg=cfg;
      if(kind==0)invalid_cfg["architecture"].erase("system_region_height_mm");
      if(kind==1)invalid_cfg["performance_profile"]=json::array();
      if(kind==2)invalid_cfg["core_resource_table"]["rows_sha256"]="incorrect";
      if(kind==3)invalid_cfg["source_revision"]="incorrect";
      if(kind==4)invalid_cfg["source_constants"]=json::array();
      if(kind==5)invalid_cfg["validation_constraints"]={{"objective","pareto"}};
      if(kind==6)invalid_cfg.erase("architecture");
      DieHandle rejected=123456;
      check(die_coarse_create_model_v2(invalid_cfg.dump().c_str(), &rejected)==
                DIE_INVALID_HARDWARE,
            "invalid hardware namespace/leaf must have the hardware status: "+
                std::string(die_coarse_last_error_v2()));
      check(rejected==0,"invalid hardware clears the output without publishing a model handle");
      const auto error=json::parse(die_coarse_last_error_v2());
      check(error.at("status_code")==DIE_INVALID_HARDWARE &&
                !error.at("field_path").get<std::string>().empty(),
            "hardware rejection retains a field diagnostic");
    }

    DieHandle model = 0;
    DieHandle runtime = 0;
    DieHandle execution = 0;
    check(die_coarse_create_model_v2(cfg.dump().c_str(), &model) == DIE_OK,
          die_coarse_last_error_v2());
    const auto hw = take(die_coarse_hardware_summary_v2(model));
    check(hw.at("schema_version") == 2, "summary version");
    check(die_coarse_create_runtime_v2(model, &runtime) == DIE_OK,
          die_coarse_last_error_v2());
    check(die_coarse_destroy_model_v2(model) == DIE_BUSY,
          "live runtime prevents model destruction");

    DieOwnerV2 owner{sizeof(DieOwnerV2), 2, 1, 2, 3, 1};
    const json endpoint{{"endpoint_id", 1}, {"transaction_id", 1},
                        {"owner_kind", "comm"}, {"direction", "read"},
                        {"physical_bytes", 1000000}, {"path_class", "hbm_to_edge"},
                        {"eta_hbm", 0.7}, {"route", {{"internal_route", false}}}};
    check(die_coarse_submit_endpoint_v2(runtime, endpoint.dump().c_str(), &owner,
                                        0, &execution) == DIE_OK,
          die_coarse_last_error_v2());
    check(die_coarse_destroy_execution_v2(execution) == DIE_BUSY,
          "active execution lifecycle");
    check(die_coarse_destroy_runtime_v2(runtime) == DIE_BUSY,
          "active runtime lifecycle");

    const auto event = take(die_coarse_next_event_v2(runtime));
    check(!event.at("next_event_ns").is_null(),
          "active endpoint must have a finite next event");
    const auto next = event.at("next_event_ns").get<std::uint64_t>();
    check(next > 1, "endpoint duration must leave an integer midpoint for partial service");

    const auto initial = take(die_coarse_query_progress_v2(execution));
    check(!initial.at("terminal").get<bool>() &&
              initial.at("activities").at(0).at("status") == "running" &&
              initial.at("activities").at(0).at("rate").get<double>() > 0.0,
          "submitted HBM endpoint receives positive service");
    check(hbm_read_bytes(initial) == 0.0,
          "endpoint has no served HBM bytes before runtime advances");

    const auto midpoint = next / 2;
    check(die_coarse_advance_to_v2(runtime, midpoint) == DIE_OK,
          die_coarse_last_error_v2());
    const auto partial = take(die_coarse_query_progress_v2(execution));
    const double partial_bytes = hbm_read_bytes(partial);
    check(partial_bytes > 0.0 && partial_bytes < 1000000.0 &&
              partial.at("activities").at(0).at("progress").get<double>() > 0.0 &&
              partial.at("activities").at(0).at("progress").get<double>() < 1.0,
          "partial endpoint progress records only the HBM service delivered so far");
    const auto counters_at_midpoint = take(die_coarse_query_counters_v2(runtime));
    check(counters_at_midpoint.at("active_activities") == 1 &&
              counters_at_midpoint.at("hbm_served_bytes_by_row_direction").at(0).at(0).get<double>() > 0.0,
          "runtime counters retain the active endpoint and its partial served bytes");

    const auto snapshot = take(die_coarse_export_snapshot_v2(runtime));
    DieHandle fork = 0;
    check(die_coarse_import_snapshot_v2(model, snapshot.dump().c_str(), &fork) == DIE_OK,
          die_coarse_last_error_v2());
    auto corrupt = snapshot;
    corrupt["schema"] = "mesh_die_v1";
    DieHandle bad = 999;
    check(die_coarse_import_snapshot_v2(model, corrupt.dump().c_str(), &bad) != DIE_OK &&
              bad == 0,
          "invalid snapshot import is atomic");

    DieHandle cancel_fork = 0;
    check(die_coarse_import_snapshot_v2(model, snapshot.dump().c_str(), &cancel_fork) == DIE_OK,
          die_coarse_last_error_v2());
    const auto listed = take(die_coarse_list_executions_v2(cancel_fork));
    check(listed.at("executions").size() == 1,
          "snapshot import exposes its private live execution through list API");
    const auto cancel_execution = listed.at("executions").at(0)
                                      .at("execution_handle").get<DieHandle>();
    check(next > midpoint + 1, "endpoint event leaves time to cancel before completion");
    const auto cancel_at = next - 1;
    check(die_coarse_cancel_execution_v2(cancel_execution, cancel_at) == DIE_OK,
          die_coarse_last_error_v2());
    const auto cancelled = take(die_coarse_query_progress_v2(cancel_execution));
    const double cancelled_bytes = hbm_read_bytes(cancelled);
    check(cancelled.at("state") == "cancelled" &&
              cancelled.at("terminal").get<bool>() &&
              cancelled.at("finish_ns").get<std::uint64_t>() == cancel_at &&
              cancelled_bytes > partial_bytes && cancelled_bytes < 1000000.0,
          "cancellation settles partial HBM service before releasing the endpoint");
    check(take(die_coarse_next_event_v2(cancel_fork)).at("next_event_ns").is_null(),
          "a cancelled and drained private runtime has no next event");
    check(die_coarse_destroy_execution_v2(cancel_execution) == DIE_OK,
          die_coarse_last_error_v2());
    check(die_coarse_destroy_runtime_v2(cancel_fork) == DIE_OK,
          die_coarse_last_error_v2());

    check(die_coarse_advance_to_v2(runtime, next) == DIE_OK, die_coarse_last_error_v2());
    check(die_coarse_advance_to_v2(fork, next) == DIE_OK, die_coarse_last_error_v2());
    const auto completed = take(die_coarse_query_progress_v2(execution));
    check(completed.at("state") == "finished" && completed.at("terminal").get<bool>() &&
              completed.at("finish_ns").get<std::uint64_t>() == next &&
              hbm_read_bytes(completed) == 1000000.0,
          "completion accounts for the full endpoint payload exactly once");
    check(take(die_coarse_query_counters_v2(runtime)) ==
              take(die_coarse_query_counters_v2(fork)),
          "private snapshot fork replays identical event order and counters");
    check(take(die_coarse_next_event_v2(runtime)).at("next_event_ns").is_null(),
          "a completed runtime has no next event");
    check(die_coarse_advance_to_v2(runtime, next - 1) == DIE_INVALID_TIME,
          "time reversal");

    check(die_coarse_destroy_execution_v2(execution) == DIE_OK,
          die_coarse_last_error_v2());
    check(die_coarse_query_progress_v2(execution) == nullptr,
          "destroyed execution handle is stale");

    // Destroying a terminal endpoint releases its endpoint/transaction identity.
    DieHandle reused_endpoint = 0;
    check(die_coarse_submit_endpoint_v2(runtime, endpoint.dump().c_str(), &owner,
                                        next, &reused_endpoint) == DIE_OK,
          die_coarse_last_error_v2());
    check(die_coarse_cancel_execution_v2(reused_endpoint, next) == DIE_OK,
          die_coarse_last_error_v2());
    const auto zero_service_cancel = take(die_coarse_query_progress_v2(reused_endpoint));
    check(zero_service_cancel.at("state") == "cancelled" &&
              hbm_read_bytes(zero_service_cancel) == 0.0,
          "cancelling at arrival terminates without charging unserved payload");
    check(die_coarse_destroy_execution_v2(reused_endpoint) == DIE_OK,
          die_coarse_last_error_v2());

    check(die_coarse_destroy_runtime_v2(runtime) == DIE_OK, die_coarse_last_error_v2());
    check(die_coarse_destroy_runtime_v2(fork) == DIE_OK, die_coarse_last_error_v2());
    check(die_coarse_destroy_model_v2(model) == DIE_OK, die_coarse_last_error_v2());
    check(die_coarse_destroy_model_v2(model) == DIE_STALE_HANDLE,
          "stale model");
    DieHandle incomplete_model = 0;
    check(die_coarse_create_model_v2("{}", &incomplete_model) != DIE_OK,
          "reject incomplete config");
    std::cout << "{\"cabi_lifecycle\":\"passed\",\"abi\":2}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
