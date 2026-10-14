#include "test_support.hh"
#include <limits>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {
void rejected(const json& summary) {
  Plan plan;
  plan.summary = summary;
  bool failed = false;
  try { (void)hbm_workspace_peak_bytes(plan); }
  catch (const std::invalid_argument& error) {
    failed = std::string(error.what()).find("invalid_workspace:") == 0;
  }
  require(failed, "malformed capacity metadata must fail closed");
}

void body() {
  for (const json& value : {json(-1), json(1.0), json(true), json(nullptr),
                            json("12"), json::array(), json::object()}) {
    rejected(json{{"hbm_workspace_peak_bytes", value}});
  }
  rejected(json::object());
  rejected(json::array());
  // JSON parses integers beyond uint64 into a floating value; it must not be
  // rounded into a capacity claim, even if it looks integral in binary64.
  rejected(json::parse("{\"hbm_workspace_peak_bytes\":18446744073709551616}"));
  for (const std::uint64_t expected : {std::uint64_t{0}, std::uint64_t{1},
                                      std::uint64_t{1} << 63,
                                      std::numeric_limits<std::uint64_t>::max()}) {
    Plan plan;
    plan.summary = {{"hbm_workspace_peak_bytes", expected}};
    require(hbm_workspace_peak_bytes(plan) == expected,
            "unsigned workspace count must preserve every uint64 bit");
  }
  Plan signed_plan;
  signed_plan.summary = {{"hbm_workspace_peak_bytes", std::int64_t{12}}};
  require(hbm_workspace_peak_bytes(signed_plan) == 12,
          "nonnegative signed JSON integer is exact");

  const auto hardware = example_hardware();
  const json task{{"op_kind", "ffn_region"},
    {"tensor_shapes", {{"m", 3}, {"hidden", 17}, {"ffn", 19},
                        {"mlp_variant", "swiglu"}}},
    {"fusion_policy", "materialized"}, {"storage_dtype", "bf16"}};
  const auto plan = compile(hardware, task);
  require(hbm_workspace_peak_bytes(*plan) > 0,
          "real materialized FFN requires incremental HBM workspace");
  const auto restored = plan_from_json(plan_to_json(*plan));
  require(hbm_workspace_peak_bytes(*restored) == hbm_workspace_peak_bytes(*plan) &&
          restored->fingerprint == plan->fingerprint,
          "typed capacity read survives canonical Plan roundtrip");
  require(plan->summary.at("hbm_read_bytes_total").get<std::uint64_t>() >
              hbm_workspace_peak_bytes(*plan),
          "workspace capacity and repeated transfer payload remain separate");
}
}

int main() { return run_test("workspace_metadata_test", body); }
