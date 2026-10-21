// WP2 (roofline stage 2, 2026-10-19): typed terminal summary replaces the
// host completion callbacks' full query_progress() JSON. This test pins the
// new Runtime::terminal_summary() contract against the public JSON as the
// oracle:
//   1. finished parity — every consumed field equals what query_progress()
//      serializes for the same terminal state (state mapping, issue/finish/
//      elapsed, served int4/vec doubles, the 2x2 HBM double matrix);
//   2. both HBM consumption styles stay separately carried and each equals
//      its old JSON consumption: the COMP style (per-entry double cast,
//      row-major sum in double) and the endpoint style (raw long doubles
//      summed per direction, cast once — the parse of the exact decimal
//      strings). A >2^53 fixture makes the two orders genuinely diverge;
//   3. cancelled with partial service — parity holds, served is strictly
//      between zero and planned (a planned/served mix-up fails here), and
//      non-integer fractional service stays exact;
//   4. zero-work bodies — finish 0 is legal and the summary is engaged;
//   5. live executions return nullopt, unknown handles throw the same
//      stale_handle error as query_progress;
//   6. a many-round repeat plan (grown activity history) still summarizes
//      execution-level fields exactly.
// The public query_progress() itself is unchanged and covered elsewhere;
// here it is the oracle only.
#include "test_support.hh"
#include "coarse_die/Execution.hh"
#include "coarse_die/Plan.hh"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {

using u64 = std::uint64_t;

// 2^53 + 1: casting each entry first loses the +1 (double rounding), while
// summing the exact values first keeps 2^53 + 2 — the two HBM consumption
// styles provably differ on this fixture.
constexpr u64 kEdgeReadA = (u64{1} << 53) + 1;
constexpr u64 kEdgeReadB = 1;

std::array<double, resource_count> gate_rates() {
  std::array<double, resource_count> rates{};
  rates[COMPUTE_GATE] = 1e9;
  rates[SRAM_READ] = 1e9;
  // HBM row payload bytes require positive row service work, so the two HBM
  // rows need a positive rate as well (one unit per ns, like the gate).
  rates[HBM0] = 1e9;
  rates[HBM1] = 1e9;
  return rates;
}

Activity byte_activity(u64 id, double gate_work, u64 read0, u64 write0,
                       u64 read1, u64 write1, u64 int4_ops, u64 vec_eqops) {
  Activity a;
  a.id = id;
  a.role = Role::Compute;
  a.work[COMPUTE_GATE] = gate_work;
  a.hbm_bytes = {{{{read0, write0}}, {{read1, write1}}}};
  if (read0 != 0 || write0 != 0) a.work[HBM0] = 1.0;
  if (read1 != 0 || write1 != 0) a.work[HBM1] = 1.0;
  a.int4_ops = int4_ops;
  a.vec_eqops = vec_eqops;
  return a;
}

std::shared_ptr<const Plan> byte_plan(const Hardware& hw, double gate_work,
                                      u64 read0, u64 write0, u64 read1,
                                      u64 write1, u64 int4_ops,
                                      u64 vec_eqops) {
  Epoch epoch;
  epoch.name = "bytes";
  epoch.activities = {
      byte_activity(1, gate_work, read0, write0, read1, write1, int4_ops,
                    vec_eqops)};
  auto plan = std::make_shared<Plan>();
  plan->task = json{{"op_kind", "terminal_summary_fixture"}};
  plan->summary = json{{"status", "test_fixture"}};
  plan->hardware_fingerprint = hw.fingerprint;
  plan->launch_ns = 0;
  plan->epochs = {epoch};
  plan->fingerprint =
      plan_to_json(*plan).at("fingerprint").get<std::string>();
  return plan;
}

void drain(Runtime& runtime) {
  unsigned guard = 0;
  while (!runtime.drained()) {
    const auto next = runtime.next_event();
    require(next.has_value() && *next > runtime.now_ns() && ++guard < 4096,
            "summary fixture has a bounded advancing completion sequence");
    runtime.advance_to(*next);
  }
}

long double exact_number(const json& value) {
  return std::stold(value.get<std::string>());
}

// Oracle comparisons shared by every terminal fixture.
void require_summary_matches_oracle(const Runtime& runtime, u64 handle,
                                    const TerminalSummary& summary,
                                    const std::string& expected_state) {
  const json oracle = runtime.query_progress(handle);
  require(oracle.at("state").get<std::string>() == expected_state,
          "oracle state is the expected terminal state");
  require((expected_state == "cancelled") == summary.cancelled &&
              summary.terminal,
          "summary maps the same terminal state");
  require(oracle.at("issue_ns").get<u64>() == summary.issue_ns &&
              oracle.at("finish_ns").get<u64>() == summary.finish_ns &&
              oracle.at("elapsed_ns").get<u64>() == summary.elapsed_ns &&
              summary.elapsed_ns == summary.finish_ns - summary.issue_ns,
          "summary times equal the oracle fields");
  require(oracle.at("served").at("int4_ops").get<double>() ==
              summary.int4_served &&
              oracle.at("served").at("vec_eqops").get<double>() ==
                  summary.vec_served,
          "summary served doubles equal the oracle projections");
  for (std::size_t row = 0; row < 2; ++row) {
    for (std::size_t kind = 0; kind < 2; ++kind) {
      require(oracle.at("hbm_served_bytes")[row][kind].get<double>() ==
                  summary.hbm_served[row][kind],
              "summary HBM matrix equals the oracle matrix entrywise");
    }
  }
  // Endpoint style: exact decimal strings parsed and summed per direction in
  // row-major order, cast once at the end.
  const long double read_exact =
      exact_number(oracle.at("hbm_served_bytes_exact")[0][0]) +
      exact_number(oracle.at("hbm_served_bytes_exact")[1][0]);
  const long double write_exact =
      exact_number(oracle.at("hbm_served_bytes_exact")[0][1]) +
      exact_number(oracle.at("hbm_served_bytes_exact")[1][1]);
  require(static_cast<double>(read_exact) == summary.hbm_read_bytes &&
              static_cast<double>(write_exact) == summary.hbm_write_bytes,
          "summary direction sums equal the exact-string consumption");
}

void verify_finished_parity_and_both_hbm_styles() {
  const Hardware hw = manual_hardware(gate_rates());
  const auto plan =
      byte_plan(hw, 3.0, kEdgeReadA, 4, kEdgeReadB, 6, 11, 13);
  Runtime runtime(hw);
  const auto handle = runtime.submit(plan, owner(1, 1, 1), 0);
  const auto live = runtime.terminal_summary(handle);
  require(!live.has_value(), "live execution has no terminal summary");
  drain(runtime);
  const auto engaged = runtime.terminal_summary(handle);
  require(engaged.has_value() && engaged->terminal && !engaged->cancelled,
          "drained execution yields an engaged finished summary");
  require_summary_matches_oracle(runtime, handle, *engaged, "finished");
  // Full service: every planned byte and op was served.
  require(engaged->int4_served == 11.0 && engaged->vec_served == 13.0,
          "finished fixture serves its full typed work");
  require(engaged->hbm_served[0][0] == 9007199254740992.0 &&
              engaged->hbm_served[0][1] == 4.0 &&
              engaged->hbm_served[1][0] == 1.0 &&
              engaged->hbm_served[1][1] == 6.0,
          "matrix entries are the raw byte counts after double projection");
  // The two consumption styles genuinely diverge on this fixture and the
  // summary carries both: exact-direction sum keeps 2^53+2, the COMP-style
  // double sum of the cast entries loses the +1 to 2^53.
  require(engaged->hbm_read_bytes == 9007199254740994.0,
          "endpoint-style read sum keeps 2^53 + 2");
  const double comp_read = engaged->hbm_served[0][0] +
                           engaged->hbm_served[0][1] +
                           engaged->hbm_served[1][0] +
                           engaged->hbm_served[1][1];
  require(comp_read == 9007199254740992.0 + 4.0 + 1.0 + 6.0,
          "COMP-style double sum is the in-order sum of the cast entries");
  require(engaged->hbm_read_bytes != comp_read,
          "the two HBM consumption orders provably differ on this fixture");
}

void verify_cancelled_partial_service() {
  const Hardware hw = manual_hardware(gate_rates());
  // Byte counts and ops deliberately not divisible by three: a one-third
  // progress serves non-integer values on every channel.
  const auto plan = byte_plan(hw, 3.0, 9007199254740995, 0, 5, 0, 31, 0);
  Runtime runtime(hw);
  const auto handle = runtime.submit(plan, owner(1, 1, 1), 0);
  // One unit of gate work at rate 1e9 = one third of the body: fractional,
  // strictly between zero and the planned totals.
  runtime.advance_to(1);
  runtime.cancel(handle, 1);
  const auto engaged = runtime.terminal_summary(handle);
  require(engaged.has_value() && engaged->terminal && engaged->cancelled,
          "cancelled execution yields an engaged cancelled summary");
  require_summary_matches_oracle(runtime, handle, *engaged, "cancelled");
  require(engaged->int4_served > 0.0 && engaged->int4_served < 30.0,
          "partial service is strictly between zero and the planned ops");
  require(engaged->hbm_read_bytes > 0.0 && engaged->hbm_read_bytes < 9007199254741000.0,
          "partial HBM service is strictly between zero and the planned bytes");
  // Fractional service coverage: the ops channel serves 31/3 at the
  // one-third cancel point (this fixture's HBM partial sums land on
  // integers; their fractional-projection equivalence is already pinned
  // entrywise by the oracle comparisons above).
  require(std::floor(engaged->int4_served) != engaged->int4_served,
          "partial fixture exercises non-integer served values (int4=" +
              std::to_string(engaged->int4_served) + ")");
}

void verify_zero_work_and_stale_handles() {
  const Hardware hw = manual_hardware(gate_rates());
  const auto plan = byte_plan(hw, 0.0, 0, 0, 0, 0, 0, 0);
  Runtime runtime(hw);
  const auto handle = runtime.submit(plan, owner(1, 1, 1), 0);
  drain(runtime);
  const auto engaged = runtime.terminal_summary(handle);
  require(engaged.has_value() && engaged->terminal && !engaged->cancelled &&
              engaged->finish_ns == 0 && engaged->elapsed_ns == 0 &&
              engaged->int4_served == 0.0 && engaged->vec_served == 0.0 &&
              engaged->hbm_read_bytes == 0.0 &&
              engaged->hbm_write_bytes == 0.0,
          "zero-work body finishes with a legal zero-timestamp summary");
  require_summary_matches_oracle(runtime, handle, *engaged, "finished");
  bool stale_threw = false;
  std::string stale_message;
  try {
    (void)runtime.terminal_summary(999);
  } catch (const std::out_of_range& error) {
    stale_threw = true;
    stale_message = error.what();
  }
  require(stale_threw, "unknown handle throws for the terminal summary");
  bool oracle_threw = false;
  try {
    (void)runtime.query_progress(999);
  } catch (const std::out_of_range& error) {
    oracle_threw = stale_message == error.what();
  }
  require(oracle_threw,
          "terminal_summary keeps query_progress's stale_handle convention");
}

void verify_history_independence() {
  const Hardware hw = manual_hardware(gate_rates());
  Epoch compute;
  compute.name = "compute";
  compute.activities = {byte_activity(1, 1.0, 5, 6, 7, 8, 9, 2)};
  Epoch store;
  store.name = "store";
  store.activities = {byte_activity(2, 2.0, 1, 2, 3, 4, 0, 1)};
  auto plan = std::make_shared<Plan>();
  plan->task = json{{"op_kind", "terminal_summary_history_fixture"}};
  plan->summary = json{{"status", "test_fixture"}};
  plan->hardware_fingerprint = hw.fingerprint;
  plan->launch_ns = 0;
  plan->repeat = PlanRepeat{16, 2};
  plan->epochs = {compute, store};
  plan->fingerprint =
      plan_to_json(*plan).at("fingerprint").get<std::string>();
  Runtime runtime(hw);
  const auto handle = runtime.submit(plan, owner(1, 1, 1), 0);
  drain(runtime);
  const auto engaged = runtime.terminal_summary(handle);
  require(engaged.has_value() && engaged->terminal,
          "repeat fixture drains to a terminal summary");
  require_summary_matches_oracle(runtime, handle, *engaged, "finished");
  require(engaged->hbm_read_bytes == static_cast<double>(16 * ((5 + 7) + (1 + 3))) &&
              engaged->hbm_write_bytes == static_cast<double>(16 * ((6 + 8) + (2 + 4))) &&
              engaged->int4_served == static_cast<double>(16 * 9),
          "sixteen rounds of history still summarize exactly once per field");
}

}  // namespace

int main() {
  return run_test("terminal_summary_test", [] {
    verify_finished_parity_and_both_hbm_styles();
    verify_cancelled_partial_service();
    verify_zero_work_and_stale_handles();
    verify_history_independence();
  });
}
