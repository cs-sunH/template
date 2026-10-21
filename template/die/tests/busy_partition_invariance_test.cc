// P2 canonical busy semantics: busy counters are exact dyadic sums of the
// per-event busy windows, so their public projections cannot depend on how
// the event grid partitions those windows (order, granularity, or which
// build produced them). Historical context: 13 estimate busy micro-diffs
// between two engine builds (8.5e-17..4.5e-15 ns, /7-type fractions) were
// long-double accumulation-order artifacts; this contract removes the class.
#include "test_support.hh"
#include <array>
#include <cstdio>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {

void require_same_json(const DyadicBusySum& a, const DyadicBusySum& b,
                       const char* what) {
  if (a.to_json().dump() != b.to_json().dump()) {
    std::printf("FAIL %s: %s vs %s\n", what,
                a.to_json().dump().c_str(), b.to_json().dump().c_str());
    throw std::runtime_error(std::string("busy partition invariance violated: ") + what);
  }
}

void require_value(const DyadicBusySum& sum, long double expected, const char* what) {
  const long double got = sum.value();
  if (got != expected) {
    std::printf("FAIL %s: %.25Lf vs %.25Lf\n", what, got, expected);
    throw std::runtime_error(std::string("busy exact value wrong: ") + what);
  }
}

// 1. Accumulator-level partition invariance.
void accumulator_partition_invariance() {
  // (a) k pieces vs 2k exact sub-splits of the same windows.
  const std::vector<long double> pieces = {0.1L, 800.0L / 7.0L, 1e-3L, 7.7L,
                                           1e9L, 3.65L, 0.5L, 2.25L};
  DyadicBusySum coarse_sum, fine_sum;
  for (long double piece : pieces) {
    coarse_sum.add(piece);
    fine_sum.add(piece / 2);   // exact halving: same total, different partition
    fine_sum.add(piece / 2);
  }
  require_same_json(coarse_sum, fine_sum, "k vs 2k partitions");
  // (b) order permutations.
  DyadicBusySum forward, reverse;
  for (long double piece : pieces) forward.add(piece);
  for (std::size_t i = pieces.size(); i-- > 0;) reverse.add(pieces[i]);
  require_same_json(forward, reverse, "order permutation");
  // (c) exact dyadic total is representable and correct.
  DyadicBusySum exact_sum;
  exact_sum.add(0.5L);
  exact_sum.add(0.25L);
  exact_sum.add(0.125L);
  require_value(exact_sum, 0.875L, "dyadic total");
  // (d) a tiny delta survives next to a huge one (long double would swallow).
  DyadicBusySum wide;
  wide.add(1e18L);
  const DyadicBusySum before = wide;
  wide.add(1e-9L);
  if (wide.to_json().dump() == before.to_json().dump())
    throw std::runtime_error("busy accumulator lost a sub-ulp-scale delta");
  // (e) wire round-trip identity.
  DyadicBusySum parsed = DyadicBusySum::from_json(wide.to_json());
  require_same_json(wide, parsed, "wire round trip");
  if (parsed.value() != wide.value())
    throw std::runtime_error("busy wire round trip changed the value");
}

// 2. Engine level: fork-continuation busy identity. The snapshot carries the
// exact dyadic accumulator, so a runtime that continues from an imported
// snapshot must publish busy counters identical to the un-imported run --
// including the last decimal digit.
void engine_fork_continuation_identity() {
  std::array<double, resource_count> capacities{};
  capacities[COMPUTE_GATE] = 7;   // /7-style fractional event grid
  capacities[HBM0] = capacities[HBM1] = 1024;
  const Hardware hw = manual_hardware(capacities);
  auto build = [&]() {
    Epoch epoch;
    epoch.name = "fractional_grid";
    const long double work = 1000;
    auto load = activity(1, Role::Load, HBM0, 1024);
    load.hbm_bytes[0][0] = 1024;
    load.credit_delta = {1, 0};
    auto compute = activity(2, Role::Compute, COMPUTE_GATE, 1000);
    compute.vec_eqops = 1000;
    compute.credit_delta = {-1, 1};
    auto store = activity(3, Role::Store, HBM1, 1024);
    store.hbm_bytes[1][1] = 1024;
    store.credit_delta = {0, -1};
    epoch.activities = {load, compute, store};
    epoch.credit_capacity = {2, 2};
    epoch.credit_initial = {1, 1};
    return manual_plan(hw, {epoch}, "busy-invariance-fixture");
  };

  Runtime direct(hw);
  const auto plan = build();
  const std::uint64_t handle = direct.submit(plan, owner(1, 1, 1), 0);
  (void)handle;
  direct.advance_to(500000000ULL);  // mid-run split with fractional events pending
  const nlohmann::json saved = direct.export_snapshot();
  auto fork = Runtime::import_snapshot(hw, saved);
  direct.advance_to(2000000000ULL);
  fork->advance_to(2000000000ULL);
  if (direct.export_snapshot() != fork->export_snapshot())
    throw std::runtime_error("busy fork continuation snapshot diverged");
  const nlohmann::json a = direct.counters();
  const nlohmann::json b = fork->counters();
  for (const char* field : {"compute_busy_ns", "compute_busy_ns_exact",
                            "hbm_busy_ns", "hbm_busy_ns_exact"})
    if (a.at(field).dump() != b.at(field).dump())
      throw std::runtime_error(std::string("busy fork counters diverged: ") + field);

  // 3. Third run without any snapshot interaction must agree as well.
  Runtime fresh(hw);
  fresh.submit(plan, owner(1, 1, 1), 0);
  fresh.advance_to(2000000000ULL);
  const nlohmann::json c = fresh.counters();
  for (const char* field : {"compute_busy_ns_exact", "hbm_busy_ns_exact"})
    if (a.at(field).dump() != c.at(field).dump())
      throw std::runtime_error(std::string("busy direct-vs-fresh diverged: ") + field);
}

}  // namespace

int main() {
  accumulator_partition_invariance();
  engine_fork_continuation_identity();
  std::printf("busy_partition_invariance_test passed\n");
  return 0;
}
