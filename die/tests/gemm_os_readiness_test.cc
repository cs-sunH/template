#include "test_support.hh"
#include "../src/GemmOSReadiness.hh"

#include <array>
#include <iostream>

using namespace coarse_die;
using namespace coarse_die_test;
namespace ready = coarse_die::detail::gemm_os_readiness;
namespace geom = coarse_die::detail::gemm_os_geometry;
namespace access = coarse_die::detail::gemm_os_access_counter;

static Hardware hardware() {
  auto config = read_json(std::string(COARSE_DIE_SOURCE_DIR) +
                          "/configs/example_hardware.json");
  config["architecture"]["core_columns_nx"] = 1;
  config["architecture"]["core_rows_ny"] = 1;
  config["architecture"]["die_geometry"] = "isotropic";
  return resolve_hardware(config);
}

static void finish(ready::Cursor& c, Role role) {
  const auto wave = c.active(role);
  require(wave.has_value(), "stage is active before completion");
  c.complete(role, *wave);
  ready::Cursor restored = c;
  restored.restore(c.state());
  require(restored.state() == c.state(), "active cursor restores exactly");
}

int main() {
  const auto hw = hardware();
  geom::Input input;
  input.M = 3; input.N = 1; input.K = 2;
  input.tm = 1; input.tn = 1; input.tk = 1; input.core_count = 1;
  input.matrix_path = access::MatrixPath::StrictFp32Vec;
  ready::Descriptor d(hw, input);
  ready::Cursor c(d);
  // Reach output occupancy two using ordinary admissions/completions. Keep
  // Store 0 active while output 1 is computed and queued behind it.
  for (unsigned output = 0; output < 2; ++output) {
    for (unsigned k = 0; k < 2; ++k) {
      require(c.admit(Role::Load).has_value(), "load next K");
      require(!c.admit(Role::Compute), "Compute waits for full Load terminal");
      finish(c, Role::Load);
      require(c.admit(Role::Compute).has_value(), "Compute after Load terminal");
      require(!c.admit(Role::Store), "no Store before final Compute terminal");
      finish(c, Role::Compute);
    }
    if (output == 0) require(c.admit(Role::Store).has_value(), "start Store 0");
  }
  require(c.output_occupancy() == 2, "two pending outputs reached");
  require(c.admit(Role::Load).has_value(), "load output 2 nonfinal K");
  finish(c, Role::Load);
  require(c.admit(Role::Compute) == std::optional<ready::u64>(4),
          "nonfinal Compute starts with both output slots occupied");
  require(c.output_occupancy() == 2, "nonfinal does not reserve an output");
  require(c.admit(Role::Load) == std::optional<ready::u64>(5), "prefetch final K");
  require(!c.admit(Role::Load), "no third input reservation");
  finish(c, Role::Load);
  finish(c, Role::Compute);
  require(!c.admit(Role::Compute), "final Compute waits for output space");
  const auto before = c.state();
  bool rejected = false;
  try { c.complete(Role::Store, 3); } catch (const std::invalid_argument&) { rejected = true; }
  require(rejected && c.state() == before, "stale completion is atomic");
  auto invalid = before; invalid.store_completed = invalid.store_started + 1;
  rejected = false;
  try { c.restore(invalid); } catch (const std::invalid_argument&) { rejected = true; }
  require(rejected && c.state() == before, "invalid restore is atomic");
  const std::array<ready::State, 8> invalid_states{{
      {0,0,1,0,0,0},  // Compute before Load terminal.
      {0,0,0,0,1,0},  // Store before final Compute terminal.
      {3,3,0,0,0,0},  // Three held input slots.
      {6,6,6,6,0,0},  // Three held output slots.
      {2,0,0,0,0,0},  // Two active Loads.
      {2,2,2,0,0,0},  // Two active Computes.
      {4,4,4,4,2,0},  // Two active Stores.
      {7,6,6,6,3,3},  // Cursor beyond the geometry.
  }};
  for (const auto& bad : invalid_states) {
    rejected = false;
    try { c.restore(bad); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && c.state() == before, "invalid dependency/slot restore is atomic");
  }
  finish(c, Role::Store);
  require(c.output_occupancy() == 1, "Store terminal releases exactly one output");
  require(c.admit(Role::Compute) == std::optional<ready::u64>(5), "final now starts");
  require(c.admit(Role::Store) == std::optional<ready::u64>(3), "Store 1 was ready");
  finish(c, Role::Compute);
  finish(c, Role::Store);
  require(c.admit(Role::Store) == std::optional<ready::u64>(5), "Store final output");
  finish(c, Role::Store);
  require(c.done() && c.input_occupancy() == 0 && c.output_occupancy() == 0,
          "all reservations drained");

  std::size_t checked = 0;
  for (auto shape : {std::array<ready::u64,3>{80,128,4096}, {80,128,4097},
                    {65,65,4097}, {1,1,1}, {3,5,7}}) {
    input.M = shape[0]; input.N = shape[1]; input.K = shape[2];
    input.tm = 16; input.tn = 64; input.tk = 64;
    ready::Descriptor descriptor(hw, input);
    ready::Cursor cursor(descriptor);
    ready::u64 final_stores = 0;
    ready::u64 read_bytes = 0, write_bytes = 0, vec_ops = 0;
    while (!cursor.done()) {
      for (auto role : {Role::Load, Role::Compute, Role::Store}) cursor.admit(role);
      bool progressed = false;
      for (auto role : {Role::Store, Role::Compute, Role::Load}) {
        if (const auto active = cursor.active(role)) {
          const auto token = descriptor.token(*active);
          const auto& fact = descriptor.facts(*active)[
              role == Role::Load ? 0 : (role == Role::Compute ? 1 : 2)];
          for (const auto& row : fact.hbm_bytes) {
            read_bytes += row[0]; write_bytes += row[1];
          }
          vec_ops += fact.vec_eqops;
          if (role == Role::Store) {
            require(token.final_k, "only final-K Store exists");
            ++final_stores;
          }
          finish(cursor, role); progressed = true;
        }
      }
      require(progressed, "no readiness deadlock");
      require(cursor.input_occupancy() <= 2 && cursor.output_occupancy() <= 2,
              "slot bounds across full/tail geometry");
    }
    require(final_stores == descriptor.output_count(), "exactly one Store per output");
    require(cursor.state().load_completed == descriptor.wave_count() &&
            cursor.state().compute_completed == descriptor.wave_count(),
            "all ordered K tokens consumed exactly once");
    const auto& ledger = descriptor.geometry().total_ledger();
    require(read_bytes == ledger.hbm_read_bytes && write_bytes == ledger.hbm_write_bytes &&
            vec_ops == ledger.vec_equivalent_ops.checked_u64_whole(),
            "selected class facts conserve geometry payload/strict matrix work");
    ++checked;
  }
  // Class construction and cursor size stay fixed for huge K/output repeats.
  input.M = 16000001; input.N = 129; input.K = 64000001;
  ready::Descriptor huge(hw, input);
  require(huge.geometry().class_count() <= 12 && sizeof(ready::Cursor) <= 64,
          "fixed class table/cursor independent of repeat count");
  ready::Cursor huge_cursor(huge);
  auto end = huge_cursor.state();
  end.load_started = end.load_completed = huge.wave_count();
  end.compute_started = end.compute_completed = huge.wave_count();
  end.store_started = end.store_completed = huge.output_count();
  huge_cursor.restore(end);
  require(huge_cursor.done(), "large exact cursor ordinals restore without expansion");
  auto unsupported = input;
  unsupported.core_count = 2;
  rejected = false;
  try { ready::Descriptor invalid_descriptor(hw, unsupported); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "multi-Core descriptor inputs are rejected");
  rejected = false;
  try { ready::Descriptor invalid_descriptor(example_hardware(), input); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "multi-Core hardware cannot masquerade as one-Core facts");
  input.M = std::numeric_limits<ready::u64>::max(); input.N = 2; input.K = 1;
  input.tm = input.tn = input.tk = 1;
  rejected = false;
  try { ready::Descriptor invalid_descriptor(hw, input); }
  catch (const std::overflow_error& error) {
    rejected = std::string(error.what()).find("readiness wave count") != std::string::npos;
  }
  require(rejected, "wide geometry cannot truncate to a uint64 cursor ordinal");
  std::cout << "PASS: final-only output reservations, atomic cursor restore, "
            << checked << " ordered full/tail lifecycles; bounded metadata only\n";
}
