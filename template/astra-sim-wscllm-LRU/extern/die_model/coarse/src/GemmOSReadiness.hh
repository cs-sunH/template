#pragma once

// Bounded single-Core OS facts and discrete readiness state. This is not yet
// a production Runtime/Plan variant: no timing, cycle skip, or alternate
// allocator lives here. The compact counters replace per-K token containers;
// the existing geometry and resource builders remain the source of work.

#include "GemmOSGeometry.hh"
#include "GemmOSResourceFacts.hh"

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace coarse_die::detail::gemm_os_readiness {

namespace geometry = gemm_os_geometry;
namespace access = gemm_os_access_counter;
using u64 = std::uint64_t;

class Descriptor {
 public:
  Descriptor(const Hardware& hardware, const geometry::Input& input)
      : geometry_(input) {
    if (input.core_count != 1)
      throw std::invalid_argument("unsupported_readiness: single-Core OS only");
    waves_ = access::detail::checked_u64(geometry_.consumption_wave_count(),
                                       "readiness wave count");
    outputs_ = access::detail::checked_u64(geometry_.output_tile_count(),
                                         "readiness output count");
    for (std::size_t i = 0; i < geometry_.class_count(); ++i) {
      const auto& c = geometry_.access_class(i);
      access::Input one;
      one.m = c.m; one.n = c.n; one.k = c.k;
      one.first_k = c.first_k; one.final_k = c.final_k;
      one.a_storage = input.a_storage; one.w_storage = input.w_storage;
      one.output_storage = input.output_storage;
      one.matrix_path = input.matrix_path;
      one.kappa_quant = input.kappa_quant; one.kappa_acc = input.kappa_acc;
      facts_[i] = gemm_os_single_core_resource_facts(hardware, one);
    }
  }

  const geometry::GemmOSGeometry& geometry() const noexcept { return geometry_; }
  u64 wave_count() const noexcept { return waves_; }
  u64 output_count() const noexcept { return outputs_; }
  geometry::WaveFact token(u64 wave) const {
    if (wave >= waves_) throw std::out_of_range("OS readiness wave ordinal");
    return geometry_.wave_fact(wave / geometry_.k_tiles(),
                               wave % geometry_.k_tiles());
  }
  const std::array<Activity, 3>& facts(u64 wave) const {
    return facts_[token(wave).class_index];
  }

 private:
  geometry::GemmOSGeometry geometry_;
  std::array<std::array<Activity, 3>, geometry::kMaxAccessClasses> facts_{};
  u64 waves_ = 0, outputs_ = 0;
};

struct State {
  u64 load_started = 0, load_completed = 0;
  u64 compute_started = 0, compute_completed = 0;
  u64 store_started = 0, store_completed = 0;

  bool operator==(const State& other) const noexcept {
    return load_started == other.load_started &&
           load_completed == other.load_completed &&
           compute_started == other.compute_started &&
           compute_completed == other.compute_completed &&
           store_started == other.store_started &&
           store_completed == other.store_completed;
  }
};

// Ordered one-Load/one-Compute/one-Store execution makes each ready FIFO an
// interval of ordinals. Six monotone counters describe both queues and all
// reservations, regardless of M/N/K. Input slots are held from Load admission
// through Compute terminal; output slots from final Compute admission through
// Store terminal. Nonfinal Compute never reserves an output slot. The caller
// keeps the immutable Descriptor alive for the entire Cursor lifetime.
class Cursor {
 public:
  explicit Cursor(const Descriptor& descriptor) : descriptor_(&descriptor) {}
  const State& state() const noexcept { return state_; }
  u64 input_occupancy() const noexcept {
    return state_.load_started - state_.compute_completed;
  }
  u64 output_occupancy() const noexcept {
    return state_.compute_started / descriptor_->geometry().k_tiles() -
           state_.store_completed;
  }
  bool done() const noexcept {
    return state_.store_completed == descriptor_->output_count();
  }

  std::optional<u64> active(Role role) const {
    switch (role) {
      case Role::Load:
        if (state_.load_started != state_.load_completed)
          return state_.load_completed;
        break;
      case Role::Compute:
        if (state_.compute_started != state_.compute_completed)
          return state_.compute_completed;
        break;
      case Role::Store:
        if (state_.store_started != state_.store_completed)
          return (state_.store_completed + 1) *
                     descriptor_->geometry().k_tiles() - 1;
        break;
      default: throw std::invalid_argument("OS readiness has no Endpoint role");
    }
    return std::nullopt;
  }

  std::optional<u64> admit(Role role) {
    if (active(role)) return std::nullopt;
    switch (role) {
      case Role::Load:
        if (state_.load_started == descriptor_->wave_count() ||
            input_occupancy() == 2) return std::nullopt;
        return state_.load_started++;
      case Role::Compute: {
        if (state_.compute_started == state_.load_completed) return std::nullopt;
        const auto wave = state_.compute_started;
        if (descriptor_->token(wave).final_k && output_occupancy() == 2)
          return std::nullopt;
        ++state_.compute_started;
        return wave;
      }
      case Role::Store:
        if (state_.store_started == state_.compute_completed /
                                           descriptor_->geometry().k_tiles())
          return std::nullopt;
        ++state_.store_started;
        return state_.store_started * descriptor_->geometry().k_tiles() - 1;
      default: throw std::invalid_argument("OS readiness has no Endpoint role");
    }
  }

  void complete(Role role, u64 wave) {
    if (active(role) != std::optional<u64>(wave))
      throw std::invalid_argument("invalid_readiness: stale stage completion");
    switch (role) {
      case Role::Load: ++state_.load_completed; break;
      case Role::Compute: ++state_.compute_completed; break;
      case Role::Store: ++state_.store_completed; break;
      default: throw std::invalid_argument("OS readiness has no Endpoint role");
    }
  }

  // Restore is atomic. These inequalities also establish that every ordinal
  // used by active()/admit() is in range and every reservation is backed by a
  // started stage. Runtime progress/owner/hardware validation is separate.
  void restore(const State& value) {
    const auto kt = descriptor_->geometry().k_tiles();
    if (value.load_started > descriptor_->wave_count() ||
        value.load_completed > value.load_started ||
        value.compute_started > value.load_completed ||
        value.compute_completed > value.compute_started ||
        value.store_started > value.compute_completed / kt ||
        value.store_completed > value.store_started ||
        value.load_started - value.load_completed > 1 ||
        value.compute_started - value.compute_completed > 1 ||
        value.store_started - value.store_completed > 1 ||
        value.load_started - value.compute_completed > 2 ||
        value.compute_started / kt - value.store_completed > 2)
      throw std::invalid_argument("invalid_readiness: cursor dependency/slot state");
    state_ = value;
  }

 private:
  const Descriptor* descriptor_;
  State state_{};
};

}  // namespace coarse_die::detail::gemm_os_readiness
