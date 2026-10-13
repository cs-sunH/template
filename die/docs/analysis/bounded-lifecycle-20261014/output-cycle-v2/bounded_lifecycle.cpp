// Scratch-only bounded K-wave lifecycle prototype. No production code changes.
// Build: g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic bounded_lifecycle.cpp -lgmpxx -lgmp -o bounded_lifecycle

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <gmpxx.h>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {
bool g_trace_outer = false;

struct Rat {
  mpz_class n = 0;
  mpz_class d = 1;

  Rat() = default;
  Rat(std::int64_t value) : n(static_cast<long>(value)), d(1) {}
  Rat(std::int64_t numerator, std::int64_t denominator)
      : n(static_cast<long>(numerator)), d(static_cast<long>(denominator)) {
    normalize();
  }
  Rat(mpz_class numerator, mpz_class denominator)
      : n(std::move(numerator)), d(std::move(denominator)) {
    normalize();
  }

  void normalize() {
    if (d == 0) throw std::runtime_error("zero rational denominator");
    if (d < 0) {
      d = -d;
      n = -n;
    }
    if (n == 0) {
      d = 1;
      return;
    }
    mpz_class g;
    mpz_gcd(g.get_mpz_t(), n.get_mpz_t(), d.get_mpz_t());
    n /= g;
    d /= g;
  }

  std::string str() const {
    if (d == 1) return n.get_str();
    return n.get_str() + "/" + d.get_str();
  }

  long double as_long_double() const {
    return static_cast<long double>(mpz_get_d(n.get_mpz_t())) /
           static_cast<long double>(mpz_get_d(d.get_mpz_t()));
  }

  bool zero() const { return n == 0; }
  bool positive() const { return n > 0; }

  friend Rat operator+(const Rat& a, const Rat& b) {
    mpz_class g;
    mpz_gcd(g.get_mpz_t(), a.d.get_mpz_t(), b.d.get_mpz_t());
    const mpz_class am = b.d / g;
    const mpz_class bm = a.d / g;
    return Rat(a.n * am + b.n * bm, a.d * am);
  }
  friend Rat operator-(const Rat& a, const Rat& b) {
    mpz_class g;
    mpz_gcd(g.get_mpz_t(), a.d.get_mpz_t(), b.d.get_mpz_t());
    const mpz_class am = b.d / g;
    const mpz_class bm = a.d / g;
    return Rat(a.n * am - b.n * bm, a.d * am);
  }
  friend Rat operator*(const Rat& a, const Rat& b) {
    mpz_class g1, g2;
    mpz_gcd(g1.get_mpz_t(), a.n.get_mpz_t(), b.d.get_mpz_t());
    mpz_gcd(g2.get_mpz_t(), b.n.get_mpz_t(), a.d.get_mpz_t());
    return Rat((a.n / g1) * (b.n / g2), (a.d / g2) * (b.d / g1));
  }
  friend Rat operator/(const Rat& a, const Rat& b) {
    if (b.n == 0) throw std::runtime_error("rational division by zero");
    return a * Rat(b.d, b.n);
  }
  Rat& operator+=(const Rat& rhs) { return *this = *this + rhs; }
  Rat& operator-=(const Rat& rhs) { return *this = *this - rhs; }
  Rat& operator*=(const Rat& rhs) { return *this = *this * rhs; }
  Rat& operator/=(const Rat& rhs) { return *this = *this / rhs; }

  friend bool operator==(const Rat& a, const Rat& b) {
    return a.n == b.n && a.d == b.d;
  }
  friend bool operator!=(const Rat& a, const Rat& b) { return !(a == b); }
  friend bool operator<(const Rat& a, const Rat& b) {
    return a.n * b.d < b.n * a.d;
  }
  friend bool operator>(const Rat& a, const Rat& b) { return b < a; }
  friend bool operator<=(const Rat& a, const Rat& b) { return !(b < a); }
  friend bool operator>=(const Rat& a, const Rat& b) { return !(a < b); }
};

std::int64_t floor_nonnegative(const Rat& r) {
  if (r.n < 0) throw std::runtime_error("negative value passed to floor");
  const mpz_class value = r.n / r.d;
  const mpz_class max(static_cast<long>(std::numeric_limits<std::int64_t>::max()));
  if (value > max)
    throw std::overflow_error("cycle count exceeds int64");
  return static_cast<std::int64_t>(value.get_si());
}

std::uint64_t bit_length(const mpz_class& value) {
  const mpz_class magnitude = value < 0 ? -value : value;
  return magnitude == 0 ? 0 : static_cast<std::uint64_t>(
      mpz_sizeinbase(magnitude.get_mpz_t(), 2));
}

#ifndef FIXTURE_M
#define FIXTURE_M 65
#endif
#ifndef FIXTURE_N
#define FIXTURE_N 65
#endif
#ifndef FIXTURE_K
#define FIXTURE_K 4097
#endif
constexpr std::int64_t kM = FIXTURE_M;
constexpr std::int64_t kN = FIXTURE_N;
constexpr std::int64_t kK = FIXTURE_K;
constexpr std::int64_t kTm = 16;
constexpr std::int64_t kTn = 64;
constexpr std::int64_t kTk = 64;
constexpr std::int64_t kTmCount = (kM + kTm - 1) / kTm;
constexpr std::int64_t kTnCount = (kN + kTn - 1) / kTn;
constexpr std::int64_t kTkCount = (kK + kTk - 1) / kTk;
constexpr std::int64_t kOutputs = kTmCount * kTnCount;
constexpr std::int64_t kWaves = kOutputs * kTkCount;
constexpr std::int64_t kComputeDenom = 7910800000LL * 2 / 100;  // P_vec * eta_vec = 158216000.
constexpr std::int64_t kMaxProbes = 256;

static_assert(kTmCount > 0 && kTnCount > 0 && kTkCount > 0,
              "fixture tile geometry changed");
static_assert(kOutputs > 0 && kWaves > 0, "fixture wave counts changed");

enum Resource : std::size_t { kCompute = 0, kHbm0 = 1, kHbm1 = 2, kResourceCount = 3 };
enum Stage : std::size_t { kLoad = 0, kComp = 1, kStore = 2, kStageCount = 3 };
enum class Policy { kEqualProgress, kGroupedQ1, kGroupedQLocal };

const char* policy_name(Policy policy) {
  if (policy == Policy::kEqualProgress) return "equal_progress";
  return policy == Policy::kGroupedQ1 ? "grouped_q_alpha_q1" : "grouped_q_alpha_qlocal";
}

bool uses_grouped_q(Policy policy) {
  return policy != Policy::kEqualProgress;
}

struct Wave {
  std::int64_t output = 0;
  std::int64_t k_index = 0;
  std::int64_t m_elems = 0;
  std::int64_t n_elems = 0;
  std::int64_t k_elems = 0;
  std::array<std::int64_t, 2> read_bytes{{0, 0}};
  std::array<std::int64_t, 2> write_bytes{{0, 0}};
  std::int64_t vec_ops = 0;
  std::int64_t local_bytes = 0;
  bool final_k = false;
  int category = 0;  // 0..7: M-tail, N-tail, K-tail bits.
};

std::int64_t row0_partition(std::int64_t total, std::int64_t prefix) {
  return (prefix + total + 1) / 2 - (prefix + 1) / 2;
}

Wave make_wave(std::int64_t index) {
  if (index < 0 || index >= kWaves) throw std::out_of_range("wave index");
  Wave w;
  w.output = index / kTkCount;
  w.k_index = index % kTkCount;
  const std::int64_t mt = w.output / kTnCount;
  const std::int64_t nt = w.output % kTnCount;
  w.m_elems = std::min(kTm, kM - mt * kTm);
  w.n_elems = std::min(kTn, kN - nt * kTn);
  w.k_elems = std::min(kTk, kK - w.k_index * kTk);
  w.final_k = w.k_index + 1 == kTkCount;
  const std::int64_t read = 4 * w.k_elems * (w.m_elems + w.n_elems);
  const std::int64_t write = w.final_k ? 4 * w.m_elems * w.n_elems : 0;
  w.vec_ops = 2 * w.m_elems * w.n_elems * w.k_elems;
  const std::int64_t accumulator = 4 * w.m_elems * w.n_elems;
  w.local_bytes = read + accumulator + (w.k_index > 0 ? accumulator : 0);

  // Closed-prefix form of the independent per-wave oracle's ordered row split.
  const std::int64_t n_before = nt * kTn;
  const std::int64_t area_before_outputs =
      mt * (kTn * kTm + kN) + nt * (w.m_elems + kTn);
  const std::int64_t read_prefix =
      4 * kK * area_before_outputs +
      4 * (w.m_elems + w.n_elems) * (w.k_index * kTk);
  const std::int64_t write_prefix =
      4 * (mt * kTm * kN + w.m_elems * n_before);
  w.read_bytes[0] = row0_partition(read, read_prefix);
  w.read_bytes[1] = read - w.read_bytes[0];
  w.write_bytes[0] = row0_partition(write, write_prefix);
  w.write_bytes[1] = write - w.write_bytes[0];
  const int mtail = w.m_elems < kTm ? 1 : 0;
  const int ntail = w.n_elems < kTn ? 1 : 0;
  const int ktail = w.k_elems < kTk ? 1 : 0;
  w.category = mtail * 4 + ntail * 2 + ktail;
  return w;
}

Rat compute_work(const Wave& wave) {
  return Rat(wave.vec_ops, kComputeDenom);
}

std::int64_t payload(const Wave& wave, Stage stage) {
  if (stage == kLoad) return wave.read_bytes[0] + wave.read_bytes[1];
  if (stage == kStore) return wave.write_bytes[0] + wave.write_bytes[1];
  return 0;
}

struct Activity {
  int owner = -1;
  Stage stage = kLoad;
  std::int64_t wave_index = -1;
  Rat progress;
  std::array<Rat, kResourceCount> work{};
  Rat private_time;
  Rat q;
  std::int64_t local_bytes = 0;
};

Activity make_activity(int owner, Stage stage, std::int64_t wave_index, Policy policy) {
  const Wave wave = make_wave(wave_index);
  Activity activity;
  activity.owner = owner;
  activity.stage = stage;
  activity.wave_index = wave_index;
  if (stage == kLoad) {
    activity.work[kHbm0] = Rat(wave.read_bytes[0]);
    activity.work[kHbm1] = Rat(wave.read_bytes[1]);
    activity.q = Rat(payload(wave, stage));
  } else if (stage == kComp) {
    activity.work[kCompute] = compute_work(wave);
    activity.private_time = compute_work(wave);
    activity.local_bytes = wave.local_bytes;
    activity.q = policy == Policy::kGroupedQLocal ? Rat(wave.local_bytes) : Rat(1);
  } else {
    activity.work[kHbm0] = Rat(wave.write_bytes[0]);
    activity.work[kHbm1] = Rat(wave.write_bytes[1]);
    activity.q = Rat(payload(wave, stage));
  }
  if (!activity.q.positive()) throw std::runtime_error("activity canonical Q must be positive");
  return activity;
}

struct LiveActivity {
  bool active = false;
  Activity activity;
};

struct OwnerMetrics {
  std::int64_t read_bytes = 0;
  std::int64_t write_bytes = 0;
  std::int64_t vec_ops = 0;
  std::int64_t local_bytes = 0;
  std::int64_t load_done = 0;
  std::int64_t compute_done = 0;
  std::int64_t store_done = 0;
  struct Class {
    std::int64_t read_bytes = 0;
    std::int64_t write_bytes = 0;
    std::int64_t vec_ops = 0;
    std::int64_t local_bytes = 0;
    std::int64_t load_done = 0;
    std::int64_t compute_done = 0;
    std::int64_t store_done = 0;
    std::int64_t retired = 0;
  };
  std::array<Class, 8> by_class{};
};

struct Counters {
  Rat hbm_busy;
  Rat hbm_shared;
  Rat compute_busy;
  Rat compute_shared;
  std::array<Rat, 2> row_busy{};
  std::array<Rat, 2> row_shared{};
  std::array<Rat, kResourceCount> served_work{};
};

struct Owner {
  bool started = false;
  bool finished = false;
  std::int64_t next_load = 0;
  std::array<std::int64_t, 2> loaded{{-1, -1}};
  int loaded_count = 0;
  std::array<std::int64_t, 2> computed{{-1, -1}};
  int computed_count = 0;
  std::array<LiveActivity, kStageCount> active{};
  std::int64_t retired_outputs = 0;
  std::optional<Rat> finish;
  OwnerMetrics metrics;
};

struct World {
  Policy policy = Policy::kEqualProgress;
  Rat now;
  std::array<Rat, 2> arrivals{};
  std::array<Owner, 2> owners{};
  int owner_count = 1;
  Counters counters;
  std::int64_t intervals = 0;
  std::int64_t transitions = 0;
  std::int64_t skipped_cycles = 0;
  std::int64_t inner_skipped_cycles = 0;
  std::int64_t outer_skipped_cycles = 0;
  std::int64_t inner_skipped_waves = 0;
  std::int64_t outer_skipped_outputs = 0;
  std::int64_t outer_exact_cycles = 0;
  std::int64_t outer_cached_cycle_hits = 0;
  std::int64_t outer_unskippable_cycle_hits = 0;
  std::int64_t outer_rejected_key_repeats = 0;
  std::int64_t outer_cycle_output_step = 0;
  std::int64_t max_probe_steps = 0;
  std::int64_t inner_probe_steps = 0;
  std::int64_t outer_probe_steps = 0;
  std::int64_t max_inner_probe_steps = 0;
  std::int64_t max_outer_probe_steps = 0;
  std::uint64_t max_progress_num_bits = 0;
  std::uint64_t max_progress_den_bits = 0;
  std::uint64_t max_rate_num_bits = 0;
  std::uint64_t max_rate_den_bits = 0;
  std::vector<std::string> outer_trace;
  std::uint64_t capacity_generation = 0;
  bool fail_closed = false;
  std::string fail_reason;
};

struct ActivityRef {
  int owner = 0;
  Stage stage = kLoad;
  Activity* activity = nullptr;
};

struct Allocation {
  std::array<Rat, 6> rates{};
  std::array<ActivityRef, 6> refs{};
  int count = 0;
  std::vector<std::string> basis;
};

Allocation allocate(World& world) {
  Allocation result;
  for (int owner = 0; owner < world.owner_count; ++owner) {
    if (world.owners[owner].finished) continue;
    for (Stage stage : {kLoad, kComp, kStore}) {
      LiveActivity& live = world.owners[owner].active[stage];
      if (!live.active) continue;
      if (result.count >= 6) throw std::runtime_error("active activity bound exceeded");
      result.refs[result.count++] = ActivityRef{owner, stage, &live.activity};
    }
  }
  if (result.count == 0) return result;

  std::array<bool, 6> free{};
  std::array<Rat, 2> owner_q{};
  for (int i = 0; i < result.count; ++i) {
    free[i] = true;
    if (uses_grouped_q(world.policy))
      owner_q[result.refs[i].owner] += result.refs[i].activity->q;
  }

  const std::array<Rat, kResourceCount> capacity{{Rat(1), Rat(50000000), Rat(50000000)}};
  int remaining = result.count;
  for (int tier = 0; tier <= result.count && remaining > 0; ++tier) {
    std::optional<Rat> step;
    std::set<std::size_t> limiting_resources;
    std::set<int> private_limiter;
    std::array<Rat, kResourceCount> used{};
    for (int i = 0; i < result.count; ++i) {
      for (std::size_t r = 0; r < kResourceCount; ++r)
        used[r] += result.refs[i].activity->work[r] * result.rates[i];
    }
    for (std::size_t r = 0; r < kResourceCount; ++r) {
      Rat slope;
      for (int i = 0; i < result.count; ++i) {
        if (!free[i]) continue;
        Rat multiplier(1);
        if (uses_grouped_q(world.policy))
          multiplier /= owner_q[result.refs[i].owner];
        slope += result.refs[i].activity->work[r] * multiplier;
      }
      if (!slope.positive()) continue;
      const Rat available = capacity[r] - used[r];
      if (available < Rat(0)) throw std::runtime_error("allocator starts over capacity");
      const Rat candidate = available / slope;
      if (!step || candidate < *step) {
        step = candidate;
        limiting_resources = {r};
        private_limiter.clear();
      } else if (candidate == *step) {
        limiting_resources.insert(r);
      }
    }
    for (int i = 0; i < result.count; ++i) {
      if (!free[i]) continue;
      const Rat private_time = result.refs[i].activity->private_time;
      if (!private_time.positive()) continue;
      const Rat max_rate = Rat(1) / private_time;
      Rat multiplier(1);
      if (uses_grouped_q(world.policy))
        multiplier = owner_q[result.refs[i].owner];
      const Rat candidate = (max_rate - result.rates[i]) * multiplier;
      if (!step || candidate < *step) {
        step = candidate;
        limiting_resources.clear();
        private_limiter = {i};
      } else if (candidate == *step) {
        private_limiter.insert(i);
      }
    }
    if (!step) throw std::runtime_error("allocator has no finite resource tier");
    if (*step < Rat(0)) throw std::runtime_error("allocator found negative rate increment");
    for (int i = 0; i < result.count; ++i) {
      if (!free[i]) continue;
      Rat multiplier(1);
      if (uses_grouped_q(world.policy))
        multiplier /= owner_q[result.refs[i].owner];
      result.rates[i] += *step * multiplier;
    }

    std::set<int> freeze = private_limiter;
    std::string tier_name;
    for (std::size_t r : limiting_resources) {
      if (!tier_name.empty()) tier_name += "+";
      tier_name += (r == kCompute ? "COMPUTE_GATE" :
                    r == kHbm0 ? "HBM0" : "HBM1");
      for (int i = 0; i < result.count; ++i) {
        if (free[i] && result.refs[i].activity->work[r].positive()) freeze.insert(i);
      }
    }
    if (!private_limiter.empty()) {
      if (!tier_name.empty()) tier_name += "+";
      tier_name += "PRIVATE";
    }
    if (freeze.empty()) throw std::runtime_error("allocator tier froze no activity");
    result.basis.push_back(tier_name);
    for (int i : freeze) {
      if (free[i]) {
        free[i] = false;
        --remaining;
      }
    }
  }
  if (remaining != 0) throw std::runtime_error("allocator exceeded fixed tier count");
  for (std::size_t r = 0; r < kResourceCount; ++r) {
    Rat used;
    for (int i = 0; i < result.count; ++i)
      used += result.refs[i].activity->work[r] * result.rates[i];
    if (used > capacity[r]) throw std::runtime_error("allocator exceeded resource capacity");
  }
  return result;
}

void observe_sizes(World& world, const Allocation& allocation) {
  for (int i = 0; i < allocation.count; ++i) {
    const Activity& activity = *allocation.refs[i].activity;
    const Rat& rate = allocation.rates[i];
    world.max_progress_num_bits = std::max(
        world.max_progress_num_bits, bit_length(activity.progress.n));
    world.max_progress_den_bits = std::max(
        world.max_progress_den_bits, bit_length(activity.progress.d));
    world.max_rate_num_bits = std::max(
        world.max_rate_num_bits, bit_length(rate.n));
    world.max_rate_den_bits = std::max(
        world.max_rate_den_bits, bit_length(rate.d));
  }
}

int input_occupancy(const Owner& owner) {
  return owner.loaded_count + static_cast<int>(owner.active[kLoad].active) +
         static_cast<int>(owner.active[kComp].active);
}

int output_occupancy(const Owner& owner) {
  int final_compute = 0;
  if (owner.active[kComp].active && make_wave(owner.active[kComp].activity.wave_index).final_k)
    final_compute = 1;
  return owner.computed_count + static_cast<int>(owner.active[kStore].active) + final_compute;
}

void push_bounded(std::array<std::int64_t, 2>& queue, int& count, std::int64_t value) {
  if (count >= 2) throw std::runtime_error("bounded slot queue overflow");
  queue[count++] = value;
}

std::int64_t pop_bounded(std::array<std::int64_t, 2>& queue, int& count) {
  if (count <= 0) throw std::runtime_error("bounded slot queue underflow");
  const std::int64_t value = queue[0];
  if (count == 2) queue[0] = queue[1];
  queue[1] = -1;
  --count;
  return value;
}

void start_stage(Owner& owner, Stage stage, int owner_id, std::int64_t wave_index,
                 Policy policy) {
  LiveActivity& live = owner.active[stage];
  if (live.active) throw std::runtime_error("stage already active");
  live.active = true;
  live.activity = make_activity(owner_id, stage, wave_index, policy);
}

void settle(World& world) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (int oi = 0; oi < world.owner_count; ++oi) {
      Owner& owner = world.owners[oi];
      if (!owner.started || owner.finished) continue;
      if (owner.active[kLoad].active && owner.active[kLoad].activity.progress == Rat(1)) {
        const std::int64_t wave = owner.active[kLoad].activity.wave_index;
        const Wave w = make_wave(wave);
        owner.active[kLoad] = LiveActivity{};
        push_bounded(owner.loaded, owner.loaded_count, wave);
        ++owner.metrics.load_done;
        const std::int64_t bytes = payload(w, kLoad);
        owner.metrics.read_bytes += bytes;
        auto& class_metrics = owner.metrics.by_class[w.category];
        ++class_metrics.load_done;
        class_metrics.read_bytes += bytes;
        changed = true;
      }
    }
    for (int oi = 0; oi < world.owner_count; ++oi) {
      Owner& owner = world.owners[oi];
      if (!owner.started || owner.finished) continue;
      if (owner.active[kComp].active && owner.active[kComp].activity.progress == Rat(1)) {
        const std::int64_t wave_index = owner.active[kComp].activity.wave_index;
        const Wave wave = make_wave(wave_index);
        owner.active[kComp] = LiveActivity{};
        ++owner.metrics.compute_done;
        owner.metrics.vec_ops += wave.vec_ops;
        owner.metrics.local_bytes += wave.local_bytes;
        auto& class_metrics = owner.metrics.by_class[wave.category];
        ++class_metrics.compute_done;
        class_metrics.vec_ops += wave.vec_ops;
        class_metrics.local_bytes += wave.local_bytes;
        if (wave.final_k) push_bounded(owner.computed, owner.computed_count, wave_index);
        changed = true;
      }
    }
    for (int oi = 0; oi < world.owner_count; ++oi) {
      Owner& owner = world.owners[oi];
      if (!owner.started || owner.finished) continue;
      if (owner.active[kStore].active && owner.active[kStore].activity.progress == Rat(1)) {
        const std::int64_t wave = owner.active[kStore].activity.wave_index;
        const Wave w = make_wave(wave);
        owner.active[kStore] = LiveActivity{};
        ++owner.metrics.store_done;
        const std::int64_t bytes = payload(w, kStore);
        owner.metrics.write_bytes += bytes;
        auto& class_metrics = owner.metrics.by_class[w.category];
        ++class_metrics.store_done;
        ++class_metrics.retired;
        class_metrics.write_bytes += bytes;
        ++owner.retired_outputs;
        changed = true;
      }
    }
    for (int oi = 0; oi < world.owner_count; ++oi) {
      Owner& owner = world.owners[oi];
      if (!owner.started || owner.finished) continue;
      if (!owner.active[kLoad].active && owner.next_load < kWaves && input_occupancy(owner) < 2) {
        const std::int64_t wave = owner.next_load++;
        start_stage(owner, kLoad, oi, wave, world.policy);
        changed = true;
      }
      if (!owner.active[kComp].active && owner.loaded_count > 0 && output_occupancy(owner) < 2) {
        const std::int64_t wave = pop_bounded(owner.loaded, owner.loaded_count);
        start_stage(owner, kComp, oi, wave, world.policy);
        changed = true;
      }
      if (!owner.active[kStore].active && owner.computed_count > 0) {
        const std::int64_t wave = pop_bounded(owner.computed, owner.computed_count);
        start_stage(owner, kStore, oi, wave, world.policy);
        changed = true;
      }
      if (owner.retired_outputs == kOutputs && !owner.finish) {
        owner.finished = true;
        owner.finish = world.now;
        changed = true;
      }
      const int active_count = static_cast<int>(owner.active[kLoad].active) +
                               static_cast<int>(owner.active[kComp].active) +
                               static_cast<int>(owner.active[kStore].active);
      if (active_count > 3) throw std::runtime_error("per-owner activity bound exceeded");
    }
  }
}

bool all_finished(const World& world) {
  for (int i = 0; i < world.owner_count; ++i)
    if (!world.owners[i].finished) return false;
  return true;
}

struct IntervalMasks {
  std::array<std::set<int>, 2> hbm_owners;
  std::set<int> compute_owners;
};

IntervalMasks masks_for(const Allocation& allocation) {
  IntervalMasks masks;
  for (int i = 0; i < allocation.count; ++i) {
    const Activity& a = *allocation.refs[i].activity;
    if (!allocation.rates[i].positive()) continue;
    if (a.work[kHbm0].positive()) masks.hbm_owners[0].insert(a.owner);
    if (a.work[kHbm1].positive()) masks.hbm_owners[1].insert(a.owner);
    if (a.work[kCompute].positive()) masks.compute_owners.insert(a.owner);
  }
  return masks;
}

void integrate(World& world, const Allocation& allocation, const Rat& dt) {
  const IntervalMasks masks = masks_for(allocation);
  if (!masks.hbm_owners[0].empty() || !masks.hbm_owners[1].empty())
    world.counters.hbm_busy += dt;
  if (masks.compute_owners.size() > 0) world.counters.compute_busy += dt;
  bool shared_hbm = false;
  for (int row = 0; row < 2; ++row) {
    if (!masks.hbm_owners[row].empty()) world.counters.row_busy[row] += dt;
    if (masks.hbm_owners[row].size() >= 2) {
      world.counters.row_shared[row] += dt;
      shared_hbm = true;
    }
  }
  if (shared_hbm) world.counters.hbm_shared += dt;
  if (masks.compute_owners.size() >= 2) world.counters.compute_shared += dt;

  for (int i = 0; i < allocation.count; ++i) {
    Activity& activity = *allocation.refs[i].activity;
    const Rat rate = allocation.rates[i];
    activity.progress += rate * dt;
    for (std::size_t r = 0; r < kResourceCount; ++r)
      world.counters.served_work[r] += activity.work[r] * rate * dt;
  }
}

std::optional<Rat> next_arrival(const World& world) {
  std::optional<Rat> result;
  for (int i = 0; i < world.owner_count; ++i) {
    if (world.owners[i].started) continue;
    if (!result || world.arrivals[i] < *result) result = world.arrivals[i];
  }
  return result;
}

bool admit_arrivals(World& world) {
  bool changed = false;
  for (int i = 0; i < world.owner_count; ++i) {
    Owner& owner = world.owners[i];
    if (!owner.started && world.arrivals[i] <= world.now) {
      owner.started = true;
      changed = true;
    }
  }
  return changed;
}

std::string wave_signature(std::int64_t wave_index) {
  if (wave_index < 0 || wave_index >= kWaves) return "end";
  const Wave w = make_wave(wave_index);
  std::ostringstream out;
  out << w.category << ':' << w.m_elems << ':' << w.n_elems << ':' << w.k_elems
      << ':' << w.read_bytes[0] << ':' << w.read_bytes[1]
      << ':' << w.write_bytes[0] << ':' << w.write_bytes[1]
      << ':' << w.vec_ops << ':' << w.local_bytes << ':' << w.final_k;
  return out.str();
}

std::string inner_segment_signature(const World& world) {
  std::ostringstream out;
  for (int i = 0; i < world.owner_count; ++i) {
    const Owner& owner = world.owners[i];
    if (!owner.started) out << "U;";
    else if (owner.finished || owner.next_load >= kWaves) out << "E;";
    else out << (owner.next_load / kTkCount) << ';';
  }
  return out.str();
}

std::string state_key(World& world, const Allocation& allocation) {
  std::ostringstream out;
  out << "bounded-lifecycle-v2|" << policy_name(world.policy) << "|g"
      << world.capacity_generation << '|';
  for (int i = 0; i < world.owner_count; ++i) {
    const Owner& owner = world.owners[i];
    out << "o" << i << ':' << owner.started << ':' << owner.finished << ':';
    out << "next=" << wave_signature(owner.next_load) << ':';
    out << "li" << owner.loaded_count << '[';
    for (int q = 0; q < owner.loaded_count; ++q)
      out << wave_signature(owner.loaded[q]) << ',';
    out << "]co" << owner.computed_count << '[';
    for (int q = 0; q < owner.computed_count; ++q)
      out << wave_signature(owner.computed[q]) << ',';
    out << ']';
    for (Stage stage : {kLoad, kComp, kStore}) {
      const LiveActivity& live = owner.active[stage];
      out << "s" << static_cast<int>(stage) << '=' << live.active;
      if (live.active) {
        out << ',' << wave_signature(live.activity.wave_index)
            << ',' << live.activity.progress.str();
      }
      out << ';';
    }
  }
  const IntervalMasks masks = masks_for(allocation);
  out << "mask=" << !masks.hbm_owners[0].empty() << !masks.hbm_owners[1].empty()
      << !masks.compute_owners.empty() << ':';
  for (int row = 0; row < 2; ++row) {
    out << '[';
    for (int owner : masks.hbm_owners[row]) out << owner;
    out << ']';
  }
  out << '[';
  for (int owner : masks.compute_owners) out << owner;
  out << "]rates=";
  for (int i = 0; i < allocation.count; ++i) {
    const auto& ref = allocation.refs[i];
    out << ref.owner << '.' << static_cast<int>(ref.stage) << '='
        << allocation.rates[i].str() << "/q=" << ref.activity->q.str()
        << "/private=" << ref.activity->private_time.str() << "/work=";
    for (std::size_t r = 0; r < kResourceCount; ++r)
      out << ref.activity->work[r].str() << ',';
    out << ';';
  }
  out << "basis=";
  for (const std::string& tier : allocation.basis) out << tier << ',';
  return out.str();
}

struct Snapshot {
  Rat now;
  Counters counters;
  std::array<Owner, 2> owner_state{};
  std::array<OwnerMetrics, 2> owner_metrics{};
  std::array<std::int64_t, 2> next_load{{0, 0}};
  std::array<std::int64_t, 2> retired{{0, 0}};
};

Snapshot snapshot(const World& world) {
  Snapshot s;
  s.now = world.now;
  s.counters = world.counters;
  for (int i = 0; i < world.owner_count; ++i) {
    s.owner_state[i] = world.owners[i];
    s.owner_metrics[i] = world.owners[i].metrics;
    s.next_load[i] = world.owners[i].next_load;
    s.retired[i] = world.owners[i].retired_outputs;
  }
  return s;
}

struct CycleDelta {
  Rat period;
  Counters counters;
  std::array<OwnerMetrics, 2> owner_metrics{};
  std::array<std::int64_t, 2> next_load{{0, 0}};
  std::array<std::int64_t, 2> retired{{0, 0}};
};

OwnerMetrics operator-(const OwnerMetrics& a, const OwnerMetrics& b) {
  OwnerMetrics out;
  out.read_bytes = a.read_bytes - b.read_bytes;
  out.write_bytes = a.write_bytes - b.write_bytes;
  out.vec_ops = a.vec_ops - b.vec_ops;
  out.local_bytes = a.local_bytes - b.local_bytes;
  out.load_done = a.load_done - b.load_done;
  out.compute_done = a.compute_done - b.compute_done;
  out.store_done = a.store_done - b.store_done;
  for (std::size_t c = 0; c < out.by_class.size(); ++c) {
    const auto& x = a.by_class[c];
    const auto& y = b.by_class[c];
    auto& z = out.by_class[c];
    z.read_bytes = x.read_bytes - y.read_bytes;
    z.write_bytes = x.write_bytes - y.write_bytes;
    z.vec_ops = x.vec_ops - y.vec_ops;
    z.local_bytes = x.local_bytes - y.local_bytes;
    z.load_done = x.load_done - y.load_done;
    z.compute_done = x.compute_done - y.compute_done;
    z.store_done = x.store_done - y.store_done;
    z.retired = x.retired - y.retired;
  }
  return out;
}

Counters operator-(const Counters& a, const Counters& b) {
  Counters out;
  out.hbm_busy = a.hbm_busy - b.hbm_busy;
  out.hbm_shared = a.hbm_shared - b.hbm_shared;
  out.compute_busy = a.compute_busy - b.compute_busy;
  out.compute_shared = a.compute_shared - b.compute_shared;
  for (int i = 0; i < 2; ++i) {
    out.row_busy[i] = a.row_busy[i] - b.row_busy[i];
    out.row_shared[i] = a.row_shared[i] - b.row_shared[i];
  }
  for (std::size_t i = 0; i < kResourceCount; ++i)
    out.served_work[i] = a.served_work[i] - b.served_work[i];
  return out;
}

CycleDelta cycle_delta(const Snapshot& old, const World& world) {
  CycleDelta d;
  d.period = world.now - old.now;
  d.counters = world.counters - old.counters;
  for (int i = 0; i < world.owner_count; ++i) {
    d.owner_metrics[i] = world.owners[i].metrics - old.owner_metrics[i];
    d.next_load[i] = world.owners[i].next_load - old.next_load[i];
    d.retired[i] = world.owners[i].retired_outputs - old.retired[i];
  }
  return d;
}

void add_scaled(Counters& target, const Counters& delta, std::int64_t cycles) {
  const Rat n(cycles);
  target.hbm_busy += delta.hbm_busy * n;
  target.hbm_shared += delta.hbm_shared * n;
  target.compute_busy += delta.compute_busy * n;
  target.compute_shared += delta.compute_shared * n;
  for (int i = 0; i < 2; ++i) {
    target.row_busy[i] += delta.row_busy[i] * n;
    target.row_shared[i] += delta.row_shared[i] * n;
  }
  for (std::size_t i = 0; i < kResourceCount; ++i)
    target.served_work[i] += delta.served_work[i] * n;
}

bool same_translated_wave_window(const Owner& old_owner,
                                 const Owner& now_owner,
                                 std::int64_t shift) {
  if (shift <= 0 || shift % kTkCount != 0) return false;
  std::int64_t first = old_owner.next_load;
  auto include = [&](std::int64_t wave) { if (wave >= 0) first = std::min(first, wave); };
  for (Stage stage : {kLoad, kComp, kStore})
    if (old_owner.active[stage].active) include(old_owner.active[stage].activity.wave_index);
  for (int q = 0; q < old_owner.loaded_count; ++q) include(old_owner.loaded[q]);
  for (int q = 0; q < old_owner.computed_count; ++q) include(old_owner.computed[q]);
  const std::int64_t end = old_owner.next_load + shift;
  // `shift` comes from at most 256 observed transitions, so this verification
  // is bounded independently of the request's output count.
  for (std::int64_t wave = first; wave < end; ++wave) {
    if (wave < 0 || wave + shift >= kWaves) return false;
    if (wave_signature(wave) != wave_signature(wave + shift)) return false;
  }
  for (Stage stage : {kLoad, kComp, kStore}) {
    const LiveActivity& before = old_owner.active[stage];
    const LiveActivity& now = now_owner.active[stage];
    if (before.active != now.active) return false;
    if (before.active && now.activity.wave_index - before.activity.wave_index != shift)
      return false;
  }
  auto queues_match = [shift](const std::array<std::int64_t, 2>& before, int before_count,
                             const std::array<std::int64_t, 2>& now, int now_count) {
    if (before_count != now_count) return false;
    for (int q = 0; q < before_count; ++q)
      if (now[q] - before[q] != shift) return false;
    return true;
  };
  return queues_match(old_owner.loaded, old_owner.loaded_count,
                      now_owner.loaded, now_owner.loaded_count) &&
         queues_match(old_owner.computed, old_owner.computed_count,
                      now_owner.computed, now_owner.computed_count);
}

bool is_whole_output_cycle(const Snapshot& old, const World& now,
                           const CycleDelta& delta) {
  if (delta.period <= Rat(0)) return false;
  for (int i = 0; i < now.owner_count; ++i) {
    const Owner& a = old.owner_state[i];
    const Owner& b = now.owners[i];
    const std::int64_t shift = delta.next_load[i];
    if (!a.started && !b.started) {
      if (shift != 0 || delta.retired[i] != 0) return false;
      continue;
    }
    if (!a.started || !b.started || a.finished || b.finished || shift <= 0 ||
        shift % kTkCount != 0 || !same_translated_wave_window(a, b, shift))
      return false;
    const std::int64_t outputs = shift / kTkCount;
    const OwnerMetrics& m = delta.owner_metrics[i];
    if (delta.retired[i] != outputs || m.load_done != shift ||
        m.compute_done != shift || m.store_done != outputs)
      return false;
    // Exact class deltas must agree with the full output cycle. This includes
    // the K-final completion, store, and retirement inside each period.
    for (const auto& c : m.by_class) {
      if (c.load_done < 0 || c.compute_done < 0 || c.store_done < 0 ||
          c.retired < 0 || c.store_done != c.retired)
        return false;
    }
    std::int64_t class_loads = 0, class_computes = 0, class_stores = 0;
    for (const auto& c : m.by_class) {
      class_loads += c.load_done;
      class_computes += c.compute_done;
      class_stores += c.store_done;
    }
    if (class_loads != shift || class_computes != shift || class_stores != outputs)
      return false;
  }
  return true;
}

bool is_inner_full_k_cycle(const Snapshot& old, const World& now,
                           const CycleDelta& delta) {
  if (delta.period <= Rat(0)) return false;
  for (int i = 0; i < now.owner_count; ++i) {
    const Owner& a = old.owner_state[i];
    const Owner& b = now.owners[i];
    const std::int64_t shift = delta.next_load[i];
    if (!a.started && !b.started) {
      if (shift != 0 || delta.retired[i] != 0) return false;
      continue;
    }
    if (!a.started || !b.started || a.finished || b.finished || shift <= 0 ||
        a.next_load >= kWaves || b.next_load >= kWaves ||
        a.next_load / kTkCount != b.next_load / kTkCount ||
        a.next_load % kTkCount >= kTkCount - 1 ||
        b.next_load % kTkCount >= kTkCount - 1)
      return false;
    const std::int64_t output = a.next_load / kTkCount;
    if (b.next_load - a.next_load != shift || delta.retired[i] != 0 ||
        delta.owner_metrics[i].store_done != 0 ||
        delta.owner_metrics[i].write_bytes != 0 ||
        delta.owner_metrics[i].load_done != shift ||
        delta.owner_metrics[i].compute_done != shift)
      return false;
    for (Stage stage : {kLoad, kComp, kStore}) {
      const LiveActivity& before = a.active[stage];
      const LiveActivity& after = b.active[stage];
      if (before.active != after.active) return false;
      if (!before.active) continue;
      if (before.activity.wave_index / kTkCount != output ||
          after.activity.wave_index - before.activity.wave_index != shift ||
          before.activity.wave_index % kTkCount >= kTkCount - 1 ||
          after.activity.wave_index % kTkCount >= kTkCount - 1)
        return false;
    }
    auto queue_ok = [output, shift](const std::array<std::int64_t, 2>& before,
                                    int before_count,
                                    const std::array<std::int64_t, 2>& after,
                                    int after_count) {
      if (before_count != after_count) return false;
      for (int q = 0; q < before_count; ++q)
        if (before[q] / kTkCount != output || after[q] - before[q] != shift ||
            before[q] % kTkCount >= kTkCount - 1 ||
            after[q] % kTkCount >= kTkCount - 1)
          return false;
      return true;
    };
    if (!queue_ok(a.loaded, a.loaded_count, b.loaded, b.loaded_count) ||
        !queue_ok(a.computed, a.computed_count, b.computed, b.computed_count))
      return false;
    const std::int64_t end = a.next_load + shift;
    for (std::int64_t wave = a.next_load; wave < end; ++wave) {
      if (wave >= kWaves || wave / kTkCount != output ||
          wave % kTkCount >= kTkCount - 1 || wave + shift >= kWaves ||
          wave_signature(wave) != wave_signature(wave + shift))
        return false;
    }
    for (const auto& c : delta.owner_metrics[i].by_class)
      if (c.store_done != 0 || c.retired != 0 || c.write_bytes != 0) return false;
  }
  return true;
}

std::int64_t max_inner_k_repetitions(const World& world,
                                     const CycleDelta& delta) {
  std::int64_t allowed = std::numeric_limits<std::int64_t>::max();
  bool any = false;
  for (int i = 0; i < world.owner_count; ++i) {
    const Owner& owner = world.owners[i];
    const std::int64_t shift = delta.next_load[i];
    if (shift == 0) continue;
    if (shift < 0 || owner.finished || owner.next_load >= kWaves) return 0;
    any = true;
    const std::int64_t output = owner.next_load / kTkCount;
    const std::int64_t last_full = output * kTkCount + (kTkCount - 2);
    if (owner.next_load % kTkCount >= kTkCount - 1) return 0;
    std::int64_t furthest = owner.next_load;
    auto include = [&](std::int64_t wave) {
      if (wave < 0 || wave >= kWaves || wave / kTkCount != output ||
          wave % kTkCount >= kTkCount - 1)
        furthest = last_full + 1;
      else
        furthest = std::max(furthest, wave);
    };
    for (Stage stage : {kLoad, kComp, kStore})
      if (owner.active[stage].active) include(owner.active[stage].activity.wave_index);
    for (int q = 0; q < owner.loaded_count; ++q) include(owner.loaded[q]);
    for (int q = 0; q < owner.computed_count; ++q) include(owner.computed[q]);
    if (furthest > last_full) return 0;
    allowed = std::min(allowed, (last_full - furthest) / shift);
  }
  return any ? std::max<std::int64_t>(0, allowed) : 0;
}

std::int64_t max_output_cycle_repetitions(const World& world,
                                          const CycleDelta& delta) {
  std::int64_t allowed = std::numeric_limits<std::int64_t>::max();
  bool any = false;
  const std::int64_t full_m_outputs = (kM / kTm) * kTnCount;
  for (int i = 0; i < world.owner_count; ++i) {
    const Owner& owner = world.owners[i];
    const std::int64_t shift = delta.next_load[i];
    if (shift == 0) continue;
    if (shift < 0 || shift % kTkCount != 0) return 0;
    any = true;
    const std::int64_t output_step = shift / kTkCount;
    if (kN % kTn != 0 && output_step % kTnCount != 0) return 0;

    // Do not cross an M-full/M-tail boundary or request end. The N-full/N-tail
    // sequence may recur inside a full M row, but only if the shift preserves
    // the whole row pattern. K-tail transitions occur within each observed
    // output cycle and are represented by exact per-class counter deltas.
    std::int64_t furthest = owner.next_load;
    std::int64_t earliest = owner.next_load;
    for (Stage stage : {kLoad, kComp, kStore})
      if (owner.active[stage].active) {
        furthest = std::max(furthest, owner.active[stage].activity.wave_index);
        earliest = std::min(earliest, owner.active[stage].activity.wave_index);
      }
    for (int q = 0; q < owner.loaded_count; ++q) {
      furthest = std::max(furthest, owner.loaded[q]);
      earliest = std::min(earliest, owner.loaded[q]);
    }
    for (int q = 0; q < owner.computed_count; ++q) {
      furthest = std::max(furthest, owner.computed[q]);
      earliest = std::min(earliest, owner.computed[q]);
    }
    if (earliest >= kWaves || furthest >= kWaves) return 0;
    const bool m_tail = (make_wave(earliest).category & 4) != 0;
    for (std::int64_t wave = earliest; wave <= furthest; ++wave)
      if (((make_wave(wave).category & 4) != 0) != m_tail) return 0;
    const std::int64_t class_begin = m_tail ? full_m_outputs : 0;
    const std::int64_t class_end = m_tail ? kOutputs : full_m_outputs;
    if (class_end <= class_begin) return 0;
    const std::int64_t remaining = class_end - owner.retired_outputs;
    if (remaining < output_step) return 0;
    allowed = std::min(allowed, remaining / output_step);
    const std::int64_t furthest_output = furthest / kTkCount;
    if (furthest_output < class_begin || furthest_output >= class_end) return 0;
    const std::int64_t to_boundary = class_end - 1 - furthest_output;
    allowed = std::min(allowed, to_boundary / output_step);
  }
  return any ? std::max<std::int64_t>(0, allowed) : 0;
}

void apply_cycles(World& world, const CycleDelta& delta, std::int64_t cycles) {
  if (cycles <= 0) return;
  world.now += delta.period * Rat(cycles);
  add_scaled(world.counters, delta.counters, cycles);
  for (int i = 0; i < world.owner_count; ++i) {
    Owner& owner = world.owners[i];
    const std::int64_t shift = delta.next_load[i] * cycles;
    owner.next_load += shift;
    owner.retired_outputs += delta.retired[i] * cycles;
    const OwnerMetrics& d = delta.owner_metrics[i];
    owner.metrics.read_bytes += d.read_bytes * cycles;
    owner.metrics.write_bytes += d.write_bytes * cycles;
    owner.metrics.vec_ops += d.vec_ops * cycles;
    owner.metrics.local_bytes += d.local_bytes * cycles;
    owner.metrics.load_done += d.load_done * cycles;
    owner.metrics.compute_done += d.compute_done * cycles;
    owner.metrics.store_done += d.store_done * cycles;
    for (std::size_t c = 0; c < owner.metrics.by_class.size(); ++c) {
      auto& target = owner.metrics.by_class[c];
      const auto& source = d.by_class[c];
      target.read_bytes += source.read_bytes * cycles;
      target.write_bytes += source.write_bytes * cycles;
      target.vec_ops += source.vec_ops * cycles;
      target.local_bytes += source.local_bytes * cycles;
      target.load_done += source.load_done * cycles;
      target.compute_done += source.compute_done * cycles;
      target.store_done += source.store_done * cycles;
      target.retired += source.retired * cycles;
    }
    for (Stage stage : {kLoad, kComp, kStore})
      if (owner.active[stage].active) owner.active[stage].activity.wave_index += shift;
    for (int j = 0; j < owner.loaded_count; ++j) owner.loaded[j] += shift;
    for (int j = 0; j < owner.computed_count; ++j) owner.computed[j] += shift;
  }
}

Rat interval_length(World& world, const Allocation& allocation) {
  std::optional<Rat> next;
  for (int i = 0; i < allocation.count; ++i) {
    const Rat rate = allocation.rates[i];
    if (!rate.positive()) continue;
    const Rat remain = Rat(1) - allocation.refs[i].activity->progress;
    const Rat dt = remain / rate;
    if (!next || dt < *next) next = dt;
  }
  const auto arrival = next_arrival(world);
  if (arrival) {
    const Rat dt = *arrival - world.now;
    if (dt < Rat(0)) throw std::runtime_error("arrival scheduler moved backwards");
    if (!next || dt < *next) next = dt;
  }
  if (!next) throw std::runtime_error("unfinished world has no advancing stage");
  if (*next <= Rat(0)) throw std::runtime_error("zero-time interval before stage completion");
  return *next;
}

struct RunResult {
  World world;
  std::string status = "completed";
};

void trace_outer_state(World& world, const Allocation& allocation, std::int64_t probe) {
  std::ostringstream out;
  out << "trace probe=" << probe << " now=" << world.now.str();
  for (int owner_id = 0; owner_id < world.owner_count; ++owner_id) {
    const Owner& owner = world.owners[owner_id];
    out << " owner" << owner_id << "_retired=" << owner.retired_outputs
        << " owner" << owner_id << "_next=" << owner.next_load;
    for (Stage stage : {kLoad, kComp, kStore}) {
      const LiveActivity& live = owner.active[stage];
      const char* stage_label = stage == kLoad ? "L" : stage == kComp ? "C" : "S";
      out << " owner" << owner_id << '_' << stage_label << '=';
      if (!live.active) {
        out << "idle";
        continue;
      }
      Rat rate;
      for (int i = 0; i < allocation.count; ++i)
        if (allocation.refs[i].owner == owner_id && allocation.refs[i].stage == stage)
          rate = allocation.rates[i];
      out << live.activity.wave_index << '@' << live.activity.progress.str()
          << '|' << rate.str();
    }
  }
  world.outer_trace.push_back(out.str());
}

std::int64_t cap_at_next_arrival(const World& world, const CycleDelta& delta,
                                 std::int64_t proposed) {
  if (proposed <= 0) return 0;
  const auto arrival = next_arrival(world);
  if (!arrival) return proposed;
  const Rat until_arrival = *arrival - world.now;
  if (until_arrival < Rat(0)) throw std::runtime_error("arrival prefix moved backwards");
  return std::min(proposed, floor_nonnegative(until_arrival / delta.period));
}

RunResult run(Policy policy, const std::vector<Rat>& arrivals,
              bool use_cycle_detector, std::int64_t probe_limit = kMaxProbes) {
  if (arrivals.empty() || arrivals.size() > 2) throw std::invalid_argument("need one or two owners");
  World world;
  world.policy = policy;
  world.owner_count = static_cast<int>(arrivals.size());
  for (int i = 0; i < world.owner_count; ++i) world.arrivals[i] = arrivals[i];

  std::map<std::string, Snapshot> inner_seen;
  std::map<std::string, CycleDelta> inner_certified;
  std::map<std::string, Snapshot> outer_seen;
  std::map<std::string, CycleDelta> outer_certified;
  std::int64_t inner_probes = 0;
  std::int64_t outer_probes = 0;
  std::string inner_segment;
  bool inner_segment_valid = false;
  std::array<std::int64_t, 2> last_outer_retired{{0, 0}};
  const std::int64_t max_events = 1000000;

  while (!all_finished(world)) {
    if (admit_arrivals(world)) {
      inner_seen.clear();
      inner_certified.clear();
      outer_seen.clear();
      outer_certified.clear();
      inner_probes = 0;
      outer_probes = 0;
      world.max_probe_steps = 0;
      world.max_inner_probe_steps = 0;
      world.max_outer_probe_steps = 0;
      inner_segment_valid = false;
      for (int i = 0; i < world.owner_count; ++i)
        last_outer_retired[i] = world.owners[i].retired_outputs;
    }
    settle(world);
    if (all_finished(world)) break;

    Allocation allocation = allocate(world);
    observe_sizes(world, allocation);
    if (use_cycle_detector) {
      const std::string key = state_key(world, allocation);
      bool retired_boundary = false;
      for (int i = 0; i < world.owner_count; ++i) {
        if (world.owners[i].retired_outputs != last_outer_retired[i])
          retired_boundary = true;
        last_outer_retired[i] = world.owners[i].retired_outputs;
      }

      // The outer detector samples only after an output has retired. It has a
      // single 256-boundary budget for the current arrival prefix.
      if (retired_boundary) {
        ++outer_probes;
        world.outer_probe_steps = outer_probes;
        world.max_outer_probe_steps = std::max(world.max_outer_probe_steps, outer_probes);
        world.max_probe_steps = std::max(world.max_probe_steps,
                                         std::max(inner_probes, outer_probes));
        if (g_trace_outer) trace_outer_state(world, allocation, outer_probes);
        if (outer_probes > probe_limit) {
          world.fail_closed = true;
          world.fail_reason = "no_exact_joint_repeat_within_probe_limit";
          return RunResult{std::move(world), "fail_closed"};
        }
        const auto cached = outer_certified.find(key);
        if (cached != outer_certified.end()) {
          ++world.outer_cached_cycle_hits;
          const CycleDelta& delta = cached->second;
          const std::int64_t cycles = cap_at_next_arrival(
              world, delta, max_output_cycle_repetitions(world, delta));
          if (cycles > 0) {
            apply_cycles(world, delta, cycles);
            world.skipped_cycles += cycles;
            world.outer_skipped_cycles += cycles;
            for (int i = 0; i < world.owner_count; ++i)
              world.outer_skipped_outputs += delta.retired[i] * cycles;
            continue;
          }
          ++world.outer_unskippable_cycle_hits;
        } else {
          const auto found = outer_seen.find(key);
          if (found == outer_seen.end()) {
            if (outer_probes >= probe_limit ||
                static_cast<std::int64_t>(outer_seen.size()) >= probe_limit) {
              world.fail_closed = true;
              world.fail_reason = "no_exact_joint_repeat_within_probe_limit";
              return RunResult{std::move(world), "fail_closed"};
            }
            outer_seen.emplace(key, snapshot(world));
          } else {
            const CycleDelta delta = cycle_delta(found->second, world);
            if (is_whole_output_cycle(found->second, world, delta)) {
              ++world.outer_exact_cycles;
              if (world.owner_count > 0 && delta.next_load[0] % kTkCount == 0)
                world.outer_cycle_output_step = std::max(
                    world.outer_cycle_output_step, delta.next_load[0] / kTkCount);
              outer_certified.emplace(key, delta);
              const std::int64_t cycles = cap_at_next_arrival(
                  world, delta, max_output_cycle_repetitions(world, delta));
              if (cycles > 0) {
                apply_cycles(world, delta, cycles);
                world.skipped_cycles += cycles;
                world.outer_skipped_cycles += cycles;
                for (int i = 0; i < world.owner_count; ++i)
                  world.outer_skipped_outputs += delta.retired[i] * cycles;
                continue;
              }
              ++world.outer_unskippable_cycle_hits;
            } else {
              ++world.outer_rejected_key_repeats;
              // Keep the most recent exact phase at this boundary key so a
              // later period cannot accidentally include an earlier class.
              outer_seen[key] = snapshot(world);
            }
            if (outer_probes >= probe_limit) {
              world.fail_closed = true;
              world.fail_reason = "no_exact_joint_repeat_within_probe_limit";
              return RunResult{std::move(world), "fail_closed"};
            }
          }
        }
        if (outer_probes >= probe_limit) {
          world.fail_closed = true;
          world.fail_reason = "no_exact_joint_repeat_within_probe_limit";
          return RunResult{std::move(world), "fail_closed"};
        }
      }

      // Inner K-full detector: reset only when a request crosses an output
      // cursor. The outer 256-retirement cap bounds how many such segments
      // can be explored before either a skip or fail-closed.
      const std::string current_inner_segment = inner_segment_signature(world);
      if (!inner_segment_valid || current_inner_segment != inner_segment) {
        inner_seen.clear();
        inner_certified.clear();
        inner_probes = 0;
        inner_segment = current_inner_segment;
        inner_segment_valid = true;
      }
      const auto inner_cached = inner_certified.find(key);
      if (inner_cached != inner_certified.end()) {
        const CycleDelta& delta = inner_cached->second;
        const std::int64_t cycles = cap_at_next_arrival(
            world, delta, max_inner_k_repetitions(world, delta));
        if (cycles > 0) {
          apply_cycles(world, delta, cycles);
          world.skipped_cycles += cycles;
          world.inner_skipped_cycles += cycles;
          for (int i = 0; i < world.owner_count; ++i)
            world.inner_skipped_waves += delta.next_load[i] * cycles;
          continue;
        }
      } else {
        const auto found = inner_seen.find(key);
        if (found == inner_seen.end()) {
          if (inner_probes >= probe_limit ||
              static_cast<std::int64_t>(inner_seen.size()) >= probe_limit) {
            world.fail_closed = true;
            world.fail_reason = "no_exact_joint_repeat_within_probe_limit";
            return RunResult{std::move(world), "fail_closed"};
          }
          inner_seen.emplace(key, snapshot(world));
        } else {
          const CycleDelta delta = cycle_delta(found->second, world);
          if (is_inner_full_k_cycle(found->second, world, delta)) {
            inner_certified.emplace(key, delta);
            const std::int64_t cycles = cap_at_next_arrival(
                world, delta, max_inner_k_repetitions(world, delta));
            if (cycles > 0) {
              apply_cycles(world, delta, cycles);
              world.skipped_cycles += cycles;
              world.inner_skipped_cycles += cycles;
              for (int i = 0; i < world.owner_count; ++i)
                world.inner_skipped_waves += delta.next_load[i] * cycles;
              continue;
            }
          } else {
            inner_seen[key] = snapshot(world);
          }
        }
      }
    }

    const Rat dt = interval_length(world, allocation);
    integrate(world, allocation, dt);
    world.now += dt;
    ++world.intervals;
    ++world.transitions;
    if (use_cycle_detector) {
      ++inner_probes;
      world.inner_probe_steps = inner_probes;
      world.max_inner_probe_steps = std::max(world.max_inner_probe_steps, inner_probes);
      world.max_probe_steps = std::max(world.max_probe_steps,
                                       std::max(inner_probes, outer_probes));
      if (inner_probes > probe_limit) {
        world.fail_closed = true;
        world.fail_reason = "no_exact_joint_repeat_within_probe_limit";
        return RunResult{std::move(world), "fail_closed"};
      }
    }
    if (world.transitions > max_events)
      throw std::runtime_error("event-by-event path exceeded fixed event budget");
    settle(world);
  }
  return RunResult{std::move(world), "completed"};
}

std::string finish_string(const Owner& owner) {
  return owner.finish ? owner.finish->str() : "unavailable";
}

std::string metrics_line(const char* kind, Policy policy, const char* scenario,
                         const RunResult& result) {
  const World& w = result.world;
  std::ostringstream out;
  out << "record kind=" << kind << " status=" << result.status
      << " policy=" << policy_name(policy) << " scenario=" << scenario
      << " now=" << w.now.str()
      << " finish0=" << finish_string(w.owners[0])
      << " finish1=" << (w.owner_count > 1 ? finish_string(w.owners[1]) : "na")
      << " hbm_busy=" << w.counters.hbm_busy.str()
      << " hbm_shared=" << w.counters.hbm_shared.str()
      << " compute_busy=" << w.counters.compute_busy.str()
      << " compute_shared=" << w.counters.compute_shared.str()
      << " row_busy0=" << w.counters.row_busy[0].str()
      << " row_busy1=" << w.counters.row_busy[1].str()
      << " row_shared0=" << w.counters.row_shared[0].str()
      << " row_shared1=" << w.counters.row_shared[1].str()
      << " served_compute=" << w.counters.served_work[kCompute].str()
      << " served_hbm0=" << w.counters.served_work[kHbm0].str()
      << " served_hbm1=" << w.counters.served_work[kHbm1].str()
      << " intervals=" << w.intervals
      << " transitions=" << w.transitions
      << " executed_events=" << w.transitions
      << " skipped_cycles=" << w.skipped_cycles
      << " inner_skipped_cycles=" << w.inner_skipped_cycles
      << " outer_skipped_cycles=" << w.outer_skipped_cycles
      << " inner_skipped_waves=" << w.inner_skipped_waves
      << " outer_skipped_outputs=" << w.outer_skipped_outputs
      << " outer_exact_cycles=" << w.outer_exact_cycles
      << " outer_cached_cycle_hits=" << w.outer_cached_cycle_hits
      << " outer_unskippable_cycle_hits=" << w.outer_unskippable_cycle_hits
      << " outer_rejected_key_repeats=" << w.outer_rejected_key_repeats
      << " outer_cycle_output_step=" << w.outer_cycle_output_step
      << " max_probes=" << w.max_probe_steps
      << " inner_probes=" << w.inner_probe_steps
      << " outer_probes=" << w.outer_probe_steps
      << " max_inner_probes=" << w.max_inner_probe_steps
      << " max_outer_probes=" << w.max_outer_probe_steps
      << " max_progress_num_bits=" << w.max_progress_num_bits
      << " max_progress_den_bits=" << w.max_progress_den_bits
      << " max_rate_num_bits=" << w.max_rate_num_bits
      << " max_rate_den_bits=" << w.max_rate_den_bits
      << " fail_reason=" << (w.fail_reason.empty() ? "none" : w.fail_reason);
  for (int i = 0; i < w.owner_count; ++i) {
    const Owner& o = w.owners[i];
    out << " owner" << i << "_read=" << o.metrics.read_bytes
        << " owner" << i << "_write=" << o.metrics.write_bytes
        << " owner" << i << "_ops=" << o.metrics.vec_ops
      << " owner" << i << "_local=" << o.metrics.local_bytes
        << " owner" << i << "_load_done=" << o.metrics.load_done
        << " owner" << i << "_compute_done=" << o.metrics.compute_done
        << " owner" << i << "_store_done=" << o.metrics.store_done
        << " owner" << i << "_retired=" << o.retired_outputs
        << " owner" << i << "_next_load=" << o.next_load;
    for (std::size_t c = 0; c < o.metrics.by_class.size(); ++c) {
      const auto& cm = o.metrics.by_class[c];
      out << " owner" << i << "_class" << c << "_read=" << cm.read_bytes
          << " owner" << i << "_class" << c << "_write=" << cm.write_bytes
          << " owner" << i << "_class" << c << "_ops=" << cm.vec_ops
          << " owner" << i << "_class" << c << "_local=" << cm.local_bytes
          << " owner" << i << "_class" << c << "_load=" << cm.load_done
          << " owner" << i << "_class" << c << "_compute=" << cm.compute_done
          << " owner" << i << "_class" << c << "_store=" << cm.store_done
          << " owner" << i << "_class" << c << "_retired=" << cm.retired;
    }
  }
  return out.str();
}

std::int64_t expected_read_bytes() {
  return 4 * kK * (kTnCount * kM + kTmCount * kN);
}

std::int64_t expected_write_bytes() { return 4 * kM * kN; }
std::int64_t expected_ops() { return 2 * kM * kN * kK; }
std::int64_t expected_local_bytes() {
  return expected_read_bytes() + (2 * kTkCount - 1) * 4 * kM * kN;
}

const char* stage_name(Stage stage) {
  return stage == kLoad ? "L" : stage == kComp ? "C" : "S";
}

void emit_grouped_lp_probe(Policy policy) {
  World world;
  world.policy = policy;
  world.owner_count = 2;
  world.owners[0].started = true;
  world.owners[1].started = true;
  // A lifecycle-reachable four-activity shape: owner 0 has S(output 0 final
  // K), C(output 1 K0), and L(output 1 K1); owner 1 has C(output 0 K0).
  // It exercises all three production stages while respecting <=3 active per
  // owner and <=4 total active activities in the independent LP check.
  const std::array<std::tuple<int, Stage, std::int64_t>, 4> descriptors{{
      {0, kStore, 64}, {0, kComp, 65}, {0, kLoad, 66}, {1, kComp, 0}}};
  for (const auto& row : descriptors) {
    const int owner = std::get<0>(row);
    const Stage stage = std::get<1>(row);
    const std::int64_t wave = std::get<2>(row);
    world.owners[owner].active[stage].active = true;
    world.owners[owner].active[stage].activity = make_activity(owner, stage, wave, policy);
  }
  const Allocation allocation = allocate(world);
  const std::string name = policy == Policy::kGroupedQLocal
      ? "three_stage_four_active_qlocal" : "three_stage_four_active_q1";
  std::cout << "lp-case name=" << name << " policy=" << policy_name(policy)
            << " active=" << allocation.count << " basis=";
  for (std::size_t i = 0; i < allocation.basis.size(); ++i) {
    if (i) std::cout << '>';
    std::cout << allocation.basis[i];
  }
  std::cout << '\n';
  for (int i = 0; i < allocation.count; ++i) {
    const auto& ref = allocation.refs[i];
    const Activity& a = *ref.activity;
    std::cout << "lp-activity case=" << name << " owner=" << ref.owner
              << " stage=" << stage_name(ref.stage) << " wave=" << a.wave_index
              << " q=" << a.q.str() << " compute=" << a.work[kCompute].str()
              << " hbm0=" << a.work[kHbm0].str()
              << " hbm1=" << a.work[kHbm1].str()
              << " local=" << a.local_bytes
              << " private=" << a.private_time.str()
              << " rate=" << allocation.rates[i].str() << '\n';
  }
}

bool counters_equal(const Counters& a, const Counters& b) {
  if (a.hbm_busy != b.hbm_busy || a.hbm_shared != b.hbm_shared ||
      a.compute_busy != b.compute_busy || a.compute_shared != b.compute_shared)
    return false;
  for (int i = 0; i < 2; ++i)
    if (a.row_busy[i] != b.row_busy[i] || a.row_shared[i] != b.row_shared[i])
      return false;
  for (std::size_t i = 0; i < kResourceCount; ++i)
    if (a.served_work[i] != b.served_work[i]) return false;
  return true;
}

void emit_scenario(Policy policy, const char* scenario, const std::vector<Rat>& arrivals,
                   bool compare_event = true) {
  std::optional<RunResult> oracle;
  if (compare_event) oracle.emplace(run(policy, arrivals, false));
  const RunResult accelerated = run(policy, arrivals, true, kMaxProbes);
  std::cout << "config policy=" << policy_name(policy) << " scenario=" << scenario
            << " owners=" << arrivals.size() << " m=" << kM << " n=" << kN
            << " k=" << kK << " tm=" << kTm << " tn=" << kTn << " tk=" << kTk
            << " mt_count=" << kTmCount << " nt_count=" << kTnCount
            << " kt_count=" << kTkCount << " outputs=" << kOutputs << " waves=" << kWaves
            << " expected_read=" << expected_read_bytes()
            << " expected_write=" << expected_write_bytes()
            << " expected_local=" << expected_local_bytes()
            << " expected_ops=" << expected_ops()
            << " probe_cap=" << kMaxProbes
            << " event_oracle=" << (compare_event ? "run" : "skipped") << '\n';
  if (oracle) std::cout << metrics_line("event", policy, scenario, *oracle) << '\n';
  std::cout << metrics_line("cycle", policy, scenario, accelerated) << '\n';
  if (g_trace_outer)
    for (const std::string& trace : accelerated.world.outer_trace)
      std::cout << trace << " policy=" << policy_name(policy)
                << " scenario=" << scenario << '\n';

  if (!oracle) {
    std::cout << "check policy=" << policy_name(policy) << " scenario=" << scenario
              << " cycle_equals_event=not_run fail_closed="
              << (accelerated.status == "fail_closed" ? "true" : "false") << '\n';
    return;
  }

  const bool complete_equal = oracle->status == "completed" &&
      accelerated.status == "completed" &&
      oracle->world.now == accelerated.world.now &&
      counters_equal(oracle->world.counters, accelerated.world.counters) &&
      oracle->world.owner_count == accelerated.world.owner_count;
  bool owner_equal = complete_equal;
  if (owner_equal) {
    for (int i = 0; i < oracle->world.owner_count; ++i) {
      const Owner& a = oracle->world.owners[i];
      const Owner& b = accelerated.world.owners[i];
      owner_equal = owner_equal && a.finish == b.finish &&
          a.metrics.read_bytes == b.metrics.read_bytes &&
          a.metrics.write_bytes == b.metrics.write_bytes &&
          a.metrics.vec_ops == b.metrics.vec_ops &&
          a.metrics.local_bytes == b.metrics.local_bytes &&
          a.metrics.load_done == b.metrics.load_done &&
          a.metrics.compute_done == b.metrics.compute_done &&
          a.metrics.store_done == b.metrics.store_done &&
          a.retired_outputs == b.retired_outputs &&
          a.next_load == b.next_load;
      for (std::size_t c = 0; owner_equal && c < a.metrics.by_class.size(); ++c) {
        const auto& x = a.metrics.by_class[c];
        const auto& y = b.metrics.by_class[c];
        owner_equal = x.read_bytes == y.read_bytes && x.write_bytes == y.write_bytes &&
            x.vec_ops == y.vec_ops && x.local_bytes == y.local_bytes && x.load_done == y.load_done &&
            x.compute_done == y.compute_done && x.store_done == y.store_done &&
            x.retired == y.retired;
      }
    }
  }
  std::cout << "check policy=" << policy_name(policy) << " scenario=" << scenario
            << " cycle_equals_event=" << (owner_equal ? "true" : "false")
            << " read_bytes_per_owner=" << expected_read_bytes()
            << " write_bytes_per_owner=" << expected_write_bytes()
            << " local_bytes_per_owner=" << expected_local_bytes()
            << " ops_per_owner=" << expected_ops() << '\n';
}

void check_fail_closed_guard() {
  for (Policy policy : {Policy::kGroupedQ1, Policy::kGroupedQLocal}) {
    const RunResult capped = run(policy, {Rat(0)}, true, 1);
    const bool passed = capped.status == "fail_closed" &&
        capped.world.fail_reason == "no_exact_joint_repeat_within_probe_limit" &&
        capped.world.max_probe_steps == 1;
    std::cout << "guard policy=" << policy_name(policy) << " probe_limit=1 status=" << capped.status
              << " steps=" << capped.world.max_probe_steps
              << " reason=" << (capped.world.fail_reason.empty() ? "none" : capped.world.fail_reason)
              << " fail_closed=" << (passed ? "true" : "false") << '\n';
    if (!passed) throw std::runtime_error("probe-limit guard did not fail closed");
  }
}

Rat parse_rational(const std::string& text) {
  const std::size_t slash = text.find('/');
  const std::int64_t numerator = std::stoll(text.substr(0, slash));
  const std::int64_t denominator = slash == std::string::npos
      ? 1 : std::stoll(text.substr(slash + 1));
  if (numerator < 0 || denominator <= 0)
    throw std::invalid_argument("custom arrival offset must be nonnegative rational");
  return Rat(numerator, denominator);
}

std::string offset_scenario_name(const std::string& offset) {
  std::string name = "two_offset_r_";
  for (char c : offset) name += c == '/' ? '_' : c;
  return name;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc >= 2 && std::string(argv[1]) == "--scale") {
      bool compare_event = true;
      bool two_only = false;
      bool solo_only = false;
      for (int i = 2; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--cycle-only") compare_event = false;
        else if (arg == "--two-only") two_only = true;
        else if (arg == "--solo-only") solo_only = true;
        else if (arg == "--trace") g_trace_outer = true;
        else throw std::invalid_argument("unknown scale option: " + arg);
      }
      if (!two_only)
        emit_scenario(Policy::kGroupedQLocal, "scale_solo", {Rat(0)}, compare_event);
      if (!solo_only)
        emit_scenario(Policy::kGroupedQLocal, "scale_two_offset_005",
                      {Rat(0), Rat(1, 20)}, compare_event);
      return 0;
    }
    emit_grouped_lp_probe(Policy::kGroupedQ1);
    emit_grouped_lp_probe(Policy::kGroupedQLocal);
    for (Policy policy : {Policy::kEqualProgress, Policy::kGroupedQ1,
                          Policy::kGroupedQLocal}) {
      emit_scenario(policy, "solo", {Rat(0)});
      emit_scenario(policy, "two_offset_0", {Rat(0), Rat(0)});
      emit_scenario(policy, "two_offset_005", {Rat(0), Rat(1, 20)});
      for (int i = 1; i < argc; ++i) {
        const Rat offset = parse_rational(argv[i]);
        const std::string name = offset_scenario_name(argv[i]);
        emit_scenario(policy, name.c_str(), {Rat(0), offset});
      }
    }
    check_fail_closed_guard();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "bounded_lifecycle: " << error.what() << '\n';
    return 1;
  }
}
