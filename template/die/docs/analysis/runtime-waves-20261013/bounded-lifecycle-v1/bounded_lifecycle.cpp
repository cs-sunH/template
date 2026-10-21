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
#include <utility>
#include <vector>

namespace {
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

constexpr std::int64_t kM = 65;
constexpr std::int64_t kN = 65;
constexpr std::int64_t kK = 4097;
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

static_assert(kTmCount == 5 && kTnCount == 2 && kTkCount == 65,
              "fixture tile geometry changed");
static_assert(kOutputs == 10 && kWaves == 650, "fixture wave counts changed");

enum Resource : std::size_t { kCompute = 0, kHbm0 = 1, kHbm1 = 2, kResourceCount = 3 };
enum Stage : std::size_t { kLoad = 0, kComp = 1, kStore = 2, kStageCount = 3 };
enum class Policy { kEqualProgress, kGroupedQ };

const char* policy_name(Policy policy) {
  return policy == Policy::kEqualProgress ? "equal_progress" : "grouped_q_alpha";
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
};

Activity make_activity(int owner, Stage stage, std::int64_t wave_index) {
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
    activity.q = Rat(1);
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
  std::int64_t load_done = 0;
  std::int64_t compute_done = 0;
  std::int64_t store_done = 0;
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
  std::int64_t max_probe_steps = 0;
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
    if (world.policy == Policy::kGroupedQ)
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
        if (world.policy == Policy::kGroupedQ)
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
      if (world.policy == Policy::kGroupedQ)
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
      if (world.policy == Policy::kGroupedQ)
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

void start_stage(Owner& owner, Stage stage, int owner_id, std::int64_t wave_index) {
  LiveActivity& live = owner.active[stage];
  if (live.active) throw std::runtime_error("stage already active");
  live.active = true;
  live.activity = make_activity(owner_id, stage, wave_index);
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
        owner.active[kLoad] = LiveActivity{};
        push_bounded(owner.loaded, owner.loaded_count, wave);
        ++owner.metrics.load_done;
        owner.metrics.read_bytes += payload(make_wave(wave), kLoad);
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
        if (wave.final_k) push_bounded(owner.computed, owner.computed_count, wave_index);
        changed = true;
      }
    }
    for (int oi = 0; oi < world.owner_count; ++oi) {
      Owner& owner = world.owners[oi];
      if (!owner.started || owner.finished) continue;
      if (owner.active[kStore].active && owner.active[kStore].activity.progress == Rat(1)) {
        const std::int64_t wave = owner.active[kStore].activity.wave_index;
        owner.active[kStore] = LiveActivity{};
        ++owner.metrics.store_done;
        owner.metrics.write_bytes += payload(make_wave(wave), kStore);
        ++owner.retired_outputs;
        changed = true;
      }
    }
    for (int oi = 0; oi < world.owner_count; ++oi) {
      Owner& owner = world.owners[oi];
      if (!owner.started || owner.finished) continue;
      if (!owner.active[kLoad].active && owner.next_load < kWaves && input_occupancy(owner) < 2) {
        const std::int64_t wave = owner.next_load++;
        start_stage(owner, kLoad, oi, wave);
        changed = true;
      }
      if (!owner.active[kComp].active && owner.loaded_count > 0 && output_occupancy(owner) < 2) {
        const std::int64_t wave = pop_bounded(owner.loaded, owner.loaded_count);
        start_stage(owner, kComp, oi, wave);
        changed = true;
      }
      if (!owner.active[kStore].active && owner.computed_count > 0) {
        const std::int64_t wave = pop_bounded(owner.computed, owner.computed_count);
        start_stage(owner, kStore, oi, wave);
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

int wave_category(std::int64_t wave_index) {
  if (wave_index < 0 || wave_index >= kWaves) return -1;
  return make_wave(wave_index).category;
}

std::string wave_signature(std::int64_t wave_index) {
  if (wave_index < 0 || wave_index >= kWaves) return "end";
  const Wave w = make_wave(wave_index);
  std::ostringstream out;
  out << w.category << ':' << w.m_elems << ':' << w.n_elems << ':' << w.k_elems
      << ':' << w.read_bytes[0] << ':' << w.read_bytes[1]
      << ':' << w.write_bytes[0] << ':' << w.write_bytes[1]
      << ':' << w.vec_ops << ':' << w.final_k;
  return out.str();
}

std::string segment_signature(const World& world) {
  std::ostringstream out;
  for (int i = 0; i < world.owner_count; ++i) {
    const Owner& owner = world.owners[i];
    if (!owner.started || owner.finished || owner.next_load >= kWaves) out << "E;";
    else out << wave_category(owner.next_load) << ';';
  }
  return out.str();
}

std::string state_key(World& world, const Allocation& allocation) {
  std::ostringstream out;
  out << "bounded-lifecycle-v1|" << policy_name(world.policy) << "|g0|";
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
  return OwnerMetrics{a.read_bytes - b.read_bytes, a.write_bytes - b.write_bytes,
                      a.vec_ops - b.vec_ops, a.load_done - b.load_done,
                      a.compute_done - b.compute_done, a.store_done - b.store_done};
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

std::int64_t bounded_cycle_count(const World& world,
                                 const CycleDelta& delta,
                                 const Owner& old_owner,
                                 int owner_index,
                                 std::int64_t proposed) {
  const Owner& owner = world.owners[owner_index];
  const std::int64_t shift = delta.next_load[owner_index];
  if (shift < 0) return 0;
  std::int64_t allowed = proposed;
  if (shift > 0) {
    if (owner.next_load >= kWaves) return 0;
    const std::int64_t next_k = owner.next_load % kTkCount;
    if (next_k >= kTkCount - 1) return 0;
    const std::int64_t tail_index = (owner.next_load / kTkCount) * kTkCount + (kTkCount - 1);
    allowed = std::min(allowed, (tail_index - owner.next_load) / shift);
  }
  // Every in-flight/queued wave must remain in this same nonfinal full-K
  // segment under the repeated cursor shift.
  for (Stage stage : {kLoad, kComp, kStore}) {
    const LiveActivity& current = owner.active[stage];
    const LiveActivity& before = old_owner.active[stage];
    if (current.active != before.active) return 0;
    if (!current.active) continue;
    const std::int64_t id_delta = current.activity.wave_index - before.activity.wave_index;
    if (id_delta != shift) return 0;
    const Wave w = make_wave(current.activity.wave_index);
    if (w.final_k) return 0;
    if (shift > 0) {
      const std::int64_t last_full = (current.activity.wave_index / kTkCount) * kTkCount + (kTkCount - 2);
      allowed = std::min(allowed, (last_full - current.activity.wave_index) / shift);
    }
  }
  auto check_queue = [&](const std::array<std::int64_t, 2>& now_queue,
                         int now_count,
                         const std::array<std::int64_t, 2>& old_queue,
                         int old_count) {
    if (now_count != old_count) { allowed = 0; return; }
    for (int j = 0; j < now_count; ++j) {
      if (now_queue[j] - old_queue[j] != shift) { allowed = 0; return; }
      const Wave w = make_wave(now_queue[j]);
      if (w.final_k) { allowed = 0; return; }
      if (shift > 0) {
        const std::int64_t last_full = (now_queue[j] / kTkCount) * kTkCount + (kTkCount - 2);
        allowed = std::min(allowed, (last_full - now_queue[j]) / shift);
      }
    }
  };
  check_queue(owner.loaded, owner.loaded_count, old_owner.loaded, old_owner.loaded_count);
  check_queue(owner.computed, owner.computed_count, old_owner.computed, old_owner.computed_count);
  return std::max<std::int64_t>(0, allowed);
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
    owner.metrics.load_done += d.load_done * cycles;
    owner.metrics.compute_done += d.compute_done * cycles;
    owner.metrics.store_done += d.store_done * cycles;
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

RunResult run(Policy policy, const std::vector<Rat>& arrivals,
              bool use_cycle_detector, std::int64_t probe_limit = kMaxProbes) {
  if (arrivals.empty() || arrivals.size() > 2) throw std::invalid_argument("need one or two owners");
  World world;
  world.policy = policy;
  world.owner_count = static_cast<int>(arrivals.size());
  for (int i = 0; i < world.owner_count; ++i) world.arrivals[i] = arrivals[i];

  std::map<std::string, Snapshot> seen;
  std::string current_segment;
  bool segment_valid = false;
  std::int64_t probes = 0;
  const std::int64_t max_events = 1000000;

  while (!all_finished(world)) {
    if (admit_arrivals(world)) {
      seen.clear();
      probes = 0;
      segment_valid = false;
    }
    settle(world);
    if (all_finished(world)) break;

    const std::string segment = segment_signature(world);
    if (!segment_valid || segment != current_segment) {
      seen.clear();
      probes = 0;
      current_segment = segment;
      segment_valid = true;
    }

    Allocation allocation = allocate(world);
    if (use_cycle_detector) {
      const std::string key = state_key(world, allocation);
      const auto found = seen.find(key);
      const bool repeated = found != seen.end();
      if (!repeated && probes >= probe_limit) {
        world.fail_closed = true;
        world.fail_reason = "no_exact_joint_repeat_within_probe_limit";
        world.max_probe_steps = std::max(world.max_probe_steps, probes);
        return RunResult{std::move(world), "fail_closed"};
      }
      if (repeated) {
        const Snapshot& old = found->second;
        const CycleDelta delta = cycle_delta(old, world);
        if (delta.period <= Rat(0)) {
          world.fail_closed = true;
          world.fail_reason = "zero_or_negative_exact_cycle_period";
          return RunResult{std::move(world), "fail_closed"};
        }
        // Do not skip to request completion or across a full/tail boundary.
        // The stream's K-final boundary supplies the finite skip cap here.
        std::int64_t cycles = std::numeric_limits<std::int64_t>::max();
        bool has_cap = false;
        for (int i = 0; i < world.owner_count; ++i) {
          const std::int64_t shift = delta.next_load[i];
          if (shift <= 0) continue;
          has_cap = true;
          const Owner& owner = world.owners[i];
          if (owner.next_load >= kWaves) { cycles = 0; break; }
          const std::int64_t k = owner.next_load % kTkCount;
          if (k >= kTkCount - 1) { cycles = 0; break; }
          const std::int64_t tail_index = (owner.next_load / kTkCount) * kTkCount + (kTkCount - 1);
          cycles = std::min(cycles, (tail_index - owner.next_load) / shift);
          const Owner& old_owner = old.owner_state[i];
          cycles = bounded_cycle_count(world, delta, old_owner, i, cycles);
        }
        if (!has_cap) cycles = 0;
        for (int i = 0; i < world.owner_count; ++i) {
          if (delta.owner_metrics[i].store_done > 0 || delta.retired[i] > 0)
            cycles = 0;
        }
        // A pending owner arrival is an external prefix boundary. Repeating
        // states may be used only for whole cycles that end no later than it;
        // the normal event loop then admits the owner and starts a fresh key.
        if (cycles > 0) {
          const auto arrival = next_arrival(world);
          if (arrival) {
            const Rat until_arrival = *arrival - world.now;
            if (until_arrival < Rat(0)) {
              world.fail_closed = true;
              world.fail_reason = "arrival_prefix_moved_backwards";
              return RunResult{std::move(world), "fail_closed"};
            }
            cycles = std::min(cycles, floor_nonnegative(until_arrival / delta.period));
          }
        }
        if (cycles > 0) {
          apply_cycles(world, delta, cycles);
          world.skipped_cycles += cycles;
          seen.clear();
          probes = 0;
          continue;
        }
        // A repeat at a segment boundary is still useful evidence, but it
        // cannot authorize a skip across the full/tail transition.
        seen.clear();
        probes = 0;
      } else {
        seen.emplace(key, snapshot(world));
      }
    }

    const Rat dt = interval_length(world, allocation);
    integrate(world, allocation, dt);
    world.now += dt;
    ++world.intervals;
    ++world.transitions;
    if (use_cycle_detector) {
      ++probes;
      world.max_probe_steps = std::max(world.max_probe_steps, probes);
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
      << " skipped_cycles=" << w.skipped_cycles
      << " max_probes=" << w.max_probe_steps
      << " fail_reason=" << (w.fail_reason.empty() ? "none" : w.fail_reason);
  for (int i = 0; i < w.owner_count; ++i) {
    const Owner& o = w.owners[i];
    out << " owner" << i << "_read=" << o.metrics.read_bytes
        << " owner" << i << "_write=" << o.metrics.write_bytes
        << " owner" << i << "_ops=" << o.metrics.vec_ops
        << " owner" << i << "_load_done=" << o.metrics.load_done
        << " owner" << i << "_compute_done=" << o.metrics.compute_done
        << " owner" << i << "_store_done=" << o.metrics.store_done
        << " owner" << i << "_retired=" << o.retired_outputs
        << " owner" << i << "_next_load=" << o.next_load;
  }
  return out.str();
}

std::int64_t expected_read_bytes() {
  return 4 * kK * (kTnCount * kM + kTmCount * kN);
}

std::int64_t expected_write_bytes() { return 4 * kM * kN; }
std::int64_t expected_ops() { return 2 * kM * kN * kK; }

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

void emit_scenario(Policy policy, const char* scenario, const std::vector<Rat>& arrivals) {
  const RunResult oracle = run(policy, arrivals, false);
  const RunResult accelerated = run(policy, arrivals, true, kMaxProbes);
  std::cout << "config policy=" << policy_name(policy) << " scenario=" << scenario
            << " owners=" << arrivals.size() << " m=" << kM << " n=" << kN
            << " k=" << kK << " tm=" << kTm << " tn=" << kTn << " tk=" << kTk
            << " outputs=" << kOutputs << " waves=" << kWaves
            << " expected_read=" << expected_read_bytes()
            << " expected_write=" << expected_write_bytes()
            << " expected_ops=" << expected_ops()
            << " probe_cap=" << kMaxProbes << '\n';
  std::cout << metrics_line("event", policy, scenario, oracle) << '\n';
  std::cout << metrics_line("cycle", policy, scenario, accelerated) << '\n';

  const bool complete_equal = oracle.status == "completed" &&
      accelerated.status == "completed" &&
      oracle.world.now == accelerated.world.now &&
      counters_equal(oracle.world.counters, accelerated.world.counters) &&
      oracle.world.owner_count == accelerated.world.owner_count;
  bool owner_equal = complete_equal;
  if (owner_equal) {
    for (int i = 0; i < oracle.world.owner_count; ++i) {
      const Owner& a = oracle.world.owners[i];
      const Owner& b = accelerated.world.owners[i];
      owner_equal = owner_equal && a.finish == b.finish &&
          a.metrics.read_bytes == b.metrics.read_bytes &&
          a.metrics.write_bytes == b.metrics.write_bytes &&
          a.metrics.vec_ops == b.metrics.vec_ops &&
          a.metrics.load_done == b.metrics.load_done &&
          a.metrics.compute_done == b.metrics.compute_done &&
          a.metrics.store_done == b.metrics.store_done &&
          a.retired_outputs == b.retired_outputs &&
          a.next_load == b.next_load;
    }
  }
  std::cout << "check policy=" << policy_name(policy) << " scenario=" << scenario
            << " cycle_equals_event=" << (owner_equal ? "true" : "false")
            << " read_bytes_per_owner=" << expected_read_bytes()
            << " write_bytes_per_owner=" << expected_write_bytes()
            << " ops_per_owner=" << expected_ops() << '\n';
}

void check_fail_closed_guard() {
  const RunResult capped = run(Policy::kGroupedQ, {Rat(0)}, true, 1);
  const bool passed = capped.status == "fail_closed" &&
      capped.world.fail_reason == "no_exact_joint_repeat_within_probe_limit" &&
      capped.world.max_probe_steps == 1;
  std::cout << "guard probe_limit=1 status=" << capped.status
            << " steps=" << capped.world.max_probe_steps
            << " reason=" << (capped.world.fail_reason.empty() ? "none" : capped.world.fail_reason)
            << " fail_closed=" << (passed ? "true" : "false") << '\n';
  if (!passed) throw std::runtime_error("probe-limit guard did not fail closed");
}

}  // namespace

int main() {
  try {
    for (Policy policy : {Policy::kEqualProgress, Policy::kGroupedQ}) {
      emit_scenario(policy, "solo", {Rat(0)});
      emit_scenario(policy, "two_offset_0", {Rat(0), Rat(0)});
      emit_scenario(policy, "two_offset_005", {Rat(0), Rat(1, 20)});
    }
    check_fail_closed_guard();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "bounded_lifecycle: " << error.what() << '\n';
    return 1;
  }
}
