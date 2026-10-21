#include "coarse_die/Hardware.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace coarse_die {
namespace {

using Json = nlohmann::json;

constexpr char kSourceRevision[] = "41f119135fd2d4cc87292f59b4b00adc918eece0";
constexpr char kCoreTableId[] = "core32_v1";
constexpr char kCoreTableSha256[] =
    "d66b4a76122742ed91102d75305e01f10a8241213c96ed2289ecd51a1f7aa346";
constexpr std::array<double, 8> kSramMiB =
    {{0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00}};
constexpr std::array<int, 4> kNocWidths = {{64, 128, 256, 512}};
constexpr std::array<double, 8> kL0DisplayMm =
    {{0.40662, 0.81324, 1.21986, 1.62648, 2.03310, 2.43972, 2.84634, 3.25296}};

// All 32 rows from 架构建模_单Core版_严格复核修订.docx §1.3.
// SRAM capacity and NoC width select resources within the one Core design.
constexpr int kNVec[4][8] = {
    {2, 15, 40, 77, 126, 204, 280, 368},
    {2, 15, 40, 77, 126, 198, 273, 360},
    {2, 15, 40, 70, 117, 180, 252, 336},
    {2, 12, 32, 56, 99, 156, 210, 272},
};
constexpr int kNPe[4][8] = {
    {40, 297, 791, 1523, 2492, 4035, 5538, 7278},
    {40, 297, 791, 1523, 2492, 3916, 5399, 7120},
    {40, 297, 791, 1384, 2314, 3560, 4984, 6645},
    {40, 237, 633, 1108, 1958, 3085, 4153, 5379},
};

struct CoreRow {
  double sram_mib;
  int noc_width_bits;
  int n_vec;
  int n_pe;
  double displayed_l0_mm;
};

std::array<CoreRow, 32> makeCoreRows() {
  std::array<CoreRow, 32> rows{};
  std::size_t cursor = 0;
  for (std::size_t b = 0; b < kNocWidths.size(); ++b) {
    for (std::size_t s = 0; s < kSramMiB.size(); ++s) {
      rows[cursor++] = CoreRow{kSramMiB[s], kNocWidths[b],
                              kNVec[b][s], kNPe[b][s], kL0DisplayMm[s]};
    }
  }
  return rows;
}

const std::array<CoreRow, 32>& coreRows() {
  static const auto rows = makeCoreRows();
  return rows;
}

[[noreturn]] void invalid(const std::string& path, const std::string& why) {
  throw std::invalid_argument(path + ": " + why);
}

Json requireField(const Json& object, const char* key,
                  const std::string& path) {
  if (!object.is_object()) invalid(path, "must be an object");
  const auto it = object.find(key);
  if (it == object.end()) invalid(path + "." + key, "is required");
  return *it;
}

std::string stringField(const Json& object, const char* key,
                        const std::string& path) {
  const auto& value = requireField(object, key, path);
  if (!value.is_string()) invalid(path + "." + key, "must be a string");
  return value.get<std::string>();
}

double numberValue(const Json& value, const std::string& path) {
  if (!value.is_number()) invalid(path, "must be a number");
  const double result = value.get<double>();
  if (!std::isfinite(result)) invalid(path, "must be finite");
  return result;
}

double numberField(const Json& object, const char* key,
                   const std::string& path) {
  return numberValue(requireField(object, key, path), path + "." + key);
}

bool optionalBool(const Json& object, const char* key, bool fallback,
                  const std::string& path) {
  const auto it = object.find(key);
  if (it == object.end()) return fallback;
  if (!it->is_boolean()) invalid(path + "." + key, "must be a boolean");
  return it->get<bool>();
}

int positiveIntegerValue(const Json& value, const std::string& path,
                         int maximum = std::numeric_limits<int>::max()) {
  const double d = numberValue(value, path);
  if (std::floor(d) != d || d < 1.0 || d > maximum)
    invalid(path, "must be a positive integer within the supported range");
  return static_cast<int>(d);
}

int zeroBasedIndexValue(const Json& value, const std::string& path) {
  const double d = numberValue(value, path);
  if (std::floor(d) != d || d < 0.0 || d > 1.0)
    invalid(path, "must be the zero-based index 0 or 1");
  return static_cast<int>(d);
}

int positiveIntegerField(const Json& object, const char* key,
                         const std::string& path,
                         int maximum = std::numeric_limits<int>::max()) {
  return positiveIntegerValue(requireField(object, key, path),
                              path + "." + key, maximum);
}

double positiveNumberField(const Json& object, const char* key,
                           const std::string& path) {
  const double value = numberField(object, key, path);
  if (value <= 0.0) invalid(path + "." + key, "must be greater than zero");
  return value;
}

void nonnegative(double value, const std::string& path) {
  if (value < 0.0) invalid(path, "must be nonnegative");
}

bool nearlyEqual(double a, double b, double relative = 1.0e-10,
                 double absolute = 1.0e-10) {
  return std::abs(a - b) <= absolute + relative * std::max(std::abs(a), std::abs(b));
}

Json sourceConstants() {
  return Json{{"process_node_nm", 7},
              {"clock_hz", 1.0e9},
              {"sram_wordwidth_bits", 1024},
              {"sram_quarter_mib_long_side_mm", 0.40662},
              {"sram_short_side_mm", 0.13925},
              {"vec_lane_width_mm", 0.1897},
              {"vec_lane_height_mm", 0.1265},
              {"vec_lane_fp32_eq_ops_per_s", 3.9554e9},
              {"pe_area_mm2", 0.0000376},
              {"pe_int4_ops_per_mac", 2},
              {"pe_per_vec_ratio", 19.777},
              {"noc_base_area_fraction", 0.15},
              {"noc_fixed_fraction", 0.20},
              {"noc_width_fraction", 0.80},
              {"noc_base_width_bits", 256},
              {"hbm_sku", "Micron MT65B18G16120A00QH-92:A"},
              {"hbm_stack_capacity_bytes", 36.0e9},
              {"hbm_stack_peak_bytes_per_s", 1.1776e12},
              {"hbm_stack_width_bits", 1024},
              {"hbm_transfer_per_s", 9.2e9},
              {"hbm_stack_width_mm", 10.98},
              {"hbm_stack_height_mm", 10.98},
              {"hbm_stack_thickness_mm", 0.78},
              {"hbm_gap_mm", 0.15},
              {"bump_pitch_mm", 0.055},
              {"bump_communication_fraction", 0.40},
              {"bump_rate_bits_per_s", 2.5e9}};
}

void validateSourceConstants(const Json& input) {
  const auto it = input.find("source_constants");
  if (it == input.end()) return;
  if (!it->is_object()) invalid("source_constants", "must be an object");
  const Json expected = sourceConstants();
  if (it->size() != expected.size())
    invalid("source_constants", "must contain the complete v2 source constant mirror");
  for (auto expectedIt = expected.begin(); expectedIt != expected.end(); ++expectedIt) {
    const auto actualIt = it->find(expectedIt.key());
    if (actualIt == it->end()) invalid("source_constants." + expectedIt.key(), "is required");
    if (expectedIt->is_number()) {
      const double actual = numberValue(*actualIt, "source_constants." + expectedIt.key());
      const double wanted = expectedIt->get<double>();
      if (!nearlyEqual(actual, wanted, 1.0e-12, 1.0e-12))
        invalid("source_constants." + expectedIt.key(), "does not match the immutable source constant");
    } else if (*actualIt != *expectedIt) {
      invalid("source_constants." + expectedIt.key(), "does not match the immutable source constant");
    }
  }
}

void validateDerivedMirrorValue(const Json& expected, const Json& supplied,
                                const std::string& path) {
  if (expected.is_object()) {
    if (!supplied.is_object()) invalid(path, "must be an object matching the resolved derived value");
    for (auto it = supplied.begin(); it != supplied.end(); ++it) {
      const auto expectedIt = expected.find(it.key());
      if (expectedIt == expected.end())
        invalid(path + "." + it.key(), "is not a field in the resolved derived value");
      validateDerivedMirrorValue(*expectedIt, it.value(), path + "." + it.key());
    }
    return;
  }
  if (expected.is_array()) {
    if (!supplied.is_array() || expected.size() != supplied.size())
      invalid(path, "must have the same array shape as the resolved derived value");
    for (std::size_t i = 0; i < expected.size(); ++i)
      validateDerivedMirrorValue(expected[i], supplied[i], path + "[" + std::to_string(i) + "]");
    return;
  }
  if (expected.is_number()) {
    if (!supplied.is_number()) invalid(path, "must be numeric like the resolved derived value");
    const double actual = numberValue(supplied, path);
    const double wanted = expected.get<double>();
    if (!nearlyEqual(actual, wanted, 1.0e-10, 1.0e-9))
      invalid(path, "does not match the resolved derived value");
    return;
  }
  if (expected != supplied)
    invalid(path, "does not match the resolved derived value");
}

void validateDerivedMirrors(const Json& input, const Json& derived) {
  for (const char* key : {"derived", "derived_hardware"}) {
    const auto it = input.find(key);
    if (it == input.end()) continue;
    if (!it->is_object()) invalid(key, "must be an object mirror of resolver-derived hardware");
    validateDerivedMirrorValue(derived, *it, key);
  }
}

// Small self-contained SHA-256 implementation keeps the library independent of
// a crypto runtime and gives the hardware descriptor a stable content identity.
std::string sha256(std::string_view input) {
  static constexpr std::uint32_t k[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
      0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
      0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
      0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
      0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
      0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
      0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  auto rotate = [](std::uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); };
  std::vector<std::uint8_t> bytes(input.begin(), input.end());
  const std::uint64_t bitLength = static_cast<std::uint64_t>(bytes.size()) * 8;
  bytes.push_back(0x80);
  while (bytes.size() % 64 != 56) bytes.push_back(0);
  for (int i = 7; i >= 0; --i)
    bytes.push_back(static_cast<std::uint8_t>((bitLength >> (i * 8)) & 0xff));

  std::array<std::uint32_t, 8> h = {{0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                      0xa54ff53a, 0x510e527f, 0x9b05688c,
                                      0x1f83d9ab, 0x5be0cd19}};
  for (std::size_t offset = 0; offset < bytes.size(); offset += 64) {
    std::uint32_t w[64]{};
    for (std::size_t i = 0; i < 16; ++i) {
      const std::size_t p = offset + i * 4;
      w[i] = (static_cast<std::uint32_t>(bytes[p]) << 24) |
             (static_cast<std::uint32_t>(bytes[p + 1]) << 16) |
             (static_cast<std::uint32_t>(bytes[p + 2]) << 8) |
             static_cast<std::uint32_t>(bytes[p + 3]);
    }
    for (std::size_t i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotate(w[i - 15], 7) ^ rotate(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 = rotate(w[i - 2], 17) ^ rotate(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a=h[0], b=h[1], c=h[2], d=h[3], e=h[4], f=h[5], g=h[6], z=h[7];
    for (std::size_t i = 0; i < 64; ++i) {
      const std::uint32_t s1 = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
      const std::uint32_t ch = (e & f) ^ (~e & g);
      const std::uint32_t t1 = z + s1 + ch + k[i] + w[i];
      const std::uint32_t s0 = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t t2 = s0 + maj;
      z=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=z;
  }
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (const auto word : h) out << std::setw(8) << word;
  return out.str();
}

Json coreRowsJson() {
  Json rows = Json::array();
  for (const auto& row : coreRows()) {
    rows.push_back(Json{{"sram_mib_per_core", row.sram_mib},
                        {"noc_width_bits", row.noc_width_bits},
                        {"n_vec", row.n_vec},
                        {"n_pe", row.n_pe},
                        {"l0_mm_display", row.displayed_l0_mm}});
  }
  return rows;
}

void validateCoreTable() {
  static const bool validated = [] {
    const auto digest = sha256(coreRowsJson().dump());
    if (coreRows().size() != 32 || digest != kCoreTableSha256)
      throw std::logic_error("compiled core32_v1 table does not match its source SHA-256: " + digest);
    std::set<std::pair<double, int>> seen;
    for (const auto& row : coreRows()) {
      if (!seen.emplace(row.sram_mib, row.noc_width_bits).second)
        throw std::logic_error("duplicate row in core32_v1");
      if (row.n_pe != static_cast<int>(std::floor(19.777 * row.n_vec + 0.5)))
        throw std::logic_error("PE count violates the source round-half-up rule");
    }
    return true;
  }();
  (void)validated;
}

const CoreRow& findCoreRow(double sram, int noc) {
  validateCoreTable();
  for (const auto& row : coreRows()) {
    if (nearlyEqual(row.sram_mib, sram, 0.0, 1.0e-12) &&
        row.noc_width_bits == noc)
      return row;
  }
  invalid("architecture", "configuration has no matching row in core32_v1");
}

double coreEdgeMm(double sramMiB) {
  return sramMiB * 0.40662 / 0.25;
}

// Constraints validate one supplied instance; they never generate hardware.
struct GeometryConstraints {
  double areaMin = 0.0;
  double areaMax = std::numeric_limits<double>::infinity();
  double reticleWidth = std::numeric_limits<double>::infinity();
  double reticleHeight = std::numeric_limits<double>::infinity();
  bool allowRotation = true;
  double maxAspect = std::numeric_limits<double>::infinity();
  bool requireNonnegativeD2d = false;
  bool requireHbmWithinDie = false;
};

GeometryConstraints validateConstraints(const Json& p, const std::string& path,
                                        bool legacy = false) {
  if (!p.is_object()) invalid(path, "must be an object");
  GeometryConstraints result;
  if (!legacy) {
    for (auto it = p.begin(); it != p.end(); ++it) {
      if (it.key() != "die_area_min_mm2" && it.key() != "die_area_max_mm2" &&
          it.key() != "reticle_width_mm" && it.key() != "reticle_height_mm" &&
          it.key() != "allow_reticle_rotation" && it.key() != "die_max_aspect_ratio" &&
          it.key() != "require_nonnegative_d2d_budget" &&
          it.key() != "require_hbm_row_within_die")
        invalid(path + "." + it.key(), "is not an instance validation constraint");
    }
  }
  const auto number = [&](const char* key, double fallback) {
    return p.contains(key) ? positiveNumberField(p, key, path) : fallback;
  };
  result.areaMin = number("die_area_min_mm2", result.areaMin);
  result.areaMax = number("die_area_max_mm2", result.areaMax);
  if (result.areaMin > result.areaMax)
    invalid(path, "die_area_min_mm2 must not exceed die_area_max_mm2");
  if (p.contains("reticle_width_mm") != p.contains("reticle_height_mm"))
    invalid(path, "reticle_width_mm and reticle_height_mm must be supplied together");
  result.reticleWidth = number("reticle_width_mm", result.reticleWidth);
  result.reticleHeight = number("reticle_height_mm", result.reticleHeight);
  result.allowRotation = optionalBool(p, "allow_reticle_rotation", true, path);
  result.maxAspect = number("die_max_aspect_ratio", result.maxAspect);
  if (result.maxAspect < 1.0)
    invalid(path + ".die_max_aspect_ratio", "must be at least one");
  result.requireNonnegativeD2d = optionalBool(p, "require_nonnegative_d2d_budget", false, path);
  result.requireHbmWithinDie = optionalBool(p, "require_hbm_row_within_die", false, path);
  return result;
}

struct PerformanceProfile {
  double etaPe;
  double etaVec;
  double etaHbmMatrix;
  double etaHbmVector;
  double etaSram;
  double etaLocal;
  double etaMesh;
  double etaInject;
  double rhoSram;
  double launchNs;
  double setupNs;
  double kappaQuant;
  double kappaAccConvert;
  std::string topology;
  std::string routing;
};

double efficiencyField(const Json& p, const char* key) {
  const double value = positiveNumberField(p, key, "performance_profile");
  if (value > 1.0) invalid(std::string("performance_profile.") + key, "must be in (0,1]");
  return value;
}

double nonnegativeField(const Json& p, const char* key) {
  const double value = numberField(p, key, "performance_profile");
  nonnegative(value, std::string("performance_profile.") + key);
  return value;
}

PerformanceProfile validatePerformanceProfile(const Json& p) {
  PerformanceProfile profile{};
  (void)stringField(p, "execution_profile", "performance_profile");
  (void)stringField(p, "calibration_status", "performance_profile");
  if (!p.contains("timing_topology_profile"))
    invalid("performance_profile.timing_topology_profile",
            "missing_topology_profile: select an explicit v2 topology assumption");
  profile.topology = stringField(p, "timing_topology_profile", "performance_profile");
  if (profile.topology != "distributed_ideal_v2" &&
      profile.topology != "edge_single_word_v2" &&
      profile.topology != "explicit_ingress_v2")
    invalid("performance_profile.timing_topology_profile", "must select an explicit v2 topology profile");
  profile.routing = stringField(p, "routing_profile", "performance_profile");
  if (profile.routing != "mapped_uniform_sources_v2" &&
      profile.routing != "uniform_grid_v2" &&
      profile.routing != "edge_uniform_v2" &&
      profile.routing != "uniform_proxy_v2")
    invalid("performance_profile.routing_profile", "is not a supported v2 routing profile");
  (void)stringField(p, "pipeline_profile", "performance_profile");
  (void)stringField(p, "granularity_validation", "performance_profile");
  (void)stringField(p, "physical_topology_validation", "performance_profile");

  profile.etaPe = efficiencyField(p, "eta_pe");
  profile.etaVec = efficiencyField(p, "eta_vec");
  profile.etaHbmMatrix = efficiencyField(p, "eta_hbm_matrix");
  profile.etaHbmVector = efficiencyField(p, "eta_hbm_vector");
  profile.etaSram = efficiencyField(p, "eta_sram");
  profile.etaLocal = efficiencyField(p, "eta_local");
  profile.etaMesh = efficiencyField(p, "eta_mesh");
  profile.etaInject = efficiencyField(p, "eta_inject");
  profile.rhoSram = efficiencyField(p, "rho_sram");
  profile.launchNs = nonnegativeField(p, "launch_ns_per_macro");
  profile.setupNs = nonnegativeField(p, "memory_setup_ns_per_macro");
  profile.kappaQuant = nonnegativeField(p, "kappa_quant_eqops");
  profile.kappaAccConvert = nonnegativeField(p, "kappa_acc_convert_eqops");
  return profile;
}

void checkCoreTableReference(const Json& input) {
  const auto it = input.find("core_resource_table");
  if (it == input.end()) return;
  if (!it->is_object()) invalid("core_resource_table", "must be an object");
  const std::string id = stringField(*it, "id", "core_resource_table");
  const std::string digest = stringField(*it, "rows_sha256", "core_resource_table");
  if (id != kCoreTableId) invalid("core_resource_table.id", "unknown source table identity");
  if (digest != kCoreTableSha256)
    invalid("core_resource_table.rows_sha256", "does not match the validated 32-row source table");
}

struct Architecture {
  double sramMiB;
  int nocBits;
  int nx;
  int ny;
  double pitchX;
  double pitchY;
  double extraWidth;
  double extraHeight;
  std::string geometry;
  std::string geometryCandidate;
  int hbmN;
  std::string placement;
  double regionWidth;
  double regionHeight;
  bool hasRegion;
};

Architecture parseArchitecture(const Json& a) {
  Architecture arch{};
  if (a.contains("core_type"))
    invalid("architecture.core_type", "is no longer supported; specify SRAM and NoC width for the single Core design");
  arch.sramMiB = positiveNumberField(a, "sram_mib_per_core", "architecture");
  bool validSram = false;
  for (const double s : kSramMiB) validSram = validSram || nearlyEqual(s, arch.sramMiB, 0.0, 1.0e-12);
  if (!validSram) invalid("architecture.sram_mib_per_core", "must be one of the eight source SRAM sizes");
  arch.nocBits = positiveIntegerField(a, "noc_width_bits", "architecture", 512);
  if (std::find(kNocWidths.begin(), kNocWidths.end(), arch.nocBits) == kNocWidths.end())
    invalid("architecture.noc_width_bits", "must be 64, 128, 256, or 512");
  arch.nx = positiveIntegerField(a, "core_columns_nx", "architecture", 100000);
  arch.ny = positiveIntegerField(a, "core_rows_ny", "architecture", 100000);
  arch.pitchX = numberField(a, "core_pitch_x_mm", "architecture");
  arch.pitchY = numberField(a, "core_pitch_y_mm", "architecture");
  arch.extraWidth = numberField(a, "die_extra_width_mm", "architecture");
  arch.extraHeight = numberField(a, "die_extra_height_mm", "architecture");
  nonnegative(arch.pitchX, "architecture.core_pitch_x_mm");
  nonnegative(arch.pitchY, "architecture.core_pitch_y_mm");
  nonnegative(arch.extraWidth, "architecture.die_extra_width_mm");
  nonnegative(arch.extraHeight, "architecture.die_extra_height_mm");
  arch.geometry = a.contains("die_geometry")
      ? stringField(a, "die_geometry", "architecture") : "explicit";
  if (arch.geometry != "explicit" && arch.geometry != "minimum_grid" && arch.geometry != "grow_width" &&
      arch.geometry != "grow_height" && arch.geometry != "isotropic" &&
      arch.geometry != "resolved_source_boundary")
    invalid("architecture.die_geometry", "must be explicit or a supported legacy geometry identity");
  arch.geometryCandidate = arch.geometry;
  if (arch.geometry == "resolved_source_boundary") {
    arch.geometryCandidate = stringField(a, "geometry_candidate", "architecture");
    if (arch.geometryCandidate != "minimum_grid" && arch.geometryCandidate != "grow_width" &&
        arch.geometryCandidate != "grow_height" && arch.geometryCandidate != "isotropic")
      invalid("architecture.geometry_candidate", "must identify one of the four source boundary candidates");
  }
  arch.hbmN = positiveIntegerField(a, "hbm_stacks_per_row", "architecture", 14);
  arch.placement = stringField(a, "hbm_placement_mode", "architecture");
  if (arch.placement != "same_side_two_rows" && arch.placement != "two_sides_one_row_each")
    invalid("architecture.hbm_placement_mode", "must be same_side_two_rows or two_sides_one_row_each");
  if (a.contains("system_region_width_mm") != a.contains("system_region_height_mm"))
    invalid("architecture", "system region width and height must be supplied together");
  arch.hasRegion = a.contains("system_region_width_mm");
  arch.regionWidth = arch.hasRegion ? positiveNumberField(a, "system_region_width_mm", "architecture") : 0.0;
  arch.regionHeight = arch.hasRegion ? positiveNumberField(a, "system_region_height_mm", "architecture") : 0.0;
  return arch;
}

double bumpBudgetBytesPerSecond(double areaMm2) {
  constexpr double pitchMm = 0.055;
  constexpr double fraction = 0.40;
  constexpr double rateBitsPerSecond = 2.5e9;
  return areaMm2 / (pitchMm * pitchMm) * fraction * rateBitsPerSecond / 8.0;
}

double requiredDieArea(double minArea, int hbmPerRow) {
  const double hbmBytesPerSecond = 2.0 * hbmPerRow * 1.1776e12;
  return std::max(minArea, hbmBytesPerSecond / bumpBudgetBytesPerSecond(1.0));
}

std::pair<double, double> sourceGeometry(const std::string& candidate,
                                         double minW, double minH,
                                         double areaReq) {
  const double baseArea = minW * minH;
  if (candidate == "minimum_grid") return {minW, minH};
  if (baseArea >= areaReq)
    invalid("architecture.die_geometry", "an area extension is not a valid source boundary when the minimum grid already meets A_req");
  if (candidate == "grow_width") return {areaReq / minH, minH};
  if (candidate == "grow_height") return {minW, areaReq / minW};
  if (candidate == "isotropic") {
    const double scale = std::sqrt(areaReq / baseArea);
    return {minW * scale, minH * scale};
  }
  invalid("architecture.die_geometry", "unsupported source geometry candidate");
}

struct ExplicitIngress {
  std::array<double, 2> rawBps{{0.0, 0.0}};
  std::array<int, 2> rowToIngress{{0, 1}};
  std::array<int, 2> sharedGroup{{0, 1}};
  std::array<bool, 2> enabled{{true, true}};
  Json sourcePositions = Json::array();
  std::string source;
};

ExplicitIngress validateExplicitIngress(const Json& p, int nx) {
  const auto it = p.find("explicit_ingress");
  if (it == p.end() || !it->is_object())
    invalid("performance_profile.explicit_ingress", "is required by explicit_ingress_v2");
  const auto& ingress = *it;
  ExplicitIngress result;
  result.source = stringField(ingress, "source", "performance_profile.explicit_ingress");
  if (result.source.empty()) invalid("performance_profile.explicit_ingress.source", "must not be empty");

  const auto& raw = requireField(ingress, "ingress_raw_bytes_per_s", "performance_profile.explicit_ingress");
  const auto& rowMap = requireField(ingress, "row_to_ingress", "performance_profile.explicit_ingress");
  const auto& groups = requireField(ingress, "shared_group_ids", "performance_profile.explicit_ingress");
  const auto& enabled = requireField(ingress, "ingress_enabled", "performance_profile.explicit_ingress");
  const auto& positions = requireField(ingress, "source_positions", "performance_profile.explicit_ingress");
  if (!raw.is_array() || raw.size() != 2 || !rowMap.is_array() || rowMap.size() != 2 ||
      !groups.is_array() || groups.size() != 2 || !enabled.is_array() || enabled.size() != 2 ||
      !positions.is_array() || positions.size() != 2)
    invalid("performance_profile.explicit_ingress", "two-entry arrays are required for rates, row mapping, groups, enables, and positions");
  for (int i = 0; i < 2; ++i) {
    const double rawRate = numberValue(raw[i], "performance_profile.explicit_ingress.ingress_raw_bytes_per_s");
    nonnegative(rawRate, "performance_profile.explicit_ingress.ingress_raw_bytes_per_s");
    result.rawBps[i] = rawRate;
    result.rowToIngress[i] = zeroBasedIndexValue(rowMap[i], "performance_profile.explicit_ingress.row_to_ingress");
    result.sharedGroup[i] = zeroBasedIndexValue(groups[i], "performance_profile.explicit_ingress.shared_group_ids");
    if (!enabled[i].is_boolean()) invalid("performance_profile.explicit_ingress.ingress_enabled", "entries must be booleans");
    result.enabled[i] = enabled[i].get<bool>();
    if (result.enabled[i] && rawRate <= 0.0)
      invalid("performance_profile.explicit_ingress.ingress_raw_bytes_per_s", "enabled entrances need positive rates");
    if (!positions[i].is_array() || positions[i].empty())
      invalid("performance_profile.explicit_ingress.source_positions", "each entrance needs at least one [column, weight] pair");
    double weightSum = 0.0;
    for (const auto& pair : positions[i]) {
      if (!pair.is_array() || pair.size() != 2)
        invalid("performance_profile.explicit_ingress.source_positions", "entries must be [column, weight] pairs");
      const double columnValue = numberValue(pair[0], "performance_profile.explicit_ingress.source_positions.column");
      if (std::floor(columnValue) != columnValue || columnValue < 0 || columnValue >= nx)
        invalid("performance_profile.explicit_ingress.source_positions", "source column is outside the Core grid");
      const double weight = numberValue(pair[1], "performance_profile.explicit_ingress.source_positions.weight");
      if (weight <= 0.0) invalid("performance_profile.explicit_ingress.source_positions", "position weights must be positive");
      weightSum += weight;
      if (!std::isfinite(weightSum))
        invalid("performance_profile.explicit_ingress.source_positions", "position weight sum must be finite");
    }
    if (!nearlyEqual(weightSum, 1.0, 1.0e-10, 1.0e-10))
      invalid("performance_profile.explicit_ingress.source_positions", "weights for each entrance must sum to one");
  }
  for (int row = 0; row < 2; ++row) {
    if (!result.enabled[result.rowToIngress[row]])
      invalid("performance_profile.explicit_ingress.row_to_ingress", "HBM rows cannot map to a disabled entrance");
  }
  if (result.sharedGroup[0] == result.sharedGroup[1] && result.enabled[0] && result.enabled[1] &&
      !nearlyEqual(result.rawBps[0], result.rawBps[1], 1.0e-10, 1.0e-6))
    invalid("performance_profile.explicit_ingress.ingress_raw_bytes_per_s", "members of a shared group must declare the same shared service rate");
  result.sourcePositions = positions;
  return result;
}

struct Resource {
  const char* id;
  bool enabled;
  double rate;
  const char* unit;
};

}  // namespace

Hardware resolve_hardware(const nlohmann::json& input) {
  if (!input.is_object()) invalid("input", "must be a JSON object");
  if (stringField(input, "schema", "input") != "coarse_die_v2")
    invalid("schema", "must be coarse_die_v2");
  if (stringField(input, "source_revision", "input") != kSourceRevision)
    invalid("source_revision", "does not match the source architecture revision");
  validateCoreTable();
  checkCoreTableReference(input);
  validateSourceConstants(input);

  const auto& aJson = requireField(input, "architecture", "input");
  const auto profileIt = input.find("performance_profile");
  if (profileIt == input.end())
    invalid("performance_profile.timing_topology_profile",
            "missing_topology_profile: performance_profile must explicitly select a topology");
  const auto& pJson = *profileIt;
  const Architecture arch = parseArchitecture(aJson);
  const bool hasWidth = aJson.contains("die_width_mm");
  const bool hasHeight = aJson.contains("die_height_mm");
  if (hasWidth != hasHeight)
    invalid("architecture.die_width_mm/die_height_mm", "must be supplied together");
  const bool suppliedOutline = hasWidth && hasHeight;
  if (!suppliedOutline && arch.geometry == "explicit")
    invalid("architecture.die_width_mm/die_height_mm", "are required for an explicit single-Die instance");
  // Existing configs may name one deterministic source geometry without W/H.
  // This legacy decoding does not enumerate, rank or recommend configurations.
  const GeometryConstraints legacyConstraints = !suppliedOutline && input.contains("search_profile")
      ? validateConstraints(input.at("search_profile"), "search_profile", true)
      : GeometryConstraints{};
  const GeometryConstraints constraints = input.contains("validation_constraints")
      ? validateConstraints(input.at("validation_constraints"), "validation_constraints")
      : legacyConstraints;
  const PerformanceProfile perf = validatePerformanceProfile(pJson);
  const CoreRow& core = findCoreRow(arch.sramMiB, arch.nocBits);
  const double l0 = coreEdgeMm(arch.sramMiB);
  if (!nearlyEqual(l0, core.displayed_l0_mm, 0.0, 5.1e-6))
    throw std::logic_error("source L0 formula disagrees with the §1.3 displayed Core edge");

  const double minW = arch.nx * l0 + (arch.nx - 1.0) * arch.pitchX + arch.extraWidth;
  const double minH = arch.ny * l0 + (arch.ny - 1.0) * arch.pitchY + arch.extraHeight;
  if (!std::isfinite(minW) || !std::isfinite(minH) || minW <= 0.0 || minH <= 0.0)
    invalid("architecture", "Core grid produces an invalid minimum Die outline");
  const double areaReq = requiredDieArea(legacyConstraints.areaMin, arch.hbmN);
  const auto outline = suppliedOutline
      ? std::make_pair(positiveNumberField(aJson, "die_width_mm", "architecture"),
                       positiveNumberField(aJson, "die_height_mm", "architecture"))
      : sourceGeometry(arch.geometryCandidate, minW, minH, areaReq);
  const double dieW = outline.first, dieH = outline.second;
  if (!std::isfinite(dieW) || !std::isfinite(dieH) || dieW <= 0.0 || dieH <= 0.0)
    invalid("architecture.die_width_mm/die_height_mm", "must define a finite positive outline");
  if (dieW < minW - 1.0e-9 || dieH < minH - 1.0e-9)
    invalid("architecture.die_width_mm/die_height_mm", "do not contain the specified Core grid");
  const double dieArea = dieW * dieH;
  if (!std::isfinite(dieArea) || dieArea <= 0.0)
    invalid("architecture.die_width_mm/die_height_mm", "produce a nonfinite or nonpositive area");
  if (aJson.contains("die_area_mm2") &&
      !nearlyEqual(positiveNumberField(aJson, "die_area_mm2", "architecture"), dieArea))
    invalid("architecture.die_area_mm2", "must equal the supplied width times height");
  if (aJson.contains("core_count") &&
      numberField(aJson, "core_count", "architecture") != static_cast<double>(arch.nx) * arch.ny)
    invalid("architecture.core_count", "must equal core_columns_nx times core_rows_ny");
  if (dieArea < constraints.areaMin - 1.0e-9 || dieArea > constraints.areaMax + 1.0e-9)
    invalid("architecture.die_width_mm/die_height_mm", "fall outside the explicit validation area window");
  const double aspect = std::max(dieW, dieH) / std::min(dieW, dieH);
  const bool fitsReticle =
      (dieW <= constraints.reticleWidth && dieH <= constraints.reticleHeight) ||
      (constraints.allowRotation && dieW <= constraints.reticleHeight && dieH <= constraints.reticleWidth);
  if (!fitsReticle)
    invalid("architecture.die_width_mm/die_height_mm", "do not fit the explicit validation reticle");
  if (aspect > constraints.maxAspect + 1.0e-12)
    invalid("architecture.die_width_mm/die_height_mm", "exceed the explicit validation aspect ratio");

  const double hbmRowWidth = arch.hbmN * 10.98 + (arch.hbmN - 1.0) * 0.15;
  if (constraints.requireHbmWithinDie && hbmRowWidth > dieW + 1.0e-12)
    invalid("architecture.hbm_stacks_per_row", "HBM row does not fit within the Die width");
  const double bumpBps = bumpBudgetBytesPerSecond(dieArea);
  if (!std::isfinite(bumpBps))
    invalid("architecture.die_width_mm/die_height_mm", "overflow the derived bump budget");
  const double hbmTotalBps = 2.0 * arch.hbmN * 1.1776e12;
  const double rawD2dBiBps = bumpBps - hbmTotalBps;
  const double d2dToleranceBps = std::max(1.0e-6, hbmTotalBps * 1.0e-12);
  if (constraints.requireNonnegativeD2d && rawD2dBiBps < -d2dToleranceBps)
    invalid("architecture.hbm_stacks_per_row", "candidate has a negative bump-minus-HBM D2D budget");
  const double d2dBiBps = constraints.requireNonnegativeD2d
                              ? std::max(0.0, rawD2dBiBps)
                              : rawD2dBiBps;

  const double moduleW = std::max(dieW, hbmRowWidth) + 0.30;
  const double moduleH = dieH + 22.56;
  const double effectiveW = moduleW / std::sqrt(0.90);
  const double effectiveH = moduleH / std::sqrt(0.90);
  const auto moduleAxis = [](double region, double extent) {
    const double count = std::floor(region / extent + 1.0e-12);
    if (!std::isfinite(count) || count < 0.0 || count > std::numeric_limits<int>::max())
      invalid("architecture.system_region_width_mm/system_region_height_mm", "overflow the optional module count");
    return static_cast<int>(count);
  };
  const int moduleColumns = arch.hasRegion ? moduleAxis(arch.regionWidth, effectiveW) : 0;
  const int moduleRows = arch.hasRegion ? moduleAxis(arch.regionHeight, effectiveH) : 0;
  const long long moduleCount = static_cast<long long>(moduleColumns) * moduleRows;
  const long long coreCount = static_cast<long long>(arch.nx) * arch.ny;
  if (coreCount <= 0 || moduleCount < 0) invalid("architecture", "resource count overflow");

  const double sramBytesPerCore = arch.sramMiB * 1048576.0;
  const double pInt4 = 2.0 * core.n_pe * 1.0e9;
  const double pVec = 3.9554 * core.n_vec * 1.0e9;
  const double sramRate = coreCount * 128.0e9 * perf.etaSram;
  const double localRate = coreCount * (arch.nocBits / 8.0) * 1.0e9 * perf.etaLocal;
  const double linkRate = (arch.nocBits / 8.0) * 1.0e9 * perf.etaMesh;
  const long long meshEdges = 2LL * ((arch.nx - 1LL) * arch.ny + (arch.ny - 1LL) * arch.nx);
  const bool hasX = arch.nx > 1;
  const bool hasY = arch.ny > 1;

  std::array<double, 2> injectRate{{0.0, 0.0}};
  std::array<bool, 2> injectEnabled{{false, false}};
  std::array<int, 2> rowToIngress{{0, 0}};
  Json ingressEvidence = Json::object();
  if (perf.topology == "edge_single_word_v2") {
    const double sideRate = arch.nx * (arch.nocBits / 8.0) * 1.0e9 * perf.etaInject;
    if (arch.placement == "same_side_two_rows") {
      injectRate[0] = sideRate;
      injectEnabled[0] = true;
      rowToIngress = {{0, 0}};
    } else {
      injectRate = {{sideRate, sideRate}};
      injectEnabled = {{true, true}};
      rowToIngress = {{0, 1}};
    }
    ingressEvidence = Json{{"profile", perf.topology},
                          {"physical_topology_validation", "unverified"},
                          {"side_rate_Bps", sideRate},
                          {"row_to_ingress", rowToIngress}};
  } else if (perf.topology == "explicit_ingress_v2") {
    const ExplicitIngress explicitIngress = validateExplicitIngress(pJson, arch.nx);
    std::array<int, 2> groupToResource{{-1, -1}};
    int nextResource = 0;
    for (int i = 0; i < 2; ++i) {
      const int group = explicitIngress.sharedGroup[i];
      if (groupToResource[group] == -1 && explicitIngress.enabled[i])
        groupToResource[group] = nextResource++;
      if (explicitIngress.enabled[i]) {
        const int resource = groupToResource[group];
        injectEnabled[resource] = true;
        injectRate[resource] = explicitIngress.rawBps[i] * perf.etaInject;
      }
    }
    for (int row = 0; row < 2; ++row)
      rowToIngress[row] = groupToResource[explicitIngress.sharedGroup[explicitIngress.rowToIngress[row]]];
    ingressEvidence = Json{{"profile", perf.topology},
                          {"physical_topology_validation", "unverified"},
                          {"source", explicitIngress.source},
                          {"raw_ingress_bytes_per_s", explicitIngress.rawBps},
                          {"shared_group_ids", explicitIngress.sharedGroup},
                          {"row_to_ingress", rowToIngress},
                          {"source_positions", explicitIngress.sourcePositions}};
  } else {
    rowToIngress = {{0, 0}};
    ingressEvidence = Json{{"profile", perf.topology},
                          {"physical_topology_validation", "unverified"},
                          {"row_to_ingress", Json::array({nullptr, nullptr})},
                          {"unlimited", true}};
  }

  std::array<Resource, 13> resources = {{
      {"COMPUTE_GATE", true, 1.0, "second_of_service/s"},
      {"SRAM_READ", true, sramRate, "B/s"},
      {"SRAM_WRITE", true, sramRate, "B/s"},
      {"LOCAL", true, localRate, "B/s"},
      {"HBM0", true, arch.hbmN * 1.1776e12, "B/s"},
      {"HBM1", true, arch.hbmN * 1.1776e12, "B/s"},
      {"INJECT0", injectEnabled[0], injectRate[0], "B/s"},
      {"INJECT1", injectEnabled[1], injectRate[1], "B/s"},
      {"XPOS", hasX, hasX ? arch.ny * linkRate : 0.0, "B/s"},
      {"XNEG", hasX, hasX ? arch.ny * linkRate : 0.0, "B/s"},
      {"YPOS", hasY, hasY ? arch.nx * linkRate : 0.0, "B/s"},
      {"YNEG", hasY, hasY ? arch.nx * linkRate : 0.0, "B/s"},
      {"HOP", meshEdges > 0, meshEdges > 0 ? meshEdges * linkRate : 0.0, "hop_B/s"},
  }};

  std::array<double, 13> rates{};
  Json resourceJson = Json::array();
  for (std::size_t i = 0; i < resources.size(); ++i) {
    rates[i] = resources[i].rate;
    resourceJson.push_back(Json{{"resource_id", i},
                                {"id", resources[i].id},
                                {"enabled", resources[i].enabled},
                                {"rate", resources[i].rate},
                                {"unit", resources[i].unit}});
  }

  if (moduleCount > std::numeric_limits<long long>::max() / coreCount)
    invalid("architecture.system_region_width_mm/system_region_height_mm", "overflow the optional system Core count");
  const long long systemCoreCount = moduleCount * coreCount;
  Json coreDerived{{"sram_mib_per_core", arch.sramMiB},
                   {"noc_width_bits", arch.nocBits},
                   {"N_vec", core.n_vec},
                   {"N_pe", core.n_pe},
                   {"L0_mm", l0},
                   {"P_int4_per_core_ops_per_s", pInt4},
                   {"P_vec_per_core_ops_per_s", pVec},
                   {"sram_bytes_per_core", static_cast<std::uint64_t>(sramBytesPerCore)},
                   {"N_core", coreCount}};
  Json dieDerived{{"geometry_candidate", arch.geometryCandidate},
                  {"minimum_width_mm", minW},
                  {"minimum_height_mm", minH},
                  {"A_req_mm2", areaReq},
                  {"width_mm", dieW},
                  {"height_mm", dieH},
                  {"area_mm2", dieArea},
                  {"core_columns_nx", arch.nx},
                  {"core_rows_ny", arch.ny},
                  {"core_count", coreCount},
                  {"core_pitch_x_mm", arch.pitchX},
                  {"core_pitch_y_mm", arch.pitchY},
                  {"die_extra_width_mm", arch.extraWidth},
                  {"die_extra_height_mm", arch.extraHeight},
                  {"aspect_ratio", aspect},
                  {"bump_budget_Bps", bumpBps},
                  {"bump_budget_GBps", bumpBps / 1.0e9},
                  {"d2d_bidirectional_Bps", d2dBiBps},
                  {"d2d_oneway_Bps", d2dBiBps / 2.0}};
  const double capacity = 2.0 * arch.hbmN * 36.0e9;
  Json hbmDerived{{"sku", "Micron MT65B18G16120A00QH-92:A"},
                  {"stacks_per_row", arch.hbmN},
                  {"total_stacks", 2 * arch.hbmN},
                  {"capacity_bytes", capacity},
                  {"peak_Bps", hbmTotalBps},
                  {"row_peak_Bps", arch.hbmN * 1.1776e12},
                  {"row_width_mm", hbmRowWidth},
                  {"row_overhang", hbmRowWidth > dieW},
                  {"module_width_mm", moduleW},
                  {"module_height_mm", moduleH},
                  {"placement_mode", arch.placement},
                  {"row_to_ingress", perf.topology == "distributed_ideal_v2"
                                          ? Json::array({nullptr, nullptr})
                                          : Json(rowToIngress)},
                  {"ingress_evidence", ingressEvidence}};
  Json systemDerived;
  if (arch.hasRegion) systemDerived = Json{{"layout_profile", "legacy_region_fit_10pct_v1"},
                     {"layout_provenance", "compatibility_assumption_not_hardware_docx"},
                     {"module_interconnect_fraction", 0.10},
                     {"region_width_mm", arch.regionWidth},
                     {"region_height_mm", arch.regionHeight},
                     {"effective_module_width_mm", effectiveW},
                     {"effective_module_height_mm", effectiveH},
                     {"module_columns", moduleColumns},
                     {"module_rows", moduleRows},
                     {"module_count", moduleCount},
                     {"core_count", systemCoreCount},
                     {"hbm_capacity_bytes", moduleCount * capacity},
                     {"peak_int4_ops_per_s", systemCoreCount * pInt4},
                     {"peak_vec_fp32_eq_ops_per_s", systemCoreCount * pVec},
                     {"eta_core_die", coreCount * l0 * l0 / dieArea},
                     {"eta_core_region", systemCoreCount * l0 * l0 /
                                             (arch.regionWidth * arch.regionHeight)},
                     {"eta_module_region", moduleCount * moduleW * moduleH /
                                               (0.90 * arch.regionWidth * arch.regionHeight)}};
  else systemDerived = Json{{"status", "not_requested"},
                         {"eta_core_die", coreCount * l0 * l0 / dieArea}};
  Json hardwareSummary{{"die_core_count", coreCount},
                       {"hbm_capacity_bytes", capacity},
                       {"hbm_row_peak_Bps", Json::array({arch.hbmN * 1.1776e12,
                                                          arch.hbmN * 1.1776e12})},
                       {"peak_int4_ops_per_s", coreCount * pInt4},
                       {"peak_vec_fp32_eq_ops_per_s", coreCount * pVec}};
  Json derived{{"core", coreDerived},
               {"die", dieDerived},
               {"hbm", hbmDerived},
               {"system", systemDerived},
               {"hardware_summary", hardwareSummary},
               {"resources", resourceJson},
               {"source_revision", kSourceRevision},
               {"core_resource_table", {{"id", kCoreTableId}, {"rows_sha256", kCoreTableSha256}}},
               {"performance_identity", {{"timing_topology_profile", perf.topology},
                                         {"routing_profile", perf.routing},
                                         {"physical_topology_validation", "unverified"}}}};

  validateDerivedMirrors(input, derived);
  Json canonical = input;
  canonical.erase("derived");
  canonical.erase("derived_hardware");
  canonical["source_constants"] = sourceConstants();
  canonical["core_resource_table"] = Json{{"id", kCoreTableId}, {"rows_sha256", kCoreTableSha256}};
  Json fingerprintInput = canonical;
  fingerprintInput.erase("search_profile");
  fingerprintInput.erase("validation_constraints");
  if (fingerprintInput.contains("architecture"))
    fingerprintInput["architecture"].erase("upstream_search_profile");
  Json fingerprintDerived = derived;
  fingerprintDerived["die"].erase("A_req_mm2");
  Json fingerprintPayload{{"input", fingerprintInput}, {"derived", fingerprintDerived}};
  Hardware result;
  result.input = std::move(canonical);
  result.derived = std::move(derived);
  result.rates = rates;
  result.fingerprint = "sha256:" + sha256(fingerprintPayload.dump());
  return result;
}

}  // namespace coarse_die
