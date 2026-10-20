#include "test_support.hh"
#include "coarse_die/Capi.h"
#include "coarse_die/Memory.hh"

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {

// Child-process protocol: "plan_disk_cache_test --child <cache-dir> <out>"
// compiles the standard task in a brand-new process (empty in-memory cache,
// environment inherited from the parent) and writes the public compile row
// plus cache diagnostics to <out>. Real processes keep the disk-hit scenario
// honest instead of approximating it with a second in-process model.
constexpr const char* kEnvName = "COARSE_DIE_PLAN_CACHE_DIR";
constexpr int kEnvelopeSchemaVersion = 1;
constexpr const char* kEnvelopeFormat = "coarse_die_plan_cache_v1";
std::string g_self;

json gemm_task(std::uint64_t m, std::uint64_t n = 1, std::uint64_t k = 1) {
  return {{"op_kind", "gemm"},
          {"tensor_shapes", {{"m", m}, {"n", n}, {"k", k}}},
          {"storage_dtype", {{"weights", "bf16"}, {"activations", "bf16"},
                             {"output", "bf16"}}},
          {"execution_profile", "int4_matrix_vec_fp32_assumed_v1"},
          {"layer_instances", 1},
          {"weight_passes", 1}};
}

class CModel {
 public:
  explicit CModel(const Hardware& hardware) {
    const auto config = hardware.input.dump();
    require(die_coarse_create_model_v2(config.c_str(), &handle_) == DIE_OK,
            std::string("C ABI model create failed: ") + die_coarse_last_error_v2());
  }
  CModel(const CModel&) = delete;
  CModel& operator=(const CModel&) = delete;
  ~CModel() {
    if (handle_) (void)die_coarse_destroy_model_v2(handle_);
  }
  DieHandle get() const { return handle_; }

 private:
  DieHandle handle_ = 0;
};

class PlanHandles {
 public:
  PlanHandles() = default;
  PlanHandles(const PlanHandles&) = delete;
  PlanHandles& operator=(const PlanHandles&) = delete;
  ~PlanHandles() {
    for (const auto handle : handles_)
      if (handle) (void)die_coarse_release_plan_v2(handle);
  }
  void track(const json& response) {
    for (const auto& row : response.at("results")) {
      if (row.is_object() && row.contains("plan_handle"))
        handles_.push_back(row.at("plan_handle").get<DieHandle>());
    }
  }

 private:
  std::vector<DieHandle> handles_;
};

json take_json(const char* raw, const char* api) {
  require(raw != nullptr,
          std::string(api) + " failed: " + die_coarse_last_error_v2());
  const std::string text(raw);
  die_coarse_free_result_v2(raw);
  return json::parse(text);
}

json compile_batch(DieHandle model, const json& specs) {
  const auto request = json{{"specs", specs}}.dump();
  return take_json(die_coarse_compile_batch_v2(model, request.c_str()),
                   "compile_batch");
}

class TempDir {
 public:
  TempDir() {
    std::vector<char> pattern{'/', 't', 'm', 'p', '/', 'c', 'o', 'a', 'r',
                              's', 'e', '_', 'd', 'i', 'e', '_', 'p', 'l',
                              'a', 'n', '_', 'd', 'i', 's', 'k', '_', 'c',
                              'a', 'c', 'h', 'e', '_', 'X', 'X', 'X', 'X',
                              'X', 'X', '\0'};
    const char* made = ::mkdtemp(pattern.data());
    require(made != nullptr, "mkdtemp failed for plan cache test directory");
    path_ = made;
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  ~TempDir() {
    if (path_.empty()) return;
    for (const auto& name : list()) (void)::remove((path_ + "/" + name).c_str());
    (void)::rmdir(path_.c_str());
  }
  const std::string& path() const { return path_; }
  std::vector<std::string> list() const {
    DIR* handle = ::opendir(path_.c_str());
    require(handle != nullptr, "opendir failed: " + path_);
    std::vector<std::string> names;
    while (const dirent* entry = ::readdir(handle)) {
      const std::string name = entry->d_name;
      if (name != "." && name != "..") names.push_back(name);
    }
    ::closedir(handle);
    std::sort(names.begin(), names.end());
    return names;
  }

 private:
  std::string path_;
};

void write_file(const std::string& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  require(out.good(), "cannot write " + path);
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
  out.flush();
  require(out.good(), "short write to " + path);
}

struct FileStamp {
  std::string bytes;
  std::int64_t sec = 0;
  std::int64_t nsec = 0;
};

FileStamp stamp_of(const std::string& path) {
  struct stat st {};
  require(::stat(path.c_str(), &st) == 0, "stat failed: " + path);
  return {read_text(path), static_cast<std::int64_t>(st.st_mtim.tv_sec),
          static_cast<std::int64_t>(st.st_mtim.tv_nsec)};
}

std::string single_entry_path(const TempDir& dir, const char* where) {
  const auto names = dir.list();
  std::string joined;
  for (const auto& name : names) joined += (joined.empty() ? "" : ", ") + name;
  require(names.size() == 1,
          std::string(where) + ": cache directory holds exactly one entry, found " +
              std::to_string(names.size()) + ": " + joined);
  const auto& name = names.front();
  require(name.size() == 64 + 5 && name.compare(64, 5, ".json") == 0 &&
              name.compare(0, 5, ".tmp-") != 0,
          "entry filename is the 64-hex cache key plus .json with no temp residue: " +
              name);
  for (std::size_t i = 0; i < 64; ++i) {
    const char c = name[i];
    require((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'),
            "entry filename uses only lowercase hex key characters");
  }
  return dir.path() + "/" + name;
}

void run_child(const std::string& cache_dir, const std::string& out,
               std::uint64_t m, std::uint64_t n, std::uint64_t k) {
  // std::to_string() temporaries stay alive through the execl() call.
  const std::string ms = std::to_string(m), ns = std::to_string(n), ks = std::to_string(k);
  const pid_t pid = ::fork();
  require(pid >= 0, "fork for plan cache child failed");
  if (pid == 0) {
    ::execl(g_self.c_str(), g_self.c_str(), "--child", cache_dir.c_str(),
            out.c_str(), ms.c_str(), ns.c_str(), ks.c_str(),
            static_cast<char*>(nullptr));
    ::_exit(127);
  }
  int status = 0;
  require(::waitpid(pid, &status, 0) == pid, "waitpid for plan cache child failed");
  if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0)) {
    std::string report;
    try {
      report = read_text(out);
    } catch (const std::exception&) {
      report = "<no child report>";
    }
    require(false, "plan cache child exited unsuccessfully: " + report);
  }
}

int child_main(const std::string& cache_dir, const std::string& out,
               std::uint64_t m, std::uint64_t n, std::uint64_t k) {
  try {
    (void)cache_dir;  // the environment variable inherited from the parent selects it
    const auto hardware = example_hardware();
    const json task = gemm_task(m, n, k);
    CModel model(hardware);
    PlanHandles handles;
    const auto response = compile_batch(model.get(), json::array({task}));
    handles.track(response);
    const auto& row = response.at("results").at(0);
    write_file(out, json{{"plan_hash", row.at("plan_hash")},
                         {"summary", row.at("summary")},
                         {"cache_entries", response.at("cache_entries")},
                         {"cache_accounted_bytes",
                          response.at("cache_accounted_bytes")}}.dump());
    return 0;
  } catch (const std::exception& e) {
    write_file(out, json{{"error", e.what()}}.dump());
    return 1;
  }
}

void require_reference_row(const json& row, const Plan& reference,
                           const std::string& message) {
  require(row.is_object() && row.at("plan_hash").is_string() &&
              row.at("plan_hash") == reference.fingerprint &&
              row.at("summary") == reference.summary,
          message);
}

// Three reported regimes share one equivalence claim: compile without any
// cache, compile hitting the disk entry in a new process, and a same-process
// in-memory hit all return the reference plan hash, summary, result-row
// shape, and entry count. cache_accounted_bytes is deliberately NOT compared
// across regimes: it accounts the retained container capacities of the live
// Plan object, and a Plan rebuilt by plan_from_json legitimately retains
// different capacities than a freshly compiled one, so only same-entry
// repeats (regime 3) must keep it bit-stable.
void three_regimes_behavior_equivalence() {
  const auto hardware = example_hardware();
  const json task = gemm_task(23, 7, 5);
  const auto reference = compile(hardware, task);

  // Regime 1: no cache at all (default OFF).
  ::unsetenv(kEnvName);
  TempDir untouched;
  std::size_t cold_bytes = 0;
  {
    CModel model(hardware);
    PlanHandles handles;
    const auto cold = compile_batch(model.get(), json::array({task}));
    handles.track(cold);
    require(cold.at("errors").at(0).is_null() &&
                cold.at("cache_entries") == 1,
            "cold compile without the variable succeeds and retains one entry");
    require_reference_row(cold.at("results").at(0), *reference,
                          "cold compile without the variable matches the reference plan");
    cold_bytes = cold.at("cache_accounted_bytes").get<std::size_t>();
  }
  require(untouched.list().empty(),
          "with the variable unset no cache directory is created or written");

  // Regime 2: disk hit in a genuinely new process.
  TempDir cache;
  TempDir work;
  ::setenv(kEnvName, cache.path().c_str(), 1);
  run_child(cache.path(), work.path() + "/first.json", 23, 7, 5);
  const auto first = read_json(work.path() + "/first.json");
  require(first.at("plan_hash") == reference->fingerprint &&
              first.at("summary") == reference->summary &&
              first.at("cache_entries") == 1 &&
              first.at("cache_accounted_bytes").get<std::size_t>() == cold_bytes,
          "first process cold-compiles with the cache enabled and matches the reference row and diagnostics");
  const std::string entry = single_entry_path(cache, __func__);

  run_child(cache.path(), work.path() + "/second.json", 23, 7, 5);
  const auto second = read_json(work.path() + "/second.json");
  require(second.at("plan_hash") == reference->fingerprint &&
              second.at("summary") == reference->summary &&
              second.at("cache_entries") == 1 &&
              second.at("cache_accounted_bytes").is_number_unsigned() &&
              second.at("cache_accounted_bytes").get<std::size_t>() > 0,
          "new-process disk hit returns the identical plan hash, summary, and entry count");
  require(single_entry_path(cache, __func__) == entry,
          "a disk hit neither adds entries nor rewrites the existing one");
  const std::string stored = read_text(entry);
  const json envelope = json::parse(stored);
  require(envelope.at("cache_schema_version") == kEnvelopeSchemaVersion &&
              envelope.at("cache_format") == kEnvelopeFormat &&
              envelope.at("source_identity").is_string() &&
              !envelope.at("source_identity").get<std::string>().empty(),
          "stored entry carries the expected envelope version metadata");
  require(envelope.at("plan").at("fingerprint") == reference->fingerprint,
          "stored canonical plan payload carries the reference fingerprint");

  // Cross-model in-process disk hit: a fresh model has an empty in-memory
  // LRU, so the same load path runs without fork/exec.
  std::size_t disk_hit_bytes = 0;
  {
    CModel model(hardware);
    PlanHandles handles;
    const auto disk_hit = compile_batch(model.get(), json::array({task}));
    handles.track(disk_hit);
    require(disk_hit.at("cache_entries") == 1 &&
                disk_hit.at("cache_accounted_bytes").is_number_unsigned(),
            "in-process cross-model disk hit retains exactly one entry");
    disk_hit_bytes = disk_hit.at("cache_accounted_bytes").get<std::size_t>();
    require_reference_row(disk_hit.at("results").at(0), *reference,
                          "in-process cross-model disk hit matches the reference plan");

    // Regime 3: same-process in-memory hit on the same model.
    const auto memory_hit = compile_batch(model.get(), json::array({task}));
    handles.track(memory_hit);
    require(memory_hit.at("cache_entries") == 1 &&
                memory_hit.at("cache_accounted_bytes").get<std::size_t>() ==
                    disk_hit_bytes,
            "same-process repeat stays an in-memory hit with unchanged accounting");
    require_reference_row(memory_hit.at("results").at(0), *reference,
                          "same-process in-memory hit matches the reference plan");
    require(memory_hit.at("results").at(0).at("plan_handle") !=
                disk_hit.at("results").at(0).at("plan_handle"),
            "every successful row still receives a fresh plan handle");
  }
  require(read_text(entry) == stored,
          "in-memory hits leave the disk entry byte-identical");
}

// A compile through a tampered entry must fail closed: normal compilation,
// correct result, and the unusable file removed and rewritten fresh.
void corrupt_entry_fails_closed() {
  const auto hardware = example_hardware();
  const json task = gemm_task(29, 3, 11);
  const auto reference = compile(hardware, task);
  TempDir cache;
  TempDir work;
  ::setenv(kEnvName, cache.path().c_str(), 1);
  run_child(cache.path(), work.path() + "/seed.json", 29, 3, 11);
  const std::string entry = single_entry_path(cache, __func__);
  const std::string good_text = read_text(entry);

  const auto repair_check = [&](const char* what) {
    CModel model(hardware);
    PlanHandles handles;
    const auto response = compile_batch(model.get(), json::array({task}));
    handles.track(response);
    require(response.at("errors").at(0).is_null() &&
                response.at("cache_entries") == 1,
            std::string(what) + ": fail-closed compile succeeds normally");
    require_reference_row(response.at("results").at(0), *reference,
                          std::string(what) + ": fail-closed compile returns the reference plan");
    require(single_entry_path(cache, __func__) == entry &&
                read_text(entry) == good_text,
            std::string(what) + ": the unusable entry is deleted and rewritten byte-identically");
  };

  write_file(entry, "{ this is not valid json");
  repair_check("unparseable cache bytes");
  write_file(entry, good_text.substr(0, good_text.size() / 2));
  repair_check("truncated cache file");

  json tampered = json::parse(good_text);
  tampered["plan"]["launch_ns"] =
      tampered["plan"].at("launch_ns").get<std::uint64_t>() + 1;
  write_file(entry, tampered.dump());
  repair_check("inner plan payload tampering (fingerprint mismatch)");
}

// Envelope version metadata is a hard gate: schema, format tag, source
// identity, and structural completeness must all match this build.
void version_mismatch_fails_closed() {
  const auto hardware = example_hardware();
  const json task = gemm_task(31, 2, 13);
  const auto reference = compile(hardware, task);
  TempDir cache;
  TempDir work;
  ::setenv(kEnvName, cache.path().c_str(), 1);
  run_child(cache.path(), work.path() + "/seed.json", 31, 2, 13);
  const std::string entry = single_entry_path(cache, __func__);
  const std::string good_text = read_text(entry);

  const auto mismatch_check = [&](const char* what) {
    CModel model(hardware);
    PlanHandles handles;
    const auto response = compile_batch(model.get(), json::array({task}));
    handles.track(response);
    require(response.at("errors").at(0).is_null(),
            std::string(what) + ": version-mismatched entry fails closed to a normal compile");
    require_reference_row(response.at("results").at(0), *reference,
                          std::string(what) + ": fallback compile returns the reference plan");
    require(read_text(single_entry_path(cache, __func__)) == good_text,
            std::string(what) + ": foreign entry is replaced by a fresh valid one");
  };

  json future = json::parse(good_text);
  future["cache_schema_version"] = kEnvelopeSchemaVersion + 7;
  write_file(entry, future.dump());
  mismatch_check("future schema version");

  json foreign = json::parse(good_text);
  foreign["source_identity"] = "foreign-build-stamp";
  write_file(entry, foreign.dump());
  mismatch_check("foreign source identity");

  json format = json::parse(good_text);
  format["cache_format"] = "coarse_die_plan_cache_v999";
  write_file(entry, format.dump());
  mismatch_check("foreign format tag");

  json missing = json::parse(good_text);
  missing.erase("plan");
  write_file(entry, missing.dump());
  mismatch_check("missing plan member");

  write_file(entry, "{}");
  mismatch_check("empty envelope");
}

// Default OFF purity: with the variable unset the library neither reads,
// rewrites, deletes, nor creates cache files even when a valid entry sits on
// disk and the in-memory cache is empty.
void disabled_default_never_touches_disk() {
  const auto hardware = example_hardware();
  const json task = gemm_task(37, 11, 2);
  const auto reference = compile(hardware, task);
  TempDir cache;
  TempDir work;
  ::setenv(kEnvName, cache.path().c_str(), 1);
  run_child(cache.path(), work.path() + "/enabled.json", 37, 11, 2);
  const std::string entry = single_entry_path(cache, __func__);
  const FileStamp before = stamp_of(entry);

  ::unsetenv(kEnvName);
  {
    CModel model(hardware);
    PlanHandles handles;
    const auto response = compile_batch(model.get(), json::array({task}));
    handles.track(response);
    require(response.at("errors").at(0).is_null() &&
                response.at("cache_entries") == 1,
            "disabled compile succeeds with a cold in-memory cache");
    require_reference_row(response.at("results").at(0), *reference,
                          "disabled compile returns the reference plan");
  }
  const FileStamp after = stamp_of(entry);
  require(after.bytes == before.bytes && after.sec == before.sec &&
              after.nsec == before.nsec,
          "with the variable unset the existing disk entry is left byte-identical with an untouched mtime");
  const auto names = cache.list();
  require(names.size() == 1 && cache.path() + "/" + names.front() == entry,
          "with the variable unset no entries are created, deleted, or renamed");
}

void body() {
  require(die_coarse_abi_version() == 2,
          "disk plan cache keeps C ABI version 2");
  three_regimes_behavior_equivalence();
  corrupt_entry_fails_closed();
  version_mismatch_fails_closed();
  disabled_default_never_touches_disk();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 7 && std::string(argv[1]) == "--child")
    return child_main(argv[2], argv[3], std::stoull(argv[4]),
                      std::stoull(argv[5]), std::stoull(argv[6]));
  if (argc > 0 && argv[0]) g_self = argv[0];
  return run_test("plan_disk_cache_test", body);
}
