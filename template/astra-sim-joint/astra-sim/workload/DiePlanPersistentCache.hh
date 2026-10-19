/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
DiePlanPersistentCache.hh -- opt-in cross-run persistent DiePlan cache on
the AstraSim adapter side (W5 of the joint提速改造, 2026-10-17).

Mirrors the C2 persistent estimate cache of extern/die_model/src/Capi.cc at
the plan layer: compiled DiePlan objects survive the process through one
binary file so a warm run skips DieModel::compile for every (spec,
hardware) key the previous clean run already compiled.  Semantics:

  - Opt-in via the environment variable DIE_PLAN_CACHE_PATH.  Unset or
    empty keeps the store fully disabled: no file is read, no file is
    written, get() returns null and put() is a no-op -- byte-for-byte the
    pre-W5 dispatch behavior of DieModelAdapter::issue_die.
  - Keys are exactly DieModel::plan_cache_key(spec, hw) (content
    addressing: same key implies same plan), so the store inherits the
    library's own key versioning discipline.  The FILE-level fingerprint
    (DIE_PLAN_CACHE_VERSION_STAMP + DIE_PLAN_CACHE_BUILD_STAMP, stamped by
    the repository root CMakeLists at configure time) invalidates the
    whole store when the estimate-relevant sources, the compiler/flag set
    or the die_model version change, and the file-level hardware key
    (DieModel::plan_cache_hardware_key) rejects stores saved for another
    resolved hardware.  Known blind spot (accepted, same scope as the C2
    estimate cache): the stamp folds implementation .cc files only -- a
    header-only change (inline logic, struct defaults) does NOT
    invalidate a saved store.  The content-addressed keys plus the
    per-entry plan_cache_key self-check cover the spec/hardware surface;
    fold the headers in when the fingerprint format tag is next bumped.
  - Fail-closed load: absent file, fingerprint/hardware-key mismatch, or
    ANY decode anomaly (including a per-entry plan_cache_key self-check
    mismatch) discards the whole file and starts from an empty store; the
    corrupt file itself is not touched until a fresh compile dirties the
    store again.
  - Save happens ONLY on the clean-exit path (door B): main_online.cc
    calls flush_clean_exit() after the Sys objects are gone, and the save
    itself is best effort (temp file + atomic rename, every failure
    swallowed).  A harvested process (SIGTERM reaping) intentionally
    persists nothing; no signal handler is installed.
  - Loaded entries warm the in-memory LRU WITHOUT marking the store dirty
    (insert_loaded semantics, same as C2), so a read-only run never
    rewrites the file.  Only put() of a freshly compiled plan marks it
    dirty.
  - DiePlanData::flat_shape is deliberately NOT serialized: a loaded plan
    starts with a null flat_shape and rebuilds it lazily through
    DiePlan::flat_shape_handle() at the first Execution construction
    (joint perf item semantics, implemented by detail::build_flat_shape
    in extern/die_model/src/Execution.cc).

The store is a process-level singleton.  The simulator drives every
Workload on one event thread (the same single-driver contract as
DieModel::PlanCache and DieModelAdapter's cumulative timing counters), so
the singleton state carries no locking.
*/

#ifndef __DIE_PLAN_PERSISTENT_CACHE_HH__
#define __DIE_PLAN_PERSISTENT_CACHE_HH__

#include <memory>
#include <list>
#include <string>
#include <unordered_map>

#include "DieModel.hh"

namespace AstraSim {

class DiePlanPersistentCache {
  public:
    // Process-wide instance (single event thread; no locking).
    static DiePlanPersistentCache& instance();

    // Whether DIE_PLAN_CACHE_PATH opted in.  False when the env var is
    // unset or empty; every other member is then a no-op.
    bool enabled();

    // Persistent lookup for one plan_cache_key.  `runtime_hw` seeds the
    // first call: it fixes the expected DieModel::plan_cache_hardware_key
    // the loaded file header must carry (mismatch = whole file discarded,
    // fail-closed) and is ignored on every later call.  Returns null when
    // disabled, on a miss, or when the file failed to load; never
    // compiles and never throws.
    std::shared_ptr<const DieModel::DiePlan> get(
        const std::string& key, const DieModel::Hardware& runtime_hw);

    // Records a freshly compiled plan and marks the store dirty (the next
    // flush_clean_exit() persists it).  No-op when disabled; never throws.
    void put(const std::string& key,
             std::shared_ptr<const DieModel::DiePlan> plan);

    // Clean-exit persistence (door B): the ONLY save site, reserved for
    // the main_online.cc success path.  Writes solely when a put() marked
    // the store dirty since load; best effort and noexcept so it can sit
    // directly in front of `return 0`.
    static void flush_clean_exit() noexcept;

    // Test-only seam: return the singleton to its pristine pre-first-touch
    // state (env unread, store empty, not dirty).  Production code never
    // calls this; the unit tests use it to simulate a fresh process
    // between the save and the load halves of a round trip.
    static void reset_for_test();

  private:
    DiePlanPersistentCache() = default;

    void init_once();
    void load_file();
    void save_file() const noexcept;

    bool initialized_ = false;
    bool enabled_ = false;
    bool loaded_ = false;
    bool dirty_ = false;
    std::string path_;
    std::string fingerprint_;
    std::string hw_key_;
    // LRU mirror of DieModel::PlanCache (capacity 4096), newest first.
    std::list<std::pair<std::string,
                        std::shared_ptr<const DieModel::DiePlan>>> lru_;
    std::unordered_map<
        std::string,
        std::list<std::pair<
            std::string, std::shared_ptr<const DieModel::DiePlan>>>::iterator>
        index_;
};

}  // namespace AstraSim

#endif /* __DIE_PLAN_PERSISTENT_CACHE_HH__ */
