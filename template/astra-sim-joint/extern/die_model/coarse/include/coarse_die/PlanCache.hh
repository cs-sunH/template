#pragma once

#include <memory>

#include <json/json.hpp>

#include "coarse_die/Hardware.hh"
#include "coarse_die/Plan.hh"

namespace coarse_die {

// Opt-in disk plan-cache hooks (roofline stage 3, WP4).
//
// The C ABI cache (Capi) and host-side compile caches (the repositories'
// CoarseDieRuntime) are separate in-process caches but share one
// normalization contract (semantic_task_key) and, when the environment
// variable COARSE_DIE_PLAN_CACHE_DIR is set to a directory, one on-disk
// cache with one key scheme. These hooks let a host compile path consult
// and populate that disk cache without going through the C ABI.
//
// Both functions are no-ops that never touch the filesystem when the
// variable is unset or empty. A load returns nullptr on any miss, parse
// failure, fingerprint mismatch, or version mismatch (fail-closed: the
// caller falls back to a normal compile and the bad entry is repaired by
// the next store). A load hit is NOT "compile zero": the caller still owns
// validation (require_complete_prediction and the normalized-task
// collision check) exactly as for a freshly compiled plan.

/// Try to load a previously compiled plan from the disk cache. Returns
/// null when disabled, missing, or invalid (never throws for cache
/// states; filesystem errors are swallowed as misses).
std::shared_ptr<const Plan> try_load_disk_plan(const Hardware& hardware,
                                               const nlohmann::json& task);

/// Persist a freshly compiled plan to the disk cache (atomic
/// temporary-file rename; any failure is swallowed -- the cache is an
/// accelerator, never a correctness dependency).
void store_disk_plan_for(const Hardware& hardware,
                         const nlohmann::json& task, const Plan& plan);

}  // namespace coarse_die
