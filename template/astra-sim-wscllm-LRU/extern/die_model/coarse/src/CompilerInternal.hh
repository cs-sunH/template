#pragma once

#include "coarse_die/Plan.hh"

namespace coarse_die::detail {

// Canonicalize all accepted task spellings for internal cache keys and
// compiler entry points. The returned object is the exact task identity that
// is retained by the compiled Plan.
nlohmann::json normalize_task(const Hardware& hardware,
                             const nlohmann::json& task);

// Compile a task that has already passed normalize_task(). This is private to
// the compiler/C ABI implementation so the public compile() contract remains
// unchanged while cache misses avoid normalizing the same task twice.
std::shared_ptr<const Plan> compile_normalized(
    const Hardware& hardware, const nlohmann::json& normalized_task);

// Materialize the selected GEMM trial's region audit without changing its
// ready delay, resource work, availability status, or description budget.
void finalize_region_overhead_audit(const Hardware& hardware, Plan& plan);

// Rebuild the chosen internal ranking trial through the same GEMM builder,
// emitting its full diagnostics without another Runtime estimate or search.
std::shared_ptr<Plan> materialize_gemm_compile_candidate(const Hardware&,
                                                       const Plan& candidate);

}  // namespace coarse_die::detail
