#pragma once

#include "GemmOSAccessCounter.hh"
#include "coarse_die/Plan.hh"
#include <array>

namespace coarse_die::detail {

// One actual single-Core K-consumption, using the compiler's resource/path
// builders. These immutable stage facts are a prerequisite for a bounded
// readiness descriptor; they do not select a mapping or execute a lifecycle.
// A nonfinal K has an empty Store Activity. No buffer credit is attached here:
// the lifecycle must reserve inputs/output slots at the true dependency edges.
// Multi-Core cohorts require payload-weighted mapping and slowest-Core work,
// and are rejected until those semantics are supplied.
std::array<Activity, 3> gemm_os_single_core_resource_facts(
    const Hardware&, const gemm_os_access_counter::Input&);

}  // namespace coarse_die::detail
