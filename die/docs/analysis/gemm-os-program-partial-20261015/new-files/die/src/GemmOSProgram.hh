#pragma once

// Internal bridge from the public Plan marker to the canonical single-Core
// OS geometry and per-access-class Runtime facts. Both compiler validation
// and Runtime submit/import use this factory; no facts are serialized as
// caller-controlled Plan payload.

#include "GemmOSReadiness.hh"
#include "coarse_die/Plan.hh"

#include <memory>

namespace coarse_die::detail {

// Returns nullptr for an ordinary Epoch Plan. For an opted-in Program Plan,
// reconstructs and validates the geometry, wave budget, OS mapping identity,
// and aggregate summary before returning the immutable descriptor.
std::shared_ptr<const gemm_os_readiness::Descriptor>
make_gemm_os_program(const Hardware& hardware, const Plan& plan);

}  // namespace coarse_die::detail
