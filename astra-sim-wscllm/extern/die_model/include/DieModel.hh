/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
DieModel.hh -- convenience facade of the die-internal cost model library
(contract §1/§4).

Stage A1 delivered: make_hardware / hardware_summary_json / core_table /
tx8_reference_block (types in DieTypes.hh).  Stage A2 adds derive_workload
(types in DieTypes.hh), compile / estimate_alone and the plan cache API
below, implemented in src/Workload.cc, src/Mapping.cc and src/CostModel.cc.
Stage A3 adds the externally driven Execution state machine (declared in
DieTypes.hh, implemented in src/Execution.cc) and the C ABI
(include/DieCApi.h, src/Capi.cc, SHARED die_model_cabi target).
*/

#pragma once

#include "DieTypes.hh"

#include <vector>

namespace DieModel {

// Strictly validates the die hardware configuration and derives all resource
// scalars (contract §3.1/§3.3).  Accepted JSON shapes:
//
//   wrapped (the §3.3 resolved_die_model.json form, minus derived):
//     { "schema_version": 1, "profile": "mesh_die_v1",
//       "die": { "sram-mb-per-core": 0.75, "core-type": "large",
//                "compute-area-mm2": 800, "hbm-per-side": 3,
//                "hbm-sides": 2, "noc-width-bits": 512 } }
//   bare (the die node itself; profile defaults to mesh_die_v1):
//     { the six die keys exactly }
//
// Legal (area, per-side, sides) combinations and NoC widths are exactly the
// design doc §3.1 table (the contract §3.1 delegates there); see the stage
// A1 report for the (800, 2, 2) discrepancy in the contract's examples.
//
// schema_version/profile/die are optional/required as shown; a top-level
// "derived" object is accepted and ignored (it is this library's own §3.3
// output mirror, so resolved files round-trip).  Every other key is
// rejected.  Throws DieModelError on any violation.
Hardware make_hardware(const nlohmann::json& config);

// Serializes the §3.3 summary: schema_version, profile, the die node
// verbatim and the derived block (mesh_n/cores/pe_peak_tflops/vec_peak_teq/
// sram_usable_bytes_per_core/link_peak_bytes/hbm_count/hbm_capacity_bytes/
// hbm_bandwidth_bytes).  Used later by the C ABI's die_hardware_summary();
// the resolver must take derived values from here, never re-derive in
// Python (contract §3.3).
nlohmann::json hardware_summary_json(const Hardware& hw);

// See DieTypes.hh (embedded 14-row §3.1 table).
std::vector<CoreTableEntry> core_table();

// The TX8 reference block (design doc §2): fixed reference inputs, 13 timed
// groups computed from the §2.2 workload formulas, no Excel time lookup.
// All times are SI seconds in double.
Tx8Result tx8_reference_block();

// §4.2 workload determination of one op (design doc §4.2; types in
// DieTypes.hh).  Pure arithmetic on the spec; no tiling decisions.
WorkloadQuantities derive_workload(const DieOpSpec& spec);

// Compiles one op on one hardware into an immutable plan (contract §4).
// mesh_die_v1: enumerates the §5.1 candidate set, evaluates every candidate
// with the SAME isolated-run stage model as estimate_alone and keeps the
// shortest (ties: smaller SRAM, less traffic, fixed tile lexicographic
// order).  tx8_reference_v1: builds the §1.1 aggregated-streaming reference
// envelope (PE(Fop)->vec(Eop) budget chain + memory stream per §2.2) and
// rejects any spec that is not the §2.1 reference shape.  Both profiles
// share the workload semantic checks.  Throws DieModelError.
std::shared_ptr<const DiePlan> compile(const DieOpSpec& spec,
                                       const Hardware& hw);

// Isolated-run estimate of a plan: a die-level conservative list schedule of
// the stage DAG on the PE/VEC/NOC/MEM slots (§6.1).  Candidate evaluation
// inside compile() uses exactly this model.
CostResult estimate_alone(const DiePlan& plan);

// Plan cache key (frozen FNV-1a hash; see DieTypes.hh / the stage A2
// report) and the LRU cache itself (capacity 4096 by default).
std::string plan_cache_key(const DieOpSpec& spec, const Hardware& hw);

// Canonical hardware sub-key shared by every plan cache key of one hardware
// (stage E2): the exact "hw{...}" substring plan_cache_key embeds, exposed so
// callers that want to SHARE one PlanCache across model instances (e.g. all
// Sys of a simulation process) can index their registry by the same resolved
// hardware summary the plan keys use -- two hardware objects with equal keys
// produce equal plan keys, so sharing never changes hit semantics.
std::string plan_cache_hardware_key(const Hardware& hw);

// class Execution (see DieTypes.hh): construct it directly from the plan
// shared_ptr returned by compile().  The C ABI wraps the same machine for
// the Python side (include/DieCApi.h).

}  // namespace DieModel
