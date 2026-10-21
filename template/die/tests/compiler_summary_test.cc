#include "test_support.hh"
#include "../src/CompilerInternal.hh"
#include <array>
#include <limits>
#include <string>

namespace coarse_die::detail {
std::vector<std::shared_ptr<Plan>> enumerate_mappings(const Hardware&,const nlohmann::json&);
std::vector<std::shared_ptr<Plan>> enumerate_mappings_for_compile(const Hardware&,const nlohmann::json&);
}
using namespace coarse_die;
using namespace coarse_die_test;
using u64=std::uint64_t;
namespace {
json spec(u64 m,u64 n,u64 k,const std::string& storage,u64 repeats=1) {
  return {{"op_kind","gemm"},{"tensor_shapes",{{"m",m},{"n",n},{"k",k}}},
      {"storage_dtype",storage},{"layer_instances",repeats},{"weight_passes",1}};
}
void all_candidates_have_identical_public_materialization() {
  auto config=read_json(std::string(COARSE_DIE_SOURCE_DIR)+"/configs/example_hardware_explicit.json");
  std::size_t compared=0;
  for(const std::string profile:{"int4_matrix_vec_fp32_assumed_v1","fp32_vec_matrix_assumed_v1"})
      for(double coefficient:{0.5,4.0}) {
    config["performance_profile"]["execution_profile"]=profile;
    config["performance_profile"]["kappa_quant_eqops"]=coefficient;
    const auto hw=resolve_hardware(config);
    for(const auto shape:{std::array<u64,3>{{1,1,1}},{{7,5,9}},{{128,128,128}},{{65,129,257}}})
        for(const std::string storage:{"int4","bf16"})for(u64 repeats:{1,3}) {
      const auto normalized=detail::normalize_task(hw,spec(shape[0],shape[1],shape[2],storage,repeats));
      const auto full=detail::enumerate_mappings(hw,normalized);
      const auto trials=detail::enumerate_mappings_for_compile(hw,normalized);
      require(!full.empty() && full.size()==trials.size(),"diagnostic deferral preserves every software candidate and order");
      for(std::size_t i=0;i<full.size();++i) {
        const auto before=plan_to_json(*trials[i]);
        auto restored=detail::materialize_gemm_compile_candidate(hw,*trials[i]);
        detail::finalize_region_overhead_audit(hw,*restored);
        restored->fingerprint=plan_to_json(*restored).at("fingerprint").get<std::string>();
        require(plan_to_json(*restored)==plan_to_json(*full[i]),
            "every candidate's task, physical work, curves, readiness, complete diagnostics and fingerprint remain identical");
        require(before==plan_to_json(*trials[i]),"materializing a winner cannot mutate its ranking trial");
        ++compared;
      }
    }
  }
  std::cout<<"all_candidate_full_wire_comparisons="<<compared<<'\n';
}
void range_errors_are_preserved() {
  const auto hw=example_hardware();
  const u64 maximum=std::numeric_limits<u64>::max();
  for(const auto& task:{spec(maximum,2,2,"bf16"),spec(2,maximum,2,"int4"),
      spec(64,64,64,"bf16",maximum),spec(1,1,1,"fp32",maximum)}) {
    const auto normalized=detail::normalize_task(hw,task);
    std::string full_error,trial_error;
    try {(void)detail::enumerate_mappings(hw,normalized);}catch(const std::exception& e){full_error=e.what();}
    try {(void)detail::enumerate_mappings_for_compile(hw,normalized);}catch(const std::exception& e){trial_error=e.what();}
    require(!full_error.empty() && full_error==trial_error,
        "omitting diagnostic storage cannot omit a range, workload or credit representability rejection");
  }
}
void rank_key_drift_is_rejected() {
  const auto hw=example_hardware();const auto normalized=detail::normalize_task(hw,spec(7,5,9,"bf16"));
  const auto trials=detail::enumerate_mappings_for_compile(hw,normalized);
  Plan changed=*trials.front();changed.summary["hbm_read_bytes_total"]=changed.summary.at("hbm_read_bytes_total").get<u64>()+1;
  const auto before=plan_to_json(changed);bool rejected=false;
  try {(void)detail::materialize_gemm_compile_candidate(hw,changed);}
  catch(const std::invalid_argument& e){rejected=std::string(e.what()).find("rank key changed")!=std::string::npos;}
  require(rejected && before==plan_to_json(changed),"a materialized identity/rank mismatch fails without modifying the trial");
}
void run(){all_candidates_have_identical_public_materialization();range_errors_are_preserved();rank_key_drift_is_rejected();}
}
int main(){return run_test("compiler_summary_test",run);}
