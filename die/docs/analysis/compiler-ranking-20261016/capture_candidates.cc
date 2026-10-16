// Capture the unchanged public enumerator, including losing candidates.
#include "coarse_die/Plan.hh"
#include "../../../src/CompilerInternal.hh"
#include <array>
#include <fstream>
#include <iostream>

namespace coarse_die::detail {
std::vector<std::shared_ptr<Plan>> enumerate_mappings(const Hardware&,const nlohmann::json&);
}
int main(int argc,char** argv) {
  using namespace coarse_die;using json=nlohmann::json;using u64=std::uint64_t;
  try {
    if(argc!=2)throw std::invalid_argument("one fresh output path required");
    std::ifstream input(std::string(COARSE_DIE_SOURCE_DIR)+"/configs/example_hardware_explicit.json");
    const auto base=json::parse(input);json result={{"scope","public software candidates, not hardware calibration"},{"cases",json::array()}};
    std::size_t count=0;
    for(const std::string profile:{"int4_matrix_vec_fp32_assumed_v1","fp32_vec_matrix_assumed_v1"})
        for(double coefficient:{0.5,4.0}) {
      auto config=base;config["performance_profile"]["execution_profile"]=profile;
      config["performance_profile"]["kappa_quant_eqops"]=coefficient;
      const auto hw=resolve_hardware(config);
      for(const auto shape:{std::array<u64,3>{{1,1,1}},{{7,5,9}},{{128,128,128}},{{65,129,257}}})
          for(const std::string storage:{"int4","bf16"})for(u64 repeats:{1,3}) {
        const json task={{"op_kind","gemm"},{"tensor_shapes",{{"m",shape[0]},{"n",shape[1]},{"k",shape[2]}}},
            {"storage_dtype",storage},{"layer_instances",repeats},{"weight_passes",1}};
        const auto normalized=detail::normalize_task(hw,task);json plans=json::array();
        for(const auto& plan:detail::enumerate_mappings(hw,normalized)){plans.push_back(plan_to_json(*plan));++count;}
        result["cases"].push_back({{"hardware",hw.input},{"task",normalized},{"candidates",plans}});
      }
    }
    result["candidate_count"]=count;std::ofstream output(argv[1]);if(!output)throw std::runtime_error("cannot open candidate output");
    output<<result.dump(2)<<'\n';std::cout<<"candidate_count="<<count<<'\n';return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
