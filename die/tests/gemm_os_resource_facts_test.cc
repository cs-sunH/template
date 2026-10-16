#include "test_support.hh"
#include "../src/GemmOSResourceFacts.hh"

#include <array>
#include <iostream>

namespace coarse_die::detail {
std::vector<std::shared_ptr<Plan>> enumerate_mappings(
    const Hardware&, const nlohmann::json&);
}
using namespace coarse_die;
using namespace coarse_die_test;
namespace access = coarse_die::detail::gemm_os_access_counter;

int main() {
  auto config=read_json(std::string(COARSE_DIE_SOURCE_DIR)+"/configs/example_hardware.json");
  config["architecture"]["sram_mib_per_core"]=0.25;
  config["architecture"]["noc_width_bits"]=64;
  config["architecture"]["core_columns_nx"]=1;
  config["architecture"]["core_rows_ny"]=1;
  config["architecture"]["die_geometry"]="isotropic";
  config["performance_profile"]["kappa_quant_eqops"]=0.3125;
  config["performance_profile"]["kappa_acc_convert_eqops"]=0.375;
  const auto hw=resolve_hardware(config);
  const std::array<const char*,4> names{{"fp32","bf16","int4","int32"}};
  const std::array<access::StorageDType,4> dtypes{{access::StorageDType::Fp32,
      access::StorageDType::Bf16,access::StorageDType::Int4,access::StorageDType::Int32}};
  std::size_t checked=0;
  for(bool int4 : {true,false}) for(std::size_t dtype=0;dtype<names.size();++dtype) {
    json task={{"op_kind","gemm"},{"tensor_shapes",{{"m",3},{"n",5},{"k",7}}},
        {"storage_dtype",{{"activations",names[dtype]},{"weights",names[dtype]},
                          {"output",names[dtype]}}},
        {"execution_profile",int4?"int4_matrix_vec_fp32_assumed_v1":"fp32_vec_matrix_assumed_v1"}};
    auto normalized=compile(hw,task)->task;
    std::shared_ptr<const Plan> one;
    for(const auto& candidate : detail::enumerate_mappings(hw,normalized)) {
      if(candidate->summary.at("tm")==3 && candidate->summary.at("tn")==5 &&
         candidate->summary.at("tk")==7 &&
         candidate->summary.at("dataflow_id")=="output_stationary_v2") {
        one=candidate; break;
      }
    }
    require(static_cast<bool>(one),"single-output OS candidate exists");
    access::Input in;
    in.m=3; in.n=5; in.k=7; in.first_k=true; in.final_k=true;
    in.a_storage=dtypes[dtype]; in.w_storage=dtypes[dtype]; in.output_storage=dtypes[dtype];
    in.matrix_path=int4?access::MatrixPath::Int4PE:access::MatrixPath::StrictFp32Vec;
    in.kappa_quant=0.3125; in.kappa_acc=0.375;
    const auto facts=detail::gemm_os_single_core_resource_facts(hw,in);
    require(one->epochs.size()==3,"one output/one K has ordered L,C,S");
    for(std::size_t stage=0;stage<3;++stage) {
      require(one->epochs[stage].activities.size()==1,"solo stage has one activity");
      const auto& old=one->epochs[stage].activities.front();
      const auto& actual=facts[stage];
      require(actual.role==old.role && actual.hbm_bytes==old.hbm_bytes &&
              actual.int4_ops==old.int4_ops && actual.vec_eqops==old.vec_eqops &&
              actual.vec_eqops_fraction==old.vec_eqops_fraction,
              "single-K stage preserves physical payload and typed operations");
      for(std::size_t r=0;r<resource_count;++r)
        require_close(actual.work[r],old.work[r],1e-13*std::max(1.0,old.work[r]),
                      "single-K resource work matches actual compiler stage");
      require_close(actual.private_time_s,old.private_time_s,1e-18,
                    "PRIVATE local supply/compute bound matches compiler");
      require(actual.canonical_q==old.canonical_q,"same canonical fairness basis");
      require(actual.credit_delta==std::array<double,2>{},
              "resource facts do not invent readiness credit curves");
    }
    in.final_k=false;
    const auto interior=detail::gemm_os_single_core_resource_facts(hw,in);
    require(std::all_of(interior[2].work.begin(),interior[2].work.end(),
                       [](double v){return v==0;}),"nonfinal K produces no Store work");
    require(interior[2].hbm_bytes==Activity{}.hbm_bytes,"nonfinal K produces no HBM write");
    ++checked;
  }
  bool rejected=false;
  try {
    access::Input in; in.m=1; in.n=1; in.k=1; in.first_k=true; in.final_k=true;
    detail::gemm_os_single_core_resource_facts(example_hardware(),in);
  } catch(const std::invalid_argument& e) {
    rejected=std::string(e.what()).find("single-Core")!=std::string::npos;
  }
  require(rejected,"multi-Core grids require their own mapping facts");
  std::cout<<"PASS: "<<checked<<" typed single-K compiler/resource-fact comparisons\n";
}
