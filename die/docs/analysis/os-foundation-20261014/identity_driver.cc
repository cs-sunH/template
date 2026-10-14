#include "coarse_die/Execution.hh"
#include <fstream>
#include <iostream>
namespace coarse_die::detail {
std::vector<std::shared_ptr<Plan>> enumerate_mappings(const Hardware&, const nlohmann::json&);
}
using nlohmann::json;
json read(const char* path) { std::ifstream in(path); return json::parse(in); }
int main() {
  json config=read("die/configs/example_hardware.json");
  json result=json::array();
  for (double kappa : {4.0, 0.3125}) {
    config["performance_profile"]["kappa_quant_eqops"]=kappa;
    config["performance_profile"]["kappa_acc_convert_eqops"]=kappa;
    auto hw=coarse_die::resolve_hardware(config);
    const json examples=read("die/configs/example_tasks.json");
    for (auto task : examples.at("specs")) {
      auto plan=coarse_die::compile(hw,task);
      result.push_back({{"task",task},{"hardware",hw.fingerprint},{"hash",plan->fingerprint},{"summary",plan->summary}});
    }
    for (const char* profile : {"int4_matrix_vec_fp32_assumed_v1","fp32_vec_matrix_assumed_v1"})
      for (const char* dtype : {"fp32","bf16","int4","int32"})
        for (auto shape : {json{{"m",17},{"n",65},{"k",129}},json{{"m",1},{"n",1},{"k",1}},json{{"m",33},{"n",129},{"k",65}}}) {
          json task={{"op_kind","gemm"},{"tensor_shapes",shape},{"execution_profile",profile},
            {"storage_dtype",{{"activations",dtype},{"weights",dtype},{"output",dtype}}},
            {"layer_instances",2},{"weight_passes",3}};
          try {
            auto plan=coarse_die::compile(hw,task);
            result.push_back({{"task",task},{"hardware",hw.fingerprint},{"hash",plan->fingerprint},{"summary",plan->summary}});
          } catch(const std::exception& e) {
            result.push_back({{"task",task},{"hardware",hw.fingerprint},{"error",e.what()}});
          }
        }
  }
  std::cout<<result.dump(2)<<'\n';
}
