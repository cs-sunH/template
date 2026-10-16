#include "coarse_die/Execution.hh"
#include "coarse_die/Prediction.hh"
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

using namespace coarse_die;
using nlohmann::json;

json read_json(const std::string& path) {
  std::ifstream file(path);
  if(!file)throw std::runtime_error("cannot read "+path);
  return json::parse(std::string(std::istreambuf_iterator<char>(file),{}));
}

json probe(const Hardware& hw,const std::string& name,json task) {
  json result={{"name",name},{"task",task}};
  try {
    const auto plan=compile(hw,task);
    result["compiled"]=true;
    result["plan_wire"]=plan_to_json(*plan);
    result["summary"]=plan->summary;
    result["estimate"]=estimate_alone(hw,*plan);
    try {
      require_complete_prediction(*plan);
      result["consumer_policy"]="accepted";
    } catch(const std::exception& error) {
      result["consumer_policy"]="rejected";
      result["consumer_policy_error"]=error.what();
    }
  } catch(const std::exception& error) {
    result["compiled"]=false;result["error"]=error.what();
  }
  return result;
}

int main(int argc,char** argv) {
  if(argc!=2)return 2;
  auto config=read_json(argv[1]);
  config["architecture"]["core_columns_nx"]=1;
  config["architecture"]["core_rows_ny"]=1;
  config["performance_profile"]["launch_ns_per_macro"]=0;
  config["performance_profile"]["memory_setup_ns_per_macro"]=0;
  const auto zero=resolve_hardware(config);
  json residual={{"op_kind","residual"},{"tensor_shapes",{{"m",1},{"hidden",257}}},
      {"storage_dtype","bf16"},{"execution_profile","fp32_vec_matrix_assumed_v1"}};
  json ffn={{"op_kind","ffn_region"},{"tensor_shapes",{{"m",1},{"hidden",8},{"ffn",16},
      {"mlp_variant","swiglu"}}},{"storage_dtype","bf16"},{"fusion_policy","materialized"}};
  json attention={{"op_kind","attention_region"},{"tensor_shapes",{{"q_heads",1},{"kv_heads",1},{"head_dim",8}}},
      {"spans",json::array({{{"q_tokens",1},{"kv_tokens",2}}})},
      {"storage_dtype","bf16"},{"fusion_policy","materialized"}};
  json qkv={{"op_kind","qkv_rope"},{"tensor_shapes",{{"m",1},{"hidden",8},{"q_heads",2},
      {"kv_heads",1},{"head_dim",4},{"position_encoding","rope"}}},{"storage_dtype","bf16"}};
  json norm={{"op_kind","norm"},{"tensor_shapes",{{"m",1},{"hidden",257}}},{"storage_dtype","bf16"}};
  json output={{"hardware_input",config},{"hardware_fingerprint",zero.fingerprint},{"cases",json::array()}};
  for(const auto& entry:std::vector<std::pair<std::string,json>>{{"residual",residual},{"ffn_materialized",ffn},
      {"attention_materialized",attention},{"qkv_rope",qkv},{"norm_control",norm}}) {
    for(const auto& repeats:std::vector<std::pair<unsigned,unsigned>>{{1,1},{2,1},{1,2},{2,2}}) {
      auto task=entry.second;task["layer_instances"]=repeats.first;task["weight_passes"]=repeats.second;
      output["cases"].push_back(probe(zero,entry.first+"_layers"+std::to_string(repeats.first)+
          "_passes"+std::to_string(repeats.second),task));
    }
  }
  auto untyped=residual;untyped["layer_instances"]=2;untyped["region_ops"]=json::array({residual,residual});
  output["cases"].push_back(probe(zero,"untyped_region_ops_expected_unsupported",untyped));
  config["performance_profile"]["launch_ns_per_macro"]=1000;
  const auto delayed=resolve_hardware(config);residual["layer_instances"]=2;
  output["cases"].push_back(probe(delayed,"positive_launch_residual_R2_control",residual));
  std::cout<<output.dump(2)<<"\n";
}
