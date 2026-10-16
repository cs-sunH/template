#include "test_support.hh"
#include "coarse_die/Capi.h"
#include "coarse_die/Prediction.hh"

#include <array>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;
namespace {
Hardware hardware(double setup=0) {
  auto config=read_json(std::string(COARSE_DIE_SOURCE_DIR)+"/configs/example_hardware_explicit.json");
  config["performance_profile"]["launch_ns_per_macro"]=0;
  config["performance_profile"]["memory_setup_ns_per_macro"]=setup;
  return resolve_hardware(config);
}
std::vector<json> tasks() {
  return {
      {{"op_kind","residual"},{"tensor_shapes",{{"m",1},{"hidden",257}}}},
      {{"op_kind","norm"},{"tensor_shapes",{{"m",1},{"hidden",257}}}},
      {{"op_kind","gemm"},{"tensor_shapes",{{"m",7},{"n",5},{"k",9}}}},
      {{"op_kind","ffn_region"},{"tensor_shapes",{{"m",1},{"hidden",8},{"ffn",16},
          {"mlp_variant","swiglu"}}},{"fusion_policy","materialized"}},
      {{"op_kind","attention_region"},{"tensor_shapes",{{"q_heads",1},{"kv_heads",1},{"head_dim",8}}},
          {"spans",json::array({{{"q_tokens",1},{"kv_tokens",2}}})},{"fusion_policy","materialized"}},
      {{"op_kind","qkv_rope"},{"tensor_shapes",{{"m",1},{"hidden",8},{"q_heads",2},
          {"kv_heads",1},{"head_dim",4},{"position_encoding","rope"}}}}};
}
void require_unavailable(const Plan& plan) {
  require(plan.summary.value("prediction_status",std::string())=="not_available" &&
      plan.summary.at("alone_estimate_status")=="not_available",
      "zero launch/setup cannot certify an aggregated repeated region's unmodeled barriers");
  bool rejected=false;
  try {require_complete_prediction(plan);}catch(const std::invalid_argument& e) {
    rejected=std::string(e.what()).find("prediction_not_available")!=std::string::npos;
  }
  require(rejected,"the same production consumer policy must reject zero-overhead repeated body estimates");
}
json take(const char* text) {
  require(text!=nullptr,std::string("ABI query failed: ")+die_coarse_last_error_v2());
  const std::string copy(text);die_coarse_free_result_v2(text);return json::parse(copy);
}
void verify() {
  const auto zero=hardware();const auto setup=hardware(17);
  auto specifications=tasks();
  for(auto task:specifications)for(const auto counts:{std::array<unsigned,2>{{2,1}},{{1,2}},{{2,3}}}) {
    task["layer_instances"]=counts[0];task["weight_passes"]=counts[1];
    const auto plan=compile(zero,task);const auto delayed=compile(setup,task);
    require_unavailable(*plan);require_unavailable(*delayed);
    require(plan->launch_ns==0 && delayed->launch_ns==0 &&
        plan_to_json(*plan).at("epochs")==plan_to_json(*delayed).at("epochs"),
        "unavailable repeated bodies retain their work, credit, fairness and physical execution");
    const auto& audit=plan->summary.at("fixed_region_overheads");
    require(audit.at("instances")==counts[0]*counts[1] && audit.at("modeled_instances")==0 &&
        audit.at("required_total_delay_ns_ceil")==0 && audit.at("applied_ready_delay_ns")==0 &&
        audit.at("ready_gate_description_count")==0,
        "a zero-delay audit must report missing repeated barriers without inventing waiting work");
    const auto result=estimate_alone(zero,*plan);
    require(result.at("status")=="not_available" && result.contains("estimated_latency_ns"),
        "full estimates must keep reviewable body diagnostics with an explicit unavailable status");
  }

  DieHandle model=0;
  const auto config=zero.input.dump();
  require(die_coarse_create_model_v2(config.c_str(),&model)==DIE_OK,"create zero-overhead ABI model");
  for(auto& task:specifications)task["layer_instances"]=2;
  const auto request=json{{"specs",specifications}}.dump();
  const auto compiled=take(die_coarse_compile_batch_v2(model,request.c_str()));
  const auto full=take(die_coarse_estimate_batch_v2(model,request.c_str()));
  const auto cost=take(die_coarse_estimate_cost_batch_v2(model,request.c_str()));
  for(std::size_t i=0;i<specifications.size();++i) {
    require(compiled.at("errors")[i].is_null() && full.at("errors")[i].is_null() && cost.at("errors")[i].is_null(),
        "unavailable predictions remain explicit successful diagnostic rows across ABI query modes");
    const auto& c=compiled.at("results")[i];const auto& f=full.at("results")[i];const auto& k=cost.at("results")[i];
    require(c.at("summary").at("prediction_status")=="not_available" &&
        c.at("summary").at("alone_estimate_status")=="not_available" &&
        f.at("status")=="not_available" && k.at("status")=="not_available" &&
        c.at("plan_hash")==f.at("plan_fingerprint") && c.at("plan_hash")==k.at("plan_hash"),
        "compile, full and cached cost queries must expose the same unavailable repeated Plan");
    require(die_coarse_release_plan_v2(c.at("plan_handle").get<DieHandle>())==DIE_OK,"release diagnostic plan");
  }
  require(die_coarse_destroy_model_v2(model)==DIE_OK,"release zero-overhead model");
}
}
int main(){return run_test("repeated_region_availability_test",verify);}
