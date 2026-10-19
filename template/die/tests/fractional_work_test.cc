#include "test_support.hh"
#include "coarse_die/Capi.h"
#include <limits>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {
Hardware hardware(double quant, double acc) {
  auto input=read_json(std::string(COARSE_DIE_SOURCE_DIR)+"/configs/example_hardware.json");
  input["performance_profile"]["kappa_quant_eqops"]=quant;
  input["performance_profile"]["kappa_acc_convert_eqops"]=acc;
  auto hw=resolve_hardware(input);
  // Synthetic slow gate permits observable partial service in nanosecond ticks.
  hw.rates[COMPUTE_GATE]=1e-6;
  hw.derived["resources"][COMPUTE_GATE]["rate"]=1e-6;
  return hw;
}
json task(bool strict=false) {
  return {{"op_kind","gemm"},{"tensor_shapes",{{"m",1},{"n",1},{"k",1}}},
      {"storage_dtype",{{"activations",strict?"fp32":"bf16"},
                         {"weights",strict?"fp32":"bf16"},{"output","int4"}}},
      {"execution_profile",strict?"fp32_vec_matrix_assumed_v1":"int4_matrix_vec_fp32_assumed_v1"}};
}
long double planned(const Plan& p) {
  long double result=0;
  for(const auto& e:p.epochs)for(const auto& a:e.activities)result+=planned_vec_eqops(a);
  return result;
}
void finish(Runtime& r,std::uint64_t handle) {
  for(unsigned i=0;i<128&&!r.terminal_finish_ns(handle);++i)r.advance_to(r.next_event().value());
  require(r.terminal_finish_ns(handle).has_value(),"bounded fractional execution finishes");
}
template<class F> void rejected(F body,const std::string& message) {
  try { body(); } catch(const std::exception&) { return; }
  throw std::runtime_error(message);
}
json take(const char* text) {
  require(text!=nullptr,die_coarse_last_error_v2());
  const auto result=json::parse(text);
  die_coarse_free_result_v2(text);
  return result;
}
void capi_roundtrip(const Hardware& hw) {
  DieHandle model=0,runtime=0,execution=0,clone=0;
  require(die_coarse_create_model_v2(hw.input.dump().c_str(),&model)==DIE_OK,"fractional ABI model");
  const auto batch=take(die_coarse_compile_batch_v2(model,json{{"specs",json::array({task()})}}.dump().c_str()));
  require(batch.at("errors").at(0).is_null(),"fractional ABI compile: "+batch.at("errors").dump());
  require(batch.at("results").at(0).at("plan_schema")=="coarse_die_plan_v3" &&
          batch.at("results").at(0).at("equivalent_work_schema")=="dyadic_equivalent_work_v1",
          "ABI JSON identifies the fractional payload contract");
  const auto plan=batch.at("results").at(0).at("plan_handle").get<DieHandle>();
  require_close(batch.at("results").at(0).at("summary").at("vec_equivalent_ops").get<double>(),4.5,0,"ABI summary fractional cost");
  require(die_coarse_create_runtime_v2(model,&runtime)==DIE_OK,"fractional ABI runtime");
  require(die_coarse_update_capacity_v2(runtime,COMPUTE_GATE,1e-6,0)==DIE_OK,"ABI slow-gate fixture");
  const DieOwnerV2 own{sizeof(DieOwnerV2),2,1,1,1,1};
  require(die_coarse_submit_plan_v2(runtime,plan,&own,0,&execution)==DIE_OK,"fractional ABI submit");
  const auto snap=take(die_coarse_export_snapshot_v2(runtime));
  require(die_coarse_import_snapshot_v2(model,snap.dump().c_str(),&clone)==DIE_OK,"ABI imports fractional embedded Plan");
  for(unsigned i=0;i<128&&!take(die_coarse_query_progress_v2(execution)).at("terminal").get<bool>();++i) {
    const auto next=take(die_coarse_next_event_v2(runtime)).at("next_event_ns").get<std::uint64_t>();
    require(die_coarse_advance_to_v2(runtime,next)==DIE_OK,"fractional ABI advance");
  }
  require_close(take(die_coarse_query_progress_v2(execution)).at("served").at("vec_eqops").get<double>(),4.5,0,"ABI serves fraction without integer rounding");
  require(die_coarse_destroy_execution_v2(execution)==DIE_OK,"ABI execution cleanup");
  const auto clone_execution=take(die_coarse_list_executions_v2(clone)).at("executions").at(0).at("execution_handle").get<DieHandle>();
  require(die_coarse_cancel_execution_v2(clone_execution,0)==DIE_OK,"ABI clone cancellation");
  require(die_coarse_destroy_execution_v2(clone_execution)==DIE_OK,"ABI clone execution cleanup");
  require(die_coarse_destroy_runtime_v2(clone)==DIE_OK,"ABI clone cleanup");
  require(die_coarse_destroy_runtime_v2(runtime)==DIE_OK,"ABI runtime cleanup");
  require(die_coarse_release_plan_v2(plan)==DIE_OK,"ABI plan cleanup");
  require(die_coarse_destroy_model_v2(model)==DIE_OK,"ABI model cleanup");
}
void body() {
  require(die_coarse_abi_version()==2,"C ABI v2 version preserved");
  auto hw=hardware(1.25,0.75);
  const auto p=compile(hw,task());
  // Two BF16 operands, one INT32 partial and one INT4 output conversion.
  require(planned(*p)==4.5L,"fractional conversion ledger equals independent element count");
  require_close(p->summary.at("vec_equivalent_ops").get<double>(),4.5,0,"fractional summary");
  long double gate=0;
  for(const auto& e:p->epochs)for(const auto& a:e.activities)gate+=a.work[COMPUTE_GATE];
  const long double expected=2.0L/(hw.derived.at("core").at("P_int4_per_core_ops_per_s").get<double>()*0.9L)+
      4.5L/(hw.derived.at("core").at("P_vec_per_core_ops_per_s").get<double>()*0.9L);
  require_close(static_cast<double>(gate),static_cast<double>(expected),1e-20,"budget and fractional ledger agree");
  const auto wire=plan_to_json(*p);
  require(wire.at("schema")=="coarse_die_plan_v3","fractional Plan explicitly versioned");
  require(plan_to_json(*plan_from_json(wire))==wire,"fractional Plan canonical roundtrip");
  auto broken=wire;broken["schema"]="coarse_die_plan_v2";
  rejected([&]{plan_from_json(broken);},"v2 cannot silently accept fractional work");
  for(double invalid:{-0.25,1.0,std::numeric_limits<double>::infinity()}) {
    auto bad=*p;bad.fingerprint.clear();
    bad.epochs.at(1).activities.at(0).vec_eqops_fraction=invalid;
    rejected([&]{plan_to_json(bad);},"invalid fractional remainder rejected");
  }
  Runtime r(hw);const auto h=r.submit(p,owner(1,1,1),0);
  r.advance_to(p->launch_ns);
  // Stop after the load, then inside compute, before completion correction.
  r.advance_to(r.next_event().value());
  auto end=r.next_event().value();
  r.advance_to(r.now_ns()+(end-r.now_ns())/2);
  const double partial=r.query_progress(h).at("served").at("vec_eqops").get<double>();
  require(partial>0&&partial<4.5,"partial fractional service is retained");
  auto clone=Runtime::import_snapshot(hw,r.export_snapshot());
  require(clone->query_progress(h)==r.query_progress(h),"fractional snapshot preserves partial progress");
  finish(r,h);finish(*clone,h);
  require(r.query_progress(h)==clone->query_progress(h),"fractional fork terminal equality");
  require_close(r.query_progress(h).at("served").at("vec_eqops").get<double>(),4.5,0,"terminal work exact");
  auto strict=compile(hw,task(true));
  require(planned(*strict)==3.25L,"strict FP32 matrix plus INT4 output uses kappa_quant");
  auto integer_hw=hardware(4,2);
  auto integer=compile(integer_hw,task());
  require(plan_to_json(*integer).at("schema")=="coarse_die_plan_v2","integer Plan retains v2");
  require(planned(*integer)==14,"integer coefficient charge remains exact");
  auto zero=compile(hardware(0,0),task());
  require(planned(*zero)==0,"zero conversion coefficient adds no synthetic Vec work");
  rejected([&]{compile(hardware(1e300,2),task());},"huge finite coefficient rejected before integer cast");
  // 0.1 and 0.2 are binary64 inputs whose exact weighted sum needs more
  // significand bits than binary64. The dyadic wire format retains those bits.
  auto decimal_coefficients=compile(hardware(0.1,0.2),task());
  const auto decimal_wire=plan_to_json(*decimal_coefficients);
  require(plan_to_json(*plan_from_json(decimal_wire))==decimal_wire,"non-binary-decimal coefficients roundtrip exactly");
  require(planned(*decimal_coefficients)==3*static_cast<long double>(0.1)+static_cast<long double>(0.2),"declared binary64 coefficients retained without truncation");
  // Both complete loss (.25) and rounding upward (.75) must be rejected.
  for (const std::uint64_t significand : {1u,3u}) {
    auto precision_bad=wire;
    precision_bad["epochs"][1]["activities"][0]["vec_eqops"]=std::uint64_t{1}<<63;
    precision_bad["epochs"][1]["activities"][0]["vec_eqops_fraction"]={{"significand",significand},{"exponent",-2}};
    precision_bad.erase("fingerprint");
    precision_bad["fingerprint"]=stable_hash(precision_bad.dump());
    try {
      plan_from_json(precision_bad);
      throw std::runtime_error("unrepresentable imported fraction accepted");
    } catch(const std::overflow_error& e) {
      require(std::string(e.what()).find("precision")!=std::string::npos,
              "fraction rejected on precision boundary before hash or runtime mutation");
    }
  }
  json attention={{"op_kind","attention"},{"tensor_shapes",{{"q_heads",1},{"kv_heads",1},{"head_dim",1}}},
      {"spans",json::array({json{{"q_len",1},{"kv_len",1}}})},
      {"storage_dtype",{{"activations","int4"},{"kv","int4"},{"output","fp32"}}},
      {"execution_profile","fp32_vec_matrix_assumed_v1"}};
  for(const char* fusion:{"fused","materialized"}) {
    attention["fusion_policy"]=fusion;
    const auto a=compile(hw,attention);
    // QK/PV: four matrix ops; 13 score and one output-normalization op;
    // one Q, K, V element each decoded at κ=1.25.
    require(planned(*a)==21.75L,"strict Attention decodes all INT4 source roles at kappa_quant");
    const bool full=std::string(fusion)=="materialized";
    require(a->summary.at("hbm_read_bytes_total")== (full?11:3) &&
            a->summary.at("hbm_write_bytes_total")== (full?12:4),
            "FP32 Attention output has no extra storage cast or duplicate write");
    require(plan_to_json(*plan_from_json(plan_to_json(*a)))==plan_to_json(*a),"Attention fractional primitive roundtrip");
  }
  json ffn={{"op_kind","ffn_region"},{"tensor_shapes",{{"m",1},{"hidden",1},{"ffn",1}}},
      {"storage_dtype",{{"activations","int4"},{"weights","int4"},{"output","fp32"}}},
      {"execution_profile","fp32_vec_matrix_assumed_v1"},{"fusion_policy","fused"}};
  const auto f=compile(hw,ffn);
  // Three scalar matrix MACs: six ops; SwiGLU math: sixteen; input and
  // three source weights each decoded once: four * 1.25.
  require(planned(*f)==27.0L,"strict fused FFN INT4 decoding is weighted independently of math");
  const auto tiny_hw=hardware(std::numeric_limits<double>::denorm_min(),0);
  const auto tiny_plan=compile(tiny_hw,task());
  Runtime tiny_runtime(tiny_hw);
  const auto tiny_handle=tiny_runtime.submit(tiny_plan,owner(1,1,1),0);
  finish(tiny_runtime,tiny_handle);
  const long double tiny_expected=3*static_cast<long double>(std::numeric_limits<double>::denorm_min());
  const auto tiny_counts=tiny_runtime.counters();
  const auto tiny_text=tiny_counts.at("served_by_category").at("comp").at("vec_eqops_exact_decimal").get<std::string>();
  require(std::strtold(tiny_text.c_str(),nullptr)==tiny_expected,
          "small nonzero work survives exact counter serialization");
  const auto tiny_clone=Runtime::import_snapshot(tiny_hw,tiny_runtime.export_snapshot());
  require(tiny_clone->counters().at("served_by_category").at("comp").at("vec_eqops_exact_decimal")==tiny_text,
          "snapshot preserves subnormal weighted-work history");
  capi_roundtrip(hw);
}
}
int main(){return run_test("fractional_work_test",body);}
