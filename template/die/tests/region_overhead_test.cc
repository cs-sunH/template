#include "test_support.hh"
#include "coarse_die/Capi.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {
using u64=std::uint64_t;

Hardware hardware(double launch=0,double setup=0) {
  auto cfg=read_json(std::string(COARSE_DIE_SOURCE_DIR)+"/configs/example_hardware_explicit.json");
  cfg["architecture"]["core_columns_nx"]=1;
  cfg["architecture"]["core_rows_ny"]=1;
  cfg["performance_profile"]["launch_ns_per_macro"]=launch;
  cfg["performance_profile"]["memory_setup_ns_per_macro"]=setup;
  return resolve_hardware(cfg);
}
json residual(u64 repeats=1,u64 elements=1) {
  return {{"op_kind","residual"},{"tensor_shapes",{{"m",1},{"hidden",elements}}},
      {"storage_dtype","bf16"},{"execution_profile","fp32_vec_matrix_assumed_v1"},
      {"layer_instances",repeats},{"weight_passes",1}};
}
void drain(Runtime& runtime) {
  unsigned count=0;
  while(!runtime.drained()) {
    const auto next=runtime.next_event();
    require(next.has_value() && *next>runtime.now_ns() && ++count<1000,
        "region fixture has a bounded advancing completion sequence");
    runtime.advance_to(*next);
  }
}
u64 finish(const Hardware& hw,std::shared_ptr<const Plan> plan) {
  Runtime runtime(hw);const auto handle=runtime.submit(std::move(plan),owner(1,1,1),0);
  drain(runtime);return *runtime.terminal_finish_ns(handle);
}
void same_body(const Plan& first,const Plan& second) {
  // Compare the public epoch wire, including physical work, Q, PRIVATE,
  // payload, credits and activity identities. A ready delay adds no activity.
  require(plan_to_json(first).at("epochs")==plan_to_json(second).at("epochs"),
      "nonservice region delay must preserve every physical activity and Q/PRIVATE");
}
void no_service(const json& progress) {
  const auto& served=progress.at("served");
  require(served.at("vec_eqops")==0 && served.at("int4_ops")==0 &&
      served.at("compute_gate_work")==0,"waiting/cancelled region served no arithmetic");
  for(const auto& row:served.at("hbm_bytes_by_row_direction"))for(const auto& value:row)
    require(value==0,"waiting/cancelled region served no physical HBM payload");
  for(const auto& a:progress.at("activities"))
    require(a.at("progress")==0 && a.at("rate")==0 && a.at("alpha")==0,
        "initial ready wait must have no progress, resource share or owner alpha");
}
void no_busy(const Runtime& runtime) {
  const auto c=runtime.counters();
  require(c.at("active_activities")==0 && c.at("active_owners")==0 &&
      c.at("hbm_busy_ns")==0 && c.at("hbm_shared_ns")==0 && c.at("compute_busy_ns")==0,
      "ready wait itself must create neither active owners nor busy time");
}

void single_region_response_and_quantization() {
  const auto zero=hardware();const auto task=residual(1,257);const auto body=compile(zero,task);
  const auto body_finish=finish(zero,body);
  struct Case {double launch,setup;u64 expected;};
  for(const auto c:std::array<Case,4>{{{750,0,750},{0,1250,1250},{750,1250,2000},{0.25,0.5,1}}}) {
    const auto hw=hardware(c.launch,c.setup);const auto p=compile(hw,task);
    same_body(*body,*p);
    require(p->launch_ns==c.expected && finish(hw,p)==body_finish+c.expected,
        "integer ready wait adds exactly the hand-calculated region delay");
    require(p->summary.at("estimated_latency_ns")==body_finish+c.expected,
        "compiler estimate must include exactly the same initial wait");
    const auto& audit=p->summary.at("fixed_region_overheads");
    require(audit.at("instances")==1 && audit.at("modeled_instances")==1 &&
        audit.at("hbm_instances")==1 && audit.at("applied_ready_delay_ns")==c.expected &&
        audit.at("timing_status")=="modeled_integer_ns_ready_gate",
        "one complete region is modeled once, regardless of tile count");
    require(service_description_count(*p)==service_description_count(*body)+1 &&
        p->summary.at("service_description_count")==service_description_count(*p) &&
        audit.at("ready_gate_description_count")==1,
        "the ready gate must be included in the finite description budget");
    const auto& old_curve=body->summary.at("service_curves")[0];
    const auto& curve=p->summary.at("service_curves")[0];
    require(curve.at("description_count").get<u64>()==old_curve.at("description_count").get<u64>()+1 &&
        curve.at("description_count").get<u64>()<=curve.at("description_budget").get<u64>(),
        "the first primitive's budget must include the extra ready switch");
    const auto report=estimate_alone(hw,*p);
    require(report.at("status")=="ok_assumed" && report.at("plan_schema")=="coarse_die_plan_v4" &&
        report.at("granularity_identity").at("schema")=="coarse_die_plan_v4",
        "full estimate must report the actual curved Plan v4 schema");
    if(c.launch==0.25) {
      require(audit.at("required_total_delay_ns_whole")==0 &&
          audit.at("required_total_delay_ns_ceil")==1 &&
          equivalent_fraction_from_json(audit.at("required_total_delay_ns_fraction"))==0.75L,
          "0.25ns launch plus 0.5ns setup is 0.75ns, rounded once to 1ns, never 2ns");
    }
  }
}

void zero_overheads_preserve_wire_and_hash() {
  const auto explicit_zero=hardware();auto absent=explicit_zero;
  // Keep the same resolved physical identity so the comparison isolates the
  // optional zero-valued timing fields, rather than hashing different inputs.
  absent.input["performance_profile"].erase("launch_ns_per_macro");
  absent.input["performance_profile"].erase("memory_setup_ns_per_macro");
  const auto a=compile(explicit_zero,residual());const auto b=compile(absent,residual());
  require(a->launch_ns==0 && !a->summary.contains("fixed_region_overheads") &&
      !a->summary.contains("prediction_status"),"zero overhead must add no audit/status wire fields");
  require(a->fingerprint==b->fingerprint && plan_to_json(*a)==plan_to_json(*b),
      "explicit zero and the prior absent-field default preserve identical Plan wire/hash");
  require(plan_to_json(*plan_from_json(plan_to_json(*a)))==plan_to_json(*a),
      "zero-overhead public Plan roundtrip must remain byte-for-byte canonical");
}

void composite_region_charged_once() {
  const std::vector<json> tasks={
      {{"op_kind","ffn_region"},{"tensor_shapes",{{"m",1},{"hidden",8},{"ffn",16},
          {"mlp_variant","swiglu"}}},{"storage_dtype","bf16"},{"fusion_policy","materialized"}},
      {{"op_kind","attention_region"},{"tensor_shapes",{{"q_heads",1},{"kv_heads",1},{"head_dim",8}}},
          {"spans",json::array({{{"q_tokens",1},{"kv_tokens",2}}})},
          {"storage_dtype","bf16"},{"fusion_policy","materialized"}},
      {{"op_kind","qkv_rope"},{"tensor_shapes",{{"m",1},{"hidden",8},{"q_heads",2},
          {"kv_heads",1},{"head_dim",4},{"position_encoding","rope"}}},{"storage_dtype","bf16"}}};
  const auto zero=hardware();const auto delayed=hardware(700,300);
  for(const auto& task:tasks) {
    const auto a=compile(zero,task);const auto b=compile(delayed,task);
    require(a->epochs.size()>3,"composite fixture must contain multiple internal stages");
    same_body(*a,*b);
    require(b->launch_ns==1000 && finish(delayed,b)==finish(zero,a)+1000,
        "FFN/Attention/QKV+RoPE pay one 1000ns source-region delay, not per internal kernel");
    require(b->summary.at("fixed_region_overheads").at("modeled_instances")==1 &&
        service_description_count(*b)==service_description_count(*a)+1,
        "composition must expose one ready gate for the entire region");
    for(const auto& a0:b->epochs.front().activities)
      require(a0.role==Role::Load,"supported setup relocation requires an actual initial Load-only epoch");
  }
}

void unmodeled_repeats_and_large_values() {
  auto task=residual(3);const auto zero=hardware();const auto body=compile(zero,task);
  const auto hw=hardware(0.25,0.5);const auto repeated=compile(hw,task);
  same_body(*body,*repeated);
  require(repeated->launch_ns==0 && finish(hw,repeated)==finish(zero,body) &&
      repeated->summary.at("estimated_latency_ns")==body->summary.at("estimated_latency_ns"),
      "unmodeled repeat delays must never be concentrated at the front or alter body timing");
  const auto& audit=repeated->summary.at("fixed_region_overheads");
  require(audit.at("instances")==3 && audit.at("modeled_instances")==0 &&
      audit.at("timing_status")=="not_available" && audit.at("required_total_delay_ns_whole")==2 &&
      audit.at("required_total_delay_ns_ceil")==3 &&
      equivalent_fraction_from_json(audit.at("required_total_delay_ns_fraction"))==0.25L,
      "three 0.75ns delays require an exact 2.25ns audit, not three separate ceilings");
  require(repeated->summary.at("prediction_status")=="not_available" &&
      repeated->summary.at("alone_estimate_status")=="not_available" &&
      estimate_alone(hw,*repeated).at("status")=="not_available" &&
      service_description_count(*repeated)==service_description_count(*body),
      "incomplete repeat timing must propagate not_available without inventing a gate");
  auto passes=residual();passes["weight_passes"]=3;
  require(compile(hw,passes)->summary.at("fixed_region_overheads").at("instances")==3,
      "weight_passes contributes actual region instances");
  const auto large=compile(hardware(std::ldexp(1.0,61),0.5),task);
  const u64 whole=3*(u64{1}<<61)+1;
  require(large->summary.at("fixed_region_overheads").at("required_total_delay_ns_whole")==whole &&
      large->summary.at("fixed_region_overheads").at("required_total_delay_ns_ceil")==whole+1 &&
      equivalent_fraction_from_json(large->summary.at("fixed_region_overheads").at(
          "required_total_delay_ns_fraction"))==0.5L,
      "large exact integer delay and fractional setup must not lose low bits in double arithmetic");
  for(const auto c:std::array<std::array<double,3>,2>{{
      {{std::ldexp(1.0,64),0,1}},{{std::ldexp(1.0,62),0,5}}}}) {
    bool overflow=false;
    try {compile(hardware(c[0],c[1]),residual(static_cast<u64>(c[2])));}
    catch(const std::exception& e){overflow=std::string(e.what()).find("overflow")!=std::string::npos;}
    require(overflow,"per-region or repeated delay exceeding uint64 must explicitly reject");
  }
}

void widely_separated_fractional_delays() {
  const auto task=residual();const auto zero=hardware();const auto body=compile(zero,task);
  for(const auto c:std::array<std::array<double,2>,3>{{
      {{0.5,std::ldexp(1.0,-65)}},{{1.0,std::numeric_limits<double>::denorm_min()}},
      {{std::numeric_limits<double>::denorm_min(),0.0}}}}) {
    const auto hw=hardware(c[0],c[1]);const auto p=compile(hw,task);
    const u64 delay=c[0]==1.0 ? 2 : 1;
    same_body(*body,*p);
    require(p->launch_ns==delay && finish(hw,p)==finish(zero,body)+delay &&
        estimate_alone(hw,*p).at("status")=="ok_assumed",
        "legal fractional coefficients must reach integer ready timing without lossy audit rejection");
    const auto& a=p->summary.at("fixed_region_overheads");
    require(a.at("required_total_delay_ns_whole")==static_cast<u64>(c[0]) &&
        a.at("required_total_delay_ns_ceil")==delay && a.at("launch_ns_per_instance")==c[0] &&
        a.at("memory_setup_ns_per_hbm_instance")==c[1],
        "retain the exact binary64 input expression alongside exact floor/ceil");
  }
}

void waiting_resource_shares_snapshot_cancel_and_capacity() {
  constexpr u64 delay=1000000;
  const auto hw=hardware(delay,0);const auto p=compile(hw,residual());
  Runtime waiting(hw);const auto h=waiting.submit(p,owner(1,1,1),0);
  require(waiting.next_event()==delay,"ready delay is the first event, with no synthetic service activity");
  waiting.advance_to(300000);no_service(waiting.query_progress(h));no_busy(waiting);
  auto imported=Runtime::import_snapshot(hw,waiting.export_snapshot());
  require(imported->export_snapshot()==waiting.export_snapshot(),"queued ready wait snapshot roundtrips exactly");
  waiting.update_capacity(HBM0,hw.rates[HBM0]/2,300000);
  imported->update_capacity(HBM0,hw.rates[HBM0]/2,300000);
  waiting.advance_to(delay-1);imported->advance_to(delay-1);
  no_service(waiting.query_progress(h));no_busy(waiting);
  while(!waiting.drained()) {
    require(waiting.next_event()==imported->next_event(),"waiting source/import next events agree after capacity update");
    const auto next=waiting.next_event();require(next.has_value(),"resumed region remains reachable");
    waiting.advance_to(*next);imported->advance_to(*next);
  }
  require(waiting.query_progress(h)==imported->query_progress(h) &&
      waiting.counters()==imported->counters(),"restored wait resumes with identical completion, served work and busy");

  Runtime cancelled(hw);const auto cancelled_h=cancelled.submit(p,owner(2,2,2),0);
  cancelled.advance_to(delay-1);cancelled.cancel(cancelled_h,delay-1);
  const auto cancelled_progress=cancelled.query_progress(cancelled_h);
  require(cancelled_progress.at("cancelled")==true && cancelled_progress.at("finish_ns")==delay-1 &&
      cancelled.drained() && !cancelled.next_event(),"cancel during initial wait finishes immediately without output");
  no_service(cancelled_progress);no_busy(cancelled);
  require(Runtime::import_snapshot(hw,cancelled.export_snapshot())->export_snapshot()==cancelled.export_snapshot(),
      "cancelled initial-wait snapshot retains its zero-service terminal state");

  Runtime competition(hw);const auto queued=competition.submit(p,owner(3,3,3),0);
  Activity c=activity(1,Role::Compute,COMPUTE_GATE,1e-7*hw.rates[COMPUTE_GATE]);
  c.private_time_s=1e-7;c.canonical_q=1;
  Epoch e;e.name="independent_100ns_compute";e.activities={c};
  const auto competitor=competition.submit(manual_plan(hw,{e}),owner(4,4,4),0);
  competition.advance_to(100);
  require(competition.terminal_finish_ns(competitor)==100,
      "a queued region cannot take a fairness share from a hand-calculated 100ns competitor");
  no_service(competition.query_progress(queued));
  require_close(competition.counters().at("compute_busy_ns").get<double>(),100,1e-9,
      "only actual competitor service contributes compute busy during the wait");
  require(competition.counters().at("hbm_busy_ns")==0,"queued HBM work must not accrue busy");
  competition.cancel(queued,100);
}

void ready_time_overflow_is_atomic() {
  const auto hw=hardware(1000,0);const auto p=compile(hw,residual());
  Runtime runtime(hw);runtime.advance_to(7);const auto before=runtime.export_snapshot();
  bool overflow=false;
  try {runtime.submit(p,owner(1,1,1),std::numeric_limits<u64>::max()-999);}
  catch(const std::overflow_error&){overflow=true;}
  require(overflow && runtime.export_snapshot()==before && runtime.now_ns()==7 &&
      runtime.execution_handles().empty(),"overflowing issue+ready time must not advance or mutate the world");
  const auto valid=runtime.submit(p,owner(1,1,1),7);
  require(valid==1,"rejected ready-time overflow must not consume a local Runtime handle");
  runtime.cancel(valid,7);
}

json take(const char* raw) {
  require(raw!=nullptr,std::string("C ABI JSON call failed: ")+die_coarse_last_error_v2());
  const std::string text(raw);die_coarse_free_result_v2(raw);return json::parse(text);
}
struct CModel {
  DieHandle handle=0;std::vector<DieHandle> plans;
  explicit CModel(const Hardware& hw) {
    const auto cfg=hw.input.dump();require(die_coarse_create_model_v2(cfg.c_str(),&handle)==DIE_OK,
        std::string("C ABI model creation failed: ")+die_coarse_last_error_v2());
  }
  ~CModel(){for(auto p:plans)(void)die_coarse_release_plan_v2(p);if(handle)(void)die_coarse_destroy_model_v2(handle);}
};

void cabi_status_and_schema_consistency() {
  CModel model(hardware(0.25,0.5));
  const auto request=json{{"specs",json::array({residual(),residual(3)})}}.dump();
  const auto compiled=take(die_coarse_compile_batch_v2(model.handle,request.c_str()));
  for(const auto& row:compiled.at("results"))if(row.is_object() && row.contains("plan_handle"))
    model.plans.push_back(row.at("plan_handle").get<DieHandle>());
  const auto full=take(die_coarse_estimate_batch_v2(model.handle,request.c_str()));
  const auto cost=take(die_coarse_estimate_cost_batch_v2(model.handle,request.c_str()));
  for(std::size_t i=0;i<2;++i) {
    require(compiled.at("errors")[i].is_null() && full.at("errors")[i].is_null() && cost.at("errors")[i].is_null(),
        "modeled and explicitly unavailable tasks retain reviewable successful ABI rows");
    const std::string status=i ? "not_available" : "ok_assumed";
    const auto& c=compiled.at("results")[i];const auto& f=full.at("results")[i];const auto& k=cost.at("results")[i];
    require(c.at("summary").at("alone_estimate_status")==status && f.at("status")==status && k.at("status")==status,
        "compile/full/cost must consistently expose incomplete repeat timing as not_available");
    require(c.at("plan_hash")==k.at("plan_hash") && c.at("plan_hash")==f.at("plan_fingerprint") &&
        c.at("summary").at("estimated_latency_ns")==k.at("estimated_latency_ns") &&
        k.at("estimated_latency_ns")==f.at("estimated_latency_ns"),
        "all ABI query modes must identify and price the same compiled body/ready Plan");
    require(c.at("plan_schema")=="coarse_die_plan_v4" && f.at("plan_schema")=="coarse_die_plan_v4" &&
        f.at("granularity_identity").at("schema")=="coarse_die_plan_v4",
        "compile and full diagnostics report actual curved v4 instead of legacy v2");
    if(i)require(c.at("summary").at("prediction_status")=="not_available" &&
        c.at("summary").at("fixed_region_overheads").at("modeled_instances")==0,
        "repeat ABI row cannot claim all instances modeled");
  }
  DieHandle runtime=0;require(die_coarse_create_runtime_v2(model.handle,&runtime)==DIE_OK,"create ABI overflow world");
  const auto before=take(die_coarse_export_snapshot_v2(runtime));
  DieOwnerV2 o{};o.struct_size=sizeof(o);o.schema_version=2;o.node_id=1;o.execution_id=1;o.generation=1;
  DieHandle execution=12345;
  const int status=die_coarse_submit_plan_v2(runtime,model.plans.at(0),&o,std::numeric_limits<u64>::max(),&execution);
  require(status==DIE_OVERFLOW && execution==0 && take(die_coarse_export_snapshot_v2(runtime))==before,
      "C ABI ready-time overflow returns no handle and atomically preserves the runtime world");
  require(die_coarse_destroy_runtime_v2(runtime)==DIE_OK,"empty ABI overflow world remains destroyable");
}

void test_region_overheads() {
  single_region_response_and_quantization();
  zero_overheads_preserve_wire_and_hash();
  composite_region_charged_once();
  unmodeled_repeats_and_large_values();
  widely_separated_fractional_delays();
  waiting_resource_shares_snapshot_cancel_and_capacity();
  ready_time_overflow_is_atomic();
  cabi_status_and_schema_consistency();
}
}  // namespace

int main(){return run_test("region_overhead_test",test_region_overheads);}
