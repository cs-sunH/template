#include "coarse_die/Execution.hh"

#include <array>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef COARSE_DIE_SOURCE_DIR
#error "COARSE_DIE_SOURCE_DIR must point at die/"
#endif

namespace {
using json = nlohmann::json;
using namespace coarse_die;
constexpr std::array<const char*, resource_count> names{{
    "COMPUTE_GATE", "SRAM_READ", "SRAM_WRITE", "LOCAL", "HBM0", "HBM1",
    "INJECT0", "INJECT1", "XPOS", "XNEG", "YPOS", "YNEG", "HOP"}};
struct Args {
  std::uint64_t elements=257, executions=1, offset_ns=5000000;
  std::string op="residual", input_dtype="bf16", output_dtype="fp32", profile="balanced";
};
std::uint64_t number(const std::string& value, bool zero=false) {
  std::uint64_t n=0;
  const auto parsed=std::from_chars(value.data(),value.data()+value.size(),n);
  if(parsed.ec!=std::errc{} || parsed.ptr!=value.data()+value.size() || (!zero && !n))
    throw std::invalid_argument("expected a positive integer: "+value);
  return n;
}
Args arguments(int argc,char** argv) {
  Args a;
  for(int i=1;i<argc;++i) {
    const std::string key=argv[i];
    if(key=="--help") {
      std::cout<<"elementwise_oracle_driver [--elements N] [--executions I] "
          "[--op residual|gelu|swiglu] [--input-dtype int4|bf16|fp32] "
          "[--output-dtype int4|bf16|fp32] [--profile balanced|memory|compute|local] "
          "[--offset-ns N]\nSynthetic single-Core Mapping/Runtime evidence; no hardware calibration.\n";
      std::exit(0);
    }
    if(i+1==argc)throw std::invalid_argument("missing argument for "+key);
    const std::string value=argv[++i];
    if(key=="--elements")a.elements=number(value);
    else if(key=="--executions")a.executions=number(value);
    else if(key=="--offset-ns")a.offset_ns=number(value,true);
    else if(key=="--op")a.op=value;
    else if(key=="--input-dtype")a.input_dtype=value;
    else if(key=="--output-dtype")a.output_dtype=value;
    else if(key=="--profile")a.profile=value;
    else throw std::invalid_argument("unknown option: "+key);
  }
  if(a.op!="residual" && a.op!="gelu" && a.op!="swiglu")throw std::invalid_argument("unsupported op");
  if(a.profile!="balanced" && a.profile!="memory" && a.profile!="compute" && a.profile!="local")
    throw std::invalid_argument("unsupported synthetic profile");
  if(a.elements>1048576 || a.executions>256 || (a.elements+255)/256*a.executions>4096)
    throw std::invalid_argument("offline fixture limited to 4096 enumerated blocks");
  return a;
}
std::uint64_t storage(std::uint64_t n,const std::string& dtype) {
  if(dtype=="int4")return (n+1)/2;
  if(dtype=="bf16" || dtype=="fp16")return 2*n;
  if(dtype=="fp32")return 4*n;
  throw std::invalid_argument("unsupported fixture storage dtype");
}
std::uint64_t streams(const Args& a) {return a.op=="gelu" ? 1 : 2;}
std::uint64_t operations(const Args& a) {return a.op=="residual" ? 1 : a.op=="gelu" ? 15 : 16;}
Hardware hardware(const Args& a) {
  std::ifstream file(std::string(COARSE_DIE_SOURCE_DIR)+"/configs/example_hardware_explicit.json");
  if(!file)throw std::runtime_error("cannot open explicit hardware fixture");
  json cfg=json::parse(file);
  cfg["architecture"]["core_type"]="small";
  cfg["architecture"]["core_columns_nx"]=1;
  cfg["architecture"]["core_rows_ny"]=1;
  cfg["performance_profile"]["eta_vec"]=1.0;
  cfg["performance_profile"]["eta_hbm_vector"]=1.0;
  cfg["performance_profile"]["launch_ns_per_macro"]=0;
  cfg["performance_profile"]["memory_setup_ns_per_macro"]=0;
  Hardware hw=resolve_hardware(cfg);
  // Deliberate synthetic rates, never a fitted physical hardware profile.
  double vec=operations(a)*25600.0;
  if(a.profile=="memory" || a.profile=="local")vec*=100.0;
  if(a.profile=="compute")vec*=0.1;
  hw.derived["core"]["P_vec_per_core_ops_per_s"]=vec;
  for(std::size_t r=0;r<resource_count;++r) {
    auto& desc=hw.derived["resources"][r];
    double rate=desc.at("enabled").get<bool>() ? 1.0e12 : 0.0;
    if(r==COMPUTE_GATE)rate=1.0;
    if((r==HBM0 || r==HBM1) && a.profile!="local")rate=51200.0;
    if((r==HBM0 || r==HBM1) && a.profile=="local")rate=1.0e9;
    if(r==LOCAL && a.profile=="local")
      rate=(streams(a)*storage(256,a.input_dtype)+storage(256,a.output_dtype))/0.01;
    hw.rates[r]=rate;desc["rate"]=rate;
  }
  const std::string identity=hw.fingerprint+":"+a.profile+":"+a.op+":"+
      a.input_dtype+":"+a.output_dtype+":"+std::to_string(vec);
  hw.fingerprint="synthetic-elementwise:"+stable_hash(identity);
  return hw;
}
json task(const Args& a) {
  json shape={{"m",1}};
  if(a.op=="residual")shape["hidden"]=a.elements;
  else {shape["ffn"]=a.elements;shape["mlp_variant"]=a.op;}
  return {{"op_kind",a.op=="residual" ? "residual" : "mlp_activation"},
      {"tensor_shapes",shape},{"storage_dtype",{{"activations",a.input_dtype},{"output",a.output_dtype}}},
      {"execution_profile","fp32_vec_matrix_assumed_v1"},{"layer_instances",a.executions},{"weight_passes",1}};
}
Owner owner(std::uint64_t id) {Owner o;o.request_id=id;o.node_id=id;o.execution_id=id;return o;}
const char* role(Role r) {return r==Role::Load ? "load" : r==Role::Compute ? "compute" : "store";}
json plan_data(const Plan& p) {
  json epochs=json::array();
  for(const auto& e:p.epochs) {
    json activities=json::array();
    for(const auto& a:e.activities)
      activities.push_back({{"id",a.id},{"role",role(a.role)},{"work",a.work},
          {"hbm_bytes",a.hbm_bytes},{"vec_eqops",a.vec_eqops},{"canonical_q",a.canonical_q},
          {"private_time_s",a.private_time_s}});
    epochs.push_back({{"name",e.name},{"credit_initial",e.credit_initial},
        {"credit_capacity",e.credit_capacity},{"activities",activities}});
  }
  return {{"task",p.task},{"summary",p.summary},{"epochs",epochs},
      {"wire",plan_to_json(p)},{"description_count",service_description_count(p)}};
}
json run(const Hardware& hw,std::shared_ptr<const Plan> p,const std::vector<std::uint64_t>& arrivals) {
  Runtime rt(hw);
  std::vector<std::uint64_t> handles;
  json events=json::array();
  std::size_t next_arrival=0,advances=0;
  auto capture=[&]() {
    json states=json::array();
    for(auto h:handles) {
      const auto q=rt.query_progress(h);
      states.push_back({{"handle",h},{"epoch_index",q.at("epoch_index")},{"credit",q.at("credit")},
          {"served",q.at("served")},{"terminal",q.at("terminal")}});
    }
    events.push_back({{"tick_ns",rt.now_ns()},{"states",states}});
  };
  for(;;) {
    while(next_arrival<arrivals.size() && arrivals[next_arrival]==rt.now_ns()) {
      handles.push_back(rt.submit(p,owner(next_arrival+1),rt.now_ns()));++next_arrival;
    }
    capture();
    if(next_arrival==arrivals.size() && rt.drained())break;
    auto next=rt.next_event();
    if(next_arrival<arrivals.size() && (!next || arrivals[next_arrival]<*next))next=arrivals[next_arrival];
    if(!next || *next<=rt.now_ns())throw std::runtime_error("nonadvancing Runtime fixture");
    rt.advance_to(*next);
    if(++advances>10000)throw std::runtime_error("Runtime fixture exceeded event bound");
  }
  json finish=json::array(),first=json::array(),terminal=json::array();
  for(auto h:handles) {
    finish.push_back(*rt.terminal_finish_ns(h));terminal.push_back(rt.query_progress(h));
    json first_tick=nullptr;
    for(const auto& event:events)for(const auto& state:event.at("states"))
      if(state.at("handle")==h && state.at("epoch_index").get<std::size_t>()>=2 && first_tick.is_null())
        first_tick=event.at("tick_ns");
    first.push_back(first_tick);
  }
  return {{"arrivals_ns",arrivals},{"finish_ns",finish},{"first_output_ready_ns",first},
      {"event_advances",advances},{"counters",rt.counters()},{"event_trace",events},{"terminal_states",terminal}};
}
}  // namespace

int main(int argc,char** argv) {
  try {
    const Args a=arguments(argc,argv);const Hardware hw=hardware(a);const auto p=compile(hw,task(a));
    json resources=json::array();
    for(std::size_t r=0;r<resource_count;++r)
      resources.push_back({{"name",names[r]},{"capacity",hw.rates[r]}});
    json output={{"schema_version",1},{"measurement_source","actual_compiled_mapping_and_cpp_runtime"},
      {"scope","single_core_synthetic_model_to_model_not_hardware_calibration"},
      {"fixture",{{"op",a.op},{"elements_per_invocation",a.elements},{"executions",a.executions},
          {"input_dtype",a.input_dtype},{"output_dtype",a.output_dtype},{"profile",a.profile}}},
      {"hardware",{{"input",hw.input},{"derived",hw.derived},{"resources",resources},{"fingerprint",hw.fingerprint}}},
      {"compiled_plan",plan_data(*p)},
      {"runs",{{"solo",run(hw,p,{0})},{"sync",run(hw,p,{0,0})},{"offset",run(hw,p,{0,a.offset_ns})}}}};
    std::cout<<output.dump(2)<<'\n';return 0;
  }catch(const std::exception& e){std::cerr<<"elementwise_oracle_driver: "<<e.what()<<'\n';return 1;}
}
