#include "test_support.hh"
#include "coarse_die/Prediction.hh"
#include <algorithm>
#include <array>
#include <limits>
#include <set>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;
using u64=std::uint64_t;

namespace {
Hardware hardware(u64 cores=4) {
  auto c=read_json(std::string(COARSE_DIE_SOURCE_DIR)+"/configs/example_hardware_explicit.json");
  c["architecture"]["core_columns_nx"]=cores;c["architecture"]["core_rows_ny"]=1;
  return resolve_hardware(c);
}
json task(u64 rows,u64 hidden,bool layer=false,const std::string& x="bf16",
    const std::string& p="bf16",const std::string& y="bf16",u64 repeats=1) {
  return {{"op_kind","norm"},{"tensor_shapes",{{"m",rows},{"hidden",hidden},
      {"norm_type",layer ? "layernorm" : "rmsnorm"}}},
      {"storage_dtype",{{"activations",x},{"parameters",p},{"output",y}}},
      {"layer_instances",repeats},{"weight_passes",1}};
}
u64 bytes(u64 h,const std::string& dt) {
  return dt=="int4" ? (h+1)/2 : h*(dt=="bf16" ? 2 : 4);
}
void independent_rows() {
  for(bool layer:{false,true})for(u64 cores:{1,4})for(u64 rows:{1,9})
      for(u64 h:{3,257})for(const std::string x:{"int4","bf16","fp32"})
      for(const std::string p:{"int4","bf16","fp32"})
      for(const std::string y:{"int4","bf16","fp32"}) {
    const auto hw=hardware(cores);const auto plan=compile(hw,task(rows,h,layer,x,p,y));
    const auto& s=plan->summary;const u64 vectors=layer ? 2 : 1;
    std::set<u64> consumers;
    u64 read=0,write=0,math=0,converted=0,eq=0;
    const auto cost=[](const std::string& dtype){return dtype=="int4" ? 4 : dtype=="fp32" ? 0 : 1;};
    // Explicit independent row ownership, separately packed gamma/beta, and
    // two SRAM visits. No production curve or geometry helper is consulted.
    for(u64 row=0;row<rows;++row) {
      consumers.insert(row%cores);read+=bytes(h,x);write+=bytes(h,y);
      math+=(layer ? 8 : 6)*h;eq+=(layer ? 8 : 6)*h+2*h*cost(x)+h*cost(y);
      converted+=(x!="fp32" ? 2*h : 0)+(y!="fp32" ? h : 0);
    }
    for(u64 core:consumers)for(u64 parameter=0;parameter<vectors;++parameter) {
      (void)core;(void)parameter;read+=bytes(h,p);eq+=h*cost(p);converted+=p!="fp32" ? h : 0;
    }
    require(s.at("hbm_read_bytes_total")==read && s.at("hbm_write_bytes_total")==write,
        "Norm must load separately packed parameters on every consumer Core");
    require(s.at("source_vec_ops")==math && s.at("vec_equivalent_ops")==eq &&
        s.at("conversion_elements")==converted,"Norm math and actual conversion visits stay separate");
    require(s.at("norm_row_ledger").at("parameter_hbm_loads")==consumers.size(),"no inter-Core free parameter broadcast");
    u64 actual_read=0,actual_write=0,actual_eq=0;
    long double sr=0,sw=0,local=0;
    for(const auto& e:plan->epochs)for(const auto& a:e.activities) {
      require(e.activities.size()<=3,"one Norm owner has at most three activities");
      for(const auto& row:a.hbm_bytes){actual_read+=row[0];actual_write+=row[1];}
      actual_eq+=a.vec_eqops;sr+=a.work[SRAM_READ];sw+=a.work[SRAM_WRITE];local+=a.work[LOCAL];
    }
    require(actual_read==read && actual_write==write && actual_eq==eq,"actual Plan conserves Norm payload and Vec work");
    require(sr==s.at("sram_read_bytes").get<u64>() && sw==s.at("sram_write_bytes").get<u64>() &&
        local==s.at("local_bus_bytes").get<u64>(),"Norm compute and IO SRAM/LOCAL service matches summary");
  }
}
void exact_cache_boundary() {
  const auto hw=hardware();
  for(bool layer:{false,true}) {
    const u64 last_fitting=layer ? 14676 : 19568;
    for(u64 h:{last_fitting,last_fitting+1}) {
      const auto plan=compile(hw,task(8,h,layer));const auto& s=plan->summary;
      const bool cached=h==last_fitting;
      const u64 vectors=layer ? 2 : 1;
      require(s.at("norm_row_ledger").at("parameter_hbm_loads")==(cached ? 4 : 8),
          "reuse requires the complete parameter buffer plus the entire live row allocation to fit");
      require(s.at("hbm_read_bytes_total")==8*h*2+(cached ? 4 : 8)*vectors*h*2,
          "uncached gamma/beta are loaded once per row");
      u64 read=0;for(const auto& e:plan->epochs)for(const auto& a:e.activities)
        for(const auto& row:a.hbm_bytes)read+=row[0];
      require(read==s.at("hbm_read_bytes_total").get<u64>(),"streamed Compute retains its actual HBM payload");
      const auto result=estimate_alone(hw,*plan);
      const auto& served=result.at("execution").at("served").at("hbm_bytes_by_row_direction");
      require_close(served[0][0].get<double>()+served[1][0].get<double>(),read,1e-6,
          "completed streamed/cached Norm physical reads reconcile");
    }
  }
  bool rejected=false;
  try {(void)compile(hw,task(8,131073));}
  catch(const std::invalid_argument& e) {rejected=std::string(e.what()).find("unsupported_workspace")!=std::string::npos;}
  require(rejected,"long rows must not silently receive one-pass timing without a fitting row buffer");
}
void complete_row_readiness_and_restore() {
  const auto hw=hardware(2);const auto plan=compile(hw,task(7,4096));
  require(plan->summary.at("service_description_count").get<u64>()<=16,"Norm prefix/middle/drain plus ready gate stays in sixteen descriptions");
  Runtime source(hw);const auto handle=source.submit(plan,{1,2,3,1},0);
  source.advance_to(*source.next_event());  // startup ready gate
  const u64 load_begin=source.now_ns(),load_end=*source.next_event();
  require(load_end>load_begin+1,"fixture exposes a positive integer interval inside the row load");
  source.advance_to(load_begin+(load_end-load_begin)/2);
  const auto counters=source.counters();
  require(counters.at("compute_busy_ns").get<double>()==0,"row statistics cannot start during first incomplete input load");
  const auto& payload=counters.at("hbm_served_bytes_by_row_direction");
  require(payload[0][1].get<double>()+payload[1][1].get<double>()==0,"Norm cannot write an output before complete row statistics");
  source.update_capacity(COMPUTE_GATE,0,source.now_ns());
  for(unsigned event=0;event<4 && source.next_event();++event)
    source.advance_to(*source.next_event()); // only bounded input prefetch is allowed
  require(!source.next_event().has_value(),"first row is held when its statistics/normalization Compute is paused");
  const auto snapshot=source.export_snapshot();
  auto restored=Runtime::import_snapshot(hw,snapshot);
  auto cancelled=Runtime::import_snapshot(hw,snapshot);
  cancelled->update_capacity(COMPUTE_GATE,hw.rates[COMPUTE_GATE],cancelled->now_ns());
  const auto cancel_begin=cancelled->now_ns(),cancel_end=*cancelled->next_event();
  require(cancel_end>cancel_begin+1,"fixture exposes a positive interval in row Compute");
  cancelled->advance_to(cancel_begin+(cancel_end-cancel_begin)/2);
  cancelled->cancel(handle,cancelled->now_ns());
  require(cancelled->drained() && cancelled->query_progress(handle).at("cancelled")==true &&
      cancelled->counters().at("active_owners")==0,"partial row cancellation removes service without publishing output");
  source.update_capacity(COMPUTE_GATE,hw.rates[COMPUTE_GATE],source.now_ns());
  restored->update_capacity(COMPUTE_GATE,hw.rates[COMPUTE_GATE],restored->now_ns());
  unsigned events=0;
  while(!source.drained()) {require(++events<=20,"row count must not create row-linear events");source.advance_to(*source.next_event());}
  while(!restored->drained())restored->advance_to(*restored->next_event());
  require(source.query_progress(handle)==restored->query_progress(handle) && source.counters()==restored->counters(),"row-resident parameter/credit progress restores exactly");
  require(source.terminal_finish_ns(handle).has_value(),"whole Norm terminal is published after final row store");
}
void streamed_parameters_and_odd_routes() {
  for(bool layer:{false,true})for(u64 cores:{1,3,4})for(u64 rows:{2*cores,3*cores-1,3*cores+1})
      for(const std::string pd:{"int4","bf16","fp32"}) {
    const u64 hidden=layer ? 15001 : 20001,parameters=layer ? 2 : 1;
    const auto hw=hardware(cores);const auto plan=compile(hw,task(rows,hidden,layer,"bf16",pd,"bf16"));
    const auto& summary=plan->summary;
    require(summary.at("norm_row_ledger").at("parameter_cache")=="streamed_each_row",
        "whole FP32 parameters cannot be cached at the streamed fixture boundary");
    const u64 read=rows*(2*hidden+parameters*bytes(hidden,pd)),write=rows*2*hidden;
    u64 actual=0;for(const auto& e:plan->epochs)for(const auto& activity:e.activities)
      for(const auto& row:activity.hbm_bytes)actual+=row[0];
    require(actual==read && summary.at("hbm_read_bytes_total")==read && summary.at("hbm_write_bytes_total")==write,
        "odd gamma/beta payloads conserve across first, middle and tail Core prefixes");
    const auto result=estimate_alone(hw,*plan);const auto& served=result.at("execution").at("served").at("hbm_bytes_by_row_direction");
    require_close(served[0][0].get<double>()+served[1][0].get<double>(),read,1e-5,"streamed odd parameters reach Runtime terminal with conserved physical reads");
  }
}
void fractional_conversion() {
  auto c=hardware(2).input;c["performance_profile"]["kappa_quant_eqops"]=0.5;
  const auto hw=resolve_hardware(c);const auto plan=compile(hw,task(5,3,false,"int4","int4","int4"));
  require(plan->summary.at("source_vec_ops")==90 && plan->summary.at("vec_equivalent_ops")==115.5,
      "fractional dequant/input/output conversion costs retain exact Norm math");
  long double total=0;for(const auto& e:plan->epochs)for(const auto& a:e.activities)total+=planned_vec_eqops(a);
  require(total==115.5L,"Norm fractional work is conserved across prefix and drain");
  const auto estimate=estimate_alone(hw,*plan);
  require(estimate.at("execution").at("served").at("vec_eqops")==115.5,"Runtime serves exact fractional Norm work");
}
void per_core_row_slots() {
  for(u64 cores:{1,2,3,8})for(u64 rows:{2*cores+1,3*cores,5*cores+1}) {
    const auto plan=compile(hardware(cores),task(rows,257));
    const u64 rounds=(rows+cores-1)/cores;
    const auto& epoch=plan->epochs.at(2);
    std::vector<u64> loaded(cores),consumed(cores),produced(cores),stored(cores);
    // Enumerate actual RR row ownership, not max-Core aggregate formulas.
    for(u64 row=0;row<rows;++row) {
      const auto core=row%cores;
      if(row>=2*cores)++loaded[core];
      if(row>=cores && row<(rounds-1)*cores){++consumed[core];++produced[core];}
      if(row<(rounds-2)*cores)++stored[core];
    }
    for(unsigned l=0;l<=4;++l)for(unsigned c=0;c<=4;++c)for(unsigned s=0;s<=4;++s) {
      std::array<double,2> credits=epoch.credit_initial;
      for(const auto& activity:epoch.activities) {
        const auto values=credit_at(activity,(activity.role==Role::Load ? l : activity.role==Role::Compute ? c : s)/4.0);
        for(unsigned i=0;i<2;++i)credits[i]+=values[i];
      }
      if(credits[0]<0 || credits[0]>epoch.credit_capacity[0] || credits[1]<0 || credits[1]>epoch.credit_capacity[1])continue;
      for(u64 core=0;core<cores;++core) {
        const double in=1+loaded[core]*l/4.0-consumed[core]*c/4.0;
        const double out=1+produced[core]*c/4.0-stored[core]*s/4.0;
        require(in>=0 && in<=2 && out>=0 && out<=2,
            "feasible cumulative Norm credit cannot borrow a different Core's physical row slots");
      }
    }
  }
}
void repeats_and_scaling() {
  const auto hw=hardware(2);
  for(u64 repeat:{1,3,1000000}) {
    const auto plan=compile(hw,task(5,3,false,"int4","int4","int4",repeat));
    if(repeat>1) {
      bool rejected=false;try {require_complete_prediction(*plan);}catch(const std::invalid_argument&){rejected=true;}
      require(rejected,"aggregate repeated Norm body cannot claim complete per-region timing");
      require(plan->summary.at("norm_row_ledger").at("parameter_hbm_loads")==5*repeat,
          "diagnostic repeats reload per-row parameters and never assume cross-region residency");
    }
    Runtime runtime(hw);runtime.submit(plan,{1,1,repeat,1},0);unsigned events=0;
    while(!runtime.drained()) {require(++events<=20,"Norm repeat amplification has a fixed internal event budget");runtime.advance_to(*runtime.next_event());}
  }
}
void run() {independent_rows();exact_cache_boundary();complete_row_readiness_and_restore();streamed_parameters_and_odd_routes();fractional_conversion();per_core_row_slots();repeats_and_scaling();}
}
int main(){return run_test("norm_row_accounting_test",run);}
