#include "test_support.hh"
#include "../src/ElementwiseGeometry.hh"
#include <algorithm>
#include <limits>
#include <iomanip>
#include <sstream>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;
using u64=std::uint64_t;

namespace {
long double max_service_residual=0;
u64 max_service_events=0;
Hardware hardware(u64 cores=1) {
  auto cfg=read_json(std::string(COARSE_DIE_SOURCE_DIR)+"/configs/example_hardware_explicit.json");
  cfg["architecture"]["core_columns_nx"]=cores;
  cfg["architecture"]["core_rows_ny"]=1;
  return resolve_hardware(cfg);
}
json task(u64 elements,const std::string& op="residual",const std::string& dt="int4",u64 repeats=1) {
  return {{"op_kind",op},{"tensor_shapes",{{"m",1},{"hidden",elements},{"ffn",elements}}},
      {"storage_dtype",dt},{"layer_instances",repeats},{"weight_passes",1}};
}
u64 payload(const Activity& a,bool write=false) {
  u64 n=0;for(const auto& row:a.hbm_bytes)n+=row[write ? 1 : 0];return n;
}
void bounded_drain(Runtime& r) {
  u64 steps=0;
  while(!r.drained()) {
    const auto next=r.next_event();require(next.has_value(),"no next event");
    require(++steps<100,"elementwise repeats must not create linear events");
    r.advance_to(*next);
  }
}
void independent_enumeration() {
  for(u64 count=1;count<=1031;++count)for(u64 cores=1;cores<=8;++cores) {
    const detail::elementwise_geometry::Geometry g(count,3,cores);
    std::vector<u64> blocks;
    for(u64 begin=0;begin<count;begin+=256)blocks.push_back(std::min<u64>(256,count-begin));
    const u64 active=std::min<u64>(cores,blocks.size());
    u64 serial=0,rounds=0,first=0,last=0;
    for(u64 begin=0;begin<blocks.size();begin+=active) {
      u64 maximum=0,total=0;
      for(u64 i=begin;i<std::min<u64>(blocks.size(),begin+active);++i) {
        maximum=std::max(maximum,blocks[i]);total+=blocks[i];
      }
      if(!rounds)first=total;
      last=total;serial+=maximum;++rounds;
    }
    require(g.total_blocks==3*blocks.size()&&g.total_rounds==3*rounds&&
        g.total_elements==3*count&&g.serial_elements==3*serial&&
        g.first_elements==first&&g.last_elements==last,
        "closed form differs from independent explicit block/cohort enumeration");
  }
  bool rejected=false;
  try { detail::elementwise_geometry::Geometry g(std::numeric_limits<u64>::max(),2,1); }
  catch(const std::overflow_error&) {rejected=true;}
  require(rejected,"repeated work overflow must fail");
}
void physical_ledger_and_prefix() {
  for(const auto op:{"residual","mlp_activation"})
    for(const auto dt:{"int4","bf16","fp32"})
      for(u64 count:{1,255,256,257,513,1025})
        for(u64 repeats:{1,3}) {
          const auto hw=hardware(2);const auto p=compile(hw,task(count,op,dt,repeats));
          const u64 streams=2,ops=std::string(op)=="residual" ? 1 : 16;
          const auto b=[&](u64 n){return std::string(dt)=="int4" ? n/2+n%2 : n*(std::string(dt)=="bf16" ? 2 : 4);};
          const u64 read=streams*b(count)*repeats,write=b(count)*repeats;
          const auto& s=p->summary;
          const auto& audit=s.at("service_curves")[0];
          require(s.at("hbm_read_bytes_total")==read&&s.at("hbm_write_bytes_total")==write&&
              s.at("source_vec_ops")==count*ops*repeats,
              "actual tensors are packed independently before repetition");
          u64 actual_read=0,actual_write=0,vec=0;
          // Last input has its own reserved physical slot in the bounded
          // prefix/drain template. Its completed Load terminal transfers one
          // unit into Last Compute; ordinary middle credit omits that slot.
          long double in=audit.value("reserved_input_terminal_transfer_units",0),out=0;
          if(in) {
            const auto transfer=audit.at("reserved_input_terminal_transfer_epoch").get<u64>();
            require(in==1 && p->epochs.at(transfer).credit_initial[0]==1,
                "reserved Last input must be explicitly transferred at its Load terminal");
          }
          for(const auto& e:p->epochs) {
            out+=e.credit_initial[1];
            for(const auto& a:e.activities) {
              actual_read+=payload(a);actual_write+=payload(a,true);vec+=a.vec_eqops;
              in+=a.credit_delta[0];out+=a.credit_delta[1];
            }
          }
          require(actual_read==read&&actual_write==write&&vec==count*ops*repeats&&in==0&&out==0,
              "physical work and independent input/output unit ledgers must close");
          require(service_description_count(*p)<=16&&audit.at("description_count")==service_description_count(*p),
              "all activity terminal switches count in the fixed budget");
          const u64 first_elements=std::min<u64>(count,2*256);
          require(payload(p->epochs[0].activities[0])==streams*b(first_elements),
              "first load is a real cohort, not average total bytes divided by rounds");
          for(const auto& a:p->epochs[1].activities) {
            require(a.credit_delta[1]==0 && credit_at(a,0.5)[1]==0,
                "the first epoch must not release fractional output blocks");
            if(a.role==Role::Compute)require(a.id==audit.at("first_output_activity_id"),
                "first-output diagnostics must identify the actual Compute terminal");
          }
          Runtime r(hw);const auto h=r.submit(p,owner(1,1,1),0);bounded_drain(r);
          const auto served=r.query_progress(h).at("served");
          const auto bytes=served.at("hbm_bytes_by_row_direction_exact");
          // The *_exact strings preserve long-double integration, not rounded
          // integer counters. Check integer Plan conservation above separately
          // from this small floating arithmetic reconciliation, and report it.
          for(std::size_t dir=0;dir<2;++dir) {
            const u64 expected=dir ? write : read;
            const long double actual=std::stold(bytes[0][dir].get<std::string>())+
                std::stold(bytes[1][dir].get<std::string>());
            const long double residual=std::abs(actual-expected);
            max_service_residual=std::max(max_service_residual,residual);
            require(residual<=128*std::numeric_limits<long double>::epsilon()*std::max<u64>(1,expected),
                "Runtime service must reconcile within fixed arithmetic roundoff; not an accuracy tolerance");
          }
        }
  auto gelu=task(257,"mlp_activation");gelu["tensor_shapes"]["mlp_variant"]="gelu";
  const auto p=compile(hardware(),gelu);
  require(p->summary.at("hbm_read_bytes_total")==129&&p->summary.at("source_vec_ops")==257*15,
      "GELU has one input and its own operation template");
  const auto prefix=compile(hardware(),task(257));
  require(payload(prefix->epochs[0].activities[0])==256&&
      prefix->summary.at("service_curves")[0].at("first_load_elements")==256,
      "single-Core 257-element first load is two separately packed 256-element inputs");
}
void capacity_and_private() {
  auto hw=hardware(2);auto p=compile(hw,task(257));
  // Both blocks fit one cohort: one raw slot per tensor and one work block.
  const u64 peak=2*128+128+4*256;
  require(p->summary.at("sram_peak_bytes_per_core")==peak,"single cohort uses single slots");
  hw.input["performance_profile"]["rho_sram"]=1.0;
  hw.derived["core"]["sram_bytes_per_core"]=peak;
  require(compile(hw,task(257))->summary.at("sram_peak_bytes_per_core")==peak,"capacity equality allowed");
  hw.derived["core"]["sram_bytes_per_core"]=peak-1;
  bool rejected=false;try{compile(hw,task(257));}catch(const std::exception& e){
    rejected=std::string(e.what()).find("unsupported_workspace")!=std::string::npos;}
  require(rejected,"capacity one byte short must reject without changing hardware");
  bool overflow=false;
  try{compile(hardware(),task(1,"residual","int4",9007199254740993ULL));}
  catch(const std::exception& e){overflow=std::string(e.what()).find("overflow")!=std::string::npos;}
  require(overflow,"inexact cumulative credit counts must reject without rounding");
  overflow=false;
  try{compile(hardware(2),task(513,"residual","int4",3002399751580332ULL));}
  catch(const std::exception& e){overflow=std::string(e.what()).find("overflow")!=std::string::npos;}
  require(overflow,"bulk output and early-store credit endpoints must also remain exactly representable");
  for(const auto resource:{SRAM_READ,SRAM_WRITE,LOCAL}) {
    hw=hardware(2);hw.rates[resource]=2.0;
    hw.derived["resources"][resource]["rate"]=2.0;
    p=compile(hw,task(257,"residual","fp32"));
    const auto& first=p->epochs[1].activities;
    const auto& c=*std::find_if(first.begin(),first.end(),[](const Activity& a){return a.role==Role::Compute;});
    const double full=resource==SRAM_READ ? 2048.0 : resource==SRAM_WRITE ? 1024.0 : 3072.0;
    require(c.private_time_s>=full,"full/tail skew must retain per-Core SRAM/LOCAL service bound");
    require(c.work[COMPUTE_GATE]<1,"per-Core memory bound must not inflate compute work");
  }
}
void byte_weighted_routes() {
  for(u64 cores:{1,2,8})for(u64 count:{257,513,1025,4097})for(u64 repeats:{1,3}) {
    const auto hw=hardware(cores);const auto p=compile(hw,task(count,"residual","int4",repeats));
    std::array<long double,3> load{},store{};
    const u64 blocks=(count+255)/256,active=std::min(cores,blocks);
    const long double left=static_cast<long double>(cores/2)/cores;
    // Small independent byte/core/source enumeration. Production uses three
    // analytical groups and must agree with this full/tail byte weighting.
    for(u64 i=0;i<blocks;++i) {
      const u64 n=std::min<u64>(256,count-i*256),x=(i%active)%cores;
      const u64 read=2*((n+1)/2)*repeats,write=((n+1)/2)*repeats;
      long double hop=0;for(u64 source=0;source<cores;++source)
        hop+=static_cast<long double>(source>x ? source-x : x-source)/cores;
      const bool right=x>=cores/2;
      load[0]+=read*left*right;load[1]+=read*(1-left)*(!right);load[2]+=read*hop;
      store[0]+=write*(1-left)*(!right);store[1]+=write*left*right;store[2]+=write*hop;
    }
    std::array<long double,3> actual_load{},actual_store{};
    for(const auto& e:p->epochs)for(const auto& a:e.activities) {
      auto* target=a.role==Role::Load ? &actual_load : a.role==Role::Store ? &actual_store : nullptr;
      if(target)for(std::size_t i=0;i<3;++i)(*target)[i]+=a.work[i==0 ? XPOS : i==1 ? XNEG : HOP];
    }
    for(std::size_t i=0;i<3;++i) {
      const auto tol=[](long double v){return 8*std::numeric_limits<double>::epsilon()*std::max(1.0L,v);};
      require(std::abs(actual_load[i]-load[i])<=tol(load[i])&&std::abs(actual_store[i]-store[i])<=tol(store[i]),
          "byte-weighted directions and second-moment HOP must match actual Core assignment");
    }
    if(cores==2&&count==513&&repeats==1) {
      const auto& last=p->epochs.back().activities[0];
      require(last.work[XPOS]==0.5&&last.work[XNEG]==0,
          "final singleton tail drains from its real Core, not an averaged active prefix");
    }
  }
}
void independent_two_cohort_compute_drain() {
  auto hw=hardware(1);
  hw.derived["core"]["P_vec_per_core_ops_per_s"]=25600.0;
  hw.input["performance_profile"]["launch_ns_per_macro"]=0;
  hw.input["performance_profile"]["memory_setup_ns_per_macro"]=0;
  hw.input["performance_profile"]["eta_vec"]=1.0;
  hw.input["performance_profile"]["eta_hbm_vector"]=1.0;
  hw.fingerprint="synthetic-two-cohort-drain:"+hw.fingerprint;
  for(std::size_t r=0;r<resource_count;++r) {
    const bool enabled=hw.derived["resources"][r].at("enabled").get<bool>();
    const double rate=r==COMPUTE_GATE ? 1.0 : (r==HBM0 || r==HBM1) ? 51200.0 : enabled ? 1e12 : 0;
    hw.rates[r]=rate;hw.derived["resources"][r]["rate"]=rate;
  }
  for(u64 count:{257,321,511,512}) {
    auto t=task(count,"residual","bf16");t["storage_dtype"]={{"activations","bf16"},{"output","fp32"}};
    const auto p=compile(hw,t);
    require(p->summary.at("service_curves")[0].at("model")==
        "analytical_cumulative_elementwise_v3_two_cohort_prefetch" && service_description_count(*p)==6,
        "the two-cohort drain has six real service descriptions independent of tail size");
    Runtime runtime(hw);runtime.submit(p,owner(1,1,1),0);bounded_drain(runtime);
    const long double expected=static_cast<long double>(count)/25600.0L*1e9L;
    const long double measured=runtime.counters().at("compute_busy_ns").get<long double>();
    require(std::abs(measured-expected)<=1.1L,
        "nonbinding memory/LOCAL must not stretch positive Compute service to Store duration");
  }
  // Memory-bound prefix: ComputeFirst ends while prefetched LoadLast is
  // still active. The diagnostic ID must expose that actual activity event.
  hw.derived["core"]["P_vec_per_core_ops_per_s"]=2560000.0;
  auto t=task(512,"residual","bf16");
  t["storage_dtype"]={{"activations","bf16"},{"output","fp32"}};
  const auto p=compile(hw,t);Runtime runtime(hw);
  const auto h=runtime.submit(p,owner(1,1,1),0);
  runtime.advance_to(*runtime.next_event());runtime.advance_to(*runtime.next_event());
  const auto q=runtime.query_progress(h);
  require(q.at("epoch_index")==1,"prefetched load should outlive the short first Compute");
  const auto id=p->summary.at("service_curves")[0].at("first_output_activity_id");
  bool compute_done=false,load_active=false;
  for(const auto& activity:q.at("activities")) {
    if(activity.at("activity_id")==id)compute_done=activity.at("done").get<bool>();
    if(activity.at("epoch_index")==1 && activity.at("role")=="load")
      load_active=!activity.at("done").get<bool>();
  }
  require(compute_done && load_active && runtime.now_ns()>=10100000 && runtime.now_ns()<=10100001,
      "first output is the 10.1ms Compute event within public ceil quantization, before the 20ms epoch gate");

}

void per_core_credit_capacities_under_pause() {
  struct Case {u64 cores,blocks,repeats;};
  for(const auto c:std::array<Case,7>{{{2,7,1},{2,5,3},{8,31,1},{8,17,2},
                                    {1,3,1},{2,5,1},{8,17,1}}}) {
    const auto hw=hardware(c.cores);const auto p=compile(hw,task(c.blocks*256,"residual","bf16",c.repeats));
    Runtime runtime(hw);const auto h=runtime.submit(p,owner(1,1,1),0);
    unsigned steps=0;
    while(runtime.query_progress(h).at("epoch_index").get<u64>()<2) {
      require(++steps<20,"prefix has a fixed event bound");runtime.advance_to(*runtime.next_event());
    }
    runtime.update_capacity(COMPUTE_GATE,0,runtime.now_ns());
    runtime.advance_to(runtime.now_ns()+1000000);
    const auto q=runtime.query_progress(h);long double load=0,compute=0,store=0;
    for(const auto& a:q.at("activities"))if(a.at("epoch_index")==2) {
      if(a.at("role")=="load")load=a.at("progress").get<long double>();
      if(a.at("role")=="compute")compute=a.at("progress").get<long double>();
      if(a.at("role")=="store")store=a.at("progress").get<long double>();
    }
    const u64 active=std::min(c.cores,c.blocks),last=c.blocks%active ? c.blocks%active : active;
    // Independent small RR enumeration counts actual full blocks on each
    // Core. Pausing compute lets Load and Store approach both FIFO bounds.
    std::vector<u64> blocks_per_core(active,0);
    for(u64 execution=0;execution<c.repeats;++execution)
      for(u64 b=0;b<c.blocks;++b)++blocks_per_core[b%active];
    for(u64 core=0;core<active;++core) {
      const auto n=blocks_per_core[core];const u64 final_on_core=core<last;
      const bool prefix_drain=c.repeats==1 && (c.blocks+active-1)/active>=3;
      const long double input=prefix_drain
          ? 1+(n-2)*load-(n-1-final_on_core)*compute : (n-1)*(load-compute);
      const long double visible_output=prefix_drain
          ? 1+(n-2-final_on_core)*compute-(n-1-final_on_core)*store
          : 1+(n-1-final_on_core)*compute-(n-final_on_core)*store;
      const long double output=visible_output+(prefix_drain ? 1 : final_on_core)*compute;
      require(input>=-1e-10L && input<=2+1e-10L,
          "every Core must retain its own two input slots during a compute pause");
      require(visible_output>=-1e-10L && output<=2+1e-10L,
          "no Core may store unproduced output or exceed its two output slots");
    }
    auto fork=Runtime::import_snapshot(hw,runtime.export_snapshot());
    require(fork->query_progress(h)==q,"paused physical buffer state must survive snapshot/import");
    runtime.update_capacity(COMPUTE_GATE,hw.rates[COMPUTE_GATE],runtime.now_ns());
    fork->update_capacity(COMPUTE_GATE,hw.rates[COMPUTE_GATE],fork->now_ns());
    bounded_drain(runtime);bounded_drain(*fork);
    require(runtime.query_progress(h)==fork->query_progress(h),
        "paused buffer continuation must keep the same physical work and finish");
  }
}

void snapshots_and_scaling() {
  const auto hw=hardware(2);u64 largest=0;
  for(u64 repeats:{1,1024,1000000}) {
    const auto p=compile(hw,task(513,"residual","int4",repeats));
    Runtime r(hw);const auto h=r.submit(p,owner(1,1,1),0);
    r.advance_to(*r.next_event());
    auto restored=Runtime::import_snapshot(hw,r.export_snapshot());
    bounded_drain(r);bounded_drain(*restored);
    require(r.query_progress(h)==restored->query_progress(h),"partial source/import continuation must agree");
    largest=std::max(largest,r.counters().at("internal_events").get<u64>());
    require(largest<100&&service_description_count(*p)<=8,"repeat amplification must stay bounded");
  }
  max_service_events=largest;
  for(u64 cores:{1,2,8})for(u64 count:{65537,1048577}) {
    const auto grid=hardware(cores);const auto p=compile(grid,task(count));
    Runtime scaled(grid);scaled.submit(p,owner(1,1,1),0);bounded_drain(scaled);
    const auto events=scaled.counters().at("internal_events").get<u64>();
    require(events<100&&service_description_count(*p)<=16,
        "Core count and block amplification must preserve fixed events/descriptions");
    max_service_events=std::max(max_service_events,events);
  }
  const auto p=compile(hw,task(513));Runtime r(hw);
  const auto h=r.submit(p,owner(1,1,1),0);r.advance_to(*r.next_event());
  r.update_capacity(HBM0,hw.rates[HBM0]/2,r.now_ns());
  auto fork=Runtime::import_snapshot(hw,r.export_snapshot());
  r.cancel(h,r.now_ns());fork->cancel(h,fork->now_ns());
  require(r.query_progress(h)==fork->query_progress(h),"capacity/cancel preserves restored partial work");
}
void verify(){
  independent_enumeration();physical_ledger_and_prefix();capacity_and_private();byte_weighted_routes();independent_two_cohort_compute_drain();per_core_credit_capacities_under_pause();snapshots_and_scaling();
  std::ostringstream value;value<<std::setprecision(std::numeric_limits<long double>::max_digits10)<<max_service_residual;
  std::cout<<json{{"integer_plan_conservation","passed"},{"runtime_service_numeric_reconciliation","passed"},
      {"runtime_service_exact_integer_equality",max_service_residual==0},
      {"maximum_absolute_byte_residual",value.str()},{"maximum_repeat_internal_events",max_service_events},
      {"prediction_accuracy","not_run"},{"independent_target_calibration","not_available"}}.dump()<<'\n';
}
}
int main(){return run_test("elementwise_service_test",verify);}
