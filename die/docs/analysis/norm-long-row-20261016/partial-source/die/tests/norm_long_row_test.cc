#include "test_support.hh"
#include <array>
#include <set>
using namespace coarse_die;
using namespace coarse_die_test;
using u64=std::uint64_t;
namespace {
Hardware hardware(u64 cores) {
  auto c=read_json(std::string(COARSE_DIE_SOURCE_DIR)+"/configs/example_hardware_explicit.json");
  c["architecture"]["core_columns_nx"]=cores;c["architecture"]["core_rows_ny"]=1;
  return resolve_hardware(c);
}
json task(u64 rows,u64 h,bool layer,const std::string& x,const std::string& p,const std::string& y) {
  return {{"op_kind","norm"},{"tensor_shapes",{{"m",rows},{"hidden",h},{"norm_type",layer ? "layernorm" : "rmsnorm"}}},
      {"storage_dtype",{{"activations",x},{"parameters",p},{"output",y}}}};
}
u64 bytes(u64 n,const std::string& dt){return dt=="int4" ? (n+1)/2 : n*(dt=="bf16" ? 2 : 4);}
void independent_two_pass_ledger() {
  constexpr u64 h=1048577;
  for(bool layer:{false,true})for(u64 cores:{1,4})for(u64 rows:std::set<u64>{1,cores})
      for(const auto types:{std::array<std::string,3>{{"bf16","bf16","bf16"}},
          {{"fp32","fp32","fp32"}},{{"int4","int4","int4"}},{{"int4","bf16","fp32"}}}) {
    const auto hw=hardware(cores);const auto plan=compile(hw,task(rows,h,layer,types[0],types[1],types[2]));
    const auto& s=plan->summary;const u64 vectors=layer ? 2 : 1;
    const auto& x=types[0];const auto& p=types[1];const auto& y=types[2];
    u64 read=0,write=0,math=0,converted=0,eq=0,sr=0,sw=0,local=0;
    const auto cost=[](const std::string& dt){return dt=="int4" ? 4 : dt=="bf16" ? 1 : 0;};
    for(u64 row=0;row<rows;++row) {
      require(row%cores==row,"each row owns a distinct consumer Core");
      const u64 xb=bytes(h,x),yb=bytes(h,y),pb=vectors*bytes(h,p);
      const u64 xc=x=="fp32" ? 0 : 4*h,pc=p=="fp32" ? 0 : 4*h*vectors;
      const u64 stat_r=xb+xc,stat_w=xc+16;
      const u64 apply_r=xb+xc+4*h*vectors+(p=="fp32" ? 0 : pb)+(y=="fp32" ? 0 : 4*h)+16;
      const u64 apply_w=xc+pc+4*h+(y=="fp32" ? 0 : yb);
      read+=2*xb+pb;write+=yb;math+=(layer ? 8 : 6)*h;
      eq+=(layer ? 8 : 6)*h+h*(2*cost(x)+vectors*cost(p)+cost(y));
      converted+=h*(2*(x!="fp32")+vectors*(p!="fp32")+(y!="fp32"));
      sr+=stat_r+apply_r+yb;sw+=2*xb+pb+stat_w+apply_w;local+=stat_r+stat_w+apply_r+apply_w;
    }
    require(s.at("template_id")=="norm_two_pass_single_cohort_v1","long rows use an actual bounded two-pass mapping");
    require(s.at("hbm_read_bytes_total")==read && s.at("hbm_write_bytes_total")==write,
        "each Core rereads its input from HBM and loads its own separately packed gamma/beta");
    require(s.at("source_vec_ops")==math && s.at("vec_equivalent_ops")==eq && s.at("conversion_elements")==converted,
        "two passes preserve math and charge every conversion");
    require(s.at("sram_read_bytes")==sr && s.at("sram_write_bytes")==sw && s.at("local_bus_bytes")==local,
        "tile conversion, input reload and retained statistics have physical SRAM/LOCAL service");
    require(s.at("hbm_workspace_peak_bytes")==0,"one row per Core retains statistics on-chip without a fictitious HBM tensor");
    const auto& ledger=s.at("norm_long_row_ledger");
    const u64 peak=2*ledger.at("input_slot_bytes_per_core").get<u64>()+
        2*ledger.at("output_slot_bytes_per_core").get<u64>()+ledger.at("working_bytes_per_core").get<u64>()+16+64;
    require(s.at("sram_peak_bytes_per_core")==peak && peak<10000,"long row SRAM is a real fixed tile allocation");
    require(plan->epochs.size()==7 && s.at("service_description_count").get<u64>()<=16,
        "row width cannot create tile-linear epochs or service descriptors");
    u64 ar=0,aw=0;long double ae=0,asr=0,asw=0,al=0;
    for(const auto& e:plan->epochs) {
      require(e.activities.size()<=3,"each epoch retains at most three independent activities");
      for(const auto& a:e.activities){
        for(const auto& row:a.hbm_bytes){ar+=row[0];aw+=row[1];}
        ae+=planned_vec_eqops(a);asr+=a.work[SRAM_READ];asw+=a.work[SRAM_WRITE];al+=a.work[LOCAL];
      }
    }
    require(ar==read && aw==write && ae==eq && asr==sr && asw==sw && al==local,
        "actual activities conserve the independently derived two-pass ledger");
    Runtime runtime(hw);const auto handle=runtime.submit(plan,{1,2,3,1},0);unsigned events=0;
    while(!runtime.drained()){require(++events<=32,"width must not create tile-linear events");runtime.advance_to(*runtime.next_event());}
    require(runtime.terminal_finish_ns(handle).has_value(),"the last apply output must actually complete");
  }
}
void statistics_barrier_pause_restore() {
  const auto hw=hardware(2);const auto plan=compile(hw,task(2,131073,false,"bf16","bf16","bf16"));
  Runtime runtime(hw);const auto handle=runtime.submit(plan,{1,2,3,1},0);
  runtime.advance_to(*runtime.next_event()); // launch
  runtime.advance_to(*runtime.next_event()); // statistics first input tile
  runtime.update_capacity(COMPUTE_GATE,0,runtime.now_ns());
  unsigned events=0;
  while(runtime.next_event()){require(++events<=4,"only finite statistics input prefetch is allowed");runtime.advance_to(*runtime.next_event());}
  auto counters=runtime.counters();const auto& served=counters.at("hbm_served_bytes_by_row_direction");
  require(served[0][1].get<double>()+served[1][1].get<double>()==0,
      "no output may be written while full-row statistics are blocked");
  // next_event() is publicly ceiled to integer ns. Statistics may consume a
  // tiny input prefix before that tick is returned; bound unconsumed bytes,
  // not all bytes ever loaded. RMS BF16 statistics costs 3 eqops/element.
  const auto progress=runtime.query_progress(handle);
  const double consumed=progress.at("served").at("vec_eqops").get<double>()/3.0*2.0;
  const double resident=served[0][0].get<double>()+served[1][0].get<double>()-consumed;
  require(resident<=2*512*2+1e-6 && resident>=-1e-6,
      "statistics prefetch respects two source tiles per Core after subtracting actual consumption");
  const auto snapshot=runtime.export_snapshot();auto resumed=Runtime::import_snapshot(hw,snapshot);
  auto cancelled=Runtime::import_snapshot(hw,snapshot);cancelled->cancel(handle,cancelled->now_ns());
  require(cancelled->drained() && cancelled->query_progress(handle).at("cancelled")==true,
      "long-row cancellation retains the public terminal semantics");
  runtime.update_capacity(COMPUTE_GATE,hw.rates[COMPUTE_GATE],runtime.now_ns());
  resumed->update_capacity(COMPUTE_GATE,hw.rates[COMPUTE_GATE],resumed->now_ns());
  events=0;while(!runtime.drained()){require(++events<=32,"resumed long row remains bounded");runtime.advance_to(*runtime.next_event());}
  while(!resumed->drained())resumed->advance_to(*resumed->next_event());
  require(runtime.counters()==resumed->counters() && runtime.query_progress(handle)==resumed->query_progress(handle),
      "both passes and retained statistics restore without replay or duplicate work");
}
void unsupported_multirow_and_repeats() {
  const auto hw=hardware(2);
  for(auto spec:{task(3,1048577,false,"bf16","bf16","bf16"),task(1,1048577,true,"bf16","bf16","bf16")}) {
    if(spec["tensor_shapes"]["m"]==1)spec["layer_instances"]=2;
    bool rejected=false;try{(void)compile(hw,spec);}catch(const std::invalid_argument& e){rejected=std::string(e.what()).find("unsupported_workspace")!=std::string::npos;}
    require(rejected,"the new subset cannot silently aggregate repeated or multiple per-Core rows");
  }
}
void run(){independent_two_pass_ledger();statistics_barrier_pause_restore();unsupported_multirow_and_repeats();}
}
int main(){return run_test("norm_long_row_test",run);}
