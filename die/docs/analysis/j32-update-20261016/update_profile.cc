// Instrumented diagnostic linked to a work-only Execution object, never SDK.
#include "trace.hh"
#define HOTPATH_PROFILE_UTILITIES_ONLY
#include "../hotpath-allocation-20261016/phase_profile.cc"
int main(int argc,char** argv){
  try{
    const auto args=parse_args(argc,argv);const auto config=read_json_file(args.config_path);
    const auto hw=resolve_hardware(config);
    const auto profile=config.at("performance_profile").at("execution_profile").get<std::string>();
    const auto selected=select_os_bulk_plan(hw,gemm_task(128,128,128,profile));
    Json rows=Json::array();
    for(std::size_t owners:{1,8,32})for(std::size_t i=0;i<args.samples;++i){
      Runtime runtime(hw);std::vector<std::uint64_t> handles;
      for(std::size_t n=0;n<owners;++n)handles.push_back(runtime.submit(selected.plan,owner(n+1,1,n+1),0));
      advance_to_os_bulk(runtime,handles,selected.bulk_epoch);
      update_trace::totals={};update_trace::calls={};update_trace::enabled=true;
      auto row=measure([&]{runtime.update_capacity(HBM0,hw.rates[HBM0],runtime.now_ns());});
      update_trace::enabled=false;
      row["owners"]=owners;row["path"]=arbitration_path_name(runtime.last_arbitration().path);
      for(std::size_t k=0;k<update_trace::Count;++k)
        row["trace"][update_trace::names[k]]={{"inclusive_thread_cpu_ns",update_trace::totals[k]},
                                         {"calls",update_trace::calls[k]}};
      rows.push_back(std::move(row));
    }
    std::ofstream output(args.output_path);if(!output)throw std::runtime_error("cannot open update diagnostic output");
    output<<Json{{"scope","diagnostic clocks/global-new add overhead; not budget samples; production source unchanged"},
        {"samples",args.samples},{"rows",rows}}.dump(2)<<'\n';return 0;
  }catch(const std::exception& e){std::cerr<<"update_profile: "<<e.what()<<'\n';return 1;}
}
