// Diagnostic only: counting global C++ new adds overhead. These samples never
// replace the public latency benchmark or its failed acceptance samples.
#include <cstdlib>
#include <new>
#include <sys/resource.h>
#include <time.h>

namespace allocation_probe {
thread_local bool enabled=false;
thread_local std::size_t calls=0,bytes=0;
}
void* operator new(std::size_t n) {
  if(allocation_probe::enabled){++allocation_probe::calls;allocation_probe::bytes+=n;}
  if(void* p=std::malloc(n ? n : 1))return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t n){return ::operator new(n);}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete[](void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
void operator delete[](void* p,std::size_t) noexcept {std::free(p);}

#define main benchmark_utility_unused_main
#include "../../../tests/service_curve_benchmark.cc"
#undef main

namespace coarse_die::detail {
std::vector<std::shared_ptr<Plan>> enumerate_mappings_for_compile(const Hardware&,const Json&);
std::uint64_t estimate_candidate_latency_ns(std::shared_ptr<const Hardware>,std::shared_ptr<const Plan>);
}
namespace {
std::uint64_t thread_time() {
  timespec t{};if(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t)!=0)throw std::runtime_error("thread CPU clock unavailable");
  return static_cast<std::uint64_t>(t.tv_sec)*1000000000+t.tv_nsec;
}
template<class F> Json measure(F action) {
  rusage before{},after{};getrusage(RUSAGE_THREAD,&before);
  const auto cpu=thread_time();const auto wall=Clock::now();
  allocation_probe::calls=allocation_probe::bytes=0;allocation_probe::enabled=true;
  action();
  allocation_probe::enabled=false;
  const auto end=Clock::now();const auto cpu_end=thread_time();getrusage(RUSAGE_THREAD,&after);
  return {{"wall_ns",elapsed_ns(wall,end)},{"thread_cpu_ns",cpu_end-cpu},
      {"cpp_new_calls",allocation_probe::calls},{"cpp_new_bytes",allocation_probe::bytes},
      {"voluntary_switches",after.ru_nvcsw-before.ru_nvcsw},{"involuntary_switches",after.ru_nivcsw-before.ru_nivcsw}};
}
}
#ifndef HOTPATH_PROFILE_UTILITIES_ONLY
int main(int argc,char** argv) {
  try {
    const auto args=parse_args(argc,argv);
    const auto config=read_json_file(args.config_path);const Hardware hw=resolve_hardware(config);
    const auto profile=config.at("performance_profile").at("execution_profile").get<std::string>();
    Json report={{"scope","diagnostic allocation counts and phase CPU only; not acceptance timing"},{"samples",args.samples},{"cases",Json::array()}};
    for(auto rows:{1u,128u}) {
      const Json spec=gemm_task(rows,128,128,profile);
      Json samples=Json::array();
      for(std::size_t i=0;i<args.samples;++i) {
        Json normalized;std::vector<std::shared_ptr<Plan>> candidates;
        std::shared_ptr<const Hardware> shared;std::shared_ptr<const Plan> complete;
        Json row;
        row["normalize"]=measure([&]{normalized=detail::normalize_task(hw,spec);});
        row["enumerate"]=measure([&]{candidates=detail::enumerate_mappings_for_compile(hw,normalized);});
        row["hardware_copy"]=measure([&]{shared=std::make_shared<const Hardware>(hw);});
        row["candidate_count"]=candidates.size();
        row["candidate_structure_validation_subset"]=measure([&]{for(const auto& p:candidates)detail::validate_plan_structure(*p);});
        row["all_candidate_runtime_with_validation"]=measure([&]{for(const auto& p:candidates)(void)detail::estimate_candidate_latency_ns(shared,p);});
        row["actual_full_compile"]=measure([&]{complete=compile(hw,spec);});
        row["selected_hash"]=complete->fingerprint;samples.push_back(std::move(row));
      }
      report["cases"].push_back({{"m",rows},{"n",128},{"k",128},{"samples",samples}});
    }
    const auto selected=select_os_bulk_plan(hw,gemm_task(128,128,128,profile));
    Json updates=Json::array();
    for(std::size_t count:{1,8,32})for(std::size_t i=0;i<args.samples;++i) {
      Runtime runtime(hw);std::vector<std::uint64_t> handles;
      for(std::size_t owner_index=0;owner_index<count;++owner_index)
        handles.push_back(runtime.submit(selected.plan,owner(owner_index+1,1,owner_index+1),0));
      advance_to_os_bulk(runtime,handles,selected.bulk_epoch);
      auto sample=measure([&]{runtime.update_capacity(HBM0,hw.rates[HBM0],runtime.now_ns());});
      sample["owners"]=count;sample["arbitration_path"]=arbitration_path_name(runtime.last_arbitration().path);
      updates.push_back(std::move(sample));
    }
    report["single_os_updates"]=updates;
    std::ofstream output(args.output_path);if(!output)throw std::runtime_error("cannot open profile output");
    output<<report.dump(2)<<'\n';return 0;
  }catch(const std::exception& e){std::cerr<<"phase_profile: "<<e.what()<<'\n';return 1;}
}
#endif
