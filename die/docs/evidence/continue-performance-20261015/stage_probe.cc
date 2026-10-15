#include "coarse_die/Execution.hh"
#include "CompilerInternal.hh"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
namespace coarse_die::detail {
std::vector<std::shared_ptr<Plan>> enumerate_mappings_for_compile(const Hardware&, const nlohmann::json&);
}
using namespace coarse_die;
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
Json stats(std::vector<uint64_t> samples) {
  auto ordered=samples;std::sort(ordered.begin(),ordered.end());
  return {{"samples_ns",samples},{"p50_ns",ordered[(ordered.size()+1)/2-1]},
          {"p95_ns",ordered[(ordered.size()*95+99)/100-1]}};
}
template<class F> Json measured(F f) {
  std::vector<uint64_t> samples;
  for(int i=0;i<30;++i){auto a=Clock::now();f();auto b=Clock::now();samples.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count());}
  return stats(samples);
}
int main(int argc,char**argv) {
  if(argc!=3)return 2;
  std::ifstream input(argv[1]);Json config=Json::parse(input);auto hw=resolve_hardware(config);
  const std::string profile=config.at("performance_profile").at("execution_profile");
  Json task={{"op_kind","gemm"},{"tensor_shapes",{{"m",128},{"n",128},{"k",128}}},
             {"storage_dtype",{{"weights","bf16"},{"activations","bf16"},{"output","bf16"}}},
             {"execution_profile",profile},{"layer_instances",1},{"weight_passes",1}};
  Json normalized=detail::normalize_task(hw,task),sink;
  auto immutable_hw=std::make_shared<const Hardware>(hw);
  auto candidates=detail::enumerate_mappings_for_compile(hw,normalized);
  auto compiled=compile(hw,task);
  Json out={{"schema","read_only_frozen_compile_stage_probe_v1"},{"frozen_commit","5c2de289bc5744e5c252d782901cdba8b6f62459"},
   {"candidate_count",candidates.size()},{"selected_plan_hash",compiled->fingerprint},
   {"scope","diagnostic stage timing only; prebuilt normalized/candidate objects reused; phases are not an additive decomposition of compile()"}};
  out["normalize_task"]=measured([&]{sink=detail::normalize_task(hw,task);});
  std::vector<std::shared_ptr<Plan>> mapped;
  out["enumerate_mappings_for_compile"]=measured([&]{mapped=detail::enumerate_mappings_for_compile(hw,normalized);});
  uint64_t latency_sum=0;
  out["estimate_all_prebuilt_candidates"]=measured([&]{for(const auto&p:candidates)latency_sum+=detail::estimate_candidate_latency_ns(immutable_hw,p);});
  out["plan_to_json_winner"]=measured([&]{sink=plan_to_json(*compiled);});
  out["compile_public"]=measured([&]{compiled=compile(hw,task);});
  out["guard_latency_sum"]=latency_sum;
  std::ofstream result(argv[2]);result<<out.dump(2)<<'\n';
  std::cout<<out.dump(2)<<'\n';
}
