#include "test_support.hh"
#include "CompilerInternal.hh"
#include <fstream>
namespace coarse_die::detail {std::vector<std::shared_ptr<Plan>> enumerate_mappings(const Hardware&,const nlohmann::json&);}
int main(){using namespace coarse_die;using namespace coarse_die_test;
  auto hw=example_hardware();
  auto spec=detail::normalize_task(hw,json{{"op_kind","gemm"},{"tensor_shapes",{{"m",17},{"n",5},{"k",3}}}});
  for(auto p:detail::enumerate_mappings(hw,spec)){
    Runtime r(hw);auto h=r.submit(p,owner(1,1,1),0);unsigned n=0;
    while(!r.drained()){
      auto next=r.next_event();if(!next){std::ofstream("work/analytical-curves/deadlock-state.json")<<json{{"plan",plan_to_json(*p)},{"snapshot",r.export_snapshot()},{"progress",r.query_progress(h)}}.dump(2);return 1;}
      r.advance_to(*next);if(++n>100)return 2;
    }
  }return 0;}
