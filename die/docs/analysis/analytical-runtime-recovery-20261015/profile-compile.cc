#include "/workspace/template/die/tests/test_support.hh"
#include "/workspace/template/die/src/CompilerInternal.hh"
#include <chrono>
namespace coarse_die::detail { std::vector<std::shared_ptr<Plan>> enumerate_mappings_for_compile(const Hardware&, const nlohmann::json&); }
int main(){using namespace coarse_die;using namespace coarse_die_test;using C=std::chrono::steady_clock;
 auto hw=std::make_shared<const Hardware>(example_hardware());
 json t={{"op_kind","gemm"},{"tensor_shapes",{{"m",128},{"n",128},{"k",128}}},{"storage_dtype",{{"weights","bf16"},{"activations","bf16"},{"output","bf16"}}},{"execution_profile","int4_matrix_vec_fp32_assumed_v1"},{"layer_instances",1},{"weight_passes",1}};
 long long norm=0,enumeration=0,estimate=0;std::size_t count=0;
 for(unsigned i=0;i<100;++i){auto b=C::now();auto n=detail::normalize_task(*hw,t);auto nend=C::now();auto v=detail::enumerate_mappings_for_compile(*hw,n);auto vend=C::now();for(auto p:v)detail::estimate_candidate_latency_ns(hw,p);auto eend=C::now();count=v.size();norm+=(nend-b).count();enumeration+=(vend-nend).count();estimate+=(eend-vend).count();}
 std::cout<<"candidates="<<count<<" mean ns normalize="<<norm/100<<" enumeration="<<enumeration/100<<" Runtime trials="<<estimate/100<<"\n";
}
