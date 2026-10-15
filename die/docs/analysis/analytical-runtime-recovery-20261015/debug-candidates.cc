#include "/workspace/template/die/tests/test_support.hh"
#include "/workspace/template/die/src/CompilerInternal.hh"
namespace coarse_die::detail { std::vector<std::shared_ptr<Plan>> enumerate_mappings(const Hardware&, const nlohmann::json&); }
using namespace coarse_die; using namespace coarse_die_test;
int main() {
 auto hw=example_hardware();
 json t={{"op_kind","gemm"},{"tensor_shapes",{{"m",89},{"n",1},{"k",1}}},{"storage_dtype",{{"weights","bf16"},{"activations","bf16"},{"output","bf16"}}},{"execution_profile","int4_matrix_vec_fp32_assumed_v1"},{"layer_instances",1},{"weight_passes",1}};
 auto p=detail::enumerate_mappings(hw,detail::normalize_task(hw,t));
 for(auto c:p) { Runtime r(hw); r.submit(c,owner(1,1,1),0); unsigned guard=0;
  while(!r.drained()) { auto n=r.next_event(); if(!n) {std::cerr << c->summary.at("dataflow_id") << " tm="<<c->summary.at("tm")<<"\n"; std::ofstream("/workspace/template/work/analytical-continue/candidate-failure.json") << r.export_snapshot().dump(2); return 1;} r.advance_to(*n); if(++guard>100)return 2; }
 }
 std::cout << "PASS "<<p.size()<<" candidates\n";
}
