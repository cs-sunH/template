#include "coarse_die/Plan.hh"
#include <fstream>
#include <iostream>
using nlohmann::json;
json read(const char* p){std::ifstream f(p);return json::parse(f);}
int main(int argc,char**argv){
 auto hw=coarse_die::resolve_hardware(read(argv[1]));
 json span={{"run",{{"q_len",1},{"kv_len",4},{"kv_stride",2},{"count",2}}},{"request_id_prefix","request"},{"first_index",0}};
 json attention={{"op_kind","attention"},{"tensor_shapes",{{"q_heads",1},{"kv_heads",1},{"head_dim",1}}},{"causal",true},{"spans",json::array({span})}};
 auto tasks=read(argv[2]);
 std::ofstream(argv[3])<<coarse_die::plan_to_json(*coarse_die::compile(hw,attention)).dump(2)<<'\n';
 std::ofstream(argv[4])<<coarse_die::plan_to_json(*coarse_die::compile(hw,tasks.at("specs").at(1))).dump(2)<<'\n';
}
