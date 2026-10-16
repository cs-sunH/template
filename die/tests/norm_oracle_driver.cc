// Offline driver utilities only: production Runtime/Plan capture, synthetic
// hardware and exact first-Compute activity observation. No timing formula
// from the Python reference is linked into the simulator.
#define main elementwise_driver_unused_main
#include "elementwise_oracle_driver.cc"
#undef main

int main(int argc,char** argv) {
  try {
    std::uint64_t rows=5,hidden=257,offset=5000000;
    std::string profile="balanced",norm="rmsnorm";
    for(int i=1;i<argc;++i) {
      const std::string key=argv[i];
      if(i+1==argc)throw std::invalid_argument("missing value for "+key);
      const std::string value=argv[++i];
      if(key=="--rows")rows=number(value);
      else if(key=="--hidden")hidden=number(value);
      else if(key=="--offset-ns")offset=number(value,true);
      else if(key=="--profile")profile=value;
      else if(key=="--norm")norm=value;
      else throw std::invalid_argument("unknown Norm fixture argument "+key);
    }
    if(rows>4096 || hidden>4096 || (norm!="rmsnorm" && norm!="layernorm"))
      throw std::invalid_argument("offline Norm scope is <=4096 rows/hidden, cached BF16 parameters, one Core, R1");
    if(profile!="balanced" && profile!="memory" && profile!="compute" && profile!="local")
      throw std::invalid_argument("unknown synthetic Norm profile");
    Args a;a.elements=hidden;a.profile=profile;a.output_dtype="bf16";
    Hardware hw=hardware(a);
    const double row_ops=(norm=="rmsnorm" ? 9.0 : 11.0)*hidden;
    double vec=row_ops*100;
    if(profile=="memory" || profile=="local")vec*=100;
    if(profile=="compute")vec*=0.1;
    hw.derived["core"]["P_vec_per_core_ops_per_s"]=vec;
    for(std::size_t r=0;r<resource_count;++r) {
      if(r==HBM0 || r==HBM1)hw.rates[r]=profile=="local" ? 1.0e9 : hidden*100.0;
      if(r==LOCAL && profile=="local")hw.rates[r]=(norm=="rmsnorm" ? 34.0 : 38.0)*hidden*100;
      hw.derived["resources"][r]["rate"]=hw.rates[r];
    }
    hw.fingerprint="synthetic-norm:"+stable_hash(hw.fingerprint+":"+norm+":"+std::to_string(hidden)+":"+std::to_string(vec));
    const json spec={{"op_kind","norm"},{"tensor_shapes",{{"m",rows},{"hidden",hidden},{"norm_type",norm}}},
        {"storage_dtype","bf16"},{"layer_instances",1},{"weight_passes",1}};
    const auto plan=compile(hw,spec);
    json resources=json::array();
    for(std::size_t r=0;r<resource_count;++r)resources.push_back({{"name",names[r]},{"capacity",hw.rates[r]}});
    std::cout<<json({{"schema_version",1},{"scope","offline_single_core_cached_bf16_norm_row_reference_not_hardware_calibration"},
        {"fixture",{{"rows",rows},{"hidden",hidden},{"norm",norm},{"profile",profile}}},
        {"hardware",{{"input",hw.input},{"derived",hw.derived},{"resources",resources},{"fingerprint",hw.fingerprint}}},
        {"compiled_plan",plan_data(*plan)},
        {"runs",{{"solo",run(hw,plan,{0})},{"sync",run(hw,plan,{0,0})},{"offset",run(hw,plan,{0,offset})}}}}).dump(2)<<'\n';
    return 0;
  }catch(const std::exception& e) {std::cerr<<"norm_oracle_driver: "<<e.what()<<'\n';return 1;}
}
