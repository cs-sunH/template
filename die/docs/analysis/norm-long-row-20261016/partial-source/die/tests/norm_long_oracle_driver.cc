// Offline synthetic two-pass evidence, one Core / one row / BF16 / R1.
#define main elementwise_driver_unused_main
#include "elementwise_oracle_driver.cc"
#undef main
int main(int argc,char** argv) {
  try {
    std::uint64_t hidden=65537,offset=500000000;std::string norm="rmsnorm",profile="balanced";
    for(int i=1;i<argc;++i) {
      const std::string key=argv[i];if(i+1==argc)throw std::invalid_argument("missing option value");
      const std::string value=argv[++i];
      if(key=="--hidden")hidden=number(value);
      else if(key=="--offset-ns")offset=number(value,true);
      else if(key=="--norm")norm=value;
      else if(key=="--profile")profile=value;
      else throw std::invalid_argument("unknown option "+key);
    }
    if(hidden<65537 || hidden>262145 || (norm!="rmsnorm" && norm!="layernorm"))
      throw std::invalid_argument("offline long row scope is BF16 R1, one Core/row, hidden65537..262145");
    if(profile!="balanced" && profile!="memory" && profile!="compute" && profile!="local")
      throw std::invalid_argument("unknown synthetic long row profile");
    Args a;a.elements=hidden;a.profile=profile;a.output_dtype="bf16";auto hw=hardware(a);
    double peak=(norm=="rmsnorm" ? 10.0 : 13.0)*256*100;
    if(profile=="memory" || profile=="local")peak*=100;
    if(profile=="compute")peak*=0.1;
    hw.derived["core"]["P_vec_per_core_ops_per_s"]=peak;
    if(profile=="local")hw.rates[LOCAL]=(norm=="rmsnorm" ? 34.0 : 38.0)*256*100;
    for(std::size_t r=0;r<resource_count;++r)hw.derived["resources"][r]["rate"]=hw.rates[r];
    hw.fingerprint="synthetic-norm-long:"+stable_hash(hw.fingerprint+":"+norm+":"+std::to_string(hidden)+":"+std::to_string(peak));
    const json spec={{"op_kind","norm"},{"tensor_shapes",{{"m",1},{"hidden",hidden},{"norm_type",norm}}},
        {"storage_dtype","bf16"},{"layer_instances",1},{"weight_passes",1}};
    const auto plan=compile(hw,spec);
    if(!plan->summary.contains("norm_long_row_ledger"))throw std::runtime_error("fixture did not select the two-pass mapping");
    json resources=json::array();
    for(std::size_t r=0;r<resource_count;++r)resources.push_back({{"name",names[r]},{"capacity",hw.rates[r]}});
    std::cout<<json({{"schema_version",1},{"scope","offline_one_core_one_row_bf16_two_hbm_passes_not_hardware_calibration"},
        {"fixture",{{"hidden",hidden},{"norm",norm},{"profile",profile}}},
        {"hardware",{{"input",hw.input},{"derived",hw.derived},{"resources",resources},{"fingerprint",hw.fingerprint}}},
        {"compiled_plan",plan_data(*plan)},
        {"runs",{{"solo",run(hw,plan,{0})},{"sync",run(hw,plan,{0,0})},{"offset",run(hw,plan,{0,offset})}}}}).dump(2)<<'\n';
    return 0;
  }catch(const std::exception& e){std::cerr<<"norm_long_oracle_driver: "<<e.what()<<'\n';return 1;}
}
