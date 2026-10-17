// Fixed-width dyadic arithmetic only. No Runtime, arbitrary precision, or fma.
#include <algorithm>
#include <array>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <json/json.hpp>
using nlohmann::json;
#include "Dyadic128.hh"
using namespace norm_dyadic128;
std::string hex_real(Real x){std::ostringstream s;s<<std::hexfloat<<x;return s.str();}
std::string hex_int(U128 x){if(!x)return "0x0";std::string s;while(x){s.push_back("0123456789abcdef"[static_cast<unsigned>(x&15)]);x>>=4;}std::reverse(s.begin(),s.end());return "0x"+s;}
json result_json(const Evaluation& e){const auto&r=e.result;json terms=json::array();for(std::size_t i=0;i<e.count;++i){const auto&t=e.terms[i];terms.push_back({{"mantissa_hex",hex_int(t.mantissa)},{"exponent",t.exponent},{"negative",t.negative}});}json j={{"supported",r.supported},{"reason",r.reason},{"terms",terms}};if(r.supported){j["sign"]=r.sign;j["magnitude_hex"]=hex_int(r.magnitude);j["exponent"]=r.exponent;j["positive_group_hex"]=hex_int(r.positive);j["negative_group_hex"]=hex_int(r.negative);j["base_exponent"]=r.base_exponent;j["lower_hex"]=hex_real(r.lower);j["upper_hex"]=hex_real(r.upper);}return j;}
double parse_double(const json& j){const auto s=j.get<std::string>();char* end=nullptr;const double x=std::strtod(s.c_str(),&end);if(end!=s.c_str()+s.size())throw std::runtime_error("coefficient parse");return x;}
Real parse_progress(const json& j){const auto s=j.get<std::string>();char* end=nullptr;const Real x=std::strtold(s.c_str(),&end);if(end!=s.c_str()+s.size())throw std::runtime_error("progress parse");return x;}
Real direct_cost(const Sample& s,bool head){Real sum=head?static_cast<Real>(s.capacity)-s.initial:s.initial;for(std::size_t i=0;i<s.n;++i)sum+=(head?-static_cast<Real>(s.d[i]):s.d[i])*s.p[i];return sum;}
Real sign_cost(const Sample&s,bool head){const auto e=evaluate(s,head,false);return e.result.supported?static_cast<Real>(e.result.sign):99;}
Real interval_cost(const Sample&s,bool head){const auto e=evaluate(s,head,true);return e.result.supported?e.result.lower+e.result.upper:99;}
json controls(){
 json list=json::array();auto add=[&](const char*name,std::initializer_list<Term> ts){Evaluation e;e.count=ts.size();std::copy(ts.begin(),ts.end(),e.terms.begin());e.result=sum_terms(e.terms,e.count,true);list.push_back({{"name",name},{"result",result_json(e)}});};
 add("span_129",{{1,0,false},{1,128,false}});add("mantissa_plus_shift_129",{{1,0,false},{3,127,false}});
 add("positive_accumulator_overflow",{{max128,0,false},{1,0,false}});add("negative_accumulator_overflow",{{max128,0,true},{1,0,true}});
 add("maximum_128_integer",{{max128,0,false}});add("zero_cancel",{{max128,0,false},{max128,0,true}});
 add("output_overflow",{{1,std::numeric_limits<Real>::max_exponent,false}});add("output_subnormal",{{1,std::numeric_limits<Real>::min_exponent-2,false}});
 add("negative_rounding_flip",{{(U128{1}<<64)+1,-64,true}});add("zero_terms_supported",{{0,-1234,true},{0,999,false}});
 Sample p;p.initial=1;p.capacity=1;p.n=1;p.d[0]=-1;p.p[0]=1;list.push_back({{"name","progress_one_supported"},{"result",result_json(evaluate(p,false,true))}});
 p.p[0]=std::numeric_limits<Real>::denorm_min();list.push_back({{"name","progress_subnormal_rejected"},{"result",result_json(evaluate(p,false,true))}});
 p.p[0]=0;p.d[0]=std::numeric_limits<double>::denorm_min();list.push_back({{"name","coefficient_subnormal_rejected"},{"result",result_json(evaluate(p,false,true))}});return list;
}
int main(int argc,char**argv){if(argc!=3)return 2;std::ifstream in(argv[1]);json input;in>>input;std::ofstream out(argv[2]);
 const bool abi=std::numeric_limits<Real>::digits==64&&std::numeric_limits<double>::is_iec559&&std::numeric_limits<double>::digits==53&&sizeof(double)==8&&std::fegetround()==FE_TONEAREST;
 out<<json{{"kind","abi"},{"supported",abi},{"long_double_digits",std::numeric_limits<Real>::digits},{"long_double_min_exponent",std::numeric_limits<Real>::min_exponent},{"long_double_max_exponent",std::numeric_limits<Real>::max_exponent},{"u128_bits",128},{"rounding",std::fegetround()==FE_TONEAREST?"FE_TONEAREST":"unsupported"}}.dump()<<'\n';std::vector<Sample> eligible;
 for(const auto& row:input.at("cases")){json result={{"kind","case"},{"id",row.at("id")}};if(!abi){result["unsupported"]= "requires_LD64_IEEE_binary64_RNE";out<<result.dump()<<'\n';continue;}
  Sample s;s.initial=parse_double(row.at("initial"));s.capacity=parse_double(row.at("capacity"));s.n=row.at("coefficients").size();if(s.n>3||s.n<1)throw std::runtime_error("fixed n=1..3 required");
  for(std::size_t i=0;i<s.n;++i){s.d[i]=parse_double(row.at("coefficients")[i]);s.p[i]=parse_progress(row.at("progress")[i]);}
  if(row.contains("neighbor")){const auto k=row.at("neighbor").at("index").get<std::size_t>();s.p[k]=std::nextafter(s.p[k],row.at("neighbor").at("direction").get<int>()<0?-std::numeric_limits<Real>::infinity():std::numeric_limits<Real>::infinity());}
  result["progress_hex"]=json::array();for(std::size_t i=0;i<s.n;++i)result["progress_hex"].push_back(hex_real(s.p[i]));const auto q=evaluate(s,false,true),h=evaluate(s,true,true);result["q"]=result_json(q);result["h"]=result_json(h);if(q.result.supported&&h.result.supported)eligible.push_back(s);out<<result.dump()<<'\n';
 }
 if(!abi)return 0;out<<json{{"kind","domain_controls"},{"cases",controls()}}.dump()<<'\n';
 json benchmark={{"kind","benchmark"},{"scope","standalone dyadic arithmetic only; not Runtime/API budget"},{"eligible_samples",eligible.size()},{"batches",5},{"rounds_per_batch",100},{"methods",json::object()}};volatile Real sink=0;
 for(const auto& method:{std::string("direct"),std::string("dyadic128_checked_sign"),std::string("dyadic128_checked_interval")}){auto fn=method=="direct"?direct_cost:method=="dyadic128_checked_sign"?sign_cost:interval_cost;for(const auto&s:eligible){sink+=fn(s,false);sink+=fn(s,true);}json times=json::array();for(unsigned batch=0;batch<5;++batch){const auto start=std::chrono::steady_clock::now();for(unsigned round=0;round<100;++round)for(const auto&s:eligible){sink+=fn(s,false);sink+=fn(s,true);}const auto end=std::chrono::steady_clock::now();times.push_back(std::chrono::duration<double,std::nano>(end-start).count()/(200.0*eligible.size()));}benchmark["methods"][method]=times;}
 benchmark["observable_sink_hex"]=hex_real(sink);out<<benchmark.dump()<<'\n';
}
